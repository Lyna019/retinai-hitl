"""
SQLAlchemy models — full schema v3.

Entity graph:
  User → * AnnotationSession → * Annotation
  Patient → * FundusImage → * Annotation
                         → * ModelPrediction
  Annotation → * DiseaseLabel → 1 Disease → 1 Mechanism
             → * RegionOfInterest → * GridCell → 0..1 LesionVocabulary
             → 0..1 ClinicalNote
             → 0..1 GradCAMValidation → 1 ModelPrediction
             → 0..1 UrgencyFlag
  User → * CustomPathologyProposal → 0..1 Disease (on approval)
  Patient → * PatientSystemicDisease → 1 SystemicDisease
  AuditLog — global change journal
  ModelVersion — version registry for rollback
  ActiveLearningConfig — training loop parameters
  ImageLock — patient-level queue locking
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime,
    ForeignKey, JSON, Text, Enum, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from ..database import Base


def _uid():
    return str(uuid.uuid4())


# ─────────────────────────── Users ───────────────────────────

class User(Base):
    __tablename__ = "users"
    id            = Column(UUID(as_uuid=False), primary_key=True, default=_uid)
    username      = Column(String(64), unique=True, nullable=False)
    email         = Column(String(128))
    full_name     = Column(String(128))
    role          = Column(String(16), default="doctor")   # admin | doctor | viewer
    password_hash = Column(String(255), nullable=False)
    is_active     = Column(Boolean, default=True)
    created_at    = Column(DateTime, default=datetime.utcnow)

    annotations      = relationship("Annotation", back_populates="doctor", foreign_keys="[Annotation.doctor_id]")
    sessions         = relationship("AnnotationSession", back_populates="doctor")
    locks            = relationship("ImageLock", back_populates="locked_by_user", foreign_keys="[ImageLock.locked_by]")
    custom_proposals = relationship("CustomPathologyProposal", back_populates="doctor", foreign_keys="[CustomPathologyProposal.doctor_id]")
    audit_logs       = relationship("AuditLog", back_populates="user", foreign_keys="[AuditLog.user_id]")


# ─────────────────────────── Systemic disease catalog (admin-managed) ───────────────────────────

class SystemicDisease(Base):
    """Admin-managed list of known systemic diseases shown in patient info panel."""
    __tablename__ = "systemic_diseases"
    id         = Column(UUID(as_uuid=False), primary_key=True, default=_uid)
    name_fr    = Column(String(128), unique=True, nullable=False)
    category   = Column(String(64))   # e.g. Métabolique, Cardiovasculaire, Ophtalmique…
    is_active  = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ─────────────────────────── Patients & images ───────────────────────────

class Patient(Base):
    __tablename__ = "patients"
    id              = Column(UUID(as_uuid=False), primary_key=True, default=_uid)
    clinical_id     = Column(String(64), unique=True, index=True)   # bridge to hospital DB
    full_name       = Column(String(128))
    birth_date      = Column(DateTime)                               # extracted from DICOM (0010,0030)
    age             = Column(Integer)
    gender          = Column(String(8))                              # M | F (DICOM 0010,0040)
    historical_note = Column(Text)                                   # free-text filled by doctor
    created_at      = Column(DateTime, default=datetime.utcnow)

    systemic_diseases = relationship("PatientSystemicDisease", back_populates="patient", cascade="all, delete-orphan")
    images            = relationship("FundusImage", back_populates="patient")


class PatientSystemicDisease(Base):
    """M2M link: patient ↔ systemic disease."""
    __tablename__ = "patient_systemic_diseases"
    __table_args__ = (UniqueConstraint("patient_id", "systemic_disease_id"),)
    id                  = Column(UUID(as_uuid=False), primary_key=True, default=_uid)
    patient_id          = Column(UUID(as_uuid=False), ForeignKey("patients.id"), nullable=False)
    systemic_disease_id = Column(UUID(as_uuid=False), ForeignKey("systemic_diseases.id"), nullable=False)

    patient         = relationship("Patient", back_populates="systemic_diseases")
    systemic_disease = relationship("SystemicDisease")


class FundusImage(Base):
    __tablename__ = "fundus_images"
    id                = Column(UUID(as_uuid=False), primary_key=True, default=_uid)
    patient_id        = Column(UUID(as_uuid=False), ForeignKey("patients.id"))
    file_path         = Column(String(512), nullable=False)
    dicom_path        = Column(String(512))                          # original .dcm if available
    modality          = Column(String(16))                           # UWF | STD | OPTOMAPPLUS
    eye               = Column(String(2))                            # OD | OS
    capture_date      = Column(DateTime)
    study_date        = Column(DateTime)                             # DICOM (0008,0020)
    dicom_study_uid   = Column(String(128))                         # DICOM SOP Instance UID
    status            = Column(String(16), default="pending")        # pending | in_progress | done
    uncertainty_score = Column(Float, default=0.5)
    model_urgency     = Column(String(4))                            # P1 | P2 | P3 | None
    assigned_to       = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)
    image_quality     = Column(String(8))                            # good | fair | poor
    ingested_at       = Column(DateTime, default=datetime.utcnow)

    patient      = relationship("Patient", back_populates="images")
    lock         = relationship("ImageLock", back_populates="image", uselist=False)
    annotations  = relationship("Annotation", back_populates="image")
    predictions  = relationship("ModelPrediction", back_populates="image")


class ImageLock(Base):
    """Patient-level queue lock: when a doctor opens one image, all images of the
    same patient are locked to that doctor until the session is released."""
    __tablename__ = "image_locks"
    id          = Column(UUID(as_uuid=False), primary_key=True, default=_uid)
    image_id    = Column(UUID(as_uuid=False), ForeignKey("fundus_images.id"), unique=True, nullable=False)
    patient_id  = Column(UUID(as_uuid=False), ForeignKey("patients.id"), nullable=False)
    locked_by   = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False)
    locked_at   = Column(DateTime, default=datetime.utcnow)

    image          = relationship("FundusImage", back_populates="lock")
    locked_by_user = relationship("User", back_populates="locks", foreign_keys=[locked_by])
    patient        = relationship("Patient", foreign_keys=[patient_id])


# ─────────────────────────── Disease catalog ───────────────────────────

class Mechanism(Base):
    __tablename__ = "mechanisms"
    code        = Column(String(16), primary_key=True)   # VASC | DEGEN | INFLAM | DYST | STRUCT | TUMOR
    name_fr     = Column(String(128), nullable=False)
    description = Column(Text)

    diseases = relationship("Disease", back_populates="mechanism")


class Disease(Base):
    __tablename__ = "diseases"
    code             = Column(String(32), primary_key=True)
    name_fr          = Column(String(128), nullable=False)
    description      = Column(Text)                          # clinical description for admin review
    mechanism_code   = Column(String(16), ForeignKey("mechanisms.code"))
    is_gradable      = Column(Boolean, default=False)
    grades_json      = Column(JSON)                          # e.g. ["0","1","2","3","4"]
    grade_labels_json = Column(JSON)                         # e.g. {"0":"Absent","1":"Léger",...}
    urgency_override  = Column(String(4))                    # "P1" | "P2" | …
    is_approved       = Column(Boolean, default=True)
    created_at        = Column(DateTime, default=datetime.utcnow)
    updated_at        = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    mechanism = relationship("Mechanism", back_populates="diseases")


# ─────────────────────────── Anatomical regions ───────────────────────────

class AnatomicalRegion(Base):
    """Pre-defined + custom anatomical regions selectable before grid painting."""
    __tablename__ = "anatomical_regions"
    id        = Column(UUID(as_uuid=False), primary_key=True, default=_uid)
    name_fr   = Column(String(128), unique=True, nullable=False)
    is_custom = Column(Boolean, default=False)   # False = pre-loaded, True = doctor-added
    is_active = Column(Boolean, default=True)
    created_by = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)

    created_by_user = relationship("User", foreign_keys=[created_by])


# ─────────────────────────── Annotation session + annotation ───────────────────────────

class AnnotationSession(Base):
    __tablename__ = "annotation_sessions"
    id          = Column(UUID(as_uuid=False), primary_key=True, default=_uid)
    doctor_id   = Column(UUID(as_uuid=False), ForeignKey("users.id"))
    started_at  = Column(DateTime, default=datetime.utcnow)
    ended_at    = Column(DateTime)
    images_done = Column(Integer, default=0)

    doctor      = relationship("User", back_populates="sessions")
    annotations = relationship("Annotation", back_populates="session")


class Annotation(Base):
    __tablename__ = "annotations"
    id            = Column(UUID(as_uuid=False), primary_key=True, default=_uid)
    image_id      = Column(UUID(as_uuid=False), ForeignKey("fundus_images.id"))
    session_id    = Column(UUID(as_uuid=False), ForeignKey("annotation_sessions.id"))
    doctor_id     = Column(UUID(as_uuid=False), ForeignKey("users.id"))   # denormalised for fast query
    time_spent_sec = Column(Integer, default=0)
    status        = Column(String(16), default="draft")   # draft | submitted
    created_at    = Column(DateTime, default=datetime.utcnow)
    submitted_at  = Column(DateTime)

    disease_labels     = relationship("DiseaseLabel", back_populates="annotation", cascade="all, delete-orphan")
    regions            = relationship("RegionOfInterest", back_populates="annotation", cascade="all, delete-orphan")
    clinical_note      = relationship("ClinicalNote", back_populates="annotation", uselist=False)
    urgency_flag       = relationship("UrgencyFlag", back_populates="annotation", uselist=False)
    gradcam_validation = relationship("GradCAMValidation", back_populates="annotation", uselist=False)
    image              = relationship("FundusImage", back_populates="annotations")
    doctor             = relationship("User", back_populates="annotations", foreign_keys="[Annotation.doctor_id]")
    session            = relationship("AnnotationSession", back_populates="annotations")


class DiseaseLabel(Base):
    __tablename__ = "disease_labels"
    id            = Column(UUID(as_uuid=False), primary_key=True, default=_uid)
    annotation_id = Column(UUID(as_uuid=False), ForeignKey("annotations.id"))
    disease_code  = Column(String(32), ForeignKey("diseases.code"))
    grade         = Column(String(32))    # stored as raw value (retroactively updated on rename)
    is_custom     = Column(Boolean, default=False)

    annotation = relationship("Annotation", back_populates="disease_labels")
    disease    = relationship("Disease", foreign_keys=[disease_code])


# ─────────────────────────── Region of interest + grid ───────────────────────────

class RegionOfInterest(Base):
    """One selected anatomical region per annotation. Each has its own grid cells."""
    __tablename__ = "regions_of_interest"
    id                   = Column(UUID(as_uuid=False), primary_key=True, default=_uid)
    annotation_id        = Column(UUID(as_uuid=False), ForeignKey("annotations.id"), nullable=False)
    anatomical_region_id = Column(UUID(as_uuid=False), ForeignKey("anatomical_regions.id"), nullable=True)
    custom_region_name   = Column(String(128))          # if region is typed free-form
    max_zoom_reached     = Column(Integer, default=0)
    created_at           = Column(DateTime, default=datetime.utcnow)

    annotation = relationship("Annotation", back_populates="regions")
    cells      = relationship("GridCell", back_populates="region", cascade="all, delete-orphan")
    region_ref = relationship("AnatomicalRegion")


class GridCell(Base):
    __tablename__ = "grid_cells"
    id           = Column(UUID(as_uuid=False), primary_key=True, default=_uid)
    region_id    = Column(UUID(as_uuid=False), ForeignKey("regions_of_interest.id"), nullable=False)
    zoom_level   = Column(Integer, nullable=False)   # 0=8×8  1=16×16  2=32×32  3=64×64
    row          = Column(Integer, nullable=False)
    col          = Column(Integer, nullable=False)
    lesion_code  = Column(String(16), ForeignKey("lesion_vocabulary.code"))
    custom_label_text = Column(String(64))

    region = relationship("RegionOfInterest", back_populates="cells")
    lesion = relationship("LesionVocabulary", foreign_keys=[lesion_code])


class LesionVocabulary(Base):
    __tablename__ = "lesion_vocabulary"
    code     = Column(String(16), primary_key=True)
    name_fr  = Column(String(64), nullable=False)
    color_hex = Column(String(8))


# ─────────────────────────── Model predictions + Grad-CAM ───────────────────────────

class ModelVersion(Base):
    """Registry of all trained model checkpoints; supports rollback."""
    __tablename__ = "model_versions"
    id              = Column(UUID(as_uuid=False), primary_key=True, default=_uid)
    version_tag     = Column(String(64), unique=True, nullable=False)   # e.g. "v1.2.0"
    checkpoint_path = Column(String(512))
    description     = Column(Text)
    metrics_json    = Column(JSON)                                       # {auc, f1, …}
    is_active       = Column(Boolean, default=False)
    trained_at      = Column(DateTime)
    created_at      = Column(DateTime, default=datetime.utcnow)


class ModelPrediction(Base):
    __tablename__ = "model_predictions"
    id            = Column(UUID(as_uuid=False), primary_key=True, default=_uid)
    image_id      = Column(UUID(as_uuid=False), ForeignKey("fundus_images.id"))
    model_version = Column(String(64), nullable=False)
    top_k_json    = Column(JSON)       # [{disease_code, confidence, grade?}, ...]
    confidence    = Column(Float)
    gradcam_path  = Column(String(512))
    created_at    = Column(DateTime, default=datetime.utcnow)

    image               = relationship("FundusImage", back_populates="predictions")
    gradcam_validations = relationship("GradCAMValidation", back_populates="prediction")


class GradCAMValidation(Base):
    __tablename__ = "gradcam_validations"
    id            = Column(UUID(as_uuid=False), primary_key=True, default=_uid)
    annotation_id = Column(UUID(as_uuid=False), ForeignKey("annotations.id"))
    prediction_id = Column(UUID(as_uuid=False), ForeignKey("model_predictions.id"))
    verdict       = Column(String(16))   # correct | partial | wrong
    comment       = Column(Text)

    annotation = relationship("Annotation", back_populates="gradcam_validation")
    prediction = relationship("ModelPrediction", back_populates="gradcam_validations")


# ─────────────────────────── Urgency flag ───────────────────────────

class UrgencyFlag(Base):
    __tablename__ = "urgency_flags"
    id            = Column(UUID(as_uuid=False), primary_key=True, default=_uid)
    annotation_id = Column(UUID(as_uuid=False), ForeignKey("annotations.id"), unique=True)
    level         = Column(String(4))    # P1 | P2 | P3 | P4
    rule_triggered = Column(String(128))
    flagged_at    = Column(DateTime, default=datetime.utcnow)

    annotation = relationship("Annotation", back_populates="urgency_flag")


# ─────────────────────────── Clinical note ───────────────────────────

class ClinicalNote(Base):
    __tablename__ = "clinical_notes"
    id               = Column(UUID(as_uuid=False), primary_key=True, default=_uid)
    annotation_id    = Column(UUID(as_uuid=False), ForeignKey("annotations.id"), unique=True)
    text_content     = Column(Text)
    audio_path       = Column(String(512))
    transcription_fr = Column(Text)
    duration_sec     = Column(Integer)
    vlm_description  = Column(Text)    # Qwen2-VL auto-generated fundus description

    annotation = relationship("Annotation", back_populates="clinical_note")


# ─────────────────────────── Custom pathology proposal ───────────────────────────

class CustomPathologyProposal(Base):
    __tablename__ = "custom_pathology_proposals"
    id                     = Column(UUID(as_uuid=False), primary_key=True, default=_uid)
    doctor_id              = Column(UUID(as_uuid=False), ForeignKey("users.id"))
    image_id               = Column(UUID(as_uuid=False), ForeignKey("fundus_images.id"), nullable=True)
    proposed_name          = Column(String(128), nullable=False)
    proposed_description   = Column(Text)
    suspected_mechanism    = Column(String(16), ForeignKey("mechanisms.code"))
    proposed_grades_json   = Column(JSON)          # doctor's suggested grade list
    proposed_grade_labels  = Column(JSON)          # doctor's suggested grade label map
    urgency_suggestion     = Column(String(4))
    is_gradable            = Column(Boolean, default=False)
    status                 = Column(String(16), default="pending")   # pending | approved | rejected
    admin_notes            = Column(Text)
    reviewed_by            = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)
    reviewed_at            = Column(DateTime)
    resulting_disease_code = Column(String(32), ForeignKey("diseases.code"), nullable=True)
    created_at             = Column(DateTime, default=datetime.utcnow)

    doctor           = relationship("User", back_populates="custom_proposals", foreign_keys=[doctor_id])
    reviewer         = relationship("User", foreign_keys=[reviewed_by])
    image            = relationship("FundusImage", foreign_keys=[image_id])
    mechanism_ref    = relationship("Mechanism", foreign_keys=[suspected_mechanism])
    resulting_disease = relationship("Disease", foreign_keys=[resulting_disease_code])


# ─────────────────────────── Audit log ───────────────────────────

class AuditLog(Base):
    """Immutable change journal. Every mutation to catalog, annotations, or
    proposals by any user is appended here. Never deleted."""
    __tablename__ = "audit_logs"
    id          = Column(UUID(as_uuid=False), primary_key=True, default=_uid)
    user_id     = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)
    action      = Column(String(64), nullable=False)   # e.g. "disease.update", "proposal.approve"
    entity_type = Column(String(64))                   # "disease" | "annotation" | "proposal" | …
    entity_id   = Column(String(128))                  # PK of the modified entity
    old_value   = Column(JSON)                         # snapshot before change
    new_value   = Column(JSON)                         # snapshot after change
    detail      = Column(Text)                         # human-readable description
    created_at  = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="audit_logs", foreign_keys=[user_id])


# ─────────────────────────── Active learning config ───────────────────────────

class ActiveLearningConfig(Base):
    """Single-row config table for the active learning loop.
    Use id='singleton' as the primary key."""
    __tablename__ = "active_learning_config"
    id                  = Column(String(16), primary_key=True, default="singleton")
    uncertainty_threshold = Column(Float, default=0.7)
    n_samples_per_cycle = Column(Integer, default=50)
    auto_retrain        = Column(Boolean, default=False)   # disabled until stable
    current_version_tag = Column(String(64), nullable=True)
    last_retrain_at     = Column(DateTime, nullable=True)
    updated_at          = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
