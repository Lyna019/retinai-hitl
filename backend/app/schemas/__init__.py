"""Pydantic request/response schemas — v3."""
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel


# ─── Auth ───────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


# ─── Images ─────────────────────────────────────────────────────────────────

class ImageQualityIn(BaseModel):
    quality: str  # good | fair | poor


class ImageOut(BaseModel):
    id: str
    patient_id: str           # UUID — used for API calls
    patient_clinical_id: str = ""  # short ID like "P-2040" — used for display
    eye: str
    modality: str
    capture_date: datetime | None = None
    status: str
    uncertainty: float        # renamed from uncertainty_score for frontend consistency
    file_url: str
    locked_by: str | None = None
    image_quality: str | None = None


# ─── Patients ───────────────────────────────────────────────────────────────

class PatientSystemicIn(BaseModel):
    systemic_disease_ids: list[str]


class PatientNoteIn(BaseModel):
    historical_note: str


class SystemicDiseaseOut(BaseModel):
    id: str
    name_fr: str
    category: str | None = None
    is_active: bool = True


class SystemicDiseaseIn(BaseModel):
    name_fr: str
    category: str | None = None


class PatientOut(BaseModel):
    id: str
    clinical_id: str
    full_name: str | None = None
    age: int | None = None
    gender: str | None = None
    historical_note: str | None = None
    systemic_disease_ids: list[str] = []
    image_count: int = 0
    last_capture_date: datetime | None = None


class PatientDetailOut(BaseModel):
    id: str
    clinical_id: str
    full_name: str | None = None
    age: int | None = None
    gender: str | None = None
    historical_note: str | None = None
    systemic_diseases: list[SystemicDiseaseOut] = []
    images: list[ImageOut] = []


# ─── Disease catalog ────────────────────────────────────────────────────────

class DiseaseOut(BaseModel):
    code: str
    name_fr: str
    description: str | None = None
    mechanism: str | None = None       # frontend key (was mechanism_code)
    gradable: bool                     # frontend key (was is_gradable)
    grades: list[str] | None = None
    grade_labels_json: dict | None = None  # frontend key (was grade_labels)
    urgency: str | None = None         # frontend key (was urgency_override)
    is_approved: bool = True
    common: bool = False               # True for the 9 quick-select chips


class MechanismOut(BaseModel):
    code: str
    name_fr: str
    description: str | None = None
    diseases: list[DiseaseOut] = []


class DiseaseCreateIn(BaseModel):
    code: str
    name_fr: str
    description: str | None = None
    mechanism_code: str | None = None
    is_gradable: bool = False
    grades_json: list[str] | None = None
    grade_labels_json: dict | None = None
    urgency_override: str | None = None


class DiseaseUpdateIn(BaseModel):
    name_fr: str | None = None
    description: str | None = None
    mechanism_code: str | None = None
    is_gradable: bool | None = None
    grades_json: list[str] | None = None
    grade_labels_json: dict | None = None
    urgency_override: str | None = None


# ─── Anatomical regions ─────────────────────────────────────────────────────

class AnatomicalRegionOut(BaseModel):
    id: str
    name_fr: str
    is_custom: bool


class AnatomicalRegionIn(BaseModel):
    name_fr: str


# ─── Annotation payloads ────────────────────────────────────────────────────

class DiseaseLabelIn(BaseModel):
    disease_code: str
    grade: Optional[str] = None


class GridCellIn(BaseModel):
    zoom_level: int
    row: int
    col: int
    lesion_code: Optional[str] = None
    custom_label_text: Optional[str] = None


class RegionOfInterestIn(BaseModel):
    anatomical_region_id: Optional[str] = None
    custom_region_name: Optional[str] = None
    cells: list[GridCellIn] = []


class AnnotationSubmit(BaseModel):
    image_id: str
    disease_labels: list[DiseaseLabelIn]
    regions: list[RegionOfInterestIn] = []
    gradcam_verdict: Optional[str] = None     # correct | partial | wrong
    notes_text: Optional[str] = None
    time_spent_sec: int = 0


class AnnotationOut(BaseModel):
    id: str
    image_id: str
    status: str
    urgency_level: str | None = None
    mechanisms: list[str] = []
    doctor_id: str | None = None
    submitted_at: datetime | None = None
    # populated for draft annotations (to restore UI state)
    disease_labels: list[dict] = []
    regions: list[dict] = []
    gradcam_verdict: str | None = None
    notes_text: str | None = None


# ─── Predictions ────────────────────────────────────────────────────────────

class PredictionOut(BaseModel):
    model_version: str
    top_k: list[dict]
    gradcam_url: Optional[str] = None
    uncertainty: float


# ─── Proposals ──────────────────────────────────────────────────────────────

class ProposalIn(BaseModel):
    proposed_name: str
    proposed_description: str | None = None
    suspected_mechanism: str
    is_gradable: bool = False
    proposed_grades_json: list[str] | None = None
    proposed_grade_labels: dict | None = None
    urgency_suggestion: str | None = None
    image_id: str | None = None


class ProposalApproveIn(BaseModel):
    """Admin edits pathology details directly on the approval screen."""
    final_name: str
    final_description: str | None = None
    mechanism_code: str
    is_gradable: bool = False
    grades_json: list[str] | None = None
    grade_labels_json: dict | None = None
    urgency_override: str | None = None
    admin_notes: str | None = None


class ProposalOut(BaseModel):
    id: str
    doctor_id: str
    image_id: str | None = None
    proposed_name: str
    proposed_description: str | None = None
    suspected_mechanism: str
    is_gradable: bool
    proposed_grades_json: list[str] | None = None
    status: str
    admin_notes: str | None = None
    created_at: datetime


# ─── Transcription ──────────────────────────────────────────────────────────

class TranscriptionOut(BaseModel):
    text: str
    language: str = "fr"
    duration_sec: float | None = None


# ─── VLM ────────────────────────────────────────────────────────────────────

class VLMDescribeOut(BaseModel):
    description: str
    image_id: str
    model: str = "Qwen2-VL"


# ─── Audit log ──────────────────────────────────────────────────────────────

class AuditLogOut(BaseModel):
    id: str
    user_id: str | None = None
    action: str
    entity_type: str | None = None
    entity_id: str | None = None
    old_value: Any | None = None
    new_value: Any | None = None
    detail: str | None = None
    created_at: datetime


# ─── Model versions ─────────────────────────────────────────────────────────

class ModelVersionOut(BaseModel):
    id: str
    version_tag: str
    description: str | None = None
    metrics_json: dict | None = None
    is_active: bool
    trained_at: datetime | None = None
    created_at: datetime


class ModelVersionIn(BaseModel):
    version_tag: str
    checkpoint_path: str | None = None
    description: str | None = None
    metrics_json: dict | None = None
    trained_at: datetime | None = None


# ─── Active learning ────────────────────────────────────────────────────────

class ActiveLearningConfigOut(BaseModel):
    uncertainty_threshold: float
    n_samples_per_cycle: int
    auto_retrain: bool
    current_version_tag: str | None = None
    last_retrain_at: datetime | None = None


class ActiveLearningConfigIn(BaseModel):
    uncertainty_threshold: float | None = None
    n_samples_per_cycle: int | None = None
    auto_retrain: bool | None = None


# ─── Doctor management ──────────────────────────────────────────────────────

class DoctorCreateIn(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    password: str
    role: str = "doctor"


class DoctorOut(BaseModel):
    id: str
    username: str
    email: str | None = None
    full_name: str | None = None
    role: str
    is_active: bool
    created_at: datetime
