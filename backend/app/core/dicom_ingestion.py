"""
DICOM ingestion — parses .dcm files and upserts Patient + FundusImage records.

Extracts automatically from DICOM tags:
  (0010,0020) Patient ID      → clinical_id
  (0010,0010) Patient Name    → full_name
  (0010,0030) Birth Date      → birth_date + computed age
  (0010,0040) Patient Sex     → gender (M/F)
  (0020,0060) Laterality      → eye (OD/OS)
  (0008,0020) Study Date      → study_date / capture_date
  (0008,0018) SOP Instance UID → dicom_study_uid
  (0008,0060) Modality        → modality
  Source Application (NILAE/OPTOMAPPLUS) → modality tag enrichment

Usage:
    from app.core.dicom_ingestion import ingest_dicom_file
    record = ingest_dicom_file(db, "/path/to/image.dcm", jpeg_out_path="/app/data/images/xxx.jpg")
"""
import re
from datetime import datetime, date
from pathlib import Path
from sqlalchemy.orm import Session

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    import numpy as np
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from ..models import Patient, FundusImage


def _parse_date(tag_value: str | None) -> datetime | None:
    if not tag_value:
        return None
    try:
        return datetime.strptime(str(tag_value).strip(), "%Y%m%d")
    except ValueError:
        return None


def _compute_age(birth: datetime | None) -> int | None:
    if not birth:
        return None
    today = date.today()
    return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))


def _normalize_gender(raw: str | None) -> str | None:
    if not raw:
        return None
    r = str(raw).upper().strip()
    if r in ("M", "MALE", "MASCULIN"):
        return "M"
    if r in ("F", "FEMALE", "FEMININ", "FÉMININ"):
        return "F"
    return raw


def _normalize_laterality(ds) -> str:
    """Try (0020,0060) Laterality, then fall back to filename heuristic."""
    lat = getattr(ds, "Laterality", None) or getattr(ds, "ImageLaterality", None)
    if lat:
        lat = str(lat).upper().strip()
        if lat in ("R", "RIGHT", "OD"):
            return "OD"
        if lat in ("L", "LEFT", "OS"):
            return "OS"
    return "OD"   # default


def _normalize_modality(ds) -> str:
    src = str(getattr(ds, "SourceApplicationEntityTitle", "") or "").upper()
    img_type = getattr(ds, "ImageType", [])
    img_type_str = " ".join(str(x) for x in img_type).upper()

    if "OPTOMAPPLUS" in img_type_str or "OPTOMAP" in img_type_str:
        return "UWF"
    if "NILAE" in src or "OPTOMA" in src:
        return "UWF"
    modality = str(getattr(ds, "Modality", "STD") or "STD").upper()
    return modality if modality else "STD"


def _export_jpeg(ds, out_path: Path) -> bool:
    """Convert DICOM pixel data to JPEG. Returns True on success."""
    if not PIL_AVAILABLE:
        return False
    try:
        arr = ds.pixel_array.astype(float)
        arr = apply_voi_lut(arr, ds) if hasattr(ds, "WindowWidth") else arr
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6) * 255
        img = PILImage.fromarray(arr.astype("uint8"))
        if img.mode != "RGB":
            img = img.convert("RGB")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(out_path), "JPEG", quality=92)
        return True
    except Exception:
        return False


def ingest_dicom_file(
    db: Session,
    dcm_path: str,
    images_root: str = "/app/data/images",
) -> FundusImage | None:
    """
    Parse a DICOM file, upsert Patient and FundusImage in the DB.
    Returns the FundusImage record (new or existing).
    """
    if not PYDICOM_AVAILABLE:
        raise RuntimeError("pydicom is not installed — add it to requirements.txt")

    dcm_path = Path(dcm_path)
    if not dcm_path.exists():
        raise FileNotFoundError(str(dcm_path))

    ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=False)

    # ── Extract patient metadata ──────────────────────────────────────────
    clinical_id = str(getattr(ds, "PatientID", "") or "").strip() or dcm_path.stem
    full_name   = str(getattr(ds, "PatientName", "") or "").strip() or None
    birth_raw   = str(getattr(ds, "PatientBirthDate", "") or "").strip() or None
    birth_date  = _parse_date(birth_raw)
    gender      = _normalize_gender(str(getattr(ds, "PatientSex", "") or ""))

    # ── Upsert patient ────────────────────────────────────────────────────
    patient = db.query(Patient).filter(Patient.clinical_id == clinical_id).first()
    if not patient:
        patient = Patient(
            clinical_id=clinical_id,
            full_name=full_name,
            birth_date=birth_date,
            age=_compute_age(birth_date),
            gender=gender,
        )
        db.add(patient)
        db.flush()
    else:
        if full_name and not patient.full_name:
            patient.full_name = full_name
        if birth_date and not patient.birth_date:
            patient.birth_date = birth_date
            patient.age = _compute_age(birth_date)
        if gender and not patient.gender:
            patient.gender = gender

    # ── Extract image metadata ────────────────────────────────────────────
    sop_uid     = str(getattr(ds, "SOPInstanceUID", "") or "").strip() or None
    study_date  = _parse_date(str(getattr(ds, "StudyDate", "") or "").strip() or None)
    eye         = _normalize_laterality(ds)
    modality    = _normalize_modality(ds)

    # ── Avoid duplicates by SOP UID or file path ──────────────────────────
    existing = None
    if sop_uid:
        existing = db.query(FundusImage).filter(FundusImage.dicom_study_uid == sop_uid).first()
    if not existing:
        existing = db.query(FundusImage).filter(FundusImage.dicom_path == str(dcm_path)).first()

    if existing:
        db.commit()
        return existing

    # ── Export JPEG ───────────────────────────────────────────────────────
    jpeg_name = f"{sop_uid or dcm_path.stem}.jpg"
    jpeg_path = Path(images_root) / jpeg_name
    exported  = _export_jpeg(ds, jpeg_path)
    file_path = str(jpeg_path) if exported else str(dcm_path)

    # ── Create FundusImage ────────────────────────────────────────────────
    img = FundusImage(
        patient_id=patient.id,
        file_path=file_path,
        dicom_path=str(dcm_path),
        modality=modality,
        eye=eye,
        capture_date=study_date,
        study_date=study_date,
        dicom_study_uid=sop_uid,
        status="pending",
        uncertainty_score=0.5,
    )
    db.add(img)
    db.commit()
    db.refresh(img)
    return img


def ingest_dicom_folder(
    db: Session,
    folder: str,
    images_root: str = "/app/data/images",
) -> list[str]:
    """Walk a folder recursively and ingest all .dcm files. Returns list of image IDs."""
    folder = Path(folder)
    ids = []
    for dcm in folder.rglob("*.dcm"):
        try:
            img = ingest_dicom_file(db, str(dcm), images_root)
            if img:
                ids.append(img.id)
        except Exception as e:
            import logging
            logging.getLogger("retinai.dicom").warning("Failed to ingest %s: %s", dcm, e)
    return ids
