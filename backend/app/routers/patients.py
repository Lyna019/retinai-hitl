"""
Patient gallery — grouped by patient, with systemic diseases and historical note.
"""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import (
    Patient, FundusImage, PatientSystemicDisease,
    SystemicDisease, ImageLock, Annotation,
)
from ..schemas import (
    PatientOut, PatientDetailOut, PatientSystemicIn,
    PatientNoteIn, SystemicDiseaseOut, ImageOut,
)
from .auth import get_current_user

router = APIRouter(prefix="/patients", tags=["patients"])


def _image_out(img: FundusImage, lock: ImageLock | None, patient: "Patient | None" = None) -> ImageOut:
    return ImageOut(
        id=img.id,
        patient_id=img.patient_id or "",
        patient_clinical_id=patient.clinical_id if patient else "",
        eye=img.eye or "",
        modality=img.modality or "",
        capture_date=img.capture_date,
        status=img.status,
        uncertainty=img.uncertainty_score or 0.5,
        file_url=f"/api/images/{img.id}/file",
        locked_by=lock.locked_by if lock else None,
    )


@router.get("", response_model=list[PatientOut])
def list_patients(
    date_from: datetime | None = Query(None),
    date_to:   datetime | None = Query(None),
    sort:      str = Query("id", description="id | date"),
    limit:     int = 200,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    q = db.query(Patient)
    rows = q.all()

    result = []
    for p in rows:
        img_q = db.query(FundusImage).filter(FundusImage.patient_id == p.id)
        if date_from:
            img_q = img_q.filter(FundusImage.capture_date >= date_from)
        if date_to:
            img_q = img_q.filter(FundusImage.capture_date <= date_to)
        images = img_q.all()

        if not images:
            continue   # hide patients with no images in the date range

        sd_ids = [
            str(x.systemic_disease_id)
            for x in db.query(PatientSystemicDisease)
            .filter(PatientSystemicDisease.patient_id == p.id).all()
        ]
        last_date = max((i.capture_date for i in images if i.capture_date), default=None)

        result.append(PatientOut(
            id=p.id,
            clinical_id=p.clinical_id or "",
            full_name=p.full_name,
            age=p.age,
            gender=p.gender,
            historical_note=p.historical_note,
            systemic_disease_ids=sd_ids,
            image_count=len(images),
            last_capture_date=last_date,
        ))

    if sort == "date":
        result.sort(key=lambda x: x.last_capture_date or datetime.min, reverse=True)
    else:
        result.sort(key=lambda x: x.clinical_id)

    return result[:limit]


@router.get("/{patient_id}", response_model=PatientDetailOut)
def get_patient(
    patient_id: str,
    date_from: datetime | None = Query(None),
    date_to:   datetime | None = Query(None),
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    p = db.query(Patient).filter(Patient.id == patient_id).first()
    if not p:
        # Also accept clinical_id (e.g. "P-2040") from PatientInfoPanel
        p = db.query(Patient).filter(Patient.clinical_id == patient_id).first()
    if not p:
        raise HTTPException(404, "Patient introuvable")

    # Systemic diseases
    sd_links = db.query(PatientSystemicDisease).filter(
        PatientSystemicDisease.patient_id == patient_id
    ).all()
    sd_list = []
    for link in sd_links:
        sd = db.query(SystemicDisease).filter(SystemicDisease.id == link.systemic_disease_id).first()
        if sd:
            sd_list.append(SystemicDiseaseOut(id=sd.id, name_fr=sd.name_fr, category=sd.category))

    # Images
    img_q = db.query(FundusImage).filter(FundusImage.patient_id == patient_id)
    if date_from:
        img_q = img_q.filter(FundusImage.capture_date >= date_from)
    if date_to:
        img_q = img_q.filter(FundusImage.capture_date <= date_to)
    images = img_q.order_by(FundusImage.capture_date.desc()).all()

    image_outs = []
    for img in images:
        lock = db.query(ImageLock).filter(ImageLock.image_id == img.id).first()
        image_outs.append(_image_out(img, lock, p))

    return PatientDetailOut(
        id=p.id,
        clinical_id=p.clinical_id or "",
        full_name=p.full_name,
        age=p.age,
        gender=p.gender,
        historical_note=p.historical_note,
        systemic_diseases=sd_list,
        images=image_outs,
    )


@router.put("/{patient_id}/systemic-diseases")
def set_systemic_diseases(
    patient_id: str,
    payload: PatientSystemicIn,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    p = db.query(Patient).filter(Patient.id == patient_id).first()
    if not p:
        raise HTTPException(404, "Patient introuvable")

    # Replace existing links
    db.query(PatientSystemicDisease).filter(
        PatientSystemicDisease.patient_id == patient_id
    ).delete()

    for sd_id in payload.systemic_disease_ids:
        db.add(PatientSystemicDisease(patient_id=patient_id, systemic_disease_id=sd_id))

    db.commit()
    return {"status": "ok", "count": len(payload.systemic_disease_ids)}


@router.put("/{patient_id}/note")
def update_historical_note(
    patient_id: str,
    payload: PatientNoteIn,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    p = db.query(Patient).filter(Patient.id == patient_id).first()
    if not p:
        raise HTTPException(404, "Patient introuvable")
    p.historical_note = payload.historical_note
    db.commit()
    return {"status": "ok"}
