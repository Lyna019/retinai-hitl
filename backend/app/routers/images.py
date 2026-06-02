"""
Image queue + DICOM ingestion + clinical DB sync.
"""
import os
from datetime import datetime, timedelta
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import FundusImage, ModelPrediction, ImageLock, Patient, Annotation
from ..schemas import ImageOut, ImageQualityIn, PredictionOut

LOCK_TTL_HOURS = 4
from ..config import settings
from .auth import get_current_user, require_admin

router = APIRouter(prefix="/images", tags=["images"])


def _img_out(r: FundusImage, lock: ImageLock | None = None, patient: Patient | None = None) -> ImageOut:
    return ImageOut(
        id=r.id,
        patient_id=r.patient_id or "",
        patient_clinical_id=patient.clinical_id if patient else "",
        eye=r.eye or "",
        modality=r.modality or "",
        capture_date=r.capture_date,
        status=r.status,
        uncertainty=r.uncertainty_score or 0.5,
        model_urgency=r.model_urgency,
        file_url=f"/api/images/{r.id}/file",
        locked_by=lock.locked_by if lock else None,
        image_quality=r.image_quality,
    )


@router.get("", response_model=list[ImageOut])
def list_images(
    status: str | None = Query(None),
    sort: str = Query("uncertainty"),
    limit: int = 2000,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    # Auto-release locks idle for more than LOCK_TTL_HOURS
    cutoff = datetime.utcnow() - timedelta(hours=LOCK_TTL_HOURS)
    stale = db.query(ImageLock).filter(ImageLock.locked_at < cutoff).all()
    for lk in stale:
        img = db.query(FundusImage).filter(FundusImage.id == lk.image_id).first()
        if img and img.status == "in_progress":
            has_sub = db.query(Annotation).filter(
                Annotation.image_id == img.id, Annotation.status == "submitted"
            ).first()
            if not has_sub:
                img.status = "pending"
        db.delete(lk)
    if stale:
        db.commit()

    q = db.query(FundusImage)
    if status:
        q = q.filter(FundusImage.status == status)
    # Doctors with assigned images only see their own; admins see all
    if user.role == "doctor" and hasattr(FundusImage, 'assigned_to'):
        from sqlalchemy import or_
        q = q.filter(
            or_(FundusImage.assigned_to == user.id, FundusImage.assigned_to == None)
        ).filter(
            # If has assignment, only show assigned; if no assignment exists at all show unassigned P1/P2
            or_(
                FundusImage.assigned_to == user.id,
                FundusImage.model_urgency.in_(["P1", "P2", "P3"])
            )
        )

    if sort == "uncertainty":
        # P1 → P2 → P3 → P4/None, then by uncertainty desc within each group
        from sqlalchemy import case
        urgency_order = case(
            (FundusImage.model_urgency == "P1", 1),
            (FundusImage.model_urgency == "P2", 2),
            (FundusImage.model_urgency == "P3", 3),
            else_=4,
        )
        q = q.order_by(urgency_order, FundusImage.uncertainty_score.desc())
    elif sort == "date":
        q = q.order_by(FundusImage.capture_date.desc())

    rows = q.limit(limit).all()
    result = []
    for r in rows:
        lock = db.query(ImageLock).filter(ImageLock.image_id == r.id).first()
        # Hide images locked by another doctor from the pending queue
        if lock and lock.locked_by != user.id and r.status == "pending":
            continue
        patient = db.query(Patient).filter(Patient.id == r.patient_id).first() if r.patient_id else None
        result.append(_img_out(r, lock, patient))
    return result


@router.get("/{image_id}", response_model=ImageOut)
def get_image(image_id: str, db: Session = Depends(get_db), _user=Depends(get_current_user)):
    r = db.query(FundusImage).filter(FundusImage.id == image_id).first()
    if not r:
        raise HTTPException(404, "Image introuvable")
    lock = db.query(ImageLock).filter(ImageLock.image_id == image_id).first()
    patient = db.query(Patient).filter(Patient.id == r.patient_id).first() if r.patient_id else None
    return _img_out(r, lock, patient)


@router.get("/{image_id}/file")
def serve_image_file(image_id: str, db: Session = Depends(get_db), _user=Depends(get_current_user)):
    """Redirect to static file or stream from disk."""
    from fastapi.responses import FileResponse
    r = db.query(FundusImage).filter(FundusImage.id == image_id).first()
    if not r:
        raise HTTPException(404)
    fp = Path(r.file_path)
    if not fp.exists():
        raise HTTPException(404, f"Fichier introuvable sur disque : {r.file_path}")
    return FileResponse(str(fp))


@router.get("/{image_id}/predictions", response_model=PredictionOut)
def get_predictions(image_id: str, db: Session = Depends(get_db), _user=Depends(get_current_user)):
    pred = (
        db.query(ModelPrediction)
        .filter(ModelPrediction.image_id == image_id)
        .order_by(ModelPrediction.created_at.desc())
        .first()
    )
    if not pred:
        raise HTTPException(404, "Aucune prédiction disponible")
    return PredictionOut(
        model_version=pred.model_version,
        top_k=pred.top_k_json or [],
        gradcam_url=f"/api/files/gradcam/{image_id}.png" if (pred.gradcam_path and os.path.exists(pred.gradcam_path)) else None,
        uncertainty=pred.confidence or 0.5,
    )


@router.patch("/{image_id}/quality")
def set_image_quality(
    image_id: str,
    payload: ImageQualityIn,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    if payload.quality not in ("good", "fair", "poor"):
        raise HTTPException(422, "Valeur invalide : good | fair | poor")
    r = db.query(FundusImage).filter(FundusImage.id == image_id).first()
    if not r:
        raise HTTPException(404, "Image introuvable")
    r.image_quality = payload.quality
    db.commit()
    return {"image_quality": r.image_quality}


@router.post("/ingest-dicom", dependencies=[Depends(require_admin)])
async def ingest_dicom(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload a .dcm file, parse it, and create Patient + FundusImage records."""
    from ..core.dicom_ingestion import ingest_dicom_file

    dicom_dir = Path(settings.dicom_root)
    dicom_dir.mkdir(parents=True, exist_ok=True)
    dest = dicom_dir / file.filename
    with open(dest, "wb") as f:
        f.write(await file.read())

    try:
        img = ingest_dicom_file(db, str(dest), settings.images_root)
        return {"status": "ingested", "image_id": img.id, "patient_id": img.patient_id}
    except Exception as exc:
        raise HTTPException(500, f"Erreur DICOM : {exc}")


@router.post("/ingest-dicom-folder", dependencies=[Depends(require_admin)])
def ingest_dicom_folder_endpoint(
    folder_path: str = Query(..., description="Chemin absolu du dossier DICOM sur le serveur"),
    db: Session = Depends(get_db),
):
    """Ingest all .dcm files in a server-side folder."""
    from ..core.dicom_ingestion import ingest_dicom_folder

    if not Path(folder_path).exists():
        raise HTTPException(400, f"Dossier introuvable : {folder_path}")
    ids = ingest_dicom_folder(db, folder_path, settings.images_root)
    return {"status": "ok", "ingested": len(ids), "image_ids": ids}


@router.post("/sync", dependencies=[Depends(require_admin)])
def trigger_sync():
    """
    Trigger a clinical-DB → platform image sync.
    Implementation: connect to settings.clinical_db_url, query new fundus images,
    resolve patient ID, copy files, call ingest_dicom_folder.
    """
    return {
        "status": "stub",
        "message": (
            "Implémentez fetch_new_images_from_clinical_db() : "
            "connectez-vous à HOSPITAL_DB_URL, requêtez les nouvelles images, "
            "copiez les .dcm dans DICOM_ROOT, appelez POST /api/images/ingest-dicom-folder."
        ),
    }
