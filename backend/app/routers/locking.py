"""
Patient-level image locking.

When a doctor opens any image of a patient, ALL pending images of that patient
are locked for that doctor. Other doctors see those images as locked in their queue.
Lock is released when the doctor submits the annotation or explicitly releases it.
"""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import FundusImage, ImageLock, Annotation
from .auth import get_current_user

router = APIRouter(prefix="/locks", tags=["locks"])


@router.post("/{image_id}/acquire")
def acquire_lock(
    image_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    img = db.query(FundusImage).filter(FundusImage.id == image_id).first()
    if not img:
        raise HTTPException(404, "Image introuvable")

    # Check if this image is already locked by someone else
    existing = db.query(ImageLock).filter(ImageLock.image_id == image_id).first()
    if existing and existing.locked_by != user.id:
        raise HTTPException(
            423,
            f"Image verrouillée par un autre médecin (patient {img.patient_id})",
        )
    if existing:
        return {"status": "already_yours", "patient_id": img.patient_id}

    # Lock ALL images of the same patient that are still pending/in_progress
    patient_images = (
        db.query(FundusImage)
        .filter(
            FundusImage.patient_id == img.patient_id,
            FundusImage.status.in_(["pending", "in_progress"]),
        )
        .all()
    )
    locked = []
    for pi in patient_images:
        already = db.query(ImageLock).filter(ImageLock.image_id == pi.id).first()
        if not already:
            db.add(ImageLock(image_id=pi.id, patient_id=img.patient_id, locked_by=user.id))
            locked.append(pi.id)
        elif already.locked_by != user.id:
            db.rollback()
            raise HTTPException(
                423,
                f"Un autre médecin a déjà verrouillé une image de ce patient",
            )

    # Mark the clicked image as in_progress
    img.status = "in_progress"
    db.commit()
    return {"status": "locked", "locked_images": locked, "patient_id": img.patient_id}


@router.delete("/{image_id}/release")
def release_lock(
    image_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    img = db.query(FundusImage).filter(FundusImage.id == image_id).first()
    if not img:
        raise HTTPException(404)

    # Release all patient locks held by this user
    patient_locks = (
        db.query(ImageLock)
        .filter(ImageLock.patient_id == img.patient_id, ImageLock.locked_by == user.id)
        .all()
    )
    for lock in patient_locks:
        # Only reset status if no submitted annotation exists
        locked_img = db.query(FundusImage).filter(FundusImage.id == lock.image_id).first()
        if locked_img and locked_img.status == "in_progress":
            has_submission = (
                db.query(Annotation)
                .filter(Annotation.image_id == lock.image_id, Annotation.status == "submitted")
                .first()
            )
            if not has_submission:
                locked_img.status = "pending"
        db.delete(lock)

    db.commit()
    return {"status": "released", "patient_id": img.patient_id}
