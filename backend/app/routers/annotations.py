"""
Annotation create / update / submit + auto-derived urgency & mechanisms.
Includes: region-of-interest grid, audit log for edits, hospital write-back.
"""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import (
    Annotation, AnnotationSession, DiseaseLabel, RegionOfInterest, GridCell,
    GradCAMValidation, UrgencyFlag, ClinicalNote, FundusImage,
    ModelPrediction, Patient, Disease, User, AuditLog, ImageLock,
)
from ..schemas import AnnotationSubmit, AnnotationOut
from ..core.rule_engine import compute_mechanisms, compute_urgency
from .auth import get_current_user

router = APIRouter(prefix="/annotations", tags=["annotations"])


def _build_annotation_out(annotation: Annotation, urgency, mechanisms: list) -> AnnotationOut:
    return AnnotationOut(
        id=annotation.id,
        image_id=annotation.image_id,
        status=annotation.status,
        urgency_level=urgency.level if urgency else None,
        mechanisms=mechanisms,
        doctor_id=annotation.doctor_id,
        submitted_at=annotation.submitted_at,
    )


@router.get("/image/{image_id}", response_model=list[AnnotationOut])
def get_annotations_for_image(
    image_id: str,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """Returns all annotations for an image (any doctor). Used in patient gallery and draft restore."""
    rows = db.query(Annotation).filter(Annotation.image_id == image_id).all()
    result = []
    for ann in rows:
        labels = db.query(DiseaseLabel).filter(DiseaseLabel.annotation_id == ann.id).all()
        label_dicts = [{"disease_code": l.disease_code, "grade": l.grade} for l in labels]
        urgency    = compute_urgency(label_dicts)
        mechanisms = compute_mechanisms(label_dicts)
        uf = db.query(UrgencyFlag).filter(UrgencyFlag.annotation_id == ann.id).first()

        # Load regions + cells for draft restore
        regions_out = []
        for roi in ann.regions:
            cells_out = [
                {"zoom_level": c.zoom_level, "row": c.row, "col": c.col, "lesion_code": c.lesion_code}
                for c in roi.cells
            ]
            regions_out.append({
                "anatomical_region_id": roi.anatomical_region_id,
                "custom_region_name": roi.custom_region_name,
                "cells": cells_out,
            })

        note = ann.clinical_note
        gcv  = ann.gradcam_validation

        result.append(AnnotationOut(
            id=ann.id,
            image_id=ann.image_id,
            status=ann.status,
            urgency_level=uf.level if uf else (urgency.level if urgency else None),
            mechanisms=mechanisms,
            doctor_id=ann.doctor_id,
            submitted_at=ann.submitted_at,
            disease_labels=label_dicts,
            regions=regions_out,
            gradcam_verdict=gcv.verdict if gcv else None,
            notes_text=note.text_content if note else None,
        ))
    return result


@router.post("/submit", response_model=AnnotationOut)
def submit_annotation(
    payload: AnnotationSubmit,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    image = db.query(FundusImage).filter(FundusImage.id == payload.image_id).first()
    if not image:
        raise HTTPException(404, "Image introuvable")

    # Fetch draft for this doctor, or create fresh
    annotation = (
        db.query(Annotation)
        .filter(
            Annotation.image_id == payload.image_id,
            Annotation.doctor_id == user.id,
            Annotation.status == "draft",
        )
        .first()
    )
    is_new = annotation is None
    if is_new:
        annotation = Annotation(
            image_id=payload.image_id,
            doctor_id=user.id,
            status="draft",
        )
        db.add(annotation)
        db.flush()
    else:
        # Log that a different doctor is editing someone else's annotation
        if annotation.doctor_id and annotation.doctor_id != user.id:
            db.add(AuditLog(
                user_id=user.id,
                action="annotation.edit_other",
                entity_type="annotation",
                entity_id=annotation.id,
                detail=f"Dr {user.username} a modifié l'annotation de {annotation.doctor_id} sur image {payload.image_id}",
            ))

    # ── Disease labels ───────────────────────────────────────────────────
    old_labels = db.query(DiseaseLabel).filter(DiseaseLabel.annotation_id == annotation.id).all()
    old_snap = [{"code": l.disease_code, "grade": l.grade} for l in old_labels]

    db.query(DiseaseLabel).filter(DiseaseLabel.annotation_id == annotation.id).delete()
    for lbl in payload.disease_labels:
        db.add(DiseaseLabel(
            annotation_id=annotation.id,
            disease_code=lbl.disease_code,
            grade=lbl.grade,
        ))

    new_snap = [{"code": l.disease_code, "grade": l.grade} for l in payload.disease_labels]
    if not is_new and old_snap != new_snap:
        db.add(AuditLog(
            user_id=user.id,
            action="annotation.labels_changed",
            entity_type="annotation",
            entity_id=annotation.id,
            old_value=old_snap,
            new_value=new_snap,
            detail=f"Modification des pathologies sur image {payload.image_id}",
        ))

    # ── Regions of interest + grid cells ────────────────────────────────
    for roi in db.query(RegionOfInterest).filter(RegionOfInterest.annotation_id == annotation.id).all():
        db.query(GridCell).filter(GridCell.region_id == roi.id).delete()
    db.query(RegionOfInterest).filter(RegionOfInterest.annotation_id == annotation.id).delete()

    for region_in in payload.regions:
        roi = RegionOfInterest(
            annotation_id=annotation.id,
            anatomical_region_id=region_in.anatomical_region_id,
            custom_region_name=region_in.custom_region_name,
            max_zoom_reached=max((c.zoom_level for c in region_in.cells), default=0),
        )
        db.add(roi)
        db.flush()
        for c in region_in.cells:
            db.add(GridCell(
                region_id=roi.id,
                zoom_level=c.zoom_level,
                row=c.row,
                col=c.col,
                lesion_code=c.lesion_code,
                custom_label_text=c.custom_label_text,
            ))

    # ── Grad-CAM verdict ─────────────────────────────────────────────────
    if payload.gradcam_verdict:
        db.query(GradCAMValidation).filter(GradCAMValidation.annotation_id == annotation.id).delete()
        pred = (
            db.query(ModelPrediction)
            .filter(ModelPrediction.image_id == payload.image_id)
            .order_by(ModelPrediction.created_at.desc())
            .first()
        )
        db.add(GradCAMValidation(
            annotation_id=annotation.id,
            verdict=payload.gradcam_verdict,
            prediction_id=pred.id if pred else None,
        ))

    # ── Clinical note ────────────────────────────────────────────────────
    if payload.notes_text is not None:
        existing_note = db.query(ClinicalNote).filter(ClinicalNote.annotation_id == annotation.id).first()
        if existing_note:
            existing_note.text_content = payload.notes_text
        else:
            db.add(ClinicalNote(annotation_id=annotation.id, text_content=payload.notes_text))

    # ── Rule engine ──────────────────────────────────────────────────────
    label_dicts = [{"disease_code": l.disease_code, "grade": l.grade} for l in payload.disease_labels]
    urgency   = compute_urgency(label_dicts)
    mechanisms = compute_mechanisms(label_dicts)

    db.query(UrgencyFlag).filter(UrgencyFlag.annotation_id == annotation.id).delete()
    if urgency:
        db.add(UrgencyFlag(
            annotation_id=annotation.id,
            level=urgency.level,
            rule_triggered=urgency.rule,
        ))

    annotation.status         = "submitted"
    annotation.submitted_at   = datetime.utcnow()
    annotation.time_spent_sec = payload.time_spent_sec
    image.status              = "done"

    # ── Session tracking ─────────────────────────────────────────────────
    session_row = (
        db.query(AnnotationSession)
        .filter(AnnotationSession.doctor_id == user.id, AnnotationSession.ended_at == None)
        .first()
    )
    if not session_row:
        session_row = AnnotationSession(doctor_id=user.id)
        db.add(session_row)
        db.flush()
    annotation.session_id   = session_row.id
    session_row.images_done = (session_row.images_done or 0) + 1

    # ── Release patient lock ─────────────────────────────────────────────
    db.query(ImageLock).filter(
        ImageLock.patient_id == image.patient_id,
        ImageLock.locked_by == user.id,
    ).delete()

    db.commit()

    # ── Hospital write-back (non-blocking) ───────────────────────────────
    try:
        from ..core.hospital_writeback import write_back_diagnosis
        patient = db.query(Patient).filter(Patient.id == image.patient_id).first()
        if patient:
            disease_details = []
            for lbl in payload.disease_labels:
                d = db.query(Disease).filter(Disease.code == lbl.disease_code).first()
                disease_details.append({
                    "code": lbl.disease_code,
                    "grade": lbl.grade,
                    "name_fr": d.name_fr if d else lbl.disease_code,
                })
            write_back_diagnosis(
                clinical_id=patient.clinical_id or patient.id,
                eye=image.eye or "?",
                disease_labels=disease_details,
                urgency=urgency.level if urgency else None,
                notes_text=payload.notes_text,
                annotated_by=user.full_name or user.username,
                annotation_date=annotation.submitted_at,
                study_date=image.study_date,
            )
    except Exception:
        pass   # write-back failure never blocks the doctor

    return _build_annotation_out(annotation, urgency, mechanisms)


@router.post("/{image_id}/draft")
def save_draft(
    image_id: str,
    payload: AnnotationSubmit,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    image = db.query(FundusImage).filter(FundusImage.id == image_id).first()
    if not image:
        raise HTTPException(404, "Image introuvable")

    annotation = (
        db.query(Annotation)
        .filter(
            Annotation.image_id == image_id,
            Annotation.doctor_id == user.id,
            Annotation.status == "draft",
        )
        .first()
    )
    if not annotation:
        annotation = Annotation(image_id=image_id, doctor_id=user.id, status="draft")
        db.add(annotation)
        db.flush()

    # Persist disease labels
    db.query(DiseaseLabel).filter(DiseaseLabel.annotation_id == annotation.id).delete()
    for lbl in payload.disease_labels:
        db.add(DiseaseLabel(
            annotation_id=annotation.id,
            disease_code=lbl.disease_code,
            grade=lbl.grade,
        ))

    # Persist regions + grid cells
    for roi in db.query(RegionOfInterest).filter(RegionOfInterest.annotation_id == annotation.id).all():
        db.query(GridCell).filter(GridCell.region_id == roi.id).delete()
    db.query(RegionOfInterest).filter(RegionOfInterest.annotation_id == annotation.id).delete()
    for region_in in payload.regions:
        roi = RegionOfInterest(
            annotation_id=annotation.id,
            anatomical_region_id=region_in.anatomical_region_id,
            custom_region_name=region_in.custom_region_name,
            max_zoom_reached=max((c.zoom_level for c in region_in.cells), default=0),
        )
        db.add(roi)
        db.flush()
        for c in region_in.cells:
            db.add(GridCell(
                region_id=roi.id,
                zoom_level=c.zoom_level,
                row=c.row,
                col=c.col,
                lesion_code=c.lesion_code,
            ))

    # Persist clinical note
    if payload.notes_text is not None:
        note = db.query(ClinicalNote).filter(ClinicalNote.annotation_id == annotation.id).first()
        if note:
            note.text_content = payload.notes_text
        else:
            db.add(ClinicalNote(annotation_id=annotation.id, text_content=payload.notes_text))

    # Persist Grad-CAM verdict
    if payload.gradcam_verdict:
        db.query(GradCAMValidation).filter(GradCAMValidation.annotation_id == annotation.id).delete()
        pred = (
            db.query(ModelPrediction)
            .filter(ModelPrediction.image_id == image_id)
            .order_by(ModelPrediction.created_at.desc())
            .first()
        )
        db.add(GradCAMValidation(
            annotation_id=annotation.id,
            verdict=payload.gradcam_verdict,
            prediction_id=pred.id if pred else None,
        ))

    # Link to open session (create one if needed)
    if not annotation.session_id:
        session_row = (
            db.query(AnnotationSession)
            .filter(AnnotationSession.doctor_id == user.id, AnnotationSession.ended_at == None)
            .first()
        )
        if not session_row:
            session_row = AnnotationSession(doctor_id=user.id)
            db.add(session_row)
            db.flush()
        annotation.session_id = session_row.id

    image.status = "in_progress"
    db.commit()
    return {"status": "draft_saved", "annotation_id": annotation.id}
