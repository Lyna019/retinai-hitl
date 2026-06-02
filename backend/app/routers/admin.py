"""
Admin dashboard — stats, doctor management, audit log,
model version registry, active learning config, image locks.
"""
from collections import Counter
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import (
    Annotation, DiseaseLabel, FundusImage, UrgencyFlag,
    GradCAMValidation, User, CustomPathologyProposal,
    AuditLog, ModelVersion, ActiveLearningConfig, ImageLock,
)
from ..schemas import (
    AuditLogOut, ModelVersionOut, ModelVersionIn,
    ActiveLearningConfigOut, ActiveLearningConfigIn,
    DoctorCreateIn, DoctorOut,
)
from .auth import require_admin, hash_password

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


# ─── Stats ──────────────────────────────────────────────────────────────────

@router.get("/stats/progress")
def progress(db: Session = Depends(get_db)):
    total       = db.query(func.count(FundusImage.id)).scalar() or 0
    done        = db.query(func.count(FundusImage.id)).filter(FundusImage.status == "done").scalar() or 0
    in_progress = db.query(func.count(FundusImage.id)).filter(FundusImage.status == "in_progress").scalar() or 0
    pending     = db.query(func.count(FundusImage.id)).filter(FundusImage.status == "pending").scalar() or 0
    urgent      = db.query(func.count(UrgencyFlag.id)).filter(UrgencyFlag.level.in_(["P1","P2"])).scalar() or 0
    return {"total": total, "done": done, "in_progress": in_progress, "pending": pending, "urgent": urgent}


@router.get("/stats/disease-distribution")
def disease_distribution(db: Session = Depends(get_db)):
    rows = (
        db.query(DiseaseLabel.disease_code, func.count(DiseaseLabel.id))
        .group_by(DiseaseLabel.disease_code).all()
    )
    return [{"name": code, "value": n} for code, n in rows]


@router.get("/stats/urgency-distribution")
def urgency_distribution(db: Session = Depends(get_db)):
    rows = (
        db.query(UrgencyFlag.level, func.count(UrgencyFlag.id))
        .group_by(UrgencyFlag.level).all()
    )
    return [{"level": lv, "count": n} for lv, n in rows]


@router.get("/stats/gradcam-validation")
def gradcam_validation(db: Session = Depends(get_db)):
    rows = (
        db.query(GradCAMValidation.verdict, func.count(GradCAMValidation.id))
        .group_by(GradCAMValidation.verdict).all()
    )
    return [{"verdict": v, "count": n} for v, n in rows]


@router.get("/stats/doctors")
def doctors_stats(db: Session = Depends(get_db)):
    rows = (
        db.query(User.id, User.full_name, func.count(Annotation.id), func.avg(Annotation.time_spent_sec))
        .outerjoin(Annotation, Annotation.doctor_id == User.id)
        .filter(User.role == "doctor")
        .group_by(User.id, User.full_name).all()
    )
    return [
        {
            "id": uid, "name": name, "annotations": count,
            "avg_time": int(avg or 0),
            "initials": "".join(p[0] for p in (name or "U").split()[:2]).upper(),
        }
        for uid, name, count, avg in rows
    ]


@router.get("/stats/proposals-pending")
def proposals_pending(db: Session = Depends(get_db)):
    n = db.query(func.count(CustomPathologyProposal.id)).filter(
        CustomPathologyProposal.status == "pending"
    ).scalar() or 0
    return {"pending": n}


@router.get("/stats/inter-annotator-kappa")
def inter_annotator_kappa(db: Session = Depends(get_db)):
    return {"kappa": 0.78, "n_doubly_annotated": 0, "computed": False}


# ─── Doctor management ──────────────────────────────────────────────────────

@router.get("/doctors", response_model=list[DoctorOut])
def list_doctors(db: Session = Depends(get_db)):
    users = db.query(User).filter(User.role.in_(["doctor", "viewer"])).all()
    return [
        DoctorOut(
            id=u.id, username=u.username, email=u.email,
            full_name=u.full_name, role=u.role,
            is_active=u.is_active, created_at=u.created_at,
        )
        for u in users
    ]


@router.post("/doctors", response_model=DoctorOut)
def create_doctor(
    payload: DoctorCreateIn,
    db: Session = Depends(get_db),
    user=Depends(require_admin),
):
    if db.query(User).filter(User.username == payload.username).first():
        raise HTTPException(400, "Nom d'utilisateur déjà pris")
    u = User(
        username=payload.username,
        email=payload.email,
        full_name=payload.full_name,
        role=payload.role,
        password_hash=hash_password(payload.password),
        is_active=True,
    )
    db.add(u)
    db.add(AuditLog(
        user_id=None,
        action="user.create",
        entity_type="user",
        entity_id=payload.username,
        detail=f"Création du compte {payload.username} (rôle: {payload.role})",
    ))
    db.commit()
    db.refresh(u)
    return DoctorOut(
        id=u.id, username=u.username, email=u.email,
        full_name=u.full_name, role=u.role,
        is_active=u.is_active, created_at=u.created_at,
    )


@router.put("/doctors/{user_id}/toggle-active")
def toggle_doctor_active(user_id: str, db: Session = Depends(get_db)):
    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        raise HTTPException(404)
    u.is_active = not u.is_active
    db.commit()
    return {"id": u.id, "is_active": u.is_active}


# ─── Image locks ────────────────────────────────────────────────────────────

@router.get("/locks")
def list_locks(db: Session = Depends(get_db)):
    locks = db.query(ImageLock).all()
    return [
        {"image_id": l.image_id, "patient_id": l.patient_id,
         "locked_by": l.locked_by, "locked_at": l.locked_at}
        for l in locks
    ]


@router.delete("/locks/{image_id}")
def release_lock(image_id: str, db: Session = Depends(get_db)):
    lock = db.query(ImageLock).filter(ImageLock.image_id == image_id).first()
    if lock:
        db.delete(lock)
        db.commit()
    return {"status": "released"}


# ─── Audit log ──────────────────────────────────────────────────────────────

@router.get("/audit-log", response_model=list[AuditLogOut])
def get_audit_log(
    entity_type: str | None = None,
    action: str | None = None,
    limit: int = 200,
    db: Session = Depends(get_db),
):
    q = db.query(AuditLog)
    if entity_type:
        q = q.filter(AuditLog.entity_type == entity_type)
    if action:
        q = q.filter(AuditLog.action.ilike(f"%{action}%"))
    rows = q.order_by(AuditLog.created_at.desc()).limit(limit).all()
    return [
        AuditLogOut(
            id=r.id, user_id=r.user_id, action=r.action,
            entity_type=r.entity_type, entity_id=r.entity_id,
            old_value=r.old_value, new_value=r.new_value,
            detail=r.detail, created_at=r.created_at,
        )
        for r in rows
    ]


# ─── Model versions ─────────────────────────────────────────────────────────

@router.get("/model-versions", response_model=list[ModelVersionOut])
def list_model_versions(db: Session = Depends(get_db)):
    rows = db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()
    return [
        ModelVersionOut(
            id=r.id, version_tag=r.version_tag, description=r.description,
            metrics_json=r.metrics_json, is_active=r.is_active,
            trained_at=r.trained_at, created_at=r.created_at,
        )
        for r in rows
    ]


@router.post("/model-versions", response_model=ModelVersionOut)
def register_model_version(
    payload: ModelVersionIn,
    db: Session = Depends(get_db),
    user=Depends(require_admin),
):
    if db.query(ModelVersion).filter(ModelVersion.version_tag == payload.version_tag).first():
        raise HTTPException(400, "Tag de version déjà enregistré")
    v = ModelVersion(
        version_tag=payload.version_tag,
        checkpoint_path=payload.checkpoint_path,
        description=payload.description,
        metrics_json=payload.metrics_json,
        trained_at=payload.trained_at,
        is_active=False,
    )
    db.add(v)
    db.add(AuditLog(
        user_id=user.id,
        action="model.register",
        entity_type="model_version",
        entity_id=payload.version_tag,
        detail=f"Enregistrement du modèle {payload.version_tag}",
    ))
    db.commit()
    db.refresh(v)
    return ModelVersionOut(
        id=v.id, version_tag=v.version_tag, description=v.description,
        metrics_json=v.metrics_json, is_active=v.is_active,
        trained_at=v.trained_at, created_at=v.created_at,
    )


@router.post("/model-versions/{version_id}/activate")
def activate_model_version(
    version_id: str,
    db: Session = Depends(get_db),
    user=Depends(require_admin),
):
    v = db.query(ModelVersion).filter(ModelVersion.id == version_id).first()
    if not v:
        raise HTTPException(404, "Version introuvable")

    # Deactivate all others
    db.query(ModelVersion).update({"is_active": False})
    v.is_active = True

    # Update AL config
    cfg = db.query(ActiveLearningConfig).filter(ActiveLearningConfig.id == "singleton").first()
    if cfg:
        cfg.current_version_tag = v.version_tag

    db.add(AuditLog(
        user_id=user.id,
        action="model.activate",
        entity_type="model_version",
        entity_id=version_id,
        detail=f"Activation du modèle {v.version_tag}",
    ))
    db.commit()
    return {"status": "activated", "version_tag": v.version_tag}


# ─── Active learning config ─────────────────────────────────────────────────

@router.get("/active-learning", response_model=ActiveLearningConfigOut)
def get_al_config(db: Session = Depends(get_db)):
    cfg = db.query(ActiveLearningConfig).filter(ActiveLearningConfig.id == "singleton").first()
    if not cfg:
        raise HTTPException(404, "Config non initialisée — relancez le seed")
    return ActiveLearningConfigOut(
        uncertainty_threshold=cfg.uncertainty_threshold,
        n_samples_per_cycle=cfg.n_samples_per_cycle,
        auto_retrain=cfg.auto_retrain,
        current_version_tag=cfg.current_version_tag,
        last_retrain_at=cfg.last_retrain_at,
    )


@router.put("/active-learning", response_model=ActiveLearningConfigOut)
def update_al_config(
    payload: ActiveLearningConfigIn,
    db: Session = Depends(get_db),
    user=Depends(require_admin),
):
    cfg = db.query(ActiveLearningConfig).filter(ActiveLearningConfig.id == "singleton").first()
    if not cfg:
        raise HTTPException(404)

    old = {
        "uncertainty_threshold": cfg.uncertainty_threshold,
        "n_samples_per_cycle": cfg.n_samples_per_cycle,
        "auto_retrain": cfg.auto_retrain,
    }

    if payload.uncertainty_threshold is not None:
        cfg.uncertainty_threshold = payload.uncertainty_threshold
    if payload.n_samples_per_cycle is not None:
        cfg.n_samples_per_cycle = payload.n_samples_per_cycle
    if payload.auto_retrain is not None:
        cfg.auto_retrain = payload.auto_retrain

    db.add(AuditLog(
        user_id=user.id,
        action="al_config.update",
        entity_type="active_learning_config",
        entity_id="singleton",
        old_value=old,
        new_value=payload.model_dump(exclude_none=True),
        detail="Mise à jour de la configuration d'apprentissage actif",
    ))
    db.commit()
    db.refresh(cfg)
    return ActiveLearningConfigOut(
        uncertainty_threshold=cfg.uncertainty_threshold,
        n_samples_per_cycle=cfg.n_samples_per_cycle,
        auto_retrain=cfg.auto_retrain,
        current_version_tag=cfg.current_version_tag,
        last_retrain_at=cfg.last_retrain_at,
    )
