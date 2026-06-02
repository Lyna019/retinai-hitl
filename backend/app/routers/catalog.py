"""
Disease catalog — read + admin management (create / update / delete).
Grade label renames retroactively update all DiseaseLabel rows.
"""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import (
    Disease, Mechanism, LesionVocabulary, DiseaseLabel,
    AnatomicalRegion, SystemicDisease, AuditLog,
)
from ..schemas import (
    DiseaseOut, MechanismOut, DiseaseCreateIn, DiseaseUpdateIn,
    AnatomicalRegionOut, AnatomicalRegionIn,
    SystemicDiseaseOut, SystemicDiseaseIn,
)
from .auth import get_current_user, require_admin

router = APIRouter(prefix="/catalog", tags=["catalog"])

# Disease codes shown as quick-select chips in the annotation panel
_COMMON_CODES = frozenset({
    'DR', 'GLAUC', 'DMLA', 'HTN_DR',            # chronic
    'OACR', 'ABACR', 'NOIAA', 'DR_MAC_OFF', 'GLAUC_AIGU',  # emergency
})


def _disease_out(d: Disease) -> DiseaseOut:
    return DiseaseOut(
        code=d.code,
        name_fr=d.name_fr,
        description=d.description,
        mechanism=d.mechanism_code,
        gradable=d.is_gradable,
        grades=d.grades_json,
        grade_labels_json=d.grade_labels_json,
        urgency=d.urgency_override,
        is_approved=d.is_approved,
        common=d.code in _COMMON_CODES,
    )


# ─── Diseases ───────────────────────────────────────────────────────────────

@router.get("/diseases", response_model=list[DiseaseOut])
def list_diseases(
    approved_only: bool = True,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    q = db.query(Disease)
    if approved_only:
        q = q.filter(Disease.is_approved == True)
    return [_disease_out(d) for d in q.all()]


@router.post("/diseases", response_model=DiseaseOut, dependencies=[Depends(require_admin)])
def create_disease(payload: DiseaseCreateIn, db: Session = Depends(get_db), user=Depends(get_current_user)):
    if db.query(Disease).filter(Disease.code == payload.code).first():
        raise HTTPException(400, f"Code '{payload.code}' déjà existant")
    d = Disease(
        code=payload.code,
        name_fr=payload.name_fr,
        description=payload.description,
        mechanism_code=payload.mechanism_code,
        is_gradable=payload.is_gradable,
        grades_json=payload.grades_json,
        grade_labels_json=payload.grade_labels_json,
        urgency_override=payload.urgency_override,
        is_approved=True,
    )
    db.add(d)
    db.add(AuditLog(
        user_id=user.id,
        action="disease.create",
        entity_type="disease",
        entity_id=payload.code,
        new_value=payload.model_dump(),
        detail=f"Création de la pathologie {payload.name_fr}",
    ))
    db.commit()
    db.refresh(d)
    return _disease_out(d)


@router.put("/diseases/{code}", response_model=DiseaseOut, dependencies=[Depends(require_admin)])
def update_disease(
    code: str,
    payload: DiseaseUpdateIn,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    d = db.query(Disease).filter(Disease.code == code).first()
    if not d:
        raise HTTPException(404, "Pathologie introuvable")

    old_snap = {
        "name_fr": d.name_fr,
        "description": d.description,
        "mechanism_code": d.mechanism_code,
        "is_gradable": d.is_gradable,
        "grades_json": d.grades_json,
        "grade_labels_json": d.grade_labels_json,
        "urgency_override": d.urgency_override,
    }

    # Detect grade value renames → retroactively update DiseaseLabel.grade
    old_labels = d.grade_labels_json or {}
    new_labels = payload.grade_labels_json or old_labels

    if payload.name_fr is not None:            d.name_fr = payload.name_fr
    if payload.description is not None:        d.description = payload.description
    if payload.mechanism_code is not None:     d.mechanism_code = payload.mechanism_code
    if payload.is_gradable is not None:        d.is_gradable = payload.is_gradable
    if payload.grades_json is not None:        d.grades_json = payload.grades_json
    if payload.grade_labels_json is not None:  d.grade_labels_json = payload.grade_labels_json
    if payload.urgency_override is not None:   d.urgency_override = payload.urgency_override
    d.updated_at = datetime.utcnow()

    # Retroactive rename: stored grade text → new label text
    if payload.grade_labels_json and old_labels != new_labels:
        label_remap = {}
        for key, new_label in new_labels.items():
            old_label = old_labels.get(key)
            if old_label and old_label != new_label:
                label_remap[old_label] = new_label
        if label_remap:
            for lbl in db.query(DiseaseLabel).filter(DiseaseLabel.disease_code == code).all():
                if lbl.grade in label_remap:
                    lbl.grade = label_remap[lbl.grade]

    db.add(AuditLog(
        user_id=user.id,
        action="disease.update",
        entity_type="disease",
        entity_id=code,
        old_value=old_snap,
        new_value=payload.model_dump(exclude_none=True),
        detail=f"Mise à jour de la pathologie {d.name_fr}",
    ))
    db.commit()
    db.refresh(d)
    return _disease_out(d)


@router.delete("/diseases/{code}", dependencies=[Depends(require_admin)])
def delete_disease(code: str, db: Session = Depends(get_db), user=Depends(get_current_user)):
    d = db.query(Disease).filter(Disease.code == code).first()
    if not d:
        raise HTTPException(404, "Pathologie introuvable")
    db.add(AuditLog(
        user_id=user.id,
        action="disease.delete",
        entity_type="disease",
        entity_id=code,
        old_value={"name_fr": d.name_fr},
        detail=f"Suppression de la pathologie {d.name_fr}",
    ))
    db.delete(d)
    db.commit()
    return {"status": "deleted"}


# ─── Mechanisms ─────────────────────────────────────────────────────────────

@router.get("/mechanisms", response_model=list[MechanismOut])
def list_mechanisms(db: Session = Depends(get_db), _user=Depends(get_current_user)):
    mechs = db.query(Mechanism).all()
    result = []
    for m in mechs:
        diseases = [
            _disease_out(d)
            for d in db.query(Disease)
            .filter(Disease.mechanism_code == m.code, Disease.is_approved == True).all()
        ]
        result.append(MechanismOut(
            code=m.code, name_fr=m.name_fr,
            description=m.description, diseases=diseases,
        ))
    return result


# ─── Lesions ────────────────────────────────────────────────────────────────

@router.get("/lesions")
def list_lesions(db: Session = Depends(get_db), _user=Depends(get_current_user)):
    return [
        {"code": l.code, "name_fr": l.name_fr, "color": l.color_hex}
        for l in db.query(LesionVocabulary).all()
    ]


# ─── Anatomical regions ─────────────────────────────────────────────────────

@router.get("/regions", response_model=list[AnatomicalRegionOut])
def list_regions(db: Session = Depends(get_db), _user=Depends(get_current_user)):
    return [
        AnatomicalRegionOut(id=r.id, name_fr=r.name_fr, is_custom=r.is_custom)
        for r in db.query(AnatomicalRegion).filter(AnatomicalRegion.is_active == True).all()
    ]


@router.post("/regions", response_model=AnatomicalRegionOut)
def create_region(
    payload: AnatomicalRegionIn,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    existing = db.query(AnatomicalRegion).filter(AnatomicalRegion.name_fr == payload.name_fr).first()
    if existing:
        return AnatomicalRegionOut(id=existing.id, name_fr=existing.name_fr, is_custom=existing.is_custom)
    r = AnatomicalRegion(name_fr=payload.name_fr, is_custom=True, created_by=user.id)
    db.add(r)
    db.commit()
    db.refresh(r)
    return AnatomicalRegionOut(id=r.id, name_fr=r.name_fr, is_custom=r.is_custom)


# ─── Systemic diseases (admin-managed) ──────────────────────────────────────

@router.get("/systemic-diseases", response_model=list[SystemicDiseaseOut])
def list_systemic_diseases(db: Session = Depends(get_db), _user=Depends(get_current_user)):
    return [
        SystemicDiseaseOut(id=s.id, name_fr=s.name_fr, category=s.category, is_active=s.is_active)
        for s in db.query(SystemicDisease).filter(SystemicDisease.is_active == True).all()
    ]


@router.post("/systemic-diseases", response_model=SystemicDiseaseOut, dependencies=[Depends(require_admin)])
def create_systemic_disease(payload: SystemicDiseaseIn, db: Session = Depends(get_db)):
    s = SystemicDisease(name_fr=payload.name_fr, category=payload.category)
    db.add(s)
    db.commit()
    db.refresh(s)
    return SystemicDiseaseOut(id=s.id, name_fr=s.name_fr, category=s.category)


@router.put("/systemic-diseases/{sd_id}", response_model=SystemicDiseaseOut, dependencies=[Depends(require_admin)])
def update_systemic_disease(sd_id: str, payload: SystemicDiseaseIn, db: Session = Depends(get_db)):
    s = db.query(SystemicDisease).filter(SystemicDisease.id == sd_id).first()
    if not s:
        raise HTTPException(404)
    s.name_fr = payload.name_fr
    if payload.category is not None:
        s.category = payload.category
    db.commit()
    db.refresh(s)
    return SystemicDiseaseOut(id=s.id, name_fr=s.name_fr, category=s.category)


@router.delete("/systemic-diseases/{sd_id}", dependencies=[Depends(require_admin)])
def delete_systemic_disease(sd_id: str, db: Session = Depends(get_db)):
    s = db.query(SystemicDisease).filter(SystemicDisease.id == sd_id).first()
    if not s:
        raise HTTPException(404)
    s.is_active = False   # soft delete
    db.commit()
    return {"status": "deactivated"}
