"""
Custom pathology proposals — doctors propose (with linked image), admin approves.
Admin can edit name/mechanism/grades/urgency directly on the approval screen.
All actions are logged to audit_logs.
"""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import CustomPathologyProposal, Disease, FundusImage, AuditLog, User
from ..schemas import ProposalIn, ProposalApproveIn, ProposalOut
from .auth import get_current_user, require_admin

router = APIRouter(prefix="/proposals", tags=["proposals"])


def _out(p: CustomPathologyProposal) -> ProposalOut:
    return ProposalOut(
        id=p.id,
        doctor_id=p.doctor_id,
        image_id=p.image_id,
        proposed_name=p.proposed_name,
        proposed_description=p.proposed_description,
        suspected_mechanism=p.suspected_mechanism or "",
        is_gradable=p.is_gradable,
        proposed_grades_json=p.proposed_grades_json,
        status=p.status,
        admin_notes=p.admin_notes,
        created_at=p.created_at,
    )


@router.get("", response_model=list[ProposalOut])
def list_proposals(
    status: str | None = None,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    q = db.query(CustomPathologyProposal)
    if status:
        q = q.filter(CustomPathologyProposal.status == status)
    return [_out(p) for p in q.order_by(CustomPathologyProposal.created_at.desc()).all()]


@router.get("/{proposal_id}", response_model=ProposalOut)
def get_proposal(proposal_id: str, db: Session = Depends(get_db), _user=Depends(get_current_user)):
    p = db.query(CustomPathologyProposal).filter(CustomPathologyProposal.id == proposal_id).first()
    if not p:
        raise HTTPException(404, "Proposition introuvable")
    return _out(p)


@router.get("/{proposal_id}/image-url")
def get_proposal_image(
    proposal_id: str,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """Returns the file URL of the image attached to this proposal."""
    p = db.query(CustomPathologyProposal).filter(CustomPathologyProposal.id == proposal_id).first()
    if not p:
        raise HTTPException(404, "Proposition introuvable")
    if not p.image_id:
        return {"image_url": None}
    img = db.query(FundusImage).filter(FundusImage.id == p.image_id).first()
    if not img:
        return {"image_url": None}
    return {"image_url": f"/api/images/{img.id}/file", "image_id": img.id}


@router.post("", response_model=ProposalOut)
def create_proposal(
    payload: ProposalIn,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    p = CustomPathologyProposal(
        doctor_id=user.id,
        image_id=payload.image_id,
        proposed_name=payload.proposed_name,
        proposed_description=payload.proposed_description,
        suspected_mechanism=payload.suspected_mechanism,
        is_gradable=payload.is_gradable,
        proposed_grades_json=payload.proposed_grades_json,
        proposed_grade_labels=payload.proposed_grade_labels,
        urgency_suggestion=payload.urgency_suggestion,
    )
    db.add(p)
    db.add(AuditLog(
        user_id=user.id,
        action="proposal.create",
        entity_type="proposal",
        detail=f"Proposition de nouvelle pathologie : {payload.proposed_name}",
    ))
    db.commit()
    db.refresh(p)
    return _out(p)


@router.post("/{proposal_id}/approve", dependencies=[Depends(require_admin)])
def approve_proposal(
    proposal_id: str,
    payload: ProposalApproveIn,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    p = db.query(CustomPathologyProposal).filter(CustomPathologyProposal.id == proposal_id).first()
    if not p:
        raise HTTPException(404, "Proposition introuvable")
    if p.status != "pending":
        raise HTTPException(400, f"Proposition déjà {p.status}")

    # Generate stable disease code
    base_code = "X_" + "".join(c for c in payload.final_name.upper() if c.isalnum())[:24]
    code = base_code
    suffix = 1
    while db.query(Disease).filter(Disease.code == code).first():
        suffix += 1
        code = f"{base_code}_{suffix}"

    disease = Disease(
        code=code,
        name_fr=payload.final_name,
        description=payload.final_description,
        mechanism_code=payload.mechanism_code,
        is_gradable=payload.is_gradable,
        grades_json=payload.grades_json,
        grade_labels_json=payload.grade_labels_json,
        urgency_override=payload.urgency_override,
        is_approved=True,
    )
    db.add(disease)

    p.status = "approved"
    p.reviewed_by = user.id
    p.reviewed_at = datetime.utcnow()
    p.resulting_disease_code = code
    p.admin_notes = payload.admin_notes

    db.add(AuditLog(
        user_id=user.id,
        action="proposal.approve",
        entity_type="proposal",
        entity_id=proposal_id,
        new_value={
            "disease_code": code,
            "final_name": payload.final_name,
            "mechanism": payload.mechanism_code,
            "grades": payload.grades_json,
            "urgency": payload.urgency_override,
        },
        detail=f"Approbation de la proposition '{p.proposed_name}' → pathologie {code}",
    ))
    db.commit()
    return {"status": "approved", "disease_code": code}


@router.post("/{proposal_id}/reject", dependencies=[Depends(require_admin)])
def reject_proposal(
    proposal_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    p = db.query(CustomPathologyProposal).filter(CustomPathologyProposal.id == proposal_id).first()
    if not p:
        raise HTTPException(404, "Proposition introuvable")
    p.status = "rejected"
    p.reviewed_by = user.id
    p.reviewed_at = datetime.utcnow()
    db.add(AuditLog(
        user_id=user.id,
        action="proposal.reject",
        entity_type="proposal",
        entity_id=proposal_id,
        detail=f"Rejet de la proposition '{p.proposed_name}'",
    ))
    db.commit()
    return {"status": "rejected"}
