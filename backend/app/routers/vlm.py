"""
VLM integration — fundus description via Gemma4 (ENSIA HPC).

Primary:  Gemma4 multimodal VLM via vLLM + cloudflare tunnel.
No fallback — if VLM is not configured, returns a clear message.

Setup:
  1. On HPC, start the Gemma4 vLLM server (see pipeline-uwf notebook)
  2. Run: bash scripts/vlm-tunnel.sh lyna.ikhelef@hpc.ensia.edu.dz
  3. Copy the tunnel URL into .env:
       VLM_SERVICE_URL=https://xxxx.trycloudflare.com/v1
  4. docker compose up -d backend
"""
import base64
import os
import httpx
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import FundusImage, ClinicalNote, Annotation, AuditLog
from ..schemas import VLMDescribeOut
from ..config import settings
from .auth import get_current_user

router = APIRouter(prefix="/vlm", tags=["vlm"])


# ─── Gemma4 prompt ───────────────────────────────────────────────────────────

_VLM_PROMPT = """Tu es un ophtalmologue expert analysant une photographie du fond d'œil.
Décris ce que tu observes de manière clinique et précise en français.
Identifie les pathologies présentes, les signes visibles (hémorragies, exsudats, \
drusen, néovaisseaux, décollements, altérations maculaires, etc.),
la qualité de l'image, et l'état du disque optique, de la macula et des vaisseaux.
Réponds UNIQUEMENT avec ce JSON :
{
  "is_normal": true | false,
  "confidence": <float 0-1>,
  "image_quality": "good" | "fair" | "poor",
  "note": "<description clinique détaillée en français : pathologies détectées, signes observés, conclusion>"
}"""


def _parse_vlm_json(raw: str) -> str:
    try:
        parsed = json.loads(raw)
        return parsed.get("note", parsed.get("reason", raw))
    except (json.JSONDecodeError, AttributeError):
        return raw


# ─── Gemma4 call ─────────────────────────────────────────────────────────────

async def _describe_with_gemma4(file_path: str) -> str:
    vlm_url = settings.vlm_service_url
    cookie  = settings.vlm_kubeflow_cookie
    api_key = settings.vlm_api_key

    if not vlm_url:
        return "[Gemma4 non configuré — définir VLM_SERVICE_URL dans .env]"

    headers: dict = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if cookie:
        headers["Cookie"] = f"oauth2_proxy_kubeflow={cookie}"

    path = Path(file_path)
    if not path.exists():
        return f"[Image introuvable : {file_path}]"

    suffix     = path.suffix.lower()
    media_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(suffix, "image/jpeg")
    with open(file_path, "rb") as f:
        image_b64 = base64.standard_b64encode(f.read()).decode()

    # Check tunnel + get model
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as c:
        models_r = await c.get(f"{vlm_url}/models", headers=headers)
        if models_r.status_code in (302, 403, 401):
            return "[Session Kubeflow expirée — copiez un nouveau cookie oauth2_proxy_kubeflow]"
        if models_r.status_code == 404:
            return "[Tunnel VLM injoignable — relancez le notebook Gemma4 sur HPC]"
        models_r.raise_for_status()
        model_id = models_r.json()["data"][0]["id"]

    # Generate description
    async with httpx.AsyncClient(timeout=120.0) as c:
        resp = await c.post(
            f"{vlm_url}/chat/completions",
            headers=headers,
            json={
                "model": model_id,
                "max_tokens": 512,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}"}},
                        {"type": "text", "text": _VLM_PROMPT},
                    ],
                }],
            },
        )
        resp.raise_for_status()

    raw = resp.json()["choices"][0]["message"]["content"]
    return _parse_vlm_json(raw)


# ─── Describe endpoint ────────────────────────────────────────────────────────

@router.post("/describe/{image_id}", response_model=VLMDescribeOut)
async def describe_image(
    image_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    img = db.query(FundusImage).filter(FundusImage.id == image_id).first()
    if not img:
        raise HTTPException(404, "Image introuvable")

    try:
        description = await _describe_with_gemma4(img.file_path)
    except Exception as exc:
        description = f"[Erreur Gemma4 : {exc}]"

    # Persist only successful descriptions
    if not description.startswith("["):
        annotation = (
            db.query(Annotation)
            .filter(
                Annotation.image_id == image_id,
                Annotation.doctor_id == user.id,
                Annotation.status == "draft",
            )
            .order_by(Annotation.created_at.desc())
            .first()
        )
        if annotation and annotation.clinical_note:
            annotation.clinical_note.vlm_description = description
            db.commit()

    db.add(AuditLog(
        user_id=user.id,
        action="vlm.describe",
        entity_type="fundus_image",
        entity_id=image_id,
        detail=f"Description Gemma4 pour image {image_id}",
    ))
    db.commit()

    return VLMDescribeOut(description=description, image_id=image_id)


# ─── Batch import ─────────────────────────────────────────────────────────────

class VLMBatchItem(BaseModel):
    image_id:    Optional[str] = None
    clinical_id: Optional[str] = None
    eye:         Optional[str] = None
    file_path:   Optional[str] = None
    description: str


class VLMBatchResult(BaseModel):
    imported: int
    skipped:  int
    errors:   list[str] = []


@router.post("/batch-import", response_model=VLMBatchResult)
async def batch_import_vlm(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """Import pre-computed descriptions from a JSON array file."""
    import json as _json

    raw = await file.read()
    try:
        items = _json.loads(raw)
    except _json.JSONDecodeError as e:
        raise HTTPException(400, f"JSON invalide : {e}")

    if not isinstance(items, list):
        raise HTTPException(400, "Le fichier doit contenir un tableau JSON []")

    imported = skipped = 0
    errors = []

    for idx, raw_item in enumerate(items):
        try:
            item = VLMBatchItem(**raw_item)
        except Exception as e:
            errors.append(f"[{idx}] format invalide: {e}")
            skipped += 1
            continue

        img = None
        if item.image_id:
            img = db.query(FundusImage).filter(FundusImage.id == item.image_id).first()
        elif item.clinical_id and item.eye:
            from ..models import Patient
            patient = db.query(Patient).filter(Patient.clinical_id == item.clinical_id).first()
            if patient:
                img = (
                    db.query(FundusImage)
                    .filter(FundusImage.patient_id == patient.id, FundusImage.eye == item.eye)
                    .first()
                )

        if not img:
            errors.append(f"[{idx}] image introuvable: {raw_item}")
            skipped += 1
            continue

        annotation = (
            db.query(Annotation)
            .filter(Annotation.image_id == img.id, Annotation.status == "draft")
            .order_by(Annotation.created_at.desc())
            .first()
        )
        if annotation and annotation.clinical_note:
            annotation.clinical_note.vlm_description = item.description
        elif annotation and not annotation.clinical_note:
            db.add(ClinicalNote(annotation_id=annotation.id, vlm_description=item.description))
        else:
            annotation = Annotation(image_id=img.id, doctor_id=user.id, status="draft")
            db.add(annotation)
            db.flush()
            db.add(ClinicalNote(annotation_id=annotation.id, vlm_description=item.description))

        db.add(AuditLog(
            user_id=user.id,
            action="vlm.batch_import",
            entity_type="fundus_image",
            entity_id=str(img.id),
            detail=f"Description importée (batch) pour image {img.id}",
        ))
        imported += 1

    db.commit()
    return VLMBatchResult(imported=imported, skipped=skipped, errors=errors[:20])
