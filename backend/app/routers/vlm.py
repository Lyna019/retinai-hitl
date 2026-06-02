"""
VLM integration — zero-shot fundus description via RetiZero.

Primary:  RetiZero microservice (retizero-service:9001) — runs locally, no tunnel needed.
Fallback: Gemma4 vLLM on ENSIA HPC (requires VLM_SERVICE_URL + SSH tunnel).

Configure in .env:
  RETIZERO_SERVICE_URL=http://retizero-service:9001   (default, set automatically by docker-compose)

To use the vLLM fallback instead:
  VLM_SERVICE_URL=https://<cloudflared-tunnel>/v1
  VLM_KUBEFLOW_COOKIE=<cookie from browser>
"""
import os
import base64
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


# ─── RetiZero (primary) ───────────────────────────────────────────────────────

def _b64_image(file_path: str | None) -> str | None:
    if not file_path or not os.path.exists(file_path):
        return None
    import base64
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


async def _describe_with_retizero(image_id: str, file_path: str) -> str:
    """Call RetiZero microservice for zero-shot classification."""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{settings.retizero_service_url}/describe",
                json={"image_id": image_id, "file_path": file_path,
                      "image_data": _b64_image(file_path)},
            )
            r.raise_for_status()
            return r.json().get("description", "[Réponse vide de RetiZero]")
    except Exception as exc:
        return f"[Erreur RetiZero : {exc}]"


# ─── vLLM / Gemma fallback ────────────────────────────────────────────────────

_VLM_PROMPT = (
    """Tu es un ophtalmologue expert analysant une photographie du fond d'œil.
Décris ce que tu observes de manière clinique et précise en français.
Identifie les pathologies présentes, les signes visibles (hémorragies, exsudats, drusen, néovaisseaux, etc.),
la qualité de l'image, et l'état du disque optique, de la macula et des vaisseaux.
Réponds UNIQUEMENT avec ce JSON :
{
  "is_normal": true | false,
  "confidence": <float 0-1>,
  "image_quality": "good" | "fair" | "poor",
  "note": "<description clinique détaillée en français : pathologies détectées, signes observés, conclusion>"
}
"""
)


def _parse_vlm_json(raw: str) -> str:
    try:
        parsed = json.loads(raw)
        return parsed.get("note", parsed.get("reason", raw))
    except (json.JSONDecodeError, AttributeError):
        return raw


async def _describe_with_vllm(file_path: str) -> str:
    vlm_url = settings.vlm_service_url
    cookie  = settings.vlm_kubeflow_cookie
    api_key = settings.vlm_api_key

    headers: dict = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if cookie:
        headers["Cookie"] = f"oauth2_proxy_kubeflow={cookie}"

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {file_path}")

    suffix     = path.suffix.lower()
    media_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(suffix, "image/jpeg")
    with open(file_path, "rb") as f:
        image_b64 = base64.standard_b64encode(f.read()).decode()

    async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as c:
        models_r = await c.get(f"{vlm_url}/models", headers=headers)
        if models_r.status_code in (302, 403, 401):
            raise RuntimeError("Session Kubeflow expirée — copiez un nouveau cookie oauth2_proxy_kubeflow")
        if models_r.status_code == 404:
            raise RuntimeError("Proxy VLM injoignable (404) — relancez Cell 1 dans le notebook pipeline-uwf")
        models_r.raise_for_status()
        model_id = models_r.json()["data"][0]["id"]

    async with httpx.AsyncClient(timeout=120.0) as c:
        resp = await c.post(
            f"{vlm_url}/chat/completions",
            headers=headers,
            json={
                "model": model_id,
                "max_tokens": 768,
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

    # Try Gemma4 VLM first (multimodal, richer descriptions); fall back to RetiZero if not configured
    description = None
    if settings.vlm_service_url:
        try:
            description = await _describe_with_vllm(img.file_path)
        except Exception as exc:
            description = None  # fall through to RetiZero

    if not description:
        description = await _describe_with_retizero(image_id, img.file_path)

    # Persist only successful descriptions (never store error strings)
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
        detail=f"Description générée pour l'image {image_id}",
    ))
    db.commit()

    return VLMDescribeOut(description=description, image_id=image_id)


# ─── Batch import — pre-computed descriptions from HPC ───────────────────────

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
        elif item.file_path:
            img = db.query(FundusImage).filter(
                FundusImage.file_path.contains(item.file_path)
            ).first()

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
