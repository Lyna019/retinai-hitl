"""
Bridge between the platform and the model microservice.

The model service is treated as a black box behind 3 endpoints:
  POST /predict     -> top-k disease predictions
  POST /gradcam     -> heatmap PNG
  POST /uncertainty -> active learning score

file_path is resolved here from the FundusImage record and forwarded to
the model service so the preprocessing pipeline can load the image from disk.

Swap the underlying PyTorch checkpoint inside the model service without
touching this router.
"""
import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models import FundusImage, ModelPrediction
from .auth import get_current_user

router = APIRouter(prefix="/model", tags=["model"])


def _file_path_for(image_id: str, db: Session) -> str | None:
    """Look up the absolute file path for an image record."""
    img = db.query(FundusImage).filter(FundusImage.id == image_id).first()
    return img.file_path if img else None


@router.post("/predict")
async def predict(
    image_id: str,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    file_path = _file_path_for(image_id, db)
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            r = await client.post(
                f"{settings.model_service_url}/predict",
                json={"image_id": image_id, "file_path": file_path},
            )
        except httpx.RequestError as e:
            raise HTTPException(503, f"Service modèle indisponible: {e}")
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)

    data = r.json()

    model_version = data.get("model_version", "unknown")
    is_mock = "-mock" in model_version

    if not is_mock:
        top_k      = data.get("top_k", [])
        uncertainty = data.get("uncertainty")
        gradcam_url = data.get("gradcam_url")

        # Remove any stale mock predictions before storing the real one
        (
            db.query(ModelPrediction)
            .filter(
                ModelPrediction.image_id == image_id,
                ModelPrediction.model_version.like("%-mock%"),
            )
            .delete(synchronize_session=False)
        )

        pred = ModelPrediction(
            image_id=image_id,
            model_version=model_version,
            top_k_json=top_k,
            confidence=top_k[0]["confidence"] if top_k else None,
            gradcam_path=f"{settings.gradcam_root}/{image_id}.png" if gradcam_url else None,
        )
        db.add(pred)

        if uncertainty is not None:
            img = db.query(FundusImage).filter(FundusImage.id == image_id).first()
            if img:
                img.uncertainty_score = uncertainty

        db.commit()

    return data


@router.post("/uncertainty")
async def uncertainty(
    image_id: str,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    file_path = _file_path_for(image_id, db)
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{settings.model_service_url}/uncertainty",
            json={"image_id": image_id, "file_path": file_path},
        )
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    return r.json()


@router.post("/gradcam")
async def gradcam(image_id: str, _user=Depends(get_current_user)):
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            r = await client.post(
                f"{settings.model_service_url}/gradcam",
                json={"image_id": image_id},
            )
        except httpx.RequestError as e:
            raise HTTPException(503, f"Service modèle indisponible: {e}")
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    return r.json()


@router.get("/health")
async def health():
    """Quick liveness check that the model service responds."""
    async with httpx.AsyncClient(timeout=3.0) as client:
        try:
            r = await client.get(f"{settings.model_service_url}/health")
            return {"model_service": r.json()}
        except httpx.RequestError:
            return {"model_service": "unreachable"}
