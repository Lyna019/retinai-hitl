"""
RetinAI HITL — backend entry point.
All routers are mounted under /api.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .config import settings
from .database import Base, engine
from .routers import (
    auth, images, annotations, catalog, proposals,
    admin, transcription, model, patients, locking, vlm,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("retinai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting %s [%s]", settings.app_name, settings.environment)
    Base.metadata.create_all(bind=engine)
    for d in [settings.images_root, settings.audio_root, settings.dicom_root, settings.gradcam_root]:
        Path(d).mkdir(parents=True, exist_ok=True)
    # Schema migration (idempotent — safe on every restart)
    try:
        from .migrate import migrate
        migrate()
        log.info("Schema migration: OK")
    except Exception as e:
        log.warning("Schema migration failed (non-fatal): %s", e)
    # Auto-seed catalog data (idempotent — safe on every restart)
    try:
        from .seed import seed as seed_catalog
        seed_catalog()
        log.info("Catalog seed: OK")
    except Exception as e:
        log.warning("Catalog seed failed (non-fatal): %s", e)
    yield
    log.info("Shutting down")


app = FastAPI(
    title=settings.app_name,
    version="0.2.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "development" else [
        "http://localhost", "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_routers = [
    auth, images, annotations, catalog, proposals,
    admin, transcription, model, patients, locking, vlm,
]
for r in api_routers:
    app.include_router(r.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok", "app": settings.app_name, "version": "0.2.0"}


for root, name in [
    (settings.images_root, "images"),
    (settings.audio_root,  "audio"),
    (settings.dicom_root,  "dicom"),
    (settings.gradcam_root, "gradcam"),
]:
    Path(root).mkdir(parents=True, exist_ok=True)
    app.mount(f"/api/files/{name}", StaticFiles(directory=root), name=name)
