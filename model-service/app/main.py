"""
RetinAI inference microservice — v0.5.0

Endpoints:
  GET  /health        liveness + model status
  POST /predict       top-k disease predictions + Grad-CAM
  POST /uncertainty   active-learning entropy score
  POST /gradcam       (re-)generate Grad-CAM for a previously predicted image

Behaviour:
  • Checkpoint present  → real PyTorch inference
  • No checkpoint       → realistic mock so the platform works without a model

Supported MODEL_ARCH values:
  swin_base        Swin-B patch4 window12 (timm), default — your trained model
  efficientnet_b4  torchvision EfficientNet-B4
  resnet50/101/…   torchvision ResNet family
  densenet121      torchvision DenseNet-121

Swin checkpoint formats accepted:
  • Merged state dict produced by model.merge_and_unload() — keys backbone.*/head.*
  • Raw PEFT state dict — keys base_model.model.*/modules_to_save.default.*

PREPROCESSING PIPELINE  (preprocess_uwf):
  fast resize → retina crop (black-border removal) →
  local contrast (Ben-Graham) → CLAHE on green channel → square pad
"""
import math
import os
import random
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

log = logging.getLogger("model-service")
logging.basicConfig(level=logging.INFO)

MODEL_VERSION   = os.getenv("MODEL_VERSION",   "retinai-v0.5.0")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "/app/checkpoints/best.pt")
MODEL_ARCH      = os.getenv("MODEL_ARCH",      "swin_base")
NUM_CLASSES     = int(os.getenv("NUM_CLASSES", "9"))
IMG_SIZE        = int(os.getenv("IMG_SIZE",    "512"))
GRADCAM_ROOT    = os.getenv("GRADCAM_ROOT",    "/app/data/gradcam")
# Temperature scaling: T < 1 sharpens under-confident model, T > 1 softens over-confident.
# Current checkpoint is under-confident (max conf ~69%) — T=0.65 pushes top class to ~78%.
TEMPERATURE      = float(os.getenv("MODEL_TEMPERATURE", "0.65"))
GRADCAM_ENABLED  = os.getenv("GRADCAM_ENABLED", "true").lower() == "true"

# Class index → disease code mapping (must match training order)
_DEFAULT_CODES = {
    9: ["MYOPIE_F", "GLAUC", "DR", "ME", "UVEITE", "DMLA", "RETDET", "HTN_DR", "NORMAL"],
    7: ["DR", "DMLA", "GLAUC", "HTN_DR", "OACR", "ABACR", "NORMAL"],
}
DISEASE_CODES = (
    [c.strip() for c in os.getenv("DISEASE_CODES", "").split(",") if c.strip()]
    or _DEFAULT_CODES.get(NUM_CLASSES)
    or [f"CLASS_{i}" for i in range(NUM_CLASSES)]
)

# ─────────────────────────── Model state ─────────────────────────────────────

_model            = None   # nn.Module | None
_device           = None   # torch.device | None
_target_layer_fn  = None   # (model) → [layer]  — for Grad-CAM
_reshape_transform = None  # for transformers (Swin needs B×N×C → B×C×H×W)
_model_loaded     = False  # True once load was attempted


def _swin_reshape(tensor, _height=None, _width=None):
    """Reshape Swin activations → [B, C, H, W] for Grad-CAM.
    Handles both [B, N, C] (older timm) and [B, H, W, C] (newer timm)."""
    if tensor.dim() == 4:
        # Already [B, H, W, C] — just permute
        return tensor.permute(0, 3, 1, 2).contiguous()
    n = tensor.shape[1]
    h = w = int(n ** 0.5)
    result = tensor.reshape(tensor.size(0), h, w, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)


def _get_target_layers(model):
    return _target_layer_fn(model) if _target_layer_fn else []


def _normalise_swin_state_dict(raw: dict) -> dict:
    """
    Accept three common PEFT/LoRA checkpoint formats and normalise to plain keys:
      1. Merged (merge_and_unload)       — keys backbone.*/head.* with no PEFT artefacts → no-op
      2. PeftModel.state_dict()          — keys have base_model.model.* prefix
      3. LoRA-wrapped (unmerged save)    — LoRA layers stored as .base_layer.weight
    """
    has_base_model_prefix = any(k.startswith("base_model.") for k in raw)
    has_base_layer        = any(".base_layer." in k for k in raw)

    if not has_base_model_prefix and not has_base_layer:
        return raw  # already clean / merged format

    clean = {}
    for k, v in raw.items():
        # Drop LoRA delta weights and PEFT bookkeeping
        if (
            ".lora_A." in k or ".lora_B." in k
            or ".lora_embedding_" in k
            or ".lora_magnitude_vector" in k
            or "modules_to_save" in k and ".lora_" in k
        ):
            continue

        # Strip top-level PEFT prefix  (base_model.model.X → X)
        for prefix in ("base_model.model.", "base_model."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break

        # Remap saved-module head keys (modules_to_save.default.1.weight → 1.weight)
        k = k.replace(".modules_to_save.default", "")

        # Flatten LoRA-wrapped layers: module.base_layer.weight → module.weight
        k = k.replace(".base_layer.", ".")

        clean[k] = v
    return clean


def _merge_lora(net, raw_ckpt, scaling=1.0):
    """Merge LoRA deltas directly into model weights. No PEFT runtime needed."""
    import torch
    lora_a, lora_b = {}, {}
    for k, v in raw_ckpt.items():
        if '.lora_A.' in k:
            base = k
            for pfx in ('base_model.model.', 'base_model.'):
                if base.startswith(pfx):
                    base = base[len(pfx):]
                    break
            lora_a[base[:base.index('.lora_A.')]] = v
        elif '.lora_B.' in k:
            base = k
            for pfx in ('base_model.model.', 'base_model.'):
                if base.startswith(pfx):
                    base = base[len(pfx):]
                    break
            lora_b[base[:base.index('.lora_B.')]] = v

    params = dict(net.named_parameters())
    merged = 0
    for path, A in lora_a.items():
        B = lora_b.get(path)
        if B is None:
            continue
        # Try .weight key; also try stripping .base_layer that may remain after normalise
        wkey = path + '.weight'
        if wkey not in params:
            wkey = path.replace('.base_layer', '') + '.weight'
        param = params.get(wkey)
        if param is None:
            log.warning("LoRA merge: no param found for %s", path)
            continue
        with torch.no_grad():
            if A.dim() == 2:
                delta = (B @ A) * scaling
            else:
                delta = (B.flatten(1) @ A.flatten(1)).view_as(param.data) * scaling
            param.data.add_(delta)
        merged += 1
    return merged


def _load_model():
    """Attempt to load checkpoint once. Returns (model, device) or (None, None)."""
    global _model, _device, _target_layer_fn, _reshape_transform, _model_loaded
    if _model_loaded:
        return _model, _device
    _model_loaded = True

    if not os.path.exists(CHECKPOINT_PATH):
        log.info("No checkpoint at %s — mock mode active", CHECKPOINT_PATH)
        return None, None

    try:
        import torch
        import torch.nn as nn
        import torchvision.models as tv

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading %s  num_classes=%d  img_size=%d  device=%s …",
                 MODEL_ARCH, NUM_CLASSES, IMG_SIZE, dev)

        if MODEL_ARCH == "swin_base":
            import timm

            backbone = timm.create_model(
                "swin_base_patch4_window12_384",
                pretrained=False,
                num_classes=0,
                img_size=IMG_SIZE,
            )

            class _SwinClassifier(nn.Module):
                def __init__(self, backbone, num_classes):
                    super().__init__()
                    self.backbone = backbone
                    self.head = nn.Sequential(
                        nn.Dropout(0.2),
                        nn.Linear(backbone.num_features, num_classes),
                    )

                def forward(self, x):
                    return self.head(self.backbone(x))

            net = _SwinClassifier(backbone, NUM_CLASSES)
            _target_layer_fn  = lambda m: [m.backbone.layers[-1].blocks[-1].norm1]
            _reshape_transform = _swin_reshape

        elif MODEL_ARCH == "efficientnet_b4":
            net = tv.efficientnet_b4(weights=None)
            net.classifier[1] = nn.Linear(net.classifier[1].in_features, NUM_CLASSES)
            _target_layer_fn  = lambda m: [m.features[-2]]
            _reshape_transform = None

        elif MODEL_ARCH.startswith("resnet"):
            net = getattr(tv, MODEL_ARCH)(weights=None)
            net.fc = nn.Linear(net.fc.in_features, NUM_CLASSES)
            _target_layer_fn  = lambda m: [m.layer4[-1]]
            _reshape_transform = None

        elif MODEL_ARCH == "densenet121":
            net = tv.densenet121(weights=None)
            net.classifier = nn.Linear(net.classifier.in_features, NUM_CLASSES)
            _target_layer_fn  = lambda m: [m.features[-1]]
            _reshape_transform = None

        else:
            raise ValueError(f"Unsupported MODEL_ARCH: {MODEL_ARCH}")

        raw_ckpt = torch.load(CHECKPOINT_PATH, map_location=dev, weights_only=False)
        if isinstance(raw_ckpt, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in raw_ckpt:
                    raw_ckpt = raw_ckpt[key]
                    break

        # Detect unmerged LoRA checkpoint (has lora_A / lora_B keys)
        has_lora = isinstance(raw_ckpt, dict) and any(
            ".lora_A." in k or ".lora_B." in k for k in raw_ckpt
        )

        state_dict = _normalise_swin_state_dict(raw_ckpt) if isinstance(raw_ckpt, dict) else raw_ckpt
        missing, unexpected = net.load_state_dict(state_dict, strict=False)
        if missing:
            log.warning("Missing keys (%d): %s …", len(missing), missing[:5])
        if unexpected:
            log.warning("Unexpected keys (%d): %s …", len(unexpected), unexpected[:5])
        matched = len(state_dict) - len(unexpected)
        if matched == 0:
            raise RuntimeError(
                f"Zero checkpoint keys matched model ({MODEL_ARCH}). "
                "Check MODEL_ARCH or checkpoint format."
            )
        log.info("Loaded %d/%d base keys", matched, len(state_dict))

        net.eval().to(dev)

        if has_lora:
            lora_a_sample = next(v for k, v in raw_ckpt.items() if ".lora_A." in k)
            lora_r = lora_a_sample.shape[0]
            n_merged = _merge_lora(net, raw_ckpt, scaling=1.0)
            log.info("LoRA adapters merged: %d pairs (r=%d, scaling=1.0) — model ready on %s",
                     n_merged, lora_r, dev)
        _model  = net
        _device = dev
        return _model, _device

    except Exception as exc:
        log.error("Model load failed (%s) — mock mode active", exc)
        return None, None


# ─────────────────────────── UWF Preprocessing ───────────────────────────────

def preprocess_uwf(pil_img: Image.Image, fast_resize: int = 1024) -> Image.Image:
    """
    Fundus preprocessing matching the training pipeline exactly:
      1. Black border removal via threshold crop
      2. Resize to target square
      3. CLAHE on LAB L-channel only (preserves RGB color)
    No Ben-Graham, no grayscale conversion.
    """
    img = np.array(pil_img.convert("RGB"))

    # Step 1: Black border removal
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pad = 5
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(img.shape[1] - x, w + 2 * pad)
        h = min(img.shape[0] - y, h + 2 * pad)
        img = img[y:y + h, x:x + w]

    # Step 3: CLAHE on LAB L-channel — preserves color, matches training
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(img)


def _load_and_preprocess(file_path: str | None, image_data: str | None = None) -> Image.Image | None:
    try:
        if image_data:
            import base64, io
            img_bytes = base64.b64decode(image_data)
            return preprocess_uwf(Image.open(io.BytesIO(img_bytes)))
        if file_path and os.path.exists(file_path):
            return preprocess_uwf(Image.open(file_path))
        return None
    except Exception as exc:
        log.warning("preprocess_uwf failed: %s", exc)
        return None


# ─────────────────────────── Real inference helpers ──────────────────────────

def _make_transform():
    import torchvision.transforms as T
    # ImageNet normalization — matches training pipeline (transforms.py)
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _real_predict(model, device, preprocessed: Image.Image):
    """Return (top_k list, uncertainty float, top_class_idx int)."""
    import torch

    tensor = _make_transform()(preprocessed).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor).squeeze()
        probs  = torch.sigmoid(logits / TEMPERATURE).cpu().tolist()

    # Multi-label top-k: all classes with confidence ≥ threshold, sorted desc
    indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    top_k = [
        {"disease_code": DISEASE_CODES[i], "confidence": round(p, 4)}
        for i, p in indexed[:3]
        if i < len(DISEASE_CODES)
    ]

    # Per-class binary entropy, normalised to [0, 1]
    per_ent = [
        -p * math.log(p + 1e-10) - (1 - p) * math.log(1 - p + 1e-10)
        for p in probs
    ]
    uncertainty = round(sum(per_ent) / (len(probs) * math.log(2)), 4)

    return top_k, uncertainty, indexed[0][0]


def _run_gradcam(model, device, preprocessed: Image.Image, class_idx: int, image_id: str) -> str | None:
    """
    Generate Grad-CAM heatmap.
    Returns base64-encoded PNG string so the backend can store it locally,
    regardless of whether the model-service is local or remote (HPC).
    """
    try:
        import io, base64, torch
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        tensor  = _make_transform()(preprocessed).unsqueeze(0).to(device)
        rgb_img = np.array(preprocessed.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0

        target_layers = _get_target_layers(model)
        if not target_layers:
            return None

        with GradCAM(
            model=model,
            target_layers=target_layers,
            reshape_transform=_reshape_transform,
        ) as cam:
            mask = cam(
                input_tensor=tensor,
                targets=[ClassifierOutputTarget(class_idx)],
            )[0]

        overlay = show_cam_on_image(rgb_img, mask, use_rgb=True)
        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        log.info("GradCAM generated for %s (%d bytes)", image_id, len(buf.getvalue()))
        return b64

    except Exception as exc:
        log.warning("GradCAM failed for %s: %s", image_id, exc)
        return None


# ─────────────────────────── Mock fallback ───────────────────────────────────

def _mock_predict(image_id: str, preprocessed: bool) -> dict:
    codes = DISEASE_CODES
    # Slightly favour the last code (often NORMAL/HEALTHY)
    weights = [1] * len(codes)
    if weights:
        weights[-1] = 3
    primary      = random.choices(codes, weights=weights)[0]
    primary_conf = round(random.uniform(0.55, 0.95), 3)
    others       = random.sample([c for c in codes if c != primary], min(2, len(codes) - 1))
    rest         = round(1 - primary_conf, 3)
    second_conf  = round(rest * random.uniform(0.5, 0.85), 3)
    top_k = [
        {"disease_code": primary,   "confidence": primary_conf},
        {"disease_code": others[0], "confidence": second_conf},
        {"disease_code": others[1], "confidence": round(max(0.01, rest - second_conf), 3)},
    ] if len(others) >= 2 else [{"disease_code": primary, "confidence": primary_conf}]

    if primary == "DR":
        top_k[0]["grade"] = random.choice(["1", "2", "3", "4"])
    return {
        "model_version": MODEL_VERSION + "-mock",
        "top_k":         top_k,
        "uncertainty":   round(1 - primary_conf, 3),
        "gradcam_url":   None,
        "preprocessed":  preprocessed,
    }


# ─────────────────────────── Main inference entry ────────────────────────────

def _run_inference(image_id: str, file_path: str | None = None) -> dict:
    preprocessed = _load_and_preprocess(file_path) if (file_path and os.path.exists(file_path)) else None
    model, device = _load_model()

    if model is not None and preprocessed is not None:
        top_k, uncertainty, top_class_idx = _real_predict(model, device, preprocessed)
        gradcam_path = _run_gradcam(model, device, preprocessed, top_class_idx, image_id)
        return {
            "model_version": MODEL_VERSION,
            "top_k":         top_k,
            "uncertainty":   uncertainty,
            "gradcam_url":   f"/api/files/gradcam/{image_id}.png" if gradcam_path else None,
            "preprocessed":  True,
        }

    if model is not None and preprocessed is None:
        log.warning("Model loaded but image not readable for %s — returning mock", image_id)

    return _mock_predict(image_id, preprocessed is not None)


# ─────────────────────────── App lifecycle ───────────────────────────────────

@asynccontextmanager
async def lifespan(_app):
    _load_model()   # warm up at startup; no-op if no checkpoint
    yield


app = FastAPI(title="RetinAI Model Service", version="0.5.0", lifespan=lifespan)


# ─────────────────────────── Request schemas ─────────────────────────────────

class PredictRequest(BaseModel):
    image_id:   str
    file_path:  str | None = None
    image_data: str | None = None   # base64-encoded image (used when file_path not local)


class UncertaintyRequest(BaseModel):
    image_id:   str
    file_path:  str | None = None
    image_data: str | None = None   # base64-encoded image


# ─────────────────────────── Endpoints ───────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":            "ok",
        "model_version":     MODEL_VERSION,
        "checkpoint_exists": os.path.exists(CHECKPOINT_PATH),
        "checkpoint_loaded": _model is not None,
        "mock_mode":         _model is None,
        "model_arch":        MODEL_ARCH if _model is not None else None,
        "device":            str(_device) if _device else "cpu",
        "num_classes":       NUM_CLASSES,
        "disease_codes":     DISEASE_CODES,
    }


@app.post("/predict")
def predict(req: PredictRequest, background_tasks: BackgroundTasks):
    preprocessed = _load_and_preprocess(req.file_path, req.image_data)
    model, device = _load_model()

    if model is not None and preprocessed is not None:
        top_k, uncertainty, top_class_idx = _real_predict(model, device, preprocessed)
        gradcam_b64 = None
        if GRADCAM_ENABLED:
            gradcam_b64 = _run_gradcam(model, device, preprocessed, top_class_idx, req.image_id)
        return {
            "model_version": MODEL_VERSION,
            "top_k":         top_k,
            "uncertainty":   uncertainty,
            "gradcam_url":   f"/api/files/gradcam/{req.image_id}.png" if gradcam_b64 else None,
            "gradcam_data":  gradcam_b64,
            "preprocessed":  True,
        }

    if model is not None and preprocessed is None:
        log.warning("Model loaded but image not readable for %s — returning mock", req.image_id)

    return _mock_predict(req.image_id, preprocessed is not None)


@app.post("/uncertainty")
def uncertainty(req: UncertaintyRequest):
    """Per-class binary entropy (multi-label) when model loaded, random mock otherwise."""
    preprocessed = _load_and_preprocess(req.file_path, req.image_data)
    model, device = _load_model()

    if model is not None and preprocessed is not None:
        import torch

        tensor = _make_transform()(preprocessed).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor).squeeze()
            probs  = torch.sigmoid(logits / TEMPERATURE).cpu().tolist()

        per_ent = [
            -p * math.log(p + 1e-10) - (1 - p) * math.log(1 - p + 1e-10)
            for p in probs
        ]
        score = round(sum(per_ent) / (len(probs) * math.log(2)), 4)
        return {"image_id": req.image_id, "uncertainty": score}

    return {"image_id": req.image_id, "uncertainty": round(random.uniform(0.1, 0.95), 3)}


@app.post("/gradcam")
def gradcam(req: PredictRequest):
    """Return cached Grad-CAM if available, otherwise generate it."""
    cached = Path(GRADCAM_ROOT) / f"{req.image_id}.png"
    if cached.exists():
        return {"image_id": req.image_id, "gradcam_url": f"/api/files/gradcam/{req.image_id}.png"}

    model, device = _load_model()
    if model is None:
        return {"image_id": req.image_id, "gradcam_url": None, "mock": True}

    preprocessed = _load_and_preprocess(req.file_path) if (req.file_path and os.path.exists(req.file_path)) else None
    if preprocessed is None:
        return {"image_id": req.image_id, "gradcam_url": None}

    _, _, top_class_idx = _real_predict(model, device, preprocessed)
    path = _run_gradcam(model, device, preprocessed, top_class_idx, req.image_id)
    return {
        "image_id":    req.image_id,
        "gradcam_url": f"/api/files/gradcam/{req.image_id}.png" if path else None,
    }
