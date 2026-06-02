"""
RetinAI — RetiZero inference service (port 9001)

Zero-shot retinal disease recognition using the RetiZero CLIP-R foundation model.
Labels are loaded from the clinician-validated taxonomy (taxonomy.yaml).

Architecture (inferred from checkpoint):
  Vision : ViT-L/16 (embed_dim=1024, depth=24, heads=16) + LoRA r=8 on Q,V
           → linear projection 1024 → 512
  Text   : BERT-base-cased (hidden=768, layers=12) + pooler
           → linear projection 768 → 512

Endpoints:
  GET  /health
  POST /describe   { image_id, file_path } → { description, top_labels }
"""
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertModel, BertConfig, BertTokenizerFast

log = logging.getLogger("retizero-service")
logging.basicConfig(level=logging.INFO)

WEIGHTS_PATH  = os.getenv("WEIGHTS_PATH",  "/app/checkpoints/RetiZero.pth")
TAXONOMY_PATH = os.getenv("TAXONOMY_PATH", "/app/taxonomy.yaml")

# ─── Load labels from taxonomy ───────────────────────────────────────────────

def _load_taxonomy_labels(path: str) -> list[dict]:
    """
    Parse the clinician-validated taxonomy YAML and return all retinal
    pathologies and signs suitable as CLIP text labels.

    Excludes:
      - anterior-segment-only entries  (anterior_segment: true)
      - procedural markers              (procedural: true)
      - metadata / dropped blocks       (keys starting with _)
    """
    try:
        with open(path, encoding="utf-8") as f:
            tax = yaml.safe_load(f)
    except FileNotFoundError:
        log.warning("Taxonomy not found at %s — using fallback labels", path)
        return _fallback_labels()

    labels = []
    for code, entry in tax.items():
        if code.startswith("_") or not isinstance(entry, dict):
            continue
        kind = entry.get("kind")
        if kind not in ("PATH", "SIGN"):
            continue
        if entry.get("anterior_segment", False):
            continue
        if entry.get("procedural", False):
            continue

        full_name = entry.get("full_name", code)
        # French name: prefer synonym with accented chars, else first lowercase synonym
        # that isn't an acronym and isn't the English full_name
        name_fr = full_name
        synonyms = entry.get("synonyms", [])
        # Pass 1: accented character → definitely French
        for syn in synonyms:
            if any(c in syn for c in "éèêëàâùûüœçîïôÉÈÊËÀÂÙÛÜŒÇÎÏÔ"):
                name_fr = syn
                break
        else:
            # Pass 2: first lowercase synonym that isn't all-caps and isn't English
            for syn in synonyms:
                if syn and syn[0].islower() and syn != full_name.lower() and len(syn) > 3:
                    name_fr = syn
                    break

        labels.append({
            "code":      code,
            "full_name": full_name,   # English — fed to BERT text encoder
            "name_fr":   name_fr,     # French  — shown in description
            "kind":      kind,
        })

    log.info("Taxonomy loaded: %d labels (%d PATH, %d SIGN)",
             len(labels),
             sum(1 for l in labels if l["kind"] == "PATH"),
             sum(1 for l in labels if l["kind"] == "SIGN"))
    return labels


def _fallback_labels() -> list[dict]:
    """Minimal fallback when taxonomy.yaml is absent."""
    return [
        {"code": "DR",    "full_name": "Diabetic Retinopathy",          "name_fr": "Rétinopathie diabétique",         "kind": "PATH"},
        {"code": "ARMD",  "full_name": "Age-Related Macular Degeneration","name_fr": "Dégénérescence maculaire (DMLA)", "kind": "PATH"},
        {"code": "ME",    "full_name": "Macular Edema",                  "name_fr": "Œdème maculaire",                 "kind": "PATH"},
        {"code": "Glaucoma","full_name": "Glaucoma",                     "name_fr": "Glaucome",                        "kind": "PATH"},
        {"code": "HTN",   "full_name": "Hypertensive Retinopathy",       "name_fr": "Rétinopathie hypertensive",       "kind": "PATH"},
        {"code": "Retinal_Detachment","full_name":"Retinal Detachment",  "name_fr": "Décollement de rétine",           "kind": "PATH"},
        {"code": "Myopia","full_name": "Degenerative Myopia",            "name_fr": "Myopie forte",                    "kind": "PATH"},
        {"code": "CRVO",  "full_name": "Central Retinal Vein Occlusion", "name_fr": "Occlusion veine centrale rétine", "kind": "PATH"},
        {"code": "Retinitis_Pigmentosa","full_name":"Retinitis Pigmentosa","name_fr":"Rétinite pigmentaire",           "kind": "PATH"},
        {"code": "Healthy","full_name":"Normal / Healthy Fundus",        "name_fr": "Fond d'œil normal",               "kind": "SIGN"},
    ]


TAXONOMY_LABELS = _load_taxonomy_labels(TAXONOMY_PATH)

# ─── Image preprocessing ─────────────────────────────────────────────────────

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ─── Model loading ────────────────────────────────────────────────────────────

def _merge_lora_into_vit_sd(raw_sd: dict) -> dict:
    """Build a timm-ViT-L/16-compatible state dict with LoRA merged in."""
    prefix = "vision_model.model.lora_vit."
    H = 1024

    vit_sd: dict = {}
    for k, v in raw_sd.items():
        if not k.startswith(prefix):
            continue
        local = k[len(prefix):]
        if "attn.qkv.qkv." in local:
            vit_sd[local.replace("attn.qkv.qkv.", "attn.qkv.")] = v.clone()
        elif any(tag in local for tag in ("linear_a_q", "linear_b_q", "linear_a_v", "linear_b_v")):
            pass
        else:
            vit_sd[local] = v

    for i in range(24):
        bp = f"vision_model.model.lora_vit.blocks.{i}.attn.qkv."
        base = raw_sd.get(bp + "qkv.weight")
        a_q  = raw_sd.get(bp + "linear_a_q.weight")
        b_q  = raw_sd.get(bp + "linear_b_q.weight")
        a_v  = raw_sd.get(bp + "linear_a_v.weight")
        b_v  = raw_sd.get(bp + "linear_b_v.weight")
        if base is None or a_q is None:
            continue
        merged = base.clone()
        merged[:H]    = merged[:H]    + b_q @ a_q
        merged[2 * H:] = merged[2 * H:] + b_v @ a_v
        vit_sd[f"blocks.{i}.attn.qkv.weight"] = merged

    return vit_sd


def _load_model():
    """
    Load ViT-L/16 + BERT-cased + projection heads from RetiZero.pth.
    Pre-computes text embeddings for all taxonomy labels.
    Returns a component tuple or None on failure.
    """
    if not os.path.exists(WEIGHTS_PATH):
        log.warning("No weights at %s — placeholder mode.", WEIGHTS_PATH)
        return None

    try:
        import timm

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading RetiZero on %s …", dev)

        raw_sd = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)

        # Vision encoder (ViT-L/16 with LoRA merged)
        vit = timm.create_model("vit_large_patch16_224", pretrained=False,
                                 num_classes=0, global_pool="token")
        vit_sd = _merge_lora_into_vit_sd(raw_sd)
        vit.load_state_dict(vit_sd, strict=False)

        # Text encoder (BERT-base-cased, vocab 28996)
        bert_cfg = BertConfig(vocab_size=28996, hidden_size=768,
                              num_hidden_layers=12, num_attention_heads=12,
                              intermediate_size=3072, max_position_embeddings=512)
        bert = BertModel(bert_cfg, add_pooling_layer=True)
        text_pfx = "text_model.model."
        bert_sd = {k[len(text_pfx):]: v for k, v in raw_sd.items() if k.startswith(text_pfx)}
        bert.load_state_dict(bert_sd, strict=False)

        # Projection heads
        vis_proj = nn.Linear(1024, 512, bias=False)
        txt_proj = nn.Linear(768,  512, bias=False)
        vis_proj.weight = nn.Parameter(raw_sd["vision_model.projection_head_vision.projection.weight"])
        txt_proj.weight = nn.Parameter(raw_sd["text_model.projection_head_text.projection.weight"])
        logit_scale = raw_sd["logit_scale"].clone().to(dev)

        vit      = vit.to(dev).eval()
        bert     = bert.to(dev).eval()
        vis_proj = vis_proj.to(dev).eval()
        txt_proj = txt_proj.to(dev).eval()

        # Tokenizer
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

        # Pre-compute text embeddings for all taxonomy labels
        en_texts = [lbl["full_name"] for lbl in TAXONOMY_LABELS]
        with torch.no_grad():
            enc = tokenizer(en_texts, padding=True, truncation=True,
                            max_length=64, return_tensors="pt")
            bert_out  = bert(input_ids=enc["input_ids"].to(dev),
                             attention_mask=enc["attention_mask"].to(dev))
            txt_embs  = F.normalize(txt_proj(bert_out.pooler_output), dim=-1)

        log.info("RetiZero loaded on %s — %d taxonomy labels embedded.",
                 dev, len(TAXONOMY_LABELS))
        return vit, bert, vis_proj, txt_proj, logit_scale, tokenizer, txt_embs, dev

    except Exception as exc:
        log.error("RetiZero load failed: %s", exc, exc_info=True)
        return None


# ─── Model state ─────────────────────────────────────────────────────────────

_components = None


# ─── Inference ───────────────────────────────────────────────────────────────

def _run_inference(file_path: str) -> dict:
    if _components is None:
        return {
            "description": (
                "[RetiZero non disponible — placez RetiZero.pth dans "
                "retizero-service/checkpoints/ puis redémarrez le service]"
            ),
            "top_labels": [],
        }

    vit, bert, vis_proj, txt_proj, logit_scale, tokenizer, txt_embs, dev = _components

    try:
        image = Image.open(file_path).convert("RGB")
        img_t = _TRANSFORM(image).unsqueeze(0).to(dev)

        with torch.no_grad():
            vis_feat = vit(img_t)
            vis_emb  = F.normalize(vis_proj(vis_feat), dim=-1)
            # Independent sigmoid per label (multi-label — findings can co-exist)
            logits   = (vis_emb @ txt_embs.T) * logit_scale.exp()
            probs    = logits.sigmoid().squeeze(0)

        probs_list = probs.cpu().tolist()

        # Separate pathologies from signs
        path_ranked = sorted(
            [(lbl, p) for lbl, p in zip(TAXONOMY_LABELS, probs_list)
             if lbl["kind"] == "PATH"],
            key=lambda x: x[1], reverse=True,
        )
        sign_ranked = sorted(
            [(lbl, p) for lbl, p in zip(TAXONOMY_LABELS, probs_list)
             if lbl["kind"] == "SIGN"],
            key=lambda x: x[1], reverse=True,
        )

        # Adaptive thresholds based on the image's score distribution
        all_scores = probs_list
        mean_s = sum(all_scores) / len(all_scores)
        # "Probable" if score is clearly above the mean; "possible" if modestly above
        thresh_probable = mean_s + 0.12
        thresh_possible = mean_s + 0.04

        # Check if the top finding is "Healthy"
        top_label = path_ranked[0][0] if path_ranked else None
        is_healthy = top_label and top_label["code"] == "Healthy" and path_ranked[0][1] >= thresh_probable

        # Build the clinical description
        lines = ["Analyse fondoscopique RetiZero (modèle fondation) :", ""]

        # Pathologies
        # Exclude meta-labels from clinical display
        _display_skip = {"Healthy", "Other"}
        probable_paths  = [(l, p) for l, p in path_ranked[:8] if p >= thresh_probable and l["code"] not in _display_skip]
        possible_paths  = [(l, p) for l, p in path_ranked[:8] if thresh_possible <= p < thresh_probable and l["code"] not in _display_skip]
        notable_signs   = [(l, p) for l, p in sign_ranked[:6] if p >= thresh_possible and l["code"] != "Healthy"]

        if is_healthy and not probable_paths:
            lines.append("Résultat : Fond d'œil dans les limites de la normale.")
            lines.append("Aucune pathologie rétinienne significative détectée par le modèle de fondation.")
        else:
            if probable_paths:
                lines.append("Pathologies probables :")
                for lbl, p in probable_paths[:4]:
                    lines.append(f"  ► {lbl['name_fr']} ({lbl['code']}) — {p*100:.0f}%")
                lines.append("")

            if possible_paths:
                lines.append("Pathologies possibles :")
                for lbl, p in possible_paths[:3]:
                    lines.append(f"  ◦ {lbl['name_fr']} ({lbl['code']}) — {p*100:.0f}%")
                lines.append("")

            if notable_signs:
                lines.append("Signes associés :")
                for lbl, p in notable_signs[:4]:
                    lines.append(f"  • {lbl['name_fr']} ({lbl['code']})")
                lines.append("")

            # Conclusion
            if probable_paths:
                top_fr  = probable_paths[0][0]["name_fr"]
                top_code = probable_paths[0][0]["code"]
                secondary = f" avec {probable_paths[1][0]['name_fr']}" if len(probable_paths) > 1 else ""
                lines.append(
                    f"Conclusion : Image évocatrice de {top_fr}{secondary}."
                )
            elif possible_paths:
                lines.append(
                    f"Conclusion : Résultats non concluants — "
                    f"{possible_paths[0][0]['name_fr']} possible. Examen clinique recommandé."
                )
            else:
                lines.append("Conclusion : Aucune pathologie significative détectée. Fond d'œil d'aspect normal.")

        return {
            "description": "\n".join(lines),
            "top_labels": [
                {
                    "label":       lbl["full_name"],
                    "label_fr":    lbl["name_fr"],
                    "disease_code": lbl["code"],
                    "kind":        lbl["kind"],
                    "probability": round(p, 4),
                }
                for lbl, p in (path_ranked + sign_ranked)[:10]
            ],
        }

    except Exception as exc:
        log.error("RetiZero inference failed: %s", exc, exc_info=True)
        return {
            "description": f"[Erreur RetiZero : {exc}]",
            "top_labels":  [],
        }


# ─── App lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app):
    global _components
    _components = _load_model()
    yield


app = FastAPI(title="RetinAI RetiZero Service", version="3.0.0", lifespan=lifespan)


# ─── Schemas ─────────────────────────────────────────────────────────────────

class DescribeRequest(BaseModel):
    image_id:  str
    file_path: str | None = None


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    comps  = _components
    loaded = comps is not None
    dev    = str(comps[7]) if loaded else "none"
    return {
        "status":        "ok",
        "model_loaded":  loaded,
        "weights_exist": os.path.exists(WEIGHTS_PATH),
        "device":        dev,
        "weights_path":  WEIGHTS_PATH,
        "num_labels":    len(TAXONOMY_LABELS),
    }


@app.get("/labels")
def list_labels():
    """Return the full label set loaded from taxonomy."""
    return {"labels": TAXONOMY_LABELS, "count": len(TAXONOMY_LABELS)}


@app.post("/describe")
def describe(req: DescribeRequest):
    if not req.file_path or not os.path.exists(req.file_path):
        return {
            "description": f"[Image introuvable : {req.file_path}]",
            "top_labels":  [],
        }
    return _run_inference(req.file_path)
