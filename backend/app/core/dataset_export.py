"""
Auto-export submitted annotations to a curated CSV dataset.

CSV columns:
  filename, patient_clinical_id, eye, modality, capture_date,
  image_quality,                         ← doctor-assessed quality
  disease_codes, grades, urgency_level,
  gradcam_verdict,                        ← correct | partial | wrong | (empty)
  has_hemorrhage, has_exudate, has_microaneurysm, has_drusen,
  has_neovaisseau, has_cotton_wool, has_edema,
  lesion_locations,                       ← JSON: [{region, row, col, zoom, lesion}]
  notes, doctor_username, submitted_at, time_spent_sec

Located at: /app/data/curated_dataset.csv
Appended on every submit — one row per submitted annotation.
"""
import csv
import json
import os
from datetime import datetime
from pathlib import Path

DATASET_CSV = os.getenv("CURATED_DATASET_PATH", "/app/data/curated_dataset.csv")

LESION_CODES = ["HEM", "EXS", "MA", "DRUS", "NV", "CW", "OED"]
LESION_NAMES = {
    "HEM":  "has_hemorrhage",
    "EXS":  "has_exudate",
    "MA":   "has_microaneurysm",
    "DRUS": "has_drusen",
    "NV":   "has_neovaisseau",
    "CW":   "has_cotton_wool",
    "OED":  "has_edema",
}

HEADER = [
    # ── Image metadata ─────────────────────────────
    "filename",
    "patient_clinical_id",
    "eye",
    "modality",
    "capture_date",
    # ── Quality assessments ────────────────────────
    "image_quality",          # good | fair | poor
    "gradcam_verdict",        # correct | partial | wrong | ""
    # ── Diagnosis ──────────────────────────────────
    "disease_codes",          # pipe-separated: DR|GLAUC
    "grades",                 # pipe-separated, aligned with disease_codes
    "urgency_level",          # P1 | P2 | P3 | ""
    # ── Lesion presence (binary) ───────────────────
    *LESION_NAMES.values(),
    # ── Lesion localization (exact) ───────────────
    "lesion_locations",       # JSON: [{"region":"Macula","row":2,"col":3,"zoom":1,"lesion":"HEM"}]
    # ── Annotation metadata ────────────────────────
    "notes",
    "doctor_username",
    "submitted_at",
    "time_spent_sec",
]


def _ensure_header():
    path = Path(DATASET_CSV)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HEADER)


def append_annotation(
    *,
    file_path: str,
    patient_clinical_id: str,
    eye: str,
    modality: str,
    capture_date,
    image_quality: str | None,
    gradcam_verdict: str | None,
    disease_labels: list[dict],
    regions: list[dict],
    urgency_level: str | None,
    notes_text: str | None,
    doctor_username: str,
    submitted_at: datetime,
    time_spent_sec: int,
):
    """Append one row to the curated dataset CSV."""
    try:
        _ensure_header()

        # ── Lesion presence (binary) ────────────────────────────────────────
        lesion_set: set[str] = set()
        for roi in (regions or []):
            for cell in (roi.get("cells") or []):
                code = cell.get("lesion_code")
                if code:
                    lesion_set.add(code)

        # ── Lesion localization (exact coords) ─────────────────────────────
        locations = []
        for roi in (regions or []):
            region_name = roi.get("custom_region_name") or roi.get("anatomical_region_name") or ""
            for cell in (roi.get("cells") or []):
                if cell.get("lesion_code"):
                    locations.append({
                        "region":  region_name,
                        "row":     cell.get("row"),
                        "col":     cell.get("col"),
                        "zoom":    cell.get("zoom_level", 1),
                        "lesion":  cell.get("lesion_code"),
                    })

        disease_codes = "|".join(
            lbl["disease_code"] for lbl in disease_labels if lbl.get("disease_code")
        )
        grades = "|".join(str(lbl.get("grade") or "") for lbl in disease_labels)
        filename = Path(file_path).name if file_path else ""

        row = [
            filename,
            patient_clinical_id or "",
            eye or "",
            modality or "",
            capture_date.date().isoformat() if capture_date else "",
            # Quality
            image_quality or "",
            gradcam_verdict or "",
            # Diagnosis
            disease_codes,
            grades,
            urgency_level or "",
            # Lesion binary
            *[("1" if lc in lesion_set else "0") for lc in LESION_CODES],
            # Lesion exact
            json.dumps(locations, ensure_ascii=False) if locations else "[]",
            # Meta
            (notes_text or "").replace("\n", " "),
            doctor_username,
            submitted_at.isoformat() if submitted_at else "",
            str(time_spent_sec or 0),
        ]

        with open(DATASET_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    except Exception as exc:
        import logging
        logging.getLogger("retinai").warning("CSV export failed: %s", exc)
