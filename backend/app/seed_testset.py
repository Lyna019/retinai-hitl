"""
Test-set importer — loads a fundus image dataset (ZIP + CSV) into RetinAI HITL.

Expected CSV columns:
    filename, patient_id, laterality (L/R), device, split,
    Myopia, Glaucoma, DR, ME, Uveite, ARMD,
    Retinal_Detachment, HTN, Healthy, Other,
    DR_severity, Glaucoma_severity, ARMD_severity, HTN_severity

Prerequisites (run once before this script):
    docker compose exec backend python -m app.seed

Typical usage — the host directories are already volume-mounted in docker-compose.yml,
so no file-copying is needed:
    docker compose exec backend python -m app.seed_testset \\
        --zip "/tmp/clinical/fundus_dataset.zip" \\
        --csv "/tmp/downloads/master_dataset_final2 (2).csv"

Options:
    --zip         Path to fundus_dataset.zip  (required)
    --csv         Path to master_dataset CSV  (required)
    --images-out  Where to extract images     (default: <images_root>/testset)
    --split       Comma-separated splits      (default: test,val)
    --db-url      SQLAlchemy URL override     (default: from config)

Idempotent — safe to re-run; already-imported rows are skipped.
"""
import argparse
import csv
import random
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base, FundusImage, ModelPrediction, ModelVersion, Patient


# ── Label column → Disease code ───────────────────────────────────────────────

LABEL_TO_CODE: dict[str, str] = {
    "Myopia":             "MYOPIE_F",
    "Glaucoma":           "GLAUC",
    "DR":                 "DR",
    "ME":                 "OED_MAC",
    "Uveite":             "UVEITE",
    "ARMD":               "DMLA",
    "Retinal_Detachment": "DR_MAC_OFF",
    "HTN":                "HTN_DR",
    "Healthy":            "NORMAL",
    # "Other" has no stable ophthalmic code — skipped
}

# ── Severity column → grade value (per disease) ───────────────────────────────
#   Severity 0 means "not affected" → no grade attached.

SEVERITY_MAP: dict[str, tuple[str, dict[int, str]]] = {
    "DR":      ("DR_severity",       {1: "1", 2: "2", 3: "3", 4: "4"}),
    "GLAUC":   ("Glaucoma_severity", {2: "DEB", 3: "EVO", 4: "EVO"}),
    "DMLA":    ("ARMD_severity",     {2: "SEC", 3: "HUM"}),
    "HTN_DR":  ("HTN_severity",      {1: "1", 2: "2", 3: "3", 4: "4"}),
}

DEVICE_TO_MODALITY: dict[str, str] = {
    "Daytona": "UWF",   # Optos Daytona — ultra-widefield
    "Triton":  "CFP",   # Optos Triton — standard colour fundus photography
}

MODEL_VERSION_TAG = "v1.0.0-testset"

RNG = random.Random(99)   # fixed seed → reproducible confidence scores


# ── Helpers ───────────────────────────────────────────────────────────────────

def _int(value) -> int:
    """Safe int coercion (empty string → 0)."""
    try:
        return int(value or 0)
    except (ValueError, TypeError):
        return 0


def _grade_for(disease_code: str, row: dict) -> str | None:
    """Return the grade string for a disease, or None if not applicable."""
    if disease_code in SEVERITY_MAP:
        col, grade_map = SEVERITY_MAP[disease_code]
        return grade_map.get(_int(row.get(col, 0)))
    return None


def _build_top_k(positive_codes: list[str], row: dict) -> list[dict]:
    """
    Build a top-k prediction list from ground-truth labels.
    The primary disease gets a high-confidence score; co-morbidities get moderate scores.
    Returns up to 3 entries sorted by confidence descending.
    """
    if not positive_codes:
        positive_codes = ["NORMAL"]

    entries = []
    for rank, code in enumerate(positive_codes):
        # Primary label: 0.72–0.92 · secondary labels: 0.30–0.55
        if rank == 0:
            conf = round(RNG.uniform(0.72, 0.92), 3)
        else:
            conf = round(RNG.uniform(0.30, 0.55), 3)

        entry: dict = {"disease_code": code, "confidence": conf}
        grade = _grade_for(code, row)
        if grade:
            entry["grade"] = grade
        entries.append(entry)

    entries.sort(key=lambda e: e["confidence"], reverse=True)
    return entries[:3]


# ── Main import logic ─────────────────────────────────────────────────────────

def seed_testset(
    zip_path: str,
    csv_path: str,
    images_out: str,
    splits: list[str],
    db_url: str,
    limit_per_split: int = 0,
) -> None:

    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    out_dir = Path(images_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Read & filter CSV ──────────────────────────────────────────────────
    print(f"\n── Step 1 : reading CSV ──")
    with open(csv_path, encoding="utf-8-sig", errors="replace", newline="") as f:
        all_rows = list(csv.DictReader(f))

    # Group by split, apply per-split limit, then flatten
    rows: list[dict] = []
    for sp in splits:
        sp_rows = [r for r in all_rows if r.get("split", "").strip().lower() == sp]
        if limit_per_split:
            sp_rows = sp_rows[:limit_per_split]
        rows.extend(sp_rows)
        print(f"  {sp:6s} → {len(sp_rows)} rows" + (f"  (limit {limit_per_split})" if limit_per_split else ""))
    print(f"  Total  → {len(rows)} rows  (CSV total: {len(all_rows)})")

    needed_filenames: set[str] = {r["filename"] for r in rows}

    # ── 2. Extract images from ZIP ────────────────────────────────────────────
    print(f"\n── Step 2 : extracting images → {out_dir} ──")
    extracted = already_there = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            fname = Path(member).name
            if fname not in needed_filenames:
                continue
            dest = out_dir / fname
            if dest.exists():
                already_there += 1
            else:
                dest.write_bytes(zf.read(member))
                extracted += 1
    print(f"  Extracted : {extracted}   Already present : {already_there}")

    # ── 3. Ensure ModelVersion row ────────────────────────────────────────────
    print(f"\n── Step 3 : model version ──")
    mv = db.query(ModelVersion).filter(ModelVersion.version_tag == MODEL_VERSION_TAG).first()
    if not mv:
        mv = ModelVersion(
            version_tag=MODEL_VERSION_TAG,
            description="Ground-truth labels imported from master_dataset_final2 (test + val splits)",
            metrics_json={},
            is_active=False,
            trained_at=datetime(2026, 1, 1),
        )
        db.add(mv)
        db.flush()
        print(f"  + ModelVersion {MODEL_VERSION_TAG}")
    else:
        print(f"  · ModelVersion {MODEL_VERSION_TAG} already exists")

    # ── 4. Patients / Images / Predictions ───────────────────────────────────
    print(f"\n── Step 4 : patients, images, predictions ──")

    patient_cache: dict[str, str] = {}   # clinical_id → patient.id (DB UUID)
    n_patients = n_images = n_skipped = 0

    try:
        for row in rows:
            clinical_id = f"TS-{row['patient_id'].strip()}"

            # ── Patient ───────────────────────────────────────────────────────
            if clinical_id not in patient_cache:
                patient = db.query(Patient).filter(
                    Patient.clinical_id == clinical_id
                ).first()
                if not patient:
                    patient = Patient(
                        clinical_id=clinical_id,
                        full_name=f"Patient {row['patient_id'].strip()}",
                    )
                    db.add(patient)
                    db.flush()
                    n_patients += 1
                patient_cache[clinical_id] = patient.id

            patient_db_id = patient_cache[clinical_id]

            # ── Eye ───────────────────────────────────────────────────────────
            eye = "OS" if row.get("laterality", "R").strip().upper() == "L" else "OD"

            # ── Idempotency check ─────────────────────────────────────────────
            fname = row["filename"]
            existing = (
                db.query(FundusImage)
                .filter(
                    FundusImage.patient_id == patient_db_id,
                    FundusImage.file_path.like(f"%{fname}%"),
                )
                .first()
            )
            if existing:
                n_skipped += 1
                continue

            # ── FundusImage ───────────────────────────────────────────────────
            modality    = DEVICE_TO_MODALITY.get(row.get("device", "Daytona"), "UWF")
            img_path    = out_dir / fname
            capture_dt  = datetime.now() - timedelta(days=RNG.randint(30, 365))
            uncertainty = round(RNG.uniform(0.10, 0.60), 3)

            img = FundusImage(
                patient_id        = patient_db_id,
                file_path         = str(img_path),
                modality          = modality,
                eye               = eye,
                capture_date      = capture_dt,
                study_date        = capture_dt,
                status            = "pending",
                uncertainty_score = uncertainty,
            )
            db.add(img)
            db.flush()
            n_images += 1

            # ── ModelPrediction (ground-truth labels as top-k) ─────────────
            positive_codes = [
                code
                for col, code in LABEL_TO_CODE.items()
                if _int(row.get(col, 0)) == 1
            ]
            top_k    = _build_top_k(positive_codes, row)
            top_conf = top_k[0]["confidence"] if top_k else 0.5

            db.add(ModelPrediction(
                image_id      = img.id,
                model_version = MODEL_VERSION_TAG,
                top_k_json    = top_k,
                confidence    = top_conf,
            ))

            # Progress heartbeat every 500 rows
            if n_images % 500 == 0:
                print(f"  … {n_images} images imported so far")
                db.commit()   # intermediate commit to avoid giant transaction

        db.commit()

    except Exception as exc:
        db.rollback()
        print(f"\n✗ Error during import: {exc}")
        raise
    finally:
        db.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"""
✓ Test-set import complete.
  Splits imported  : {', '.join(splits)}
  Patients created : {n_patients}
  Images imported  : {n_images}
  Images skipped   : {n_skipped}  (already in DB)
  Images extracted : {extracted} new  +  {already_there} already on disk
  Output directory : {out_dir}
""")


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main() -> None:
    from .config import settings

    parser = argparse.ArgumentParser(
        description="Import a fundus image dataset (ZIP + CSV) into RetinAI HITL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--zip", required=True, metavar="PATH",
        help="Path to fundus_dataset.zip",
    )
    parser.add_argument(
        "--csv", required=True, metavar="PATH",
        help="Path to master_dataset_final2.csv",
    )
    parser.add_argument(
        "--images-out",
        default=str(Path(settings.images_root) / "testset"),
        metavar="DIR",
        help="Directory where images will be extracted (default: <images_root>/testset)",
    )
    parser.add_argument(
        "--split", default="test,val", metavar="SPLITS",
        help="Comma-separated list of splits to import (default: test,val)",
    )
    parser.add_argument(
        "--db-url", default=settings.database_url, metavar="URL",
        help="SQLAlchemy database URL (default: from config / .env)",
    )
    parser.add_argument(
        "--limit", default=0, type=int, metavar="N",
        help="Max images to import per split (default: 0 = all)",
    )

    args = parser.parse_args()
    splits = [s.strip().lower() for s in args.split.split(",") if s.strip()]

    seed_testset(
        zip_path        = args.zip,
        csv_path        = args.csv,
        images_out      = args.images_out,
        splits          = splits,
        db_url          = args.db_url,
        limit_per_split = args.limit,
    )


if __name__ == "__main__":
    main()
