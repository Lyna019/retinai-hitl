"""
Demo dataset loader — creates synthetic patients, fundus images, and model predictions
for UI simulation and testing.

Run once after seed.py:
    docker compose exec backend python -m app.seed_demo

What it creates
───────────────
  • 1 ModelVersion  (v1.0.0-demo, active)
  • 14 Patients     (realistic Algerian clinical IDs, mixed demographics)
  • 28 FundusImages (OD + OS per patient, varying uncertainty & status)
  • 28 ModelPredictions (top-3 disease predictions with confidence scores)
  • Placeholder PNG thumbnails written to MEDIA_ROOT/demo/

Idempotent — safe to run multiple times (skips existing records).
"""
import random
import struct
import zlib
from datetime import datetime, timedelta
from pathlib import Path

from .database import SessionLocal, Base, engine
from .models import (
    Patient, FundusImage, ModelVersion, ModelPrediction,
    PatientSystemicDisease, SystemicDisease,
)
from .config import settings
from .seed import seed as seed_catalog


# ─── Deterministic random for reproducibility ────────────────────────────────
RNG = random.Random(42)


# ─── Demo patients ────────────────────────────────────────────────────────────

PATIENTS = [
    {"clinical_id": "P-1001", "full_name": "Amine Boukhalfa",     "age": 58, "gender": "M",
     "systemic": ["Diabète de type 2", "Hypertension artérielle"]},
    {"clinical_id": "P-1002", "full_name": "Fatima Zerrouk",      "age": 72, "gender": "F",
     "systemic": ["DMLA connue", "Dyslipidémie"]},
    {"clinical_id": "P-1003", "full_name": "Karim Mansouri",      "age": 45, "gender": "M",
     "systemic": ["Glaucome familial"]},
    {"clinical_id": "P-1004", "full_name": "Nadia Benali",        "age": 63, "gender": "F",
     "systemic": ["Diabète de type 2", "Obésité"]},
    {"clinical_id": "P-1005", "full_name": "Rachid Amrani",       "age": 54, "gender": "M",
     "systemic": ["Hypertension artérielle"]},
    {"clinical_id": "P-1006", "full_name": "Samira Khelif",       "age": 38, "gender": "F",
     "systemic": []},
    {"clinical_id": "P-1007", "full_name": "Youcef Dahmani",      "age": 67, "gender": "M",
     "systemic": ["Diabète de type 2", "Insuffisance rénale chronique"]},
    {"clinical_id": "P-1008", "full_name": "Leïla Cheriet",       "age": 49, "gender": "F",
     "systemic": ["Lupus érythémateux systémique"]},
    {"clinical_id": "P-1009", "full_name": "Omar Hadj-Slimane",   "age": 71, "gender": "M",
     "systemic": ["Atteinte thyroïdienne", "Hypertension artérielle"]},
    {"clinical_id": "P-1010", "full_name": "Djamila Aissaoui",    "age": 55, "gender": "F",
     "systemic": ["Diabète de type 2"]},
    {"clinical_id": "P-1011", "full_name": "Sofiane Mebarki",     "age": 42, "gender": "M",
     "systemic": []},
    {"clinical_id": "P-1012", "full_name": "Houria Bensalem",     "age": 60, "gender": "F",
     "systemic": ["Polyarthrite rhumatoïde", "Sarcoïdose"]},
    {"clinical_id": "P-1013", "full_name": "Abdelkader Touati",   "age": 33, "gender": "M",
     "systemic": ["Drépanocytose"]},
    {"clinical_id": "P-1014", "full_name": "Meriem Bouchama",     "age": 78, "gender": "F",
     "systemic": ["Diabète de type 2", "Dyslipidémie", "Obésité"]},
]


# ─── Per-patient prediction scenarios ────────────────────────────────────────
# Each patient has two images (OD, OS). We assign a primary diagnosis for each eye.
# Format: (disease_code, top_confidence, grade_or_None, uncertainty, status)

SCENARIOS = [
    # P-1001 — diabetic, hypertensive → DR moderate OD, HTN retinopathy OS
    [("DR",      0.87, "2", 0.13, "pending"),   ("HTN_DR", 0.72, "2", 0.28, "pending")],
    # P-1002 — DMLA
    [("DMLA",    0.91, "HUM", 0.09, "pending"),  ("DMLA",   0.76, "SEC", 0.24, "pending")],
    # P-1003 — glaucoma
    [("GLAUC",   0.83, "EVO", 0.17, "pending"),  ("GLAUC",  0.69, "DEB", 0.31, "pending")],
    # P-1004 — diabetic, obese → DR proliferant OD, DR moderate OS
    [("DR",      0.78, "4",   0.22, "pending"),  ("DR",     0.65, "2",  0.35, "pending")],
    # P-1005 — hypertensive → HTN retinopathy both eyes
    [("HTN_DR",  0.80, "3",   0.20, "done"),     ("HTN_DR", 0.74, "2",  0.26, "done")],
    # P-1006 — young, no systemic → normal both eyes
    [("NORMAL",  0.95, None,  0.05, "done"),     ("NORMAL", 0.93, None, 0.07, "done")],
    # P-1007 — diabetic, renal → DR severe + high uncertainty
    [("DR",      0.61, "3",   0.39, "pending"),  ("DR",     0.55, "3",  0.45, "pending")],
    # P-1008 — lupus → chorioretinitis
    [("CHORIO",  0.73, None,  0.27, "pending"),  ("UVEITE", 0.68, None, 0.32, "pending")],
    # P-1009 — older hypertensive → OACR urgent OD (P1), HTN OS
    [("OACR",    0.89, None,  0.11, "pending"),  ("HTN_DR", 0.77, "3",  0.23, "pending")],
    # P-1010 — diabetic → DR mild OD done, DR moderate OS in_progress
    [("DR",      0.82, "1",   0.18, "done"),     ("DR",     0.70, "2",  0.30, "in_progress")],
    # P-1011 — young healthy → normal OD, myopie forte OS
    [("NORMAL",  0.90, None,  0.10, "pending"),  ("MYOPIE_F", 0.66, None, 0.34, "pending")],
    # P-1012 — polyarthrite, sarcoïdose → uvéite both
    [("UVEITE",  0.75, None,  0.25, "pending"),  ("CHORIO", 0.62, None, 0.38, "pending")],
    # P-1013 — drépanocytose → BRVO OD (P1 territory), normal OS
    [("BRVO",    0.71, None,  0.29, "pending"),  ("NORMAL", 0.88, None, 0.12, "pending")],
    # P-1014 — elderly diabetic → DR proliferant both eyes, very uncertain
    [("DR",      0.52, "4",   0.48, "pending"),  ("DR",     0.49, "4",  0.51, "pending")],
]

# Second and third prediction candidates (code, confidence_delta)
RUNNER_UPS = {
    "DR":       [("DMLA", -0.25), ("NORMAL", -0.35)],
    "DMLA":     [("DR",   -0.20), ("NORMAL", -0.30)],
    "GLAUC":    [("MYOPIE_F", -0.22), ("NORMAL", -0.30)],
    "HTN_DR":   [("DR",  -0.20), ("OVCR",  -0.30)],
    "OACR":     [("ABACR", -0.18), ("HTN_DR", -0.28)],
    "ABACR":    [("OACR", -0.15), ("HTN_DR", -0.25)],
    "NOIAA":    [("OACR", -0.18), ("GLAUC", -0.25)],
    "DR_MAC_OFF": [("DR", -0.20), ("BRVO", -0.28)],
    "GLAUC_AIGU": [("GLAUC", -0.18), ("HTN_DR", -0.28)],
    "OVCR":     [("BRVO", -0.18), ("HTN_DR", -0.25)],
    "BRVO":     [("OVCR", -0.15), ("DR",    -0.25)],
    "CHORIO":   [("UVEITE", -0.15), ("DR",  -0.30)],
    "UVEITE":   [("CHORIO", -0.15), ("NORMAL", -0.30)],
    "STARGARDT":[("DMLA", -0.20), ("NORMAL", -0.30)],
    "MYOPIE_F": [("NORMAL", -0.22), ("GLAUC", -0.30)],
    "NORMAL":   [("DR",   -0.30), ("DMLA",  -0.40)],
}


# ─── Minimal PNG generator (pure stdlib, no Pillow required) ─────────────────

def _png_chunk(name: bytes, data: bytes) -> bytes:
    c = zlib.crc32(name + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + name + data + struct.pack(">I", c)


def make_fundus_png(eye: str = "OD", size: int = 128) -> bytes:
    """
    Generate a tiny synthetic fundus-like PNG:
    dark background, warm circular disk with faint radial lines.
    Pure stdlib — no external dependencies.
    """
    # Build raw RGB pixel rows
    cx, cy = size // 2, size // 2
    radius = int(size * 0.46)
    disc_x = cx + (int(size * 0.16) if eye == "OD" else -int(size * 0.16))
    disc_y = cy

    rows = []
    for y in range(size):
        row = []
        for x in range(size):
            dx, dy = x - cx, y - cy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > radius:
                row.extend([4, 4, 4])           # near-black background
            else:
                # Base warm reddish-brown
                norm = dist / radius             # 0 at centre, 1 at rim
                r = int(80 - norm * 40)
                g = int(30 - norm * 15)
                b = int(10 - norm * 5)
                # Optic disc
                ddx, ddy = x - disc_x, y - disc_y
                if (ddx * ddx + ddy * ddy) ** 0.5 < size * 0.07:
                    r, g, b = 210, 160, 80
                row.extend([max(0, r), max(0, g), max(0, b)])
        # PNG filter byte (0 = None) + row data
        rows.append(bytes([0]) + bytes(row))

    raw = b"".join(rows)
    compressed = zlib.compress(raw)

    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR",
                      struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
    png += _png_chunk(b"IDAT", compressed)
    png += _png_chunk(b"IEND", b"")
    return png


# ─── Main seed function ───────────────────────────────────────────────────────

def seed_demo():
    # Ensure catalog (diseases, lesions, users, etc.) is seeded first
    print("── Étape 1 : catalogue & utilisateurs ──")
    seed_catalog()

    print("\n── Étape 2 : patients & images de démonstration ──")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    demo_dir = Path(settings.images_root) / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── ModelVersion ────────────────────────────────────────────────────
        mv = db.query(ModelVersion).filter(ModelVersion.version_tag == "v1.0.0-demo").first(
            
        )
        if not mv:
            mv = ModelVersion(
                version_tag="v1.0.0-demo",
                description="Modèle de démonstration — ResNet-50 pré-entraîné ODIR-5K",
                metrics_json={"auc": 0.91, "f1_macro": 0.83, "accuracy": 0.87},
                is_active=True,
                trained_at=datetime(2026, 1, 15),
            )
            db.add(mv)
            db.flush()
            print(f"  + ModelVersion v1.0.0-demo")
        else:
            print(f"  · ModelVersion v1.0.0-demo already exists")

        # ── Patients + Images + Predictions ─────────────────────────────────
        systemic_cache = {
            s.name_fr: s.id
            for s in db.query(SystemicDisease).all()
        }

        for i, (pdata, scenario) in enumerate(zip(PATIENTS, SCENARIOS)):
            # Patient
            patient = db.query(Patient).filter(
                Patient.clinical_id == pdata["clinical_id"]
            ).first()
            if not patient:
                birth_year = datetime.now().year - pdata["age"]
                patient = Patient(
                    clinical_id=pdata["clinical_id"],
                    full_name=pdata["full_name"],
                    age=pdata["age"],
                    gender=pdata["gender"],
                    birth_date=datetime(birth_year, 6, 1),
                )
                db.add(patient)
                db.flush()

                # Systemic diseases
                for sname in pdata["systemic"]:
                    sid = systemic_cache.get(sname)
                    if sid:
                        db.add(PatientSystemicDisease(
                            patient_id=patient.id,
                            systemic_disease_id=sid,
                        ))

                print(f"  + Patient {pdata['clinical_id']}  {pdata['full_name']}")
            else:
                print(f"  · Patient {pdata['clinical_id']} already exists")

            # Images (OD + OS)
            for eye_idx, (eye, (disease_code, conf, grade, uncertainty, status)) in enumerate(
                zip(["OD", "OS"], scenario)
            ):
                existing_img = (
                    db.query(FundusImage)
                    .filter(
                        FundusImage.patient_id == patient.id,
                        FundusImage.eye == eye,
                    )
                    .first()
                )
                if existing_img:
                    continue

                # Write placeholder PNG
                png_path = demo_dir / f"{pdata['clinical_id']}_{eye}.png"
                if not png_path.exists():
                    png_path.write_bytes(make_fundus_png(eye=eye, size=128))

                capture_date = datetime.now() - timedelta(days=RNG.randint(10, 180))

                img = FundusImage(
                    patient_id=patient.id,
                    file_path=str(png_path),
                    modality=RNG.choice(["STD", "UWF", "STD"]),   # STD more common
                    eye=eye,
                    capture_date=capture_date,
                    study_date=capture_date,
                    status=status,
                    uncertainty_score=round(uncertainty + RNG.uniform(-0.03, 0.03), 3),
                )
                db.add(img)
                db.flush()

                # ModelPrediction — top-3
                runners = RUNNER_UPS.get(disease_code, [("NORMAL", -0.30), ("DMLA", -0.40)])
                top_k = [{"disease_code": disease_code, "confidence": round(conf, 3)}]
                if grade:
                    top_k[0]["grade"] = grade
                for run_code, delta in runners[:2]:
                    run_conf = max(0.05, round(conf + delta + RNG.uniform(-0.04, 0.04), 3))
                    top_k.append({"disease_code": run_code, "confidence": run_conf})

                db.add(ModelPrediction(
                    image_id=img.id,
                    model_version="v1.0.0-demo",
                    top_k_json=top_k,
                    confidence=conf,
                ))
                print(f"    + Image {eye}  [{disease_code}  σ={uncertainty:.2f}  {status}]")

        db.commit()
        print("\n✓ Demo dataset chargé avec succès.")
        print(f"  {len(PATIENTS)} patients · {len(PATIENTS) * 2} images")

    except Exception as e:
        db.rollback()
        print(f"\n✗ Erreur : {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_demo()
