"""
Seed all PATH entries from the clinician-validated taxonomy into the diseases catalog.
Safe to run multiple times — uses ON CONFLICT DO NOTHING.

Run:
    docker compose exec backend python -m app.seed_taxonomy
"""
import yaml
from .database import SessionLocal
from .models import Disease, Mechanism

# ─── Mechanism heuristics ─────────────────────────────────────────────────────
# Maps taxonomy keywords → DB mechanism code
_MECH_HINTS = {
    "CRAO": "VASC", "BRAO": "VASC", "CRVO": "VASC", "BRVO": "VASC",
    "DR": "VASC", "HTN": "VASC", "HR": "VASC",
    "Vascular": "VASC", "Retinal_Embolism": "VASC", "Retinal_Hemorrhage": "VASC",
    "CNV": "VASC", "HPED": "VASC", "CSCR": "VASC", "PRH": "VASC", "VH": "VASC",
    "ARMD": "DEGEN", "Choroidosis": "DEGEN", "Myopia": "DEGEN",
    "Retinitis_Pigmentosa": "DYST", "Maladie_Best": "DYST", "Maladie_Stargardt": "DYST",
    "CRS": "INFLAM", "Uveite": "INFLAM", "VS": "INFLAM", "RS": "INFLAM",
    "Scar_Toxo": "INFLAM", "ON": "INFLAM",
    "Glaucoma": "STRUCT", "Retinal_Detachment": "STRUCT", "RTR": "STRUCT",
    "GRT": "STRUCT", "RHL": "STRUCT", "RT": "STRUCT", "CB": "STRUCT",
    "ERM": "STRUCT", "MHL": "STRUCT", "ODPM": "STRUCT", "HIC": "STRUCT",
    "AION": "VASC", "Stries_Angioides": "STRUCT",
}

def _get_mech(code: str) -> str | None:
    if code in _MECH_HINTS:
        return _MECH_HINTS[code]
    for key, mech in _MECH_HINTS.items():
        if key.upper() in code.upper():
            return mech
    return None


def _french_name(entry: dict, code: str) -> str:
    """Pick best French name from synonyms or fall back to full_name."""
    for syn in entry.get("synonyms", []):
        if any(c in syn for c in "éèêëàâùûüœçîïôÉÈÊËÀÂÙÛÜŒÇÎÏÔ"):
            return syn
    for syn in entry.get("synonyms", []):
        if syn and syn[0].islower() and len(syn) > 3 and syn != entry.get("full_name","").lower():
            return syn
    return entry.get("full_name", code)


def seed_taxonomy():
    taxonomy_path = "/app/taxonomy.yaml"
    try:
        with open(taxonomy_path, encoding="utf-8") as f:
            tax = yaml.safe_load(f)
    except FileNotFoundError:
        # Try local path for development
        import os
        local = os.path.join(os.path.dirname(__file__), "../../retizero-service/taxonomy.yaml")
        with open(local, encoding="utf-8") as f:
            tax = yaml.safe_load(f)

    db = SessionLocal()
    added = skipped = updated = 0

    try:
        # Collect existing mechanisms
        mechs = {m.code for m in db.query(Mechanism).all()}

        for code, entry in tax.items():
            if code.startswith("_") or not isinstance(entry, dict):
                continue
            if entry.get("kind") not in ("PATH", "SIGN"):
                continue
            if entry.get("anterior_segment", False):
                continue
            if entry.get("procedural", False):
                continue

            name_fr  = _french_name(entry, code)
            mech     = _get_mech(code)
            if mech and mech not in mechs:
                mech = None   # don't reference non-existent mechanism

            existing = db.query(Disease).filter(Disease.code == code).first()
            if existing:
                # Update name if it was just the code or English
                needs_update = (existing.name_fr == code or
                                existing.name_fr == entry.get("full_name", code))
                if needs_update:
                    existing.name_fr = name_fr
                    updated += 1
                else:
                    skipped += 1
                continue

            db.add(Disease(
                code=code,
                name_fr=name_fr,
                description=entry.get("notes", "").strip()[:300] or None,
                mechanism_code=mech,
                is_gradable=bool(entry.get("severity_scale")),
                is_approved=True,
            ))
            added += 1

        db.commit()
        print(f"✓ Taxonomy seed: {added} added, {updated} names updated, {skipped} unchanged.")

    finally:
        db.close()


if __name__ == "__main__":
    seed_taxonomy()
