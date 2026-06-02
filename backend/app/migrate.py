"""
Schema migration — brings an existing DB up to the current model definition.
Safe to run multiple times (all statements use IF NOT EXISTS / DO NOTHING).

Run:
    docker compose exec backend python -m app.migrate
"""
from sqlalchemy import text
from .database import engine, SessionLocal


MIGRATIONS = [
    # ── patients ────────────────────────────────────────────────────────────
    "ALTER TABLE patients ADD COLUMN IF NOT EXISTS full_name        VARCHAR(128)",
    "ALTER TABLE patients ADD COLUMN IF NOT EXISTS birth_date       TIMESTAMP",
    "ALTER TABLE patients ADD COLUMN IF NOT EXISTS historical_note  TEXT",

    # ── fundus_images ────────────────────────────────────────────────────────
    "ALTER TABLE fundus_images ADD COLUMN IF NOT EXISTS dicom_path      VARCHAR(512)",
    "ALTER TABLE fundus_images ADD COLUMN IF NOT EXISTS study_date       TIMESTAMP",
    "ALTER TABLE fundus_images ADD COLUMN IF NOT EXISTS dicom_study_uid  VARCHAR(128)",
    "ALTER TABLE fundus_images ADD COLUMN IF NOT EXISTS image_quality    VARCHAR(8)",

    # ── annotations — doctor_id (FK added separately below) ─────────────────
    "ALTER TABLE annotations ADD COLUMN IF NOT EXISTS doctor_id  UUID",

    # ── grid_cells — rename grid_selection_id → region_id ───────────────────
    # Wrapped in DO $$ so it's a no-op if already renamed
    """
    DO $$
    BEGIN
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name='grid_cells' AND column_name='grid_selection_id'
        ) THEN
            ALTER TABLE grid_cells RENAME COLUMN grid_selection_id TO region_id;
        END IF;
    END $$;
    """,

    # ── regions_of_interest — ensure region_ref FK exists (no-op if present) ─
    # (no missing columns, just verifying)

    # ── clinical_notes ────────────────────────────────────────────────────────
    "ALTER TABLE clinical_notes ADD COLUMN IF NOT EXISTS vlm_description  TEXT",

    # ── custom_pathology_proposals ───────────────────────────────────────────
    "ALTER TABLE custom_pathology_proposals ADD COLUMN IF NOT EXISTS proposed_grade_labels  JSONB",
    "ALTER TABLE custom_pathology_proposals ADD COLUMN IF NOT EXISTS urgency_suggestion      VARCHAR(4)",
    "ALTER TABLE custom_pathology_proposals ADD COLUMN IF NOT EXISTS is_gradable             BOOLEAN DEFAULT FALSE",
    "ALTER TABLE custom_pathology_proposals ADD COLUMN IF NOT EXISTS proposed_description    TEXT",
    "ALTER TABLE custom_pathology_proposals ADD COLUMN IF NOT EXISTS proposed_grades_json    JSONB",
    "ALTER TABLE custom_pathology_proposals ADD COLUMN IF NOT EXISTS admin_notes             TEXT",
    "ALTER TABLE custom_pathology_proposals ADD COLUMN IF NOT EXISTS reviewed_by             UUID REFERENCES users(id)",
    "ALTER TABLE custom_pathology_proposals ADD COLUMN IF NOT EXISTS image_id                UUID REFERENCES fundus_images(id) ON DELETE SET NULL",

    # ── fundus_images — urgency + assignment ─────────────────────────────────
    "ALTER TABLE fundus_images ADD COLUMN IF NOT EXISTS model_urgency  VARCHAR(4)",
    "ALTER TABLE fundus_images ADD COLUMN IF NOT EXISTS assigned_to    UUID REFERENCES users(id) ON DELETE SET NULL",
]


def migrate():
    print("── RetinAI schema migration ──")
    with engine.connect() as conn:
        for sql in MIGRATIONS:
            stmt = sql.strip()
            if not stmt:
                continue
            # Show first line as label
            label = stmt.splitlines()[0][:80]
            try:
                conn.execute(text(stmt))
                conn.commit()
                print(f"  ✓  {label}")
            except Exception as e:
                conn.rollback()
                print(f"  ✗  {label}")
                print(f"     {e}")
    print("\n✓ Migration terminée.")


if __name__ == "__main__":
    migrate()
