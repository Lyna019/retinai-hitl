"""
Hospital write-back — writes the final annotation note back to
CONSULTATION.DIAGNOSIS in the hospital's clinical DB.

Configure in .env:
    HOSPITAL_DB_TYPE=oracle       # oracle | mssql | postgresql | mysql
    HOSPITAL_DB_URL=oracle+cx_oracle://user:pw@host:1521/ORCL

The function formats the annotation as a structured clinical note string
and updates the row matching the patient's clinical_id + study date.

This is a stub: the SQL query assumes a CONSULTATION table with:
  - A patient identifier column (configurable via HOSPITAL_PATIENT_COL)
  - A diagnosis text column: DIAGNOSIS
  - A study/consultation date column (configurable via HOSPITAL_DATE_COL)

Uncomment and adapt the SQL once you have the exact schema.
"""
import logging
from datetime import datetime
from sqlalchemy import create_engine, text

from ..config import settings

log = logging.getLogger("retinai.writeback")


def _format_note(
    clinical_id: str,
    eye: str,
    disease_labels: list[dict],   # [{code, grade, name_fr}]
    urgency: str | None,
    notes_text: str | None,
    annotated_by: str,
    annotation_date: datetime,
) -> str:
    lines = [
        f"[RetinAI HITL — {annotation_date.strftime('%Y-%m-%d %H:%M')}]",
        f"Patient : {clinical_id}  |  Œil : {eye}  |  Médecin : {annotated_by}",
    ]
    if urgency:
        lines.append(f"Urgence : {urgency}")
    if disease_labels:
        lines.append("Pathologies détectées :")
        for d in disease_labels:
            grade = f" (Grade {d['grade']})" if d.get("grade") else ""
            lines.append(f"  - {d['name_fr']}{grade}")
    if notes_text:
        lines.append(f"Notes cliniques : {notes_text}")
    return "\n".join(lines)


def write_back_diagnosis(
    clinical_id: str,
    eye: str,
    disease_labels: list[dict],
    urgency: str | None,
    notes_text: str | None,
    annotated_by: str,
    annotation_date: datetime,
    study_date: datetime | None = None,
) -> bool:
    """
    Write annotation result to hospital DB.
    Returns True on success, False if write-back is not configured or fails.
    """
    if not settings.hospital_db_url:
        log.info("HOSPITAL_DB_URL not set — write-back skipped")
        return False

    note = _format_note(
        clinical_id, eye, disease_labels, urgency, notes_text, annotated_by, annotation_date
    )

    try:
        engine = create_engine(settings.hospital_db_url, pool_pre_ping=True)
        with engine.connect() as conn:
            # ── Adapt the WHERE clause to match your CONSULTATION table schema ──
            # Common patterns:
            #   Oracle/MSSQL:   WHERE PATIENT_ID = :pid AND STUDY_DATE = :sdate
            #   Some systems:   WHERE CONSULTATION_NUM = :pid
            sql = text("""
                UPDATE CONSULTATION
                   SET DIAGNOSIS = :note
                 WHERE PATIENT_ID = :pid
            """ + (" AND TRUNC(STUDY_DATE) = TRUNC(:sdate)" if study_date else ""))

            params: dict = {"note": note, "pid": clinical_id}
            if study_date:
                params["sdate"] = study_date

            result = conn.execute(sql, params)
            conn.commit()
            log.info("Write-back OK for patient %s — %d row(s) updated", clinical_id, result.rowcount)
            return True

    except Exception as exc:
        log.error("Write-back FAILED for patient %s: %s", clinical_id, exc)
        return False
