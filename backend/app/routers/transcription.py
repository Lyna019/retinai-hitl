"""French voice-to-text — Whisper FR streaming + full-file endpoints."""
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..config import settings
from ..core.whisper_service import transcribe_chunk, transcribe_file
from ..database import get_db
from ..models import Annotation, ClinicalNote
from ..schemas import TranscriptionOut
from .auth import get_current_user

router = APIRouter(prefix="/transcribe", tags=["transcription"])


@router.post("/stream", response_model=TranscriptionOut)
async def transcribe_stream(
    chunk: UploadFile = File(...),
    lang: str = Form("fr"),
    _user=Depends(get_current_user),
):
    """
    Receives a 2-second audio chunk from the browser and returns the
    transcribed text. Used as a fallback when the browser's native
    SpeechRecognition is unavailable (e.g. Firefox without polyfill).
    """
    audio = await chunk.read()
    text = await transcribe_chunk(audio, lang=lang)
    return TranscriptionOut(text=text, language=lang)


@router.post("/audio", response_model=TranscriptionOut)
async def transcribe_audio(
    audio: UploadFile = File(...),
    annotation_id: str = Form(...),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Upload a complete voice recording for an annotation draft.
    Saves the audio file to disk, transcribes it with Whisper, and
    persists both audio_path and transcription_fr in the ClinicalNote.
    Returns the full transcription text.
    """
    annotation = (
        db.query(Annotation)
        .filter(Annotation.id == annotation_id, Annotation.doctor_id == user.id)
        .first()
    )
    if not annotation:
        raise HTTPException(404, "Annotation introuvable")

    audio_dir = Path(settings.audio_root)
    audio_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
    audio_path = audio_dir / f"{annotation_id}{suffix}"

    content = await audio.read()
    with open(audio_path, "wb") as f:
        f.write(content)

    text, duration = await transcribe_file(audio_path, lang=settings.whisper_language)

    note = db.query(ClinicalNote).filter(ClinicalNote.annotation_id == annotation_id).first()
    if note:
        note.audio_path      = str(audio_path)
        note.transcription_fr = text or note.transcription_fr
        note.duration_sec    = int(duration) if duration else note.duration_sec
    else:
        db.add(ClinicalNote(
            annotation_id=annotation_id,
            audio_path=str(audio_path),
            transcription_fr=text,
            duration_sec=int(duration) if duration else None,
        ))
    db.commit()

    return TranscriptionOut(
        text=text,
        language=settings.whisper_language,
        duration_sec=duration or None,
    )
