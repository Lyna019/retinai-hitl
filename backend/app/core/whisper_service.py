"""
French voice-to-text transcription service.

CURRENT STATE: Stub. Accepts audio chunks and returns an empty transcript
when Whisper is not installed. The frontend falls back to the browser's
native `SpeechRecognition` API in the meantime — so live transcription
works for the doctor right now, even without this backend running.

TO ENABLE REAL WHISPER:
  1. In backend/requirements.txt, uncomment `openai-whisper` and `torch`.
  2. Rebuild the backend image (GPU strongly recommended).
  3. The `transcribe_chunk()` function below auto-detects the Whisper
     install and uses it. No other code changes required.
"""
import logging
import os
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

_whisper_model = None


def _load_whisper():
    """Lazy-load the Whisper model on first use."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    try:
        import whisper  # type: ignore
        from .. import config
        log.info("Loading Whisper model: %s", config.settings.whisper_model)
        _whisper_model = whisper.load_model(config.settings.whisper_model)
        return _whisper_model
    except ImportError:
        log.warning("openai-whisper not installed — transcription is stubbed.")
        return None


async def transcribe_chunk(audio_bytes: bytes, lang: str = "fr") -> str:
    """
    Transcribe a single audio chunk (typically a 2-second WebM blob)
    and return the French transcript text. Returns "" if Whisper is
    unavailable — callers should rely on the browser fallback.
    """
    model = _load_whisper()
    if model is None:
        return ""

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path, language=lang, fp16=False)
        return result.get("text", "").strip()
    finally:
        os.unlink(tmp_path)


async def transcribe_file(file_path: Path, lang: str = "fr") -> tuple[str, float]:
    """
    Transcribe a full audio file. Returns (text, duration_sec).
    Used when saving the final audio alongside the annotation.
    """
    model = _load_whisper()
    if model is None:
        return "", 0.0

    result = model.transcribe(str(file_path), language=lang, fp16=False)
    return result.get("text", "").strip(), float(result.get("duration", 0.0))
