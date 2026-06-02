import { useRef, useState, useCallback } from 'react'
import { useAuthStore } from './store'

/**
 * useLiveTranscription — French voice-to-text with two paths:
 *
 *  1. Primary: Browser Web Speech API (SpeechRecognition) with `fr-FR` locale.
 *     Real-time, zero-latency, free. Works in Chrome/Edge. Final + interim results.
 *
 *  2. Fallback (Firefox / no SpeechRecognition): MediaRecorder chunks →
 *     POST /api/transcribe/stream  (backend Whisper FR).
 *     Sends 3-second WebM chunks; each chunk is appended via onFinal.
 *
 * Bugs fixed vs previous version:
 *  • isRecordingRef (useRef) replaces React state in async callbacks —
 *    the old `recording` closure was always stale (false) so onend never restarted.
 *  • Authorization header added to Whisper fetch (was returning 401 silently).
 *  • Whisper chunks increased to 3000 ms to reduce partial-word splitting.
 */
export function useLiveTranscription({ onInterim, onFinal, lang = 'fr-FR' } = {}) {
  const [recording, setRecording] = useState(false)
  const [mode, setMode]           = useState(null)   // 'native' | 'whisper' | null

  // Mutable ref — safe to read inside async callbacks without stale closure
  const isRecordingRef   = useRef(false)
  const recognitionRef   = useRef(null)
  const mediaRecorderRef = useRef(null)
  const streamRef        = useRef(null)

  const token = useAuthStore.getState().token   // live read, not reactive

  const SR =
    typeof window !== 'undefined' &&
    (window.SpeechRecognition || window.webkitSpeechRecognition)

  const supported =
    !!SR || (typeof navigator !== 'undefined' && !!navigator.mediaDevices)

  // ── Native Web Speech API (Chrome / Edge) ──────────────────────────────
  const startNative = useCallback(() => {
    const recognition = new SR()
    recognition.lang          = lang
    recognition.continuous    = true
    recognition.interimResults = true

    recognition.onresult = (event) => {
      let interim = ''
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        const res = event.results[i]
        if (res.isFinal) {
          onFinal?.(res[0].transcript.trim())
        } else {
          interim += res[0].transcript
        }
      }
      if (interim) onInterim?.(interim)
    }

    recognition.onerror = (e) => {
      if (e.error === 'not-allowed') {
        console.error('Microphone permission denied')
        isRecordingRef.current = false
        setRecording(false)
        setMode(null)
      } else {
        console.warn('SpeechRecognition error:', e.error)
      }
    }

    recognition.onend = () => {
      // Auto-restart on silence timeout (Chrome stops after ~60s silence).
      // isRecordingRef is a ref so it always reflects current state — no stale closure.
      if (isRecordingRef.current && recognitionRef.current === recognition) {
        try { recognition.start() } catch (e) { console.warn('recognition restart:', e) }
      }
    }

    recognitionRef.current = recognition
    setMode('native')
    recognition.start()
  }, [SR, lang, onFinal, onInterim])

  // ── Whisper fallback (Firefox / no SpeechRecognition) ──────────────────
  const startWhisper = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    streamRef.current = stream

    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : 'audio/webm'

    const mr = new MediaRecorder(stream, { mimeType })
    mediaRecorderRef.current = mr
    setMode('whisper')

    mr.ondataavailable = async (e) => {
      if (!e.data || e.data.size === 0) return
      const currentToken = useAuthStore.getState().token
      const fd = new FormData()
      fd.append('chunk', e.data, 'chunk.webm')
      fd.append('lang', 'fr')
      try {
        const r = await fetch('/api/transcribe/stream', {
          method: 'POST',
          headers: currentToken ? { Authorization: `Bearer ${currentToken}` } : {},
          body: fd,
        })
        if (r.ok) {
          const j = await r.json()
          if (j.text?.trim()) onFinal?.(j.text.trim())
        }
      } catch (err) {
        console.warn('Whisper chunk failed:', err)
      }
    }

    mr.start(3000)   // 3-second chunks — better for Whisper accuracy
  }, [onFinal])

  // ── Public API ──────────────────────────────────────────────────────────
  const start = useCallback(async () => {
    isRecordingRef.current = true
    setRecording(true)
    try {
      if (SR) startNative()
      else await startWhisper()
    } catch (err) {
      console.error('Transcription start failed:', err)
      isRecordingRef.current = false
      setRecording(false)
      setMode(null)
    }
  }, [SR, startNative, startWhisper])

  const stop = useCallback(() => {
    isRecordingRef.current = false
    setRecording(false)
    setMode(null)

    if (recognitionRef.current) {
      try { recognitionRef.current.stop() } catch {}
      recognitionRef.current = null
    }
    if (mediaRecorderRef.current) {
      try { mediaRecorderRef.current.stop() } catch {}
      mediaRecorderRef.current = null
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
  }, [])

  return { recording, supported, mode, start, stop }
}
