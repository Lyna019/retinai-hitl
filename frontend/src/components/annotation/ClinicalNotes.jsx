import { useState, useEffect } from 'react'
import { Mic, Square, Sparkles, Loader2 } from 'lucide-react'
import { useAnnotationStore, useAuthStore } from '../../lib/store'
import { useLiveTranscription } from '../../lib/transcription'
import { api } from '../../lib/api'

export default function ClinicalNotes({ imageId }) {
  const token = useAuthStore((s) => s.token)
  const { text, setText, appendText } = useAnnotationStore((s) => ({
    text: s.annotations[s.currentImageId]?.notes_text || '',
    setText: s.setNotesText,
    appendText: s.appendNotesText,
  }))

  const [interim, setInterim] = useState('')
  const [vlmLoading, setVlmLoading] = useState(false)
  const [vlmError, setVlmError] = useState(null)

  const { recording, supported, mode, start, stop } = useLiveTranscription({
    onInterim: (t) => setInterim(t),
    onFinal: (t) => {
      appendText(t)
      setInterim('')
    },
  })

  // Commit any pending interim text before clearing it on stop
  const handleStop = () => {
    if (interim.trim()) {
      appendText(interim.trim())
    }
    setInterim('')
    stop()
  }

  useEffect(() => {
    if (!recording) setInterim('')
  }, [recording])

  const runVLM = async () => {
    if (!imageId) return
    setVlmLoading(true)
    setVlmError(null)
    try {
      const result = await api.describeImage(imageId, token)
      if (result.description && !result.description.startsWith('[')) {
        const prefix = text ? '\n\n' : ''
        appendText(prefix + result.description)
      } else {
        // Error or placeholder — show inline, never write to notes
        setVlmError(result.description || 'Service VLM indisponible')
      }
    } catch (e) {
      setVlmError('Service VLM indisponible')
    } finally {
      setVlmLoading(false)
    }
  }

  return (
    <section className="px-5 pt-4">
      <div className="flex items-center justify-between">
        <span className="eyebrow">Notes additionnelles · optionnel</span>
        <div className="flex items-center gap-2">
          {mode && (
            <span className="mono text-[9px] uppercase tracking-wider text-ink-tertiary">
              {mode === 'native' ? 'navigateur' : 'whisper fr'}
            </span>
          )}
        </div>
      </div>

      <div className="relative mt-2 rounded-lg border border-line bg-bg-elev1 transition-colors focus-within:border-accent/40">
        <textarea
          value={text + (interim ? (text && !text.endsWith(' ') ? ' ' : '') + interim : '')}
          onChange={(e) => setText(e.target.value)}
          placeholder="Historique, antécédents, observations cliniques…"
          rows={5}
          className="block w-full resize-none bg-transparent px-3 py-2.5 pr-12 text-[12px] leading-relaxed outline-none placeholder:text-ink-tertiary"
        />

        {interim && (
          <div className="pointer-events-none absolute left-3 bottom-2 flex items-center gap-1 rounded-full bg-accent/15 px-1.5 py-0.5">
            <span className="h-1 w-1 rounded-full bg-accent animate-pulse-soft" />
            <span className="mono text-[9px] uppercase tracking-wider text-accent">
              dictée en cours
            </span>
          </div>
        )}

        {/* Mic button */}
        <button
          onClick={() => (recording ? handleStop() : start())}
          disabled={!supported}
          title={recording ? 'Arrêter la dictée' : 'Dicter (fr-FR)'}
          className={`btn absolute right-2 top-2 flex h-8 w-8 items-center justify-center rounded-md border transition-colors ${
            recording
              ? 'border-urgency-p1/60 bg-urgency-p1/15 text-urgency-p1'
              : 'border-line bg-bg-base text-ink-secondary hover:border-line-strong hover:text-ink-primary'
          } disabled:opacity-40`}
        >
          {recording
            ? <Square size={12} strokeWidth={3} fill="currentColor" />
            : <Mic size={13} strokeWidth={2} />
          }
        </button>
      </div>

      {/* VLM button */}
      <button
        onClick={runVLM}
        disabled={vlmLoading || !imageId}
        className="btn mt-2 flex w-full items-center justify-center gap-1.5 rounded-lg border border-line bg-bg-elev1 px-3 py-2 text-[12px] text-ink-secondary hover:border-accent/40 hover:text-accent transition-colors disabled:opacity-40"
        title="Générer une description clinique automatique (Qwen2-VL)"
      >
        {vlmLoading
          ? <Loader2 size={12} className="animate-spin" />
          : <Sparkles size={12} />
        }
        {vlmLoading ? 'Génération en cours…' : 'Description VLM automatique'}
      </button>

      {vlmError && (
        <p className="mt-1 mono text-[9px] text-urgency-p1 leading-tight">
          ⚠ {vlmError.replace(/^\[|\]$/g, '')}
        </p>
      )}

      <div className="mt-1.5 flex items-center justify-between mono text-[9px] text-ink-tertiary">
        <span>{text.length} caractères</span>
        <span>{supported ? 'Micro pour dicter en français' : 'Dictée non disponible'}</span>
      </div>
    </section>
  )
}
