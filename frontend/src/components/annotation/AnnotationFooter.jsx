import { useState } from 'react'
import { motion } from 'framer-motion'
import { CornerDownLeft, Save, Loader2 } from 'lucide-react'
import { useAnnotationStore, useAuthStore } from '../../lib/store'
import { api } from '../../lib/api'

export default function AnnotationFooter() {
  const token = useAuthStore((s) => s.token)
  const { submit, canSubmit, currentImageId, annotation } = useAnnotationStore((s) => ({
    submit: s.submitAnnotation,
    canSubmit: (s.annotations[s.currentImageId]?.disease_labels || []).length > 0,
    currentImageId: s.currentImageId,
    annotation: s.annotations[s.currentImageId] || {},
  }))

  const [justSubmitted, setJustSubmitted] = useState(false)
  const [draftSaving, setDraftSaving] = useState(false)
  const [draftSaved, setDraftSaved] = useState(false)

  const handleSubmit = () => {
    if (!canSubmit) return
    setJustSubmitted(true)
    setTimeout(() => {
      submit()
      setJustSubmitted(false)
    }, 300)
  }

  const handleDraft = async () => {
    if (!currentImageId) return
    setDraftSaving(true)
    try {
      await api.saveDraft(
        currentImageId,
        {
          disease_labels: annotation.disease_labels || [],
          regions: annotation.regions || [],
          gradcam_verdict: annotation.gradcam_verdict || null,
          notes_text: annotation.notes_text || '',
        },
        token,
      )
      setDraftSaved(true)
      setTimeout(() => setDraftSaved(false), 2000)
    } catch (e) {
      console.error('saveDraft', e)
    } finally {
      setDraftSaving(false)
    }
  }

  return (
    <section className="sticky bottom-0 mt-4 border-t border-line bg-bg-elev1/80 px-5 py-3 backdrop-blur-xl">
      <div className="grid grid-cols-2 gap-2">
        <button
          onClick={handleDraft}
          disabled={draftSaving}
          className="btn flex items-center justify-center gap-1.5 rounded-lg border border-line bg-bg-base py-2.5 text-[12px] text-ink-secondary hover:border-line-strong hover:text-ink-primary disabled:opacity-60"
        >
          {draftSaving ? (
            <Loader2 size={12} className="animate-spin" />
          ) : draftSaved ? (
            <span className="text-urgency-p4">✓</span>
          ) : (
            <Save size={12} strokeWidth={2} />
          )}
          {draftSaved ? 'Sauvegardé' : 'Brouillon'}
        </button>

        <motion.button
          onClick={handleSubmit}
          disabled={!canSubmit}
          whileTap={{ scale: 0.97 }}
          className="btn flex items-center justify-center gap-1.5 rounded-lg bg-accent py-2.5 text-[12px] font-medium text-bg-base shadow-[0_0_24px_-4px_rgba(20,227,202,0.45)] transition-all disabled:bg-bg-elev3 disabled:text-ink-tertiary disabled:shadow-none"
        >
          {justSubmitted ? (
            <motion.span
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className="mono"
            >
              ✓ enregistré
            </motion.span>
          ) : (
            <>
              Soumettre
              <CornerDownLeft size={12} strokeWidth={2.5} />
            </>
          )}
        </motion.button>
      </div>
      <div className="mt-1.5 flex items-center justify-center mono text-[9px] text-ink-tertiary">
        {canSubmit
          ? 'Appuyez sur ↵ pour soumettre'
          : 'Sélectionnez au moins une pathologie pour soumettre'}
      </div>
    </section>
  )
}
