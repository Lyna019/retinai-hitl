import { useMemo, useState, useCallback } from 'react'
import { RefreshCw } from 'lucide-react'
import { useCatalogStore, useAuthStore, useAnnotationStore } from '../../lib/store'
import { DISEASES } from '../../lib/mockData'
import { api } from '../../lib/api'

export default function PredictionsPanel({ predictions, imageId }) {
  const catalogDiseases = useCatalogStore((s) => s.diseases)
  const token = useAuthStore((s) => s.token)
  const [rerunning, setRerunning] = useState(false)

  const handleRerun = useCallback(async () => {
    if (!imageId || !token || rerunning) return
    setRerunning(true)
    try {
      const preds = await api.runPrediction(imageId, token)
      useAnnotationStore.setState((st) => ({
        images: st.images.map((img) =>
          img.id === imageId ? { ...img, predictions: preds } : img,
        ),
      }))
    } catch (e) {
      console.error('Re-run prediction failed', e)
    } finally {
      setRerunning(false)
    }
  }, [imageId, token, rerunning])

  const diseaseByCode = useMemo(() => {
    const fallback = Object.fromEntries(DISEASES.map((d) => [d.code, d]))
    const fromCatalog = Object.fromEntries(catalogDiseases.map((d) => [d.code, d]))
    return { ...fallback, ...fromCatalog }
  }, [catalogDiseases])

  if (!predictions?.top_k?.length) {
    return (
      <section className="px-5 pt-5">
        <div className="eyebrow mb-3">Prédictions · Modèle</div>
        <div className="rounded-xl border border-line bg-bg-elev1 p-3 text-center text-[12px] text-ink-tertiary">
          Aucune prédiction disponible.
        </div>
      </section>
    )
  }

  const top = predictions.top_k.slice(0, 3)
  const primary = top[0]
  const others = top.slice(1)

  return (
    <section className="px-5 pt-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="eyebrow" title="Scores indépendants par pathologie (multi-label sigmoid) — ne s'additionnent pas à 100%">Prédictions · Modèle</div>
          <button
            onClick={handleRerun}
            disabled={rerunning}
            title="Relancer le modèle"
            className="flex h-5 w-5 items-center justify-center rounded text-ink-tertiary hover:bg-bg-elev2 hover:text-ink-primary transition-colors disabled:opacity-40"
          >
            <RefreshCw size={11} strokeWidth={2} className={rerunning ? 'animate-spin' : ''} />
          </button>
        </div>
        <span className="mono text-[10px] text-ink-tertiary">{predictions.model_version}</span>
      </div>

      {/* Primary prediction */}
      <div className="mt-3 rounded-xl border border-accent/30 bg-gradient-to-br from-accent/10 to-transparent p-3">
        <div className="flex items-baseline justify-between">
          <div className="display text-[15px] text-ink-primary">
            {diseaseByCode[primary.disease_code]?.name_fr || primary.disease_code}
            {primary.grade && (
              <span className="ml-1.5 mono text-[11px] text-ink-secondary">
                · grade {primary.grade}
              </span>
            )}
          </div>
          <div className="mono text-[14px] text-accent">
            {Math.round((primary.confidence ?? 0) * 100)}
            <span className="text-[10px] text-ink-tertiary">%</span>
          </div>
        </div>
        <div className="mt-2 h-[3px] overflow-hidden rounded-full bg-accent/15">
          <div
            className="h-full rounded-full bg-accent transition-all"
            style={{ width: `${(primary.confidence ?? 0) * 100}%` }}
          />
        </div>
      </div>

      {/* Alternatives */}
      {others.length > 0 && (
        <div className="mt-2 grid grid-cols-2 gap-2">
          {others.map((pred, i) => (
            <div key={i} className="rounded-lg border border-line bg-bg-elev1 px-2.5 py-1.5">
              <div className="flex items-baseline justify-between">
                <span className="text-[12px] text-ink-secondary">
                  {diseaseByCode[pred.disease_code]?.name_fr || pred.disease_code}
                </span>
                <span className="mono text-[11px] text-ink-tertiary">
                  {Math.round((pred.confidence ?? 0) * 100)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  )
}
