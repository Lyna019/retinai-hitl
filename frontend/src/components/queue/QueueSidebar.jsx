import { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useAnnotationStore } from '../../lib/store'

const FILTERS = [
  { key: 'all', label: 'Tous' },
  { key: 'pending', label: 'À faire' },
  { key: 'urgent', label: 'Urgent' },
  { key: 'done', label: 'Fait' },
]

// Predict urgency from top model prediction for the queue preview dots
const previewUrgency = (img) => {
  // Use model_urgency from backend (pre-computed from model predictions)
  if (img.model_urgency) return img.model_urgency.toLowerCase()
  // Fallback to top_k inspection for legacy images
  const top = img.predictions?.top_k?.[0]
  if (!top) return null
  if (['OACR', 'ABACR', 'NOIAA', 'CRAO', 'BRAO', 'AION'].includes(top.disease_code)) return 'p1'
  if (['DR_MAC_OFF', 'RETDET', 'GLAUC_AIGU', 'RTR', 'GRT'].includes(top.disease_code)) return 'p2'
  if (['UVEITE', 'VS', 'CRVO', 'OVCR', 'BRVO'].includes(top.disease_code)) return 'p3'
  if (top.disease_code === 'DR' && top.grade === '4') return 'p3'
  return 'p4'
}

const URGENCY_COLORS = {
  p1: 'bg-urgency-p1',
  p2: 'bg-urgency-p2',
  p3: 'bg-urgency-p3',
  p4: 'bg-urgency-p4',
}

export default function QueueSidebar() {
  const nav = useNavigate()
  const { images, currentImageId, queueFilter, openImage, setQueueFilter } = useAnnotationStore(
    (s) => ({
      images: s.images,
      currentImageId: s.currentImageId,
      queueFilter: s.queueFilter,
      openImage: s.openImage,
      setQueueFilter: s.setQueueFilter,
    }),
  )

  const handleOpen = (id) => {
    openImage(id)
    nav('/annotate')
  }

  const filtered = useMemo(() => {
    const u = (img) => previewUrgency(img)
    switch (queueFilter) {
      case 'pending':
        return images.filter((i) => i.status === 'pending' || i.status === 'in_progress')
      case 'urgent':
        return images.filter((i) => ['p1', 'p2', 'p3'].includes(u(i)))
      case 'done':
        return images.filter((i) => i.status === 'done')
      default:
        return images
    }
  }, [images, queueFilter])

  const remaining = images.filter((i) => i.status !== 'done').length

  return (
    <aside className="flex h-full w-[272px] flex-shrink-0 flex-col border-r border-line bg-bg-elev1/40">
      {/* Header */}
      <div className="flex items-end justify-between px-4 pb-2 pt-4">
        <div>
          <div className="eyebrow">File d'attente</div>
          <div className="mt-0.5 flex items-baseline gap-1.5">
            <span className="display text-lg font-medium">{remaining}</span>
            <span className="text-[11px] text-ink-secondary">restantes</span>
          </div>
        </div>
        <span className="mono text-[10px] text-ink-tertiary">
          tri · incertitude ▼
        </span>
      </div>

      {/* Filter tabs */}
      <div className="mx-4 mb-3 flex gap-1 rounded-lg border border-line bg-bg-base p-1">
        {FILTERS.map((f) => (
          <button
            key={f.key}
            onClick={() => setQueueFilter(f.key)}
            className={`flex-1 rounded-md px-2 py-1 text-[11px] transition-colors ${
              queueFilter === f.key
                ? 'bg-bg-elev2 text-ink-primary'
                : 'text-ink-secondary hover:text-ink-primary'
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto px-3 pb-4">
        <div className="space-y-1.5">
          {filtered.map((img) => {
            const selected = img.id === currentImageId
            const urg = previewUrgency(img)
            return (
              <motion.button
                key={img.id}
                layoutId={img.id}
                onClick={() => handleOpen(img.id)}
                whileHover={{ x: 2 }}
                className={`group relative flex w-full items-center gap-2.5 rounded-lg border p-2 text-left transition-all ${
                  selected
                    ? 'border-accent/40 bg-accent/5 shadow-[inset_0_0_0_0.5px_rgba(20,227,202,0.25)]'
                    : 'border-line bg-bg-elev1 hover:border-line-strong hover:bg-bg-elev2'
                }`}
              >
                {/* Thumbnail */}
                <div className="relative h-11 w-11 flex-shrink-0 overflow-hidden rounded-md border border-line bg-black">
                  <div className="absolute inset-[3px] rounded-full fundus-mock" />
                  {img.status === 'done' && (
                    <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#30D158" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12" />
                      </svg>
                    </div>
                  )}
                </div>

                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between">
                    <span className="mono text-[12px] text-ink-primary">
                      {img.patient_clinical_id || img.patient_id.slice(0, 8)}
                      <span className="text-ink-tertiary"> · {img.eye}</span>
                    </span>
                    {urg && urg !== 'p4' && (
                      <span className={`h-1.5 w-1.5 rounded-full ${URGENCY_COLORS[urg]}`} />
                    )}
                  </div>
                  <div className="mt-0.5 flex items-center justify-between">
                    <span className="mono text-[10px] text-ink-tertiary">
                      incert. {(img.uncertainty ?? 0).toFixed(2)}
                    </span>
                    <span className="mono text-[10px] uppercase text-ink-tertiary">
                      {img.status === 'in_progress' ? 'en cours' : img.status === 'done' ? 'fait' : ''}
                    </span>
                  </div>
                </div>

                {selected && (
                  <div className="absolute inset-y-1 left-0 w-[2px] rounded-r-full bg-accent" />
                )}
              </motion.button>
            )
          })}
          {filtered.length === 0 && (
            <div className="py-10 text-center text-[12px] text-ink-tertiary">
              Aucune image dans ce filtre.
            </div>
          )}
        </div>
      </div>
    </aside>
  )
}
