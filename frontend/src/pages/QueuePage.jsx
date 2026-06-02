import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Filter, ArrowUpDown } from 'lucide-react'
import { useAnnotationStore, useCatalogStore } from '../lib/store'

const URGENT_CODES = new Set(['OACR', 'ABACR', 'NOIAA', 'DR_MAC_OFF', 'GLAUC_AIGU'])

// Safe accessor — image.predictions may be null if model hasn't run yet
const topPrediction = (img) => img.predictions?.top_k?.[0] ?? null

export default function QueuePage() {
  const { images, openImage: storeOpenImage } = useAnnotationStore((s) => ({
    images: s.images,
    openImage: s.openImage,
  }))
  const { diseases } = useCatalogStore((s) => ({ diseases: s.diseases }))
  const nav = useNavigate()

  const [filter, setFilter] = useState('pending')
  const [sort, setSort] = useState('uncertainty')

  const diseaseByCode = useMemo(
    () => Object.fromEntries(diseases.map((d) => [d.code, d])),
    [diseases],
  )

  const filtered = useMemo(() => {
    let list = images
    if (filter === 'pending') list = list.filter((i) => i.status !== 'done')
    if (filter === 'done')    list = list.filter((i) => i.status === 'done')
    if (filter === 'urgent')  list = list.filter((i) => URGENT_CODES.has(topPrediction(i)?.disease_code))

    if (sort === 'uncertainty')
      list = [...list].sort((a, b) => (b.uncertainty ?? 0) - (a.uncertainty ?? 0))
    if (sort === 'date')
      list = [...list].sort((a, b) =>
        (a.capture_date || '').localeCompare(b.capture_date || ''),
      )
    return list
  }, [images, filter, sort])

  const stats = useMemo(() => ({
    total:   images.length,
    pending: images.filter((i) => i.status !== 'done').length,
    urgent:  images.filter((i) => URGENT_CODES.has(topPrediction(i)?.disease_code)).length,
    done:    images.filter((i) => i.status === 'done').length,
  }), [images])

  const openImage = (img) => {
    storeOpenImage(img.id)
    nav('/annotate')
  }

  return (
    <div className="h-full overflow-y-auto px-10 py-8">
      <div className="mx-auto max-w-[1200px]">
        <div className="flex items-start justify-between">
          <div>
            <div className="eyebrow">File d'annotation</div>
            <h1 className="display mt-1 text-[28px] font-medium leading-tight">
              Images à annoter
            </h1>
            <p className="mt-1 text-[13px] text-ink-secondary">
              Triées par incertitude décroissante · active learning
            </p>
          </div>
          <div className="grid grid-cols-4 gap-2">
            <Stat label="Total"   value={stats.total} />
            <Stat label="À faire" value={stats.pending} accent />
            <Stat label="Urgent"  value={stats.urgent} urgent />
            <Stat label="Fait"    value={stats.done} />
          </div>
        </div>

        {/* Filter & sort bar */}
        <div className="mt-6 flex items-center gap-2 border-b border-line pb-3">
          <div className="flex items-center gap-1 text-[11px] text-ink-tertiary">
            <Filter size={12} strokeWidth={2} />
            <span className="mono uppercase tracking-wider">filtre</span>
          </div>
          {[
            ['all',     'Tous'],
            ['pending', 'À faire'],
            ['urgent',  'Urgent'],
            ['done',    'Fait'],
          ].map(([k, l]) => (
            <button
              key={k}
              onClick={() => setFilter(k)}
              className={`chip rounded-md px-2.5 py-1 text-[11px] transition-colors ${
                filter === k ? 'bg-bg-elev2 text-ink-primary' : 'text-ink-secondary hover:text-ink-primary'
              }`}
            >
              {l}
            </button>
          ))}
          <div className="mx-3 h-4 w-px bg-line" />
          <div className="flex items-center gap-1 text-[11px] text-ink-tertiary">
            <ArrowUpDown size={12} strokeWidth={2} />
            <span className="mono uppercase tracking-wider">tri</span>
          </div>
          {[
            ['uncertainty', 'Incertitude'],
            ['date',        'Date'],
          ].map(([k, l]) => (
            <button
              key={k}
              onClick={() => setSort(k)}
              className={`chip rounded-md px-2.5 py-1 text-[11px] transition-colors ${
                sort === k ? 'bg-bg-elev2 text-ink-primary' : 'text-ink-secondary hover:text-ink-primary'
              }`}
            >
              {l}
            </button>
          ))}
        </div>

        {/* Image grid */}
        <div className="mt-4 grid grid-cols-2 gap-2 md:grid-cols-3 lg:grid-cols-4">
          {filtered.map((img) => {
            const top = topPrediction(img)
            const disease = top ? diseaseByCode[top.disease_code] : null
            const isUrgent = top ? URGENT_CODES.has(top.disease_code) : false
            return (
              <button
                key={img.id}
                onClick={() => openImage(img)}
                className={`group relative overflow-hidden rounded-xl border bg-bg-elev1 text-left transition-all hover:-translate-y-0.5 hover:border-accent/40 ${
                  isUrgent ? 'border-urgency-p1/40' : 'border-line'
                }`}
              >
                {/* Thumbnail */}
                <div className="relative aspect-square bg-black">
                  {img.file_url ? (
                    <img
                      src={img.file_url}
                      alt={img.patient_id}
                      className="absolute inset-0 h-full w-full object-cover"
                    />
                  ) : (
                    <div className="absolute inset-2 rounded-full fundus-mock" />
                  )}
                  {img.status === 'done' && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/70">
                      <span className="mono text-[10px] uppercase tracking-wider text-urgency-p4">
                        ✓ Annotée
                      </span>
                    </div>
                  )}
                  {img.locked_by && img.status !== 'done' && (
                    <div className="absolute bottom-2 left-2 flex items-center gap-1 rounded-full bg-black/60 px-2 py-0.5">
                      <span className="h-1 w-1 rounded-full bg-urgency-p3" />
                      <span className="mono text-[9px] text-urgency-p3">Verrouillée</span>
                    </div>
                  )}
                  {isUrgent && (
                    <div className="absolute right-2 top-2 flex items-center gap-1 rounded-full bg-urgency-p1/20 px-2 py-0.5">
                      <span className="h-1 w-1 rounded-full bg-urgency-p1 animate-pulse-soft" />
                      <span className="mono text-[9px] text-urgency-p1">P1</span>
                    </div>
                  )}
                </div>

                <div className="p-2.5">
                  <div className="flex items-baseline justify-between">
                    <span className="mono text-[12px]">
                      {img.patient_clinical_id || img.patient_id.slice(0, 8)}
                      <span className="text-ink-tertiary"> · {img.eye}</span>
                    </span>
                    <span className="mono text-[10px] text-ink-tertiary">
                      σ {(img.uncertainty ?? 0).toFixed(2)}
                    </span>
                  </div>
                  <div className="mt-1 truncate text-[11px] text-ink-secondary">
                    {disease?.name_fr || top?.disease_code || '—'}
                    {top && (
                      <span className="text-ink-tertiary"> · {Math.round((top.confidence ?? 0) * 100)}%</span>
                    )}
                  </div>
                </div>
              </button>
            )
          })}
          {filtered.length === 0 && (
            <div className="col-span-4 py-16 text-center text-[13px] text-ink-tertiary">
              {images.length === 0 ? 'Chargement…' : 'Aucune image dans ce filtre.'}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function Stat({ label, value, accent, urgent }) {
  return (
    <div className={`rounded-lg border px-3 py-2 ${
      urgent ? 'border-urgency-p1/30 bg-urgency-p1/5' :
      accent ? 'border-accent/30 bg-accent/5' :
      'border-line bg-bg-elev1'
    }`}>
      <div className="eyebrow">{label}</div>
      <div className={`display mt-0.5 text-[22px] font-medium leading-none ${
        urgent ? 'text-urgency-p1' : accent ? 'text-accent' : 'text-ink-primary'
      }`}>{value}</div>
    </div>
  )
}
