import { useState } from 'react'
import { Check, CircleDot, X, ZoomIn } from 'lucide-react'
import { useAnnotationStore } from '../../lib/store'

const OPTIONS = [
  { key: 'correct', label: 'Correct', icon: Check,      color: 'accent' },
  { key: 'partial', label: 'Partiel', icon: CircleDot,  color: 'p2'    },
  { key: 'wrong',   label: 'Faux',    icon: X,          color: 'p1'    },
]

export default function GradCamValidation() {
  const [expanded, setExpanded] = useState(false)

  const { verdict, setVerdict, gradcamUrl } = useAnnotationStore((s) => ({
    verdict:    s.annotations[s.currentImageId]?.gradcam_verdict,
    setVerdict: s.setGradCamVerdict,
    gradcamUrl: s.images.find((i) => i.id === s.currentImageId)?.predictions?.gradcam_url ?? null,
  }))

  return (
    <section className="px-5 pt-4">
      <span className="eyebrow">Validation Grad-CAM</span>

      {gradcamUrl ? (
        <div className="mt-2 overflow-hidden rounded-xl border border-line bg-bg-base">
          {/* Heatmap image */}
          <div
            className="group relative cursor-zoom-in"
            onClick={() => setExpanded(true)}
          >
            <img
              src={gradcamUrl}
              alt="Grad-CAM heatmap"
              className="w-full object-cover"
              style={{ maxHeight: expanded ? 'none' : '160px', objectFit: 'cover' }}
            />
            <div className="absolute inset-0 flex items-center justify-center bg-black/0 transition-colors group-hover:bg-black/20">
              <ZoomIn
                size={20}
                strokeWidth={1.5}
                className="text-white opacity-0 drop-shadow transition-opacity group-hover:opacity-100"
              />
            </div>
          </div>

          {/* Fullscreen overlay */}
          {expanded && (
            <div
              className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
              onClick={() => setExpanded(false)}
            >
              <img
                src={gradcamUrl}
                alt="Grad-CAM heatmap (agrandi)"
                className="max-h-full max-w-full rounded-xl object-contain shadow-2xl"
                onClick={(e) => e.stopPropagation()}
              />
            </div>
          )}
        </div>
      ) : (
        <p className="mt-2 rounded-lg border border-line bg-bg-elev1 px-3 py-2 text-[11px] text-ink-tertiary">
          Aucune carte Grad-CAM disponible pour cette image.
        </p>
      )}

      {/* Verdict buttons */}
      <div className="mt-2 grid grid-cols-3 gap-1.5">
        {OPTIONS.map(({ key, label, icon: Icon, color }) => {
          const active = verdict === key
          const colorClass = {
            accent: active ? 'border-accent bg-accent/15 text-accent' : '',
            p1:     active ? 'border-urgency-p1/60 bg-urgency-p1/15 text-urgency-p1' : '',
            p2:     active ? 'border-urgency-p2/60 bg-urgency-p2/15 text-urgency-p2' : '',
          }[color]
          return (
            <button
              key={key}
              onClick={() => setVerdict(active ? null : key)}
              className={`chip flex items-center justify-center gap-1.5 rounded-lg border py-2 text-[12px] transition-colors ${
                active
                  ? colorClass
                  : 'border-line bg-bg-elev1 text-ink-secondary hover:border-line-strong hover:text-ink-primary'
              }`}
            >
              <Icon size={12} strokeWidth={2.5} />
              {label}
            </button>
          )
        })}
      </div>
    </section>
  )
}
