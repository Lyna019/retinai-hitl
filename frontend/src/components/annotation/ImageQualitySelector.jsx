import { Check, Minus, X } from 'lucide-react'
import { useAnnotationStore } from '../../lib/store'

const OPTIONS = [
  { key: 'good', label: 'Bonne',  icon: Check, color: 'accent' },
  { key: 'fair', label: 'Moyenne', icon: Minus, color: 'p3' },
  { key: 'poor', label: 'Mauvaise', icon: X,   color: 'p1' },
]

export default function ImageQualitySelector({ imageId, quality }) {
  const setImageQuality = useAnnotationStore((s) => s.setImageQuality)

  return (
    <section className="px-5 pt-4">
      <span className="eyebrow">Qualité de l&apos;image</span>
      <div className="mt-2 grid grid-cols-3 gap-1.5">
        {OPTIONS.map(({ key, label, icon: Icon, color }) => {
          const active = quality === key
          const colorClass = {
            accent: active ? 'border-accent bg-accent/15 text-accent' : '',
            p1:     active ? 'border-urgency-p1/60 bg-urgency-p1/15 text-urgency-p1' : '',
            p3:     active ? 'border-urgency-p3/60 bg-urgency-p3/15 text-urgency-p3' : '',
          }[color]
          return (
            <button
              key={key}
              onClick={() => setImageQuality(imageId, active ? null : key)}
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
