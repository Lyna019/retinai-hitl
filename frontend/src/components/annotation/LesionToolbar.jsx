import { useMemo } from 'react'
import { useAnnotationStore, useCatalogStore } from '../../lib/store'
import { LESION_VOCAB } from '../../lib/mockData'

export default function LesionToolbar() {
  const { active, setActiveLesion, annotation } = useAnnotationStore((s) => ({
    active: s.annotations[s.currentImageId]?.active_lesion || 'HEM',
    annotation: s.annotations[s.currentImageId] || { regions: [] },
    setActiveLesion: s.setActiveLesion,
  }))

  const catalogLesions = useCatalogStore((s) => s.lesions)

  const lesions = useMemo(() => {
    if (catalogLesions?.length) return catalogLesions
    return LESION_VOCAB
  }, [catalogLesions])

  // Total cells across all regions
  const totalCells = useMemo(
    () => (annotation.regions || []).reduce((sum, r) => sum + (r.cells?.length || 0), 0),
    [annotation.regions],
  )

  return (
    <section className="px-5 pt-4">
      <div className="flex items-center justify-between">
        <span className="eyebrow">Lésion · peindre</span>
        {totalCells > 0 && (
          <span className="mono text-[10px] text-ink-tertiary">{totalCells} cellules</span>
        )}
      </div>
      <div className="mt-2 flex flex-wrap gap-1">
        {lesions.map((l) => (
          <button
            key={l.code}
            onClick={() => setActiveLesion(l.code)}
            className={`chip flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] transition-all ${
              active === l.code
                ? 'border-accent/50 bg-accent/10 text-ink-primary'
                : 'border-line bg-bg-elev1 text-ink-secondary hover:border-line-strong hover:text-ink-primary'
            }`}
          >
            <span className="h-2 w-2 rounded-sm" style={{ background: l.color }} />
            {l.name_fr}
          </button>
        ))}
      </div>
    </section>
  )
}
