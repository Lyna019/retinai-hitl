/**
 * RegionSelector — add anatomical regions before painting the grid.
 * Each region gets its own grid cells for segmentation/explainability.
 */
import { useState, useMemo } from 'react'
import { Plus, X, MapPin, ChevronRight } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { useAnnotationStore, useAuthStore, useCatalogStore } from '../../lib/store'
import { api } from '../../lib/api'

export default function RegionSelector() {
  const token = useAuthStore((s) => s.token)
  const regions = useCatalogStore((s) => s.regions)
  const fetchAll = useCatalogStore((s) => s.fetchAll)

  const { annotation, addRegion, removeRegion, setActiveRegion } = useAnnotationStore((s) => ({
    annotation: s.annotations[s.currentImageId] || { regions: [], active_region_idx: 0 },
    addRegion: s.addRegion,
    removeRegion: s.removeRegion,
    setActiveRegion: s.setActiveRegion,
  }))

  const [open, setOpen] = useState(false)
  const [customName, setCustomName] = useState('')
  const [adding, setAdding] = useState(false)

  const addFromCatalog = (region) => {
    addRegion({ anatomical_region_id: region.id, custom_region_name: null })
    setOpen(false)
  }

  const addCustom = async () => {
    if (!customName.trim()) return
    setAdding(true)
    try {
      const r = await api.createRegion(customName.trim(), token)
      await fetchAll(token)
      addRegion({ anatomical_region_id: r.id, custom_region_name: customName.trim() })
      setCustomName('')
      setOpen(false)
    } finally {
      setAdding(false)
    }
  }

  const regionByIdMap = useMemo(
    () => Object.fromEntries(regions.map((r) => [r.id, r])),
    [regions],
  )

  const regionName = (r) =>
    r.custom_region_name || regionByIdMap[r.anatomical_region_id]?.name_fr || 'Région'

  return (
    <section className="px-5 pt-4">
      <div className="flex items-center justify-between">
        <div className="eyebrow flex items-center gap-1.5">
          <MapPin size={11} />
          Régions d'intérêt
        </div>
        <button
          onClick={() => setOpen(true)}
          className="btn flex items-center gap-1 rounded border border-dashed border-line px-2 py-0.5 text-[11px] text-ink-tertiary hover:text-ink-primary hover:border-line-strong"
        >
          <Plus size={11} />
          Ajouter
        </button>
      </div>

      {/* Region tabs */}
      {annotation.regions.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {annotation.regions.map((r, i) => (
            <button
              key={i}
              onClick={() => setActiveRegion(i)}
              className={`group flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-[11px] transition-colors ${
                i === annotation.active_region_idx
                  ? 'border-accent/60 bg-accent/10 text-accent'
                  : 'border-line bg-bg-elev1 text-ink-secondary hover:border-line-strong hover:text-ink-primary'
              }`}
            >
              <span className="font-medium">{i + 1}.</span>
              <span>{regionName(r)}</span>
              {r.cells?.length > 0 && (
                <span className="mono text-[9px] opacity-70">({r.cells.length})</span>
              )}
              <span
                onClick={(e) => { e.stopPropagation(); removeRegion(i) }}
                className="opacity-0 group-hover:opacity-100 transition-opacity text-ink-tertiary hover:text-urgency-p1 ml-0.5"
              >
                <X size={10} />
              </span>
            </button>
          ))}
        </div>
      )}

      {annotation.regions.length === 0 && (
        <p className="mt-2 text-[11px] text-ink-tertiary italic">
          Sélectionnez une région avant de peindre la grille.
        </p>
      )}

      {/* Add region modal */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
            onClick={() => setOpen(false)}
          >
            <motion.div
              initial={{ y: 8, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: 8, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="w-[460px] max-w-[92vw] rounded-xl border border-line bg-bg-elev1 shadow-2xl"
            >
              <div className="border-b border-line px-5 py-3">
                <div className="display text-[14px]">Choisir une région anatomique</div>
                <div className="mt-0.5 text-[11px] text-ink-tertiary">
                  Sélectionnez une région prédéfinie ou tapez un nom personnalisé.
                </div>
              </div>

              <div className="max-h-[50vh] overflow-y-auto px-3 py-2">
                {regions.filter((r) => !r.is_custom).map((r) => (
                  <button
                    key={r.id}
                    onClick={() => addFromCatalog(r)}
                    className="flex w-full items-center justify-between rounded-md px-3 py-2 text-left hover:bg-bg-elev2 text-[12px]"
                  >
                    <span>{r.name_fr}</span>
                    <ChevronRight size={12} className="text-ink-tertiary" />
                  </button>
                ))}
                {regions.filter((r) => r.is_custom).length > 0 && (
                  <>
                    <div className="eyebrow px-3 py-2">Personnalisées</div>
                    {regions.filter((r) => r.is_custom).map((r) => (
                      <button
                        key={r.id}
                        onClick={() => addFromCatalog(r)}
                        className="flex w-full items-center justify-between rounded-md px-3 py-2 text-left hover:bg-bg-elev2 text-[12px]"
                      >
                        <span>{r.name_fr}</span>
                        <ChevronRight size={12} className="text-ink-tertiary" />
                      </button>
                    ))}
                  </>
                )}
              </div>

              {/* Custom name input */}
              <div className="border-t border-line px-4 py-3">
                <div className="text-[11px] text-ink-tertiary mb-1.5">
                  Région non listée ? Tapez son nom :
                </div>
                <div className="flex gap-2">
                  <input
                    value={customName}
                    onChange={(e) => setCustomName(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && addCustom()}
                    placeholder="Ex: Néovaisseaux supra-temporaux"
                    className="flex-1 rounded border border-line bg-bg-base px-3 py-1.5 text-[12px] outline-none focus:border-accent/60"
                  />
                  <button
                    onClick={addCustom}
                    disabled={!customName.trim() || adding}
                    className="btn rounded bg-accent px-3 py-1.5 text-[12px] font-medium text-bg-base disabled:opacity-40"
                  >
                    {adding ? '…' : 'Ajouter'}
                  </button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  )
}
