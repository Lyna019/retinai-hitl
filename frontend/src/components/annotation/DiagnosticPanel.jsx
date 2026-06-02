import { useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, Plus, X, Check, Loader2 } from 'lucide-react'
import { useAnnotationStore, useAuthStore, useCatalogStore } from '../../lib/store'
import { api } from '../../lib/api'

export default function DiagnosticPanel() {
  const { annotation, toggleDisease, setDiseaseGrade } = useAnnotationStore((s) => ({
    annotation: s.annotations[s.currentImageId] || { disease_labels: [] },
    toggleDisease: s.toggleDisease,
    setDiseaseGrade: s.setDiseaseGrade,
  }))
  const { diseases, mechanisms } = useCatalogStore((s) => ({
    diseases: s.diseases,
    mechanisms: s.mechanisms,
  }))

  const [searchOpen, setSearchOpen] = useState(false)
  const [proposeOpen, setProposeOpen] = useState(false)

  const selected = new Set(annotation.disease_labels.map((l) => l.disease_code))
  const mechByCode = useMemo(
    () => Object.fromEntries(mechanisms.map((m) => [m.code, m])),
    [mechanisms],
  )
  const diseaseByCode = useMemo(
    () => Object.fromEntries(diseases.map((d) => [d.code, d])),
    [diseases],
  )

  const chronic = diseases.filter(
    (d) => d.common && d.urgency !== 'P1' && d.urgency !== 'P2',
  )
  const emergency = diseases.filter(
    (d) => d.common && (d.urgency === 'P1' || d.urgency === 'P2'),
  )
  const otherCount = diseases.filter((d) => !d.common).length

  return (
    <section className="px-5 pt-4">
      <div className="flex items-center justify-between">
        <div className="eyebrow">Diagnostic · multi-sélection</div>
        {annotation.disease_labels.length > 0 && (
          <span className="mono text-[10px] text-ink-tertiary">
            {annotation.disease_labels.length} sélectionné{annotation.disease_labels.length > 1 ? 's' : ''}
          </span>
        )}
      </div>

      {/* Chronic row */}
      <div className="mt-3 text-[10px] tracking-widest text-ink-tertiary uppercase mono">
        Chroniques fréquentes
      </div>
      <div className="mt-2 grid grid-cols-2 gap-1.5">
        {chronic.map((d) => (
          <DiseaseChip
            key={d.code}
            disease={d}
            mech={mechByCode[d.mechanism]}
            selected={selected.has(d.code)}
            onToggle={() => toggleDisease(d.code)}
          />
        ))}
        {chronic.length === 0 && (
          <div className="col-span-2 py-3 text-center mono text-[10px] text-ink-tertiary">
            Chargement du catalogue…
          </div>
        )}
      </div>

      {/* Inline grade pickers for gradable selected diseases */}
      <AnimatePresence>
        {annotation.disease_labels
          .filter((l) => diseaseByCode[l.disease_code]?.gradable)
          .map((label) => {
            const d = diseaseByCode[label.disease_code]
            if (!d) return null
            const grades = gradeOptions(d)
            return (
              <motion.div
                key={label.disease_code}
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.18 }}
                className="overflow-hidden"
              >
                <div className="mt-3 rounded-lg border border-accent/25 bg-accent/[0.035] px-2.5 py-2">
                  <div className="mono text-[10px] uppercase tracking-wider text-ink-secondary">
                    Grade · {d.name_fr}
                  </div>
                  <div className="mt-1.5 flex flex-wrap gap-1">
                    {grades.map((g) => (
                      <button
                        key={g.code}
                        onClick={() => setDiseaseGrade(d.code, g.code)}
                        className={`chip rounded-md px-2.5 py-1 text-[11px] border transition-colors ${
                          label.grade === g.code
                            ? 'border-accent bg-accent/20 text-accent'
                            : 'border-line bg-bg-elev1 text-ink-secondary hover:border-line-strong hover:text-ink-primary'
                        }`}
                      >
                        {g.label}
                      </button>
                    ))}
                  </div>
                </div>
              </motion.div>
            )
          })}
      </AnimatePresence>

      {/* Emergency row */}
      <div className="mt-4 flex items-center gap-2 text-[10px] tracking-widest text-ink-tertiary uppercase mono">
        <span>Urgences</span>
        <span className="flex h-1 w-1 rounded-full bg-urgency-p1 animate-pulse-soft" />
      </div>
      <div className="mt-2 grid grid-cols-2 gap-1.5">
        {emergency.map((d) => (
          <DiseaseChip
            key={d.code}
            disease={d}
            mech={mechByCode[d.mechanism]}
            selected={selected.has(d.code)}
            onToggle={() => toggleDisease(d.code)}
          />
        ))}
      </div>

      {/* Search + Propose */}
      <div className="mt-4 space-y-1.5">
        <button
          onClick={() => setSearchOpen(true)}
          className="btn flex w-full items-center gap-2 rounded-lg border border-line bg-bg-elev1 px-3 py-2 text-[12px] text-ink-secondary hover:border-line-strong hover:text-ink-primary"
        >
          <Search size={12} strokeWidth={2} />
          Autre pathologie…
          <span className="ml-auto mono text-[10px] text-ink-tertiary">
            {otherCount} disponibles
          </span>
        </button>

        <button
          onClick={() => setProposeOpen(true)}
          className="btn flex w-full items-center justify-center gap-1.5 rounded-lg border border-dashed border-line px-3 py-2 text-[12px] text-ink-tertiary hover:border-line-strong hover:text-ink-secondary"
        >
          <Plus size={12} strokeWidth={2} />
          Proposer une pathologie non-listée
        </button>
      </div>

      <AnimatePresence>
        {searchOpen && (
          <SearchDialog
            diseases={diseases}
            mechByCode={mechByCode}
            onClose={() => setSearchOpen(false)}
          />
        )}
      </AnimatePresence>
      <AnimatePresence>
        {proposeOpen && (
          <ProposeDialog
            mechanisms={mechanisms}
            onClose={() => setProposeOpen(false)}
          />
        )}
      </AnimatePresence>
    </section>
  )
}

// Convert grade_labels_json ({code: label}) → [{code, label}] sorted by key
function gradeOptions(disease) {
  if (!disease.grade_labels_json) return []
  return Object.entries(disease.grade_labels_json)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([code, label]) => ({ code, label }))
}

function DiseaseChip({ disease, mech, selected, onToggle }) {
  const urgency = disease.urgency
  return (
    <motion.button
      layout
      onClick={onToggle}
      whileTap={{ scale: 0.97 }}
      className={`chip group relative flex items-center gap-2 rounded-lg border px-2.5 py-2 text-left transition-all ${
        selected
          ? 'border-accent bg-accent/10 text-ink-primary'
          : 'border-line bg-bg-elev1 text-ink-secondary hover:border-line-strong hover:text-ink-primary'
      }`}
    >
      <div
        className={`flex h-4 w-4 flex-shrink-0 items-center justify-center rounded ${
          selected
            ? 'bg-accent'
            : urgency === 'P1'
              ? 'bg-urgency-p1/20'
              : urgency === 'P2'
                ? 'bg-urgency-p2/20'
                : 'bg-bg-elev2'
        }`}
      >
        {selected ? (
          <Check size={10} strokeWidth={3} className="text-bg-base" />
        ) : urgency === 'P1' ? (
          <span className="h-1 w-1 rounded-full bg-urgency-p1" />
        ) : urgency === 'P2' ? (
          <span className="h-1 w-1 rounded-full bg-urgency-p2" />
        ) : null}
      </div>

      <div className="min-w-0 flex-1">
        <span className="block truncate text-[12px] font-medium">{chipLabel(disease)}</span>
        {mech && (
          <span className="mono text-[9px] uppercase tracking-wider text-ink-tertiary">
            {mech.name_fr}
          </span>
        )}
      </div>
    </motion.button>
  )
}

function chipLabel(d) {
  const short = {
    DR: 'DR', GLAUC: 'Glaucome', DMLA: 'DMLA', HTN_DR: 'HTN-DR',
    OACR: 'OACR', ABACR: 'ABACR', NOIAA: 'NOIAA',
    DR_MAC_OFF: 'Décoll. rétine', GLAUC_AIGU: 'Crise glauc.',
  }
  return short[d.code] || d.name_fr
}

// ============================================================
//  Search Dialog
// ============================================================

function SearchDialog({ diseases, mechByCode, onClose }) {
  const [q, setQ] = useState('')
  const { toggleDisease } = useAnnotationStore()

  const other = diseases.filter((d) => !d.common)

  const results = useMemo(() => {
    if (!q.trim()) return other
    const needle = q.toLowerCase()
    return other.filter(
      (d) =>
        d.name_fr.toLowerCase().includes(needle) ||
        d.code.toLowerCase().includes(needle),
    )
  }, [q, other])

  const grouped = useMemo(() => {
    const g = {}
    for (const d of results) {
      const key = d.mechanism || 'OTHER'
      ;(g[key] ||= []).push(d)
    }
    return g
  }, [results])

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/60 backdrop-blur-sm pt-[15vh]"
      onClick={onClose}
    >
      <motion.div
        initial={{ y: -8, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        exit={{ y: -8, opacity: 0 }}
        onClick={(e) => e.stopPropagation()}
        className="w-[560px] max-w-[92vw] overflow-hidden rounded-xl border border-line bg-bg-elev1 shadow-2xl"
      >
        <div className="flex items-center gap-2 border-b border-line px-4 py-3">
          <Search size={14} strokeWidth={2} className="text-ink-tertiary" />
          <input
            autoFocus
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Rechercher une pathologie…"
            className="flex-1 bg-transparent text-[13px] outline-none placeholder:text-ink-tertiary"
          />
          <kbd className="mono rounded border border-line bg-bg-base px-1.5 py-0.5 text-[9px] text-ink-tertiary">
            Esc
          </kbd>
          <button onClick={onClose} className="text-ink-tertiary hover:text-ink-primary">
            <X size={14} />
          </button>
        </div>
        <div className="max-h-[60vh] overflow-y-auto px-3 py-2">
          {Object.entries(grouped).map(([mechCode, list]) => (
            <div key={mechCode} className="py-2">
              <div className="eyebrow px-2 py-1">
                {mechByCode[mechCode]?.name_fr || 'Autre'}
              </div>
              {list.map((d) => (
                <button
                  key={d.code}
                  onClick={() => {
                    toggleDisease(d.code)
                    onClose()
                  }}
                  className="flex w-full items-center justify-between rounded-md px-2 py-2 text-left hover:bg-bg-elev2"
                >
                  <span className="text-[12px]">{d.name_fr}</span>
                  <span className="mono text-[10px] text-ink-tertiary">{d.code}</span>
                </button>
              ))}
            </div>
          ))}
          {results.length === 0 && (
            <div className="py-10 text-center text-[12px] text-ink-tertiary">
              Aucun résultat pour "{q}".
            </div>
          )}
        </div>
      </motion.div>
    </motion.div>
  )
}

// ============================================================
//  Propose Dialog
// ============================================================

function ProposeDialog({ mechanisms, onClose }) {
  const token = useAuthStore((s) => s.token)
  const currentImageId = useAnnotationStore((s) => s.currentImageId)

  const [name, setName] = useState('')
  const [mech, setMech] = useState(mechanisms[0]?.code || 'VASC')
  const [gradable, setGradable] = useState(false)
  const [description, setDescription] = useState('')
  const [loading, setLoading] = useState(false)
  const [submitted, setSubmitted] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async () => {
    if (!name.trim()) return
    setLoading(true)
    setError('')
    try {
      await api.createProposal(
        {
          proposed_name: name.trim(),
          suspected_mechanism: mech,
          is_gradable: gradable,
          image_id: currentImageId || null,
          proposed_description: description.trim() || null,
        },
        token,
      )
      setSubmitted(true)
    } catch (e) {
      setError(e.message || 'Erreur lors de la soumission')
    } finally {
      setLoading(false)
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <motion.div
        initial={{ y: 8, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        exit={{ y: 8, opacity: 0 }}
        onClick={(e) => e.stopPropagation()}
        className="w-[500px] max-w-[92vw] overflow-hidden rounded-xl border border-line bg-bg-elev1 shadow-2xl"
      >
        <div className="border-b border-line px-5 py-3">
          <div className="display text-[15px]">Proposer une pathologie</div>
          <div className="mt-0.5 text-[11px] text-ink-tertiary">
            Votre proposition sera revue par l'administrateur avant d'être ajoutée au catalogue.
          </div>
        </div>

        {!submitted ? (
          <div className="space-y-4 px-5 py-4">
            <div>
              <label className="eyebrow">Nom de la pathologie</label>
              <input
                autoFocus
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Ex: rétinopathie de Purtscher"
                className="mt-1.5 w-full rounded-md border border-line bg-bg-base px-3 py-2 text-[13px] outline-none focus:border-accent/60"
              />
            </div>

            <div>
              <label className="eyebrow">Mécanisme suspecté</label>
              <div className="mt-1.5 grid grid-cols-3 gap-1">
                {mechanisms.map((m) => (
                  <button
                    key={m.code}
                    onClick={() => setMech(m.code)}
                    className={`chip rounded-md border px-2 py-1.5 text-[11px] transition-colors ${
                      mech === m.code
                        ? 'border-accent bg-accent/10 text-accent'
                        : 'border-line text-ink-secondary hover:text-ink-primary'
                    }`}
                  >
                    {m.name_fr}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="eyebrow">Description / observations (optionnel)</label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={3}
                placeholder="Décrivez ce que vous observez sur l'image…"
                className="mt-1.5 w-full resize-none rounded-md border border-line bg-bg-base px-3 py-2 text-[12px] outline-none focus:border-accent/60 placeholder:text-ink-tertiary"
              />
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={() => setGradable(!gradable)}
                className={`relative h-5 w-9 rounded-full transition-colors ${
                  gradable ? 'bg-accent' : 'bg-bg-elev3'
                }`}
              >
                <span
                  className={`absolute top-0.5 h-4 w-4 rounded-full bg-bg-elev1 transition-transform ${
                    gradable ? 'translate-x-[18px]' : 'translate-x-0.5'
                  }`}
                />
              </button>
              <label className="text-[12px] text-ink-secondary">Pathologie gradable</label>
            </div>

            {error && (
              <p className="rounded-md border border-urgency-p1/30 bg-urgency-p1/10 px-3 py-2 text-[12px] text-urgency-p1">
                {error}
              </p>
            )}
          </div>
        ) : (
          <div className="px-5 py-8 text-center">
            <div className="mx-auto flex h-10 w-10 items-center justify-center rounded-full bg-accent/15">
              <Check size={18} strokeWidth={2.5} className="text-accent" />
            </div>
            <div className="mt-3 display text-[14px]">Proposition envoyée</div>
            <div className="mt-1 text-[11px] text-ink-tertiary">
              L'administrateur sera notifié pour revue.
            </div>
          </div>
        )}

        <div className="flex items-center justify-end gap-2 border-t border-line px-5 py-3">
          <button
            onClick={onClose}
            className="btn rounded-md border border-line px-3 py-1.5 text-[12px] text-ink-secondary hover:text-ink-primary"
          >
            {submitted ? 'Fermer' : 'Annuler'}
          </button>
          {!submitted && (
            <button
              disabled={!name.trim() || loading}
              onClick={handleSubmit}
              className="btn flex items-center gap-1.5 rounded-md bg-accent px-3 py-1.5 text-[12px] font-medium text-bg-base disabled:opacity-40"
            >
              {loading && <Loader2 size={12} className="animate-spin" />}
              Soumettre
            </button>
          )}
        </div>
      </motion.div>
    </motion.div>
  )
}
