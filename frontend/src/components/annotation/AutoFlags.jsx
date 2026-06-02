import { useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Zap, Activity } from 'lucide-react'
import { useAnnotationStore, useCatalogStore } from '../../lib/store'
import { computeMechanisms, computeUrgency } from '../../lib/ruleEngine'

const URGENCY_CONFIG = {
  P1: { bg: 'from-urgency-p1/25 to-urgency-p1/5', border: 'border-urgency-p1/50', text: 'text-urgency-p1', label: 'Urgence vitale', pulse: true },
  P2: { bg: 'from-urgency-p2/25 to-urgency-p2/5', border: 'border-urgency-p2/50', text: 'text-urgency-p2', label: 'Chirurgie du jour', pulse: true },
  P3: { bg: 'from-urgency-p3/20 to-urgency-p3/5', border: 'border-urgency-p3/40', text: 'text-urgency-p3', label: 'Suivi urgent', pulse: false },
  P4: { bg: 'from-urgency-p4/15 to-urgency-p4/5', border: 'border-urgency-p4/30', text: 'text-urgency-p4', label: 'Routine', pulse: false },
}

const MECH_COLOR_HEX = {
  VASC: '#5E9CFF', DEGEN: '#BF5AF2', INFLAM: '#FF6482',
  DYST: '#FF9F0A', STRUCT: '#64D2FF', TUMOR: '#FF453A',
}

export default function AutoFlags() {
  const labels = useAnnotationStore((s) => s.annotations[s.currentImageId]?.disease_labels || [])
  const { diseases, mechanisms: catalogMechs } = useCatalogStore((s) => ({
    diseases: s.diseases,
    mechanisms: s.mechanisms,
  }))

  const diseaseMap = useMemo(
    () => Object.fromEntries(diseases.map((d) => [d.code, d])),
    [diseases],
  )
  const mechByCode = useMemo(
    () => Object.fromEntries(catalogMechs.map((m) => [m.code, m])),
    [catalogMechs],
  )

  const mechanisms = computeMechanisms(labels, diseaseMap)
  const urgency = computeUrgency(labels, diseaseMap)
  const hasAny = labels.length > 0

  return (
    <section className="px-5 pt-4">
      <div className="grid grid-cols-2 gap-2">
        {/* Mechanisms */}
        <div className="rounded-xl border border-line bg-bg-elev1 p-3">
          <div className="flex items-center gap-1.5">
            <Activity size={10} strokeWidth={2} className="text-ink-tertiary" />
            <span className="eyebrow">Mécanismes · auto</span>
          </div>
          <div className="mt-2 flex min-h-[24px] flex-wrap gap-1">
            <AnimatePresence mode="popLayout">
              {mechanisms.length === 0 ? (
                <span key="empty" className="mono text-[10px] text-ink-tertiary">
                  — en attente
                </span>
              ) : (
                mechanisms.map((code) => (
                  <motion.span
                    key={code}
                    layout
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.8, opacity: 0 }}
                    className="rounded-md px-2 py-0.5 text-[11px] font-medium"
                    style={{
                      background: `${MECH_COLOR_HEX[code] || '#9A9AA3'}22`,
                      color: MECH_COLOR_HEX[code] || '#9A9AA3',
                      border: `0.5px solid ${MECH_COLOR_HEX[code] || '#9A9AA3'}55`,
                    }}
                  >
                    {mechByCode[code]?.name_fr || code}
                  </motion.span>
                ))
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Urgency */}
        <div
          className={`relative overflow-hidden rounded-xl border p-3 ${
            urgency ? URGENCY_CONFIG[urgency.level].border : 'border-line'
          } bg-gradient-to-br ${urgency ? URGENCY_CONFIG[urgency.level].bg : 'from-bg-elev1 to-bg-elev1'}`}
        >
          <div className="flex items-center gap-1.5">
            <Zap size={10} strokeWidth={2} className="text-ink-tertiary" />
            <span className="eyebrow">Urgence · auto</span>
          </div>
          <div className="mt-1.5 flex items-baseline gap-2">
            {hasAny && urgency ? (
              <>
                <span
                  className={`mono display text-[22px] font-medium leading-none ${
                    URGENCY_CONFIG[urgency.level].text
                  } ${URGENCY_CONFIG[urgency.level].pulse ? 'animate-pulse-soft' : ''}`}
                >
                  {urgency.level}
                </span>
                <span className={`text-[11px] ${URGENCY_CONFIG[urgency.level].text}`}>
                  {URGENCY_CONFIG[urgency.level].label}
                </span>
              </>
            ) : (
              <span className="mono text-[11px] text-ink-tertiary">— en attente</span>
            )}
          </div>
          {hasAny && urgency && urgency.rule !== 'Routine' && (
            <div className="mt-1.5 mono text-[10px] text-ink-secondary line-clamp-1">
              {urgency.rule}
            </div>
          )}
        </div>
      </div>
    </section>
  )
}
