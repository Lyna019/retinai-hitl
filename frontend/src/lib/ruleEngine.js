import { DISEASES } from './mockData'

const PRIORITY_ORDER = { P1: 1, P2: 2, P3: 3, P4: 4 }

// Fallback lookup from static mockData (used when catalog not yet loaded)
const fallbackByCode = Object.fromEntries(DISEASES.map((d) => [d.code, d]))

// Support both API format (d.urgency) and mockData format (d.urgency_override)
const diseaseUrgency = (d) => d.urgency || d.urgency_override || null

/**
 * Given a set of selected disease labels, return the minimum priority
 * (highest clinical urgency) and the rule that triggered it.
 *
 * P1: OACR / ABACR / NOIAA
 * P2: Décollement macula-off / Crise de glaucome aigu
 * P3: DR-4 (proliférative) / DMLA humide / Glaucome évolutive / HTN-DR stade 3-4
 * P4: everything else
 *
 * @param {Array} labels          - disease_labels from annotation store
 * @param {Object|null} diseaseMap - optional map code→disease from catalog store
 */
export function computeUrgency(labels, diseaseMap = null) {
  if (!labels || labels.length === 0) return null
  const byCode = diseaseMap || fallbackByCode

  let best = { level: 'P4', rule: 'Routine', source: null }

  for (const label of labels) {
    const d = byCode[label.disease_code]
    if (!d) continue

    // Disease-level urgency (P1/P2 emergencies)
    const urg = diseaseUrgency(d)
    if (urg === 'P1' || urg === 'P2') {
      if (PRIORITY_ORDER[urg] < PRIORITY_ORDER[best.level]) {
        best = { level: urg, rule: d.name_fr, source: d.code }
      }
      continue
    }

    // Grade-based P3 upgrades (hardcoded clinical rules)
    if (d.code === 'DR' && label.grade === '4') {
      if (PRIORITY_ORDER.P3 < PRIORITY_ORDER[best.level])
        best = { level: 'P3', rule: 'DR Proliférative (grade 4)', source: 'DR' }
    }
    if (d.code === 'DMLA' && label.grade === 'HUM') {
      if (PRIORITY_ORDER.P3 < PRIORITY_ORDER[best.level])
        best = { level: 'P3', rule: 'DMLA Humide', source: 'DMLA' }
    }
    if (d.code === 'GLAUC' && label.grade === 'EVO') {
      if (PRIORITY_ORDER.P3 < PRIORITY_ORDER[best.level])
        best = { level: 'P3', rule: 'Glaucome évolutive', source: 'GLAUC' }
    }
    if (d.code === 'HTN_DR' && (label.grade === '3' || label.grade === '4')) {
      if (PRIORITY_ORDER.P3 < PRIORITY_ORDER[best.level])
        best = { level: 'P3', rule: `HTN-DR stade ${label.grade}`, source: 'HTN_DR' }
    }
  }

  return best
}

/**
 * Derive the set of mechanism codes triggered by the selected diseases.
 *
 * @param {Array} labels          - disease_labels from annotation store
 * @param {Object|null} diseaseMap - optional map code→disease from catalog store
 */
export function computeMechanisms(labels, diseaseMap = null) {
  if (!labels || labels.length === 0) return []
  const byCode = diseaseMap || fallbackByCode
  const codes = new Set()
  for (const label of labels) {
    const d = byCode[label.disease_code]
    if (d?.mechanism) codes.add(d.mechanism)
  }
  return Array.from(codes)
}

export function isGradable(disease_code, diseaseMap = null) {
  const byCode = diseaseMap || fallbackByCode
  return !!(byCode[disease_code]?.gradable)
}
