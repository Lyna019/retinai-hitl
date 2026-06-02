// ============================================================
//  Static reference data — matches the backend Disease catalog
// ============================================================

export const MECHANISMS = [
  { code: 'VASC', name_fr: 'Vasculaire', color: 'mech.vasc' },
  { code: 'DEGEN', name_fr: 'Dégénératif', color: 'mech.degen' },
  { code: 'INFLAM', name_fr: 'Inflammatoire', color: 'mech.inflam' },
  { code: 'DYST', name_fr: 'Dystrophique', color: 'mech.dyst' },
  { code: 'STRUCT', name_fr: 'Structurel', color: 'mech.struct' },
  { code: 'TUMOR', name_fr: 'Tumoral', color: 'mech.tumor' },
]

export const DR_GRADES = [
  { code: '0', label: '0 — absent' },
  { code: '1', label: '1 — léger' },
  { code: '2', label: '2 — modéré' },
  { code: '3', label: '3 — sévère' },
  { code: '4', label: '4 — proliférative' },
]

export const GLAUCOMA_GRADES = [
  { code: 'DEB', label: 'Débutante' },
  { code: 'EVO', label: 'Évolutive' },
]

export const DMLA_GRADES = [
  { code: 'SEC', label: 'Sèche' },
  { code: 'HUM', label: 'Humide' },
]

export const HTN_DR_GRADES = [
  { code: '1', label: 'Stade 1' },
  { code: '2', label: 'Stade 2' },
  { code: '3', label: 'Stade 3' },
  { code: '4', label: 'Stade 4' },
]

// Primary catalog of diseases — code, display name, mechanism, gradability
export const DISEASES = [
  // Chronic common
  { code: 'DR', name_fr: 'Rétinopathie diabétique', mechanism: 'VASC', gradable: true, grades: DR_GRADES, common: true, group: 'chronique' },
  { code: 'GLAUC', name_fr: 'Glaucome', mechanism: 'STRUCT', gradable: true, grades: GLAUCOMA_GRADES, common: true, group: 'chronique' },
  { code: 'DMLA', name_fr: 'DMLA', mechanism: 'DEGEN', gradable: true, grades: DMLA_GRADES, common: true, group: 'chronique' },
  { code: 'HTN_DR', name_fr: 'Rétinopathie hypertensive', mechanism: 'VASC', gradable: true, grades: HTN_DR_GRADES, common: true, group: 'chronique' },

  // P1 emergencies
  { code: 'OACR', name_fr: 'OACR — occlusion artère centrale', mechanism: 'VASC', gradable: false, common: true, group: 'urgence', urgency_override: 'P1' },
  { code: 'ABACR', name_fr: 'ABACR — occlusion branche artérielle', mechanism: 'VASC', gradable: false, common: true, group: 'urgence', urgency_override: 'P1' },
  { code: 'NOIAA', name_fr: 'NOIAA — neuropathie optique ischémique', mechanism: 'VASC', gradable: false, common: true, group: 'urgence', urgency_override: 'P1' },

  // P2 same-day
  { code: 'DR_MAC_OFF', name_fr: 'Décollement de rétine (macula off)', mechanism: 'STRUCT', gradable: false, common: true, group: 'urgence', urgency_override: 'P2' },
  { code: 'GLAUC_AIGU', name_fr: 'Crise de glaucome aigu', mechanism: 'STRUCT', gradable: false, common: true, group: 'urgence', urgency_override: 'P2' },

  // Long tail — searchable
  { code: 'OVCR', name_fr: 'OVCR — occlusion veine centrale', mechanism: 'VASC', gradable: false, group: 'autre' },
  { code: 'BRVO', name_fr: 'Occlusion de branche veineuse', mechanism: 'VASC', gradable: false, group: 'autre' },
  { code: 'CHORIO', name_fr: 'Choriorétinite', mechanism: 'INFLAM', gradable: false, group: 'autre' },
  { code: 'UVEITE', name_fr: 'Uvéite postérieure', mechanism: 'INFLAM', gradable: false, group: 'autre' },
  { code: 'STARGARDT', name_fr: 'Maladie de Stargardt', mechanism: 'DYST', gradable: false, group: 'autre' },
  { code: 'BEST', name_fr: 'Maladie de Best', mechanism: 'DYST', gradable: false, group: 'autre' },
  { code: 'RP', name_fr: 'Rétinite pigmentaire', mechanism: 'DYST', gradable: false, group: 'autre' },
  { code: 'MELANOME', name_fr: 'Mélanome choroïdien', mechanism: 'TUMOR', gradable: false, group: 'autre' },
  { code: 'RETINOBLA', name_fr: 'Rétinoblastome', mechanism: 'TUMOR', gradable: false, group: 'autre' },
  { code: 'MYOPIE_F', name_fr: 'Myopie forte (fond)', mechanism: 'STRUCT', gradable: false, group: 'autre' },
  { code: 'NORMAL', name_fr: 'Fond normal', mechanism: null, gradable: false, group: 'autre' },
]

export const LESION_VOCAB = [
  { code: 'HEM', name_fr: 'Hémorragie', color: '#FF453A' },
  { code: 'EXS', name_fr: 'Exsudat dur', color: '#FFD60A' },
  { code: 'MA', name_fr: 'Microanévrisme', color: '#FF9F0A' },
  { code: 'NV', name_fr: 'Néovaisseau', color: '#BF5AF2' },
  { code: 'DRUS', name_fr: 'Drusen', color: '#64D2FF' },
  { code: 'CW', name_fr: 'Cotton-wool', color: '#FFFFFF' },
  { code: 'OED', name_fr: 'Œdème', color: '#30D158' },
]

// ============================================================
//  Mock image queue
// ============================================================

const makePredictions = (primary, primaryConf, others) => ({
  model_version: 'retinai-v0.3.1',
  top_k: [
    { disease_code: primary, confidence: primaryConf },
    ...others,
  ],
  gradcam_url: null,
  uncertainty: Math.max(0, 1 - primaryConf + Math.random() * 0.1),
})

export const MOCK_IMAGES = [
  {
    id: 'img-2040',
    patient_id: 'P-2040',
    eye: 'OS',
    modality: 'UWF',
    capture_date: '2026-04-23',
    status: 'pending',
    uncertainty: 0.91,
    predictions: makePredictions('OACR', 0.72, [
      { disease_code: 'OVCR', confidence: 0.15 },
      { disease_code: 'NORMAL', confidence: 0.13 },
    ]),
  },
  {
    id: 'img-2041',
    patient_id: 'P-2041',
    eye: 'OD',
    modality: 'UWF',
    capture_date: '2026-04-24',
    status: 'in_progress',
    uncertainty: 0.87,
    predictions: makePredictions('DR', 0.87, [
      { disease_code: 'DR', confidence: 0.09, grade: '3' },
      { disease_code: 'NORMAL', confidence: 0.04 },
    ]),
  },
  {
    id: 'img-2042',
    patient_id: 'P-2042',
    eye: 'OD',
    modality: 'STD',
    capture_date: '2026-04-24',
    status: 'pending',
    uncertainty: 0.74,
    predictions: makePredictions('DMLA', 0.81, [
      { disease_code: 'NORMAL', confidence: 0.12 },
      { disease_code: 'DR', confidence: 0.07 },
    ]),
  },
  {
    id: 'img-2043',
    patient_id: 'P-2043',
    eye: 'OS',
    modality: 'UWF',
    capture_date: '2026-04-22',
    status: 'pending',
    uncertainty: 0.68,
    predictions: makePredictions('NORMAL', 0.74, [
      { disease_code: 'DR', confidence: 0.18 },
      { disease_code: 'HTN_DR', confidence: 0.08 },
    ]),
  },
  {
    id: 'img-2044',
    patient_id: 'P-2044',
    eye: 'OD',
    modality: 'UWF',
    capture_date: '2026-04-22',
    status: 'pending',
    uncertainty: 0.61,
    predictions: makePredictions('GLAUC', 0.66, [
      { disease_code: 'NORMAL', confidence: 0.26 },
      { disease_code: 'DMLA', confidence: 0.08 },
    ]),
  },
  {
    id: 'img-2045',
    patient_id: 'P-2045',
    eye: 'OS',
    modality: 'STD',
    capture_date: '2026-04-21',
    status: 'pending',
    uncertainty: 0.55,
    predictions: makePredictions('NORMAL', 0.81, [
      { disease_code: 'DMLA', confidence: 0.12 },
      { disease_code: 'DR', confidence: 0.07 },
    ]),
  },
  {
    id: 'img-2046',
    patient_id: 'P-2046',
    eye: 'OD',
    modality: 'UWF',
    capture_date: '2026-04-20',
    status: 'done',
    uncertainty: 0.22,
    predictions: makePredictions('DR', 0.94, [
      { disease_code: 'DR', confidence: 0.04 },
      { disease_code: 'NORMAL', confidence: 0.02 },
    ]),
  },
]

// ============================================================
//  Mock admin dashboard stats
// ============================================================

export const MOCK_STATS = {
  progress: { done: 12, total: 50, in_progress: 3, pending: 35, urgent: 4 },
  avg_time_sec: 94,
  daily_annotations: [
    { day: 'Lun', count: 14 }, { day: 'Mar', count: 18 }, { day: 'Mer', count: 21 },
    { day: 'Jeu', count: 16 }, { day: 'Ven', count: 12 }, { day: 'Sam', count: 8 }, { day: 'Dim', count: 4 },
  ],
  disease_distribution: [
    { name: 'DR', value: 42 }, { name: 'Normale', value: 28 }, { name: 'DMLA', value: 12 },
    { name: 'Glaucome', value: 9 }, { name: 'HTN-DR', value: 6 }, { name: 'Autre', value: 3 },
  ],
  urgency_distribution: [
    { level: 'P1', count: 3 }, { level: 'P2', count: 5 }, { level: 'P3', count: 18 }, { level: 'P4', count: 74 },
  ],
  gradcam_validation: [
    { verdict: 'Correct', count: 58 }, { verdict: 'Partiel', count: 23 }, { verdict: 'Faux', count: 19 },
  ],
  doctors: [
    { id: 'u-01', initials: 'MM', name: 'Dr. Mekki', annotations: 128, avg_time: 96 },
    { id: 'u-02', initials: 'AB', name: 'Dr. Belkacem', annotations: 67, avg_time: 112 },
    { id: 'u-03', initials: 'KH', name: 'Dr. Haddad', annotations: 41, avg_time: 104 },
  ],
  kappa: 0.78,
}
