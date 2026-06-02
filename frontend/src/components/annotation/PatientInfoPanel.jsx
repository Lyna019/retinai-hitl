/**
 * Patient info panel — top-left of the annotation viewer.
 * Shows: Patient ID, sex, age, laterality, known systemic diseases, historical note.
 */
import { useState, useEffect, useCallback } from 'react'
import { User, Edit3, Check, X, ChevronDown, ChevronUp } from 'lucide-react'
import { useAuthStore, useCatalogStore } from '../../lib/store'
import { api } from '../../lib/api'
import { fmtDatetime } from '../../lib/utils'

export default function PatientInfoPanel({ image }) {
  const token = useAuthStore((s) => s.token)
  const systemicDiseases = useCatalogStore((s) => s.systemicDiseases)

  const [patient, setPatient] = useState(null)
  const [expanded, setExpanded] = useState(false)
  const [editingNote, setEditingNote] = useState(false)
  const [noteText, setNoteText] = useState('')
  const [selectedSDs, setSelectedSDs] = useState([])
  const [saving, setSaving] = useState(false)

  const fetchPatient = useCallback(async () => {
    if (!image?.patient_id || !token) return
    try {
      const p = await api.getPatient(image.patient_id, token)
      setPatient(p)
      setNoteText(p.historical_note || '')
      setSelectedSDs(p.systemic_diseases?.map((s) => s.id) || [])
    } catch (e) {
      console.error('fetchPatient', e)
    }
  }, [image?.patient_id, token])

  useEffect(() => { fetchPatient() }, [fetchPatient])

  const saveNote = async () => {
    if (!patient) return
    setSaving(true)
    try {
      await api.updateHistoricalNote(patient.id, noteText, token)
      setPatient((p) => ({ ...p, historical_note: noteText }))
      setEditingNote(false)
    } finally {
      setSaving(false)
    }
  }

  const toggleSD = async (sdId) => {
    if (!patient) return
    const next = selectedSDs.includes(sdId)
      ? selectedSDs.filter((x) => x !== sdId)
      : [...selectedSDs, sdId]
    setSelectedSDs(next)
    try {
      await api.setSystemicDiseases(patient.id, next, token)
    } catch (e) {
      console.error('setSystemicDiseases', e)
      setSelectedSDs(selectedSDs) // revert
    }
  }

  if (!image) return null

  const eyeLabel = image.eye === 'OD' ? 'Œil droit (OD)' : image.eye === 'OS' ? 'Œil gauche (OS)' : image.eye

  return (
    <div className="absolute top-3 left-3 z-20 w-[230px] rounded-xl border border-line/60 bg-bg-base/90 backdrop-blur-sm shadow-lg text-[11px]">
      {/* Header */}
      <button
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2.5 text-left"
      >
        <div className="flex h-6 w-6 items-center justify-center rounded-full bg-accent/15 flex-shrink-0">
          <User size={12} className="text-accent" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="font-medium text-ink-primary truncate">
            {patient?.full_name || patient?.clinical_id || image.patient_id?.slice(0, 8)}
          </div>
          <div className="text-ink-tertiary flex gap-1.5">
            <span>{patient?.gender || '—'}</span>
            {patient?.age && <><span>·</span><span>{patient.age} ans</span></>}
            <span>·</span>
            <span className="text-accent font-medium">{image.eye || '—'}</span>
          </div>
        </div>
        {expanded ? <ChevronUp size={12} className="text-ink-tertiary flex-shrink-0" /> : <ChevronDown size={12} className="text-ink-tertiary flex-shrink-0" />}
      </button>

      {/* Expanded content */}
      {expanded && (
        <div className="border-t border-line/50 px-3 py-2.5 space-y-3">
          {/* Key facts */}
          <div className="grid grid-cols-2 gap-x-3 gap-y-1">
            <Row label="ID" value={patient?.clinical_id || '—'} />
            <Row label="Œil" value={eyeLabel} />
            <Row label="Modalité" value={image.modality || '—'} />
            {image.capture_date && (
              <Row label="Date" value={fmtDatetime(image.capture_date)} />
            )}
          </div>

          {/* Systemic diseases */}
          <div>
            <div className="eyebrow mb-1.5">Maladies systémiques</div>
            <div className="flex flex-wrap gap-1">
              {systemicDiseases.map((sd) => {
                const active = selectedSDs.includes(sd.id)
                return (
                  <button
                    key={sd.id}
                    onClick={() => toggleSD(sd.id)}
                    title={sd.category}
                    className={`rounded px-1.5 py-0.5 text-[10px] border transition-colors ${
                      active
                        ? 'border-accent/60 bg-accent/10 text-accent'
                        : 'border-line bg-bg-elev1 text-ink-tertiary hover:text-ink-secondary hover:border-line-strong'
                    }`}
                  >
                    {sd.name_fr}
                  </button>
                )
              })}
              {systemicDiseases.length === 0 && (
                <span className="text-ink-tertiary italic">Chargement…</span>
              )}
            </div>
          </div>

          {/* Historical note */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <div className="eyebrow">Note historique</div>
              {!editingNote && (
                <button
                  onClick={() => setEditingNote(true)}
                  className="text-ink-tertiary hover:text-ink-secondary"
                >
                  <Edit3 size={10} />
                </button>
              )}
            </div>
            {editingNote ? (
              <div className="space-y-1.5">
                <textarea
                  autoFocus
                  value={noteText}
                  onChange={(e) => setNoteText(e.target.value)}
                  rows={3}
                  className="w-full resize-none rounded border border-line bg-bg-base px-2 py-1.5 text-[11px] outline-none focus:border-accent/60"
                  placeholder="Antécédents, observations…"
                />
                <div className="flex gap-1.5 justify-end">
                  <button
                    onClick={() => { setEditingNote(false); setNoteText(patient?.historical_note || '') }}
                    className="rounded border border-line px-2 py-0.5 text-ink-tertiary hover:text-ink-secondary"
                  >
                    <X size={10} />
                  </button>
                  <button
                    onClick={saveNote}
                    disabled={saving}
                    className="rounded bg-accent px-2 py-0.5 text-[10px] font-medium text-bg-base disabled:opacity-50"
                  >
                    <Check size={10} />
                  </button>
                </div>
              </div>
            ) : (
              <p className="text-ink-secondary leading-relaxed min-h-[24px]">
                {patient?.historical_note || (
                  <span className="italic text-ink-tertiary">Aucune note. Cliquez pour éditer.</span>
                )}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function Row({ label, value }) {
  return (
    <>
      <span className="text-ink-tertiary">{label}</span>
      <span className="text-ink-secondary font-medium truncate">{value}</span>
    </>
  )
}
