/**
 * Patient gallery — images organized by patient ID.
 * Each patient folder shows all their fundus images (annotated + pending).
 * Clicking an image opens the full annotation workflow.
 */
import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, Calendar, ChevronDown, ChevronRight, Eye, User, Clock, CheckCircle2 } from 'lucide-react'
import { useAuthStore, useAnnotationStore } from '../lib/store'
import { api } from '../lib/api'
import { fmtDate, fmtDatetime } from '../lib/utils'

const STATUS_COLOR = {
  done:        'text-green-400 bg-green-400/10 border-green-400/30',
  in_progress: 'text-yellow-400 bg-yellow-400/10 border-yellow-400/30',
  pending:     'text-ink-tertiary bg-bg-elev2 border-line',
}
const STATUS_LABEL = { done: 'Annoté', in_progress: 'En cours', pending: 'En attente' }

export default function PatientsPage() {
  const token = useAuthStore((s) => s.token)
  const navigate = useNavigate()
  const openImageStore = useAnnotationStore((s) => s.openImage)

  const [patients, setPatients] = useState([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')
  const [sort, setSort] = useState('id')
  const [expanded, setExpanded] = useState({})
  const [patientDetails, setPatientDetails] = useState({})

  const fetchPatients = useCallback(async () => {
    setLoading(true)
    try {
      const params = { sort }
      if (dateFrom) params.date_from = dateFrom
      if (dateTo)   params.date_to   = dateTo
      const data = await api.listPatients(token, params)
      setPatients(data)
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }, [token, sort, dateFrom, dateTo])

  useEffect(() => { fetchPatients() }, [fetchPatients])

  const toggleExpand = async (patientId) => {
    setExpanded((prev) => ({ ...prev, [patientId]: !prev[patientId] }))
    if (!patientDetails[patientId]) {
      try {
        const params = {}
        if (dateFrom) params.date_from = dateFrom
        if (dateTo)   params.date_to   = dateTo
        const detail = await api.getPatient(patientId, token, params)
        setPatientDetails((prev) => ({ ...prev, [patientId]: detail }))
      } catch (e) {
        console.error(e)
      }
    }
  }

  const openImage = (img) => {
    openImageStore(img.id)
    navigate('/annotate')
  }

  const filtered = patients.filter((p) => {
    const q = search.toLowerCase()
    return (
      !q ||
      p.clinical_id?.toLowerCase().includes(q) ||
      p.full_name?.toLowerCase().includes(q)
    )
  })

  return (
    <div className="flex h-full flex-col bg-bg-base">
      {/* Toolbar */}
      <div className="flex items-center gap-3 border-b border-line px-6 py-3 flex-shrink-0">
        <h1 className="display text-[15px] font-semibold text-ink-primary">Dossiers patients</h1>
        <div className="flex-1" />

        {/* Search */}
        <div className="flex items-center gap-2 rounded-lg border border-line bg-bg-elev1 px-3 py-1.5">
          <Search size={13} className="text-ink-tertiary" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Rechercher ID ou nom…"
            className="bg-transparent text-[12px] outline-none w-40 placeholder:text-ink-tertiary"
          />
        </div>

        {/* Date range */}
        <div className="flex items-center gap-1.5 text-[11px] text-ink-secondary">
          <Calendar size={12} className="text-ink-tertiary" />
          <input
            type="date"
            value={dateFrom}
            onChange={(e) => setDateFrom(e.target.value)}
            className="rounded border border-line bg-bg-elev1 px-2 py-1 text-[11px] outline-none"
          />
          <span className="text-ink-tertiary">→</span>
          <input
            type="date"
            value={dateTo}
            onChange={(e) => setDateTo(e.target.value)}
            className="rounded border border-line bg-bg-elev1 px-2 py-1 text-[11px] outline-none"
          />
        </div>

        {/* Sort */}
        <select
          value={sort}
          onChange={(e) => setSort(e.target.value)}
          className="rounded border border-line bg-bg-elev1 px-2 py-1 text-[11px] outline-none text-ink-secondary"
        >
          <option value="id">Tri : ID patient</option>
          <option value="date">Tri : Date récente</option>
        </select>
      </div>

      {/* Patient list */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-2">
        {loading && (
          <div className="py-16 text-center text-[13px] text-ink-tertiary">Chargement…</div>
        )}

        {!loading && filtered.length === 0 && (
          <div className="py-16 text-center text-[13px] text-ink-tertiary">
            Aucun patient trouvé{search ? ` pour "${search}"` : ''}.
          </div>
        )}

        {filtered.map((patient) => (
          <PatientRow
            key={patient.id}
            patient={patient}
            expanded={!!expanded[patient.id]}
            detail={patientDetails[patient.id]}
            onToggle={() => toggleExpand(patient.id)}
            onOpenImage={openImage}
          />
        ))}
      </div>
    </div>
  )
}


function PatientRow({ patient, expanded, detail, onToggle, onOpenImage }) {
  const doneCount   = detail?.images?.filter((i) => i.status === 'done').length ?? 0
  const totalCount  = detail?.images?.length ?? patient.image_count

  return (
    <div className="rounded-xl border border-line bg-bg-elev1 overflow-hidden">
      {/* Patient header */}
      <button
        onClick={onToggle}
        className="flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-bg-elev2 transition-colors"
      >
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-accent/10 flex-shrink-0">
          <User size={15} className="text-accent" />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium text-[13px] text-ink-primary">
              {patient.full_name || patient.clinical_id}
            </span>
            <span className="mono text-[10px] text-ink-tertiary">{patient.clinical_id}</span>
          </div>
          <div className="flex items-center gap-3 mt-0.5 text-[11px] text-ink-tertiary">
            {patient.gender && <span>{patient.gender === 'M' ? 'Homme' : patient.gender === 'F' ? 'Femme' : patient.gender}</span>}
            {patient.age   && <span>{patient.age} ans</span>}
            {patient.last_capture_date && (
              <span className="flex items-center gap-1">
                <Clock size={9} />
                {fmtDatetime(patient.last_capture_date)}
              </span>
            )}
          </div>
        </div>

        {/* Image count + progress */}
        <div className="flex items-center gap-3 flex-shrink-0">
          <div className="text-right">
            <div className="text-[12px] font-medium text-ink-primary">
              {patient.image_count} image{patient.image_count !== 1 ? 's' : ''}
            </div>
            {detail && (
              <div className="text-[10px] text-ink-tertiary">
                {doneCount}/{totalCount} annotées
              </div>
            )}
          </div>
          {expanded
            ? <ChevronDown size={14} className="text-ink-tertiary" />
            : <ChevronRight size={14} className="text-ink-tertiary" />
          }
        </div>
      </button>

      {/* Image grid */}
      {expanded && (
        <div className="border-t border-line px-4 py-3">
          {!detail ? (
            <div className="text-[11px] text-ink-tertiary py-4 text-center">Chargement…</div>
          ) : detail.images.length === 0 ? (
            <div className="text-[11px] text-ink-tertiary py-4 text-center italic">
              Aucune image dans cette période.
            </div>
          ) : (
            <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-2">
              {detail.images.map((img) => (
                <ImageThumb key={img.id} img={img} onClick={() => onOpenImage(img)} />
              ))}
            </div>
          )}

          {/* Systemic diseases */}
          {detail?.systemic_diseases?.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1">
              {detail.systemic_diseases.map((sd) => (
                <span
                  key={sd.id}
                  className="rounded px-1.5 py-0.5 text-[10px] border border-line text-ink-tertiary"
                >
                  {sd.name_fr}
                </span>
              ))}
            </div>
          )}

          {/* Historical note */}
          {detail?.historical_note && (
            <p className="mt-2 text-[11px] text-ink-secondary italic border-l-2 border-accent/30 pl-2">
              {detail.historical_note}
            </p>
          )}
        </div>
      )}
    </div>
  )
}


function ImageThumb({ img, onClick }) {
  return (
    <button
      onClick={onClick}
      title={`${img.eye} · ${img.modality} · ${img.status}`}
      className="group relative aspect-square rounded-lg border border-line overflow-hidden bg-bg-elev2 hover:border-accent/60 transition-all"
    >
      {/* Placeholder — replace src with img.file_url when images are real */}
      <div className="absolute inset-0 flex flex-col items-center justify-center gap-1">
        <Eye size={16} className="text-ink-tertiary group-hover:text-accent transition-colors" />
        <span className="mono text-[8px] text-ink-tertiary">{img.eye}</span>
      </div>

      {/* Status badge */}
      <div className={`absolute bottom-1 right-1 rounded px-1 py-0.5 text-[8px] border font-medium ${STATUS_COLOR[img.status] || STATUS_COLOR.pending}`}>
        {img.status === 'done' ? <CheckCircle2 size={9} className="inline" /> : STATUS_LABEL[img.status]}
      </div>

      {/* Lock indicator */}
      {img.locked_by && (
        <div className="absolute top-1 left-1 h-2 w-2 rounded-full bg-yellow-400" title="Verrouillé" />
      )}

      {/* Capture date */}
      {img.capture_date && (
        <div className="absolute top-0 left-0 right-0 bg-black/40 px-1 py-0.5 text-[8px] text-white/70 text-center">
          {fmtDate(img.capture_date)}
        </div>
      )}
    </button>
  )
}
