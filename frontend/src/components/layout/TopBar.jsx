import { useEffect, useState } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { LogOut, Pause, Play } from 'lucide-react'
import { useAuthStore, useAnnotationStore } from '../../lib/store'
import { api } from '../../lib/api'
import { fmtDate } from '../../lib/utils'

function formatTimer(s) {
  const h = String(Math.floor(s / 3600)).padStart(2, '0')
  const m = String(Math.floor((s % 3600) / 60)).padStart(2, '0')
  const sec = String(s % 60).padStart(2, '0')
  return `${h}:${m}:${sec}`
}

export default function TopBar() {
  const user = useAuthStore((s) => s.user)
  const logout = useAuthStore((s) => s.logout)
  const { images, currentImageId } = useAnnotationStore((s) => ({
    images: s.images,
    currentImageId: s.currentImageId,
  }))
  const current = images.find((i) => i.id === currentImageId)
  const done = images.filter((i) => i.status === 'done').length
  const total = images.length

  const [sessionSec, setSessionSec] = useState(0)
  const [paused, setPaused] = useState(false)

  useEffect(() => {
    if (paused) return
    const t = setInterval(() => setSessionSec((v) => v + 1), 1000)
    return () => clearInterval(t)
  }, [paused])

  const handlePause = async () => {
    const next = !paused
    setPaused(next)
    // Auto-save draft when pausing
    if (next && currentImageId) {
      const token = useAuthStore.getState().token
      const ann = useAnnotationStore.getState().annotations[currentImageId] || {}
      api.saveDraft(currentImageId, {
        disease_labels:  ann.disease_labels  || [],
        regions:         ann.regions         || [],
        gradcam_verdict: ann.gradcam_verdict || null,
        notes_text:      ann.notes_text      || '',
      }, token).catch(() => {})
    }
  }

  const location = useLocation()
  const showPatient = location.pathname.startsWith('/annotate') && current

  return (
    <header className="relative z-10 flex h-14 flex-shrink-0 items-center border-b border-line bg-bg-elev1/60 backdrop-blur-xl">
      {/* Logo */}
      <div className="flex items-center gap-2.5 pl-5 pr-6">
        <div className="relative h-5 w-5">
          <div className="absolute inset-0 rounded-full bg-accent/20" />
          <div className="absolute inset-[3px] rounded-full bg-accent" />
          <div className="absolute inset-[6px] rounded-full bg-bg-base" />
        </div>
        <div className="flex items-baseline gap-1.5">
          <span className="display text-[15px] font-medium tracking-tight">RetinAI</span>
          <span className="mono text-[10px] uppercase tracking-widest text-ink-tertiary">hitl</span>
        </div>
      </div>

      <div className="h-5 w-px bg-line" />

      {/* Nav */}
      <nav className="flex items-center gap-0 pl-2 pr-4">
        {[
          { to: '/annotate', label: 'Annotation' },
          { to: '/queue',    label: 'File' },
          { to: '/patients', label: 'Patients' },
          ...(user?.role === 'admin' ? [{ to: '/admin', label: 'Admin' }] : []),
        ].map((n) => (
          <NavLink
            key={n.to}
            to={n.to}
            className={({ isActive }) =>
              `px-3 py-1.5 text-[13px] transition-colors ${
                isActive
                  ? 'text-ink-primary'
                  : 'text-ink-secondary hover:text-ink-primary'
              }`
            }
          >
            {({ isActive }) => (
              <span className="relative">
                {n.label}
                {isActive && (
                  <span className="absolute -bottom-[18px] left-0 right-0 h-[1.5px] bg-accent" />
                )}
              </span>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Patient metadata (annotation page only) */}
      {showPatient && (
        <>
          <div className="h-5 w-px bg-line" />
          <div className="flex items-center gap-5 pl-4 pr-4 text-[11px]">
            <MetaField label="Patient" value={current.patient_clinical_id || current.patient_id?.slice(0, 8)} mono />
            <MetaField label="Œil" value={current.eye} mono />
            <MetaField label="Modalité" value={current.modality} mono />
            <MetaField label="Date" value={fmtDate(current.capture_date)} mono />
          </div>
        </>
      )}

      <div className="flex-1" />

      {/* Session cluster */}
      <div className="flex items-center gap-2 pr-4">
        <div className="flex items-center gap-2 rounded-md border border-line bg-bg-elev1 px-2.5 py-1">
          <div className={`h-1.5 w-1.5 rounded-full ${paused ? 'bg-urgency-p3' : 'bg-urgency-p4 animate-pulse-soft'}`} />
          <span className="mono text-[11px] text-ink-secondary">Session</span>
          <span className={`mono text-[11px] ${paused ? 'text-urgency-p3' : 'text-ink-primary'}`}>
            {formatTimer(sessionSec)}
          </span>
        </div>
        <div className="flex items-center gap-2 rounded-md border border-line bg-bg-elev1 px-2.5 py-1">
          <span className="mono text-[11px] text-ink-secondary">Progrès</span>
          <span className="mono text-[11px] text-ink-primary">
            {String(done).padStart(2, '0')}/{String(total).padStart(2, '0')}
          </span>
        </div>
        <button
          onClick={handlePause}
          className={`btn flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-[11px] transition-colors ${
            paused
              ? 'border-accent/50 bg-accent/10 text-accent'
              : 'border-line bg-bg-elev1 text-ink-secondary hover:border-line-strong hover:text-ink-primary'
          }`}
        >
          {paused
            ? <Play  size={12} strokeWidth={2} />
            : <Pause size={12} strokeWidth={2} />}
          {paused ? 'Reprendre' : 'Pause'}
        </button>
      </div>

      {/* User */}
      <div className="flex items-center gap-2 border-l border-line pl-4 pr-5">
        <div className="flex h-7 w-7 items-center justify-center rounded-full bg-accent/15 text-[11px] font-medium text-accent">
          {user?.initials}
        </div>
        <div className="hidden flex-col md:flex">
          <span className="text-[12px] leading-tight">{user?.full_name || user?.username}</span>
          <span className="mono text-[9px] uppercase leading-tight tracking-widest text-ink-tertiary">
            {user?.role}
          </span>
        </div>
        <button
          onClick={logout}
          className="ml-1 text-ink-tertiary hover:text-ink-primary transition-colors"
          title="Déconnexion"
        >
          <LogOut size={14} strokeWidth={1.75} />
        </button>
      </div>
    </header>
  )
}

function MetaField({ label, value, mono: isMono }) {
  return (
    <div className="flex flex-col leading-tight">
      <span className="eyebrow text-[9px]">{label}</span>
      <span className={`${isMono ? 'mono' : ''} text-[12px] text-ink-primary`}>{value}</span>
    </div>
  )
}
