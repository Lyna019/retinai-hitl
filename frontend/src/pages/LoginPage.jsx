import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowRight } from 'lucide-react'
import { useAuthStore } from '../lib/store'

export default function LoginPage() {
  const [username, setUsername] = useState('mekki')
  const [password, setPassword] = useState('demo')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const login = useAuthStore((s) => s.login)
  const nav = useNavigate()

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await login(username, password)
      nav('/annotate')
    } catch (err) {
      setError(err.message || 'Identifiants incorrects')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="relative flex min-h-screen items-center justify-center bg-bg-base text-ink-primary">
      {/* Background flourish */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute left-1/2 top-1/2 h-[900px] w-[900px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-gradient-radial from-accent/10 via-transparent to-transparent opacity-60" style={{ background: 'radial-gradient(circle, rgba(20,227,202,0.08) 0%, transparent 60%)' }} />
        <BackgroundRetinaDecor />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.2, 0.8, 0.2, 1] }}
        className="relative w-[400px] max-w-[90vw]"
      >
        {/* Brand */}
        <div className="mb-8 flex items-center gap-3">
          <div className="relative h-9 w-9">
            <div className="absolute inset-0 rounded-full bg-accent/20" />
            <div className="absolute inset-[6px] rounded-full bg-accent" />
            <div className="absolute inset-[12px] rounded-full bg-bg-base" />
          </div>
          <div>
            <div className="display text-[22px] font-medium tracking-tight leading-none">RetinAI</div>
            <div className="mt-0.5 mono text-[10px] uppercase tracking-[0.18em] text-ink-tertiary">
              Human-in-the-loop 
            </div>
          </div>
        </div>

        <div className="mb-6">
          <h1 className="display text-[26px] font-medium leading-tight">
            Plateforme d'annotation<br /><em className="text-accent not-italic">clinique assistée.</em>
          </h1>
          <p className="mt-2 text-[13px] leading-relaxed text-ink-secondary">
            Connectez-vous pour reprendre vos sessions d'annotation de fonds d'œil.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-3 rounded-xl border border-line bg-bg-elev1/60 p-5 backdrop-blur-xl">
          <Field
            label="Identifiant"
            value={username}
            onChange={setUsername}
            autoFocus
          />
          <Field
            label="Mot de passe"
            value={password}
            onChange={setPassword}
            type="password"
          />
          {error && (
            <p className="rounded-md border border-urgency-p1/30 bg-urgency-p1/10 px-3 py-2 text-[12px] text-urgency-p1">
              {error}
            </p>
          )}
          <button
            type="submit"
            disabled={loading}
            className="btn mt-2 flex w-full items-center justify-center gap-2 rounded-lg bg-accent py-2.5 text-[13px] font-medium text-bg-base shadow-[0_0_24px_-4px_rgba(20,227,202,0.45)] transition-all hover:shadow-[0_0_32px_-4px_rgba(20,227,202,0.6)] disabled:opacity-60"
          >
            {loading ? 'Connexion…' : 'Se connecter'}
            {!loading && <ArrowRight size={14} strokeWidth={2.5} />}
          </button>
        </form>

        <div className="mt-6 flex items-center justify-between mono text-[10px] text-ink-tertiary">
          <span> Service Ophtalmologie</span>
          <span>© 2026 · RetinAI</span>
        </div>
      </motion.div>
    </div>
  )
}

function Field({ label, value, onChange, type = 'text', autoFocus }) {
  return (
    <div>
      <label className="eyebrow">{label}</label>
      <input
        type={type}
        autoFocus={autoFocus}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="mt-1.5 w-full rounded-lg border border-line bg-bg-base px-3 py-2.5 text-[13px] outline-none transition-colors focus:border-accent/60"
      />
    </div>
  )
}

function BackgroundRetinaDecor() {
  return (
    <svg className="absolute right-[-12%] top-[-18%] h-[70vh] w-[70vh] opacity-[0.07]" viewBox="0 0 400 400" fill="none">
      <circle cx="200" cy="200" r="180" stroke="#14E3CA" strokeWidth="0.5" />
      <circle cx="200" cy="200" r="140" stroke="#14E3CA" strokeWidth="0.5" />
      <circle cx="200" cy="200" r="100" stroke="#14E3CA" strokeWidth="0.5" />
      <circle cx="200" cy="200" r="60" stroke="#14E3CA" strokeWidth="0.5" />
      <circle cx="150" cy="200" r="30" stroke="#14E3CA" strokeWidth="0.5" />
      <g stroke="#14E3CA" strokeWidth="0.5" fill="none">
        <path d="M 150 200 Q 200 150 260 120 T 360 80" />
        <path d="M 150 200 Q 200 250 260 280 T 360 320" />
        <path d="M 150 200 Q 110 140 100 60" />
        <path d="M 150 200 Q 110 260 100 340" />
      </g>
    </svg>
  )
}
