/**
 * Admin dashboard — tabbed: Stats | Catalog | Proposals | Doctors | Model | Audit
 */
import { useState, useEffect, useCallback } from 'react'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts'
import {
  Target, Activity, AlertOctagon, Users, BookOpen,
  CheckCircle2, XCircle, Eye, Plus, Trash2, Edit3,
  ChevronRight, RotateCcw, Settings, FileText, Loader2,
} from 'lucide-react'
import { useAuthStore } from '../lib/store'
import { api } from '../lib/api'
import { fmtDate, fmtDatetime } from '../lib/utils'

const CHART_COLORS  = ['#14E3CA', '#5E9CFF', '#BF5AF2', '#FF9F0A', '#FF6482', '#64D2FF']
const URGENCY_COLORS = { P1: '#FF453A', P2: '#FF9F0A', P3: '#FFD60A', P4: '#30D158' }

const TABS = [
  { id: 'stats',     label: 'Tableau de bord', icon: Target },
  { id: 'catalog',   label: 'Catalogue',       icon: BookOpen },
  { id: 'proposals', label: 'Propositions',     icon: CheckCircle2 },
  { id: 'doctors',   label: 'Médecins',         icon: Users },
  { id: 'model',     label: 'Modèle & AL',      icon: Settings },
  { id: 'audit',     label: 'Journal d\'audit', icon: FileText },
]

export default function AdminPage() {
  const [tab, setTab] = useState('stats')

  return (
    <div className="flex h-full flex-col bg-bg-base">
      {/* Tab bar */}
      <div className="flex items-center gap-0 border-b border-line px-6 pt-1 flex-shrink-0 overflow-x-auto">
        {TABS.map((t) => {
          const Icon = t.icon
          return (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`flex items-center gap-1.5 px-4 py-2.5 text-[12px] transition-colors whitespace-nowrap ${
                tab === t.id
                  ? 'border-b-[1.5px] border-accent text-ink-primary'
                  : 'text-ink-secondary hover:text-ink-primary'
              }`}
            >
              <Icon size={13} />
              {t.label}
            </button>
          )
        })}
      </div>

      <div className="flex-1 overflow-y-auto">
        {tab === 'stats'     && <StatsTab />}
        {tab === 'catalog'   && <CatalogTab />}
        {tab === 'proposals' && <ProposalsTab />}
        {tab === 'doctors'   && <DoctorsTab />}
        {tab === 'model'     && <ModelTab />}
        {tab === 'audit'     && <AuditTab />}
      </div>
    </div>
  )
}


// ─── Stats tab ───────────────────────────────────────────────────────────────

function StatsTab() {
  const token = useAuthStore((s) => s.token)
  const [data, setData] = useState(null)

  useEffect(() => {
    if (!token) return
    Promise.all([
      api.getStats('progress', token),
      api.getStats('disease-distribution', token),
      api.getStats('urgency-distribution', token),
      api.getStats('gradcam-validation', token),
    ]).then(([progress, diseases, urgency, gradcam]) => {
      setData({ progress, diseases, urgency, gradcam })
    }).catch(console.error)
  }, [token])

  if (!data) return <LoadingScreen />
  const pct = data.progress.total > 0
    ? Math.round((data.progress.done / data.progress.total) * 100)
    : 0

  return (
    <div className="px-8 py-6 space-y-6 max-w-[1280px] mx-auto">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Card label="Progression" value={`${pct}%`} sub={`${data.progress.done}/${data.progress.total}`} color="#14E3CA" />
        <Card label="En cours" value={data.progress.in_progress} color="#5E9CFF" />
        <Card label="En attente" value={data.progress.pending} color="#FF9F0A" />
        <Card label="Urgences P1-P2" value={data.progress.urgent} color="#FF453A" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <ChartBox title="Distribution des pathologies">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={data.diseases.slice(0, 12)}>
              <XAxis dataKey="name" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {data.diseases.slice(0, 12).map((_, i) => (
                  <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartBox>

        <ChartBox title="Distribution des urgences">
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={data.urgency} dataKey="count" nameKey="level" outerRadius={70} label={({ level }) => level}>
                {data.urgency.map((u) => (
                  <Cell key={u.level} fill={URGENCY_COLORS[u.level] || '#888'} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </ChartBox>
      </div>
    </div>
  )
}


// ─── Catalog tab ─────────────────────────────────────────────────────────────

function CatalogTab() {
  const token = useAuthStore((s) => s.token)
  const [mechanisms, setMechanisms] = useState([])
  const [systemicDiseases, setSystemicDiseases] = useState([])
  const [editingDisease, setEditingDisease] = useState(null)
  const [newDisease, setNewDisease] = useState(null)
  const [sdForm, setSdForm] = useState(null)

  const refresh = useCallback(() => {
    api.listMechanisms(token).then(setMechanisms).catch(console.error)
    api.listSystemicDiseases(token).then(setSystemicDiseases).catch(console.error)
  }, [token])

  useEffect(() => { refresh() }, [refresh])

  const saveDisease = async (code, payload, isNew) => {
    try {
      if (isNew) await api.createDisease(payload, token)
      else       await api.updateDisease(code, payload, token)
      setEditingDisease(null)
      setNewDisease(null)
      refresh()
    } catch (e) { alert(e.message) }
  }

  const deleteDisease = async (code) => {
    if (!confirm(`Supprimer la pathologie ${code} ?`)) return
    await api.deleteDisease(code, token).catch((e) => alert(e.message))
    refresh()
  }

  return (
    <div className="px-8 py-6 space-y-8 max-w-[1200px] mx-auto">
      {/* Disease catalog by mechanism */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="display text-[16px] font-semibold">Catalogue des pathologies</h2>
          <button
            onClick={() => setNewDisease({ code: '', name_fr: '', mechanism_code: 'VASC', is_gradable: false })}
            className="btn flex items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-[12px] text-ink-secondary hover:text-accent hover:border-accent/40"
          >
            <Plus size={13} /> Nouvelle pathologie
          </button>
        </div>

        {mechanisms.map((mech) => (
          <div key={mech.code} className="mb-4 rounded-xl border border-line overflow-hidden">
            <div className="flex items-center justify-between bg-bg-elev1 px-4 py-2.5">
              <div>
                <span className="font-medium text-[13px] text-ink-primary">{mech.name_fr}</span>
                <span className="ml-2 mono text-[10px] text-ink-tertiary">{mech.code}</span>
              </div>
              {mech.description && (
                <span className="text-[11px] text-ink-tertiary">{mech.description}</span>
              )}
            </div>
            <table className="w-full text-[12px]">
              <thead>
                <tr className="border-b border-line text-ink-tertiary">
                  <th className="px-4 py-2 text-left font-normal">Code</th>
                  <th className="px-4 py-2 text-left font-normal">Nom</th>
                  <th className="px-4 py-2 text-left font-normal">Grades</th>
                  <th className="px-4 py-2 text-left font-normal">Urgence</th>
                  <th className="px-4 py-2" />
                </tr>
              </thead>
              <tbody>
                {mech.diseases?.map((d) => (
                  <tr key={d.code} className="border-b border-line/50 hover:bg-bg-elev1/50">
                    <td className="px-4 py-2 mono text-[11px] text-ink-tertiary">{d.code}</td>
                    <td className="px-4 py-2">{d.name_fr}</td>
                    <td className="px-4 py-2">
                      {d.is_gradable && d.grades ? (
                        <div className="flex flex-wrap gap-0.5">
                          {d.grades.map((g) => (
                            <span key={g} className="rounded px-1.5 py-0.5 bg-bg-elev2 border border-line text-[10px]">
                              {d.grade_labels?.[g] || g}
                            </span>
                          ))}
                        </div>
                      ) : <span className="text-ink-tertiary italic">—</span>}
                    </td>
                    <td className="px-4 py-2">
                      {d.urgency_override
                        ? <span style={{ color: URGENCY_COLORS[d.urgency_override] }} className="mono font-medium">{d.urgency_override}</span>
                        : <span className="text-ink-tertiary">—</span>
                      }
                    </td>
                    <td className="px-4 py-2">
                      <div className="flex items-center gap-1.5 justify-end">
                        <button onClick={() => setEditingDisease(d)} className="text-ink-tertiary hover:text-accent">
                          <Edit3 size={12} />
                        </button>
                        <button onClick={() => deleteDisease(d.code)} className="text-ink-tertiary hover:text-urgency-p1">
                          <Trash2 size={12} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>

      {/* Systemic diseases */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="display text-[15px] font-semibold">Maladies systémiques</h2>
          <button
            onClick={() => setSdForm({ name_fr: '', category: '' })}
            className="btn flex items-center gap-1 rounded border border-line px-2.5 py-1 text-[11px] text-ink-secondary hover:text-accent hover:border-accent/40"
          >
            <Plus size={11} /> Ajouter
          </button>
        </div>
        <div className="flex flex-wrap gap-2">
          {systemicDiseases.map((sd) => (
            <div key={sd.id} className="flex items-center gap-1.5 rounded-lg border border-line bg-bg-elev1 px-3 py-1.5 text-[12px]">
              <span>{sd.name_fr}</span>
              {sd.category && <span className="text-ink-tertiary text-[10px]">· {sd.category}</span>}
              <button
                onClick={() => api.deleteSystemicDisease(sd.id, token).then(refresh)}
                className="text-ink-tertiary hover:text-urgency-p1 ml-1"
              >
                <XCircle size={12} />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Edit/Create disease modal */}
      {(editingDisease || newDisease) && (
        <DiseaseFormModal
          disease={editingDisease || newDisease}
          isNew={!!newDisease}
          mechanisms={mechanisms}
          onSave={saveDisease}
          onClose={() => { setEditingDisease(null); setNewDisease(null) }}
        />
      )}

      {/* Add systemic disease modal */}
      {sdForm && (
        <SimpleModal
          title="Nouvelle maladie systémique"
          fields={[
            { key: 'name_fr', label: 'Nom', placeholder: 'Ex: Maladie de Crohn' },
            { key: 'category', label: 'Catégorie', placeholder: 'Ex: Inflammatoire' },
          ]}
          values={sdForm}
          onChange={setSdForm}
          onSave={async () => {
            await api.createSystemicDisease(sdForm, token)
            setSdForm(null)
            refresh()
          }}
          onClose={() => setSdForm(null)}
        />
      )}
    </div>
  )
}


// ─── Proposals tab ───────────────────────────────────────────────────────────

function ProposalsTab() {
  const token = useAuthStore((s) => s.token)
  const [proposals, setProposals] = useState([])
  const [selected, setSelected] = useState(null)
  const [imageUrl, setImageUrl] = useState(null)
  const [mechanisms, setMechanisms] = useState([])
  const [form, setForm] = useState(null)

  const refresh = useCallback(() => {
    api.listProposals(token).then(setProposals).catch(console.error)
    api.listMechanisms(token).then(setMechanisms).catch(console.error)
  }, [token])

  useEffect(() => { refresh() }, [refresh])

  const openProposal = async (p) => {
    setSelected(p)
    setForm({
      final_name: p.proposed_name,
      final_description: p.proposed_description || '',
      mechanism_code: p.suspected_mechanism || 'VASC',
      is_gradable: p.is_gradable,
      grades_json: p.proposed_grades_json || [],
      grade_labels_json: {},
      urgency_override: '',
      admin_notes: '',
    })
    if (p.image_id) {
      const r = await api.getProposalImageUrl(p.id, token).catch(() => null)
      setImageUrl(r?.image_url || null)
    } else {
      setImageUrl(null)
    }
  }

  const approve = async () => {
    if (!selected) return
    await api.approveProposal(selected.id, form, token)
    setSelected(null); setImageUrl(null)
    refresh()
  }

  const reject = async () => {
    if (!selected) return
    if (!confirm('Rejeter cette proposition ?')) return
    await api.rejectProposal(selected.id, token)
    setSelected(null); setImageUrl(null)
    refresh()
  }

  const pending   = proposals.filter((p) => p.status === 'pending')
  const reviewed  = proposals.filter((p) => p.status !== 'pending')

  return (
    <div className="flex h-full">
      {/* List */}
      <div className="w-[320px] flex-shrink-0 border-r border-line overflow-y-auto">
        <div className="px-4 py-3 border-b border-line">
          <span className="eyebrow">En attente ({pending.length})</span>
        </div>
        {pending.map((p) => (
          <ProposalRow key={p.id} p={p} selected={selected?.id === p.id} onClick={() => openProposal(p)} />
        ))}
        {pending.length === 0 && (
          <p className="px-4 py-8 text-center text-[12px] text-ink-tertiary italic">Aucune proposition en attente.</p>
        )}
        <div className="px-4 py-3 border-t border-b border-line mt-2">
          <span className="eyebrow">Traitées ({reviewed.length})</span>
        </div>
        {reviewed.map((p) => (
          <ProposalRow key={p.id} p={p} selected={selected?.id === p.id} onClick={() => openProposal(p)} />
        ))}
      </div>

      {/* Detail */}
      {selected && form ? (
        <div className="flex-1 overflow-y-auto px-8 py-6 space-y-5">
          <div className="flex items-center justify-between">
            <h2 className="display text-[16px] font-semibold">Révision de la proposition</h2>
            <span className={`rounded-full px-3 py-0.5 text-[11px] font-medium border ${
              selected.status === 'pending' ? 'border-yellow-400/30 text-yellow-400 bg-yellow-400/10'
              : selected.status === 'approved' ? 'border-green-400/30 text-green-400 bg-green-400/10'
              : 'border-red-400/30 text-red-400 bg-red-400/10'
            }`}>
              {selected.status}
            </span>
          </div>

          {/* Image preview */}
          {imageUrl && (
            <div>
              <div className="eyebrow mb-2">Image source</div>
              <div className="rounded-xl border border-line overflow-hidden bg-bg-elev1 max-w-sm">
                <img src={imageUrl} alt="fundus" className="w-full object-contain max-h-64" />
              </div>
            </div>
          )}

          {/* Editable form */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="eyebrow">Nom final</label>
              <input
                value={form.final_name}
                onChange={(e) => setForm((f) => ({ ...f, final_name: e.target.value }))}
                disabled={selected.status !== 'pending'}
                className="mt-1.5 w-full rounded border border-line bg-bg-base px-3 py-2 text-[12px] outline-none focus:border-accent/60 disabled:opacity-60"
              />
            </div>
            <div>
              <label className="eyebrow">Mécanisme</label>
              <select
                value={form.mechanism_code}
                onChange={(e) => setForm((f) => ({ ...f, mechanism_code: e.target.value }))}
                disabled={selected.status !== 'pending'}
                className="mt-1.5 w-full rounded border border-line bg-bg-base px-3 py-2 text-[12px] outline-none disabled:opacity-60"
              >
                {mechanisms.map((m) => <option key={m.code} value={m.code}>{m.name_fr}</option>)}
              </select>
            </div>
            <div className="col-span-2">
              <label className="eyebrow">Description clinique</label>
              <textarea
                value={form.final_description}
                onChange={(e) => setForm((f) => ({ ...f, final_description: e.target.value }))}
                disabled={selected.status !== 'pending'}
                rows={2}
                className="mt-1.5 w-full resize-none rounded border border-line bg-bg-base px-3 py-2 text-[12px] outline-none focus:border-accent/60 disabled:opacity-60"
              />
            </div>
            <div>
              <label className="eyebrow">Grades (séparés par virgule)</label>
              <input
                value={(form.grades_json || []).join(',')}
                onChange={(e) => setForm((f) => ({ ...f, grades_json: e.target.value.split(',').map((s) => s.trim()).filter(Boolean) }))}
                disabled={selected.status !== 'pending'}
                placeholder="Ex: léger,modéré,sévère"
                className="mt-1.5 w-full rounded border border-line bg-bg-base px-3 py-2 text-[12px] outline-none focus:border-accent/60 disabled:opacity-60"
              />
            </div>
            <div>
              <label className="eyebrow">Urgence override</label>
              <select
                value={form.urgency_override}
                onChange={(e) => setForm((f) => ({ ...f, urgency_override: e.target.value }))}
                disabled={selected.status !== 'pending'}
                className="mt-1.5 w-full rounded border border-line bg-bg-base px-3 py-2 text-[12px] outline-none disabled:opacity-60"
              >
                <option value="">Aucune</option>
                <option value="P1">P1 — Urgence absolue</option>
                <option value="P2">P2 — Même jour</option>
                <option value="P3">P3 — Sous 72h</option>
              </select>
            </div>
            <div className="col-span-2">
              <label className="eyebrow">Notes admin</label>
              <textarea
                value={form.admin_notes}
                onChange={(e) => setForm((f) => ({ ...f, admin_notes: e.target.value }))}
                disabled={selected.status !== 'pending'}
                rows={2}
                className="mt-1.5 w-full resize-none rounded border border-line bg-bg-base px-3 py-2 text-[12px] outline-none focus:border-accent/60 disabled:opacity-60"
                placeholder="Commentaires internes…"
              />
            </div>
          </div>

          {selected.status === 'pending' && (
            <div className="flex gap-3">
              <button
                onClick={approve}
                className="btn flex items-center gap-2 rounded-lg bg-accent px-5 py-2 text-[13px] font-medium text-bg-base"
              >
                <CheckCircle2 size={14} /> Approuver et créer la pathologie
              </button>
              <button
                onClick={reject}
                className="btn flex items-center gap-2 rounded-lg border border-urgency-p1/40 px-4 py-2 text-[13px] text-urgency-p1"
              >
                <XCircle size={14} /> Rejeter
              </button>
            </div>
          )}
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center text-ink-tertiary text-[13px]">
          Sélectionnez une proposition pour la réviser.
        </div>
      )}
    </div>
  )
}

function ProposalRow({ p, selected, onClick }) {
  const statusColor = {
    pending:  'bg-yellow-400',
    approved: 'bg-green-400',
    rejected: 'bg-red-400',
  }
  return (
    <button
      onClick={onClick}
      className={`flex w-full items-center gap-3 px-4 py-3 text-left border-b border-line/50 hover:bg-bg-elev1 transition-colors ${selected ? 'bg-bg-elev1' : ''}`}
    >
      <div className={`h-2 w-2 rounded-full flex-shrink-0 ${statusColor[p.status] || 'bg-ink-tertiary'}`} />
      <div className="flex-1 min-w-0">
        <div className="text-[12px] font-medium text-ink-primary truncate">{p.proposed_name}</div>
        <div className="text-[10px] text-ink-tertiary">{fmtDatetime(p.created_at)}</div>
      </div>
      {p.image_id && <Eye size={11} className="text-ink-tertiary flex-shrink-0" />}
    </button>
  )
}


// ─── Doctors tab ─────────────────────────────────────────────────────────────

function DoctorsTab() {
  const token = useAuthStore((s) => s.token)
  const [doctors, setDoctors] = useState([])
  const [showForm, setShowForm] = useState(false)
  const [form, setForm] = useState({ username: '', email: '', full_name: '', password: '', role: 'doctor' })

  const refresh = () => api.listDoctors(token).then(setDoctors).catch(console.error)
  useEffect(() => { refresh() }, [token])

  const createDoctor = async () => {
    try {
      await api.createDoctor(form, token)
      setShowForm(false)
      setForm({ username: '', email: '', full_name: '', password: '', role: 'doctor' })
      refresh()
    } catch (e) { alert(e.message) }
  }

  return (
    <div className="px-8 py-6 max-w-[900px] mx-auto space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="display text-[16px] font-semibold">Comptes médecins</h2>
        <button
          onClick={() => setShowForm(!showForm)}
          className="btn flex items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-[12px] text-ink-secondary hover:text-accent hover:border-accent/40"
        >
          <Plus size={13} /> Nouveau compte
        </button>
      </div>

      {showForm && (
        <div className="rounded-xl border border-line bg-bg-elev1 p-5 grid grid-cols-2 gap-3">
          {[
            { key: 'username', label: 'Identifiant', placeholder: 'dr.ahmed' },
            { key: 'password', label: 'Mot de passe', placeholder: '••••••••', type: 'password' },
            { key: 'full_name', label: 'Nom complet', placeholder: 'Dr. Ahmed Benali' },
            { key: 'email', label: 'Email', placeholder: 'ahmed@chu.dz' },
          ].map(({ key, label, placeholder, type = 'text' }) => (
            <div key={key}>
              <label className="eyebrow">{label}</label>
              <input
                type={type}
                value={form[key]}
                onChange={(e) => setForm((f) => ({ ...f, [key]: e.target.value }))}
                placeholder={placeholder}
                className="mt-1.5 w-full rounded border border-line bg-bg-base px-3 py-2 text-[12px] outline-none focus:border-accent/60"
              />
            </div>
          ))}
          <div className="col-span-2 flex gap-3">
            <select
              value={form.role}
              onChange={(e) => setForm((f) => ({ ...f, role: e.target.value }))}
              className="rounded border border-line bg-bg-base px-3 py-2 text-[12px] outline-none"
            >
              <option value="doctor">doctor</option>
              <option value="viewer">viewer</option>
            </select>
            <button onClick={createDoctor} className="btn rounded-lg bg-accent px-4 py-2 text-[12px] font-medium text-bg-base">
              Créer le compte
            </button>
            <button onClick={() => setShowForm(false)} className="btn rounded-lg border border-line px-4 py-2 text-[12px] text-ink-secondary">
              Annuler
            </button>
          </div>
        </div>
      )}

      <table className="w-full text-[12px]">
        <thead>
          <tr className="text-ink-tertiary border-b border-line">
            <th className="text-left py-2 px-3 font-normal">Nom</th>
            <th className="text-left py-2 px-3 font-normal">Identifiant</th>
            <th className="text-left py-2 px-3 font-normal">Email</th>
            <th className="text-left py-2 px-3 font-normal">Rôle</th>
            <th className="text-left py-2 px-3 font-normal">Statut</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {doctors.map((d) => (
            <tr key={d.id} className="border-b border-line/50">
              <td className="py-2.5 px-3 font-medium">{d.full_name || d.username}</td>
              <td className="py-2.5 px-3 mono text-ink-tertiary">{d.username}</td>
              <td className="py-2.5 px-3 text-ink-secondary">{d.email || '—'}</td>
              <td className="py-2.5 px-3"><span className="mono text-[10px] bg-bg-elev2 border border-line rounded px-1.5 py-0.5">{d.role}</span></td>
              <td className="py-2.5 px-3">
                <span className={`text-[11px] ${d.is_active ? 'text-green-400' : 'text-ink-tertiary line-through'}`}>
                  {d.is_active ? 'Actif' : 'Inactif'}
                </span>
              </td>
              <td className="py-2.5 px-3">
                <button
                  onClick={() => api.toggleDoctor(d.id, token).then(refresh)}
                  className="text-[11px] text-ink-tertiary hover:text-ink-primary border border-line rounded px-2 py-0.5"
                >
                  {d.is_active ? 'Désactiver' : 'Réactiver'}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}


// ─── Model & Active Learning tab ──────────────────────────────────────────────

function ModelTab() {
  const token = useAuthStore((s) => s.token)
  const [versions, setVersions] = useState([])
  const [alConfig, setAlConfig] = useState(null)
  const [alForm, setAlForm] = useState(null)
  const [newVersion, setNewVersion] = useState(null)

  const refresh = useCallback(() => {
    api.listModelVersions(token).then(setVersions).catch(console.error)
    api.getALConfig(token).then((c) => { setAlConfig(c); setAlForm({ ...c }) }).catch(console.error)
  }, [token])

  useEffect(() => { refresh() }, [refresh])

  const saveAL = async () => {
    await api.updateALConfig(alForm, token)
    refresh()
  }

  const activate = async (id) => {
    await api.activateModelVersion(id, token)
    refresh()
  }

  const registerVersion = async () => {
    await api.registerModelVersion(newVersion, token)
    setNewVersion(null)
    refresh()
  }

  return (
    <div className="px-8 py-6 max-w-[1000px] mx-auto space-y-8">
      {/* Model versions */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="display text-[16px] font-semibold">Versions du modèle</h2>
          <button
            onClick={() => setNewVersion({ version_tag: '', description: '', checkpoint_path: '' })}
            className="btn flex items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-[12px] text-ink-secondary hover:text-accent hover:border-accent/40"
          >
            <Plus size={13} /> Enregistrer une version
          </button>
        </div>

        {newVersion && (
          <div className="rounded-xl border border-line bg-bg-elev1 p-4 mb-4 grid grid-cols-2 gap-3">
            {[
              { key: 'version_tag', label: 'Tag de version', placeholder: 'v1.0.0' },
              { key: 'checkpoint_path', label: 'Chemin checkpoint', placeholder: '/checkpoints/retinai.pth' },
              { key: 'description', label: 'Description', placeholder: 'Première version de production' },
            ].map(({ key, label, placeholder }) => (
              <div key={key} className={key === 'description' ? 'col-span-2' : ''}>
                <label className="eyebrow">{label}</label>
                <input
                  value={newVersion[key]}
                  onChange={(e) => setNewVersion((v) => ({ ...v, [key]: e.target.value }))}
                  placeholder={placeholder}
                  className="mt-1.5 w-full rounded border border-line bg-bg-base px-3 py-2 text-[12px] outline-none focus:border-accent/60"
                />
              </div>
            ))}
            <div className="col-span-2 flex gap-2">
              <button onClick={registerVersion} className="btn rounded-lg bg-accent px-4 py-2 text-[12px] font-medium text-bg-base">Enregistrer</button>
              <button onClick={() => setNewVersion(null)} className="btn rounded-lg border border-line px-4 py-2 text-[12px] text-ink-secondary">Annuler</button>
            </div>
          </div>
        )}

        <table className="w-full text-[12px]">
          <thead>
            <tr className="text-ink-tertiary border-b border-line">
              <th className="text-left py-2 px-3 font-normal">Version</th>
              <th className="text-left py-2 px-3 font-normal">Description</th>
              <th className="text-left py-2 px-3 font-normal">Enregistré le</th>
              <th className="text-left py-2 px-3 font-normal">Statut</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {versions.map((v) => (
              <tr key={v.id} className="border-b border-line/50">
                <td className="py-2.5 px-3 mono font-medium">{v.version_tag}</td>
                <td className="py-2.5 px-3 text-ink-secondary">{v.description || '—'}</td>
                <td className="py-2.5 px-3 text-ink-tertiary">{fmtDatetime(v.created_at)}</td>
                <td className="py-2.5 px-3">
                  {v.is_active
                    ? <span className="text-accent font-medium text-[11px]">● Actif</span>
                    : <span className="text-ink-tertiary text-[11px]">Inactif</span>
                  }
                </td>
                <td className="py-2.5 px-3">
                  {!v.is_active && (
                    <button
                      onClick={() => activate(v.id)}
                      className="flex items-center gap-1 text-[11px] text-ink-tertiary hover:text-accent border border-line rounded px-2 py-0.5"
                    >
                      <RotateCcw size={10} /> Activer
                    </button>
                  )}
                </td>
              </tr>
            ))}
            {versions.length === 0 && (
              <tr><td colSpan={5} className="py-8 text-center text-ink-tertiary italic">Aucune version enregistrée.</td></tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Active learning config */}
      {alForm && (
        <div>
          <h2 className="display text-[16px] font-semibold mb-4">Configuration — Apprentissage actif</h2>
          <div className="rounded-xl border border-line bg-bg-elev1 p-5 space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="eyebrow">Seuil d'incertitude</label>
                <input
                  type="number" step="0.05" min="0" max="1"
                  value={alForm.uncertainty_threshold}
                  onChange={(e) => setAlForm((f) => ({ ...f, uncertainty_threshold: +e.target.value }))}
                  className="mt-1.5 w-full rounded border border-line bg-bg-base px-3 py-2 text-[12px] outline-none focus:border-accent/60"
                />
                <p className="mt-1 text-[10px] text-ink-tertiary">Images au-dessus de ce seuil sont prioritaires dans la file.</p>
              </div>
              <div>
                <label className="eyebrow">Échantillons par cycle</label>
                <input
                  type="number" min="10"
                  value={alForm.n_samples_per_cycle}
                  onChange={(e) => setAlForm((f) => ({ ...f, n_samples_per_cycle: +e.target.value }))}
                  className="mt-1.5 w-full rounded border border-line bg-bg-base px-3 py-2 text-[12px] outline-none focus:border-accent/60"
                />
                <p className="mt-1 text-[10px] text-ink-tertiary">Nombre d'annotations avant de déclencher le ré-entraînement.</p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={() => setAlForm((f) => ({ ...f, auto_retrain: !f.auto_retrain }))}
                className={`relative h-5 w-9 rounded-full transition-colors ${alForm.auto_retrain ? 'bg-accent' : 'bg-bg-elev3'}`}
              >
                <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-transform ${alForm.auto_retrain ? 'translate-x-[18px]' : 'translate-x-0.5'}`} />
              </button>
              <label className="text-[12px] text-ink-secondary">
                Ré-entraînement automatique
                {!alForm.auto_retrain && <span className="ml-2 text-ink-tertiary italic">(désactivé pour plus de contrôle)</span>}
              </label>
            </div>

            {alConfig?.last_retrain_at && (
              <p className="text-[11px] text-ink-tertiary">
                Dernier ré-entraînement : {fmtDatetime(alConfig.last_retrain_at)}
              </p>
            )}

            <button onClick={saveAL} className="btn rounded-lg bg-accent px-5 py-2 text-[12px] font-medium text-bg-base">
              Enregistrer la configuration
            </button>
          </div>
        </div>
      )}
    </div>
  )
}


// ─── Audit log tab ───────────────────────────────────────────────────────────

function AuditTab() {
  const token = useAuthStore((s) => s.token)
  const [logs, setLogs] = useState([])
  const [filter, setFilter] = useState('')

  useEffect(() => {
    api.getAuditLog(token, { limit: 300 }).then(setLogs).catch(console.error)
  }, [token])

  const filtered = logs.filter((l) =>
    !filter || l.action?.includes(filter) || l.entity_type?.includes(filter) || l.detail?.toLowerCase().includes(filter.toLowerCase())
  )

  const actionColor = (action) => {
    if (action.includes('delete') || action.includes('reject')) return 'text-urgency-p1'
    if (action.includes('create') || action.includes('approve')) return 'text-green-400'
    if (action.includes('update') || action.includes('edit'))    return 'text-accent'
    return 'text-ink-secondary'
  }

  return (
    <div className="px-8 py-6 max-w-[1100px] mx-auto">
      <div className="flex items-center justify-between mb-4">
        <h2 className="display text-[16px] font-semibold">Journal d'audit</h2>
        <input
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filtrer par action, entité, détail…"
          className="rounded border border-line bg-bg-elev1 px-3 py-1.5 text-[12px] outline-none w-60"
        />
      </div>

      <table className="w-full text-[11px]">
        <thead>
          <tr className="text-ink-tertiary border-b border-line">
            <th className="text-left py-2 px-3 font-normal">Date</th>
            <th className="text-left py-2 px-3 font-normal">Action</th>
            <th className="text-left py-2 px-3 font-normal">Entité</th>
            <th className="text-left py-2 px-3 font-normal">Détail</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map((l) => (
            <tr key={l.id} className="border-b border-line/30 hover:bg-bg-elev1/50">
              <td className="py-2 px-3 mono text-ink-tertiary whitespace-nowrap">
                {fmtDatetime(l.created_at)}
              </td>
              <td className={`py-2 px-3 mono font-medium ${actionColor(l.action)}`}>{l.action}</td>
              <td className="py-2 px-3 text-ink-tertiary">{l.entity_type}{l.entity_id ? ` · ${l.entity_id.slice(0, 12)}…` : ''}</td>
              <td className="py-2 px-3 text-ink-secondary max-w-[400px] truncate">{l.detail || '—'}</td>
            </tr>
          ))}
          {filtered.length === 0 && (
            <tr><td colSpan={4} className="py-8 text-center text-ink-tertiary italic">Aucun log.</td></tr>
          )}
        </tbody>
      </table>
    </div>
  )
}


// ─── Shared helpers ──────────────────────────────────────────────────────────

function DiseaseFormModal({ disease, isNew, mechanisms, onSave, onClose }) {
  const [form, setForm] = useState({
    code: disease.code || '',
    name_fr: disease.name_fr || '',
    description: disease.description || '',
    mechanism_code: disease.mechanism_code || 'VASC',
    is_gradable: disease.is_gradable || false,
    grades_json: disease.grades ? [...disease.grades] : [],
    grade_labels_json: disease.grade_labels || {},
    urgency_override: disease.urgency_override || '',
  })

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div onClick={(e) => e.stopPropagation()} className="w-[600px] max-w-[96vw] rounded-xl border border-line bg-bg-elev1 shadow-2xl overflow-y-auto max-h-[90vh]">
        <div className="border-b border-line px-5 py-3">
          <div className="display text-[15px]">{isNew ? 'Nouvelle pathologie' : `Modifier ${disease.code}`}</div>
        </div>
        <div className="p-5 grid grid-cols-2 gap-4">
          <div>
            <label className="eyebrow">Code</label>
            <input
              value={form.code}
              onChange={(e) => setForm((f) => ({ ...f, code: e.target.value.toUpperCase() }))}
              disabled={!isNew}
              className="mt-1.5 w-full rounded border border-line bg-bg-base px-3 py-2 text-[12px] outline-none focus:border-accent/60 disabled:opacity-60"
            />
          </div>
          <div>
            <label className="eyebrow">Nom (FR)</label>
            <input value={form.name_fr} onChange={(e) => setForm((f) => ({ ...f, name_fr: e.target.value }))}
              className="mt-1.5 w-full rounded border border-line bg-bg-base px-3 py-2 text-[12px] outline-none focus:border-accent/60" />
          </div>
          <div className="col-span-2">
            <label className="eyebrow">Description clinique</label>
            <textarea value={form.description} onChange={(e) => setForm((f) => ({ ...f, description: e.target.value }))}
              rows={2} className="mt-1.5 w-full resize-none rounded border border-line bg-bg-base px-3 py-2 text-[12px] outline-none focus:border-accent/60" />
          </div>
          <div>
            <label className="eyebrow">Mécanisme</label>
            <select value={form.mechanism_code} onChange={(e) => setForm((f) => ({ ...f, mechanism_code: e.target.value }))}
              className="mt-1.5 w-full rounded border border-line bg-bg-base px-3 py-2 text-[12px] outline-none">
              {mechanisms.map((m) => <option key={m.code} value={m.code}>{m.name_fr}</option>)}
            </select>
          </div>
          <div>
            <label className="eyebrow">Urgence</label>
            <select value={form.urgency_override} onChange={(e) => setForm((f) => ({ ...f, urgency_override: e.target.value }))}
              className="mt-1.5 w-full rounded border border-line bg-bg-base px-3 py-2 text-[12px] outline-none">
              <option value="">Aucune</option>
              <option value="P1">P1</option><option value="P2">P2</option><option value="P3">P3</option>
            </select>
          </div>
          <div className="col-span-2">
            <div className="flex items-center gap-3 mb-2">
              <button onClick={() => setForm((f) => ({ ...f, is_gradable: !f.is_gradable }))}
                className={`relative h-5 w-9 rounded-full transition-colors ${form.is_gradable ? 'bg-accent' : 'bg-bg-elev3'}`}>
                <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-transform ${form.is_gradable ? 'translate-x-[18px]' : 'translate-x-0.5'}`} />
              </button>
              <label className="text-[12px] text-ink-secondary">Pathologie gradable</label>
            </div>
            {form.is_gradable && (
              <div className="space-y-2">
                <div>
                  <label className="eyebrow">Valeurs des grades (virgule)</label>
                  <input
                    value={form.grades_json.join(',')}
                    onChange={(e) => {
                      const grades = e.target.value.split(',').map((s) => s.trim()).filter(Boolean)
                      setForm((f) => ({ ...f, grades_json: grades }))
                    }}
                    placeholder="0,1,2,3,4"
                    className="mt-1 w-full rounded border border-line bg-bg-base px-3 py-2 text-[12px] outline-none focus:border-accent/60"
                  />
                </div>
                {form.grades_json.length > 0 && (
                  <div>
                    <label className="eyebrow">Labels des grades</label>
                    <div className="mt-1 space-y-1">
                      {form.grades_json.map((g) => (
                        <div key={g} className="flex items-center gap-2">
                          <span className="mono text-[11px] w-8 text-ink-tertiary">{g}</span>
                          <input
                            value={form.grade_labels_json[g] || ''}
                            onChange={(e) => setForm((f) => ({ ...f, grade_labels_json: { ...f.grade_labels_json, [g]: e.target.value } }))}
                            placeholder={`Label pour ${g}`}
                            className="flex-1 rounded border border-line bg-bg-base px-2 py-1 text-[11px] outline-none focus:border-accent/60"
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
        <div className="flex justify-end gap-2 border-t border-line px-5 py-3">
          <button onClick={onClose} className="btn rounded-lg border border-line px-4 py-2 text-[12px] text-ink-secondary">Annuler</button>
          <button
            onClick={() => onSave(form.code, form, isNew)}
            className="btn rounded-lg bg-accent px-4 py-2 text-[12px] font-medium text-bg-base"
          >
            {isNew ? 'Créer' : 'Enregistrer'}
          </button>
        </div>
      </div>
    </div>
  )
}

function SimpleModal({ title, fields, values, onChange, onSave, onClose }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
      <div onClick={(e) => e.stopPropagation()} className="w-[400px] rounded-xl border border-line bg-bg-elev1 shadow-2xl p-5 space-y-3">
        <div className="display text-[14px] mb-2">{title}</div>
        {fields.map(({ key, label, placeholder }) => (
          <div key={key}>
            <label className="eyebrow">{label}</label>
            <input value={values[key]} onChange={(e) => onChange((v) => ({ ...v, [key]: e.target.value }))}
              placeholder={placeholder}
              className="mt-1.5 w-full rounded border border-line bg-bg-base px-3 py-2 text-[12px] outline-none focus:border-accent/60" />
          </div>
        ))}
        <div className="flex justify-end gap-2 pt-1">
          <button onClick={onClose} className="btn rounded border border-line px-3 py-1.5 text-[12px] text-ink-secondary">Annuler</button>
          <button onClick={onSave} className="btn rounded bg-accent px-3 py-1.5 text-[12px] font-medium text-bg-base">Ajouter</button>
        </div>
      </div>
    </div>
  )
}

function Card({ label, value, sub, color }) {
  return (
    <div className="rounded-xl border border-line bg-bg-elev1 p-4">
      <div className="eyebrow">{label}</div>
      <div className="mt-1 text-[28px] font-semibold" style={{ color }}>{value}</div>
      {sub && <div className="mt-0.5 text-[11px] text-ink-tertiary">{sub}</div>}
    </div>
  )
}

function ChartBox({ title, children }) {
  return (
    <div className="rounded-xl border border-line bg-bg-elev1 p-4">
      <div className="eyebrow mb-3">{title}</div>
      {children}
    </div>
  )
}

function LoadingScreen() {
  return (
    <div className="flex h-full items-center justify-center text-ink-tertiary gap-2">
      <Loader2 size={16} className="animate-spin" />
      <span className="text-[13px]">Chargement…</span>
    </div>
  )
}
