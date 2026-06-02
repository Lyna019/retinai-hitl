/**
 * Thin API client — all calls go through the Vite proxy to /api
 */

const BASE = '/api'

async function request(method, path, body, token) {
  const headers = { 'Content-Type': 'application/json' }
  if (token) headers['Authorization'] = `Bearer ${token}`
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  if (res.status === 401) {
    // Session expired — force logout so user gets redirected to login
    const { useAuthStore } = await import('./store')
    useAuthStore.getState().logout?.()
    window.location.href = '/login'
    throw new Error('Session expirée — veuillez vous reconnecter')
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    const detail = err.detail
    const message = Array.isArray(detail)
      ? detail.map(d => d.msg || JSON.stringify(d)).join(' · ')
      : detail || 'Erreur serveur'
    throw new Error(message)
  }
  return res.json()
}

const get  = (path, token)         => request('GET',    path, undefined, token)
const post = (path, body, token)   => request('POST',   path, body, token)
const put  = (path, body, token)   => request('PUT',    path, body, token)
const del  = (path, token)         => request('DELETE', path, undefined, token)

// ─── Auth ────────────────────────────────────────────────────────────────────
export const api = {
  login:         (username, password) => post('/auth/login', { username, password }),
  logoutSession: (token) => post('/auth/logout', {}, token),

  // ─── Images ───────────────────────────────────────────────────────────────
  listImages:      (token, params = {}) => get(`/images?${new URLSearchParams(params)}`, token),
  getImage:        (id, token)          => get(`/images/${id}`, token),
  getPredictions:  (id, token)          => get(`/images/${id}/predictions`, token),
  setImageQuality: (id, quality, token) => request('PATCH', `/images/${id}/quality`, { quality }, token),
  acquireLock:     (id, token)          => post(`/locks/${id}/acquire`, {}, token),
  releaseLock:     (id, token)          => del(`/locks/${id}/release`, token),

  // ─── Patients ─────────────────────────────────────────────────────────────
  listPatients:    (token, params = {}) => get(`/patients?${new URLSearchParams(params)}`, token),
  getPatient:      (id, token, params = {}) => get(`/patients/${id}?${new URLSearchParams(params)}`, token),
  setSystemicDiseases: (patientId, ids, token) =>
    put(`/patients/${patientId}/systemic-diseases`, { systemic_disease_ids: ids }, token),
  updateHistoricalNote: (patientId, note, token) =>
    put(`/patients/${patientId}/note`, { historical_note: note }, token),

  // ─── Catalog ──────────────────────────────────────────────────────────────
  listDiseases:   (token) => get('/catalog/diseases', token),
  listMechanisms: (token) => get('/catalog/mechanisms', token),
  listLesions:    (token) => get('/catalog/lesions', token),
  listRegions:    (token) => get('/catalog/regions', token),
  createRegion:   (name, token) => post('/catalog/regions', { name_fr: name }, token),
  listSystemicDiseases: (token) => get('/catalog/systemic-diseases', token),
  createDisease:  (payload, token) => post('/catalog/diseases', payload, token),
  updateDisease:  (code, payload, token) => put(`/catalog/diseases/${code}`, payload, token),
  deleteDisease:  (code, token) => del(`/catalog/diseases/${code}`, token),
  createSystemicDisease: (payload, token) => post('/catalog/systemic-diseases', payload, token),
  deleteSystemicDisease: (id, token) => del(`/catalog/systemic-diseases/${id}`, token),

  // ─── Annotations ──────────────────────────────────────────────────────────
  submitAnnotation: (payload, token) => post('/annotations/submit', payload, token),
  saveDraft:        (imageId, payload, token) => post(`/annotations/${imageId}/draft`, payload, token),
  getAnnotationsForImage: (imageId, token) => get(`/annotations/image/${imageId}`, token),

  // ─── Proposals ────────────────────────────────────────────────────────────
  listProposals:   (token, status) => get(`/proposals${status ? `?status=${status}` : ''}`, token),
  getProposal:     (id, token) => get(`/proposals/${id}`, token),
  getProposalImageUrl: (id, token) => get(`/proposals/${id}/image-url`, token),
  createProposal:  (payload, token) => post('/proposals', payload, token),
  approveProposal: (id, payload, token) => post(`/proposals/${id}/approve`, payload, token),
  rejectProposal:  (id, token) => post(`/proposals/${id}/reject`, {}, token),

  // ─── Admin ────────────────────────────────────────────────────────────────
  getStats:        (endpoint, token) => get(`/admin/stats/${endpoint}`, token),
  listDoctors:     (token) => get('/admin/doctors', token),
  createDoctor:    (payload, token) => post('/admin/doctors', payload, token),
  toggleDoctor:    (id, token) => put(`/admin/doctors/${id}/toggle-active`, {}, token),
  getAuditLog:     (token, params = {}) => get(`/admin/audit-log?${new URLSearchParams(params)}`, token),
  listModelVersions:    (token) => get('/admin/model-versions', token),
  registerModelVersion: (payload, token) => post('/admin/model-versions', payload, token),
  activateModelVersion: (id, token) => post(`/admin/model-versions/${id}/activate`, {}, token),
  getALConfig:     (token) => get('/admin/active-learning', token),
  updateALConfig:  (payload, token) => put('/admin/active-learning', payload, token),
  listLocks:       (token) => get('/admin/locks', token),
  releaseLockAdmin: (imageId, token) => del(`/admin/locks/${imageId}`, token),

  // ─── Model service ────────────────────────────────────────────────────────
  runPrediction:  (imageId, token) => post(`/model/predict?image_id=${imageId}`, {}, token),
  runUncertainty: (imageId, token) => post(`/model/uncertainty?image_id=${imageId}`, {}, token),
  getGradcam:     (imageId, token) => post(`/model/gradcam?image_id=${imageId}`, {}, token),
  getModelHealth: (token)          => get('/model/health', token),

  // ─── VLM ──────────────────────────────────────────────────────────────────
  describeImage: (imageId, token) => post(`/vlm/describe/${imageId}`, {}, token),

  // ─── Transcription ────────────────────────────────────────────────────────
  transcribeAudio: (formData, token) => {
    const headers = {}
    if (token) headers['Authorization'] = `Bearer ${token}`
    return fetch(`${BASE}/transcribe/stream`, { method: 'POST', headers, body: formData }).then(r => r.json())
  },

  transcribeAudioFull: (formData, token) => {
    const headers = {}
    if (token) headers['Authorization'] = `Bearer ${token}`
    return fetch(`${BASE}/transcribe/audio`, { method: 'POST', headers, body: formData })
      .then(r => { if (!r.ok) throw new Error(r.statusText); return r.json() })
  },
}
