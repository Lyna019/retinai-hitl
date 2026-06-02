import { create } from 'zustand'
import { api } from './api'

// ─── Auth store ──────────────────────────────────────────────────────────────

const _loadAuth = () => {
  try {
    return {
      user:  JSON.parse(localStorage.getItem('rh_user') || 'null'),
      token: localStorage.getItem('rh_token') || null,
    }
  } catch { return { user: null, token: null } }
}

export const useAuthStore = create((set) => ({
  ..._loadAuth(),

  login: async (username, password) => {
    const data = await api.login(username, password)
    localStorage.setItem('rh_user',  JSON.stringify(data.user))
    localStorage.setItem('rh_token', data.access_token)
    set({ user: data.user, token: data.access_token })
    return data.user
  },

  logout: () => {
    const token = useAuthStore.getState().token
    // Release all locks server-side before clearing the session
    if (token) {
      api.logoutSession(token).catch(() => {})
    }
    localStorage.removeItem('rh_user')
    localStorage.removeItem('rh_token')
    set({ user: null, token: null })
  },
}))


// ─── Annotation store ────────────────────────────────────────────────────────

const defaultAnnotation = () => ({
  disease_labels: [],   // [{ disease_code, grade }]
  regions: [],          // [{ anatomical_region_id, custom_region_name, cells: [{zoom_level, row, col, lesion_code}] }]
  gradcam_verdict: null,
  notes_text: '',
  notes_audio_blob: null,
  active_lesion: 'HEM',
  active_region_idx: 0,
})

export const useAnnotationStore = create((set, get) => ({
  images: [],
  currentImageId: null,
  annotations: {},       // keyed by imageId
  queueFilter: 'all',
  _startTimes: {},

  // ─── Queue ──────────────────────────────────────────────────────────────
  fetchImages: async () => {
    const token = useAuthStore.getState().token
    try {
      const imgs = await api.listImages(token)
      set({ images: imgs })
      const pending = imgs.filter((i) => i.status !== 'done')
      if (!get().currentImageId && pending.length > 0) {
        set({ currentImageId: pending[0].id })
      }
    } catch (e) {
      console.error('fetchImages', e)
    }
  },

  // ─── Unified open-image action ───────────────────────────────────────────
  openImage: async (imageId) => {
    const token = useAuthStore.getState().token
    const s = get()

    // Set current image immediately for responsive UI
    set({ currentImageId: imageId })

    // Record start time
    set((st) => ({ _startTimes: { ...st._startTimes, [imageId]: Date.now() } }))

    // Acquire lock (non-blocking failure)
    api.acquireLock(imageId, token).catch(() => {})

    // Fetch stored prediction; re-run if none or if cached result is from mock mode
    const _setPreds = (preds) =>
      set((st) => ({
        images: st.images.map((img) =>
          img.id === imageId ? { ...img, predictions: preds } : img,
        ),
      }))

    api.getPredictions(imageId, token)
      .then((preds) => {
        const _isStale = (v) => !v || v.includes('-mock') || v.includes('testset') || v.startsWith('v1.') || v === 'retinai-v0.5.0'
        const isMock = Array.isArray(preds)
          ? preds.some((p) => _isStale(p.model_version))
          : _isStale(preds?.model_version)
        if (isMock) {
          // Overwrite stale mock prediction with real inference
          return api.runPrediction(imageId, token)
        }
        return preds
      })
      .then(_setPreds)
      .catch(() => {
        api.runPrediction(imageId, token).then(_setPreds).catch(() => {})
      })

    // Fetch uncertainty score from model service
    api.runUncertainty(imageId, token)
      .then((data) => {
        if (data?.uncertainty != null) {
          set((st) => ({
            images: st.images.map((img) =>
              img.id === imageId ? { ...img, uncertainty: data.uncertainty } : img,
            ),
          }))
        }
      })
      .catch(() => {})

    // Load existing draft if no annotation started yet
    if (!s.annotations[imageId]) {
      api.getAnnotationsForImage(imageId, token)
        .then((anns) => {
          const draft = anns?.find?.((a) => a.status === 'draft')
          if (!draft) return
          set((st) => {
            if (st.annotations[imageId]) return {}   // already set by user action
            return {
              annotations: {
                ...st.annotations,
                [imageId]: {
                  ...defaultAnnotation(),
                  disease_labels: draft.disease_labels || [],
                  regions:        draft.regions        || [],
                  gradcam_verdict: draft.gradcam_verdict ?? null,
                  notes_text:     draft.notes_text     || '',
                },
              },
            }
          })
        })
        .catch(() => {})
    }
  },

  setCurrentImage: (id) => set({ currentImageId: id }),
  setQueueFilter:  (f)  => set({ queueFilter: f }),

  getCurrentAnnotation: () => {
    const s = get()
    return s.annotations[s.currentImageId] || defaultAnnotation()
  },

  updateAnnotation: (updater) =>
    set((s) => {
      const id = s.currentImageId
      const current = s.annotations[id] || defaultAnnotation()
      const next = typeof updater === 'function' ? updater(current) : { ...current, ...updater }
      return { annotations: { ...s.annotations, [id]: next } }
    }),

  // ─── Disease labels ──────────────────────────────────────────────────────
  toggleDisease: (disease_code) =>
    set((s) => {
      const id = s.currentImageId
      const current = s.annotations[id] || defaultAnnotation()
      const exists = current.disease_labels.find((l) => l.disease_code === disease_code)
      const next = exists
        ? { ...current, disease_labels: current.disease_labels.filter((l) => l.disease_code !== disease_code) }
        : { ...current, disease_labels: [...current.disease_labels, { disease_code, grade: null }] }
      return { annotations: { ...s.annotations, [id]: next } }
    }),

  setDiseaseGrade: (disease_code, grade) =>
    set((s) => {
      const id = s.currentImageId
      const current = s.annotations[id] || defaultAnnotation()
      const next = {
        ...current,
        disease_labels: current.disease_labels.map((l) =>
          l.disease_code === disease_code ? { ...l, grade } : l,
        ),
      }
      return { annotations: { ...s.annotations, [id]: next } }
    }),

  // ─── Regions of interest ─────────────────────────────────────────────────
  addRegion: (region) =>
    set((s) => {
      const id = s.currentImageId
      const current = s.annotations[id] || defaultAnnotation()
      const next = {
        ...current,
        regions: [...current.regions, { ...region, cells: [] }],
        active_region_idx: current.regions.length,
      }
      return { annotations: { ...s.annotations, [id]: next } }
    }),

  removeRegion: (regionIdx) =>
    set((s) => {
      const id = s.currentImageId
      const current = s.annotations[id] || defaultAnnotation()
      const regions = current.regions.filter((_, i) => i !== regionIdx)
      const active = Math.min(current.active_region_idx, regions.length - 1)
      return { annotations: { ...s.annotations, [id]: { ...current, regions, active_region_idx: Math.max(0, active) } } }
    }),

  setActiveRegion: (idx) =>
    set((s) => {
      const id = s.currentImageId
      const current = s.annotations[id] || defaultAnnotation()
      return { annotations: { ...s.annotations, [id]: { ...current, active_region_idx: idx } } }
    }),

  // ─── Grid cells (within active region) ──────────────────────────────────
  toggleGridCell: (cell) =>
    set((s) => {
      const id = s.currentImageId
      const current = s.annotations[id] || defaultAnnotation()
      const regionIdx = current.active_region_idx
      if (regionIdx < 0 || regionIdx >= current.regions.length) return s

      const key = (c) => `${c.zoom_level}:${c.row}:${c.col}`
      const target = key(cell)
      const region = current.regions[regionIdx]
      const existing = region.cells.find((c) => key(c) === target)

      let cells
      if (existing) {
        cells = existing.lesion_code === cell.lesion_code
          ? region.cells.filter((c) => key(c) !== target)
          : region.cells.map((c) => (key(c) === target ? { ...c, lesion_code: cell.lesion_code } : c))
      } else {
        cells = [...region.cells, cell]
      }

      const regions = current.regions.map((r, i) => i === regionIdx ? { ...r, cells } : r)
      return { annotations: { ...s.annotations, [id]: { ...current, regions } } }
    }),

  setActiveLesion: (lesion_code) =>
    set((s) => {
      const id = s.currentImageId
      const current = s.annotations[id] || defaultAnnotation()
      return { annotations: { ...s.annotations, [id]: { ...current, active_lesion: lesion_code } } }
    }),

  // ─── Other fields ────────────────────────────────────────────────────────
  setImageQuality: async (imageId, quality) => {
    const token = useAuthStore.getState().token
    await api.setImageQuality(imageId, quality, token).catch(() => {})
    set((s) => ({
      images: s.images.map((img) =>
        img.id === imageId ? { ...img, image_quality: quality } : img,
      ),
    }))
  },

  setGradCamVerdict: (verdict) =>
    set((s) => {
      const id = s.currentImageId
      const current = s.annotations[id] || defaultAnnotation()
      return { annotations: { ...s.annotations, [id]: { ...current, gradcam_verdict: verdict } } }
    }),

  setNotesText: (text) =>
    set((s) => {
      const id = s.currentImageId
      const current = s.annotations[id] || defaultAnnotation()
      return { annotations: { ...s.annotations, [id]: { ...current, notes_text: text } } }
    }),

  appendNotesText: (chunk) =>
    set((s) => {
      const id = s.currentImageId
      const current = s.annotations[id] || defaultAnnotation()
      const sep = current.notes_text && !current.notes_text.endsWith(' ') ? ' ' : ''
      return {
        annotations: {
          ...s.annotations,
          [id]: { ...current, notes_text: current.notes_text + sep + chunk },
        },
      }
    }),

  // ─── Submit to API ───────────────────────────────────────────────────────
  submitAnnotation: async () => {
    const s = get()
    const id = s.currentImageId
    if (!id) return
    const ann = s.annotations[id] || defaultAnnotation()
    const token = useAuthStore.getState().token
    const startTime = s._startTimes?.[id] || Date.now()

    try {
      const result = await api.submitAnnotation(
        {
          image_id: id,
          disease_labels: ann.disease_labels,
          regions: ann.regions,
          gradcam_verdict: ann.gradcam_verdict,
          notes_text: ann.notes_text,
          time_spent_sec: Math.round((Date.now() - startTime) / 1000),
        },
        token,
      )

      await api.releaseLock(id, token).catch(() => {})

      set((s2) => {
        const images = s2.images.map((img) => img.id === id ? { ...img, status: 'done' } : img)
        const idx = images.findIndex((i) => i.id === id)
        const next =
          images.slice(idx + 1).find((i) => i.status !== 'done') ||
          images.find((i) => i.status !== 'done' && i.id !== id)
        return { images, currentImageId: next ? next.id : id }
      })
      return result
    } catch (e) {
      console.error('submitAnnotation', e)
      throw e
    }
  },

  setStartTime: (imageId) =>
    set((s) => ({ _startTimes: { ...(s._startTimes || {}), [imageId]: Date.now() } })),
}))


// ─── Catalog store ───────────────────────────────────────────────────────────

export const useCatalogStore = create((set) => ({
  diseases: [],
  mechanisms: [],
  lesions: [],
  regions: [],
  systemicDiseases: [],

  fetchAll: async (token) => {
    try {
      const [diseases, mechanisms, lesions, regions, systemicDiseases] = await Promise.all([
        api.listDiseases(token),
        api.listMechanisms(token),
        api.listLesions(token),
        api.listRegions(token),
        api.listSystemicDiseases(token),
      ])
      set({ diseases, mechanisms, lesions, regions, systemicDiseases })
    } catch (e) {
      console.error('fetchCatalog', e)
    }
  },
}))
