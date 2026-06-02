import { useEffect, useRef } from 'react'
import { useAnnotationStore } from '../lib/store'
import { useAuthStore } from '../lib/store'
import { api } from '../lib/api'
import QueueSidebar from '../components/queue/QueueSidebar'
import FundusViewer from '../components/viewer/FundusViewer'
import PatientInfoPanel from '../components/annotation/PatientInfoPanel'
import PredictionsPanel from '../components/annotation/PredictionsPanel'
import DiagnosticPanel from '../components/annotation/DiagnosticPanel'
import AutoFlags from '../components/annotation/AutoFlags'
import RegionSelector from '../components/annotation/RegionSelector'
import LesionToolbar from '../components/annotation/LesionToolbar'
import GradCamValidation from '../components/annotation/GradCamValidation'
import ImageQualitySelector from '../components/annotation/ImageQualitySelector'
import ClinicalNotes from '../components/annotation/ClinicalNotes'
import AnnotationFooter from '../components/annotation/AnnotationFooter'

export default function AnnotationPage() {
  const { images, currentImageId, submitAnnotation, setStartTime } =
    useAnnotationStore((s) => ({
      images: s.images,
      currentImageId: s.currentImageId,
      submitAnnotation: s.submitAnnotation,
      setStartTime: s.setStartTime,
    }))

  // Fallback: track start time when image changes (e.g. after queue advance on submit)
  useEffect(() => {
    if (currentImageId) setStartTime(currentImageId)
  }, [currentImageId])

  // Auto-save draft every 30 seconds — doctor never loses work on browser close
  const autoSaveRef = useRef(null)
  useEffect(() => {
    const doSave = () => {
      const state = useAnnotationStore.getState()
      const imgId = state.currentImageId
      const token = useAuthStore.getState().token
      if (!imgId || !token) return
      const ann = state.annotations[imgId] || {}
      // Only save if there is something to save
      if (!ann.disease_labels?.length && !ann.notes_text) return
      api.saveDraft(imgId, {
        disease_labels:  ann.disease_labels  || [],
        regions:         ann.regions         || [],
        gradcam_verdict: ann.gradcam_verdict || null,
        notes_text:      ann.notes_text      || '',
      }, token).catch(() => {})
    }
    autoSaveRef.current = setInterval(doSave, 30_000)
    // Also save on page unload
    window.addEventListener('beforeunload', doSave)
    return () => {
      clearInterval(autoSaveRef.current)
      window.removeEventListener('beforeunload', doSave)
    }
  }, [])

  // Keyboard shortcut: Enter → submit
  useEffect(() => {
    const onKey = (e) => {
      const tag = e.target?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA') return
      if (e.key === 'Enter') { e.preventDefault(); submitAnnotation() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [submitAnnotation])

  const current = images.find((i) => i.id === currentImageId)

  if (!current) {
    return (
      <div className="flex h-full items-center justify-center text-ink-secondary">
        {images.length === 0 ? 'Chargement…' : 'Aucune image disponible.'}
      </div>
    )
  }

  return (
    <div className="flex h-full">
      <QueueSidebar />
      <div className="flex flex-1 overflow-hidden">
        {/* Viewer + patient info overlay */}
        <div className="relative flex-1 overflow-hidden border-r border-line bg-bg-base">
          <FundusViewer image={current} />
          <PatientInfoPanel image={current} />
        </div>

        {/* Right annotation panel */}
        <aside className="flex w-[400px] flex-shrink-0 flex-col overflow-y-auto bg-bg-elev1/30">
          <PredictionsPanel predictions={current.predictions} imageId={currentImageId} />
          <Divider />
          <ImageQualitySelector imageId={currentImageId} quality={current.image_quality} />
          <Divider />
          <DiagnosticPanel />
          <Divider />
          <AutoFlags />
          <Divider />
          <RegionSelector />
          <Divider />
          <LesionToolbar />
          <Divider />
          <GradCamValidation />
          <Divider />
          <ClinicalNotes imageId={currentImageId} />
          <div className="flex-1" />
          <AnnotationFooter />
        </aside>
      </div>
    </div>
  )
}

function Divider() {
  return <div className="mx-5 mt-4 hairline" />
}
