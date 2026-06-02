import { useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuthStore, useAnnotationStore, useCatalogStore } from './lib/store'
import { api } from './lib/api'
import LoginPage from './pages/LoginPage'
import AnnotationPage from './pages/AnnotationPage'
import QueuePage from './pages/QueuePage'
import AdminPage from './pages/AdminPage'
import PatientsPage from './pages/PatientsPage'
import Shell from './components/layout/Shell'

function Protected({ children }) {
  const user  = useAuthStore((s) => s.user)
  const token = useAuthStore((s) => s.token)
  const fetchImages = useAnnotationStore((s) => s.fetchImages)
  const fetchAll    = useCatalogStore((s) => s.fetchAll)

  useEffect(() => {
    if (!token) return
    fetchImages()
    fetchAll(token)
  }, [token])  // re-run if token changes (login/logout)

  if (!user) return <Navigate to="/login" replace />
  return children
}

function AdminOnly({ children }) {
  const user = useAuthStore((s) => s.user)
  if (!user) return <Navigate to="/login" replace />
  if (user.role !== 'admin') return <Navigate to="/annotate" replace />
  return children
}

export default function App() {
  // Release all locks when the tab/browser closes
  useEffect(() => {
    const handler = () => {
      const token = useAuthStore.getState().token
      if (!token) return
      fetch('/api/auth/logout', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: '{}',
        keepalive: true,
      })
    }
    window.addEventListener('beforeunload', handler)
    return () => window.removeEventListener('beforeunload', handler)
  }, [])

  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route
        path="/*"
        element={
          <Protected>
            <Shell>
              <Routes>
                <Route path="/" element={<Navigate to="/annotate" replace />} />
                <Route path="/annotate" element={<AnnotationPage />} />
                <Route path="/queue"    element={<QueuePage />} />
                <Route path="/patients" element={<PatientsPage />} />
                <Route
                  path="/admin"
                  element={
                    <AdminOnly>
                      <AdminPage />
                    </AdminOnly>
                  }
                />
              </Routes>
            </Shell>
          </Protected>
        }
      />
    </Routes>
  )
}
