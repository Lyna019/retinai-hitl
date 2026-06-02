import TopBar from './TopBar'

export default function Shell({ children }) {
  return (
    <div className="flex h-screen flex-col overflow-hidden bg-bg-base text-ink-primary">
      <TopBar />
      <main className="flex-1 overflow-hidden">{children}</main>
    </div>
  )
}
