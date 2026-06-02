// "2026-01-24T06:17:37.650868" → "2026-01-24  06:17:37"
export function fmtDatetime(iso) {
  if (!iso) return '—'
  const s = String(iso)
  return s.slice(0, 10) + '  ' + s.slice(11, 19)
}

// "2026-01-24T06:17:37.650868" → "2026-01-24"
export function fmtDate(iso) {
  if (!iso) return '—'
  return String(iso).slice(0, 10)
}
