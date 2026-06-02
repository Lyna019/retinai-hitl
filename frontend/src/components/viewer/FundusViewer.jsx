import { useRef, useState, useCallback, useMemo, useEffect } from 'react'
import { Minus, Plus, Maximize2, Grid3x3, Flame, Image as ImageIcon, Undo2, Loader2 } from 'lucide-react'
import { useAnnotationStore, useCatalogStore, useAuthStore } from '../../lib/store'
import { LESION_VOCAB } from '../../lib/mockData'
import { api } from '../../lib/api'

// Zoom → grid resolution: 1x→8, 2x→16, 4x→32, 8x→64
const gridResForZoom = (z) => {
  if (z < 1.75) return 8
  if (z < 3.5) return 16
  if (z < 7) return 32
  return 64
}

// Region color palette — cycles for multiple ROIs
const REGION_COLORS = [
  '#14E3CA', // accent teal
  '#FF9F0A', // amber
  '#BF5AF2', // purple
  '#30D158', // green
  '#64D2FF', // sky
  '#FF453A', // red
]

export default function FundusViewer({ image }) {
  const stageRef = useRef(null)
  const gridRef = useRef(null)

  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [showGrid, setShowGrid] = useState(true)
  const [showGradcam, setShowGradcam] = useState(false)
  const [isPanning, setIsPanning] = useState(false)
  const [isPainting, setIsPainting] = useState(false)

  const token = useAuthStore((s) => s.token)

  const { annotation, toggleGridCell, updateAnnotation } = useAnnotationStore((s) => ({
    annotation: s.annotations[s.currentImageId] || {
      regions: [],
      active_region_idx: 0,
      active_lesion: 'HEM',
    },
    toggleGridCell: s.toggleGridCell,
    updateAnnotation: s.updateAnnotation,
  }))

  const { catalogLesions, catalogRegions } = useCatalogStore((s) => ({
    catalogLesions: s.lesions,
    catalogRegions: s.regions,
  }))

  const lesionByCode = useMemo(() => {
    const fallback = Object.fromEntries(LESION_VOCAB.map((l) => [l.code, l]))
    const fromCatalog = Object.fromEntries((catalogLesions || []).map((l) => [l.code, l]))
    return { ...fallback, ...fromCatalog }
  }, [catalogLesions])

  const regionById = useMemo(
    () => Object.fromEntries((catalogRegions || []).map((r) => [r.id, r])),
    [catalogRegions],
  )

  const gridRes = gridResForZoom(zoom)
  const zoomLevel = Math.round(Math.log2(gridRes / 8)) // 0,1,2,3

  const activeIdx = annotation.active_region_idx ?? 0
  const activeRegion = annotation.regions?.[activeIdx]
  const activeLesion = annotation.active_lesion || 'HEM'
  const hasRegions = (annotation.regions?.length ?? 0) > 0

  const regionLabel = useCallback(
    (r, idx) =>
      r?.custom_region_name ||
      regionById[r?.anatomical_region_id]?.name_fr ||
      `Région ${idx + 1}`,
    [regionById],
  )

  // Keyboard shortcuts
  useEffect(() => {
    const PAN_STEP = 60 // px per arrow-key press
    const onKey = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return
      if (e.key === 'g' || e.key === 'G') setShowGrid((v) => !v)
      if (e.key === 'h' || e.key === 'H') setShowGradcam((v) => !v)
      if (e.key === 'f' || e.key === 'F') { setZoom(1); setPan({ x: 0, y: 0 }) }
      // Arrow keys → pan image
      if (e.key === 'ArrowUp')    { e.preventDefault(); setPan((p) => ({ ...p, y: p.y + PAN_STEP })) }
      if (e.key === 'ArrowDown')  { e.preventDefault(); setPan((p) => ({ ...p, y: p.y - PAN_STEP })) }
      if (e.key === 'ArrowLeft')  { e.preventDefault(); setPan((p) => ({ ...p, x: p.x + PAN_STEP })) }
      if (e.key === 'ArrowRight') { e.preventDefault(); setPan((p) => ({ ...p, x: p.x - PAN_STEP })) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  // Record max zoom in active region
  useEffect(() => {
    if (!activeRegion) return
    if (zoom > (activeRegion.max_zoom_reached || 1)) {
      const regions = annotation.regions.map((r, i) =>
        i === activeIdx ? { ...r, max_zoom_reached: zoom } : r,
      )
      updateAnnotation((a) => ({ ...a, regions }))
    }
  }, [zoom]) // eslint-disable-line

  const handleWheel = useCallback((e) => {
    e.preventDefault()
    setZoom((z) => Math.max(1, Math.min(12, z * (e.deltaY < 0 ? 1.12 : 1 / 1.12))))
  }, [])

  const paintAtEvent = useCallback(
    (e) => {
      if (!hasRegions) return
      const grid = gridRef.current
      if (!grid) return
      const rect = grid.getBoundingClientRect()
      const x = (e.clientX - rect.left) / rect.width
      const y = (e.clientY - rect.top) / rect.height
      if (x < 0 || x > 1 || y < 0 || y > 1) return
      const col = Math.floor(x * gridRes)
      const row = Math.floor(y * gridRes)
      toggleGridCell({ zoom_level: zoomLevel, row, col, lesion_code: activeLesion })
    },
    [gridRes, zoomLevel, activeLesion, toggleGridCell, hasRegions],
  )

  const handleMouseDown = useCallback(
    (e) => {
      if (e.button === 1 || e.shiftKey) { setIsPanning(true); return }
      if (e.button === 0 && showGrid) { setIsPainting(true); paintAtEvent(e) }
    },
    [showGrid, paintAtEvent],
  )

  const handleMouseMove = useCallback(
    (e) => {
      if (isPanning) setPan((p) => ({ x: p.x + e.movementX, y: p.y + e.movementY }))
      else if (isPainting) paintAtEvent(e)
    },
    [isPanning, isPainting, paintAtEvent],
  )

  const handleMouseUp = useCallback(() => {
    setIsPanning(false)
    setIsPainting(false)
  }, [])

  const fit = () => { setZoom(1); setPan({ x: 0, y: 0 }) }

  // Tally all painted cells across ALL regions
  const cellCounts = useMemo(() => {
    const counts = {}
    for (const region of (annotation.regions || [])) {
      for (const c of region.cells || []) {
        counts[c.lesion_code] = (counts[c.lesion_code] || 0) + 1
      }
    }
    return counts
  }, [annotation.regions])

  const activeRegionLabel = regionLabel(activeRegion, activeIdx)

  return (
    <div className="relative h-full w-full overflow-hidden bg-bg-base">

      {/* Active region badge (top centre) */}
      {showGrid && hasRegions && (
        <div className="absolute left-1/2 top-4 z-30 -translate-x-1/2">
          <div
            className="flex items-center gap-1.5 rounded-md border px-2.5 py-1 backdrop-blur-sm"
            style={{
              borderColor: REGION_COLORS[activeIdx % REGION_COLORS.length] + '60',
              background:  REGION_COLORS[activeIdx % REGION_COLORS.length] + '15',
            }}
          >
            <span
              className="h-2 w-2 rounded-sm"
              style={{ background: REGION_COLORS[activeIdx % REGION_COLORS.length] }}
            />
            <span className="mono text-[10px] text-ink-primary">{activeRegionLabel}</span>
          </div>
        </div>
      )}

      {/* No-region hint (top centre) */}
      {showGrid && !hasRegions && (
        <div className="absolute left-1/2 top-4 z-30 -translate-x-1/2">
          <div className="rounded-md border border-line/60 bg-bg-elev1/80 px-3 py-1 backdrop-blur-sm">
            <span className="mono text-[10px] text-ink-tertiary">
              Sélectionner une région pour annoter
            </span>
          </div>
        </div>
      )}

      {/* Lesion tally (top right) */}
      {Object.keys(cellCounts).length > 0 && (
        <div className="absolute right-4 top-4 z-30 flex items-center gap-2 rounded-md border border-line bg-bg-elev1/80 px-2.5 py-1 backdrop-blur-sm">
          {Object.entries(cellCounts).map(([code, n]) => (
            <div key={code} className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-sm" style={{ background: lesionByCode[code]?.color }} />
              <span className="mono text-[10px] text-ink-secondary">
                {(lesionByCode[code]?.name_fr || code).slice(0, 6)} · {n}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Stage — captures pointer events */}
      <div
        ref={stageRef}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        className={`relative mx-auto h-full w-full ${showGrid && hasRegions ? 'cursor-anno' : 'cursor-grab'} no-select`}
      >
        {/* Transformed content (pan + zoom) */}
        <div
          className="pointer-events-none absolute inset-0 flex items-center justify-center"
          style={{
            transform: `translate3d(${pan.x}px, ${pan.y}px, 0) scale(${zoom})`,
            transformOrigin: 'center center',
            transition: isPanning || isPainting ? 'none' : 'transform 180ms cubic-bezier(0.2, 0.8, 0.2, 1)',
          }}
        >
          {/* Single coordinate frame: artwork + grid + gradcam share one wrapper */}
          <div className="relative aspect-square w-[min(82vh,82vw)] max-w-[760px]">
            <FundusImage imageId={image?.id} token={token} />

            {showGradcam && <GradCamOverlay imageId={image?.id} />}

            {/* Grid overlay */}
            {showGrid && (
              <div
                ref={gridRef}
                className="absolute inset-[8%] rounded-full overflow-hidden"
              >
                {/* Grid lines for current zoom level */}
                <svg className="pointer-events-none absolute inset-0 h-full w-full" style={{ opacity: 0.18 }}>
                  {Array.from({ length: gridRes + 1 }).map((_, i) => (
                    <g key={i}>
                      <line
                        x1={`${(i / gridRes) * 100}%`} y1="0"
                        x2={`${(i / gridRes) * 100}%`} y2="100%"
                        stroke="#9A9AA3" strokeWidth="0.5"
                      />
                      <line
                        x1="0" y1={`${(i / gridRes) * 100}%`}
                        x2="100%" y2={`${(i / gridRes) * 100}%`}
                        stroke="#9A9AA3" strokeWidth="0.5"
                      />
                    </g>
                  ))}
                </svg>

                {/* Painted cells — ALL zoom levels shown, projected to normalised coords */}
                {(annotation.regions || []).map((region, rIdx) => {
                  const isActive = rIdx === activeIdx
                  const cells = region.cells || []
                  if (cells.length === 0) return null
                  return (
                    <GridCells
                      key={rIdx}
                      currentZoomLevel={zoomLevel}
                      cells={cells}
                      lesionByCode={lesionByCode}
                      regionColor={REGION_COLORS[rIdx % REGION_COLORS.length]}
                      opacity={isActive ? 0.45 : 0.18}
                    />
                  )
                })}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ═══ Bottom control bar — merged: view mode + zoom + undo ═══ */}
      <div className="absolute bottom-4 left-4 z-30 flex items-center gap-1 rounded-lg border border-line bg-bg-elev1/90 p-1 backdrop-blur-sm shadow-lg">
        {/* View-mode tabs */}
        <ViewTab
          active={!showGradcam && !showGrid}
          onClick={() => { setShowGradcam(false); setShowGrid(false) }}
          icon={ImageIcon} label="Image"
        />
        <ViewTab
          active={showGrid}
          onClick={() => { setShowGradcam(false); setShowGrid(true) }}
          icon={Grid3x3} label="Grille"
        />
        <ViewTab
          active={showGradcam}
          onClick={() => { setShowGrid(false); setShowGradcam(true) }}
          icon={Flame} label="Grad-CAM"
        />

        <div className="mx-1 h-5 w-px bg-line" />

        {/* Zoom controls */}
        <ControlBtn onClick={() => setZoom((z) => Math.max(1, z / 1.4))} icon={Minus} title="Dézoomer" />
        <div className="min-w-[52px] text-center">
          <span className="mono text-[11px] text-ink-primary">{zoom.toFixed(1)}×</span>
          <span className="mono text-[9px] text-ink-tertiary ml-1">{gridRes}²</span>
        </div>
        <ControlBtn onClick={() => setZoom((z) => Math.min(12, z * 1.4))} icon={Plus} title="Zoomer" />

        <div className="mx-1 h-5 w-px bg-line" />

        <ControlBtn onClick={fit} icon={Maximize2} title="Recadrer (F)" />
        <ControlBtn
          onClick={() => {
            if (!activeRegion) return
            const regions = (annotation.regions || []).map((r, i) =>
              i === activeIdx ? { ...r, cells: r.cells.slice(0, -1) } : r,
            )
            updateAnnotation((a) => ({ ...a, regions }))
          }}
          icon={Undo2}
          title="Annuler dernière cellule"
        />
      </div>

      {/* Keyboard hints (bottom right) */}
      <div className="absolute bottom-4 right-4 z-30 hidden items-center gap-3 text-[10px] text-ink-tertiary md:flex">
        <ShortcutHint k="G" desc="grille" />
        <ShortcutHint k="H" desc="heatmap" />
        <ShortcutHint k="F" desc="recadrer" />
        <ShortcutHint k="↑↓←→" desc="déplacer" />
        <ShortcutHint k="⇧ drag" desc="glisser" />
      </div>
    </div>
  )
}

/* ─── View mode tab ────────────────────────────────────────────────────── */

function ViewTab({ active, onClick, icon: Icon, label }) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-[11px] transition-all ${
        active ? 'bg-accent/15 text-accent' : 'text-ink-secondary hover:bg-bg-elev2 hover:text-ink-primary'
      }`}
    >
      <Icon size={12} strokeWidth={2} />
      {label}
    </button>
  )
}

function ControlBtn({ onClick, icon: Icon, title }) {
  return (
    <button
      onClick={onClick}
      title={title}
      className="flex h-7 w-7 items-center justify-center rounded-md text-ink-secondary hover:bg-bg-elev2 hover:text-ink-primary transition-colors"
    >
      <Icon size={13} strokeWidth={2} />
    </button>
  )
}

function ShortcutHint({ k, desc }) {
  return (
    <div className="flex items-center gap-1.5">
      <kbd className="mono rounded border border-line bg-bg-elev1 px-1 py-0.5 text-[9px] text-ink-secondary">
        {k}
      </kbd>
      <span>{desc}</span>
    </div>
  )
}

/* ─── Real fundus image ─────────────────────────────────────────────────── */

function FundusImage({ imageId, token }) {
  const [src, setSrc] = useState(null)

  useEffect(() => {
    if (!imageId || !token) return
    let objectUrl = null
    fetch(`/api/images/${imageId}/file`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((r) => {
        if (!r.ok) throw new Error(r.status)
        return r.blob()
      })
      .then((blob) => {
        objectUrl = URL.createObjectURL(blob)
        setSrc(objectUrl)
      })
      .catch(() => setSrc(null))

    return () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [imageId, token])

  if (!src) {
    return (
      <div className="absolute inset-0 rounded-full bg-bg-elev1 animate-pulse" />
    )
  }

  return (
    <img
      src={src}
      alt="Fundus"
      className="absolute inset-0 h-full w-full object-contain"
      draggable={false}
    />
  )
}

/* ─── Grad-CAM heatmap overlay ─────────────────────────────────────────── */

function GradCamOverlay({ imageId }) {
  const [gradcamUrl, setGradcamUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [failed, setFailed] = useState(false)
  const token = useAuthStore((s) => s.token)

  useEffect(() => {
    if (!imageId || !token) return
    setLoading(true)
    setFailed(false)
    setGradcamUrl(null)
    api.getGradcam(imageId, token)
      .then((data) => {
        if (data?.gradcam_url) setGradcamUrl(data.gradcam_url)
        else setFailed(true)
      })
      .catch(() => setFailed(true))
      .finally(() => setLoading(false))
  }, [imageId, token])

  if (loading) {
    return (
      <div className="pointer-events-none absolute inset-[8%] rounded-full overflow-hidden flex items-center justify-center">
        <Loader2 size={28} className="animate-spin text-accent/60" />
      </div>
    )
  }

  if (gradcamUrl && !failed) {
    return (
      <div className="pointer-events-none absolute inset-[8%] rounded-full overflow-hidden mix-blend-screen opacity-70">
        <img
          src={gradcamUrl}
          alt="Grad-CAM heatmap"
          className="absolute inset-0 h-full w-full rounded-full object-cover"
        />
      </div>
    )
  }

  // Fallback placeholder gradient
  return (
    <div className="pointer-events-none absolute inset-[8%] rounded-full overflow-hidden mix-blend-screen opacity-70">
      <div
        className="absolute inset-0 rounded-full"
        style={{
          background:
            'radial-gradient(circle at 60% 55%, rgba(255,70,40,0.75) 0%, rgba(255,140,40,0.45) 18%, rgba(255,210,60,0.25) 30%, rgba(0,0,0,0) 48%)',
        }}
      />
    </div>
  )
}

/* ─── Grid cells — absolute positioning from normalised coords ─────────
 *
 * Each cell is stored with {zoom_level, row, col, lesion_code}.
 * To show cells from ALL zoom levels at any current magnification:
 *   cellRes = 8 × 2^zoom_level   (the grid resolution when the cell was painted)
 *   left%  = col / cellRes × 100
 *   top%   = row / cellRes × 100
 *   size%  = 1   / cellRes × 100
 *
 * Cells painted at the current zoom level look full-opacity.
 * Cells from other zoom levels are dimmer but still visible.
 * ──────────────────────────────────────────────────────────────────────── */

function GridCells({ currentZoomLevel, cells, lesionByCode, regionColor, opacity }) {
  return (
    <div className="absolute inset-0">
      {cells.map((c, i) => {
        const cellRes = 8 * (1 << c.zoom_level)         // 8, 16, 32, or 64
        const sizePct = (1 / cellRes) * 100
        const leftPct = (c.col / cellRes) * 100
        const topPct  = (c.row / cellRes) * 100
        const color   = lesionByCode[c.lesion_code]?.color || regionColor
        const isCurrent = c.zoom_level === currentZoomLevel

        return (
          <div
            key={`${c.zoom_level}-${c.row}-${c.col}-${i}`}
            style={{
              position: 'absolute',
              left:   `${leftPct}%`,
              top:    `${topPct}%`,
              width:  `${sizePct}%`,
              height: `${sizePct}%`,
              background: color,
              opacity: isCurrent ? opacity : opacity * 0.55,
              boxShadow: `inset 0 0 0 ${isCurrent ? '1px' : '0.5px'} ${color}`,
              transition: 'opacity 150ms',
            }}
          />
        )
      })}
    </div>
  )
}
