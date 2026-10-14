import React, { useState, useEffect, useRef } from 'react';

// Live captions over the editor's player (GET /api/jobs/{id}/captions): the same fixed blocks and style
// presets that the export burns in (backend app/tasks/subtitles.py), drawn on top of the playing video
// instead of into it. Sized like the ASS file: font, outline and margins are fractions of the displayed
// picture (the video's content box inside its letterbox), so the look matches the exported file.
// Drag the caption to move it, drag its corner handle to resize, double-click to reset: the placement
// ({scale, x, y}: x / y = bottom centre as a fraction of the frame, null = the preset's place) is saved per
// job (PUT /api/jobs/{id}/captions/placement) and used by the captioned exports too.

const KEY = (jobId) => `vf-captions-${jobId}`;
export const savedCaptions = (jobId) => { try { return localStorage.getItem(KEY(jobId)) || ''; } catch { return ''; } };
export const saveCaptions = (jobId, style) => { try { localStorage.setItem(KEY(jobId), style); } catch { /* private mode */ } };

export const DEFAULT_PLACEMENT = { scale: 1, x: null, y: null };
export const isMoved = (p) => !!p && (p.scale !== 1 || p.x !== null);

// Caption blocks + presets of a job's last render (refetched when the render changes: reloadKey), and the
// job's placement. setPlacement(p) updates the overlay; setPlacement(p, true) also saves it.
export function useCaptions(jobId, reloadKey) {
  const [data, setData] = useState(null);
  const [placement, setLocal] = useState(DEFAULT_PLACEMENT);
  useEffect(() => {
    let live = true;
    fetch(`/api/jobs/${jobId}/captions`).then(r => r.ok ? r.json() : Promise.reject(r.statusText))
      .then(d => { if (live) { setData(d); setLocal(d.placement || DEFAULT_PLACEMENT); } })
      .catch(e => { console.warn(`captions unavailable for job ${jobId}: ${e}`); live && setData(null); });
    return () => { live = false; };
  }, [jobId, reloadKey]);
  const setPlacement = (p, commit = false) => {
    setLocal(p);
    if (!commit) return;
    fetch(`/api/jobs/${jobId}/captions/placement`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(p) })
      .then(r => r.ok ? r.json() : Promise.reject(r.statusText))
      .then(setLocal)                                  // the server's clamped copy
      .catch(e => console.error(`saving caption placement failed for job ${jobId}: ${e}`));
  };
  return { data, placement, setPlacement };
}

const rgba = (hex, a = 1) => `rgba(${parseInt(hex.slice(0, 2), 16)},${parseInt(hex.slice(2, 4), 16)},${parseInt(hex.slice(4, 6), 16)},${a})`;

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export default function CaptionOverlay({ videoRef, videoKey, data, style, placement = DEFAULT_PLACEMENT, onPlacement }) {
  const [t, setT] = useState(0);
  const [box, setBox] = useState(null);                // displayed picture inside the <video> element
  const drag = useRef(null);                           // { mode: move | scale, start pointer, start placement }

  // Follow the playing frame (per video frame where supported; captions change on whole seconds)
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !style) return;
    let id, stop = false;
    const tick = (_, meta) => { if (stop) return; setT(meta ? meta.mediaTime : v.currentTime); id = v.requestVideoFrameCallback?.(tick); };
    if (v.requestVideoFrameCallback) id = v.requestVideoFrameCallback(tick);
    const onTime = () => setT(v.currentTime);          // seeks while paused, and browsers without frame callbacks
    v.addEventListener('timeupdate', onTime);
    v.addEventListener('seeked', onTime);
    return () => { stop = true; v.removeEventListener('timeupdate', onTime); v.removeEventListener('seeked', onTime); if (id) v.cancelVideoFrameCallback?.(id); };
  }, [videoRef, videoKey, style]);

  // Where the picture sits in the element (object-fit: contain)
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const measure = () => {
      const ew = v.clientWidth, eh = v.clientHeight, vw = v.videoWidth, vh = v.videoHeight;
      if (!ew || !eh || !vw || !vh) return setBox(null);
      const k = Math.min(ew / vw, eh / vh), w = vw * k, h = vh * k;
      setBox({ left: v.offsetLeft + (ew - w) / 2, top: v.offsetTop + (eh - h) / 2, width: w, height: h });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(v);
    v.addEventListener('loadedmetadata', measure);
    return () => { ro.disconnect(); v.removeEventListener('loadedmetadata', measure); };
  }, [videoRef, videoKey]);

  const s = data?.styles?.[style];
  if (!s || !box) return null;
  const cue = data.cues.find(c => t >= c[0] && t < c[1]);
  if (!cue) return null;
  const short = Math.min(box.width, box.height) * placement.scale;
  const ol = s.outline * short, sh = s.shadow * short;
  const marginV = box.height > box.width ? Math.max(s.margin_v, data.portrait_margin_v) : s.margin_v;
  const x = placement.x ?? 0.5, y = placement.y ?? 1 - marginV;       // bottom centre, fraction of the frame
  const [lo, hi] = data.scale_range || [0.4, 3];

  const start = (mode) => (e) => {
    if (e.button !== 0) return;
    e.preventDefault(); e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    const ax = box.left + x * box.width, ay = box.top + y * box.height;     // anchor, element coords
    const r = e.currentTarget.closest('.vf-cc-layer').getBoundingClientRect();
    drag.current = { mode, px: e.clientX, py: e.clientY, x, y, scale: placement.scale,
                     dist: Math.max(8, Math.hypot(e.clientX - r.left - ax, e.clientY - r.top - ay)), ax, ay, r };
  };
  const move = (e) => {
    const d = drag.current;
    if (!d) return;
    if (d.mode === 'move') {
      d.last = { scale: d.scale, x: clamp(d.x + (e.clientX - d.px) / box.width, 0.05, 0.95),
                 y: clamp(d.y + (e.clientY - d.py) / box.height, 0.05, 1) };
    } else {                                           // scale by distance from the anchor, like a pinch
      const dist = Math.hypot(e.clientX - d.r.left - d.ax, e.clientY - d.r.top - d.ay);
      d.last = { scale: Math.round(clamp(d.scale * dist / d.dist, lo, hi) * 100) / 100, x: placement.x, y: placement.y };
    }
    onPlacement(d.last);
  };
  // Save what the last move computed: the render-time placement can be one move behind
  const end = () => { const d = drag.current; drag.current = null; if (d?.last) onPlacement(d.last, true); };
  const edit = { onPointerMove: move, onPointerUp: end, onPointerCancel: end };
  const text = {
    fontFamily: `"${s.font}", Arial, sans-serif`, fontWeight: s.bold ? 700 : 400, fontStyle: s.italic ? 'italic' : 'normal',
    fontSize: s.size * short / s.em_height, lineHeight: s.em_height,       // libass: size = ascent + descent
    color: `#${s.colour}`, textTransform: s.upper ? 'uppercase' : 'none',
    ...(s.box
      ? { background: rgba('000000', (255 - s.box_alpha) / 255), padding: `0 ${ol}px`, boxDecorationBreak: 'clone', WebkitBoxDecorationBreak: 'clone' }
      : { WebkitTextStroke: `${2 * ol}px #${s.outline_colour}`, paintOrder: 'stroke fill',
          textShadow: sh > 0 ? `${sh}px ${sh}px 0 ${rgba('000000', 0.5)}` : 'none' }),
  };
  return (
    <div className="vf-cc-layer" style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
      <div style={{ position: 'absolute', ...box, overflow: 'hidden' }}>
        <div style={{ position: 'absolute', width: (1 - 2 * data.side_margin) * box.width, left: (x - 0.5 + data.side_margin) * box.width,
                      bottom: (1 - y) * box.height, textAlign: 'center' }}>
          <div className={`vf-cc-box${drag.current ? ' dragging' : ''}`} {...edit} onPointerDown={start('move')}
            onDoubleClick={() => onPlacement(DEFAULT_PLACEMENT, true)}
            title="Drag to move · drag the corner to resize · double-click to reset">
            <span key={cue[0]} className={s.pop ? 'vf-cc-pop' : undefined} style={{ ...text, display: s.box ? 'inline' : 'inline-block', maxWidth: '100%' }}>{cue[2]}</span>
            <span className="vf-cc-handle" onPointerDown={start('scale')} />
          </div>
        </div>
      </div>
    </div>);
}
