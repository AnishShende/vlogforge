import React, { useState, useEffect, useRef, useLayoutEffect, useMemo } from 'react';
import { ZoomIn, ZoomOut, Maximize2, AlertTriangle } from 'lucide-react';

// Bottom timeline of the review screen: the output segments (the EDL) laid out on a time ruler,
// coloured by story function and labelled with the moment's summary, over the output's audio
// waveform (A1). Click or drag empty space to scrub; click a clip to seek there and select its
// words (then ⌫ cuts it); drag a clip sideways to move it; drag a clip's edge to trim or extend it
// by whole words (trim() decides the words). Pending edits show on the clips: cut words hatched,
// restored words as +N at the edge. All of it applies on re-compile.
// ctrl/cmd + wheel (or pinch) zooms; the view follows the playhead while it plays.

const FAMILY = [
  [['hook'], '#8B5CF6'],
  [['orientation', 'goal'], '#06B6D4'],
  [['activity', 'explanation', 'progress', 'transition'], '#3B82F6'],
  [['evidence', 'discovery', 'obstacle'], '#F59E0B'],
  [['reaction', 'atmosphere'], '#EC4899'],
  [['payoff', 'reflection', 'conclusion'], '#10B981'],
];
export const functionColor = (fn) => (FAMILY.find(([names]) => names.includes(fn)) || [null, '#64748B'])[1];

const fmt = (s) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, '0')}`;
const TICK_STEPS = [0.5, 1, 2, 5, 10, 15, 30, 60, 120];
const LABEL_W = 36, PAD = 12, MIN_PPS = 4, MAX_PPS = 400, DRAG_PX = 4;

// Mirrored waveform polygon in a viewBox of (n peaks) x 1
const wavePath = (p) => {
  if (!p.length) return '';
  const top = p.map((v, k) => `${k + 0.5},${(0.5 - v / 2).toFixed(3)}`);
  const bottom = p.map((v, k) => `${k + 0.5},${(0.5 + v / 2).toFixed(3)}`).reverse();
  return `M0,0.5 L${top.join(' L')} L${p.length},0.5 L${bottom.join(' L')} Z`;
};

export default function EditTimeline({ segments, order, wordsById, wordOut, currentTime, onSeek, onClipClick, onReorder, warnings, wave, selectedClip,
  kept, added, trim, onTrim }) {
  const scrollRef = useRef(null);
  const [pps, setPps] = useState(20);                  // pixels per second
  const [fit, setFit] = useState(true);
  const [viewW, setViewW] = useState(0);
  const [drag, setDrag] = useState(null);              // { i: segment, dx: px } while a clip is dragged
  const scrubbing = useRef(false);
  const press = useRef(null);                          // pointer-down on a clip: { i, x0, sl0, moved }
  const [trimDrag, setTrimDrag] = useState(null);      // { i, side, result } while an edge is dragged

  // Clips in display order (may differ from the render until re-compile)
  const layout = useMemo(() => {
    let t = 0;
    return order.map(i => { const s = segments[i], d = s.rec_out - s.rec_in, c = { i, s, start: t, dur: d }; t += d; return c; });
  }, [segments, order]);
  const total = layout.length ? layout[layout.length - 1].start + layout[layout.length - 1].dur : 0;
  const pending = order.some((v, k) => v !== k);

  // Rendered time <-> display time (they differ only while the clip order is pending)
  const toDisplay = (t) => { const c = layout.find(c => t >= c.s.rec_in && t < c.s.rec_out) || layout[layout.length - 1]; return c ? c.start + Math.min(c.dur, Math.max(0, t - c.s.rec_in)) : 0; };
  const toRendered = (x) => { const c = layout.find(c => x >= c.start && x < c.start + c.dur) || layout[layout.length - 1]; return c ? c.s.rec_in + Math.min(c.dur - 0.01, Math.max(0, x - c.start)) : 0; };

  const waves = useMemo(() => {
    if (!wave?.peaks?.length) return null;
    return segments.map(s => wavePath(wave.peaks.slice(Math.round(s.rec_in * wave.rate), Math.round(s.rec_out * wave.rate))));
  }, [segments, wave]);

  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setViewW(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (fit && viewW && total) setPps(Math.max(MIN_PPS, (viewW - 2 * PAD) / total));
  }, [fit, viewW, total]);

  const zoom = (factor, anchorX) => {
    const el = scrollRef.current;
    setFit(false);
    setPps(p => {
      const next = Math.min(MAX_PPS, Math.max(MIN_PPS, p * factor));
      if (el) {                                         // keep the time under the anchor still
        const ax = anchorX ?? el.clientWidth / 2;
        const t = (el.scrollLeft + ax - PAD) / p;
        requestAnimationFrame(() => { el.scrollLeft = t * next + PAD - ax; });
      }
      return next;
    });
  };

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onWheel = (e) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      zoom(Math.exp(-e.deltaY * 0.01), e.clientX - el.getBoundingClientRect().left);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  const playX = PAD + toDisplay(currentTime) * pps;

  // Follow the playhead when it leaves the visible window
  useEffect(() => {
    const el = scrollRef.current;
    if (!el || scrubbing.current || drag) return;
    if (playX < el.scrollLeft || playX > el.scrollLeft + el.clientWidth - 40) el.scrollLeft = playX - el.clientWidth * 0.2;
  }, [playX]);

  // Scrubbing on the ruler / empty track space
  const timeAt = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    return Math.min(total, Math.max(0, (e.clientX - r.left - PAD) / pps));
  };
  const onPointerDown = (e) => { scrubbing.current = true; e.currentTarget.setPointerCapture(e.pointerId); onSeek(toRendered(timeAt(e))); };
  const onPointerMove = (e) => { if (scrubbing.current) onSeek(toRendered(timeAt(e))); };
  const onPointerUp = () => { scrubbing.current = false; };

  // Where a dragged clip would land: its centre among the other clips' centres
  const dropIndex = (d) => {
    const c = layout.find(c => c.i === d.i);
    const centre = c.start + c.dur / 2 + d.dx / pps;
    return layout.filter(o => o.i !== d.i && o.start + o.dur / 2 < centre).length;
  };

  // While dragging near the edge of the visible window, scroll it; drag distances include the scroll
  const edgeScroll = (e) => {
    const el = scrollRef.current, r = el.getBoundingClientRect();
    if (e.clientX > r.right - 40) el.scrollLeft += 14;
    else if (e.clientX < r.left + 40) el.scrollLeft -= 14;
  };
  const dragDx = (e, p) => e.clientX - p.x0 + scrollRef.current.scrollLeft - p.sl0;

  const clipDown = (e, i) => {
    e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    press.current = { i, x0: e.clientX, sl0: scrollRef.current.scrollLeft, moved: false };
  };
  const clipMove = (e) => {
    const p = press.current;
    if (!p || p.trim) return;                          // an edge drag bubbles here too
    const dx = dragDx(e, p);
    if (!p.moved && Math.abs(dx) < DRAG_PX) return;
    p.moved = true;
    edgeScroll(e);
    setDrag({ i: p.i, dx });
  };

  // Trimming: the edge follows the pointer in source time; trim() snaps it to whole words
  const trimDown = (e, c, side) => {
    e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    press.current = { trim: true, c, side, x0: e.clientX, sl0: scrollRef.current.scrollLeft };
    setTrimDrag({ i: c.i, side, result: null });
  };
  const trimMove = (e) => {
    const p = press.current;
    if (!p?.trim) return;
    edgeScroll(e);
    const edge = (p.side === 'out' ? p.c.s.src_out : p.c.s.src_in) + dragDx(e, p) / pps;
    setTrimDrag({ i: p.c.i, side: p.side, result: trim(p.c.i, p.side, edge) });
  };
  const trimUp = (e) => {
    e.stopPropagation();
    const r = trimDrag?.result;
    press.current = null; setTrimDrag(null);
    if (r && (r.added.length || r.removed.length)) onTrim(r);
  };
  const clipUp = (e, c) => {
    const p = press.current;
    press.current = null;
    if (!p) return;
    if (!p.moved) {                                     // a click: seek to that point of the clip, select its words
      const r = e.currentTarget.getBoundingClientRect();
      onSeek(c.s.rec_in + Math.min(c.dur - 0.01, Math.max(0, (e.clientX - r.left) / pps)));
      onClipClick(c.i);
      return;
    }
    const d = { i: p.i, dx: dragDx(e, p) };
    setDrag(null);
    const rest = order.filter(x => x !== d.i), k = dropIndex(d);
    const next = [...rest.slice(0, k), d.i, ...rest.slice(k)];
    if (next.some((v, n) => v !== order[n])) onReorder(next);
  };

  const width = Math.max(viewW, total * pps + 2 * PAD);
  const step = TICK_STEPS.find(s => s * pps >= 70) || 120;
  const ticks = [];
  for (let t = 0; t <= total + 1e-6; t += step / 5) ticks.push(+t.toFixed(3));

  let marker = null;                                    // insertion point while dragging
  if (drag) {
    const k = dropIndex(drag), rest = layout.filter(c => c.i !== drag.i);
    marker = rest.slice(0, k).reduce((a, c) => a + c.dur, 0);
  }

  return (
    <div style={wrap}>
      <div style={header}>
        <span style={title}>TIMELINE</span>
        <span style={{ fontVariantNumeric: 'tabular-nums', fontSize: '0.8rem', color: 'var(--text-main)' }}>
          {fmt(currentTime)} <span style={{ color: 'var(--text-muted)' }}>/ {fmt(total)}</span></span>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{segments.length} clips</span>
        {pending
          ? <span style={{ fontSize: '0.75rem', color: 'var(--secondary)' }}>New clip order — re-compile to hear it</span>
          : <span style={{ fontSize: '0.72rem', color: 'var(--text-disabled)' }}>Drag a clip to move it</span>}
        <span style={{ flex: 1 }} />
        <button className="tl-btn" title="Zoom out" onClick={() => zoom(1 / 1.5)}><ZoomOut size={14} /></button>
        <input type="range" min={Math.log(MIN_PPS)} max={Math.log(MAX_PPS)} step="0.01" value={Math.log(pps)}
          onChange={(e) => { setFit(false); setPps(Math.exp(+e.target.value)); }} style={{ width: 110, accentColor: 'var(--primary)' }} />
        <button className="tl-btn" title="Zoom in" onClick={() => zoom(1.5)}><ZoomIn size={14} /></button>
        <button className={`tl-btn${fit ? ' active' : ''}`} title="Fit the whole edit" onClick={() => setFit(true)}><Maximize2 size={13} /> Fit</button>
      </div>

      <div style={{ display: 'flex', minHeight: 0 }}>
        <div style={{ width: LABEL_W, flexShrink: 0 }}>
          <div style={{ height: RULER_H }} />
          <div style={{ ...laneLabel, height: V_H }}>V1</div>
          <div style={{ ...laneLabel, height: A_H }}>A1</div>
        </div>
        <div ref={scrollRef} className="tl-scroll" style={{ flex: 1, overflowX: 'auto', overflowY: 'hidden' }}>
          <div onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp}
            style={{ position: 'relative', width, height: RULER_H + V_H + A_H + 6, cursor: 'text', userSelect: 'none' }}>
            {/* ruler */}
            <div style={{ position: 'absolute', left: 0, right: 0, top: 0, height: RULER_H, borderBottom: '1px solid var(--card-border)' }}>
              {ticks.map((t, i) => {
                const major = Math.abs(t / step - Math.round(t / step)) < 1e-6;
                return (
                  <div key={i} style={{ position: 'absolute', left: PAD + t * pps, bottom: 0, height: major ? 8 : 4, borderLeft: '1px solid var(--text-disabled)' }}>
                    {major && <span style={{ position: 'absolute', bottom: 9, left: 3, fontSize: '0.65rem', color: 'var(--text-muted)', whiteSpace: 'nowrap', fontVariantNumeric: 'tabular-nums' }}>{fmt(t).replace(/\.0$/, '')}</span>}
                  </div>);
              })}
            </div>

            {layout.map((c, n) => {
              const { s, i } = c;
              const fn = s.moments[0]?.function;
              const color = functionColor(fn);
              const active = currentTime >= s.rec_in && currentTime < s.rec_out;
              const text = s.word_ids.map(id => wordsById.get(id)?.text).join(' ');
              const label = s.moments.map(m => m.summary).join(' / ') || `Clip ${i + 1}`;
              const w = Math.max(2, c.dur * pps - 2);
              const dragged = drag?.i === i;
              const moved = order[n] !== n;
              const shift = dragged ? { transform: `translateX(${drag.dx}px)`, zIndex: 3, opacity: 0.85, boxShadow: '0 6px 18px rgba(0,0,0,0.5)' } : {};
              return (
                <React.Fragment key={i}>
                  {/* V1 clip */}
                  <div onPointerDown={(e) => clipDown(e, i)} onPointerMove={clipMove} onPointerUp={(e) => clipUp(e, c)}
                    title={`Clip ${i + 1} · ${fmt(c.dur)} · ${s.source_file} ${fmt(s.src_in)}–${fmt(s.src_out)}\n${s.moments.map(m => `${m.function}: ${m.summary}`).join('\n')}\n“${text}”${warnings[i] ? '\n⚠ ' + warnings[i].join('\n⚠ ') : ''}\nClick to select · drag to move`}
                    style={{
                      position: 'absolute', top: RULER_H + 4, height: V_H - 8, left: PAD + c.start * pps, width: w,
                      background: `${color}${active ? '55' : '2e'}`, borderLeft: `3px solid ${color}`, borderRadius: 4,
                      outline: selectedClip === i ? '2px solid var(--text-main)' : active ? `1px solid ${color}` : moved ? '1px dashed var(--secondary)' : 'none',
                      overflow: 'hidden', padding: w > 24 ? '4px 6px' : 0, cursor: dragged ? 'grabbing' : 'grab', ...shift,
                    }}>
                    {/* pending cuts inside this clip */}
                    {s.word_ids.filter(id => !kept.has(id)).map(id => {
                      const wd = wordsById.get(id), t = wordOut[id];
                      return t === undefined ? null : <div key={id} className="tl-cut" style={{ left: (t - s.rec_in) * pps - 3, width: Math.max(2, (wd.end - wd.start) * pps) }} />;
                    })}
                    {w > 24 && <>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '0.7rem', fontWeight: 600, color, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {warnings[i] && <AlertTriangle size={10} style={{ color: 'var(--warning)', flexShrink: 0 }} />}
                        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{label}</span>
                      </div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', marginTop: 2 }}>“{text}”</div>
                    </>}
                    {w > 16 && ['in', 'out'].map(side => (
                      <div key={side} className={`tl-handle ${side}`} title={side === 'in' ? 'Drag to trim or extend the start' : 'Drag to trim or extend the end'}
                        onPointerDown={(e) => trimDown(e, c, side)} onPointerMove={trimMove} onPointerUp={trimUp} />))}
                  </div>
                  {added[i]?.before > 0 && <span className="tl-add" style={{ top: RULER_H + 6, left: PAD + c.start * pps + 4 }} title="Words restored before this clip">+{added[i].before}</span>}
                  {added[i]?.after > 0 && <span className="tl-add" style={{ top: RULER_H + 6, left: PAD + (c.start + c.dur) * pps - 30 }} title="Words restored after this clip">+{added[i].after}</span>}

                  {/* A1: the output's waveform for this clip (or its words, without a waveform) */}
                  <div style={{ position: 'absolute', top: RULER_H + V_H + 3, height: A_H - 6, left: PAD + c.start * pps, width: w, pointerEvents: 'none', ...shift, boxShadow: 'none' }}>
                    {waves ? (
                      <svg width="100%" height="100%" viewBox={`0 0 ${Math.max(1, Math.round(c.dur * wave.rate))} 1`} preserveAspectRatio="none" style={{ display: 'block' }}>
                        <path d={waves[i]} fill={color} fillOpacity={0.6} />
                      </svg>
                    ) : s.word_ids.map(id => {
                      const wd = wordsById.get(id), t = wordOut[id];
                      if (!wd || t === undefined) return null;
                      return <div key={id} style={{ position: 'absolute', top: '25%', height: '50%', left: (t - s.rec_in) * pps, width: Math.max(1, (wd.end - wd.start) * pps - 1), background: `${color}99`, borderRadius: 2 }} />;
                    })}
                  </div>
                </React.Fragment>);
            })}

            {trimDrag?.result && (() => {                   // the snapped new edge and what it changes
              const c = layout.find(c => c.i === trimDrag.i), r = trimDrag.result;
              const x = PAD + (c.start + r.edge - c.s.src_in) * pps;
              return <>
                <div style={{ position: 'absolute', top: RULER_H + 2, height: V_H + A_H - 4, left: x - 1, width: 2, background: 'var(--secondary)', zIndex: 6, pointerEvents: 'none' }} />
                <div className="tl-trim-label" style={{ top: 2, left: Math.max(4, x - 90) }}>{r.label}</div>
              </>;
            })()}

            {marker !== null && (
              <div style={{ position: 'absolute', top: RULER_H + 2, height: V_H + A_H - 4, left: PAD + marker * pps - 2, width: 3, background: 'var(--secondary)', borderRadius: 2, zIndex: 4, pointerEvents: 'none' }} />
            )}

            {/* playhead */}
            <div style={{ position: 'absolute', top: 0, bottom: 0, left: playX, width: 0, borderLeft: '2px solid var(--danger)', pointerEvents: 'none', zIndex: 5 }}>
              <div style={{ position: 'absolute', top: 0, left: -6, width: 10, height: 10, background: 'var(--danger)', clipPath: 'polygon(0 0, 100% 0, 50% 100%)' }} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

const RULER_H = 22, V_H = 56, A_H = 38;
const wrap = { minWidth: 0, background: 'var(--card-bg)', border: '1px solid var(--card-border)', borderRadius: 'var(--radius-lg)', padding: '0.6rem 0.75rem 0.4rem' };
const header = { display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.4rem' };
const title = { fontSize: '0.72rem', fontWeight: 600, letterSpacing: '0.06em', color: 'var(--text-muted)' };
const laneLabel = { display: 'flex', alignItems: 'center', fontSize: '0.65rem', fontWeight: 600, color: 'var(--text-disabled)', letterSpacing: '0.05em' };
