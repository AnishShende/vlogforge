import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Loader2, RotateCcw, Scissors, Play, Square, AlertTriangle, CheckCircle2, Film, FileText } from 'lucide-react';

// Roadmap Phase 6: transcript review of a word-grid job. Words come from GET /api/jobs/{id}/edit;
// the edit is a set of kept word ids. Click a removed range's chip to restore it (or a kept range's
// to remove it); click a word to seek, shift-click to select a run of words, then remove/restore it.
// Re-compile posts the plan (consecutive kept words of one file = one segment) to /recompile.

const WS_BASE = import.meta.env.DEV ? 'ws://localhost:8000'
  : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`;
const PARAGRAPH_GAP_SEC = 1.0;

const keptFromPlan = (words, plan) => {
  const index = new Map(words.map((w, i) => [w.id, i]));
  const kept = new Set();
  for (const s of plan?.segments || []) {
    for (let i = index.get(s.word_start); i <= index.get(s.word_end); i++) kept.add(words[i].id);
  }
  return kept;
};

// Kept words -> plan, keeping the CURRENT plan's segment order (story order, Phase 9). A kept word
// joins the segment holding its nearest earlier word of the same file (else the next one); runs of
// grid-consecutive words become segments.
const planFromKept = (words, kept, currentPlan) => {
  const index = new Map(words.map((w, i) => [w.id, i]));
  const group = new Array(words.length).fill(-1);
  (currentPlan?.segments || []).forEach((s, g) => {
    for (let i = index.get(s.word_start); i <= index.get(s.word_end); i++) group[i] = g;
  });
  const owner = (i) => {
    for (let j = i; j >= 0 && words[j].source_file === words[i].source_file; j--) if (group[j] >= 0) return group[j];
    for (let j = i + 1; j < words.length && words[j].source_file === words[i].source_file; j++) if (group[j] >= 0) return group[j] - 0.5;
    return 1e9 + index.get(words[i].id);              // no stored order: recording order at the end
  };
  const byGroup = new Map();
  words.forEach((w, i) => { if (kept.has(w.id)) { const g = owner(i); byGroup.set(g, [...(byGroup.get(g) || []), i]); } });
  const segments = [];
  [...byGroup.keys()].sort((a, b) => a - b).forEach(g => byGroup.get(g).forEach((i, k, arr) => {
    if (k && i === arr[k - 1] + 1) segments[segments.length - 1].word_end = words[i].id;
    else segments.push({ word_start: words[i].id, word_end: words[i].id, reason: 'review' });
  }));
  return { segments };
};

const short = (t, n = 70) => (t && t.length > n ? t.slice(0, n - 1) + '…' : t);
const reasonText = (r) => {
  if (r.label === 'remove' && r.reason === 'retake') return `retake (kept take at ${r.by?.start?.toFixed(1)}s)`;
  if (r.label === 'review') return `review: ${r.by?.alternatives?.length || 0} other clean take(s)`;
  if (r.label === 'suggest') return `suggested cut · ${short(r.reason)}`;
  if (r.label === 'suggest_restore') return `restore? · ${short(r.reason)}`;
  if (r.label === 'remove' && r.review) return `⚑ check take · ${short(r.reason)}`;
  return short(r.reason);
};
const chipColors = (r) => r.label === 'remove' && !r.review ? ['rgba(239,68,68,0.12)', 'var(--danger)']
  : r.label === 'suggest' ? ['rgba(139,92,246,0.15)', '#c4b5fd']
  : r.label === 'suggest_restore' ? ['rgba(16,185,129,0.15)', 'var(--success)'] : ['rgba(245,158,11,0.12)', 'var(--warning)'];

const fmt = (s) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, '0')}`;

export default function TranscriptEditor({ jobId, onReset }) {
  const [view, setView] = useState(null);
  const [error, setError] = useState(null);
  const [kept, setKept] = useState(new Set());
  const [baseKept, setBaseKept] = useState(new Set());
  const [selection, setSelection] = useState(null);       // [fromIndex, toIndex] inclusive
  const [anchor, setAnchor] = useState(null);
  const [recompiling, setRecompiling] = useState(false);
  const [videoKey, setVideoKey] = useState(Date.now());
  const [currentTime, setCurrentTime] = useState(0);
  const [auditioning, setAuditioning] = useState(null);   // range key being played from source
  const videoRef = useRef(null);
  const sourceRef = useRef(null);

  const load = async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}/edit`);
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const v = await res.json();
      const k = keptFromPlan(v.words, v.plan);
      setView(v); setKept(k); setBaseKept(new Set(k)); setSelection(null); setAnchor(null);
    } catch (e) {
      setError(`Could not load the edit: ${e.message}`);
    }
  };

  useEffect(() => { load(); }, [jobId]);

  const words = view?.words || [];
  const index = useMemo(() => new Map(words.map((w, i) => [w.id, i])), [view]);

  // Cleanup ranges with word indices, for chips and paragraph layout
  const ranges = useMemo(() => (view?.ranges || []).map((r, n) => ({
    ...r, key: n, from: index.get(r.word_start), to: index.get(r.word_end),
  })), [view, index]);

  const changed = useMemo(() => words.filter(w => kept.has(w.id) !== baseKept.has(w.id)).length, [words, kept, baseKept]);

  // Word playing now in the output (kept words only, by their output start time)
  const playing = useMemo(() => {
    if (!view) return null;
    for (const w of words) {
      const t = view.word_out[w.id];
      if (t !== undefined && currentTime >= t && currentTime < t + (w.end - w.start) + 0.05) return w.id;
    }
    return null;
  }, [view, currentTime]);

  const setRange = (from, to, keep) => {
    const next = new Set(kept);
    for (let i = from; i <= to; i++) keep ? next.add(words[i].id) : next.delete(words[i].id);
    setKept(next);
  };

  const onWordClick = (e, i) => {
    if (e.shiftKey && anchor !== null) {
      setSelection([Math.min(anchor, i), Math.max(anchor, i)]);
      return;
    }
    setAnchor(i); setSelection([i, i]);
    const t = view.word_out[words[i].id];
    if (t !== undefined && videoRef.current) videoRef.current.currentTime = t;
  };

  const audition = (r) => {
    const v = sourceRef.current;
    if (!v) return;
    if (auditioning === r.key) { v.pause(); setAuditioning(null); return; }
    const src = `/api/jobs/${jobId}/raw/${encodeURIComponent(r.source_file)}`;
    if (!v.src.endsWith(src)) v.src = src;
    v.currentTime = Math.max(0, r.start - 0.3);
    v.ontimeupdate = () => { if (v.currentTime >= r.end + 0.3) { v.pause(); setAuditioning(null); } };
    v.onerror = () => { setAuditioning(null); setError(`Source file ${r.source_file} not available for preview`); };
    v.play(); setAuditioning(r.key);
  };

  const recompile = (override) => {
    const plan = override || planFromKept(words, kept, view.plan);
    if (!plan.segments.length) { setError('Nothing kept: restore some words first.'); return; }
    setRecompiling(true); setError(null);
    let started = false;
    const ws = new WebSocket(`${WS_BASE}/ws/${jobId}`);
    const finish = (msg) => { ws.close(); setRecompiling(false); if (msg) setError(msg); };
    ws.onmessage = (ev) => {
      const d = JSON.parse(ev.data);
      if (d.stage === 'assembling') started = true;
      else if (started && d.stage === 'complete') { finish(); setVideoKey(Date.now()); load(); }
      else if (started && d.stage === 'failed') finish(d.message);
    };
    ws.onerror = () => finish('Lost connection to the server during re-compile.');
    ws.onopen = async () => {
      const res = await fetch(`/api/jobs/${jobId}/recompile`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(plan),
      });
      if (!res.ok) finish(`Re-compile refused: ${(await res.json()).detail || res.statusText}`);
    };
  };

  if (error && !view) return <div style={{ padding: '2rem', color: 'var(--danger)' }}>{error}</div>;
  if (!view) return <div style={{ display: 'flex', justifyContent: 'center', padding: '4rem' }}><Loader2 size={32} className="spinner" style={{ color: 'var(--primary)' }} /></div>;

  const validation = view.validation || {};
  const notPass = (validation.checks || []).filter(c => c.status !== 'pass');
  const selKept = selection && words.slice(selection[0], selection[1] + 1).some(w => kept.has(w.id));
  const removedCount = words.length - kept.size;

  // Paragraphs: break at a file change or a long pause; ranges start a new chip inline
  const rangeAt = new Map();
  ranges.forEach(r => rangeAt.set(r.from, [...(rangeAt.get(r.from) || []), r]));
  const out = [];
  words.forEach((w, i) => {
    const prev = words[i - 1];
    if (prev && (prev.source_file !== w.source_file || w.start - prev.end >= PARAGRAPH_GAP_SEC)) out.push(<br key={`br${i}`} />);
    if (!prev || prev.source_file !== w.source_file) {
      out.push(<div key={`f${i}`} style={{ fontSize: '0.7rem', color: 'var(--text-muted)', letterSpacing: '0.05em', margin: '0.75rem 0 0.25rem' }}>{w.source_file}</div>);
    }
    for (const r of (rangeAt.get(i) || []).filter(x => x.label !== 'keep')) {
      const rangeKept = words.slice(r.from, r.to + 1).every(x => kept.has(x.id));
      const [bg, fg] = chipColors(r);
      out.push(
        <span key={`c${i}-${r.key}`} title={r.reason} style={{
          display: 'inline-flex', alignItems: 'center', gap: '0.25rem', margin: '0 0.25rem', padding: '0 0.4rem',
          borderRadius: 'var(--radius-sm)', fontSize: '0.7rem', verticalAlign: 'middle', background: bg, color: fg,
        }}>
          {reasonText(r)}
          <button title={rangeKept ? 'Remove this range' : 'Restore this range'} onClick={() => setRange(r.from, r.to, !rangeKept)} style={chipBtn}>
            {rangeKept ? <Scissors size={11} /> : <RotateCcw size={11} />}
          </button>
          <button title="Play this range from the source" onClick={() => audition(r)} style={chipBtn}>
            {auditioning === r.key ? <Square size={11} /> : <Play size={11} />}
          </button>
        </span>
      );
    }
    const isKept = kept.has(w.id);
    const wasKept = baseKept.has(w.id);
    const inSel = selection && i >= selection[0] && i <= selection[1];
    const inReview = ranges.some(x => (x.label === 'review' || (x.label === 'remove' && x.review)) && i >= x.from && i <= x.to);
    out.push(
      <span key={w.id} onClick={(e) => onWordClick(e, i)} style={{
        cursor: 'pointer', padding: '0 1px', borderRadius: '3px',
        textDecoration: isKept ? 'none' : 'line-through',
        color: isKept ? 'var(--text-main)' : 'var(--text-disabled)',
        background: inSel ? 'rgba(109,40,217,0.35)' : playing === w.id ? 'rgba(6,182,212,0.35)' : 'transparent',
        borderBottom: isKept !== wasKept ? '2px dashed var(--secondary)' : inReview ? '2px solid rgba(245,158,11,0.6)' : '2px solid transparent',
      }}>{w.text}</span>
    );
    out.push(' ');
  });

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(320px, 2fr) 3fr', gap: '1.5rem', padding: '1.5rem', height: '100%', minHeight: 0 }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', minWidth: 0 }}>
        <div style={panel}>
          <div style={panelTitle}><Film size={14} style={{ color: 'var(--primary)' }} /> OUTPUT · {view.duration_sec?.toFixed(1)}s</div>
          <video key={videoKey} ref={videoRef} src={`/api/jobs/${jobId}/download?t=${videoKey}`} controls
            onTimeUpdate={(e) => setCurrentTime(e.target.currentTime)} style={{ width: '100%', borderRadius: 'var(--radius-md)', background: '#000' }} />
          <video ref={sourceRef} style={{ display: 'none' }} />
        </div>

        <div style={panel}>
          <div style={panelTitle}>
            {validation.status === 'pass' ? <CheckCircle2 size={14} style={{ color: 'var(--success)' }} /> : <AlertTriangle size={14} style={{ color: validation.status === 'fail' ? 'var(--danger)' : 'var(--warning)' }} />}
            VALIDATION · {validation.status || 'unknown'}
          </div>
          {notPass.length === 0 ? <div style={muted}>All {validation.checks?.length || 0} checks pass.</div> :
            notPass.map((c, n) => <div key={n} style={{ ...muted, marginBottom: '0.25rem' }}><b style={{ color: c.status === 'fail' ? 'var(--danger)' : 'var(--warning)' }}>{c.status}</b> {c.check}{c.segment !== null && c.segment !== undefined ? ` (segment ${c.segment})` : ''}: {c.detail}</div>)}
        </div>

        <div style={panel}>
          <div style={{ ...muted, marginBottom: '0.75rem' }}>
            {words.length} words · {removedCount} removed · {changed ? `${changed} word(s) changed since last render` : 'no changes'}
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            <button className="btn btn-primary" disabled={!changed || recompiling} onClick={() => recompile()}>
              {recompiling ? <><Loader2 size={16} className="spinner" /> Re-compiling…</> : <>Re-compile{changed ? ` (${changed})` : ''}</>}
            </button>
            <button className="btn" disabled={!changed || recompiling} onClick={() => setKept(new Set(baseKept))}>Discard changes</button>
            {onReset && <button className="btn" onClick={onReset}>New project</button>}
          </div>
          {error && <div style={{ color: 'var(--danger)', fontSize: '0.8rem', marginTop: '0.75rem' }}>{error}</div>}
        </div>

        {view.story && (() => {
          const st = view.story, same = (p) => JSON.stringify(p.edit_plan.segments.map(x => [x.word_start, x.word_end]))
            === JSON.stringify((view.plan?.segments || []).map(x => [x.word_start, x.word_end]));
          const fn = new Map(st.moments.map(m => [m.id, m]));
          const best = st.plans[0];
          return (
            <div style={panel}>
              <div style={panelTitle}>STORY PLANS · full edit {st.full_sec?.toFixed(0)}s{st.target_sec ? ` · target ${st.target_sec.toFixed(0)}s` : ''}</div>
              {st.target_sec && best.duration_sec > st.target_sec * 1.15 && (
                <div style={{ ...muted, color: 'var(--warning)', marginBottom: '0.5rem' }}>
                  Target not reachable without breaking the story: shortest valid cut is {best.duration_sec.toFixed(0)}s.</div>)}
              {st.plans.map((p, k) => (
                <div key={k} title={p.reasoning} style={{ borderTop: k ? '1px solid var(--card-border)' : 'none', padding: '0.5rem 0' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                    <b style={{ fontSize: '0.85rem' }}>{k === 0 ? '★ ' : ''}{p.strategy.replace('_', ' ')}</b>
                    <span style={muted}>{p.duration_sec.toFixed(1)}s · {p.order.length}/{st.moments.length} moments</span>
                    {same(p) ? <span style={{ ...muted, color: 'var(--success)' }}>in use</span>
                      : <button className="btn" disabled={recompiling} onClick={() => recompile(p.edit_plan)}>Use this plan</button>}
                  </div>
                  <div style={{ ...muted, marginTop: '0.25rem' }}>
                    {p.order.map(id => `${fn.get(id)?.function || id}`).join(' → ')}</div>
                </div>))}
              <div style={{ ...muted, marginTop: '0.25rem' }}>Hover a plan for the editor's reasoning.</div>
            </div>);
        })()}
      </div>

      <div style={{ ...panel, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <div style={{ ...panelTitle, justifyContent: 'space-between' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}><FileText size={14} style={{ color: 'var(--primary)' }} /> TRANSCRIPT</span>
          {selection && (
            <span style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
              <span style={muted}>{selection[1] - selection[0] + 1} word(s) selected</span>
              <button className="btn" onClick={() => { setRange(selection[0], selection[1], !selKept); setSelection(null); }}>
                {selKept ? <><Scissors size={14} /> Remove</> : <><RotateCcw size={14} /> Restore</>}
              </button>
            </span>
          )}
        </div>
        <div style={{ ...muted, marginBottom: '0.75rem' }}>Click a word to jump to it · shift-click to select a run · <span style={{ color: 'var(--danger)' }}>red</span> = cut, <span style={{ color: 'var(--warning)' }}>⚑ amber</span> = take choice to check, <span style={{ color: '#c4b5fd' }}>purple</span> = suggested cut (✂ applies), <span style={{ color: 'var(--success)' }}>green</span> = suggested restore (↺ restores) · ▶ plays it from the source</div>
        <div style={{ overflowY: 'auto', lineHeight: 1.9, fontSize: '0.95rem', paddingRight: '0.5rem' }}>{out}</div>
      </div>
    </div>
  );
}

const panel = { background: 'var(--card-bg)', border: '1px solid var(--card-border)', borderRadius: 'var(--radius-lg)', padding: '1rem' };
const panelTitle = { display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.75rem', fontWeight: 600, letterSpacing: '0.05em', color: 'var(--text-muted)', marginBottom: '0.75rem' };
const muted = { fontSize: '0.8rem', color: 'var(--text-muted)' };
const chipBtn = { background: 'transparent', border: 'none', color: 'inherit', cursor: 'pointer', padding: '2px', display: 'flex' };
