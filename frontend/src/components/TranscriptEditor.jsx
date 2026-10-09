import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Loader2, RotateCcw, Scissors, Play, Square, AlertTriangle, CheckCircle2, FileText, Table, Flag, BookOpen, Eye, EyeOff, ChevronLeft, Plus, X, Undo2, Redo2, Keyboard, Download } from 'lucide-react';
import EditTimeline, { functionColor } from './EditTimeline';

// Roadmap Phase 6: review screen of a word-grid job. Words come from GET /api/jobs/{id}/edit;
// the edit is a set of kept word ids. Layout: output player + render bar (left), tabs for the
// transcript / EDL / review list / story plans (right), and the output timeline along the bottom.
// In the transcript, runs of cut words collapse to a ✂ pill (click to open); click a word to seek,
// shift-click to select a run of words, then remove/restore it. On the timeline, drag a clip to
// change the clip order. Every edit (kept words + clip order) is undoable. Re-compile posts the
// plan (consecutive kept words of one file = one segment, in clip order) to /recompile.

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
  const [tab, setTab] = useState('transcript');
  const [showCuts, setShowCuts] = useState(false);        // open every cut run
  const [opened, setOpened] = useState(new Set());        // word ids whose cut run is open
  const [showChecks, setShowChecks] = useState(false);
  const [order, setOrder] = useState([]);                 // EDL rows in display order (differs = pending reorder)
  const [past, setPast] = useState([]);                   // undo stack of { kept, order }
  const [future, setFuture] = useState([]);
  const [selectedClip, setSelectedClip] = useState(null);
  const [wave, setWave] = useState(null);                 // output waveform { rate, peaks }
  const [showKeys, setShowKeys] = useState(false);
  const keyHandler = useRef(null);
  const videoRef = useRef(null);
  const sourceRef = useRef(null);

  const load = async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}/edit`);
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const v = await res.json();
      const k = keptFromPlan(v.words, v.plan);
      setView(v); setKept(k); setBaseKept(new Set(k)); setSelection(null); setAnchor(null);
      setOrder((v.segments || []).map((_, i) => i)); setPast([]); setFuture([]); setSelectedClip(null);
    } catch (e) {
      setError(`Could not load the edit: ${e.message}`);
    }
    fetch(`/api/jobs/${jobId}/waveform`).then(r => r.ok ? r.json() : Promise.reject(r.statusText))
      .then(setWave).catch(e => { console.warn('Waveform unavailable, showing words on A1:', e); setWave(null); });
  };

  useEffect(() => { load(); }, [jobId]);

  // Keyboard shortcuts: the handler is rebuilt each render (below) so it sees current state
  useEffect(() => {
    const onKey = (e) => keyHandler.current?.(e);
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const words = view?.words || [];
  const segments = view?.segments || [];
  const index = useMemo(() => new Map(words.map((w, i) => [w.id, i])), [view]);
  const wordsById = useMemo(() => new Map(words.map(w => [w.id, w])), [view]);

  // Cleanup ranges with word indices, for chips and paragraph layout
  const ranges = useMemo(() => (view?.ranges || []).map((r, n) => ({
    ...r, key: n, from: index.get(r.word_start), to: index.get(r.word_end),
  })), [view, index]);

  const reviewItems = useMemo(() => ranges.filter(r =>
    r.label === 'suggest' || r.label === 'suggest_restore' || r.label === 'review' || (r.label === 'remove' && r.review)), [ranges]);

  // Validation problems per output segment (= EDL row), for the EDL and timeline markers
  const segWarnings = useMemo(() => {
    const m = {};
    for (const c of view?.validation?.checks || []) {
      if (c.status !== 'pass' && c.segment !== null && c.segment !== undefined) (m[c.segment] = m[c.segment] || []).push(`${c.check}: ${c.detail}`);
    }
    return m;
  }, [view]);

  const changed = useMemo(() => words.filter(w => kept.has(w.id) !== baseKept.has(w.id)).length, [words, kept, baseKept]);
  const orderChanged = order.some((v, k) => v !== k);

  // Words restored right before / after each rendered clip (shown as +N on the timeline)
  const clipAdds = useMemo(() => segments.map(s => {
    const run = (k, step, file) => { let n = 0; for (; k >= 0 && k < words.length && words[k].source_file === file && kept.has(words[k].id) && !baseKept.has(words[k].id); k += step) n++; return n; };
    const a = index.get(s.word_ids[0]), b = index.get(s.word_ids[s.word_ids.length - 1]);
    return { before: run(a - 1, -1, s.source_file), after: run(b + 1, 1, s.source_file) };
  }), [segments, kept, baseKept, index]);

  // Word playing now in the output (kept words only, by their output start time)
  const playing = useMemo(() => {
    if (!view) return null;
    for (const w of words) {
      const t = view.word_out[w.id];
      if (t !== undefined && currentTime >= t && currentTime < t + (w.end - w.start) + 0.05) return w.id;
    }
    return null;
  }, [view, currentTime]);

  const seek = (t) => { if (videoRef.current) videoRef.current.currentTime = t; setCurrentTime(t); };

  // Every edit goes through commit so it can be undone
  const commit = (next) => {
    setPast(p => [...p.slice(-199), { kept, order }]); setFuture([]);
    if (next.kept) setKept(next.kept);
    if (next.order) setOrder(next.order);
  };
  const undo = () => {
    if (!past.length) return;
    setFuture(f => [{ kept, order }, ...f]); setPast(past.slice(0, -1));
    setKept(past[past.length - 1].kept); setOrder(past[past.length - 1].order);
  };
  const redo = () => {
    if (!future.length) return;
    setPast(p => [...p, { kept, order }]); setFuture(future.slice(1));
    setKept(future[0].kept); setOrder(future[0].order);
  };

  const setRange = (from, to, keep) => {
    const next = new Set(kept);
    for (let i = from; i <= to; i++) keep ? next.add(words[i].id) : next.delete(words[i].id);
    commit({ kept: next });
  };

  // Timeline edge drag: the clip's new word range for an edge at source time `edge`. Snaps to whole
  // words (a word belongs inside when its middle is); keeps at least one word; extends only over
  // words of the same file that no other clip keeps. Returns the words it cuts / restores.
  const trimTarget = (si, side, edge) => {
    const ids = segments[si].word_ids, inClip = new Set(ids);
    const a = index.get(ids[0]), b = index.get(ids[ids.length - 1]), file = words[a].source_file;
    const mid = (w) => (w.start + w.end) / 2;
    const free = (k) => words[k].source_file === file && (inClip.has(words[k].id) || !kept.has(words[k].id));
    let a2 = a, b2 = b;
    if (side === 'out') {
      b2 = a;
      for (let k = a + 1; k < words.length && free(k) && mid(words[k]) < edge; k++) b2 = k;
    } else {
      a2 = b;
      for (let k = b - 1; k >= 0 && free(k) && mid(words[k]) > edge; k--) a2 = k;
    }
    const removed = [], added = [];
    for (let k = Math.min(a, a2); k <= Math.max(b, b2); k++) {
      const id = words[k].id, inside = k >= a2 && k <= b2;
      if (!inside && inClip.has(id) && kept.has(id)) removed.push(id);
      if (inside && !inClip.has(id) && !kept.has(id)) added.push(id);
    }
    const said = (list) => `“${short(list.map(id => wordsById.get(id).text).join(' '), 40)}”`;
    const label = removed.length ? `−${removed.length} word${removed.length > 1 ? 's' : ''} ${said(removed)}`
      : added.length ? `+${added.length} word${added.length > 1 ? 's' : ''} ${said(added)}` : 'no change';
    return { removed, added, label, edge: side === 'out' ? words[b2].end : words[a2].start };
  };
  const applyTrim = (r) => {
    const next = new Set(kept);
    r.removed.forEach(id => next.delete(id)); r.added.forEach(id => next.add(id));
    commit({ kept: next });
  };

  // Timeline clip click: select the clip's words in the transcript
  const selectClip = (i) => {
    const ids = segments[i].word_ids, a = index.get(ids[0]);
    setSelectedClip(i); setSelection([a, index.get(ids[ids.length - 1])]); setAnchor(a);
    if (tab === 'transcript') requestAnimationFrame(() =>
      document.querySelector(`[data-wid="${ids[0]}"]`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }));
  };

  const onWordClick = (e, i) => {
    if (e.shiftKey && anchor !== null) {
      setSelection([Math.min(anchor, i), Math.max(anchor, i)]);
      return;
    }
    setAnchor(i); setSelection([i, i]); setSelectedClip(null);
    const t = view.word_out[words[i].id];
    if (t !== undefined) seek(t);
  };

  // Show a range in the transcript: open its cut run, select it and scroll to it
  const reveal = (r) => {
    const id = words[r.from].id;
    setTab('transcript'); setOpened(new Set([...opened, id])); setSelection([r.from, r.to]); setAnchor(r.from);
    requestAnimationFrame(() => requestAnimationFrame(() =>
      document.querySelector(`[data-wid="${id}"]`)?.scrollIntoView({ block: 'center', behavior: 'smooth' })));
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
    // A pending clip order replaces the stored plan's order: one segment per clip, in display order
    const base = orderChanged ? { segments: order.map(i => ({
      word_start: segments[i].word_ids[0], word_end: segments[i].word_ids[segments[i].word_ids.length - 1],
      reason: 'review' })) } : view.plan;
    const plan = override || planFromKept(words, kept, base);
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
  const rangeIsKept = (r) => words.slice(r.from, r.to + 1).every(x => kept.has(x.id));
  const rangeText = (r) => r.text || words.slice(r.from, r.to + 1).map(x => x.text).join(' ');
  const pendingEdits = [changed && `${changed} word edit${changed > 1 ? 's' : ''}`, orderChanged && 'a new clip order'].filter(Boolean);
  const dirty = pendingEdits.length > 0;

  keyHandler.current = (e) => {
    const tag = e.target.tagName, v = videoRef.current, mod = e.metaKey || e.ctrlKey;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || e.target.isContentEditable) return;
    const step = (dt) => v && seek(Math.min(v.duration || view.duration_sec || 0, Math.max(0, v.currentTime + dt)));
    if (mod && (e.key === 'z' || e.key === 'Z')) { e.preventDefault(); e.shiftKey ? redo() : undo(); return; }
    if (mod && (e.key === 'y' || e.key === 'Y')) { e.preventDefault(); redo(); return; }
    if (mod || e.altKey) return;
    switch (e.key) {
      case ' ':
        if (tag === 'VIDEO') return;                 // the player handles its own space
        e.preventDefault(); document.activeElement?.blur?.();
        if (v) v.paused ? v.play() : v.pause();
        break;
      case 'k': case 'K': if (v) { v.pause(); v.playbackRate = 1; } break;
      case 'l': case 'L': if (v) { v.playbackRate = v.paused ? 1 : Math.min(4, v.playbackRate * 2); v.play(); } break;
      case 'j': case 'J': step(-5); break;
      case 'ArrowLeft': case 'ArrowRight':
        if (tag === 'VIDEO') return;
        e.preventDefault(); step((e.key === 'ArrowLeft' ? -1 : 1) * (e.shiftKey ? 5 : 1)); break;
      case ',': case '.': if (v) { v.pause(); step(e.key === ',' ? -1 / 30 : 1 / 30); } break;
      case 'Delete': case 'Backspace':
        if (!selection) return;
        e.preventDefault(); setRange(selection[0], selection[1], !selKept); setSelection(null); break;
      case 'Escape': setSelection(null); setSelectedClip(null); setShowKeys(false); break;
      case '?': setShowKeys(k => !k); break;
      default: return;
    }
  };

  // ── Transcript: kept words as text; each run of cut words is a ✂ pill unless opened ──
  const renderTranscript = () => {
    const rangeAt = new Map();
    ranges.forEach(r => rangeAt.set(r.from, [...(rangeAt.get(r.from) || []), r]));
    const out = [];
    let atBreak = true, afterPill = false;               // a closed cut run leads the text after it
    for (let i = 0; i < words.length; i++) {
      const w = words[i], prev = words[i - 1];
      if (!prev || prev.source_file !== w.source_file) {
        out.push(<div key={`f${i}`} style={fileHead}>{w.source_file}</div>); atBreak = true;
      } else if (w.start - prev.end >= PARAGRAPH_GAP_SEC && !atBreak && !afterPill) {
        out.push(<div key={`p${i}`} style={{ height: '0.7rem' }} />); atBreak = true;
      }
      const isKept = kept.has(w.id);
      if (!isKept && (!prev || kept.has(prev.id) || prev.source_file !== w.source_file)) {
        let j = i;                                       // this cut run is words i..j
        while (j + 1 < words.length && !kept.has(words[j + 1].id) && words[j + 1].source_file === w.source_file) j++;
        const run = words.slice(i, j + 1);
        if (!showCuts && !run.some(x => opened.has(x.id))) {
          const inRun = ranges.filter(r => r.label !== 'keep' && r.from <= j && r.to >= i);
          const flagged = inRun.some(r => r.review || r.label === 'review' || r.label === 'suggest_restore');
          const reasons = [...new Set(inRun.map(r => r.reason))];
          out.push(
            <button key={`cut${i}`} className={`re-pill${run.some(x => baseKept.has(x.id)) ? ' changed' : ''}`}
              title={`${run.length} word(s) cut — click to show\n“${short(run.map(x => x.text).join(' '), 160)}”${reasons.length ? '\n\n' + reasons.map(r => '• ' + short(r, 90)).join('\n') : ''}`}
              onClick={() => setOpened(new Set([...opened, w.id]))}>
              <Scissors size={10} /> {run.length}
              {flagged && <span style={{ width: 6, height: 6, borderRadius: 3, background: 'var(--warning)' }} />}
            </button>, ' ');
          atBreak = false; afterPill = true;
          i = j;
          continue;
        }
        if (!showCuts) out.push(
          <button key={`close${i}`} className="re-pill" title="Hide these cut words"
            onClick={() => { const n = new Set(opened); run.forEach(x => n.delete(x.id)); setOpened(n); }}><ChevronLeft size={11} /></button>);
      }
      for (const r of (rangeAt.get(i) || []).filter(x => x.label !== 'keep')) {
        const rangeKept = rangeIsKept(r);
        const [bg, fg] = chipColors(r);
        out.push(
          <span key={`c${i}-${r.key}`} className="re-chip" title={r.reason} style={{ background: bg, color: fg }}>
            <span>{reasonText(r)}</span>
            <button title={rangeKept ? 'Remove this range' : 'Restore this range'} onClick={() => setRange(r.from, r.to, !rangeKept)}>
              {rangeKept ? <Scissors size={11} /> : <RotateCcw size={11} />}
            </button>
            <button title="Play this range from the source" onClick={() => audition(r)}>
              {auditioning === r.key ? <Square size={11} /> : <Play size={11} />}
            </button>
          </span>
        );
      }
      const wasKept = baseKept.has(w.id);
      const inSel = selection && i >= selection[0] && i <= selection[1];
      const inReview = ranges.some(x => (x.label === 'review' || (x.label === 'remove' && x.review)) && i >= x.from && i <= x.to);
      out.push(
        <span key={w.id} data-wid={w.id} className={`re-word${isKept ? '' : ' cut'}`} onClick={(e) => onWordClick(e, i)} style={{
          background: inSel ? 'rgba(109,40,217,0.35)' : playing === w.id ? 'rgba(6,182,212,0.35)' : undefined,
          borderBottom: isKept !== wasKept ? '2px dashed var(--secondary)' : inReview ? '2px solid rgba(245,158,11,0.6)' : undefined,
        }}>{w.text}</span>, ' ');
      atBreak = false; afterPill = false;
    }
    return out;
  };

  const tabs = [
    ['transcript', <FileText size={14} />, 'Transcript', null],
    ['edl', <Table size={14} />, 'EDL', segments.length],
    ['review', <Flag size={14} />, 'Review', reviewItems.length],
    ...(view.story ? [['story', <BookOpen size={14} />, 'Story', view.story.plans.length]] : []),
  ];

  return (
    <div style={{ display: 'grid', gridTemplateRows: 'minmax(0, 1fr) auto', gridTemplateColumns: 'minmax(0, 1fr)', gap: '0.75rem', padding: '0.75rem 1rem 1rem', height: '100%', minHeight: 0 }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(340px, 5fr) minmax(0, 7fr)', gap: '0.75rem', minHeight: 0 }}>

        {/* ── Left: output player + render bar ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', minHeight: 0, minWidth: 0 }}>
          <div style={{ ...panel, flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', padding: '0.75rem' }}>
            <div style={{ flex: 1, minHeight: 0, background: '#000', borderRadius: 'var(--radius-md)', overflow: 'hidden', display: 'flex' }}>
              <video key={videoKey} ref={videoRef} src={`/api/jobs/${jobId}/download?t=${videoKey}`} controls
                onTimeUpdate={(e) => setCurrentTime(e.target.currentTime)} style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
            </div>
            <video ref={sourceRef} style={{ display: 'none' }} />
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.65rem', flexWrap: 'wrap' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-main)', fontWeight: 600 }}>{fmt(view.duration_sec || 0)}</span>
              <span style={muted}>· {segments.length} clips · {kept.size} of {words.length} words kept</span>
              <span style={{ flex: 1 }} />
              <button className="re-btn small" onClick={() => setShowChecks(!showChecks)} title="Render checks"
                style={{ color: validation.status === 'pass' ? 'var(--success)' : validation.status === 'fail' ? 'var(--danger)' : 'var(--warning)' }}>
                {validation.status === 'pass' ? <CheckCircle2 size={13} /> : <AlertTriangle size={13} />}
                {notPass.length ? `${notPass.length} check${notPass.length > 1 ? 's' : ''} to look at` : 'All checks pass'}
              </button>
            </div>
            {showChecks && notPass.length > 0 && (
              <div style={{ marginTop: '0.5rem', maxHeight: 120, overflowY: 'auto' }} className="re-scroll">
                {notPass.map((c, n) => (
                  <div key={n} style={{ ...muted, fontSize: '0.75rem', marginBottom: '0.2rem' }}>
                    <b style={{ color: c.status === 'fail' ? 'var(--danger)' : 'var(--warning)' }}>{c.status}</b> {c.check}
                    {c.segment !== null && c.segment !== undefined ? ` · clip ${c.segment + 1}` : ''}: {c.detail}</div>))}
              </div>)}
          </div>

          <div style={{ ...panel, display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: dirty ? 'var(--secondary)' : 'var(--text-main)' }}>
                {dirty ? `${pendingEdits.join(' and ')} not rendered yet` : 'Render is up to date'}</div>
              <div style={{ ...muted, fontSize: '0.75rem' }}>{dirty ? 'Re-compile to hear the change.' : 'Edit the transcript, move clips or pick a story plan.'}</div>
              {error && <div style={{ color: 'var(--danger)', fontSize: '0.75rem', marginTop: '0.25rem' }}>{error}</div>}
            </div>
            <button className="re-btn icon" title="Undo (⌘Z)" disabled={!past.length || recompiling} onClick={undo}><Undo2 size={15} /></button>
            <button className="re-btn icon" title="Redo (⇧⌘Z)" disabled={!future.length || recompiling} onClick={redo}><Redo2 size={15} /></button>
            <button className="re-btn" disabled={!dirty || recompiling} onClick={() => commit({ kept: new Set(baseKept), order: segments.map((_, i) => i) })}>Discard</button>
            <button className="re-btn primary" disabled={!dirty || recompiling} onClick={() => recompile()}>
              {recompiling ? <><Loader2 size={14} className="spinner" /> Re-compiling…</> : 'Re-compile'}
            </button>
            <span style={{ position: 'relative' }}>
              <button className="re-btn icon" title="Keyboard shortcuts (?)" onClick={() => setShowKeys(!showKeys)}><Keyboard size={15} /></button>
              {showKeys && (
                <div style={keysPop}>
                  <div style={{ ...muted, fontWeight: 600, fontSize: '0.72rem', letterSpacing: '0.05em', marginBottom: '0.5rem' }}>KEYBOARD SHORTCUTS</div>
                  {SHORTCUTS.map(([k, what]) => (
                    <div key={k} style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem', fontSize: '0.78rem', padding: '0.18rem 0' }}>
                      <span style={{ color: 'var(--text-muted)' }}>{what}</span><kbd style={kbd}>{k}</kbd></div>))}
                </div>)}
            </span>
            {onReset && <button className="re-btn icon" title="New project" onClick={onReset}><Plus size={15} /></button>}
          </div>
        </div>

        {/* ── Right: tabs ── */}
        <div style={{ ...panel, padding: 0, display: 'flex', flexDirection: 'column', minHeight: 0, minWidth: 0 }}>
          <div className="re-tabs">
            {tabs.map(([id, icon, label, count]) => (
              <button key={id} className={`re-tab${tab === id ? ' active' : ''}`} onClick={() => setTab(id)}>
                {icon} {label}
                {count !== null && <span className={`re-count${id === 'review' && count ? ' warn' : ''}`}>{count}</span>}
              </button>))}
          </div>

          {tab === 'transcript' && <>
            <div style={toolbar}>
              {selection ? <>
                <span style={muted}>{selection[1] - selection[0] + 1} word(s) selected</span>
                <button className="re-btn small" onClick={() => { setRange(selection[0], selection[1], !selKept); setSelection(null); }}>
                  {selKept ? <><Scissors size={12} /> Cut</> : <><RotateCcw size={12} /> Restore</>}
                </button>
                <button className="re-btn small icon" title="Clear selection" onClick={() => setSelection(null)}><X size={12} /></button>
              </> : <span style={muted}>Click a word to jump to it · shift-click to select a run · <Scissors size={11} style={{ verticalAlign: '-1px' }} /> = cut words, click to show</span>}
              <span style={{ flex: 1 }} />
              <button className="re-btn small" onClick={() => { setShowCuts(!showCuts); setOpened(new Set()); }}>
                {showCuts ? <><EyeOff size={12} /> Hide cuts</> : <><Eye size={12} /> Show all cuts</>}
              </button>
            </div>
            <div className="re-scroll" style={{ flex: 1, minHeight: 0, overflowY: 'auto', lineHeight: 1.9, fontSize: '0.95rem', padding: '0.25rem 1.25rem 1.25rem' }}>
              {renderTranscript()}
            </div>
          </>}

          {tab === 'edl' && (
            <>
            <div style={toolbar}>
              <span style={{ ...muted, color: dirty ? 'var(--secondary)' : 'var(--text-muted)' }}>
                {dirty ? `The last render — re-compile to include ${pendingEdits.join(' and ')}` : 'The last render, clip by clip · click a row to jump to it'}</span>
              <span style={{ flex: 1 }} />
              <span style={muted}>Export</span>
              <a className="re-btn small" href={`/api/jobs/${jobId}/export/edl`} download title="CMX3600 EDL — Premiere, Resolve, Avid"><Download size={12} /> EDL</a>
              <a className="re-btn small" href={`/api/jobs/${jobId}/export/fcpxml`} download title="FCPXML 1.9 — Final Cut Pro, Resolve"><Download size={12} /> FCPXML</a>
            </div>
            <div className="re-scroll" style={{ flex: 1, minHeight: 0, overflow: 'auto', padding: '0 0.75rem 0.75rem' }}>
              <table className="re-table">
                <colgroup><col style={{ width: 56 }} /><col style={{ width: 220 }} /><col /><col style={{ width: 90 }} />
                  {[0, 1, 2, 3, 4].map(n => <col key={n} style={{ width: 72 }} />)}</colgroup>
                <thead><tr><th>#</th><th>MOMENT</th><th>LINE</th><th>SOURCE</th><th>SRC IN</th><th>SRC OUT</th><th>REC IN</th><th>REC OUT</th><th>DUR</th></tr></thead>
                <tbody>
                  {segments.map((s, i) => {
                    const text = s.word_ids.map(id => wordsById.get(id)?.text).join(' ');
                    const active = currentTime >= s.rec_in && currentTime < s.rec_out;
                    return (
                      <tr key={i} className={active || selectedClip === i ? 'active' : ''} onClick={() => { seek(s.rec_in + 0.01); selectClip(i); }}
                        title={`${s.moments.map(m => `${m.function}: ${m.summary}`).join('\n')}\n“${text}”${segWarnings[i] ? '\n\n⚠ ' + segWarnings[i].join('\n⚠ ') : ''}`}>
                        <td style={{ color: 'var(--text-muted)' }}>{i + 1}{segWarnings[i] && <AlertTriangle size={11} style={{ color: 'var(--warning)', marginLeft: 4, verticalAlign: '-1px' }} />}</td>
                        <td><span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 2, background: functionColor(s.moments[0]?.function), marginRight: 6 }} />{s.moments.map(m => m.summary).join(' / ') || '—'}</td>
                        <td style={{ color: 'var(--text-muted)' }}>{text}</td>
                        <td style={{ color: 'var(--text-muted)' }}>{s.source_file.replace(/\.[^.]+$/, '')}</td>
                        <td>{fmt(s.src_in)}</td><td>{fmt(s.src_out)}</td>
                        <td>{fmt(s.rec_in)}</td><td>{fmt(s.rec_out)}</td>
                        <td style={{ color: 'var(--text-muted)' }}>{(s.rec_out - s.rec_in).toFixed(1)}s</td>
                      </tr>);
                  })}
                </tbody>
              </table>
              {segments.length === 0 && <div style={{ ...muted, padding: '1rem 0.5rem' }}>No render yet.</div>}
            </div>
            </>
          )}

          {tab === 'review' && (
            <div className="re-scroll" style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '0.25rem 1.25rem 1rem' }}>
              {reviewItems.length === 0 && <div style={{ ...muted, padding: '1rem 0' }}>Nothing to check.</div>}
              {REVIEW_GROUPS.map(([title, hint, test, color]) => {
                const items = reviewItems.filter(test);
                if (!items.length) return null;
                return (
                  <div key={title} style={{ marginTop: '0.9rem' }}>
                    <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
                      <span style={{ fontSize: '0.75rem', fontWeight: 600, letterSpacing: '0.05em', color }}>{title.toUpperCase()} · {items.length}</span>
                      <span style={{ ...muted, fontSize: '0.72rem' }}>{hint}</span>
                    </div>
                    {items.map(r => {
                      const rk = rangeIsKept(r);
                      return (
                        <div key={r.key} className="re-item">
                          <span style={{ width: 3, alignSelf: 'stretch', borderRadius: 2, background: color, flexShrink: 0 }} />
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{ fontSize: '0.88rem', color: rk ? 'var(--text-main)' : 'var(--text-muted)', textDecoration: rk ? 'none' : 'line-through' }}>“{short(rangeText(r), 140)}”</div>
                            <div style={{ ...muted, fontSize: '0.74rem', marginTop: '0.15rem' }}>{r.reason}</div>
                          </div>
                          <button className="re-btn small icon" title="Play from the source" onClick={() => audition(r)}>{auditioning === r.key ? <Square size={12} /> : <Play size={12} />}</button>
                          <button className="re-btn small" onClick={() => reveal(r)}>Show</button>
                          <button className="re-btn small" style={{ minWidth: 78, justifyContent: 'center' }} onClick={() => setRange(r.from, r.to, !rk)}>
                            {rk ? <><Scissors size={12} /> Cut</> : <><RotateCcw size={12} /> Restore</>}</button>
                        </div>);
                    })}
                  </div>);
              })}
            </div>
          )}

          {tab === 'story' && view.story && (() => {
            const st = view.story, same = (p) => JSON.stringify(p.edit_plan.segments.map(x => [x.word_start, x.word_end]))
              === JSON.stringify((view.plan?.segments || []).map(x => [x.word_start, x.word_end]));
            const fn = new Map(st.moments.map(m => [m.id, m]));
            const best = st.plans[0];
            return (
              <div className="re-scroll" style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '0.75rem 1.25rem 1rem' }}>
                <div style={muted}>Full edit {st.full_sec?.toFixed(0)}s{st.target_sec ? ` · target ${st.target_sec.toFixed(0)}s` : ''}</div>
                {st.target_sec && best.duration_sec > st.target_sec * 1.15 && (
                  <div style={{ ...muted, color: 'var(--warning)', marginTop: '0.25rem' }}>
                    Target not reachable without breaking the story: shortest valid cut is {best.duration_sec.toFixed(0)}s.</div>)}
                {st.plans.map((p, k) => (
                  <div key={k} style={{ border: '1px solid var(--card-border)', borderRadius: 'var(--radius-md)', padding: '0.75rem', marginTop: '0.75rem', background: same(p) ? 'rgba(16,185,129,0.05)' : 'transparent' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                      <b style={{ fontSize: '0.88rem', textTransform: 'capitalize' }}>{k === 0 ? '★ ' : ''}{p.strategy.replace('_', ' ')}</b>
                      <span style={muted}>{p.duration_sec.toFixed(1)}s · {p.order.length}/{st.moments.length} moments</span>
                      <span style={{ flex: 1 }} />
                      {same(p) ? <span style={{ ...muted, color: 'var(--success)', display: 'flex', alignItems: 'center', gap: 4 }}><CheckCircle2 size={13} /> In use</span>
                        : <button className="re-btn small" disabled={recompiling} onClick={() => recompile(p.edit_plan)}>Use this plan</button>}
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem', marginTop: '0.55rem' }}>
                      {p.order.map((id, n) => {
                        const m = fn.get(id), f = m?.function || id;
                        return <span key={n} title={`${f}: ${m?.summary || ''}`} style={{ maxWidth: 280, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: '0.72rem', padding: '0.15rem 0.5rem', borderRadius: 999, background: `${functionColor(f)}26`, color: functionColor(f) }}>{n + 1}. {m?.summary || f}</span>;
                      })}
                    </div>
                    {p.reasoning && <div style={{ ...muted, fontSize: '0.75rem', marginTop: '0.55rem' }}>{p.reasoning}</div>}
                  </div>))}
              </div>);
          })()}
        </div>
      </div>

      {segments.length > 0 && (
        <EditTimeline segments={segments} order={order} wordsById={wordsById} wordOut={view.word_out} wave={wave}
          currentTime={currentTime} onSeek={seek} warnings={segWarnings} selectedClip={selectedClip}
          onClipClick={selectClip} onReorder={(next) => commit({ order: next })}
          kept={kept} added={clipAdds} trim={trimTarget} onTrim={applyTrim} />
      )}
    </div>
  );
}

const SHORTCUTS = [
  ['Space', 'Play / pause'], ['J', 'Back 5 s'], ['K', 'Pause'], ['L', 'Play · faster (1×, 2×, 4×)'],
  ['← →', 'Back / forward 1 s (⇧ 5 s)'], [', .', 'Step one frame'], ['⌫', 'Cut / restore the selection'],
  ['⌘Z', 'Undo'], ['⇧⌘Z', 'Redo'], ['Esc', 'Clear selection'], ['?', 'Show these shortcuts'],
];
const keysPop = { position: 'absolute', bottom: 'calc(100% + 8px)', right: 0, zIndex: 20, width: 280, padding: '0.75rem 0.9rem',
  background: '#1c1c1c', border: '1px solid var(--card-border)', borderRadius: 'var(--radius-md)', boxShadow: '0 12px 32px rgba(0,0,0,0.5)' };
const kbd = { fontFamily: 'inherit', fontSize: '0.7rem', padding: '0.05rem 0.4rem', borderRadius: 4, border: '1px solid #444', background: '#262626', color: 'var(--text-main)', whiteSpace: 'nowrap' };
const REVIEW_GROUPS = [
  ['Suggested cuts', 'the editor would remove these', r => r.label === 'suggest', '#c4b5fd'],
  ['Suggested restores', 'the editor would bring these back', r => r.label === 'suggest_restore', 'var(--success)'],
  ['Take choices to check', 'close calls between takes', r => r.label === 'review' || (r.label === 'remove' && r.review), 'var(--warning)'],
];
const panel = { background: 'var(--card-bg)', border: '1px solid var(--card-border)', borderRadius: 'var(--radius-lg)', padding: '1rem' };
const toolbar = { display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.6rem 1.25rem', minHeight: 44, borderBottom: '1px solid #2a2a2a' };
const fileHead = { fontSize: '0.68rem', fontWeight: 600, color: 'var(--text-muted)', letterSpacing: '0.06em', margin: '0.75rem 0 0.3rem' };
const muted = { fontSize: '0.8rem', color: 'var(--text-muted)' };
