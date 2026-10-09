import React, { useState, useEffect, useRef } from 'react';
import { Download, RefreshCw, Copy, Check, FileText, Loader2, AlertTriangle, Plus, CheckCircle2, Play, Pause } from 'lucide-react';

// Export panel of a word-grid job: the post copy for the last render (GET /api/jobs/{id}/publish) and
// the files: the video in standard layouts (16:9 is the main render; 9:16, 1:1, 4:5 render on demand
// from the same edit, GET/POST /api/jobs/{id}/layouts) and the EDL / FCPXML. Under a minute: 3 Instagram caption variations + a grouped hashtag
// block; from a minute: 3 YouTube title styles, the 3-part description with the chapter timeline,
// pinned-comment options and hashtags. Every text is editable before copying; the selected
// variation / title is what "Copy all" uses.

const FORMATS = [['short', 'Instagram Reel'], ['long', 'YouTube']];
const IG_CAPTION_MAX = 2200, YT_TITLE_MAX = 100, YT_DESC_MAX = 5000;
const TAG_GROUPS = [['niche', 'Niche / audience'], ['content', 'Content'], ['broad', 'Broad']];
const len = (s) => { const t = Math.round(s || 0); return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}`; };
const join = (...parts) => parts.map(p => (p || '').trim()).filter(Boolean).join('\n\n');

// Editable texts of a copy response, keyed for the inputs
const fieldsOf = (d) => d.format === 'short'
  ? Object.fromEntries(d.variations.map((v, i) => [`v${i}`, join(v.hook, v.body, v.cta)]))
  : { ...Object.fromEntries(d.titles.map((t, i) => [`t${i}`, t.text])),
      ...Object.fromEntries(d.pinned_comments.map((c, i) => [`p${i}`, c])),
      desc: join(d.description_hook, d.description_outline), links: d.links };

// Preview of the edit in a layout. A rendered layout plays its own file. Otherwise it is simulated
// from the 16:9 main render: every layout fits each clip whole with black bars, so the clip's area
// of the main frame (known from the source's shape), scaled to fit the layout frame, is exactly what
// the layout file will show (same frames and audio, lower resolution). Per clip, so mixed shapes work.
const STAGE_W = 560, STAGE_H = 300;

function LayoutPreview({ jobId, layout, segments, sources, mainSize }) {
  const ref = useRef(null);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [dur, setDur] = useState(0);
  const poster = useRef(false);                       // resting on a still past the fade-in, not yet played
  const real = layout.status === 'ready';
  const src = layout.key === '16x9' ? `/api/jobs/${jobId}/download` : real ? `/api/jobs/${jobId}/layouts/${layout.key}/download` : `/api/jobs/${jobId}/download`;

  // Follow the playing frame (per video frame where supported, so the framing switches on the cut)
  useEffect(() => {
    const v = ref.current;
    if (!v) return;
    let id, stop = false;
    const tick = (_, meta) => { if (stop) return; setT(meta ? meta.mediaTime : v.currentTime); id = v.requestVideoFrameCallback?.(tick); };
    if (v.requestVideoFrameCallback) id = v.requestVideoFrameCallback(tick);
    const onTime = () => !v.requestVideoFrameCallback && setT(v.currentTime);
    v.addEventListener('timeupdate', onTime);
    return () => { stop = true; v.removeEventListener('timeupdate', onTime); if (id) v.cancelVideoFrameCallback?.(id); };
  }, [src]);

  const [w, h] = layout.size, ac = w / h;
  const [cw, chh] = ac >= STAGE_W / STAGE_H ? [STAGE_W, STAGE_W / ac] : [STAGE_H * ac, STAGE_H];   // the layout frame
  let box = { left: 0, top: 0, width: cw, height: chh };
  if (!real) {
    const seg = segments.find(s => t >= s.rec_in && t < s.rec_out) || segments[0];
    const [sw, sh] = (seg && sources[seg.source_file]) || mainSize;
    const r = sw / sh, am = mainSize[0] / mainSize[1];
    const [pw, ph] = r < ac ? [chh * r, chh] : [cw, cw / r];          // the clip fitted in the layout frame
    const [fx, fy] = r < am ? [r / am, 1] : [1, am / r];              // the clip's share of the main frame
    const vw = pw / fx, vh = ph / fy;
    box = { left: (cw - vw) / 2, top: (chh - vh) / 2, width: vw, height: vh };
  }
  const toggle = () => {
    const v = ref.current;
    if (!v) return;
    if (v.paused && poster.current) { v.currentTime = 0; poster.current = false; }   // play from the start
    v.paused ? v.play() : v.pause();
  };
  const onMeta = (e) => {                             // the render fades in from black: show a frame after it
    const v = e.target;
    setDur(v.duration);
    if (v.currentTime === 0) { v.currentTime = Math.min(1, v.duration / 2); poster.current = true; }
  };

  return (
    <div style={{ marginTop: '0.9rem', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--card-border)', background: '#161616' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: '1rem', marginBottom: '0.75rem', flexWrap: 'wrap' }}>
        <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Preview · {layout.ratio} {layout.name}</span>
        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
          {real ? `the rendered ${layout.size[0]}×${layout.size[1]} file` : 'simulated from your 16:9 render; the rendered file will look the same'}</span>
      </div>
      <div style={{ height: STAGE_H, display: 'grid', placeItems: 'center' }}>
        <div onClick={toggle} style={{ position: 'relative', width: cw, height: chh, background: '#000', overflow: 'hidden', borderRadius: 4, outline: '1px solid #333', cursor: 'pointer' }}>
          <video key={src} ref={ref} src={src} playsInline preload="metadata"
            onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onLoadedMetadata={onMeta}
            style={{ position: 'absolute', ...box, objectFit: 'fill', display: 'block' }} />
        </div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginTop: '0.75rem' }}>
        <button className="re-btn small icon" onClick={toggle} title={playing ? 'Pause' : 'Play'}>{playing ? <Pause size={13} /> : <Play size={13} />}</button>
        <input type="range" min={0} max={dur || 0} step="0.01" value={Math.min(t, dur || 0)} style={{ flex: 1, accentColor: 'var(--primary)' }}
          onChange={(e) => { const v = ref.current; if (v) { poster.current = false; v.currentTime = +e.target.value; setT(+e.target.value); } }} />
        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums', minWidth: 72, textAlign: 'right' }}>{len(t)} / {len(dur)}</span>
      </div>
    </div>);
}

export default function PublishPanel({ jobId, onReset }) {
  const [info, setInfo] = useState(null);             // { duration, clips }
  const [copy, setCopy] = useState(null);
  const [fields, setFields] = useState({});
  const [pick, setPick] = useState(0);                // selected variation (short) / title (long)
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(null);
  const [requested, setRequested] = useState('auto');   // format asked for (shown even when the call fails)
  const [layouts, setLayouts] = useState(null);
  const [layoutMeta, setLayoutMeta] = useState({ sources: {}, main: [1920, 1080] });
  const [previewKey, setPreviewKey] = useState(null);   // layout shown in the preview (default: the suggested one)
  const [layoutError, setLayoutError] = useState(null);

  const loadLayouts = () => fetch(`/api/jobs/${jobId}/layouts`)
    .then(async r => { const d = await r.json(); if (!r.ok) throw new Error(d.detail || r.statusText); setLayouts(d.layouts); setLayoutMeta({ sources: d.sources || {}, main: d.main_size || [1920, 1080] }); })
    .catch(e => setLayoutError(e.message));
  const renderLayout = async (key) => {
    setLayouts(ls => ls.map(l => l.key === key ? { ...l, status: 'rendering', error: null } : l));
    const r = await fetch(`/api/jobs/${jobId}/layouts/${key}/render`, { method: 'POST' });
    if (!r.ok) setLayoutError((await r.json()).detail || r.statusText);
    loadLayouts();
  };

  const load = async (format = 'auto', regenerate = false) => {
    setLoading(true); setError(null); setRequested(format);
    try {
      const r = await fetch(`/api/jobs/${jobId}/publish?format=${format}${regenerate ? '&regenerate=true' : ''}`);
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || r.statusText);
      setCopy(d); setFields(fieldsOf(d)); setPick(0);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetch(`/api/jobs/${jobId}/edit`).then(r => r.ok ? r.json() : null)
      .then(v => v && setInfo({ duration: v.duration_sec, clips: v.segments?.length || 0, segments: v.segments || [] })).catch(() => {});
    load(); loadLayouts();
  }, [jobId]);

  // Poll while a layout renders
  const rendering = layouts?.some(l => l.status === 'rendering');
  useEffect(() => {
    if (!rendering) return;
    const t = setInterval(loadLayouts, 2000);
    return () => clearInterval(t);
  }, [rendering, jobId]);

  const doCopy = (key, text) => navigator.clipboard.writeText(text).then(() => { setCopied(key); setTimeout(() => setCopied(null), 1600); });
  const CopyBtn = ({ k, text, label = 'Copy' }) => (
    <button className="re-btn small" onClick={(e) => { e.stopPropagation(); doCopy(k, text); }} disabled={!text}>
      {copied === k ? <><Check size={12} style={{ color: 'var(--success)' }} /> Copied</> : <><Copy size={12} /> {label}</>}
    </button>);
  const set = (k) => (e) => setFields({ ...fields, [k]: e.target.value });
  const Count = ({ n, max, warn = max }) => <span style={{ ...count, color: n > warn ? 'var(--warning)' : 'var(--text-disabled)' }}>{n}/{max}</span>;

  const duration = copy?.duration_sec ?? info?.duration;
  const auto = copy?.auto_format ?? (duration ? (duration < 60 ? 'short' : 'long') : null);
  const format = copy?.format ?? (requested === 'auto' ? auto : requested);
  const shown = layouts && (layouts.find(l => l.key === previewKey) || layouts.find(l => l.key === (auto === 'short' ? '9x16' : '16x9')) || layouts[0]);
  const tags = (copy?.hashtags || []).join(' ');
  const chapters = (copy?.chapters || []).map(c => `${c.time} ${c.label}`).join('\n');
  const fullText = !copy ? '' : copy.format === 'short'
    ? join(fields[`v${pick}`], tags)
    : join(fields.desc, chapters, fields.links, tags);

  // A selectable card: a variation (short) or a title (long). Called as a function, not <Choice>:
  // a component defined in render would remount its textarea on every keystroke (focus lost).
  const choice = ({ key, i, title, hint, right, selectable = true }, children) => (
    <div key={key} className={`pp-choice${selectable && pick === i ? ' active' : ''}`} style={selectable ? undefined : { cursor: 'default' }}
      onClick={selectable ? () => setPick(i) : undefined}>
      <div style={labelRow}>
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', minWidth: 0 }}>
          {selectable && <span className="pp-radio" />}
          <span style={labelText}>{title}</span>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{hint}</span>
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexShrink: 0 }}>{right}</span>
      </div>
      {children}
    </div>);

  const heading = (children, right) => (
    <div style={{ ...labelRow, marginTop: '1.4rem' }}><span style={subTitle}>{children}</span>{right}</div>);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* ── Header: what was made, and the video ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap', paddingRight: '2.5rem' }}>
        <div style={{ width: 44, height: 44, borderRadius: 12, background: 'rgba(16,185,129,0.12)', display: 'grid', placeItems: 'center', flexShrink: 0 }}>
          <CheckCircle2 size={22} style={{ color: 'var(--success)' }} />
        </div>
        <div style={{ flex: 1, minWidth: 200 }}>
          <div style={{ fontSize: '1.15rem', fontWeight: 700 }}>Ready to post</div>
          <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: 2 }}>
            {duration ? `${len(duration)} long` : '—'}{info ? ` · ${info.clips} clips` : ''}
            {auto && ` · ${auto === 'short' ? 'under a minute: Instagram Reel' : 'a minute or more: YouTube'}`}
          </div>
        </div>
        {onReset && <button className="re-btn" onClick={onReset}><Plus size={15} /> New project</button>}
      </div>

      {/* ── Video: one MP4 per layout ── */}
      <section>
        <div style={{ ...sectionTitle, marginBottom: '0.6rem' }}>VIDEO <span style={{ fontWeight: 400, letterSpacing: 0 }}>· same edit, pick the shapes you need</span></div>
        {layoutError && <div style={{ ...stateBox, marginTop: 0, marginBottom: '0.6rem', color: 'var(--danger)' }}><AlertTriangle size={16} /> {layoutError}</div>}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: '0.75rem' }}>
          {(layouts || []).map(l => {
            const [w, h] = l.size, k = 40 / Math.max(w, h);
            const suggested = (auto === 'short' && l.key === '9x16') || (auto === 'long' && l.key === '16x9');
            return (
              <div key={l.key} className={`pp-layout${suggested ? ' suggested' : ''}${shown?.key === l.key ? ' previewing' : ''}`}
                onClick={() => setPreviewKey(l.key)} title="Click to preview this layout">
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
                  <div style={{ width: 44, height: 44, display: 'grid', placeItems: 'center' }}>
                    <div style={{ width: w * k, height: h * k, border: '2px solid var(--text-muted)', borderRadius: 3 }} />
                  </div>
                  {suggested && <span className="pp-badge">Suggested</span>}
                </div>
                <div style={{ marginTop: '0.5rem', fontWeight: 600, fontSize: '0.9rem' }}>{l.ratio} {l.name}</div>
                <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>{w}×{h}</div>
                <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginTop: 2 }}>{l.for}</div>
                {l.status === 'stale' && <div style={{ fontSize: '0.72rem', color: 'var(--warning)', marginTop: '0.35rem' }}>Outdated: made from an older edit</div>}
                <div style={{ marginTop: 'auto', paddingTop: '0.7rem' }}>
                  {l.status === 'ready' && <a className="re-btn small primary pp-full" href={`/api/jobs/${jobId}/layouts/${l.key}/download`} download><Download size={12} /> Download</a>}
                  {l.status === 'rendering' && <button className="re-btn small pp-full" disabled><Loader2 size={12} className="spinner" /> Rendering…</button>}
                  {l.status === 'none' && <button className="re-btn small pp-full" onClick={() => renderLayout(l.key)}>Render</button>}
                  {l.status === 'stale' && <button className="re-btn small pp-full" onClick={() => renderLayout(l.key)} title="Made from an older version of the edit"><RefreshCw size={12} /> Re-render</button>}
                  {l.status === 'failed' && <button className="re-btn small pp-full" style={{ color: 'var(--danger)' }} onClick={() => renderLayout(l.key)} title={l.error}><AlertTriangle size={12} /> Failed · retry</button>}
                </div>
                {l.status === 'failed' && <div style={{ fontSize: '0.7rem', color: 'var(--danger)', marginTop: '0.35rem', wordBreak: 'break-word' }}>{l.error}</div>}
              </div>);
          })}
        </div>
        {shown && info?.segments && <LayoutPreview jobId={jobId} layout={shown} segments={info.segments} sources={layoutMeta.sources} mainSize={layoutMeta.main} />}
        <div style={{ fontSize: '0.72rem', color: 'var(--text-disabled)', marginTop: '0.5rem' }}>Click a layout to preview it. Footage of a different shape is fitted whole, with black bars. A new layout takes about as long as a re-compile.</div>
      </section>

      {/* ── Post copy ── */}
      <section style={card}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          <span style={sectionTitle}>POST COPY</span>
          <div className="pp-seg">
            {FORMATS.map(([f, name]) => (
              <button key={f} className={format === f ? 'active' : ''} disabled={loading} onClick={() => format !== f && load(f)}>
                {name}{auto === f && <span style={{ opacity: 0.6 }}> · auto</span>}
              </button>))}
          </div>
          <span style={{ flex: 1 }} />
          <button className="re-btn small" disabled={loading || !copy} onClick={() => load(format, true)} title="Write a different version">
            <RefreshCw size={12} className={loading ? 'spinner' : ''} /> Regenerate</button>
        </div>

        {loading && !copy && <div style={stateBox}><Loader2 size={18} className="spinner" style={{ color: 'var(--primary)' }} /> Writing your {format === 'long' ? 'YouTube metadata' : 'Reel captions'} from the final cut…</div>}
        {error && (
          <div style={{ ...stateBox, color: 'var(--danger)', justifyContent: 'space-between' }}>
            <span style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}><AlertTriangle size={16} style={{ flexShrink: 0, marginTop: 2 }} /> {error}</span>
            <button className="re-btn small" style={{ whiteSpace: 'nowrap', flexShrink: 0 }} onClick={() => load(requested)}>Try again</button>
          </div>)}

        {copy && (
          <div style={{ opacity: loading ? 0.5 : 1, transition: 'opacity 0.2s' }}>
            <div style={{ ...count, color: 'var(--text-muted)', marginTop: '0.6rem' }}>
              Written for {copy.audience || '—'}{copy.format === 'short' ? ` · tone: ${copy.tone || '—'}` : ` · keyword: ${copy.keyword || '—'}`}
            </div>

            {copy.format === 'short' ? <>
              {heading(<>Caption — pick one</>)}
              <div style={{ display: 'grid', gap: '0.6rem' }}>
                {copy.variations.map((v, i) => (
                  choice({ key: v.key, i, title: v.name, hint: v.best_for,
                    right: <><Count n={fields[`v${i}`]?.length || 0} max={IG_CAPTION_MAX} /><CopyBtn k={`v${i}`} text={fields[`v${i}`]} /></> },
                    <textarea className="pp-input" rows={Math.min(8, (fields[`v${i}`] || '').split('\n').length + 1)} value={fields[`v${i}`] || ''}
                      onChange={set(`v${i}`)} onClick={(e) => e.stopPropagation()} onFocus={() => setPick(i)} />)))}
              </div>
              {heading(<>Hashtags</>, <CopyBtn k="tags" text={tags} label={`Copy all ${copy.hashtags.length}`} />)}
              <div className="pp-box" style={{ display: 'grid', gap: '0.55rem' }}>
                {TAG_GROUPS.map(([g, name]) => (
                  <div key={g} style={{ display: 'flex', gap: '0.75rem', alignItems: 'baseline' }}>
                    <span style={{ ...count, color: 'var(--text-muted)', width: 110, flexShrink: 0 }}>{name}</span>
                    <span style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem' }}>
                      {(copy.hashtag_groups[g] || []).map(h => <span key={h} className="pp-tag">{h}</span>)}
                      {!(copy.hashtag_groups[g] || []).length && <span style={count}>—</span>}
                    </span>
                  </div>))}
              </div>
            </> : <>
              {heading(<>Title <span style={hintText}>· 3 styles</span></>)}
              <div style={{ display: 'grid', gap: '0.6rem' }}>
                {copy.titles.map((t, i) => (
                  choice({ key: t.key, i, title: t.name, hint: t.hint, selectable: false,
                    right: <><Count n={fields[`t${i}`]?.length || 0} max={YT_TITLE_MAX} warn={t.key === 'curiosity' ? 50 : YT_TITLE_MAX} /><CopyBtn k={`t${i}`} text={fields[`t${i}`]} /></> },
                    <input className="pp-input" value={fields[`t${i}`] || ''} onChange={set(`t${i}`)} />)))}
              </div>

              {heading(<>Description <span style={hintText}>· hook + overview</span></>,
                <span style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}><Count n={fullText.length} max={YT_DESC_MAX} /><CopyBtn k="desc" text={fields.desc} /></span>)}
              <textarea className="pp-input" rows={9} value={fields.desc || ''} onChange={set('desc')} />

              {heading(<>Timeline</>, <CopyBtn k="chapters" text={chapters} />)}
              {copy.chapters.length
                ? <div className="pp-box" style={{ fontVariantNumeric: 'tabular-nums' }}>
                    {copy.chapters.map((c, i) => (
                      <div key={i} style={{ display: 'flex', gap: '0.9rem', padding: '0.15rem 0' }}>
                        <span style={{ color: 'var(--secondary)', minWidth: 44 }}>{c.time}</span><span>{c.label}</span></div>))}
                  </div>
                : <div className="pp-box" style={{ color: 'var(--text-muted)' }}>No chapters: {copy.chapter_note}.</div>}

              {heading(<>Links <span style={hintText}>· fill in your own</span></>, <CopyBtn k="links" text={fields.links} />)}
              <textarea className="pp-input" rows={4} value={fields.links || ''} onChange={set('links')} />

              {heading(<>Hashtags</>, <CopyBtn k="tags" text={tags} />)}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                {copy.hashtags.map(h => <span key={h} className="pp-tag">{h}</span>)}
              </div>

              {heading(<>Pinned comment — 2 options</>)}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '0.6rem' }}>
                {copy.pinned_comments.map((_, i) => (
                  <div key={i} className="pp-choice" style={{ cursor: 'default' }}>
                    <div style={labelRow}><span style={labelText}>Option {i + 1}</span><CopyBtn k={`p${i}`} text={fields[`p${i}`]} /></div>
                    <textarea className="pp-input" rows={3} value={fields[`p${i}`] || ''} onChange={set(`p${i}`)} />
                  </div>))}
              </div>
            </>}

            <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: '0.75rem', marginTop: '1.4rem' }}>
              <span style={{ ...count, color: 'var(--text-muted)' }}>
                {copy.format === 'short' ? `Variation ${pick + 1} + hashtags` : 'Description + timeline + links + hashtags'}</span>
              <button className="re-btn primary" onClick={() => doCopy('all', fullText)}>
                {copied === 'all' ? <><Check size={14} /> Copied</> : <><Copy size={14} /> {copy.format === 'short' ? 'Copy caption' : 'Copy full description'}</>}
              </button>
            </div>
          </div>)}
      </section>

      {/* ── Edit files ── */}
      <section>
        <div style={{ ...sectionTitle, marginBottom: '0.6rem' }}>EDIT FILES <span style={{ fontWeight: 400, letterSpacing: 0 }}>· to finish in another editor</span></div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '0.75rem' }}>
          {[[`/api/jobs/${jobId}/export/edl`, 'EDL', 'CMX3600 · Premiere, Resolve, Avid'],
            [`/api/jobs/${jobId}/export/fcpxml`, 'FCPXML', 'Final Cut Pro, Resolve']].map(([href, name, sub]) => (
            <a key={name} className="pp-file" href={href} download>
              <span style={{ color: 'var(--primary)', display: 'flex' }}><FileText size={18} /></span>
              <span style={{ flex: 1, minWidth: 0 }}>
                <span style={{ display: 'block', fontWeight: 600, fontSize: '0.88rem' }}>{name}</span>
                <span style={{ display: 'block', fontSize: '0.74rem', color: 'var(--text-muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{sub}</span>
              </span>
              <Download size={15} style={{ color: 'var(--text-muted)' }} />
            </a>))}
        </div>
      </section>
    </div>
  );
}

const card = { border: '1px solid var(--card-border)', borderRadius: 'var(--radius-lg)', padding: '1.1rem 1.25rem 1.25rem', background: 'var(--card-bg)' };
const sectionTitle = { fontSize: '0.72rem', fontWeight: 600, letterSpacing: '0.06em', color: 'var(--text-muted)' };
const labelRow = { display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.4rem' };
const labelText = { fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-main)', whiteSpace: 'nowrap' };
const subTitle = { fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)' };
const hintText = { fontSize: '0.75rem', fontWeight: 400, color: 'var(--text-muted)' };
const count = { fontSize: '0.7rem', fontVariantNumeric: 'tabular-nums' };
const stateBox = { display: 'flex', alignItems: 'center', gap: '0.6rem', marginTop: '1rem', padding: '0.9rem 1rem', borderRadius: 'var(--radius-md)', background: 'rgba(255,255,255,0.03)', fontSize: '0.85rem', color: 'var(--text-muted)' };
