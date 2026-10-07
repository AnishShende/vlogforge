"""Verbatim take-picking page for a job: the whole recording word by word (as the ASR heard it,
stutters and repeats included), split into numbered attempts at pauses, each playable from the
source. Starts from the job's current edit; the user keeps the best attempt of each line, cuts
the rest, and can cut single words (stutters, false starts). Exports the choices as JSON.
Local only (git-ignored location); nothing is uploaded.

Each attempt lists the other attempts that share its wording ("also said in #12, #15"): pairwise
links, not groups, because restarts often glue the end of one sentence to the next.
Optional second verbatim ASR (e.g. CrisperWhisper from the bake-off) shown per attempt when it
heard something different (off by default).

Usage (from backend/):
    PYTHONPATH=. python -m eval.verbatim_page --job <job_id> [--clip IMG_1614] \
        [--alt ../eval-set/_work/asr/F_large.json] [--pause 0.4]
Writes eval-set/<clip>/verbatim.html with --clip, else artifacts/jobs/<job>/verbatim.html.
"""

import argparse
import difflib
import html
import json
import os
import re
from typing import Dict, List, Set

from app.config import settings
from app.utils import artifacts
from eval.gold import EVAL_SET_DIR

LOW_CONF = 0.3
SAME_WORDING = 0.5      # share of the shorter attempt's word pairs found in the other attempt


def norm(t: str) -> str:
    return re.sub(r"[^\w'-]", "", t.lower())


def lines_of(words: List[Dict], pause: float) -> List[List[Dict]]:
    out: List[List[Dict]] = []
    for w in words:
        if out and out[-1][-1]["source_file"] == w["source_file"] and w["start"] - out[-1][-1]["end"] < pause:
            out[-1].append(w)
        else:
            out.append([w])
    return out


def _pairs(ws: List[Dict]) -> Set:
    t = [norm(w["text"]) for w in ws]
    return set(zip(t, t[1:])) or set(t)


def similar(atts: List[List[Dict]]) -> List[List[int]]:
    """For each attempt, the other attempts (same file) that share most of its wording."""
    P = [_pairs(a) for a in atts]
    out: List[List[int]] = [[] for _ in atts]
    for i in range(len(atts)):
        for j in range(i + 1, len(atts)):
            if atts[i][0]["source_file"] != atts[j][0]["source_file"]:
                continue
            shared, m = len(P[i] & P[j]), min(len(P[i]), len(P[j]))
            if m and shared >= min(2, m) and shared / m >= SAME_WORDING:
                out[i].append(j)
                out[j].append(i)
    return out


def alt_words(path: str, source_file: str) -> List[Dict]:
    """Words of the alternative ASR for one source file (matched by file stem), run 0."""
    res = json.load(open(path))["results"]
    stem = os.path.splitext(source_file)[0]
    key = next((k for k in res if os.path.splitext(os.path.basename(k))[0] == stem), None)
    if key is None:
        print(f"[verbatim] alt ASR has no transcript for {source_file}")
        return []
    return res[key][0]["words"]


def alt_diff(ours: List[Dict], alt: List[Dict]) -> str:
    """Other model's words for this attempt; words it heard that ours lacks are marked."""
    a, b = [norm(w["text"]) for w in ours], [norm(w["text"]) for w in alt]
    if a == b:
        return ""
    parts = []
    for tag, a1, a2, b1, b2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        for w in alt[b1:b2]:
            t = html.escape(w["text"])
            parts.append(t if tag == "equal" else f'<mark data-s="{w["start"]}" data-e="{w["end"]}">{t}</mark>')
    return " ".join(parts) or "<i>(nothing)</i>"


def build(job: str, alt_path: str, pause: float, out_dir: str) -> str:
    grid, files = artifacts.load_job(job, "grid"), artifacts.load_job(job, "files")
    if grid is None or files is None:
        raise SystemExit(f"job {job}: no stored word grid (artifacts/jobs/{job}/)")
    words = grid["words"]
    index = {w["id"]: i for i, w in enumerate(words)}
    kept: Set[str] = set()
    for s in (artifacts.load_job(job, "plan") or {}).get("segments", []):
        kept |= {words[i]["id"] for i in range(index[s["word_start"]], index[s["word_end"]] + 1)}
    video = {f["filename"]: os.path.relpath(f["path"], out_dir) for f in files}
    alts = {f: alt_words(alt_path, f) for f in video} if alt_path else {}

    atts = lines_of(words, pause)
    sims = similar(atts)
    data = []
    for n, a in enumerate(atts):
        f, t0, t1 = a[0]["source_file"], a[0]["start"], a[-1]["end"]
        alt = [x for x in alts.get(f, []) if t0 - 0.15 <= (x["start"] + x["end"]) / 2 <= t1 + 0.15]
        data.append({
            "file": f, "start": round(t0, 2), "end": round(t1, 2), "similar": sims[n],
            "alt": alt_diff(a, alt) if alts else "",
            "words": [{"id": w["id"], "t": w["text"], "s": w["start"], "e": w["end"],
                       "unsure": w.get("conf") is not None and w["conf"] < LOW_CONF,
                       "rep": i > 0 and norm(a[i - 1]["text"]) == norm(w["text"]),
                       "kept": w["id"] in kept} for i, w in enumerate(a)],
        })
    n_rep = sum(1 for s in sims if s)
    print(f"[verbatim] {len(words)} words, {len(atts)} attempts (pause >= {pause}s), {n_rep} share wording with "
          f"another attempt, {len(kept)} words in the current edit"
          + (f", other model differs on {sum(1 for d in data if d['alt'])} attempts" if alts else ""))
    return (PAGE.replace("__TITLE__", html.escape(job[:8]))
            .replace("__DATA__", json.dumps({"job": job, "videos": video, "attempts": data, "has_alt": bool(alts)})))


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Pick your takes · __TITLE__</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#141414;--card:#1e1e1e;--line:#2c2c2c;--text:#eee;--muted:#9a9a9a;--keep:#10b981;--cut:#ef4444;--accent:#8b5cf6}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;background:#191919;border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
video{width:300px;max-width:100%;border-radius:8px;background:#000}
.bar{flex:1;min-width:260px} .bar b{font-size:16px} .muted{color:var(--muted);font-size:13px}
.btn{background:#2a2a2a;color:var(--text);border:1px solid #3a3a3a;border-radius:6px;padding:5px 12px;cursor:pointer;font:inherit;font-size:13px}
.btn.primary{background:var(--accent);border-color:var(--accent);color:#fff}
main{max-width:900px;margin:0 auto;padding:16px}
.help{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin-bottom:16px}
.help h2{margin:0 0 6px;font-size:17px} .help ol{margin:6px 0 0;padding-left:20px} .help li{margin:4px 0}
.chip{display:inline-block;border-radius:4px;padding:0 6px;font-size:12px;font-weight:600}
.chip.keep{background:rgba(16,185,129,.15);color:var(--keep)} .chip.cut{background:rgba(239,68,68,.15);color:var(--cut)}
.chip.rep{background:rgba(139,92,246,.18);color:#c4b5fd}
.filters{display:flex;gap:8px;align-items:center;margin:0 0 12px;flex-wrap:wrap}
.card{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--line);border-radius:8px;padding:10px 14px;margin-bottom:8px}
.card.k{border-left-color:var(--keep)} .card.c{border-left-color:var(--cut);opacity:.75} .card.mixed{border-left-color:#f59e0b}
.card.changed{box-shadow:0 0 0 1px var(--accent)} .card.flash{outline:2px solid var(--accent)}
.top{display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-size:13px;color:var(--muted)}
.num{font-weight:700;color:var(--text);font-size:14px} .play{min-width:34px}
.seg{display:inline-flex;border:1px solid #3a3a3a;border-radius:6px;overflow:hidden;margin-left:auto}
.seg button{background:transparent;color:var(--muted);border:0;padding:3px 12px;cursor:pointer;font:inherit;font-size:13px}
.seg button.on.k{background:var(--keep);color:#04130d} .seg button.on.c{background:var(--cut);color:#fff}
.words{margin-top:6px;font-size:16px;line-height:1.9}
.w{cursor:pointer;border-radius:3px;padding:0 1px} .w:hover{background:#333}
.w.cutw{text-decoration:line-through;color:#666} .w.unsure{border-bottom:1px dotted #f59e0b} .w.rep{color:#c4b5fd}
.w.playing{background:#0e7490;color:#fff}
.sim a{color:#c4b5fd;cursor:pointer;text-decoration:underline dotted}
.alt{margin-top:4px;font-size:13px;color:var(--muted)} .alt mark{background:rgba(16,185,129,.2);color:#a7f3d0;border-radius:3px;cursor:pointer}
body:not(.show-alt) .alt{display:none}
footer{color:var(--muted);font-size:12px;text-align:center;padding:24px}
</style></head><body>
<header>
  <video id="v" controls preload="metadata"></video>
  <div class="bar">
    <b>Pick your takes</b> <span class="muted" id="sum"></span>
    <div style="margin-top:6px;display:flex;gap:8px;flex-wrap:wrap">
      <button class="btn primary" id="copy">Copy my choices</button>
      <button class="btn" id="reset">Reset to current edit</button>
      <span class="muted" id="saved"></span>
    </div>
  </div>
</header>
<main>
  <div class="help">
    <h2>What this page shows</h2>
    Your whole recording, <b>word for word as the speech model heard it</b>, stutters and repeats included,
    split into numbered <b>attempts</b> wherever you paused. Press <b>▶</b> to hear an attempt; click any word to hear just that word.
    <div style="margin-top:8px">
      <span class="chip keep">KEEP</span> = in your video now &nbsp;
      <span class="chip cut">CUT</span> = left out now &nbsp;
      <span class="chip rep">also said in #12</span> = you said the same words in another attempt (a retake) — click to jump there
    </div>
    <h2 style="margin-top:12px">What you need to do</h2>
    <ol>
      <li>Everything starts as VlogForge's current edit. <b>You only change what is wrong.</b></li>
      <li>For a line you said several times (purple chip), listen to the attempts, set the <b>best one to Keep</b> and the others to <b>Cut</b>.</li>
      <li>If a kept attempt has a <b>stutter or false start</b> ("I— I think"), <b>double-click</b> those words to cut just them (they get struck through). Double-click again to bring them back.</li>
      <li>When done, press <b>Copy my choices</b> and paste it to Claude. Your choices are saved in this browser if you close the page.</li>
    </ol>
    <div class="muted" style="margin-top:8px">Small marks: <span class="w unsure">dotted underline</span> = the speech model was unsure of this word ·
      <span class="w rep">purple word</span> = same word twice in a row (likely a stutter).</div>
  </div>
  <div class="filters">
    <span class="muted">Show:</span>
    <button class="btn" data-f="all">All attempts</button>
    <button class="btn" data-f="rep">Only lines said more than once</button>
    <button class="btn" data-f="changed">Only what I changed</button>
    <label class="muted" id="altbox" style="margin-left:auto;display:none"><input type="checkbox" id="alt"> Show what a second speech model heard (for checking the words)</label>
  </div>
  <div id="list"></div>
  <footer>Local page — nothing here is uploaded.</footer>
</main>
<script>
const D=__DATA__, V=document.getElementById('v'), KEY='verbatim-choices-'+D.job;
const A=D.attempts; let stopAt=null, filter='all';
const base={}; A.forEach(a=>a.words.forEach(w=>base[w.id]=w.kept));
let cut={}; try{cut=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
const isKept=id=>(id in cut)?!cut[id]:base[id];
function save(){try{localStorage.setItem(KEY,JSON.stringify(cut));document.getElementById('saved').textContent='saved';}catch(e){}}
function setKept(id,k){if(k===base[id])delete cut[id];else cut[id]=!k;}
let curFile=null;
function play(f,s,e){if(curFile!==f){V.src=D.videos[f];curFile=f;}
  const go=()=>{V.currentTime=Math.max(0,s);stopAt=e;V.play();};V.readyState>0?go():V.addEventListener('loadedmetadata',go,{once:true});}
V.ontimeupdate=()=>{if(stopAt!==null&&V.currentTime>=stopAt){V.pause();stopAt=null;}};
const fmt=s=>`${Math.floor(s/60)}:${(s%60).toFixed(1).padStart(4,'0')}`;
function state(a){const k=a.words.filter(w=>isKept(w.id)).length;return k===0?'c':k===a.words.length?'k':'mixed';}
function changed(a){return a.words.some(w=>isKept(w.id)!==base[w.id]);}
function render(){
  const L=document.getElementById('list');L.innerHTML='';
  A.forEach((a,n)=>{
    if(filter==='rep'&&!a.similar.length)return; if(filter==='changed'&&!changed(a))return;
    const st=state(a),c=document.createElement('div');c.className=`card ${st}${changed(a)?' changed':''}`;c.id='a'+n;
    const sim=a.similar.length?`<span class="chip rep sim">also said in ${a.similar.slice(0,6).map(j=>`<a data-j="${j}">#${j+1}</a>`).join(', ')}${a.similar.length>6?' …':''}</span>`:'';
    c.innerHTML=`<div class="top"><span class="num">#${n+1}</span><button class="btn play">▶</button><span>${fmt(a.start)} · ${(a.end-a.start).toFixed(1)}s</span>
      <span class="chip ${st==='c'?'cut':'keep'}">${st==='k'?'KEEP':st==='c'?'CUT':'PARTLY KEPT'}</span>${sim}
      <span class="seg"><button class="k ${st==='k'?'on':''}">Keep</button><button class="c ${st==='c'?'on':''}">Cut</button></span></div>
      <div class="words"></div>${a.alt?`<div class="alt">Second model heard: ${a.alt}</div>`:''}`;
    const W=c.querySelector('.words');
    a.words.forEach(w=>{const s=document.createElement('span');
      s.className='w'+(isKept(w.id)?'':' cutw')+(w.unsure?' unsure':'')+(w.rep?' rep':'');s.textContent=w.t;s.dataset.id=w.id;
      s.title=`${fmt(w.s)} — click: hear it · double-click: keep/cut this word`;
      let t=null;s.onclick=()=>{clearTimeout(t);t=setTimeout(()=>play(a.file,w.s-0.25,w.e+0.25),220);};
      s.ondblclick=()=>{clearTimeout(t);setKept(w.id,!isKept(w.id));save();render();};
      W.append(s,' ');});
    c.querySelector('.play').onclick=()=>play(a.file,a.start-0.2,a.end+0.3);
    c.querySelector('.seg .k').onclick=()=>{a.words.forEach(w=>setKept(w.id,true));save();render();};
    c.querySelector('.seg .c').onclick=()=>{a.words.forEach(w=>setKept(w.id,false));save();render();};
    c.querySelectorAll('.sim a').forEach(x=>x.onclick=()=>{filter='all';render();const el=document.getElementById('a'+x.dataset.j);el.scrollIntoView({behavior:'smooth',block:'center'});el.classList.add('flash');setTimeout(()=>el.classList.remove('flash'),1500);});
    c.querySelectorAll('.alt mark').forEach(m=>m.onclick=()=>play(a.file,+m.dataset.s-0.3,+m.dataset.e+0.3));
    L.append(c);});
  const kept=A.reduce((s,a)=>s+a.words.filter(w=>isKept(w.id)).length,0),tot=A.reduce((s,a)=>s+a.words.length,0);
  document.getElementById('sum').textContent=` · ${A.length} attempts · ${A.filter(a=>a.similar.length).length} are retakes of another · keeping ${kept} of ${tot} words · ${A.filter(changed).length} attempts changed by you`;
  document.querySelectorAll('.filters [data-f]').forEach(b=>b.classList.toggle('primary',b.dataset.f===filter));
}
document.querySelectorAll('.filters [data-f]').forEach(b=>b.onclick=()=>{filter=b.dataset.f;render();});
if(D.has_alt){document.getElementById('altbox').style.display='';document.getElementById('alt').onchange=e=>document.body.classList.toggle('show-alt',e.target.checked);}
document.getElementById('reset').onclick=()=>{if(confirm('Discard your changes and go back to the current edit?')){cut={};save();render();}};
document.getElementById('copy').onclick=()=>{
  const out={job:D.job,kept_attempts:[],cut_attempts:[],word_changes:[]};
  A.forEach((a,n)=>{const st=state(a);(st==='c'?out.cut_attempts:out.kept_attempts).push(n+1);
    a.words.forEach(w=>{if(isKept(w.id)!==base[w.id])out.word_changes.push({attempt:n+1,word:w.id,now:isKept(w.id)?'keep':'cut'});});});
  const t=JSON.stringify(out);navigator.clipboard.writeText(t).then(()=>alert(`Copied: ${out.word_changes.length} word changes. Paste it to Claude.`),()=>prompt('Copy this:',t));};
curFile=A[0].file;V.src=D.videos[curFile];render();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="job id with a stored word grid (artifacts/jobs/<job>)")
    ap.add_argument("--clip", help="eval clip id: write to eval-set/<clip>/verbatim.html")
    ap.add_argument("--alt", help="second verbatim ASR JSON (asr_bakeoff external format)")
    ap.add_argument("--pause", type=float, default=0.4)
    a = ap.parse_args()
    out_dir = os.path.join(EVAL_SET_DIR, a.clip) if a.clip else os.path.join(settings.artifact_dir, "jobs", a.job)
    out = os.path.join(out_dir, "verbatim.html")
    with open(out, "w") as fh:
        fh.write(build(a.job, a.alt, a.pause, out_dir))
    print(f"[verbatim] wrote {out}")


if __name__ == "__main__":
    main()
