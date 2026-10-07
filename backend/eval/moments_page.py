"""Phase 8 review page: the moment tables of one or more jobs, one card per moment, to be judged by ear.
Local only (eval-set/review_moments.html); nothing is uploaded.

Usage (from backend/):
    PYTHONPATH=. python -m eval.moments_page --job IMG_1614=<job> --job IMG_0248=<job> ...
"""

import argparse
import json
import os

from app.utils import artifacts
from eval.gold import EVAL_SET_DIR

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Check story moments</title><style>
:root{--bg:#141414;--card:#1e1e1e;--line:#2e2e2e;--text:#eee;--muted:#a3a3a3;--acc:#8b5cf6;--ok:#10b981}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.55 system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:#191919;border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
video{width:280px;max-width:100%;border-radius:8px;background:#000}
main{max-width:860px;margin:0 auto;padding:16px}
.intro,.clip{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin-bottom:14px}
h1{font-size:20px;margin:0 0 6px}h2{font-size:18px;margin:28px 0 8px}
.card{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--line);border-radius:10px;padding:12px 16px;margin:10px 0}
.card.done{border-left-color:var(--ok)}
.top{display:flex;gap:10px;align-items:center;flex-wrap:wrap;color:var(--muted);font-size:14px}
.num{font-weight:700;color:var(--text)}.fn{background:rgba(139,92,246,.18);color:#c4b5fd;border-radius:4px;padding:0 8px;font-weight:600}
.imp{display:inline-block;width:80px;height:8px;background:#333;border-radius:4px;vertical-align:middle;overflow:hidden}.imp i{display:block;height:100%;background:#f59e0b}
.sum{margin:6px 0 2px;font-weight:600}.text{color:#cfcfcf;font-size:15px}.dep{color:var(--muted);font-size:14px;margin-top:4px}
.row{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0;align-items:center}.lbl{color:var(--muted);font-size:13px;width:110px}
button{font:inherit;font-size:14px;border-radius:8px;border:1px solid #444;background:#2a2a2a;color:var(--text);padding:4px 12px;cursor:pointer}
button.play{background:#252545;border-color:#3d3d7a}button.on{background:var(--acc);border-color:var(--acc);color:#fff}
textarea{width:100%;font:inherit;font-size:14px;background:#181818;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:6px 10px;margin-top:4px}
#copy{background:var(--acc);border-color:var(--acc);color:#fff}
</style></head><body>
<header><video id="v" controls preload="metadata"></video>
<div><b>Check story moments</b> <span id="prog" style="color:var(--muted)"></span><div class="row"><button id="copy">Copy my answers</button></div></div></header>
<main>
<div class="intro"><h1>What is this?</h1>
The editor split each video into <b>moments</b> (story beats) and labelled each one. Later this decides what to keep when a video must be shorter.
For each moment, press <b>▶</b>, then judge three things (skip what looks fine):
<ul><li><b>Role</b>: is the purple label right? (hook = grabs attention, explanation, payoff = the reward/answer, conclusion…)</li>
<li><b>Importance</b>: the orange bar: how much the video loses without this moment.</li>
<li><b>Needs</b>: the earlier moments it needs to make sense (e.g. an answer needs its question).</li></ul>
Then press <b>Copy my answers</b> and paste them to Claude.</div>
<div id="list"></div></main>
<script>
const D=__DATA__,V=document.getElementById('v'),KEY='moments-review-1';
let st={};try{st=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}};
let queue=[],cur=null,stopAt=null;
function playSegs(src,segs){if(cur!==src){V.src=src;cur=src;}queue=segs.slice();next();}
function next(){if(!queue.length){V.pause();stopAt=null;return;}const [s,e]=queue.shift();const go=()=>{V.currentTime=Math.max(0,s);stopAt=e;V.play();};V.readyState>0?go():V.addEventListener('loadedmetadata',go,{once:true});}
V.ontimeupdate=()=>{if(stopAt!==null&&V.currentTime>=stopAt){stopAt=null;next();}};
const esc=t=>(t||'').replace(/&/g,'&amp;').replace(/</g,'&lt;');
function set(k,f,v){st[k]=st[k]||{};st[k][f]=st[k][f]===v?undefined:v;save();render();}
function note(k,v){st[k]=st[k]||{};st[k].note=v;save();}
function opts(k,f,list){return list.map(([v,l])=>`<button class="${(st[k]||{})[f]===v?'on':''}" onclick="set('${k}','${f}','${v}')">${l}</button>`).join('');}
const fmt=s=>`${Math.floor(s/60)}:${(s%60).toFixed(1).padStart(4,'0')}`;
function render(){let h='';
  D.forEach((c,ci)=>{const ck=`${ci}`;
    h+=`<h2>${esc(c.clip)}</h2><div class="clip"><b>Synopsis:</b> ${esc(c.synopsis)}<br><b>Creator's goal:</b> ${esc(c.goal)}
      <div class="row"><span class="lbl">Is this right?</span>${opts(ck+'s','v',[['ok','Yes'],['wrong','No']])}</div>
      <textarea rows="1" placeholder="Remark (optional)…" oninput="note('${ck}s',this.value)">${esc((st[ck+'s']||{}).note)}</textarea></div>`;
    c.moments.forEach((m,mi)=>{const k=`${ci}-${mi}`,a=st[k]||{},done=a.role||a.imp||a.dep||a.note?' done':'';
      const deps=m.depends_on.length?m.depends_on.map(d=>{const x=c.moments.find(y=>y.id===d);return `${d.slice(1)}. ${esc(x?x.summary:'')}`}).join(' · '):'nothing (stands alone)';
      h+=`<div class="card${done}"><div class="top"><span class="num">${mi+1}.</span><button class="play" onclick="playSegs(D[${ci}].videos[D[${ci}].moments[${mi}].file],D[${ci}].moments[${mi}].runs)">▶ Play</button>
        <span>${fmt(m.start)}</span><span class="fn">${esc(m.function)}</span><span class="imp" title="importance ${m.importance}"><i style="width:${m.importance*100}%"></i></span><span>${m.importance}</span></div>
        <div class="sum">${esc(m.summary)}</div><div class="text">${esc(m.text)}</div><div class="dep"><b>Needs:</b> ${deps}</div>
        <div class="row"><span class="lbl">Role</span>${opts(k,'role',[['ok','Right'],['wrong','Wrong']])}</div>
        <div class="row"><span class="lbl">Importance</span>${opts(k,'imp',[['low','Too low'],['ok','Right'],['high','Too high']])}</div>
        <div class="row"><span class="lbl">Needs</span>${opts(k,'dep',[['ok','Right'],['wrong','Wrong'],['missing','Missing one']])}</div>
        <textarea rows="1" placeholder="Remark (optional), e.g. 'should be payoff' or 'needs moment 3'…" oninput="note('${k}',this.value)">${esc(a.note)}</textarea></div>`;});});
  document.getElementById('list').innerHTML=h;
  const n=D.reduce((s,c)=>s+c.moments.length,0),d=Object.keys(st).filter(k=>k.includes('-')&&Object.values(st[k]).some(Boolean)).length;
  document.getElementById('prog').textContent=`· ${d} of ${n} moments judged`;}
document.getElementById('copy').onclick=()=>{const out=[];
  D.forEach((c,ci)=>{const s=st[ci+'s']||{};out.push(`${c.clip}: synopsis ${s.v||'-'}${s.note?' ('+s.note+')':''}`);
    c.moments.forEach((m,mi)=>{const a=st[`${ci}-${mi}`]||{};if(a.role||a.imp||a.dep||(a.note&&a.note.trim()))
      out.push(`  ${mi+1}. role ${a.role||'-'}, importance ${a.imp||'-'}, needs ${a.dep||'-'}${a.note&&a.note.trim()?' ('+a.note.trim()+')':''}`);});});
  const t=out.join('\n');navigator.clipboard.writeText(t).then(()=>alert('Copied'),()=>prompt('Copy this:',t));};
render();
</script></body></html>"""


def clip_data(name: str, job: str, out_dir: str) -> dict:
    r = artifacts.load_job(job, "moments")
    if r is None:
        raise SystemExit(f"job {job}: no moments; run app.tasks.moments.moments_for_job first")
    words = artifacts.load_job(job, "grid")["words"]
    grid = {w["id"]: w for w in words}
    pos = {w["id"]: i for i, w in enumerate(words)}
    files = {f["filename"]: os.path.relpath(f["path"], out_dir) for f in artifacts.load_job(job, "files")}
    moments = []
    for m in r["moments"]:
        ws = [grid[i] for i in m["word_ids"]]
        runs, prev = [], None                 # play the moment as edited: grid-consecutive kept words only
        for w in ws:
            if prev is not None and pos[w["id"]] == pos[prev["id"]] + 1:
                runs[-1][1] = w["end"] + 0.15
            else:
                runs.append([max(0.0, w["start"] - 0.15), w["end"] + 0.15])
            prev = w
        moments.append({**{k: m[k] for k in ("id", "start", "function", "importance", "depends_on", "summary", "text")},
                        "file": m["source_file"], "runs": runs})
    return {"clip": name, "synopsis": r["synopsis"], "goal": r["creator_goal"], "videos": files, "moments": moments}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", required=True, metavar="NAME=JOB")
    a = ap.parse_args()
    out = os.path.join(EVAL_SET_DIR, "review_moments.html")
    data = [clip_data(*x.split("=", 1), EVAL_SET_DIR) for x in a.job]
    open(out, "w").write(PAGE.replace("__DATA__", json.dumps(data)))
    print(f"[moments] wrote {out}: {sum(len(c['moments']) for c in data)} moments in {len(data)} clip(s)")


if __name__ == "__main__":
    main()
