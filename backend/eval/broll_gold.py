"""Phase 10 S0: B-roll answer keys from FINISHED vlogs (local only, eval-set/broll/).

In a finished vlog the editor already decided, for every spoken line, whether the viewer sees the
speaker (A-roll) or other footage (B-roll), and which shot. This script recovers those decisions:

  prepare  per vlog: ASR words (cached), shot index (app.tasks.shots, cached), a pre-label per shot,
           two review windows of ~WINDOW_SEC with dense speech and both kinds of shot, and one local
           review page (eval-set/broll/review_shots.html) to correct the pre-labels by eye.
  apply    the answers copied from the page -> eval-set/broll/<clip>/gold.json

Labels per shot:  a_roll   speaker on camera, talking to it
                  b_roll   other footage while someone speaks (voice-over / speech over cutaways)
                  silent   no speech over the shot (music, ambience, montage) -> not scored
Pre-label: the caption model's `to_camera` / `speaking` + the share of the shot covered by ASR words. ASR can turn
song lyrics into words, which is one reason every window is reviewed.

Usage (from backend/):
    PYTHONPATH=. python -m eval.broll_gold prepare
    PYTHONPATH=. python -m eval.broll_gold apply answers.json
"""

import argparse
import html
import json
import logging
import os
import sys
from typing import Dict, List, Tuple

from app.tasks import shots as shot_index
from app.tasks.transcribe import transcribe_audio
from app.utils.ffmpeg import extract_audio
from eval.gold import EVAL_SET_DIR

SRC_DIR = os.path.join(EVAL_SET_DIR, "_broll_src")
OUT_DIR = os.path.join(EVAL_SET_DIR, "broll")
CLIPS = ["nz_travel", "cycling_europe", "gym_chest"]
# gym_chest is filmed in long unbroken takes (12 shots in 38 min, checked by eye 2026-10-11): too few
# editor decisions for an answer key. It stays in the shot library (wrong-vlog negatives).
ANSWER_KEY_CLIPS = ["nz_travel", "cycling_europe"]
WINDOW_SEC = 240.0
WINDOWS_PER_CLIP = 2
SPEECH_SHARE = 0.3          # a shot counts as "over speech" when words cover >= 30% of it
LABELS = ["a_roll", "b_roll", "silent"]


def words_of(clip: str) -> List[Dict]:
    wav = os.path.join(SRC_DIR, f"{clip}.wav")
    if not os.path.exists(wav) and not extract_audio(os.path.join(SRC_DIR, f"{clip}.mp4"), wav):
        raise RuntimeError(f"audio extraction failed for {clip}")
    return [w for w in transcribe_audio(wav) if w.get("start") is not None and w.get("end") is not None]


def speech_in(words: List[Dict], s: float, e: float) -> Tuple[float, str]:
    """Share of [s, e] covered by words, and the words' text."""
    inside = [w for w in words if w["end"] > s and w["start"] < e]
    covered = sum(min(w["end"], e) - max(w["start"], s) for w in inside)
    return covered / max(e - s, 1e-6), " ".join(w["text"].strip() for w in inside)


def prelabel(shot: Dict, share: float) -> str:
    if share < SPEECH_SHARE:
        return "silent"
    return "a_roll" if shot["to_camera"] or shot.get("speaking") else "b_roll"


def pick_windows(rows: List[Dict], n: int = WINDOWS_PER_CLIP, length: float = WINDOW_SEC) -> List[Tuple[int, int]]:
    """n non-overlapping runs of whole shots, each ~length seconds, with the most seconds of BOTH
    A-roll and B-roll over speech (score = the smaller of the two, ties broken by speech)."""
    cands = []
    for a in range(len(rows)):
        b, secs = a, {"a_roll": 0.0, "b_roll": 0.0, "silent": 0.0}
        while b < len(rows) and (b == a or rows[b]["end"] - rows[a]["start"] < length):
            secs[rows[b]["pre"]] += rows[b]["end"] - rows[b]["start"]
            b += 1
        if b == len(rows) and rows[-1]["end"] - rows[a]["start"] < length * 0.75:
            break
        cands.append((min(secs["a_roll"], secs["b_roll"]), secs["a_roll"] + secs["b_roll"], a, b))
    chosen: List[Tuple[int, int]] = []
    for _, _, a, b in sorted(cands, reverse=True):
        if all(b <= ca or a >= cb for ca, cb in chosen):
            chosen.append((a, b))
        if len(chosen) == n:
            break
    return sorted(chosen)


def prepare_clip(clip: str) -> Dict:
    video = os.path.join(SRC_DIR, f"{clip}.mp4")
    words = words_of(clip)
    index = shot_index.index_file(video)
    rows = []
    for s in index["shots"]:
        share, text = speech_in(words, s["start"], s["end"])
        rows.append({"i": s["i"], "start": s["start"], "end": s["end"], "caption": s["caption"], "kind": s["kind"],
                     "to_camera": s["to_camera"], "speaking": s.get("speaking", False), "speech_share": round(share, 2), "text": text,
                     "pre": prelabel(s, share),
                     "frame": os.path.join(shot_index.index_dir(index["digest"]), s["frames"][1])})
    windows = pick_windows(rows)
    out = {"clip": clip, "video": video, "digest": index["digest"], "n_shots": len(rows), "n_words": len(words),
           "windows": [[rows[a]["i"], rows[b - 1]["i"]] for a, b in windows], "shots": rows}
    os.makedirs(os.path.join(OUT_DIR, clip), exist_ok=True)
    with open(os.path.join(OUT_DIR, clip, "prelabels.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    pre = {k: sum(1 for r in rows if r["pre"] == k) for k in LABELS}
    print(f"{clip}: {len(rows)} shots, {len(words)} words, pre-labels {pre}, windows "
          + ", ".join(f"{rows[a]['start']:.0f}-{rows[b - 1]['end']:.0f}s ({b - a} shots)" for a, b in windows), flush=True)
    return out


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Check B-roll labels</title><style>
:root{--bg:#141414;--card:#1e1e1e;--line:#2e2e2e;--text:#eee;--muted:#a3a3a3;--acc:#8b5cf6;--a:#3b82f6;--b:#f59e0b;--s:#6b7280}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:#191919;border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
video{width:320px;max-width:100%;border-radius:8px;background:#000}
main{max-width:1200px;margin:0 auto;padding:16px}
.intro{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin-bottom:14px}
h1{font-size:20px;margin:0 0 6px}h2{font-size:18px;margin:26px 0 8px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-top:4px solid var(--s);border-radius:10px;padding:8px;font-size:13px}
.card.a_roll{border-top-color:var(--a)}.card.b_roll{border-top-color:var(--b)}.card.changed{outline:2px solid var(--acc)}
.card img{width:100%;border-radius:6px;cursor:pointer;display:block}
.t{color:var(--muted)}.cap{margin:4px 0}.words{color:#cfcfcf;font-style:italic;max-height:4.5em;overflow:auto}
.row{display:flex;gap:4px;margin-top:6px;flex-wrap:wrap}
button{font:inherit;font-size:13px;border-radius:7px;border:1px solid #444;background:#2a2a2a;color:var(--text);padding:3px 9px;cursor:pointer}
button.on.a_roll{background:var(--a);border-color:var(--a)}button.on.b_roll{background:var(--b);border-color:var(--b);color:#111}button.on.silent{background:var(--s);border-color:var(--s)}
#copy{background:var(--acc);border-color:var(--acc);color:#fff}
.key span{display:inline-block;padding:0 8px;border-radius:4px;margin-right:6px}
</style></head><body>
<header><video id="v" controls preload="metadata"></video>
<div><b>Check B-roll labels</b> <span id="prog" class="t"></span><div class="row"><button id="copy">Copy my answers</button></div></div></header>
<main>
<div class="intro"><h1>What is this?</h1>
Each card is one <b>shot</b> of a finished vlog (a stretch between two cuts). The editor of that vlog decided what you
see while someone talks: the speaker (A-roll) or other footage (B-roll). Those decisions become the answer key that
VlogForge's B-roll is scored against. The labels below are a machine guess; fix the wrong ones.
<p class="key"><span style="background:var(--a)">A-roll</span>we see the person who is speaking (facing the camera or not)
<span style="background:var(--b);color:#111">B-roll</span>someone is speaking but we see other footage (voice-over)
<span style="background:var(--s)">Silent</span>nobody speaks over this shot (music, ambience; sung lyrics count as silent)</p>
Click a picture to play the shot. Only the windows below need checking; untouched cards keep the guess.
When done, press <b>Copy my answers</b> and paste them to Claude.</div>
<div id="app"></div></main>
<script>
const D=__DATA__;const st={};
const v=document.getElementById('v');let stopAt=null;
v.addEventListener('timeupdate',()=>{if(stopAt!==null&&v.currentTime>=stopAt){v.pause();stopAt=null;}});
function play(ci,si){const c=D[ci],s=c.shots[si];if(!v.src.endsWith(c.video_url))v.src=c.video_url;
  const go=()=>{v.currentTime=s.start;stopAt=s.end;v.play();};if(v.readyState>=1)go();else v.addEventListener('loadedmetadata',go,{once:true});}
function label(ci,si){const k=D[ci].clip+':'+D[ci].shots[si].i;return st[k]||D[ci].shots[si].pre;}
function set(ci,si,l){const k=D[ci].clip+':'+D[ci].shots[si].i;if(l===D[ci].shots[si].pre)delete st[k];else st[k]=l;render();}
const NAMES={a_roll:'A-roll',b_roll:'B-roll',silent:'Silent'};
function render(){let h='',n=0;D.forEach((c,ci)=>{c.windows.forEach(([a,b],wi)=>{
  h+=`<h2>${c.clip} — window ${wi+1} (${c.shots_by_i[a].start.toFixed(0)}–${c.shots_by_i[b].end.toFixed(0)} s)</h2><div class="grid">`;
  c.shots.forEach((s,si)=>{if(s.i<a||s.i>b)return;const l=label(ci,si),ch=st[c.clip+':'+s.i]?' changed':'';
    h+=`<div class="card ${l}${ch}"><img loading="lazy" src="${s.frame_url}" onclick="play(${ci},${si})">
    <div class="t">#${s.i} · ${s.start.toFixed(1)}–${s.end.toFixed(1)} s · speech ${Math.round(s.speech_share*100)}%</div>
    <div class="cap">${s.caption}</div><div class="words">${s.text||''}</div>
    <div class="row">${['a_roll','b_roll','silent'].map(x=>`<button class="${x}${l===x?' on':''}" onclick="set(${ci},${si},'${x}')">${NAMES[x]}</button>`).join('')}</div></div>`;});
  h+='</div>';});});
  document.getElementById('app').innerHTML=h;document.getElementById('prog').textContent=Object.keys(st).length+' changed';}
document.getElementById('copy').onclick=()=>{navigator.clipboard.writeText(JSON.stringify({broll_labels:st}));
  document.getElementById('copy').textContent='Copied ✓';setTimeout(()=>document.getElementById('copy').textContent='Copy my answers',1500);};
render();
</script></body></html>"""


def write_page(clips: List[Dict]) -> str:
    page_path = os.path.join(OUT_DIR, "review_shots.html")
    rel = lambda p: os.path.relpath(p, OUT_DIR)
    data = []
    for c in clips:
        shots = [{**{k: s[k] for k in ("i", "start", "end", "speech_share", "pre")},
                  "caption": html.escape(s["caption"]), "text": html.escape(s["text"]), "frame_url": rel(s["frame"])}
                 for s in c["shots"]]
        data.append({"clip": c["clip"], "video_url": rel(c["video"]), "windows": c["windows"], "shots": shots,
                     "shots_by_i": {s["i"]: {"start": s["start"], "end": s["end"]} for s in shots}})
    with open(page_path, "w") as f:
        f.write(PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False)))
    return page_path


def apply(answers_path: str) -> None:
    labels = json.load(open(answers_path))["broll_labels"]
    for clip in ANSWER_KEY_CLIPS:
        p = os.path.join(OUT_DIR, clip, "prelabels.json")
        if not os.path.exists(p):
            continue
        pre = json.load(open(p))
        changed = 0
        windows = pre["windows"]
        gold_shots = []
        for s in pre["shots"]:
            if not any(a <= s["i"] <= b for a, b in windows):
                continue
            lab = labels.get(f"{clip}:{s['i']}", s["pre"])
            changed += lab != s["pre"]
            gold_shots.append({"i": s["i"], "start": s["start"], "end": s["end"], "label": lab, "pre": s["pre"]})
        gold = {"clip": clip, "digest": pre["digest"], "windows": windows, "shots": gold_shots,
                "source": "user review of eval/broll_gold.py pre-labels"}
        with open(os.path.join(OUT_DIR, clip, "gold.json"), "w") as f:
            json.dump(gold, f, indent=1)
        n = len(gold_shots)
        print(f"{clip}: {n} shots in windows, {changed} corrected ({changed / max(n, 1):.0%} pre-label error)")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare")
    pp.add_argument("--clip", action="append", choices=CLIPS)
    sub.add_parser("apply").add_argument("answers")
    a = ap.parse_args()
    if a.cmd == "prepare":
        clips = [prepare_clip(c) for c in (a.clip or CLIPS)]
        print("review page:", write_page([c for c in clips if c["clip"] in ANSWER_KEY_CLIPS]))
    else:
        apply(a.answers)


if __name__ == "__main__":
    sys.exit(main())
