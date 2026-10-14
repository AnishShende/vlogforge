"""Phase 10 E1: rebuild "raw footage" projects from the reviewed windows of finished vlogs (local only).

User design (2026-10-12): per reviewed window, split the finished cut back into what a creator would upload:
  A-roll  stretches -> video clips with their own sound
  B-roll  stretches -> the sound becomes a VOICE-OVER audio file (.m4a); the picture becomes one MUTED video
                       per original shot (silent audio track, like a camera that recorded nothing usable)
  silent  stretches -> video clips kept as they are (picture and sound)
The editor's cut is the answer key: order of every piece, its category, and which shot played under which
voice-over words.

Speech crosses picture cuts in finished edits (J/L cuts). Every word gets ONE owner, so no word is lost twice
or heard twice:
  - a word fully inside an A-roll stretch belongs to that A-roll clip;
  - a word whose middle lies in a B-roll stretch, or that hangs over the edge of an A-roll stretch next to
    B-roll, belongs to the voice-over of the nearest B-roll stretch;
  - other words (over silent stretches) stay in that clip's own sound.
The part of a word that belongs elsewhere is muted at the edge of an A-roll / silent clip.
Voice-over files run from just before their first word to just after their last, never into another owner's word.

File names are random-looking (content hash): neither the name nor the upload order gives the category or story
position away (upload order = original order in run 1; `shuffled_order` for the stress test).

Usage (from backend/):
    PYTHONPATH=. python -m eval.broll_synth build [--clip nz_travel] [--force]
    PYTHONPATH=. python -m eval.broll_synth check        # rebuild each window's sound from the pieces vs the original
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.utils.ffmpeg import get_ffmpeg_path
from eval.broll_gold import ANSWER_KEY_CLIPS, OUT_DIR, SRC_DIR, words_of

SYNTH_DIR = os.path.join(OUT_DIR, "synth")
VERSION = "1"
VO_PAD_SEC = 0.25           # voice-over file: lead-in / tail around its words (shortened if another word is nearer)
EDGE_GAP_SEC = 0.02         # keep this far from a neighbouring owner's word
CHECK_SR = 16000


# ---------------------------------------------------------------- split

def runs_of(shots: List[Dict]) -> List[Dict]:
    """Consecutive shots with the same label -> runs [{label, start, end, shots}]."""
    runs: List[Dict] = []
    for s in shots:
        if runs and runs[-1]["label"] == s["label"]:
            runs[-1]["end"] = s["end"]
            runs[-1]["shots"].append(s)
        else:
            runs.append({"label": s["label"], "start": s["start"], "end": s["end"], "shots": [s]})
    return runs


def _run_at(runs: List[Dict], t: float) -> int:
    for k, r in enumerate(runs):
        if r["start"] <= t < r["end"]:
            return k
    return len(runs) - 1 if t >= runs[-1]["end"] else 0


def assign_words(words: List[Dict], runs: List[Dict]) -> List[Tuple[int, str]]:
    """Owner per word: (run index, "aroll" | "vo" | "silent"). Words outside the runs are skipped (-1)."""
    lo, hi = runs[0]["start"], runs[-1]["end"]
    owners = []
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        if not lo <= mid < hi:
            owners.append((-1, ""))
            continue
        k = _run_at(runs, mid)
        lab = runs[k]["label"]
        if lab == "a_roll":
            inside = w["start"] >= runs[k]["start"] and w["end"] <= runs[k]["end"]
            if inside:
                owners.append((k, "aroll"))
                continue
            # hangs over an edge: give it to the B-roll next to that edge, if there is one
            side = k - 1 if w["start"] < runs[k]["start"] else k + 1
            if 0 <= side < len(runs) and runs[side]["label"] == "b_roll":
                owners.append((side, "vo"))
            else:
                owners.append((k, "aroll"))         # partly muted at the edge (reported)
        elif lab == "b_roll":
            owners.append((k, "vo"))
        else:
            owners.append((k, "silent"))
    return owners


def build_plan(words: List[Dict], shots: List[Dict]) -> Dict:
    """Pieces of one window in original order + per-word owners (pure; no files)."""
    runs = runs_of(shots)
    owners = assign_words(words, runs)
    pieces: List[Dict] = []
    for k, r in enumerate(runs):
        mine = [i for i, (ok, kind) in enumerate(owners) if ok == k]
        if r["label"] in ("a_roll", "silent"):
            kind = "aroll" if r["label"] == "a_roll" else "silent"
            mutes = []                                   # parts of OTHER owners' words inside this picture
            for i, w in enumerate(words):
                if owners[i][0] in (-1, k) or w["end"] <= r["start"] or w["start"] >= r["end"]:
                    continue
                mutes.append([max(w["start"], r["start"]), min(w["end"], r["end"])])
            pieces.append({"kind": kind, "run": k, "start": r["start"], "end": r["end"], "words": mine,
                           "mute": mutes, "shots": [s["i"] for s in r["shots"]]})
        else:
            for s in r["shots"]:
                pieces.append({"kind": "broll", "run": k, "start": s["start"], "end": s["end"], "words": [],
                               "mute": [], "shots": [s["i"]]})
            if mine:
                first, last = words[mine[0]], words[mine[-1]]
                prev_end = max((w["end"] for i, w in enumerate(words) if w["end"] <= first["start"] and i not in mine),
                               default=-1e9)
                next_start = min((w["start"] for i, w in enumerate(words) if w["start"] >= last["end"] and i not in mine),
                                 default=1e9)
                a = max(first["start"] - VO_PAD_SEC, prev_end + EDGE_GAP_SEC, 0.0)
                b = min(last["end"] + VO_PAD_SEC, next_start - EDGE_GAP_SEC)
                pieces.append({"kind": "vo", "run": k, "start": a, "end": b, "words": mine, "mute": [],
                               "shots": [s["i"] for s in r["shots"]]})
    return {"runs": [{k: r[k] for k in ("label", "start", "end")} for r in runs], "pieces": pieces,
            "owners": owners}


# ---------------------------------------------------------------- files

def _name(clip: str, p: Dict) -> str:
    h = hashlib.sha256(f"{VERSION}|{clip}|{p['kind']}|{p['start']:.3f}|{p['end']:.3f}".encode()).hexdigest()[:10]
    return f"{h}.m4a" if p["kind"] == "vo" else f"{h}.mp4"


def _cut(src: str, p: Dict, out: str) -> None:
    ff = get_ffmpeg_path()
    dur = p["end"] - p["start"]
    base = [ff, "-y", "-v", "error", "-ss", f"{p['start']:.6f}", "-i", src]
    if p["kind"] == "vo":
        cmd = base + ["-t", f"{dur:.6f}", "-vn", "-c:a", "aac", "-b:a", "160k", "-map_metadata", "-1", out]
    elif p["kind"] == "broll":
        cmd = base + ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", f"{dur:.6f}", "-map", "0:v",
                      "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                      "-c:a", "aac", "-b:a", "128k", "-map_metadata", "-1", "-shortest", out]
    else:
        af = ",".join(f"volume=enable='between(t,{a - p['start']:.4f},{b - p['start']:.4f})':volume=0"
                      for a, b in p["mute"]) or "anull"
        cmd = base + ["-t", f"{dur:.6f}", "-af", af, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                      "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-map_metadata", "-1", out]
    subprocess.run(cmd, check=True)


def build_window(clip: str, wi: int, window: List[int], gold: Dict, words: List[Dict], force: bool) -> Dict:
    a, b = window
    shots = [s for s in gold["shots"] if a <= s["i"] <= b]
    lo, hi = shots[0]["start"], shots[-1]["end"]
    wwords = [w for w in words if lo <= (w["start"] + w["end"]) / 2 < hi]
    plan = build_plan(wwords, shots)
    proj = f"{clip}_w{wi}"
    files_dir = os.path.join(SYNTH_DIR, proj, "files")
    os.makedirs(files_dir, exist_ok=True)
    src = os.path.join(SRC_DIR, f"{clip}.mp4")
    for p in plan["pieces"]:
        p["file"] = _name(clip, p)
        out = os.path.join(files_dir, p["file"])
        if force or not os.path.exists(out):
            _cut(src, p, out)
    # answer key: the editor's cut, in original order, on the window clock (0 = window start)
    pieces = plan["pieces"]
    vo_words, under = [], []
    for p in pieces:
        if p["kind"] != "vo":
            continue
        for i in p["words"]:
            w = wwords[i]
            mid = (w["start"] + w["end"]) / 2
            shot = next((s["i"] for s in shots if s["start"] <= mid < s["end"]), None)
            shot_file = next((q["file"] for q in pieces if q["kind"] == "broll" and q["shots"] == [shot]), None)
            vo_words.append({"vo_file": p["file"], "start": round(w["start"] - p["start"], 3),
                             "end": round(w["end"] - p["start"], 3), "text": w["text"].strip(),
                             "shot_under": shot, "shot_file": shot_file})
    timeline = [{"file": p["file"], "kind": p["kind"], "at": round(p["start"] - lo, 3),
                 "src_start": round(p["start"], 3), "src_end": round(p["end"], 3), "shots": p["shots"],
                 "n_words": len(p["words"]), "muted": [[round(x - p["start"], 3) for x in m] for m in p["mute"]]}
                for p in pieces]
    upload = [p["file"] for p in pieces]
    shuffled = upload[:]
    random.Random(f"{proj}|{VERSION}").shuffle(shuffled)
    kinds = {k: sum(1 for p in pieces if p["kind"] == k) for k in ("aroll", "vo", "broll", "silent")}
    partly = sum(1 for p in pieces if p["kind"] in ("aroll", "silent") for _ in p["mute"])
    gold_out = {"project": proj, "clip": clip, "window": window, "version": VERSION,
                "source": "eval/broll_synth.py from eval-set/broll/<clip>/gold.json (user-reviewed labels)",
                "duration": round(hi - lo, 3), "upload_order": upload, "shuffled_order": shuffled,
                "category": {p["file"]: {"aroll": "a_roll", "vo": "voice_over", "broll": "b_roll",
                                         "silent": "silent"}[p["kind"]] for p in pieces},
                "timeline": timeline, "vo_words": vo_words,
                "words_total": len(wwords), "words_owned": {k: sum(1 for _, o in plan["owners"] if o == k)
                                                            for k in ("aroll", "vo", "silent")},
                "edge_mutes": partly}
    with open(os.path.join(SYNTH_DIR, proj, "gold.json"), "w") as f:
        json.dump(gold_out, f, ensure_ascii=False, indent=1)
    print(f"{proj}: {hi - lo:.0f}s, files {kinds}, words {gold_out['words_owned']} of {len(wwords)}, "
          f"edge mutes {partly}", flush=True)
    return gold_out


# ---------------------------------------------------------------- check

def _decode(path: str, start: Optional[float] = None, dur: Optional[float] = None) -> np.ndarray:
    cmd = [get_ffmpeg_path(), "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.6f}"]
    cmd += ["-i", path]
    if dur is not None:
        cmd += ["-t", f"{dur:.6f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(CHECK_SR), "-f", "s16le", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def check(proj: str) -> Dict:
    """Lay every piece's sound back at its original place; per word of the window: is it heard (energy of the
    rebuild vs the original over the word) and is it heard ONCE (no two pieces sounding over it)."""
    g = json.load(open(os.path.join(SYNTH_DIR, proj, "gold.json")))
    src = os.path.join(SRC_DIR, f"{g['clip']}.mp4")
    lo = g["timeline"][0]["src_start"]
    orig = _decode(src, lo, g["duration"])
    rebuilt = np.zeros_like(orig)
    layers = np.zeros_like(orig)
    for t in g["timeline"]:
        if t["kind"] == "broll":
            continue
        x = _decode(os.path.join(SYNTH_DIR, proj, "files", t["file"]))
        a = int(round(t["at"] * CHECK_SR))
        if a < 0:                                    # a voice-over lead-in can start before the window
            x, a = x[-a:], 0
        n = min(len(x), len(rebuilt) - a)
        if n <= 0:
            continue
        rebuilt[a:a + n] += x[:n]
        layers[a:a + n] += (np.abs(x[:n]) > 1e-4)
    words = [w for w in words_of(g["clip"]) if lo <= (w["start"] + w["end"]) / 2 < lo + g["duration"]]
    lost, doubled, ratios = [], 0, []
    for w in words:
        a, b = max(int((w["start"] - lo) * CHECK_SR), 0), min(int((w["end"] - lo) * CHECK_SR), len(orig))
        if b <= a:
            continue
        eo = float(np.sqrt(np.mean(orig[a:b] ** 2)) + 1e-9)
        er = float(np.sqrt(np.mean(rebuilt[a:b] ** 2)) + 1e-9)
        ratios.append(er / eo)
        if er / eo < 0.5:
            lost.append(w["text"].strip())
        if np.mean(layers[a:b] > 1) > 0.5:
            doubled += 1
    r = np.array(ratios)
    out = {"project": proj, "words": len(words), "heard_ok": int(np.sum((r >= 0.5) & (r <= 1.5))),
           "lost": lost, "doubled": doubled, "ratio_median": round(float(np.median(r)), 3)}
    print(f"{proj}: {out['heard_ok']}/{out['words']} words heard at the original level, "
          f"{len(lost)} lost {lost[:8]}, {doubled} heard twice, median level ratio {out['ratio_median']}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--clip", action="append", choices=ANSWER_KEY_CLIPS)
    b.add_argument("--force", action="store_true")
    sub.add_parser("check")
    a = ap.parse_args()
    if a.cmd == "build":
        for clip in a.clip or ANSWER_KEY_CLIPS:
            gold = json.load(open(os.path.join(OUT_DIR, clip, "gold.json")))
            words = words_of(clip)
            for wi, window in enumerate(gold["windows"], 1):
                build_window(clip, wi, window, gold, words, a.force)
    else:
        for proj in sorted(os.listdir(SYNTH_DIR)):
            if os.path.exists(os.path.join(SYNTH_DIR, proj, "gold.json")):
                check(proj)


if __name__ == "__main__":
    sys.exit(main())
