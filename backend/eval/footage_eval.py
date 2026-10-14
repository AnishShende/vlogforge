"""Phase 10 V1: footage categories on the synthetic raw projects (eval/broll_synth.py), scored per file
against gold.json. Gold `silent` pieces are the finished vlog's no-speech stretches, which the editor KEPT, so
for them both b_roll and silent count as right (reported separately); `usable` is judged on own footage.

Usage (from backend/):
    PYTHONPATH=. python -m eval.footage_eval [--project nz_travel_w1] [--job JOB_ID]
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from app.tasks import footage
from app.tasks import shots as shot_index
from app.tasks.transcribe import transcribe_audio
from app.utils.ffmpeg import extract_audio, get_video_duration
from eval.broll_synth import SYNTH_DIR

ACCEPT = {"a_roll": {"a_roll"}, "voice_over": {"voice_over"}, "b_roll": {"b_roll"}, "silent": {"b_roll", "silent"}}


def words_for(path: str, work: str) -> List[Dict]:
    wav = os.path.join(work, os.path.splitext(os.path.basename(path))[0] + ".wav")
    if not os.path.exists(wav) and not extract_audio(path, wav):
        raise RuntimeError(f"audio extraction failed: {path}")
    return [w for w in transcribe_audio(wav) if w.get("start") is not None and w.get("end") is not None]


def classify_files(paths: List[str], work: str) -> Dict[str, Dict]:
    os.makedirs(work, exist_ok=True)
    words = {p: words_for(p, work) for p in paths}                       # ASR: one model, in order
    audio_only = {p: footage.is_audio_only(p) for p in paths}
    with ThreadPoolExecutor(max_workers=4) as ex:                        # shot index: caption calls overlap
        idx = dict(zip(paths, ex.map(lambda p: None if audio_only[p] else shot_index.index_file(p), paths)))
    out = {}
    for p in paths:
        r = footage.classify(words[p], idx[p], get_video_duration(p))
        r["n_words"] = len(words[p])
        out[os.path.basename(p)] = r
    return out


def score_project(proj: str) -> Dict:
    g = json.load(open(os.path.join(SYNTH_DIR, proj, "gold.json")))
    files_dir = os.path.join(SYNTH_DIR, proj, "files")
    res = classify_files([os.path.join(files_dir, f) for f in g["upload_order"]], os.path.join(SYNTH_DIR, proj, "work"))
    with open(os.path.join(SYNTH_DIR, proj, "categories.json"), "w") as f:
        json.dump(res, f, indent=1)
    conf: Dict[str, Dict[str, int]] = {}
    wrong = []
    for t in g["timeline"]:
        gold, got = g["category"][t["file"]], res[t["file"]]["category"]
        conf.setdefault(gold, {}).setdefault(got, 0)
        conf[gold][got] += 1
        if got not in ACCEPT[gold]:
            wrong.append((t["file"], gold, got, round(t["src_end"] - t["src_start"], 1), t["n_words"],
                          res[t["file"]]["n_words"], res[t["file"]]["seconds"]))
    n = len(g["timeline"])
    secs = sum(t["src_end"] - t["src_start"] for t in g["timeline"])
    wrong_secs = sum(w[3] for w in wrong)
    print(f"{proj}: files right {n - len(wrong)}/{n} ({(n - len(wrong)) / n:.0%}), "
          f"by duration {(secs - wrong_secs) / secs:.0%} | gold -> got {conf}", flush=True)
    for w in wrong:
        print(f"   WRONG {w[0]} gold {w[1]} got {w[2]} ({w[3]}s, gold words {w[4]}, ASR words {w[5]}, {w[6]})")
    return {"project": proj, "files": n, "right": n - len(wrong), "confusion": conf}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", action="append")
    ap.add_argument("--job", help="classify the files of a stored job")
    a = ap.parse_args()
    projs = a.project or sorted(p for p in os.listdir(SYNTH_DIR) if os.path.exists(os.path.join(SYNTH_DIR, p, "gold.json")))
    tot = [score_project(p) for p in projs]
    print(f"ALL: {sum(t['right'] for t in tot)}/{sum(t['files'] for t in tot)} files right")
    if a.job:
        from app.utils import artifacts
        files = artifacts.load_job(a.job, "files")
        res = classify_files([f["path"] for f in files], os.path.join(SYNTH_DIR, "_job_work"))
        for name, r in res.items():
            print(f"{name}: {r['category']} {r['seconds']}")
            for s in r["shots"]:
                print(f"    {s['start']:6.2f}-{s['end']:6.2f} {s['category']:10} speech {s['speech_share']:.0%}")


if __name__ == "__main__":
    sys.exit(main())
