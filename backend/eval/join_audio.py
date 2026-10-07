"""Phase 7 S1: how smooth is the audio at every join of a rendered word-grid edit?

For each join (end of output segment k / start of k+1) in the rendered MP4, compares the audio just
before and just after the cut:
  speech jump   level of speech (90th percentile of 10 ms frame levels) in 1.5 s each side, dB
  floor jump    background level: median of the QUIET frames (25 dB or more below the speech level) in
                3 s each side, frames within 50 ms of the cut (join fades) excluded; n/a with < 10 quiet frames
A join is flagged when either jump exceeds JUMP_DB. Joins between different source files are
reported separately (different rooms or mics are where jumps are expected).

Usage (from backend/, ffmpeg on PATH):
    PYTHONPATH=. python -m eval.join_audio <job_id> [<job_id> ...]
"""

import argparse
import os
from typing import Dict, List

import numpy as np

from app.config import settings
from app.utils import artifacts
from app.utils.speech_activity import SR, load_audio

FRAME_SEC = 0.010
SIDE_SEC = 1.5
JUMP_DB = 6.0


def frame_db(x: np.ndarray) -> np.ndarray:
    n = int(SR * FRAME_SEC)
    f = x[: len(x) // n * n].reshape(-1, n)
    return 20 * np.log10(np.sqrt((f ** 2).mean(axis=1)) + 1e-9)


def joins(job: str) -> List[Dict]:
    tl = artifacts.load_job(job, "timeline")
    if tl is None:
        raise SystemExit(f"job {job}: no stored timeline")
    audio = load_audio(os.path.join(settings.output_dir, f"{job}.mp4"))
    db = frame_db(audio)
    out, t = [], 0.0
    segs = tl["segments"]
    for k, s in enumerate(segs[:-1]):
        t += round((s["src_out"] - s["src_in"]) * tl["fps"]) / tl["fps"]
        i = int(round(t / FRAME_SEC))
        w = int(SIDE_SEC / FRAME_SEC)
        before, after = db[max(0, i - w): i], db[i: i + w]
        if len(before) < 20 or len(after) < 20:
            continue
        g, w2 = 5, 2 * w                                   # skip the join fades; wider window for quiet frames
        sp = min(np.percentile(before, 90), np.percentile(after, 90))
        qb = db[max(0, i - w2): i - g]
        qa = db[i + g: i + w2]
        qb, qa = qb[qb < sp - 25], qa[qa < sp - 25]
        floor_jump = round(float(np.median(qa) - np.median(qb)), 1) if len(qb) >= 10 and len(qa) >= 10 else None
        out.append({"join": k, "out_time": round(t, 2),
                    "files": "same" if s["source_file"] == segs[k + 1]["source_file"] else "different",
                    "speech_jump_db": round(float(np.percentile(after, 90) - np.percentile(before, 90)), 1),
                    "floor_jump_db": floor_jump,
                    "floor_before_db": round(float(np.median(qb)), 1) if len(qb) else None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+")
    a = ap.parse_args()
    for job in a.jobs:
        js = joins(job)
        bad = [j for j in js if abs(j["speech_jump_db"]) > JUMP_DB or abs(j["floor_jump_db"] or 0) > JUMP_DB]
        sp = [abs(j["speech_jump_db"]) for j in js]
        fl = [abs(j["floor_jump_db"]) for j in js if j["floor_jump_db"] is not None]
        print(f"{job}: {len(js)} joins ({sum(j['files'] == 'different' for j in js)} between files) | "
              f"speech jump median {np.median(sp) if sp else 0:.1f} dB, max {max(sp, default=0):.1f} | "
              f"floor jump median {np.median(fl) if fl else 0:.1f} dB, max {max(fl, default=0):.1f} ({len(fl)} measurable) | "
              f"flagged (> {JUMP_DB:.0f} dB): {len(bad)}")
        for j in bad:
            print(f"    join {j['join']:3} at {j['out_time']:7.2f}s ({j['files']} file): speech {j['speech_jump_db']:+.1f} dB, "
                  f"floor {'n/a' if j['floor_jump_db'] is None else format(j['floor_jump_db'], '+.1f')} dB (floor before {j['floor_before_db']} dB)")


if __name__ == "__main__":
    main()
