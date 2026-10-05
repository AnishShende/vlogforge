"""Snap gold span edges to the waveform (Archdoc Phase 0 / S2).

ASR timings are wrong exactly where we cut: aligners park words on silence and
stretch them over untranscribed speech. Gold edges must not inherit that, so
every keep/exclude edge is moved to the nearest speech onset/offset found from
the audio alone, never from ASR timings.

Speech detection and edge snapping live in app/utils/speech_activity.py (shared
with the pipeline's word grid checks); this module adds the gold-specific parts.

Also reports UNACCOUNTED speech: detected speech covered by no span
(e.g. speech the ASR never transcribed). Reported only, never auto-labelled.

Usage (from backend/, ffmpeg on PATH):
    PYTHONPATH=. python -m eval.snap_gold --clip-id IMG_1614          # dry run
    PYTHONPATH=. python -m eval.snap_gold --clip-id IMG_1614 --write
"""

import argparse
import os
from typing import Dict, List, Tuple

from app.utils.speech_activity import (HOP_SEC, MIN_UNACCOUNTED_SEC, SR, Envelope,  # noqa: F401 (re-exported)
                                       envelope, load_audio, snap_end, snap_start, uncovered_speech)
from eval.gold import REPO_ROOT, Gold, all_spans, load_gold, save_gold


def snap(gold: Gold, envs: Dict[str, Envelope]) -> List[dict]:
    """Snap every span edge in place. Returns one report row per span."""
    rows = []
    for kind, spans in (("keep", gold.keep), ("exclude", gold.exclude), ("optional", gold.optional)):
        for s in spans:
            if s.edges == "hand":                       # hand-verified edges are never moved
                rows.append({"kind": kind, "id": s.id, "old": (s.start, s.end), "new": (s.start, s.end),
                             "start_flag": "hand", "end_flag": "hand"})
                continue
            env = envs[s.source_file]
            others = [o for o in all_spans(gold) if o is not s and o.source_file == s.source_file]
            labelled = lambda t: any(o.start <= t <= o.end for o in others)
            ns, fs = snap_start(env, s.start, s.end, labelled(s.start - 0.05))
            ne, fe = snap_end(env, s.end, s.start, labelled(s.end + 0.05))
            if ne <= ns:                                # degenerate: keep original, flag
                ns, ne, fs, fe = s.start, s.end, "kept-original", "kept-original"
            rows.append({"kind": kind, "id": s.id, "old": (s.start, s.end), "new": (ns, ne),
                         "start_flag": fs, "end_flag": fe})
            s.start, s.end = ns, ne
    return rows


def unaccounted(gold: Gold, envs: Dict[str, Envelope]) -> List[Tuple[str, float, float]]:
    """Detected speech covered by no gold span, per source file."""
    return [(f, s, e) for f, env in envs.items()
            for s, e in uncovered_speech(env, [(x.start, x.end) for x in all_spans(gold) if x.source_file == f])]


def run(clip_id: str, write: bool):
    gold = load_gold(clip_id)
    envs = {}
    for src in gold.sources:
        envs[src.file] = envelope(load_audio(os.path.join(REPO_ROOT, src.path)))
        e = envs[src.file]
        print(f"[{src.file}] method={e.method}  level floor {e.floor_db:.1f} dB  peak {e.peak_db:.1f} dB  "
              f"speech frames {e.speech.mean():.1%}")
    rows = snap(gold, envs)
    print(f"\n{'span':>8} {'old':>17} {'new':>17} {'d_start':>8} {'d_end':>7}  flags")
    for r in rows:
        (os_, oe), (ns, ne) = r["old"], r["new"]
        print(f"{r['kind'][:4]} {r['id']:>3} [{os_:7.2f}-{oe:7.2f}] [{ns:7.2f}-{ne:7.2f}] "
              f"{ns - os_:+8.2f} {ne - oe:+7.2f}  {r['start_flag']}/{r['end_flag']}")
    gold = Gold.model_validate(gold.model_dump())       # re-check overlaps after moving edges
    gaps = unaccounted(gold, envs)
    print(f"\nUNACCOUNTED speech (>= {MIN_UNACCOUNTED_SEC}s, in no span): {len(gaps)}")
    for f, s, e in gaps:
        print(f"  {f} [{s:7.2f}-{e:7.2f}] {e - s:5.2f}s")
    if write:
        print(f"\nwrote {save_gold(gold)}")
    else:
        print("\ndry run (pass --write to save)")
    return rows, gaps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    run(a.clip_id, a.write)


if __name__ == "__main__":
    main()
