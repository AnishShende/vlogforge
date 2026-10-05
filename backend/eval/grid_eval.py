"""Word grid evaluation (Archdoc roadmap Phase 1).

S2: build the word grid for every eval clip (word grid ON => whisper cache v2,
with confidence + aligner-dropped words interpolated) and check stability.
S3: score the grid against gold, before and after speech-activity checks:
  boundary error      gold span edge -> nearest word edge of the same kind (start/end)
  gold speech no-word seconds of speech (Silero VAD) inside gold spans that no word
                      covers (words < WORD_BRIDGE_SEC apart cover the gap between them)
  flags / unaccounted / possible speech  from app.tasks.word_grid.check_word_grid
Caveat: gold edges were themselves snapped with the same Silero VAD, so edge
refinement toward VAD boundaries is not an independent improvement measure.

Usage (from backend/, ffmpeg on PATH):
    PYTHONPATH=. python -m eval.grid_eval                    # build (replays v2 cache if present)
    PYTHONPATH=. python -m eval.grid_eval --fresh            # bypass the cache: new transcription
    PYTHONPATH=. python -m eval.grid_eval --compare A.json B.json
"""

import argparse
import json
import os
import sys
from datetime import datetime

from app.config import settings
from app.tasks.transcribe import transcribe_audio
import copy

import numpy as np

from app.tasks.word_grid import build_word_grid, check_word_grid, summarize_word_grid, word_coverage
from app.utils.speech_activity import HOP_SEC, envelope, load_audio
from eval.draft_gold import _default_job_dir
from eval.gold import EVAL_SET_DIR, REPO_ROOT, all_spans, load_gold, load_manifest


def clip_audio(clip_id: str) -> dict:
    """{source_file: analysis wav path} for a clip, from its ingest job dir."""
    gold = load_gold(clip_id, verify_sources=False)
    out = {}
    for src in gold.sources:
        video = os.path.join(REPO_ROOT, src.path)
        wav = os.path.join(_default_job_dir(video, clip_id), "audio",
                           os.path.splitext(src.file)[0] + ".wav")
        if not os.path.exists(wav):
            sys.exit(f"{clip_id}: analysis audio not found at {wav} (run the pipeline/draft once)")
        out[src.file] = wav
    return out


def build_clip_grid(clip_id: str):
    transcript = []
    for source_file, wav in clip_audio(clip_id).items():
        for t in transcribe_audio(wav):
            transcript.append({**t, "video_file": source_file})
    return build_word_grid(transcript, asr={"transcriber": "faster-whisper turbo",
                                            "aligner": "whisperx wav2vec2" if settings.enable_forced_alignment else None})


def grid_vs_gold(grid, gold, envs) -> dict:
    """Boundary error and gold-speech coverage of a grid against a gold."""
    errs, missing = [], {}
    speech_sec = missing_sec = 0.0
    for f, env in envs.items():
        ws = [w for w in grid.words if w.source_file == f]
        if not ws:
            continue
        starts = np.array([w.start for w in ws]); ends = np.array([w.end for w in ws])
        covered = np.zeros(len(env.speech), bool)
        for a, b in word_coverage(ws):
            covered[env.t2i(a): env.t2i(b) + 1] = True
        for span in [x for x in all_spans(gold) if x.source_file == f]:
            errs.append(float(np.min(np.abs(starts - span.start))))
            errs.append(float(np.min(np.abs(ends - span.end))))
            a, b = env.t2i(span.start), env.t2i(span.end)
            sp = env.speech[a:b]
            speech_sec += sp.sum() * HOP_SEC
            m = (sp & ~covered[a:b]).sum() * HOP_SEC
            if m >= 0.3:
                missing_sec += m
                missing[span.id] = round(float(m), 1)
    e = np.array(errs)
    return {"boundary_err_ms": {"median": round(float(np.median(e)) * 1000), "p90": round(float(np.percentile(e, 90)) * 1000),
                                "max": round(float(e.max()) * 1000), "over_100ms": int((e > 0.1).sum()), "edges": len(e)},
            "gold_speech_sec": round(speech_sec, 1), "gold_speech_without_words_sec": round(missing_sec, 1),
            "gold_spans_without_words": missing}


def score_clip(clip_id: str, grid) -> dict:
    gold = load_gold(clip_id)
    envs = {src.file: envelope(load_audio(os.path.join(REPO_ROOT, src.path))) for src in gold.sources}
    raw = grid_vs_gold(grid, gold, envs)
    checked = check_word_grid(copy.deepcopy(grid), envs)
    after = grid_vs_gold(checked, gold, envs)
    poss = checked.checks["possible_speech"]
    poss_hits = [s.id for s in all_spans(gold)
                 if any(a < s.end and s.start < b for a, b in poss.get(s.source_file, []))]
    return {"raw": raw, "checked": after, "summary": summarize_word_grid(checked),
            "possible_speech_overlapping_gold": poss_hits}


def compare(a_path: str, b_path: str) -> dict:
    a, b = json.load(open(a_path)), json.load(open(b_path))
    out = {}
    for clip in sorted(set(a) | set(b)):
        wa = {w["id"]: w for w in a.get(clip, {}).get("words", [])}
        wb = {w["id"]: w for w in b.get(clip, {}).get("words", [])}
        same_ids = list(wa) == list(wb)
        text_diff = sum(1 for i in set(wa) & set(wb) if wa[i]["text"] != wb[i]["text"])
        time_diff = max((abs(wa[i]["start"] - wb[i]["start"]) + abs(wa[i]["end"] - wb[i]["end"])
                         for i in set(wa) & set(wb)), default=0.0)
        out[clip] = {"ids_identical": same_ids, "words": (len(wa), len(wb)),
                     "text_differs": text_diff, "max_time_diff_sec": round(time_diff, 4)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", action="append")
    ap.add_argument("--fresh", action="store_true", help="bypass the whisper cache (new transcription)")
    ap.add_argument("--out", default=None, help="write grids JSON here")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--score", action="store_true", help="S3: score grids against gold")
    a = ap.parse_args()
    if a.compare:
        print(json.dumps(compare(*a.compare), indent=2))
        return

    settings.enable_word_grid = True
    settings.enable_mock_whisper = not a.fresh
    clips = [c["clip_id"] for c in load_manifest()["clips"] if not a.clip or c["clip_id"] in a.clip]
    grids = {}
    for clip_id in clips:
        g = build_clip_grid(clip_id)
        grids[clip_id] = g.model_dump()
        print(f"[grid] {clip_id}: {json.dumps(summarize_word_grid(g))}")
        if a.score:
            r = score_clip(clip_id, g)
            print(f"[score] {clip_id}: raw     {json.dumps(r['raw'])}")
            print(f"[score] {clip_id}: checked {json.dumps(r['checked'])}")
            print(f"[score] {clip_id}: summary {json.dumps(r['summary'])}")
            print(f"[score] {clip_id}: possible speech overlapping gold spans: {r['possible_speech_overlapping_gold']}")
    out = a.out or os.path.join(EVAL_SET_DIR, "_work", "grids",
                                f"grids_{'fresh' if a.fresh else 'cached'}_{datetime.now():%Y%m%d-%H%M%S}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(grids, f)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
