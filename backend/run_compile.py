"""Compile a hand-written edit plan into a video (Archdoc roadmap Phase 2 / S5).

Word grid (Phase 1, ASR choice E) -> EditPlan -> compiler (S2 cuts, S3 pause shortening)
-> single-pass render (S4). No LLM, no EDL. The whisper cache is replayed when the
video's analysis audio was transcribed before (same <job>/audio/<name>.wav).

Usage (from backend/, ffmpeg on PATH):
    # 1. see the words and their IDs
    PYTHONPATH=. python run_compile.py VIDEO [VIDEO ...] --dump-grid grid.json
    # 2. write plan.json: {"segments": [{"word_start": "w_..", "word_end": "w_..", "reason": ".."}],
    #                      "grid_fingerprint": "<from grid.json, optional>"}
    # 3. compile + render (writes OUT and OUT.timeline.json)
    PYTHONPATH=. python run_compile.py VIDEO [VIDEO ...] --plan plan.json --out out.mp4 [--no-pauses]
    # or let Phase 4 speech cleanup write the plan (retakes removed, all unique speech kept;
    # writes OUT.cleanup.json with every word range's label and reason)
    PYTHONPATH=. python run_compile.py VIDEO [VIDEO ...] --cleanup --out out.mp4
"""

import argparse
import json
import os
import sys
from collections import Counter

from app.config import settings
from app.models import EditPlan
from app.tasks.compiler import PlanError, compile_and_render
from app.tasks.transcribe import transcribe_audio
from app.tasks.word_grid import build_word_grid, check_word_grid, summarize_word_grid
from app.utils.ffmpeg import extract_audio
from app.utils.speech_activity import envelope, load_audio


def analysis_wav(video: str) -> str:
    """The ingest stage's audio path for this video (reused so whisper cache keys match)."""
    parent = os.path.dirname(os.path.abspath(video))
    job_dir = (os.path.dirname(parent) if os.path.basename(parent) == "raw"
               else os.path.join(settings.output_dir, "compile", os.path.splitext(os.path.basename(video))[0]))
    wav = os.path.join(job_dir, "audio", os.path.splitext(os.path.basename(video))[0] + ".wav")
    if not os.path.exists(wav):
        os.makedirs(os.path.dirname(wav), exist_ok=True)
        print(f"[compile] extracting analysis audio -> {wav}")
        if not extract_audio(video, wav):
            sys.exit(f"audio extraction failed for {video}")
    return wav


def build_grid(videos):
    settings.enable_word_grid = True
    settings.enable_mock_whisper = True       # replay when cached; otherwise transcribe (and record)
    transcript, envs = [], {}
    for v in videos:
        f, wav = os.path.basename(v), analysis_wav(v)
        transcript += [{**t, "video_file": f} for t in transcribe_audio(wav)]
        envs[f] = envelope(load_audio(wav))
    grid = check_word_grid(build_word_grid(transcript, asr={
        "transcriber": "faster-whisper turbo", "aligner": "whisperx wav2vec2" if settings.enable_forced_alignment else None}), envs)
    print(f"[compile] grid {grid.fingerprint()}: {json.dumps(summarize_word_grid(grid))}")
    return grid, envs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--dump-grid", metavar="JSON")
    ap.add_argument("--plan", metavar="JSON")
    ap.add_argument("--cleanup", action="store_true", help="plan = Phase 4 speech cleanup of the grid")
    ap.add_argument("--out", metavar="MP4")
    ap.add_argument("--no-pauses", action="store_true", help="S3 pause shortening off")
    a = ap.parse_args()
    if not a.dump_grid and not ((a.plan or a.cleanup) and a.out):
        ap.error("give --dump-grid, or --plan / --cleanup with --out")
    for v in a.videos:
        if not os.path.exists(v):
            sys.exit(f"video not found: {v}")

    grid, envs = build_grid(a.videos)
    if a.dump_grid:
        with open(a.dump_grid, "w") as f:
            json.dump({"grid_fingerprint": grid.fingerprint(),
                       "words": [{"id": w.id, "text": w.text, "source_file": w.source_file, "start": round(w.start, 3),
                                  "end": round(w.end, 3), "flags": w.flags} for w in grid.words]}, f, indent=1, ensure_ascii=False)
        print(f"[compile] wrote {len(grid.words)} words -> {a.dump_grid}")
    if a.plan or a.cleanup:
        if a.cleanup:
            from app.tasks.speech_cleanup import REMOVE, REVIEW, cleanup_plan, label_grid, label_ranges
            settings.enable_mock_jev = True                # replay recorded JEV scores; new ones are recorded
            labels = label_grid(grid)
            plan = cleanup_plan(grid, labels)
            ranges = label_ranges(grid, labels)
            with open(a.out + ".cleanup.json", "w") as f:
                json.dump(ranges, f, indent=1, ensure_ascii=False, default=str)
            print(f"[compile] cleanup: removed {sum(l['label'] == REMOVE for l in labels)} of {len(labels)} words; "
                  f"{sum(r['label'] == REVIEW for r in ranges)} line(s) to review (best take used) -> {a.out}.cleanup.json")
        else:
            with open(a.plan) as f:
                plan = EditPlan.model_validate(json.load(f))
        file_map = {os.path.basename(v): os.path.abspath(v) for v in a.videos}
        try:
            timeline, info = compile_and_render(plan, grid, envs, file_map, a.out, pauses=not a.no_pauses)
        except PlanError as e:
            sys.exit(str(e))
        with open(a.out + ".timeline.json", "w") as f:
            json.dump({**timeline.model_dump(), "render": info}, f, indent=1)
        flags = Counter(fl for s in timeline.segments for c in (s.cut_in, s.cut_out) for fl in c.flags)
        shortened = [s.pause_shortened_before_sec for s in timeline.segments if s.pause_shortened_before_sec]
        print(f"[compile] {len(plan.segments)} plan segments -> {len(timeline.segments)} cuts, {timeline.duration_sec:.2f}s; "
              f"cut flags {dict(flags) or 'none'}; pauses shortened {len(shortened)} (targets {info['pause_targets']})")
        v = info["validation"]
        print(f"[compile] validation {v['status']}: " + (", ".join(f"{c['status']} {c['check']} (segment {c['segment']})"
              for c in v["checks"] if c["status"] != "pass") or "all checks pass"))
        print(f"[compile] wrote {a.out} and {a.out}.timeline.json")


if __name__ == "__main__":
    main()
