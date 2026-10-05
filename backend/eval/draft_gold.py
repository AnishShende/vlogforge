"""Draft a gold file for one clip from the current pipeline's output (Phase 0 / S2).

The draft is a starting point for human review, NOT ground truth:
  keep     = the word-timeline selector's kept clean takes
  exclude  = every other run of spoken words, reason="unreviewed"
  then every edge is snapped to the waveform (eval.snap_gold) and speech the
  ASR never transcribed is listed for the reviewer
A human then fixes spans, sets real exclude reasons, and flips status to
"reviewed". A reviewed gold is never overwritten.

Usage (from backend/, ffmpeg on PATH):
    PYTHONPATH=. python -m eval.draft_gold --clip-id IMG_1614 \
        --video ../uploads/<job>/raw/IMG_1614.MOV [--job-dir DIR] [--force]

Single-video clips only for now (build_egt ingests one file); the gold schema
already supports multiple sources.
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone

import run_mock_pipeline as rmp   # importing enables the whisper/LLM/JEV mocks
from app.config import settings
from app.tasks.word_timeline_redundancy import build_word_timeline, detect_redundancy_on_timeline
from eval.gold import (REPO_ROOT, EVAL_SET_DIR, ExcludeSpan, Gold, KeepSpan, Source,
                       gold_path, load_gold, save_gold, sha256_file)
from eval.snap_gold import envelope, load_audio, snap, unaccounted

logger = logging.getLogger("VlogForge.DraftGold")

# Uncovered words further apart than this start a new exclude span. Draft
# granularity only (one reviewable chunk per attempt), not an editing threshold.
EXCLUDE_SPLIT_GAP_SEC = 1.0


def _default_job_dir(video: str, clip_id: str) -> str:
    """Reuse the upload job dir when the video lives in <job>/raw/ (keeps mock
    cache keys stable); otherwise use a scratch work dir under eval-set/."""
    parent = os.path.dirname(os.path.abspath(video))
    if os.path.basename(parent) == "raw":
        return os.path.dirname(parent)
    return os.path.join(EVAL_SET_DIR, "_work", clip_id)


def _git_rev() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _exclude_runs(timeline, keep):
    """Group words not inside any keep span into contiguous runs, clamped so no
    run overlaps a keep span (keep ends can be capped mid-word by the selector)."""
    seen, words = set(), []
    for w in timeline:                       # overlapping segments can repeat words
        key = (w.source_file, round(w.start_sec, 3), round(w.end_sec, 3), w.word)
        if key not in seen:
            seen.add(key)
            words.append(w)

    def covered(w):
        mid = (w.start_sec + w.end_sec) / 2
        return any(k.source_file == w.source_file and k.start <= mid <= k.end for k in keep)

    runs, cur = [], []
    for w in words:
        if covered(w):
            if cur:
                runs.append(cur)
            cur = []
            continue
        if cur and (w.source_file != cur[-1].source_file
                    or w.start_sec - cur[-1].end_sec > EXCLUDE_SPLIT_GAP_SEC):
            runs.append(cur)
            cur = []
        cur.append(w)
    if cur:
        runs.append(cur)

    out, dropped = [], 0
    for r in runs:
        f, s, e = r[0].source_file, r[0].start_sec, r[-1].end_sec
        for k in keep:
            if k.source_file != f:
                continue
            if k.start < e and s < k.end:      # overlap: trim toward the run's side
                if s < k.start:
                    e = min(e, k.start)
                else:
                    s = max(s, k.end)
        if e - s <= 0:
            dropped += 1
            logger.warning(f"[DRAFT-GOLD] exclude run fully inside a keep span after clamp, dropped: "
                           f"{' '.join(w.word for w in r)!r}")
            continue
        out.append((f, round(s, 3), round(e, 3), " ".join(w.word for w in r)))
    return out, dropped


def draft(clip_id: str, video: str, job_dir: str, force: bool = False) -> str:
    path = gold_path(clip_id)
    if os.path.exists(path):
        existing = load_gold(clip_id, verify_sources=False)
        if existing.status == "reviewed":
            sys.exit(f"REFUSING: {path} is reviewed; edit it by hand instead.")
        if not force:
            sys.exit(f"REFUSING: draft exists at {path}; pass --force to regenerate.")

    egt_doc, _ctx = rmp.build_egt(video=video, job_dir=job_dir)
    timeline = build_word_timeline(egt_doc)
    report = detect_redundancy_on_timeline(egt_doc)

    keep = [KeepSpan(id=f"k{i + 1}", source_file=k["source_file"], start=round(k["start"], 3),
                     end=round(k["end"], 3), text=k["text"],
                     note=",".join(k["flags"]))
            for i, k in enumerate(report["kept"])]
    runs, n_dropped = _exclude_runs(timeline, keep)
    exclude = [ExcludeSpan(id=f"x{i + 1}", source_file=f, start=s, end=e, text=t, reason="unreviewed")
               for i, (f, s, e, t) in enumerate(runs)]

    rel = os.path.relpath(os.path.abspath(video), REPO_ROOT)
    gold = Gold(
        clip_id=clip_id, status="draft",
        sources=[Source(file=os.path.basename(video), path=rel, sha256=sha256_file(video))],
        keep=keep, exclude=exclude,
        drafted_from={
            "tool": "eval.draft_gold",
            "git": _git_rev(),
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "selector": "word_timeline_redundancy",
            "accounting": report["accounting"],
            "mocks": {"llm": settings.enable_mock_llm, "whisper": settings.enable_mock_whisper,
                      "jev": settings.enable_mock_jev},
            "timeline_words": len(timeline),
            "exclude_runs_dropped_by_clamp": n_dropped,
        },
    )
    # ASR edges are unreliable exactly where we cut: snap every edge to the waveform
    # and surface speech the ASR never transcribed (for the reviewer to label).
    envs = {os.path.basename(video): envelope(load_audio(video))}
    rows = snap(gold, envs)
    gaps = unaccounted(gold, envs)
    gold.drafted_from["edges_snapped"] = sum(r["new"] != r["old"] for r in rows)
    gold.drafted_from["edge_flags"] = sorted({f"{r['id']}:{r[k]}" for r in rows for k in ("start_flag", "end_flag")
                                              if r[k] in ("no-boundary", "no-speech", "moved-in-past-speech")})
    gold.drafted_from["unaccounted_speech"] = [[f, s, e] for f, s, e in gaps]
    out = save_gold(gold)
    print_review_sheet(gold)
    print(f"\nflagged edges: {gold.drafted_from['edge_flags']}")
    print(f"UNACCOUNTED speech (label these by hand): {len(gaps)}")
    for f, s, e in gaps:
        print(f"  {f} [{s:7.2f}-{e:7.2f}] {e - s:5.2f}s")
    print(f"\nwrote {out}")
    return out


def print_review_sheet(gold: Gold):
    rows = [("KEEP", s.id, s.source_file, s.start, s.end, s.text, s.note) for s in gold.keep]
    rows += [("excl", s.id, s.source_file, s.start, s.end, s.text, s.reason) for s in gold.exclude]
    rows.sort(key=lambda r: (r[2], r[3]))
    print("\n" + "#" * 78)
    print(f"# DRAFT GOLD — {gold.clip_id} — {len(gold.keep)} keep / {len(gold.exclude)} exclude")
    print("#" * 78)
    for kind, i, _f, s, e, text, extra in rows:
        print(f"{kind:4} {i:>4} [{s:7.2f}-{e:7.2f}] {extra:>10}  {text}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--job-dir", default=None)
    ap.add_argument("--force", action="store_true", help="overwrite an existing DRAFT (never a reviewed gold)")
    a = ap.parse_args()
    draft(a.clip_id, a.video, a.job_dir or _default_job_dir(a.video, a.clip_id), a.force)


if __name__ == "__main__":
    main()
