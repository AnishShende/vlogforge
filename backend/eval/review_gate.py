"""Phase 6 exit gate: restore dropped words through the review edit path -> re-compile -> the new
render carries exactly those words more (validator + timeline word ids prove it).

Per clip: run the real orchestrator (flags ON, mocked perception), read the stored edit view
(GET /api/jobs/{id}/edit), then:
  - if cleanup removed words: restore the first removed range;
  - else: remove the first kept run between two pauses, re-compile, then restore it.
Each step builds the plan from a kept-word set, the same way the review UI does.

Usage (from backend/, ffmpeg on PATH):
    PYTHONPATH=. python -m eval.review_gate                     # every clip in the eval set
    PYTHONPATH=. python -m eval.review_gate --clip IMG_1614
"""

import argparse
import os
import sys
from typing import Dict, List, Set

import run_mock_pipeline  # noqa: F401  importing enables the whisper/LLM/JEV mocks
from app.config import settings
from app.models import EditPlan, EditSegment
from app.tasks.orchestrator import run_pipeline_sync
from app.tasks.recompile import load_job_edit, recompile_job
from eval.gold import REPO_ROOT, load_gold, load_manifest

PAUSE_SEC = 0.5   # a "run" for the remove-then-restore case ends at a pause this long


def plan_from_kept(words: List[Dict], kept: Set[str]) -> EditPlan:
    """Consecutive kept words of one file -> one segment, in source order (mirrors the UI)."""
    segs: List[EditSegment] = []
    prev = None
    for w in words:
        if w["id"] in kept:
            if prev is not None and prev["id"] in kept and prev["source_file"] == w["source_file"]:
                segs[-1].word_end = w["id"]
            else:
                segs.append(EditSegment(word_start=w["id"], word_end=w["id"], reason="review"))
        prev = w
    return EditPlan(segments=segs)


def kept_ids(view: Dict) -> Set[str]:
    index = {w["id"]: i for i, w in enumerate(view["words"])}
    out: Set[str] = set()
    for s in view["plan"]["segments"]:
        out |= {view["words"][i]["id"] for i in range(index[s["word_start"]], index[s["word_end"]] + 1)}
    return out


def range_ids(view: Dict, start: str, end: str) -> List[str]:
    index = {w["id"]: i for i, w in enumerate(view["words"])}
    return [view["words"][i]["id"] for i in range(index[start], index[end] + 1)]


def step(job_id: str, view: Dict, kept: Set[str], label: str) -> Dict:
    """Re-compile with `kept`, return the new view; print the evidence."""
    out = os.path.join(settings.output_dir, f"{job_id}.mp4")
    timeline, info = recompile_job(job_id, plan_from_kept(view["words"], kept), out)
    rendered = {wid for s in timeline.segments for wid in s.word_ids}
    v = info["validation"]
    bad = [f"{c['status']}:{c['check']}@{c['segment']}" for c in v["checks"] if c["status"] != "pass"]
    cov = [c for c in v["checks"] if c["check"] in ("plan_coverage", "unintended_word")]
    print(f"  [{label}] {timeline.duration_sec:.2f}s, {len(rendered)} words rendered, validation {v['status']} "
          f"{bad or ''}")
    for c in cov:
        print(f"      {c['status']:4} {c['check']}: {c['detail']}")
    return {"rendered": rendered, "validation": v, "view": load_job_edit(job_id)}


def gate_clip(clip_id: str) -> bool:
    entry = next(c for c in load_manifest()["clips"] if c["clip_id"] == clip_id)
    gold = load_gold(clip_id)
    job_id = f"review-gate-{clip_id}"
    view = load_job_edit(job_id)
    if view is None:
        videos = [os.path.join(REPO_ROOT, s.path) for s in gold.sources]
        print(f"[{clip_id}] running orchestrator (word-grid compiler, mocks) on {len(videos)} file(s)...")
        run_pipeline_sync(job_id, videos, entry.get("context_text", ""), entry.get("target_duration_sec", 60.0),
                          entry.get("genre", "default"))
        view = load_job_edit(job_id)
        assert view is not None, f"{clip_id}: job stored no edit"
    else:
        print(f"[{clip_id}] reusing stored job {job_id}")
    before = kept_ids(view)
    removed = [r for r in view["ranges"] if r["label"] == "remove"]
    print(f"  {len(view['words'])} words, {len(before)} kept, {len(removed)} removed range(s) "
          f"({sum(len(range_ids(view, r['word_start'], r['word_end'])) for r in removed)} words), "
          f"output {view['duration_sec']}s, {len(view['word_out'])} words mapped to output time")

    if removed:
        r = removed[0]
        target = range_ids(view, r["word_start"], r["word_end"])
        print(f"  restore range {r['word_start']}..{r['word_end']} ({len(target)} words, reason {r['reason']}, "
              f"{r['start']:.2f}-{r['end']:.2f}s)")
        base = {"rendered": before, "view": view}
    else:
        run, words = [], view["words"]
        for i, w in enumerate(words):
            if w["id"] not in before:
                continue
            if run and (w["source_file"] != words[i - 1]["source_file"] or w["start"] - words[i - 1]["end"] >= PAUSE_SEC):
                if len(run) >= 3:
                    break
                run = []
            run.append(w["id"])
        target = run
        print(f"  no removals: remove then restore run {target[0]}..{target[-1]} ({len(target)} words)")
        base = step(job_id, view, before - set(target), "remove")
        if base["rendered"] != before - set(target):
            print("  FAIL remove step: rendered words != kept set")
            return False
        view = base["view"]

    after = step(job_id, view, kept_ids(view) | set(target), "restore")
    added, lost = after["rendered"] - base["rendered"], base["rendered"] - after["rendered"]
    ok = added == set(target) and not lost and after["validation"]["status"] != "fail"
    print(f"  new render: +{len(added)} words (expected {len(target)}), -{len(lost)}; "
          f"exactly the restored words: {added == set(target)} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", help="clip id (repeatable); default: whole eval set")
    a = ap.parse_args()
    settings.enable_word_grid = settings.enable_word_grid_compiler = True
    clips = a.clip or [c["clip_id"] for c in load_manifest()["clips"]]
    results = {c: gate_clip(c) for c in clips}
    print("\nPhase 6 gate:", ", ".join(f"{c} {'PASS' if ok else 'FAIL'}" for c, ok in results.items()))
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
