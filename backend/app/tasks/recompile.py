"""Re-compile only (roadmap Phase 5): apply an edited plan to a finished job without re-running
ingest, transcription or cleanup. Loads the job's stored word grid and source files (artifact
store, so it works after a restart), rebuilds the speech envelopes (~1.4 s per 6-min file) and
runs compile + render + validation."""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

from app.models import CompiledTimeline, EditPlan, WordGrid
from app.tasks.compiler import FPS, compile_and_render
from app.utils import artifacts
from app.utils.speech_activity import envelope, load_audio

logger = logging.getLogger("VlogForge.Recompile")


def save_job_edit(job_id: str, grid: WordGrid, files: List[Dict], plan: EditPlan, timeline: CompiledTimeline,
                  info: Dict, cleanup_ranges=None) -> None:
    """Everything a later re-compile needs. files: [{filename, path, audio_path}]. A re-run of the
    pipeline must not silently destroy the user's edit: an existing plan that differs from the new
    one is archived as plan.prev-<time> first."""
    old_plan = artifacts.load_job(job_id, "plan")
    if old_plan is not None and old_plan != plan.model_dump():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        artifacts.save_job(job_id, f"plan.prev-{stamp}", old_plan)
        logger.warning(f"[RECOMPILE] job {job_id}: re-run replaces a different stored plan; archived as plan.prev-{stamp}")
    artifacts.save_job(job_id, "grid", grid.model_dump())
    artifacts.save_job(job_id, "files", files)
    artifacts.save_job(job_id, "plan", plan.model_dump())
    artifacts.save_job(job_id, "timeline", timeline.model_dump())
    artifacts.save_job(job_id, "validation", info.get("validation"))
    if cleanup_ranges is not None:
        artifacts.save_job(job_id, "cleanup", cleanup_ranges)


def can_recompile(job_id: str) -> bool:
    return artifacts.load_job(job_id, "grid") is not None and artifacts.load_job(job_id, "files") is not None


def load_job_edit(job_id: str) -> Optional[Dict]:
    """Review view of a job's stored edit (roadmap Phase 6): the words, cleanup ranges with their
    reasons, the current plan, and where each kept word plays in the output. None = no stored edit."""
    grid_d, files = artifacts.load_job(job_id, "grid"), artifacts.load_job(job_id, "files")
    if grid_d is None or files is None:
        return None
    grid = WordGrid(**grid_d)
    timeline_d = artifacts.load_job(job_id, "timeline")
    word_out: Dict[str, float] = {}
    segments: List[Dict] = []                     # the EDL: output segments in play order
    if timeline_d:
        by_id, t = grid.by_id(), 0.0
        moments = (artifacts.load_job(job_id, "moments") or {}).get("moments", [])
        moment_words = [(m, set(m["word_ids"])) for m in moments]
        for s in timeline_d["segments"]:          # output time = segment's output offset + offset in source
            for wid in s["word_ids"]:
                word_out[wid] = round(t + by_id[wid].start - s["src_in"], 3)
            dur = round((s["src_out"] - s["src_in"]) * FPS) / FPS
            segments.append({"source_file": s["source_file"], "src_in": round(s["src_in"], 3), "src_out": round(s["src_out"], 3),
                             "rec_in": round(t, 3), "rec_out": round(t + dur, 3), "word_ids": s["word_ids"],
                             "moments": [{k: m[k] for k in ("id", "function", "summary")}   # by shared words:
                                         for m, ids in moment_words if ids & set(s["word_ids"])]})  # ids change on refresh
            t += dur
    return {
        "job_id": job_id,
        "files": [f["filename"] for f in files],
        "words": [{"id": w.id, "text": w.text, "source_file": w.source_file, "start": w.start, "end": w.end}
                  for w in grid.words],
        "ranges": artifacts.load_job(job_id, "cleanup") or [],
        "plan": artifacts.load_job(job_id, "plan"),
        "word_out": word_out,
        "segments": segments,
        "duration_sec": timeline_d["duration_sec"] if timeline_d else None,
        "validation": artifacts.load_job(job_id, "validation"),
        "story": _story_view(job_id),
    }


def _story_view(job_id: str) -> Optional[Dict]:
    """Phase 9 candidate plans for the editor (None when the job has no story plan)."""
    story = artifacts.load_job(job_id, "storyplan")
    if story is None:
        return None
    moments = artifacts.load_job(job_id, "moments") or {"moments": []}
    return {"target_sec": story["target_sec"], "full_sec": story["full_sec"],
            "moments": [{k: m[k] for k in ("id", "function", "importance", "summary", "depends_on")} for m in moments["moments"]],
            "plans": [{k: p.get(k) for k in ("strategy", "reasoning", "order", "duration_sec", "score", "repairs", "edit_plan")}
                      for p in story["plans"]]}


def recompile_job(job_id: str, plan: EditPlan, output_path: str) -> Tuple[CompiledTimeline, Dict]:
    grid_d, files = artifacts.load_job(job_id, "grid"), artifacts.load_job(job_id, "files")
    if grid_d is None or files is None:
        raise LookupError(f"job {job_id}: no stored word grid / files (was it made with the word-grid compiler?)")
    grid = WordGrid(**grid_d)
    missing = [f["audio_path"] for f in files if not os.path.exists(f["audio_path"])]
    missing += [f["path"] for f in files if not os.path.exists(f["path"])]
    if missing:
        raise FileNotFoundError(f"job {job_id}: source media gone: {missing}")
    envs = {f["filename"]: envelope(load_audio(f["audio_path"])) for f in files}
    timeline, info = compile_and_render(plan, grid, envs, {f["filename"]: f["path"] for f in files}, output_path)
    artifacts.save_job(job_id, "plan", plan.model_dump())
    artifacts.save_job(job_id, "timeline", timeline.model_dump())
    artifacts.save_job(job_id, "validation", info.get("validation"))
    logger.info(f"[RECOMPILE] job {job_id}: {len(plan.segments)} plan segments -> {timeline.duration_sec:.2f}s, "
                f"validation {info['validation']['status']}")
    return timeline, info


def refresh_story(job_id: str, plan: EditPlan) -> str:
    """After a manual re-compile: rebuild the moment map and the story-plan OPTIONS for the edited
    version (Phases 8-9). Never changes the user's plan. Skipped when the kept words did not change
    (a plan switch only reorders) or the job has no story data."""
    old = artifacts.load_job(job_id, "moments")
    story_old = artifacts.load_job(job_id, "storyplan")
    if old is None and story_old is None:
        return "no story data"
    grid = WordGrid(**artifacts.load_job(job_id, "grid"))
    index = {w.id: i for i, w in enumerate(grid.words)}
    kept = {grid.words[i].id for s in plan.segments for i in range(index[s.word_start], index[s.word_end] + 1)}
    if old is not None and kept == {i for m in old["moments"] for i in m["word_ids"]}:
        return "unchanged"
    from app.tasks.compiler import speaker_pause_targets
    from app.tasks.moments import build_moments
    from app.tasks.storyplan import edit_plan, plan_story
    mo = build_moments(grid, kept)
    artifacts.save_job(job_id, "moments", mo)
    if story_old is not None:
        files = artifacts.load_job(job_id, "files")
        envs = {f["filename"]: envelope(load_audio(f["audio_path"])) for f in files}
        story = plan_story(grid, mo, envs, speaker_pause_targets(grid, envs), story_old.get("target_sec"))
        for p in story["plans"]:
            p["edit_plan"] = edit_plan(grid, mo["moments"], p["order"]).model_dump()
        artifacts.save_job(job_id, "storyplan", story)
    logger.info(f"[RECOMPILE] job {job_id}: story refreshed for the edited version ({len(mo['moments'])} moments)")
    return "refreshed"
