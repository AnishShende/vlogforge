"""Re-compile only (roadmap Phase 5): apply an edited plan to a finished job without re-running
ingest, transcription or cleanup. Loads the job's stored word grid and source files (artifact
store, so it works after a restart), rebuilds the speech envelopes (~1.4 s per 6-min file) and
runs compile + render + validation."""

import logging
import os
from typing import Dict, List, Tuple

from app.models import CompiledTimeline, EditPlan, WordGrid
from app.tasks.compiler import compile_and_render
from app.utils import artifacts
from app.utils.speech_activity import envelope, load_audio

logger = logging.getLogger("VlogForge.Recompile")


def save_job_edit(job_id: str, grid: WordGrid, files: List[Dict], plan: EditPlan, timeline: CompiledTimeline,
                  info: Dict, cleanup_ranges=None) -> None:
    """Everything a later re-compile needs. files: [{filename, path, audio_path}]."""
    artifacts.save_job(job_id, "grid", grid.model_dump())
    artifacts.save_job(job_id, "files", files)
    artifacts.save_job(job_id, "plan", plan.model_dump())
    artifacts.save_job(job_id, "timeline", timeline.model_dump())
    artifacts.save_job(job_id, "validation", info.get("validation"))
    if cleanup_ranges is not None:
        artifacts.save_job(job_id, "cleanup", cleanup_ranges)


def can_recompile(job_id: str) -> bool:
    return artifacts.load_job(job_id, "grid") is not None and artifacts.load_job(job_id, "files") is not None


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
