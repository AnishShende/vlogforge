"""Pipeline Orchestrator.

    Stage 1: Ingest (parallel per file) -> proxy + audio WAV
    Stage 2: Transcribe (faster-whisper + WhisperX alignment, skipped-speech recovery)
    Stage 3: Word grid: every word with its time, checked against the audio
    Stage 4: Edit passes (LLM): dead air, stutters/fillers, retakes, false starts, final review -> EditPlan
    Stage 5: Story plan (LLM): moments -> order + target duration
    Stage 6: Compile + render (frame-accurate cuts on the word grid), then validation
    Review: the transcript editor edits the plan; a re-compile re-runs Stage 6 only.
"""

import os
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Set, Optional
from fastapi import WebSocket
from sqlalchemy.future import select
from app.config import settings
from app.database import AsyncSessionLocal
from app.db_models import Project
from app.models import JobStatus, VideoFileInfo, WSProgressEvent
from app.tasks.ingest import ingest_video
from app.tasks.transcribe import transcribe_audio
from app.tasks.word_grid import build_word_grid, check_word_grid, summarize_word_grid
from app.utils.speech_activity import envelope, load_audio
from app.tasks.compiler import compile_and_render
from app.tasks.edit_passes import edit_with_passes
from app.utils import artifacts as artifacts_mod
from app.tasks.recompile import recompile_job, save_job_edit
from app.models import EditPlan
from app.utils.interaction_logger import interaction_logger

logger = logging.getLogger("VlogForge.Orchestrator")

# Global in-memory job store
jobs_db: Dict[str, JobStatus] = {}
# Global active websocket connections: job_id -> set of WebSockets
websockets_db: Dict[str, Set[WebSocket]] = {}

def get_job(job_id: str) -> Optional[JobStatus]:
    return jobs_db.get(job_id)

def cancel_job(job_id: str):
    if job_id in jobs_db:
        jobs_db[job_id].status = "cancelled"
        jobs_db[job_id].message = "Job cancelled by user."
        logger.info(f"Job {job_id} marked as cancelled.")

def register_websocket(job_id: str, websocket: WebSocket):
    if job_id not in websockets_db:
        websockets_db[job_id] = set()
    websockets_db[job_id].add(websocket)
    logger.info(f"WebSocket registered for job: {job_id}. Active: {len(websockets_db[job_id])}")

def unregister_websocket(job_id: str, websocket: WebSocket):
    if job_id in websockets_db:
        websockets_db[job_id].discard(websocket)
        if not websockets_db[job_id]:
            del websockets_db[job_id]
        logger.info(f"WebSocket unregistered for job: {job_id}")

async def broadcast_progress(job_id: str, stage: str, progress: int, message: str, download_url: Optional[str] = None):
    """Update internal job state and broadcast update to connected WebSockets."""
    warnings_list = []
    # Update JobStatus
    if job_id in jobs_db and stage != "heartbeat":
        job = jobs_db[job_id]
        job.status = stage
        job.progress = progress
        job.message = message
        warnings_list = job.warnings
        if stage == "complete" and download_url:
            job.output_video_url = download_url
            job.completed_at = datetime.utcnow()
        elif stage == "failed":
            job.completed_at = datetime.utcnow()

    # Update Postgres database for terminal states
    if stage in ["complete", "failed"]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(Project).where(Project.id == job_id))
            project = result.scalars().first()
            if project:
                project.status = stage
                await session.commit()

    # Broadcast
    websockets = websockets_db.get(job_id, set())
    if websockets:
        event = WSProgressEvent(
            stage=stage,
            progress=progress,
            message=message,
            download_url=download_url,
            warnings=warnings_list
        )
        event_json = event.model_dump_json()

        # Gather all websocket sending tasks
        disconnected_ws = set()
        for ws in list(websockets):
            try:
                await ws.send_text(event_json)
            except Exception as e:
                logger.warning(f"Failed to send websocket progress update: {e}")
                disconnected_ws.add(ws)

        # Clean up any dead connections
        for ws in disconnected_ws:
            websockets.discard(ws)

def run_pipeline_sync(job_id: str, video_paths: List[str], context_text: str, target_duration: float = 10.0, main_loop: asyncio.AbstractEventLoop = None):
    """Synchronous pipeline run (to be run in a separate thread)."""

    # Lock to serialize WebSocket broadcasts from concurrent worker threads
    broadcast_lock = threading.Lock()

    def safe_broadcast(stage: str, progress: int, message: str, download_url=None):
        """Thread-safe WebSocket progress broadcast wrapper."""
        with broadcast_lock:
            if main_loop:
                asyncio.run_coroutine_threadsafe(
                    broadcast_progress(job_id, stage, progress, message, download_url), main_loop
                ).result()

    job_dir = os.path.join(settings.upload_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Set up per-job file logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_log_file = os.path.join(settings.log_dir, f"{timestamp}_job_{job_id}.log")
    job_handler = logging.FileHandler(job_log_file, encoding='utf-8')
    job_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(job_handler)
    logger.info(f"--- STARTING PIPELINE JOB: {job_id} ---")

    # Heartbeat thread: sends a ping every 10s to keep the Vite proxy's WebSocket alive
    # during long stages (transcription, LLM passes) that send no progress for a while
    heartbeat_stop = threading.Event()

    def heartbeat_worker():
        while not heartbeat_stop.is_set():
            heartbeat_stop.wait(timeout=10.0)
            if heartbeat_stop.is_set():
                break
            try:
                with broadcast_lock:
                    if main_loop:
                        asyncio.run_coroutine_threadsafe(
                            broadcast_progress(job_id, "heartbeat", -1, "ping"), main_loop
                        ).result()
            except Exception:
                pass  # Ignore errors — websocket may have disconnected cleanly

    heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
    heartbeat_thread.start()

    def check_cancelled():
        if job_id in jobs_db and jobs_db[job_id].status == "cancelled":
            raise RuntimeError("Job cancelled by user.")


    try:
        # ---- Stage 1: Ingest (parallel per file): proxy + audio ----
        check_cancelled()
        safe_broadcast("ingesting", 5, "Validating and extracting audio...")

        # Concurrent per-file ingestion via ThreadPoolExecutor
        ingest_results: Dict[int, dict] = {}
        ingest_errors: Dict[int, str] = {}
        total_files = len(video_paths)

        def ingest_one(idx: int, video_path: str) -> tuple:
            """Worker: ingest a single video file and broadcast progress."""
            filename = os.path.basename(video_path)
            safe_broadcast(
                "ingesting",
                5 + int(10 * (idx / total_files)),
                f"Ingesting video {idx + 1} of {total_files}: {filename}..."
            )
            result = ingest_video(video_path, job_dir)
            return idx, result

        max_workers = min(total_files, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(ingest_one, idx, vp): idx
                for idx, vp in enumerate(video_paths)
            }
            for future in as_completed(futures):
                check_cancelled()
                try:
                    idx, result = future.result()
                    ingest_results[idx] = result
                except Exception as ingest_err:
                    failed_idx = futures[future]
                    failed_path = video_paths[failed_idx]
                    logger.error(f"Ingestion failed for {failed_path}: {ingest_err}")
                    ingest_errors[failed_idx] = str(ingest_err)

        if ingest_errors:
            raise RuntimeError(
                f"Ingestion failed for {len(ingest_errors)} file(s): "
                + "; ".join(f"{video_paths[i]}: {e}" for i, e in ingest_errors.items())
            )

        # Merge results in original upload order to preserve chronology
        files_info = []
        for idx in range(total_files):
            result = ingest_results[idx]
            file_info_dict = result["file_info"].model_dump()
            file_info_dict["cfr_path"] = result.get("cfr_path", video_paths[idx])
            files_info.append(file_info_dict)

        check_cancelled()
        if job_id in jobs_db:
            jobs_db[job_id].files = [VideoFileInfo(**f) for f in files_info]

        # ---- Stage 2: Transcribing (faster-whisper + WhisperX alignment) ----
        check_cancelled()
        safe_broadcast("transcribing", 20, "Starting audio transcription...")

        full_transcript_segments = []
        for idx, f_info in enumerate(files_info):
            check_cancelled()
            audio_path = f_info.get("audio_path")
            filename = f_info.get("filename")

            if audio_path and os.path.exists(audio_path):
                def make_status_callback(file_idx, fname):
                    def status_callback(status_msg: str):
                        safe_broadcast(
                            "transcribing",
                            20 + int(15 * (file_idx / len(files_info))),
                            f"Transcribing {fname} ({status_msg})..."
                        )
                    return status_callback

                transcript = transcribe_audio(
                    audio_path,
                    status_callback=make_status_callback(idx, filename)
                )

                # Tag transcript segments with their original video file
                for t in transcript:
                    t["video_file"] = filename
                    full_transcript_segments.append(t)

        check_cancelled()
        # ---- Stage 3: Word grid (the time reference for every speech edit) ----
        word_grid = build_word_grid(full_transcript_segments, asr={
            "transcriber": "faster-whisper turbo",
            "aligner": "whisperx wav2vec2" if settings.enable_forced_alignment else None,
        })
        envs = {f["filename"]: envelope(load_audio(f["audio_path"]))
                for f in files_info if f.get("audio_path") and os.path.exists(f["audio_path"])}
        missing = {w.source_file for w in word_grid.words} - set(envs)
        if missing:
            raise RuntimeError(f"No analysis audio for {sorted(missing)}: cannot compile the edit")
        word_grid = check_word_grid(word_grid, envs)
        logger.info(f"[WORD-GRID] {summarize_word_grid(word_grid)}")

        check_cancelled()
        final_video_name = f"{job_id}.mp4"
        final_video_path = os.path.join(settings.output_dir, final_video_name)
        file_map = {f["filename"]: f.get("original_path") or f.get("cfr_path") for f in files_info}
        # ---- Stage 4: Edit passes (dead air, stutters, retakes, false starts, final review) ----
        safe_broadcast("edl_generating", 72, "Editing speech: retakes, false starts, final review...")
        plan, ranges, _pass_stats = edit_with_passes(word_grid)
        story, moments_result, story_warnings = None, None, []
        # ---- Stage 5: Story plan (order + target duration over the moment map) ----
        if settings.enable_story_plan:
            safe_broadcast("edl_generating", 80, "Mapping the story and planning the cut...")
            try:
                from app.tasks.compiler import speaker_pause_targets
                from app.tasks.moments import build_moments, clip_order
                from app.tasks.storyplan import edit_plan, plan_story
                index = {w.id: i for i, w in enumerate(word_grid.words)}
                kept = {word_grid.words[i].id for s_ in plan.segments
                        for i in range(index[s_.word_start], index[s_.word_end] + 1)}
                clips = clip_order([{"filename": f["filename"], "path": file_map[f["filename"]]} for f in files_info])
                moments_result = build_moments(word_grid, kept, clips)
                story = plan_story(word_grid, moments_result, envs, speaker_pause_targets(word_grid, envs),
                                   target_duration or None)   # 0 = "Auto": no target, keep every moment
                story_warnings += story["warnings"]
                for p_ in story["plans"]:
                    p_["edit_plan"] = edit_plan(word_grid, moments_result["moments"], p_["order"]).model_dump()
                plan = EditPlan(**story["best"]["edit_plan"])
                best = story["best"]
                if target_duration and best["duration_sec"] > target_duration * 1.15:
                    story_warnings.append(f"Target {target_duration:.0f}s not reachable without breaking the story: "
                                          f"shortest valid cut is {best['duration_sec']:.0f}s.")
            except Exception as e:          # never fail the job: fall back to the cleaned edit in source order
                logger.warning(f"[STORYPLAN] job {job_id}: story plan failed, using source order: {e}", exc_info=True)
                story_warnings.append(f"Story plan failed ({e}); edit kept in recording order.")
        overlay_planner = None
        # ---- Stage 6: Compile + render ----
        if settings.enable_broll:           # B-roll over voice-over, planned on the compiled timeline (not in the MVP)
            from app.tasks.broll import plan_overlays
            broll_files = [{"filename": f["filename"], "path": file_map[f["filename"]],
                            "audio_only": bool(f.get("audio_only"))} for f in files_info]
            overlay_planner = lambda tl: plan_overlays(tl, word_grid, broll_files, moments_result)
            safe_broadcast("assembling", 85, "Choosing B-roll for the voice-over, then rendering...")
        else:
            safe_broadcast("assembling", 85, "Compiling and rendering the edit...")
        timeline, render_info = compile_and_render(plan, word_grid, envs, file_map, final_video_path,
                                                   overlay_planner=overlay_planner)
        job_warnings = [] if settings.enable_story_plan else [
            f"Story plan off: target duration ({target_duration}s) not applied (output {timeline.duration_sec:.1f}s)."]
        n_review = sum(1 for r in ranges if r["label"] == "remove" and r.get("review"))
        n_sug = sum(1 for r in ranges if r["label"] == "suggest")
        if n_review or n_sug:
            job_warnings.append(f"Edit passes: {n_review} take choice(s) to check, {n_sug} suggestion(s) "
                                "in the transcript editor.")
        if render_info.get("broll") is not None:
            job_warnings += render_info["broll"]["warnings"]
            artifacts_mod.save_job(job_id, "broll", render_info["broll"])
        vo_files = {f["filename"] for f in files_info if f.get("audio_only")}
        vo_sec = sum(s.src_out - s.src_in for s in timeline.segments if s.source_file in vo_files)
        if vo_sec and render_info.get("broll") is None:
            job_warnings.append(f"Voice-over: {vo_sec:.1f}s of the edit come from audio-only files and show a black "
                                "picture until B-roll is placed over them.")
        if render_info["validation"]["status"] != "pass":
            job_warnings.append(f"Validation {render_info['validation']['status']}: " + ", ".join(
                f"{c['check']} (segment {c['segment']})" for c in render_info["validation"]["checks"]
                if c["status"] != "pass"))
        if job_id in jobs_db:
            jobs_db[job_id].warnings.extend(job_warnings)
        for w in job_warnings:
            logger.warning(f"[WORD-GRID-COMPILER] {w}")
        # Persist what a later re-compile needs (survives a restart)
        save_job_edit(job_id, word_grid, [{"filename": f["filename"], "path": file_map[f["filename"]],
                                           "audio_path": f["audio_path"], "audio_only": bool(f.get("audio_only"))}
                                          for f in files_info],
                      plan, timeline, render_info, cleanup_ranges=ranges)
        job_story_note = story_warnings
        if story is not None:
            artifacts_mod.save_job(job_id, "moments", moments_result)
            artifacts_mod.save_job(job_id, "storyplan", story)
        if job_story_note and job_id in jobs_db:
            jobs_db[job_id].warnings.extend(job_story_note)
        for w_ in job_story_note:
            logger.warning(f"[STORYPLAN] {w_}")
        if settings.enable_moments and story is None:  # story plan off: story beats of the saved edit
            safe_broadcast("assembling", 95, "Mapping the story (moments)...")
            try:
                from app.tasks.moments import moments_for_job
                mo = moments_for_job(job_id)
                logger.info(f"[MOMENTS] job {job_id}: {len(mo['moments'])} moments stored")
            except Exception as e:      # data for later phases: never fails the job
                logger.warning(f"[MOMENTS] job {job_id}: moment table not built: {e}")
        from app.tasks.publish_copy import schedule_background
        schedule_background(job_id)             # export panel post copy, once the render settles

        # ---- Complete ----
        download_url = f"/api/jobs/{job_id}/download"

        if main_loop:
            asyncio.run_coroutine_threadsafe(broadcast_progress(
                job_id, "complete", 100,
                "Editing complete! Final video is ready.",
                download_url=download_url
            ), main_loop).result()
        
        interaction_logger.log_pipeline_completion(job_id, [s_.model_dump() for s_ in timeline.segments])

    except Exception as e:
        logger.error(f"Pipeline failed for job {job_id}: {e}", exc_info=True)
        if main_loop:
            if job_id in jobs_db and jobs_db[job_id].status == "cancelled":
                asyncio.run_coroutine_threadsafe(broadcast_progress(
                    job_id, "cancelled", 0, "Job cancelled by user."
                ), main_loop).result()
            else:
                asyncio.run_coroutine_threadsafe(broadcast_progress(
                    job_id, "failed", 0, f"Error: {str(e)}"
                ), main_loop).result()
    finally:
        heartbeat_stop.set()
        logger.info(f"--- ENDING PIPELINE JOB: {job_id} ---")
        logging.getLogger().removeHandler(job_handler)
        job_handler.close()

async def start_pipeline(job_id: str, video_paths: List[str], context_text: str, target_duration: float = 10.0):
    """Spawn the pipeline run in a background worker thread."""
    main_loop = asyncio.get_running_loop()
    asyncio.create_task(
        asyncio.to_thread(run_pipeline_sync, job_id, video_paths, context_text, target_duration, main_loop)
    )


def run_recompile_sync(job_id: str, plan: Dict, main_loop: asyncio.AbstractEventLoop = None):
    """Phase 5 re-compile only: stored grid + edited plan -> compile -> render (no ingest/ASR/cleanup)."""
    def broadcast(stage, progress, message, download_url=None):
        if main_loop:
            asyncio.run_coroutine_threadsafe(
                broadcast_progress(job_id, stage, progress, message, download_url), main_loop).result()
    try:
        broadcast("assembling", 85, "Re-compiling the edit...")
        final_video_path = os.path.join(settings.output_dir, f"{job_id}.mp4")
        timeline, info = recompile_job(job_id, EditPlan.model_validate(plan), final_video_path)
        if job_id in jobs_db and info["validation"]["status"] != "pass":
            jobs_db[job_id].warnings.append(f"Re-compile validation {info['validation']['status']}")
        broadcast("complete", 100, "Edit re-compiled.", download_url=f"/api/jobs/{job_id}/download")
    except Exception as e:
        logger.error(f"Re-compile failed for job {job_id}: {e}", exc_info=True)
        broadcast("failed", 0, f"Re-compile error: {e}")
        return
    try:                                   # Phases 8-9: moments + plan options follow the edited version
        from app.tasks.recompile import refresh_story
        refresh_story(job_id, EditPlan.model_validate(plan))
    except Exception as e:                 # the video is already done; story data is optional
        logger.warning(f"[RECOMPILE] job {job_id}: story refresh failed: {e}")
    from app.tasks.publish_copy import schedule_background
    schedule_background(job_id)            # post copy for the new render (after the refreshed moments)


async def start_recompile(job_id: str, plan: Dict):
    """Spawn the re-compile in a background worker thread."""
    main_loop = asyncio.get_running_loop()
    asyncio.create_task(asyncio.to_thread(run_recompile_sync, job_id, plan, main_loop))
