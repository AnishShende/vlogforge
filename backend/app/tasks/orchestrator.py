"""Pipeline Orchestrator — Phase 0 Architecture.

8-stage pipeline mapped to the canonical 3-pass model:

    Pass 1 (Perception):
        Stage 1: Ingest (parallel per-file) → raw EGTSegments
        Stage 2: Transcribe (Gemini STT / Whisper) → aligned EGT segments
        Stage 3: Visual Analysis → EGT segments enriched with descriptions + tags
        Stage 4: Quality Scoring → EGT segments with quality_score, segment_type, is_bad_take
        Stage 5: EGT Assembly → validated EGTDocument written to job data store

    Pass 2 (Reasoning — stub in P0):
        Stage 6: EDL Generation → mechanical chronological filter

    Pass 3 (Assembly):
        Stage 7: Video Assembly → FFmpeg render from EDL
        Stage 8: Human Review → timeline editor, EDL mutations trigger Stage 7 re-run only
"""

import os
import shutil
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Set, Optional
from fastapi import WebSocket
from app.config import settings
from app.models import JobStatus, VideoFileInfo, WSProgressEvent, EGTSegment, EGTDocument
from app.tasks.ingest import ingest_video
from app.tasks.scene_detect import detect_scenes, subdivide_by_speech_gaps
from app.tasks.transcribe import transcribe_audio, align_transcript_with_segments
from app.tasks.analyze import analyze_segments
from app.tasks.score import score_segments, recompute_bad_takes
from app.tasks.egt import build_egt_document, egt_to_serializable
from app.tasks.edl import generate_edl
from app.tasks.assemble import assemble_vlog
from app.utils.interaction_logger import interaction_logger

logger = logging.getLogger("VlogForge.Orchestrator")

# Global in-memory job store
jobs_db: Dict[str, JobStatus] = {}
# Global active websocket connections: job_id -> set of WebSockets
websockets_db: Dict[str, Set[WebSocket]] = {}
# Global job data store (EGT, EDL, transcripts, etc. that are too large for status)
jobs_data_db: Dict[str, Dict] = {}

def get_job(job_id: str) -> Optional[JobStatus]:
    return jobs_db.get(job_id)

def get_job_data(job_id: str) -> Optional[Dict]:
    return jobs_data_db.get(job_id)

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
    # Update JobStatus
    if job_id in jobs_db:
        job = jobs_db[job_id]
        job.status = stage
        job.progress = progress
        job.message = message
        if stage == "complete" and download_url:
            job.output_video_url = download_url
            job.completed_at = datetime.utcnow()
        elif stage == "failed":
            job.completed_at = datetime.utcnow()

    # Broadcast
    websockets = websockets_db.get(job_id, set())
    if websockets:
        event = WSProgressEvent(
            stage=stage,
            progress=progress,
            message=message,
            download_url=download_url
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

def run_pipeline_sync(job_id: str, video_paths: List[str], context_text: str, target_duration: float = 10.0, vlog_genre: str = "default", quality_threshold: float = 0.35):
    """Synchronous pipeline run (to be run in a separate thread)."""


    # Create event loop for this thread to call async broadcast function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Lock to serialize WebSocket broadcasts from concurrent worker threads
    broadcast_lock = threading.Lock()

    def safe_broadcast(stage: str, progress: int, message: str, download_url=None):
        """Thread-safe WebSocket progress broadcast wrapper."""
        with broadcast_lock:
            loop.run_until_complete(
                broadcast_progress(job_id, stage, progress, message, download_url)
            )

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
    # during long Gemini rate-limit waits (which can block for 50-60s with no WS activity)
    heartbeat_stop = threading.Event()

    def heartbeat_worker():
        while not heartbeat_stop.is_set():
            heartbeat_stop.wait(timeout=10.0)
            if heartbeat_stop.is_set():
                break
            try:
                with broadcast_lock:
                    loop.run_until_complete(
                        broadcast_progress(job_id, "heartbeat", -1, "ping")
                    )
            except Exception:
                pass  # Ignore errors — websocket may have disconnected cleanly

    heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
    heartbeat_thread.start()

    def check_cancelled():
        if job_id in jobs_db and jobs_db[job_id].status == "cancelled":
            raise RuntimeError("Job cancelled by user.")


    try:
        # ==================================================================
        # PASS 1 — PERCEPTION (cheap, classification-shaped)
        # ==================================================================

        # ---- Stage 1: Ingest & Pre-processing (Parallel) ----
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
        all_segments: List[EGTSegment] = []
        files_info = []
        total_raw_duration = 0.0
        for idx in range(total_files):
            result = ingest_results[idx]
            file_info_dict = result["file_info"].model_dump()
            file_info_dict["cfr_path"] = result.get("cfr_path", video_paths[idx])
            files_info.append(file_info_dict)
            all_segments.extend(result["segments"])
            total_raw_duration += result["file_info"].duration

        check_cancelled()
        if job_id in jobs_db:
            jobs_db[job_id].files = [VideoFileInfo(**f) for f in files_info]

        # ---- Stage 2: Transcribing (Gemini STT / Whisper) ----
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
        # Align transcripts with EGT segments
        all_segments = align_transcript_with_segments(all_segments, full_transcript_segments)

        # ---- Stage 3a: Speech-gap Refinement (duration-relative) ----
        check_cancelled()
        safe_broadcast("refining", 40, "Refining long segments using speech gaps...")

        # Compute dynamic thresholds from target_duration
        dynamic_long_scene = max(
            settings.long_scene_floor_sec,
            target_duration * settings.long_scene_ratio,
        )
        dynamic_speech_gap = max(
            settings.speech_gap_floor_sec,
            target_duration * settings.speech_gap_ratio,
        )
        logger.info(
            f"Dynamic thresholds (target_duration={target_duration:.0f}s): "
            f"long_scene={dynamic_long_scene:.1f}s, speech_gap={dynamic_speech_gap:.1f}s"
        )

        # Build per-file CFR path lookup for keyframe extraction
        for f_info in files_info:
            cfr_path = f_info.get("cfr_path", "")
            filename = f_info.get("filename", "")
            keyframes_dir = os.path.join(job_dir, "keyframes")

            if cfr_path and os.path.exists(cfr_path):
                all_segments = subdivide_by_speech_gaps(
                    segments=all_segments,
                    transcript_segments=full_transcript_segments,
                    long_scene_threshold_sec=dynamic_long_scene,
                    speech_gap_sec=dynamic_speech_gap,
                    video_path=cfr_path,
                    keyframes_dir=keyframes_dir,
                    context_notes=context_text,
                )

        # ---- Stage 3b: Visual Analysis (Keyframe Description + Tags) ----
        check_cancelled()
        safe_broadcast("analyzing", 48, "Running visual keyframe analysis...")

        analysis_result = analyze_segments(
            segments=all_segments, 
            user_context=context_text,
            job_dir=job_dir,
            files_info=files_info
        )
        all_segments = analysis_result["segments"]
        context_summary = analysis_result["context_summary"]

        # ---- Stage 4: Quality Scoring ----
        check_cancelled()
        safe_broadcast("classifying", 55, "Classifying segments...")

        def scoring_progress(done: int, total: int):
            pct = 55 + int(9 * done / total)  # 55% -> 64%
            safe_broadcast("classifying", pct, f"Classifying segment {done}/{total}...")

        all_segments = score_segments(
            all_segments, total_raw_duration, context_summary, quality_threshold,
            progress_callback=scoring_progress
        )

        # ---- Stage 5: EGT Assembly & Validation ----
        check_cancelled()
        safe_broadcast("classifying", 64, "Building Editorial Ground Truth document...")

        egt_doc = build_egt_document(
            segments=all_segments,
            context_summary=context_summary,
            source_file_count=total_files,
            total_duration=total_raw_duration,
        )

        # Store EGT and transcript data
        jobs_data_db[job_id] = {
            "egt": egt_to_serializable(egt_doc),
            "transcript": [seg.model_dump() for seg in all_segments],
            "context_document": context_summary,
        }

        # ==================================================================
        # PASS 2 — REASONING (stub in Phase 0: mechanical filter only)
        # ==================================================================

        # ---- Stage 6: EDL Generation ----
        check_cancelled()
        safe_broadcast("edl_generating", 75, "Generating Edit Decision List (chronological filter)...")

        edl, warning = generate_edl(egt_doc, full_transcript_segments, target_duration, context_text)
        jobs_data_db[job_id]["edl"] = edl
        if warning:
            if "warnings" not in jobs_data_db[job_id]:
                jobs_data_db[job_id]["warnings"] = []
            jobs_data_db[job_id]["warnings"].append(warning)
            # You could add safe_broadcast here if a "warning" event type existed on frontend

        # Build set of valid clip_ids for assembly validation
        egt_clip_ids = {seg.clip_id for seg in all_segments}

        # ==================================================================
        # PASS 3 — MECHANICAL ASSEMBLY
        # ==================================================================

        # ---- Stage 7: Video Assembly (FFmpeg) ----
        check_cancelled()
        safe_broadcast("assembling", 85, "Assembling video cuts with FFmpeg...")

        final_video_name = f"{job_id}.mp4"
        final_video_path = os.path.join(settings.output_dir, final_video_name)

        assembly_success = assemble_vlog(
            edl, files_info, job_dir, final_video_path,
            egt_clip_ids=egt_clip_ids
        )
        if not assembly_success:
            raise RuntimeError("FFmpeg assembly pipeline failed.")

        # Clean up CFR temp files after successful assembly to save disk space
        cfr_dir = os.path.join(job_dir, "cfr")
        if os.path.isdir(cfr_dir):
            try:
                shutil.rmtree(cfr_dir)
                logger.info(f"CFR temp directory cleaned up: {cfr_dir}")
            except Exception as cleanup_err:
                logger.warning(f"Failed to clean up CFR directory {cfr_dir}: {cleanup_err}")

        # ---- Stage 8: Complete (Human Review happens in the frontend) ----
        download_url = f"/api/jobs/{job_id}/download"
        loop.run_until_complete(broadcast_progress(
                job_id, "complete", 100,
                "Vlog creation successful! Your video is ready for review.",
                download_url=download_url
            ))
        logger.info(f"Pipeline completed successfully for job: {job_id}")
        interaction_logger.log_pipeline_completion(
            job_id,
            jobs_data_db.get(job_id, {}).get("egt", {}),
            jobs_data_db.get(job_id, {}).get("edl", [])
        )

    except Exception as e:
        logger.error(f"Pipeline failed for job {job_id}: {e}", exc_info=True)
        if job_id in jobs_db and jobs_db[job_id].status == "cancelled":
            loop.run_until_complete(broadcast_progress(
                job_id, "cancelled", 0, "Job cancelled by user."
            ))
        else:
            loop.run_until_complete(broadcast_progress(
                job_id, "failed", 0, f"Error: {str(e)}"
            ))
    finally:
        heartbeat_stop.set()
        logger.info(f"--- ENDING PIPELINE JOB: {job_id} ---")
        logging.getLogger().removeHandler(job_handler)
        job_handler.close()
        loop.close()

async def start_pipeline(job_id: str, video_paths: List[str], context_text: str, target_duration: float = 10.0, vlog_genre: str = "default", quality_threshold: float = 0.35):
    """Spawn the pipeline run in a background worker thread."""
    asyncio.create_task(
        asyncio.to_thread(run_pipeline_sync, job_id, video_paths, context_text, target_duration, vlog_genre, quality_threshold)
    )

def run_re_reasoning_sync(job_id: str, quality_threshold: float):
    """Re-run just the Reasoning (Pass 2) and Assembly (Pass 3) after threshold change."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    broadcast_lock = threading.Lock()

    def safe_broadcast(stage: str, progress: int, message: str, download_url=None):
        with broadcast_lock:
            loop.run_until_complete(broadcast_progress(job_id, stage, progress, message, download_url))

    try:
        job = get_job(job_id)
        job_data = get_job_data(job_id)
        if not job or not job_data or "egt" not in job_data:
            raise RuntimeError("Missing EGT data for re-reasoning.")

        job_dir = os.path.join(settings.upload_dir, job_id)
        
        safe_broadcast("classifying", 70, "Applying new quality threshold...")
        
        # 1. Update is_bad_take on EGT Document directly
        egt_doc_dict = job_data["egt"]
        # re-hydrate EGTDocument
        egt_doc = EGTDocument(**egt_doc_dict)
        egt_doc.segments = recompute_bad_takes(egt_doc.segments, quality_threshold)
        
        # Save back EGT
        job_data["egt"] = egt_to_serializable(egt_doc)

        safe_broadcast("edl_generating", 75, "Re-generating Edit Decision List (AI Reasoning)...")

        # 2. Re-run EDL generation
        edl, warning = generate_edl(egt_doc, [], target_duration=None, user_prompt="")
        job_data["edl"] = edl
        if warning:
            if "warnings" not in job_data:
                job_data["warnings"] = []
            job_data["warnings"].append(warning)

        safe_broadcast("assembling", 85, "Assembling video cuts with FFmpeg...")

        # 3. Assemble
        final_video_name = f"{job_id}.mp4"
        final_video_path = os.path.join(settings.output_dir, final_video_name)
        egt_clip_ids = {seg.clip_id for seg in egt_doc.segments}
        files_info = [f.model_dump() for f in job.files]

        assembly_success = assemble_vlog(
            edl, files_info, job_dir, final_video_path,
            egt_clip_ids=egt_clip_ids
        )
        if not assembly_success:
            raise RuntimeError("FFmpeg assembly pipeline failed.")

        download_url = f"/api/jobs/{job_id}/download"
        loop.run_until_complete(broadcast_progress(
            job_id, "complete", 100,
            "Re-reasoning successful! Vlog updated.",
            download_url=download_url
        ))
        logger.info(f"Re-reasoning completed successfully for job: {job_id}")
        interaction_logger.log_pipeline_completion(
            job_id,
            job_data.get("egt", {}),
            job_data.get("edl", [])
        )

    except Exception as e:
        logger.error(f"Re-reasoning failed for job {job_id}: {e}", exc_info=True)
        loop.run_until_complete(broadcast_progress(job_id, "failed", 0, f"Error: {str(e)}"))
    finally:
        loop.close()

async def start_re_reasoning(job_id: str, quality_threshold: float):
    """Spawn the re-reasoning in a background worker thread."""
    asyncio.create_task(
        asyncio.to_thread(run_re_reasoning_sync, job_id, quality_threshold)
    )

