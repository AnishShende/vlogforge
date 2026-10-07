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
from sqlalchemy.future import select
from app.config import settings
from app.database import AsyncSessionLocal
from app.db_models import Project
from app.models import JobStatus, VideoFileInfo, WSProgressEvent, EGTSegment, EGTDocument
from app.tasks.ingest import ingest_video
from app.tasks.scene_detect import detect_scenes, subdivide_by_speech_gaps, editorial_subdivide
from app.tasks.transcribe import transcribe_audio, align_transcript_with_segments
from app.tasks.word_grid import build_word_grid, check_word_grid, summarize_word_grid
from app.utils.speech_activity import envelope, load_audio
from app.tasks.analyze import analyze_segments
from app.tasks.score import score_segments, recompute_bad_takes
from app.tasks.egt import build_egt_document, egt_to_serializable
from app.tasks.retake_detect import detect_and_resolve_retakes
from app.tasks.word_timeline_redundancy import apply_word_timeline_selection, clamp_edl_to_word_timeline
from app.tasks.edl import generate_edl
from app.tasks.assemble import assemble_vlog
from app.tasks.compiler import compile_and_render
from app.tasks.speech_cleanup import REVIEW, cleanup_plan, label_grid, label_ranges
from app.tasks.edit_passes import edit_with_passes
from app.tasks.recompile import recompile_job, save_job_edit
from app.models import EditPlan
from app.tasks.metadata import generate_metadata
from app.utils.interaction_logger import interaction_logger
from app.utils.word_snap import snap_edl_to_word_boundaries

logger = logging.getLogger("VlogForge.Orchestrator")

# Global in-memory job store
jobs_db: Dict[str, JobStatus] = {}
# Global active websocket connections: job_id -> set of WebSockets
websockets_db: Dict[str, Set[WebSocket]] = {}
# Global job data store (EGT, EDL, transcripts, etc. that are too large for status)
jobs_data_db: Dict[str, Dict] = {}

def get_job(job_id: str) -> Optional[JobStatus]:
    job = jobs_db.get(job_id)
    if job and getattr(settings, "enable_mock_llm", False):
        from app.utils.llm import job_llm_stats
        stats = job_llm_stats.get(job_id, {"real": 0, "mocked": 0})
        r, m = stats["real"], stats["mocked"]
        if r > 0 and m > 0:
            job.llm_mode = f"partial (Real: {r}, Mocked: {m})"
        elif r > 0:
            job.llm_mode = f"real ({r} calls)"
        elif m > 0:
            job.llm_mode = f"mocked ({m} calls)"
        else:
            job.llm_mode = "mocked (waiting)"
    return job

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

def run_pipeline_sync(job_id: str, video_paths: List[str], context_text: str, target_duration: float = 10.0, vlog_genre: str = "default", quality_threshold: float = 0.35, main_loop: asyncio.AbstractEventLoop = None):
    """Synchronous pipeline run (to be run in a separate thread)."""
    from app.utils.llm import current_job_id
    current_job_id.set(job_id)
    if settings.enable_word_grid_compiler and not settings.enable_word_grid:
        raise RuntimeError("enable_word_grid_compiler requires enable_word_grid (the cleanup edits the word grid)")

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
    # during long Gemini rate-limit waits (which can block for 50-60s with no WS activity)
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
        # ---- Stage 2.5: Word grid (Archdoc Phase 1, flag-gated; no effect on output yet) ----
        word_grid = None
        if settings.enable_word_grid:
            word_grid = build_word_grid(full_transcript_segments, asr={
                "transcriber": "faster-whisper turbo",
                "aligner": "whisperx wav2vec2" if settings.enable_forced_alignment else None,
            })
            envs = {f["filename"]: envelope(load_audio(f["audio_path"]))
                    for f in files_info if f.get("audio_path") and os.path.exists(f["audio_path"])}
            missing = {w.source_file for w in word_grid.words} - set(envs)
            if missing:
                logger.warning(f"[WORD-GRID] no analysis audio for {sorted(missing)}: speech checks skipped")
            else:
                word_grid = check_word_grid(word_grid, envs)
            logger.info(f"[WORD-GRID] {summarize_word_grid(word_grid)}")

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
        dynamic_speech_gap = settings.speech_gap_floor_sec
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

        # ---- Stage 3a.5: Editorial subdivision for LLM Director ----
        check_cancelled()
        safe_broadcast("refining", 44, "Creating editorial sub-segments...")

        all_segments = editorial_subdivide(
            segments=all_segments,
            transcript_segments=full_transcript_segments,
            target_duration=target_duration,
            files_info=files_info,
            job_dir=job_dir,
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

        # M4: Classify using batch path. Progress fires per-batch, so for 1000 segments
        # at batch_size=10 this is 100 callbacks instead of 1000 — much less lock contention.
        total_segs = len(all_segments)
        batch_size = settings.classification_batch_size
        total_batches = max(1, (total_segs + batch_size - 1) // batch_size)

        def scoring_progress(done: int, total: int):
            # `done` and `total` here are segment counts (batch func fires per segment internally)
            # We remap to 55–65% range, scaled dynamically on segment count.
            pct = 55 + int(10 * done / max(1, total))
            batch_num = max(1, (done + batch_size - 1) // batch_size)
            safe_broadcast(
                "classifying", pct,
                f"Classifying segments: batch {batch_num}/{total_batches} ({done}/{total} segments)..."
            )

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
        
        # ---- Stage 5.5: Retake Detection & Resolution ----
        check_cancelled()
        safe_broadcast("classifying", 65, "Detecting and clustering retakes...")
        egt_doc = detect_and_resolve_retakes(egt_doc)

        # ---- Stage 5.6: Word-timeline clean-take selection (flag-gated) ----
        # Replaces the SPEECH portion of the EGT with punctuation-free,
        # JEV-judged clean takes (one per distinct line), keeping non-speech /
        # B-roll for the legacy path. Downstream EDL/word-snap/assembly consume
        # the synthetic clip_ids unchanged.
        # Skipped when the word-grid compiler makes the edit: its result would be unused.
        if settings.enable_word_timeline_redundancy and not settings.enable_word_grid_compiler:
            check_cancelled()
            safe_broadcast("classifying", 66, "Selecting clean takes from the word timeline...")
            egt_doc, wt_warnings = apply_word_timeline_selection(egt_doc)
            all_segments = egt_doc.segments  # keep segments_by_clip_id / egt_clip_ids consistent
            if wt_warnings and job_id in jobs_db:
                jobs_db[job_id].warnings.extend(wt_warnings)

        # Store EGT and transcript data
        jobs_data_db[job_id] = {
            "egt": egt_to_serializable(egt_doc),
            "transcript": [seg.model_dump() for seg in all_segments],
            "context_document": context_summary,
        }
        if word_grid is not None:
            jobs_data_db[job_id]["word_grid"] = word_grid.model_dump()

        # ==================================================================
        # Archdoc Phase 4 path (flag-gated): grid cleanup -> compiler -> render.
        # Replaces EDL + assembly; the earlier stages still run for metadata/storage.
        # ==================================================================
        if settings.enable_word_grid_compiler:
            check_cancelled()
            safe_broadcast("edl_generating", 70, "Cleaning up speech on the word grid (retakes, dead air)...")
            if word_grid is None or not {w.source_file for w in word_grid.words} <= set(envs):
                raise RuntimeError("word-grid compiler: word grid or analysis audio missing; cannot compile")
            final_video_name = f"{job_id}.mp4"
            final_video_path = os.path.join(settings.output_dir, final_video_name)
            file_map = {f["filename"]: f.get("original_path") or f.get("cfr_path") for f in files_info}
            if settings.enable_edit_passes:     # Phase 6.5: LLM passes (retakes, incomplete, review, suggestions)
                safe_broadcast("edl_generating", 72, "Editing speech: retakes, false starts, final review...")
                plan, ranges, _pass_stats = edit_with_passes(word_grid)
            else:
                labels = label_grid(word_grid)
                plan = cleanup_plan(word_grid, labels)
                ranges = label_ranges(word_grid, labels)
            safe_broadcast("assembling", 85, "Compiling and rendering the edit...")
            timeline, render_info = compile_and_render(plan, word_grid, envs, file_map, final_video_path)
            job_warnings = [
                "Word-grid edit: speech only; B-roll and target duration "
                f"({target_duration}s) are not applied yet (output {timeline.duration_sec:.1f}s).",
            ]
            if settings.enable_edit_passes:
                n_review = sum(1 for r in ranges if r["label"] == "remove" and r.get("review"))
                n_sug = sum(1 for r in ranges if r["label"] == "suggest")
                if n_review or n_sug:
                    job_warnings.append(f"Edit passes: {n_review} take choice(s) to check, {n_sug} suggestion(s) "
                                        "in the transcript editor.")
            else:
                review = [r for r in ranges if r["label"] == REVIEW]
                job_warnings += [f"Review take at {r['start']:.1f}-{r['end']:.1f}s ({len(r['by']['alternatives'])} other "
                                 f"clean take(s)): \"{r['text'][:60]}\"" for r in review]
            if render_info["validation"]["status"] != "pass":
                job_warnings.append(f"Validation {render_info['validation']['status']}: " + ", ".join(
                    f"{c['check']} (segment {c['segment']})" for c in render_info["validation"]["checks"]
                    if c["status"] != "pass"))
            if job_id in jobs_db:
                jobs_db[job_id].warnings.extend(job_warnings)
            for w in job_warnings:
                logger.warning(f"[WORD-GRID-COMPILER] {w}")
            # EDL-shaped view of the timeline for metadata / storage consumers
            edl = [{"clip_id": "", "source_file": s.source_file, "start_sec": s.src_in, "end_sec": s.src_out,
                    "editorial_type": "KEEP", "sequence_index": k} for k, s in enumerate(timeline.segments)]
            jobs_data_db[job_id].update({"edl": edl, "reasoning_mode": "word_grid_compiler",
                                         "cleanup": ranges, "timeline": timeline.model_dump(),
                                         "validation": render_info["validation"]})
            # Phase 5: persist what a later re-compile needs (survives a restart)
            save_job_edit(job_id, word_grid, [{"filename": f["filename"], "path": file_map[f["filename"]],
                                               "audio_path": f["audio_path"]} for f in files_info],
                          plan, timeline, render_info, cleanup_ranges=ranges)
        else:
            # ---- Legacy path: EDL reasoning + assembly ----

            # ---- Stage 6: EDL Generation ----
            check_cancelled()
            total_segs_for_edl = len(egt_doc.segments)
            using_map_reduce = total_segs_for_edl > settings.edl_chunk_threshold
            edl_stage_msg = (
                f"Generating Edit Decision List (Map-Reduce: {total_segs_for_edl} segments, "
                f"{max(1, (total_segs_for_edl + settings.edl_chunk_size - 1) // settings.edl_chunk_size)} chunks)..."
                if using_map_reduce
                else "Generating Edit Decision List (AI Reasoning)..."
            )
            safe_broadcast("edl_generating", 65, edl_stage_msg)

            edl, _warning, reasoning_mode = generate_edl(egt_doc, full_transcript_segments, target_duration, context_text)

            # ---- Stage 6.5: Word-boundary snap post-pass ----
            check_cancelled()
            safe_broadcast("edl_generating", 75, "Snapping cut points to word boundaries...")
            segments_by_clip_id = {
                seg.clip_id: seg.model_dump() for seg in all_segments
            }
            edl = snap_edl_to_word_boundaries(edl, segments_by_clip_id)

            # When word-timeline selection is on, its spans ARE the cut decision.
            # Restore their exact bounds so the LLM reasoner / word-snap cannot move
            # the cut points (which otherwise slices into real speech mid-sentence).
            if settings.enable_word_timeline_redundancy:
                clamped = clamp_edl_to_word_timeline(edl, egt_doc)
                logger.info(f"Word-timeline: clamped {clamped} EDL entries to clean span bounds")

            jobs_data_db[job_id]["edl"] = edl
            jobs_data_db[job_id]["reasoning_mode"] = reasoning_mode

            # Build set of valid clip_ids for assembly validation (exclude bad/superseded/stutter)
            egt_clip_ids = {
                seg.clip_id for seg in all_segments
                if not seg.is_bad_take and not getattr(seg, "is_superseded_take", False) and not getattr(seg, "is_stutter_repeat", False)
            }

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

        # ---- Stage 7.5: Metadata Generation (M5) ----
        check_cancelled()
        safe_broadcast("metadata_generating", 92, "Generating YouTube metadata (title, description, tags, chapters)...")

        try:
            metadata = generate_metadata(
                context_summary=context_summary,
                transcript_segments=full_transcript_segments,
                edl=edl,
                target_duration=target_duration,
                user_prompt=context_text,
            )
            jobs_data_db[job_id]["metadata"] = metadata
            if job_id in jobs_db:
                from app.models import VideoMetadata
                jobs_db[job_id].metadata = VideoMetadata(**metadata)
            logger.info(f"M5 metadata generated for job {job_id}.")
        except Exception as meta_err:
            logger.warning(f"Metadata generation failed (non-fatal): {meta_err}")
            # Non-fatal — video is still ready, just without metadata

        # ---- Stage 8: Complete ----
        download_url = f"/api/jobs/{job_id}/download"

        # M4: Record pipeline metrics for diagnostic reporting
        if job_id in jobs_db:
            chunks_used = max(1, (len(all_segments) + settings.edl_chunk_size - 1) // settings.edl_chunk_size)
            jobs_db[job_id].pipeline_metrics = {
                "total_segments": len(all_segments),
                "raw_footage_sec": round(total_raw_duration, 1),
                "edl_entries": len(edl),
                "chunks_used": chunks_used if len(all_segments) > settings.edl_chunk_threshold else 1,
                "map_reduce_active": len(all_segments) > settings.edl_chunk_threshold,
            }
            logger.info(
                f"Pipeline metrics for job {job_id}: "
                f"segments={len(all_segments)}, raw_footage={total_raw_duration:.0f}s, "
                f"edl_entries={len(edl)}, map_reduce={'YES' if len(all_segments) > settings.edl_chunk_threshold else 'NO'}"
            )

        if main_loop:
            asyncio.run_coroutine_threadsafe(broadcast_progress(
                job_id, "complete", 100,
                "Editing complete! Final video is ready.",
                download_url=download_url
            ), main_loop).result()
        
        interaction_logger.log_pipeline_completion(
            job_id,
            jobs_data_db.get(job_id, {}).get("egt", {}),
            jobs_data_db.get(job_id, {}).get("edl", [])
        )

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

async def start_pipeline(job_id: str, video_paths: List[str], context_text: str, target_duration: float = 10.0, vlog_genre: str = "default", quality_threshold: float = 0.35):
    """Spawn the pipeline run in a background worker thread."""
    main_loop = asyncio.get_running_loop()
    asyncio.create_task(
        asyncio.to_thread(run_pipeline_sync, job_id, video_paths, context_text, target_duration, vlog_genre, quality_threshold, main_loop)
    )

def run_re_reasoning_sync(job_id: str, quality_threshold: float, main_loop: asyncio.AbstractEventLoop = None):
    """Re-run just the Reasoning (Pass 2) and Assembly (Pass 3) after threshold change."""
    broadcast_lock = threading.Lock()

    def safe_broadcast(stage: str, progress: int, message: str, download_url=None):
        with broadcast_lock:
            if main_loop:
                asyncio.run_coroutine_threadsafe(broadcast_progress(job_id, stage, progress, message, download_url), main_loop).result()

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
        # Pass through the original job's target_duration and context_text
        job_target_duration = job.target_duration if job else None
        job_context_text = job.context_text if job else ""
        edl, _warning, reasoning_mode = generate_edl(egt_doc, [], target_duration=job_target_duration, user_prompt=job_context_text)
        job_data["edl"] = edl
        job_data["reasoning_mode"] = reasoning_mode

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
        if main_loop:
            asyncio.run_coroutine_threadsafe(broadcast_progress(
                job_id, "complete", 100,
                "Re-reasoning successful! Vlog updated.",
                download_url=download_url
            ), main_loop).result()
        logger.info(f"Re-reasoning completed successfully for job: {job_id}")
        interaction_logger.log_pipeline_completion(
            job_id,
            job_data.get("egt", {}),
            job_data.get("edl", [])
        )

    except Exception as e:
        logger.error(f"Re-reasoning failed for job {job_id}: {e}", exc_info=True)
        if main_loop:
            asyncio.run_coroutine_threadsafe(broadcast_progress(job_id, "failed", 0, f"Error: {str(e)}"), main_loop).result()
    finally:
        pass

async def start_re_reasoning(job_id: str, quality_threshold: float):
    """Spawn the re-reasoning in a background worker thread."""
    main_loop = asyncio.get_running_loop()
    asyncio.create_task(
        asyncio.to_thread(run_re_reasoning_sync, job_id, quality_threshold, main_loop)
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
        if job_id in jobs_data_db:
            jobs_data_db[job_id].update({"timeline": timeline.model_dump(), "validation": info["validation"]})
        if job_id in jobs_db and info["validation"]["status"] != "pass":
            jobs_db[job_id].warnings.append(f"Re-compile validation {info['validation']['status']}")
        broadcast("complete", 100, "Edit re-compiled.", download_url=f"/api/jobs/{job_id}/download")
    except Exception as e:
        logger.error(f"Re-compile failed for job {job_id}: {e}", exc_info=True)
        broadcast("failed", 0, f"Re-compile error: {e}")


async def start_recompile(job_id: str, plan: Dict):
    """Spawn the re-compile in a background worker thread."""
    main_loop = asyncio.get_running_loop()
    asyncio.create_task(asyncio.to_thread(run_recompile_sync, job_id, plan, main_loop))
