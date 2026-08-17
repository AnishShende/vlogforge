"""Pass 3 — Mechanical Assembly (EDL-first rendering).

The rendered video is generated strictly as an execution of the EDL — never an
independently generated creative path. This prevents EDL/video drift; the EDL
is the single source of truth.

Phase 0 changes:
- Input is List[Dict] (serialized EDLEntry) with clip_id grounding
- clip_id validation against EGT before assembly (anti-hallucination guardrail)
- Assembly logic unchanged: single-pass FFmpeg filtergraph with multi-pass fallback
"""

import os
import shutil
import logging
from typing import List, Dict, Optional, Set
import concurrent.futures

from app.utils.ffmpeg import (
    process_clip,
    concatenate_clips_with_crossfade,
    apply_fade_effects,
    assemble_single_pass,
)

logger = logging.getLogger("VlogForge.Assemble")


def validate_edl_against_egt(
    edl: List[Dict],
    egt_clip_ids: Set[str],
) -> List[str]:
    """Validate that every EDL entry references a real clip_id from the EGT.

    Anti-hallucination guardrail: if any clip_id doesn't resolve, return error messages.
    """
    errors = []
    for idx, entry in enumerate(edl):
        clip_id = entry.get("clip_id", "")
        if clip_id and clip_id not in egt_clip_ids:
            errors.append(
                f"EDL entry {idx} references clip_id '{clip_id}' "
                f"which does not exist in the EGT"
            )
    return errors


def assemble_vlog(
    edl: List[Dict],
    files_info: List[Dict],
    job_dir: str,
    final_output_path: str,
    egt_clip_ids: Optional[Set[str]] = None,
) -> bool:
    """Execute the Edit Decision List (EDL) using FFmpeg to assemble the final video.

    Single-pass filtergraph (primary path):
    Attempts to assemble the final video in a single FFmpeg pass using assemble_single_pass(),
    which trims, scales, normalises audio (dynaudnorm), crossfades, and fades in one execution.

    Falls back to the multi-pass pipeline (slice -> crossfade concat -> fade) if the single-pass
    filtergraph fails.

    Args:
        edl: List of EDLEntry dicts (from edl.py or from human-modified re-render).
        files_info: List of VideoFileInfo dicts with filename, cfr_path, original_path.
        job_dir: Job working directory for temp files.
        final_output_path: Destination path for the final video.
        egt_clip_ids: Set of valid clip_ids from EGT (for validation). If None, skip validation.

    Returns:
        True on success, False on failure.
    """
    logger.info("Starting video assembly...")

    if not edl:
        logger.error("Cannot assemble: EDL is empty.")
        return False

    # --- Anti-hallucination guardrail: validate clip_ids ---
    if egt_clip_ids is not None:
        validation_errors = validate_edl_against_egt(edl, egt_clip_ids)
        if validation_errors:
            for err in validation_errors:
                logger.error(f"Assembly blocked: {err}")
            logger.error(
                f"Assembly aborted: {len(validation_errors)} clip_id(s) "
                "failed EGT validation."
            )
            return False

    # Build file_map: filename -> original_path (preferred for assembly) or proxy_path
    file_map = {}
    for f in files_info:
        name = f["filename"]
        file_map[name] = f.get("original_path") or f.get("cfr_path")

    # Normalize EDL to the format expected by FFmpeg utils
    # The new EDLEntry uses 'source_file', legacy uses 'video_file'
    normalized_edl = []
    for entry in edl:
        normalized = {
            "video_file": entry.get("source_file") or entry.get("video_file", ""),
            "start_sec": entry["start_sec"],
            "end_sec": entry["end_sec"],
            "type": entry.get("editorial_type") or entry.get("type", "KEEP"),
        }
        normalized_edl.append(normalized)

    # -----------------------------------------------------------------------
    # Primary Path: Single-pass filtergraph
    # -----------------------------------------------------------------------
    if len(normalized_edl) <= 40:
        logger.info("Attempting single-pass filtergraph assembly...")
        success = assemble_single_pass(normalized_edl, file_map, final_output_path)
        if success:
            logger.info("Single-pass assembly succeeded.")
            return True

        logger.warning(
            "Single-pass assembly failed. Falling back to multi-pass pipeline "
            "(process_clip -> crossfade concat -> fade)."
        )
    else:
        logger.info(
            f"EDL contains {len(normalized_edl)} clips (>{40}). "
            "Bypassing single-pass filtergraph to prevent resource exhaustion. "
            "Routing directly to parallel multi-pass pipeline."
        )

    # -----------------------------------------------------------------------
    # Fallback: Multi-pass pipeline
    # -----------------------------------------------------------------------
    clips_dir = os.path.join(job_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    processed_clip_paths = []

    def _process_single_clip(args):
        idx, item = args
        video_filename = item["video_file"]
        raw_video_path = file_map.get(video_filename)

        if not raw_video_path or not os.path.exists(raw_video_path):
            logger.error(f"Source video {video_filename} not found at {raw_video_path}.")
            return None

        start = item["start_sec"]
        end = item["end_sec"]
        clip_name = f"clip_{idx:03d}_{video_filename}"
        clip_path = os.path.join(clips_dir, clip_name)

        logger.info(f"Processing clip {idx+1}/{len(normalized_edl)}: {clip_name} ({start:.2f}s to {end:.2f}s)...")
        if not process_clip(raw_video_path, start, end, clip_path):
            logger.error(f"Failed to slice clip {clip_name}.")
            return None

        return clip_path

    # Process clips in parallel using ThreadPoolExecutor
    max_workers = 4  # Safe limit for concurrent hardware encodes
    logger.info(f"Rendering {len(normalized_edl)} clips in parallel (max_workers={max_workers})...")
    
    tasks = list(enumerate(normalized_edl))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_process_single_clip, tasks))
    
    # Filter out failures and keep order
    for res in results:
        if res:
            processed_clip_paths.append(res)
        else:
            return False  # Abort if any clip failed

    if not processed_clip_paths:
        logger.error("No clips were successfully processed.")
        return False

    raw_concat_path = os.path.join(job_dir, "raw_concat.mp4")
    logger.info(f"Concatenating {len(processed_clip_paths)} clips with audio crossfade...")
    if not concatenate_clips_with_crossfade(processed_clip_paths, raw_concat_path):
        logger.error("Failed to concatenate clips.")
        return False

    logger.info(f"Applying fade effects to final cut -> {final_output_path}...")
    if not apply_fade_effects(raw_concat_path, final_output_path):
        logger.warning("Failed to apply fade effects. Using raw concat as final output.")
        shutil.copy(raw_concat_path, final_output_path)

    logger.info("Vlog assembly completed (multi-pass fallback).")
    return True
