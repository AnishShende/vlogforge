import os
import hashlib
import logging
from typing import List, Dict
from app.utils.ffmpeg import (
    get_video_duration, extract_audio, extract_keyframe,
    get_video_info, transcode_to_cfr
)
from app.models import VideoFileInfo, EGTSegment, generate_clip_id
from app.tasks.scene_detect import detect_scenes
from app.config import settings

logger = logging.getLogger("VlogForge.Ingest")


def _compute_file_hash(filepath: str, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file for dedup/validation.

    Reads in chunks to avoid loading multi-GB video files into memory.
    """
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()




def ingest_video(video_path: str, job_dir: str) -> Dict:
    """Validate video, pre-transcode to CFR, extract audio WAV, detect scene
    boundaries using the two-pass cascade detector, and extract keyframes.

    Returns a dict with:
        - file_info: VideoFileInfo
        - segments: List[EGTSegment]  (partially populated — transcript/visual/quality filled later)
        - cfr_path: str
        - source_file_hash: str
    """
    logger.info(f"Ingesting video: {video_path}")

    filename = os.path.basename(video_path)
    size_bytes = os.path.getsize(video_path)

    # Compute file hash for provenance/dedup
    logger.info(f"Computing source file hash for {filename}...")
    source_file_hash = _compute_file_hash(video_path)

    # Define paths
    cfr_dir = os.path.join(job_dir, "cfr")
    audio_dir = os.path.join(job_dir, "audio")
    keyframes_dir = os.path.join(job_dir, "keyframes")
    os.makedirs(cfr_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(keyframes_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 0: CFR Pre-transcoding
    # Must run BEFORE PySceneDetect and Whisper to align audio/video timestamps.
    # -----------------------------------------------------------------------
    cfr_filename = f"{os.path.splitext(filename)[0]}_cfr.mp4"
    cfr_path = os.path.join(cfr_dir, cfr_filename)
    logger.info(f"Pre-transcoding to CFR 30fps: {video_path} -> {cfr_path}")
    cfr_success = transcode_to_cfr(video_path, cfr_path)

    # Use CFR output for all downstream processing; fall back to original on failure
    active_video_path = cfr_path if cfr_success else video_path
    if not cfr_success:
        logger.warning(
            f"CFR transcode failed for {filename}. "
            "Continuing with original (VFR audio drift may occur)."
        )

    # Duration is read from the CFR-normalised file for accuracy
    duration = get_video_duration(active_video_path)

    # -----------------------------------------------------------------------
    # Step 1: Extract Audio WAV (from CFR source)
    # -----------------------------------------------------------------------
    audio_filename = f"{os.path.splitext(filename)[0]}.wav"
    audio_path = os.path.join(audio_dir, audio_filename)
    logger.info(f"Extracting audio to: {audio_path}")
    audio_success = extract_audio(active_video_path, audio_path)
    actual_audio_path = audio_path if audio_success else None

    # -----------------------------------------------------------------------
    # Step 2: Detect Scene Boundaries (two-pass cascade)
    # ContentDetector for hard cuts, then AdaptiveDetector on long scenes.
    # Replaces the former single-detector + expand_long_scenes approach.
    # -----------------------------------------------------------------------
    logger.info(f"Detecting scenes for {filename}...")
    raw_scenes = detect_scenes(
        active_video_path,
        duration=duration,
        long_scene_threshold_sec=settings.long_scene_threshold_sec,
        content_threshold=settings.content_detector_threshold,
        adaptive_threshold=settings.adaptive_detector_threshold,
        min_scene_duration_sec=settings.min_scene_duration_sec,
    )

    # -----------------------------------------------------------------------
    # Step 3: Convert raw scene dicts to EGTSegment objects with keyframes
    # -----------------------------------------------------------------------
    logger.info(f"Creating EGT segments and extracting keyframes for {len(raw_scenes)} scenes...")
    segments: List[EGTSegment] = []

    for s in raw_scenes:
        start_sec = s["start"]
        end_sec = s["end"]
        detection_method = s.get("detection_method", "unknown")

        # Extract keyframe at midpoint
        midpoint = start_sec + (end_sec - start_sec) / 2.0
        kf_filename = f"{os.path.splitext(filename)[0]}_scene_{start_sec:.0f}.jpg"
        kf_path = os.path.join(keyframes_dir, kf_filename)
        kf_success = extract_keyframe(active_video_path, midpoint, kf_path)

        # Tag segment with detection method provenance
        scene_tags = [f"scene_{detection_method}"]

        segment = EGTSegment(
            clip_id=generate_clip_id(filename, start_sec, end_sec),
            source_file=filename,
            source_file_hash=source_file_hash,
            start_sec=start_sec,
            end_sec=end_sec,
            keyframe_path=kf_path if kf_success else None,
            tags=scene_tags,
        )
        segments.append(segment)

    logger.info(f"Final segment count for {filename}: {len(segments)}")

    return {
        "file_info": VideoFileInfo(
            filename=filename,
            original_path=video_path,
            duration=duration,
            size_bytes=size_bytes,
            audio_path=actual_audio_path
        ),
        "segments": segments,
        "cfr_path": active_video_path,
        "source_file_hash": source_file_hash,
    }
