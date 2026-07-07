import os
import hashlib
import logging
from typing import List, Dict
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from app.utils.ffmpeg import (
    get_video_duration, extract_audio, extract_keyframe,
    get_video_info, transcode_to_cfr
)
from app.models import VideoFileInfo, EGTSegment, generate_clip_id

logger = logging.getLogger("VlogForge.Ingest")

# Thresholds for spatio-temporal hybrid ingestion
LONG_SCENE_THRESHOLD_SEC = 15.0      # Scenes longer than this trigger sub-sampling
TEMPORAL_SAMPLE_INTERVAL_SEC = 12.0  # Interval between forced keyframes in long scenes


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


def expand_long_scenes(
    segments: List[EGTSegment],
    video_path: str,
    keyframes_dir: str,
    filename: str,
    source_file_hash: str,
) -> List[EGTSegment]:
    """For any segment longer than LONG_SCENE_THRESHOLD_SEC, subdivide it into
    temporal sub-intervals of TEMPORAL_SAMPLE_INTERVAL_SEC and extract a keyframe
    at each sub-interval midpoint. Short segments pass through unchanged.

    Returns a flat list of EGTSegment objects, where each long segment is replaced
    by its sub-interval entries with unique clip_ids.
    """
    expanded = []
    for seg in segments:
        if seg.duration_sec <= LONG_SCENE_THRESHOLD_SEC:
            expanded.append(seg)
            continue

        # Long segment: subdivide into temporal sub-intervals
        logger.info(
            f"Long scene detected ({seg.duration_sec:.1f}s) in {filename} "
            f"[{seg.start_sec:.1f}s - {seg.end_sec:.1f}s]. "
            f"Expanding with {TEMPORAL_SAMPLE_INTERVAL_SEC}s temporal sampling."
        )
        sub_start = seg.start_sec
        while sub_start < seg.end_sec:
            sub_end = min(sub_start + TEMPORAL_SAMPLE_INTERVAL_SEC, seg.end_sec)
            if sub_end - sub_start < 1.0:
                break

            # Extract keyframe at midpoint of this sub-interval
            kf_time = sub_start + (sub_end - sub_start) / 2.0
            kf_filename = (
                f"{os.path.splitext(filename)[0]}"
                f"_scene_t{sub_start:.0f}.jpg"
            )
            kf_path = os.path.join(keyframes_dir, kf_filename)
            kf_success = extract_keyframe(video_path, kf_time, kf_path)

            sub_segment = EGTSegment(
                clip_id=generate_clip_id(filename, sub_start, sub_end),
                source_file=filename,
                source_file_hash=source_file_hash,
                start_sec=sub_start,
                end_sec=sub_end,
                keyframe_path=kf_path if kf_success else None,
            )
            expanded.append(sub_segment)
            sub_start = sub_end

    return expanded


def ingest_video(video_path: str, job_dir: str) -> Dict:
    """Validate video, pre-transcode to CFR, extract audio WAV, find scene cuts,
    expand long scenes spatiotemporally, and extract keyframes.

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
    # Step 2: Detect Scene Cuts using PySceneDetect (on CFR source)
    # -----------------------------------------------------------------------
    logger.info(f"Detecting scenes for {filename}...")
    raw_scenes = []

    try:
        video = open_video(active_video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=27.0))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        for idx, scene in enumerate(scene_list):
            start_sec = scene[0].get_seconds()
            end_sec = scene[1].get_seconds()

            # Avoid extremely short scenes
            if end_sec - start_sec < 1.0:
                continue

            raw_scenes.append({"start": start_sec, "end": end_sec})
    except Exception as e:
        logger.warning(f"PySceneDetect failed for {filename}: {e}. Using fixed interval fallback.")

    # Fallback to fixed intervals (8-second segments) if PySceneDetect yields nothing or fails
    if not raw_scenes:
        interval = 8.0
        current_time = 0.0
        while current_time < duration:
            end_time = min(current_time + interval, duration)
            if end_time - current_time >= 1.0:
                raw_scenes.append({"start": current_time, "end": end_time})
            current_time = end_time

    # -----------------------------------------------------------------------
    # Step 3: Convert raw scene dicts to EGTSegment objects with keyframes
    # -----------------------------------------------------------------------
    logger.info(f"Creating EGT segments and extracting keyframes for {len(raw_scenes)} scenes...")
    segments: List[EGTSegment] = []

    for s in raw_scenes:
        start_sec = s["start"]
        end_sec = s["end"]

        # Extract keyframe at midpoint
        midpoint = start_sec + (end_sec - start_sec) / 2.0
        kf_filename = f"{os.path.splitext(filename)[0]}_scene_{start_sec:.0f}.jpg"
        kf_path = os.path.join(keyframes_dir, kf_filename)
        kf_success = extract_keyframe(active_video_path, midpoint, kf_path)

        segment = EGTSegment(
            clip_id=generate_clip_id(filename, start_sec, end_sec),
            source_file=filename,
            source_file_hash=source_file_hash,
            start_sec=start_sec,
            end_sec=end_sec,
            keyframe_path=kf_path if kf_success else None,
        )
        segments.append(segment)

    # -----------------------------------------------------------------------
    # Step 4: Spatio-temporal expansion for long scenes
    # -----------------------------------------------------------------------
    segments = expand_long_scenes(
        segments, active_video_path, keyframes_dir, filename, source_file_hash
    )
    logger.info(f"Segment count after temporal expansion: {len(segments)}")

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
