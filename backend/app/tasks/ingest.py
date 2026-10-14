import os
import logging
from typing import Dict
from app.utils.ffmpeg import get_video_duration, extract_audio
from app.models import VideoFileInfo

logger = logging.getLogger("VlogForge.Ingest")


def ingest_video(video_path: str, job_dir: str) -> Dict:
    """Make a lightweight proxy (picture files) and extract the audio WAV the word grid is built from.

    Returns a dict with:
        - file_info: VideoFileInfo
        - cfr_path: str  (the proxy, or the original when there is no proxy)
    """
    logger.info(f"Ingesting video: {video_path}")

    filename = os.path.basename(video_path)
    size_bytes = os.path.getsize(video_path)

    # Define paths
    proxy_dir = os.path.join(job_dir, "proxy")
    audio_dir = os.path.join(job_dir, "audio")
    os.makedirs(proxy_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 0: Proxy Generation
    # Generates a fast 360p proxy for audio extraction.
    # -----------------------------------------------------------------------
    from app.utils.ffmpeg import generate_proxy, is_audio_only
    audio_only = is_audio_only(video_path)       # voice-over recording: no picture, no proxy
    proxy_filename = f"{os.path.splitext(filename)[0]}_proxy.mp4"
    proxy_path = os.path.join(proxy_dir, proxy_filename)
    if audio_only:
        logger.info(f"{filename}: audio only (voice-over), no proxy")
        proxy_success = False
    elif os.path.exists(proxy_path):
        logger.info(f"Using cached lightweight proxy: {proxy_path}")
        proxy_success = True
    else:
        logger.info(f"Generating lightweight proxy: {video_path} -> {proxy_path}")
        proxy_success = generate_proxy(video_path, proxy_path)

    # Use proxy for downstream processing; fall back to original on failure
    active_video_path = proxy_path if proxy_success else video_path
    if not proxy_success and not audio_only:
        logger.warning(f"Proxy generation failed for {filename}. Continuing with original.")

    # Duration is read from the proxy file for accuracy
    duration = get_video_duration(active_video_path)

    # -----------------------------------------------------------------------
    # Step 1: Extract Audio WAV (from CFR source)
    # -----------------------------------------------------------------------
    audio_filename = f"{os.path.splitext(filename)[0]}.wav"
    audio_path = os.path.join(audio_dir, audio_filename)
    if os.path.exists(audio_path):
        logger.info(f"Using cached audio: {audio_path}")
        audio_success = True
    else:
        logger.info(f"Extracting audio to: {audio_path}")
        audio_success = extract_audio(active_video_path, audio_path)
    actual_audio_path = audio_path if audio_success else None

    file_info = VideoFileInfo(
        filename=filename,
        original_path=video_path,
        duration=duration,
        size_bytes=size_bytes,
        audio_path=actual_audio_path,
        audio_only=audio_only,
    )
    return {"file_info": file_info, "cfr_path": active_video_path}
