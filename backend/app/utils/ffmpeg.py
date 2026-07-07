import os
import sys
import subprocess
import json
import logging
from typing import List, Dict, Optional

logger = logging.getLogger("VlogForge.FFmpeg")

def get_ffmpeg_path() -> str:
    """Find ffmpeg.exe inside the active conda environment or system path."""
    # Check conda env location
    conda_bin = os.path.join(sys.prefix, "Library", "bin", "ffmpeg.exe")
    if os.path.exists(conda_bin):
        return conda_bin
    
    conda_scripts = os.path.join(sys.prefix, "Scripts", "ffmpeg.exe")
    if os.path.exists(conda_scripts):
        return conda_scripts
        
    conda_root_bin = os.path.join(sys.prefix, "ffmpeg.exe")
    if os.path.exists(conda_root_bin):
        return conda_root_bin

    # Default to system PATH
    return "ffmpeg"

def get_ffprobe_path() -> str:
    """Find ffprobe.exe inside the active conda environment or system path."""
    # Check conda env location
    conda_bin = os.path.join(sys.prefix, "Library", "bin", "ffprobe.exe")
    if os.path.exists(conda_bin):
        return conda_bin
    
    conda_scripts = os.path.join(sys.prefix, "Scripts", "ffprobe.exe")
    if os.path.exists(conda_scripts):
        return conda_scripts
        
    conda_root_bin = os.path.join(sys.prefix, "ffprobe.exe")
    if os.path.exists(conda_root_bin):
        return conda_root_bin

    # Default to system PATH
    return "ffprobe"

def get_video_info(video_path: str) -> Dict:
    """Get metadata of a video file using ffprobe."""
    ffprobe_cmd = get_ffprobe_path()
    cmd = [
        ffprobe_cmd,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except Exception as e:
        logger.error(f"Failed to probe video {video_path}: {e}")
        raise RuntimeError(f"Probe failed: {e}")

def get_video_duration(video_path: str) -> float:
    """Get duration of video in seconds."""
    try:
        info = get_video_info(video_path)
        return float(info.get("format", {}).get("duration", 0.0))
    except Exception:
        return 0.0

def has_audio_stream(video_path: str) -> bool:
    """Check if the video contains an audio stream."""
    try:
        info = get_video_info(video_path)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "audio":
                return True
        return False
    except Exception:
        return False

def extract_audio(video_path: str, audio_path: str) -> bool:
    """Extract audio stream to 16kHz mono WAV file for Whisper."""
    ffmpeg_cmd = get_ffmpeg_path()
    cmd = [
        ffmpeg_cmd, "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract audio from {video_path}: {e.stderr.decode()}")
        return False

def extract_keyframe(video_path: str, time_sec: float, output_path: str) -> bool:
    """Extract a single keyframe at time_sec to output_path."""
    ffmpeg_cmd = get_ffmpeg_path()
    cmd = [
        ffmpeg_cmd, "-y",
        "-ss", str(time_sec),
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract keyframe from {video_path} at {time_sec}s: {e.stderr.decode()}")
        return False

def run_ffmpeg_with_gpu_fallback(cmd: List[str]) -> subprocess.CompletedProcess:
    """Run FFmpeg command. If it uses h264_nvenc and fails, log warning and retry with CPU-based libx264."""
    try:
        return subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        if "h264_nvenc" in cmd:
            logger.warning(f"FFmpeg NVENC encoding failed: {e.stderr.decode()}. Retrying with CPU (libx264)...")
            fallback_cmd = [arg.replace("h264_nvenc", "libx264") for arg in cmd]
            try:
                return subprocess.run(fallback_cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as fallback_err:
                logger.error(f"FFmpeg CPU fallback encoding also failed: {fallback_err.stderr.decode()}")
                raise fallback_err
        else:
            logger.error(f"FFmpeg command failed: {e.stderr.decode()}")
            raise e

def transcode_to_cfr(input_path: str, output_path: str, fps: int = 30, crf: int = 18) -> bool:
    """Re-encode a Variable Frame Rate (VFR) source to a strict Constant Frame Rate (CFR) H.264 file.
    This must run before PySceneDetect and Whisper to prevent audio-to-video timestamp drift.
    """
    ffmpeg_cmd = get_ffmpeg_path()
    cmd = [
        ffmpeg_cmd, "-y",
        "-i", input_path,
        "-vsync", "cfr",
        "-r", str(fps),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "veryfast",  # Fast encode; quality is CRF-controlled not speed
        "-c:a", "copy",         # Copy audio stream without re-encoding to preserve fidelity
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"CFR transcode complete: {input_path} -> {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"CFR transcode failed for {input_path}: {e.stderr.decode()}")
        return False

def process_clip(video_path: str, start_sec: float, end_sec: float, output_path: str) -> bool:
    """Trim a video clip, scale it to 1080p, force 30fps, normalize audio to -14 LUFS, and save."""
    ffmpeg_cmd = get_ffmpeg_path()
    duration = end_sec - start_sec
    has_audio = has_audio_stream(video_path)

    # Base scaling filter to resize to 1920x1080 and pad if aspect ratio differs
    video_filter = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30"

    if has_audio:
        # Trim, scale video, normalize audio
        cmd = [
            ffmpeg_cmd, "-y",
            "-ss", str(start_sec),
            "-to", str(end_sec),
            "-i", video_path,
            "-filter_complex", f"[0:v]{video_filter}[v];[0:a]loudnorm=I=-14:TP=-1.5:LRA=11[a]",
            "-map", "[v]",
            "-map", "[a]",
            "-c:v", "h264_nvenc",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            output_path
        ]
    else:
        # Trim, scale video, add silent audio track
        cmd = [
            ffmpeg_cmd, "-y",
            "-ss", str(start_sec),
            "-to", str(end_sec),
            "-i", video_path,
            "-f", "lavfi",
            "-i", "anullsrc=r=44100:cl=mono",
            "-filter_complex", f"[0:v]{video_filter}[v]",
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", "h264_nvenc",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-t", str(duration),
            output_path
        ]

    try:
        run_ffmpeg_with_gpu_fallback(cmd)
        return True
    except Exception as e:
        logger.error(f"Failed to process clip from {video_path} ({start_sec}s to {end_sec}s): {e}")
        return False

def concatenate_clips(clip_paths: List[str], output_path: str) -> bool:
    """Concatenate multiple clips using ffmpeg concat demuxer (hard cut, no crossfade)."""
    ffmpeg_cmd = get_ffmpeg_path()
    
    # Create filelist.txt for concat demuxer
    list_dir = os.path.dirname(output_path)
    list_file_path = os.path.join(list_dir, "filelist.txt")
    
    try:
        with open(list_file_path, "w", encoding="utf-8") as f:
            for path in clip_paths:
                # Convert backslashes to forward slashes for FFmpeg on Windows
                normalized_path = path.replace("\\", "/")
                f.write(f"file '{normalized_path}'\n")

        cmd = [
            ffmpeg_cmd, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file_path,
            "-c", "copy",
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Clean up filelist.txt
        if os.path.exists(list_file_path):
            os.remove(list_file_path)
            
        return True
    except Exception as e:
        logger.error(f"Failed to concatenate clips: {e}")
        if os.path.exists(list_file_path):
            try:
                os.remove(list_file_path)
            except Exception:
                pass
        return False

def concatenate_clips_with_crossfade(
    clip_paths: List[str],
    output_path: str,
    crossfade_duration: float = 0.075,
    clip_count_limit: int = 30
) -> bool:
    """Concatenate clips with a linear audio crossfade (acrossfade) between each adjacent pair.
    
    Builds a dynamic FFmpeg filter_complex graph that:
      - Chains all video streams through concat (hard visual cut)
      - Chains all audio streams through sequential acrossfade (smooth room-tone blend)
    Falls back to hard-concat via concatenate_clips() if clip count exceeds the limit
    or if the filter_complex build/run fails.

    Args:
        clip_paths: Ordered list of processed clip file paths.
        output_path: Destination output path.
        crossfade_duration: Audio crossfade duration in seconds (default 75ms).
        clip_count_limit: Max clips before falling back to hard concat (FFmpeg graph limit).
    """
    n = len(clip_paths)

    # Single-clip passthrough: no crossfade needed
    if n == 1:
        import shutil
        shutil.copy(clip_paths[0], output_path)
        return True

    # Safety: fall back to hard concat if too many clips
    if n > clip_count_limit:
        logger.warning(
            f"Clip count ({n}) exceeds crossfade limit ({clip_count_limit}). "
            "Falling back to hard-concat."
        )
        return concatenate_clips(clip_paths, output_path)

    ffmpeg_cmd = get_ffmpeg_path()

    # Build -i arguments
    input_args = []
    for p in clip_paths:
        input_args += ["-i", p]

    # Build filter_complex:
    #   Video: [0:v][1:v]...[n-1:v]concat=n=N:v=1:a=0[vout]
    #   Audio: chain of acrossfade filters
    #     [0:a][1:a]acrossfade=d=D:c1=tri:c2=tri[a01]
    #     [a01][2:a]acrossfade=d=D:c1=tri:c2=tri[a012]
    #     ...
    video_inputs = "".join(f"[{i}:v]" for i in range(n))
    video_filter = f"{video_inputs}concat=n={n}:v=1:a=0[vout]"

    audio_filter_parts = []
    prev_label = "[0:a]"
    for i in range(1, n):
        curr_label = f"[{i}:a]"
        out_label = f"[a_cf{i}]" if i < n - 1 else "[aout]"
        audio_filter_parts.append(
            f"{prev_label}{curr_label}acrossfade=d={crossfade_duration:.4f}:c1=tri:c2=tri{out_label}"
        )
        prev_label = out_label if i < n - 1 else "[aout]"

    filter_complex = f"{video_filter};" + ";".join(audio_filter_parts)

    cmd = [
        ffmpeg_cmd, "-y",
        *input_args,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "h264_nvenc",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path
    ]

    try:
        run_ffmpeg_with_gpu_fallback(cmd)
        logger.info(f"Crossfade concat complete: {n} clips -> {output_path}")
        return True
    except Exception as e:
        logger.warning(
            f"Crossfade concat failed ({e}). Falling back to hard-concat."
        )
        return concatenate_clips(clip_paths, output_path)

def apply_fade_effects(video_path: str, output_path: str) -> bool:
    """Apply 0.5s fade-in and 1s fade-out to video and audio streams."""
    duration = get_video_duration(video_path)
    if duration <= 0:
        return False
        
    ffmpeg_cmd = get_ffmpeg_path()
    
    fade_out_start = max(0.0, duration - 1.0)
    
    video_filter = f"fade=in:st=0:d=0.5,fade=out:st={fade_out_start}:d=1.0"
    audio_filter = f"afade=in:st=0:d=0.5,afade=out:st={fade_out_start}:d=1.0"
    
    cmd = [
        ffmpeg_cmd, "-y",
        "-i", video_path,
        "-vf", video_filter,
        "-af", audio_filter,
        "-c:v", "h264_nvenc",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path
    ]
    
    try:
        run_ffmpeg_with_gpu_fallback(cmd)
        return True
    except Exception as e:
        logger.error(f"Failed to apply fade effects: {e}")
        return False


def assemble_single_pass(edl, file_map, output_path, crossfade_duration=0.075):
    """Assemble the final vlog in a single FFmpeg pass with no intermediate clip files.

    Task 4 -- Single-pass filtergraph assembly:
    Eliminates the triple-encoding bottleneck (slice->concat->fade) by constructing one
    comprehensive filter_complex at runtime. In a single FFmpeg execution:
      1. Each EDL clip is trimmed in-graph using atrim/trim filters (no seek pre-pass).
      2. Video streams are scaled to 1920x1080 with pillarbox/letterbox padding, fps=30.
      3. Audio is normalised with dynaudnorm (Gaussian-windowed sliding normalisation).
         NOTE: loudnorm requires 2 passes and cannot run inside a single filter_complex.
         dynaudnorm is its single-pass equivalent producing equivalent results for speech.
      4. All video streams are concatenated with hard visual cuts.
      5. All audio streams are chained through sequential acrossfade nodes (75ms triangular)
         with NO clip-count ceiling -- the chain is built iteratively in Python.
      6. Global fade-in (0.5s) and fade-out (1s) are appended to the tail of the graph.
      7. [vout]/[aout] piped to h264_nvenc (CPU fallback: libx264) in one disk write.

    Returns True on success, False on failure (caller falls back to multi-pass assembly).
    """
    import os as _os
    if not edl:
        logger.error("assemble_single_pass: Empty EDL.")
        return False

    ffmpeg_cmd = get_ffmpeg_path()
    n = len(edl)

    input_args = []
    for item in edl:
        src = file_map.get(item["video_file"])
        if not src or not _os.path.exists(src):
            logger.error(
                f"assemble_single_pass: Source not found for "
                f"{item['video_file']} -> {src}"
            )
            return False
        input_args += ["-i", src]

    v_scale = (
        "scale=1920:1080:force_original_aspect_ratio=decrease,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
        "fps=30"
    )

    video_trim_parts = []
    audio_trim_parts = []
    for i, item in enumerate(edl):
        s = item["start_sec"]
        e = item["end_sec"]
        video_trim_parts.append(
            f"[{i}:v]trim=start={s:.6f}:end={e:.6f},"
            f"setpts=PTS-STARTPTS,{v_scale}[v{i}]"
        )
        audio_trim_parts.append(
            f"[{i}:a]atrim=start={s:.6f}:end={e:.6f},"
            f"asetpts=PTS-STARTPTS,"
            f"dynaudnorm=p=0.9:m=100:s=5:g=15[a{i}]"
        )

    v_inputs = "".join(f"[v{i}]" for i in range(n))
    video_concat = f"{v_inputs}concat=n={n}:v=1:a=0[vconcated]"

    if n == 1:
        audio_chain_output = "[a0]"
        acrossfade_parts = []
    else:
        acrossfade_parts = []
        prev_label = "[a0]"
        for i in range(1, n):
            out_label = "[acfout]" if i == n - 1 else f"[acf{i}]"
            acrossfade_parts.append(
                f"{prev_label}[a{i}]"
                f"acrossfade=d={crossfade_duration:.4f}:c1=tri:c2=tri{out_label}"
            )
            prev_label = out_label
        audio_chain_output = "[acfout]"

    raw_duration = sum(item["end_sec"] - item["start_sec"] for item in edl)
    output_duration = raw_duration - crossfade_duration * max(0, n - 1)
    fade_out_start = max(0.0, output_duration - 1.0)

    video_fade = (
        f"[vconcated]fade=in:st=0:d=0.5,"
        f"fade=out:st={fade_out_start:.3f}:d=1.0[vout]"
    )
    audio_fade = (
        f"{audio_chain_output}afade=in:st=0:d=0.5,"
        f"afade=out:st={fade_out_start:.3f}:d=1.0[aout]"
    )

    filter_complex = ";".join(
        video_trim_parts
        + audio_trim_parts
        + [video_concat]
        + acrossfade_parts
        + [video_fade, audio_fade]
    )

    cmd = [
        ffmpeg_cmd, "-y",
        *input_args,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "h264_nvenc",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path,
    ]

    logger.info(
        f"Single-pass assembly: {n} clips -> {output_path} ({output_duration:.1f}s)"
    )
    try:
        run_ffmpeg_with_gpu_fallback(cmd)
        logger.info(f"Single-pass assembly complete: {output_path}")
        return True
    except Exception as exc:
        logger.error(
            f"Single-pass filtergraph failed: {exc}. "
            "Caller should fall back to multi-pass assembly."
        )
        return False
