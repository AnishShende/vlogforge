import os
import sys
import subprocess
import json
import shutil
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger("VlogForge.FFmpeg")

def get_hw_encoder() -> str:
    """Return the optimal hardware encoder for the current platform."""
    return "h264_videotoolbox" if sys.platform == "darwin" else "h264_nvenc"

def get_ffmpeg_path() -> str:
    """Find ffmpeg in the active conda environment or system path."""
    # shutil.which respects the active PATH (and Conda env) on all OSes
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        return ffmpeg_bin
    
    # Fallback if not found (should rarely happen in a properly activated conda env)
    return "ffmpeg"

def get_ffprobe_path() -> str:
    """Find ffprobe in the active conda environment or system path."""
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin:
        return ffprobe_bin
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


def run_ffmpeg_with_gpu_fallback(cmd: List[str]) -> subprocess.CompletedProcess:
    """Run FFmpeg command. If it uses a hardware encoder and fails, log warning and retry with CPU-based libx264."""
    try:
        return subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        hw_encoder = get_hw_encoder()
        if hw_encoder in cmd:
            logger.warning(f"FFmpeg {hw_encoder} encoding failed: {e.stderr.decode()}. Retrying with CPU (libx264)...")
            fallback_cmd = [arg.replace(hw_encoder, "libx264") for arg in cmd]
            try:
                return subprocess.run(fallback_cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as fallback_err:
                logger.error(f"FFmpeg CPU fallback encoding also failed: {fallback_err.stderr.decode()}")
                raise fallback_err
        else:
            logger.error(f"FFmpeg command failed: {e.stderr.decode()}")
            raise e

def generate_proxy(input_path: str, output_path: str, fps: int = 30) -> bool:
    """Generate a lightweight 360p proxy for audio extraction.
    Also enforces CFR to prevent audio/video timestamp drift.
    """
    ffmpeg_cmd = get_ffmpeg_path()
    cmd = [
        ffmpeg_cmd, "-y",
        "-i", input_path,
        "-vsync", "cfr",
        "-r", str(fps),
        "-vf", "scale=-1:360",  # Downscale to 360p for lightning-fast proxy generation
        "-c:v", get_hw_encoder(),
        "-b:v", "1M",           # Very low target bitrate
        "-c:a", "copy",         # Copy audio stream without re-encoding to preserve fidelity
        output_path
    ]
    try:
        run_ffmpeg_with_gpu_fallback(cmd)
        logger.info(f"Proxy generation complete: {input_path} -> {output_path}")
        return True
    except Exception as e:
        logger.error(f"Proxy generation failed for {input_path}: {e}")
        return False








# ---------------------------------------------------------------------------
# Compiled-timeline render: the word-grid compiler's output, cut on frames.
# ---------------------------------------------------------------------------

RENDER_SR = 48000
RENDER_SIZE = (1920, 1080)


def _render_scale(w: int, h: int) -> str:
    """Fit the whole frame inside w x h, centred, black bars where the shapes differ."""
    return f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1"
_LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"


def is_audio_only(path: str) -> bool:
    """True for a file without a picture (a voice-over recording); cover art does not count as a picture."""
    return not any(s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")
                   for s in get_video_info(path).get("streams", []))


def _compiled_inputs(segments: List[Dict], file_map: Dict[str, str]) -> List[str]:
    """One accurately seeked input per segment (+0.5 s slack; exact length is set in-graph)."""
    args = []
    for s in segments:
        src = file_map.get(s["source_file"])
        if not src or not os.path.exists(src):
            raise FileNotFoundError(f"source for {s['source_file']!r} not found at {src!r}")
        args += ["-ss", f"{s['src_in']:.6f}", "-t", f"{s['src_out'] - s['src_in'] + 0.5:.6f}", "-i", src]
    return args


def _picture_inputs(segments: List[Dict], file_map: Dict[str, str], fps: int, size: Tuple[int, int]) -> Tuple[List[str], Dict[int, int]]:
    """Segments from audio-only files (voice-over) have no picture: one black lavfi input each, appended
    after the segment inputs. Returns (extra args, {segment index: input index of its picture})."""
    args, pic, audio_only = [], {}, {}
    for i, s in enumerate(segments):
        src = file_map[s["source_file"]]
        if src not in audio_only:
            audio_only[src] = is_audio_only(src)
        if audio_only[src]:
            pic[i] = len(segments) + len(pic)
            args += ["-f", "lavfi", "-t", f"{s['src_out'] - s['src_in'] + 0.5:.6f}", "-i",
                     f"color=c=black:s={size[0]}x{size[1]}:r={fps}"]
    return args, pic


def _compiled_audio_graph(segments: List[Dict], fps: int, join_fade: float, head_fade: float,
                          tail_fade: float, loudnorm: str) -> List[str]:
    """Each segment: exactly round(frames * SR / fps) samples at 48 kHz stereo, short fades at
    both ends; concat; global fades no longer than the first/last pause; one loudnorm pass."""
    parts, total = [], 0.0
    for i, s in enumerate(segments):
        frames = round((s["src_out"] - s["src_in"]) * fps)
        dur = frames / fps
        total += dur
        fade = min(join_fade, dur / 4)
        parts.append(
            f"[{i}:a]aresample={RENDER_SR},aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"asetpts=PTS-STARTPTS,apad=pad_dur=0.5,atrim=end_sample={frames * RENDER_SR // fps},"
            f"asetpts=PTS-STARTPTS,afade=t=in:d={fade:.4f},afade=t=out:st={dur - fade:.6f}:d={fade:.4f}[a{i}]")
    ins = "".join(f"[a{i}]" for i in range(len(segments)))
    chain = f"{ins}concat=n={len(segments)}:v=0:a=1"
    if head_fade > 0:
        chain += f",afade=t=in:d={head_fade:.4f}"
    if tail_fade > 0:
        chain += f",afade=t=out:st={total - tail_fade:.6f}:d={tail_fade:.4f}"
    parts.append(f"{chain},{loudnorm},aresample={RENDER_SR}[aout]")
    return parts


def _measure_loudness(cmd_inputs: List[str], audio_graph: List[str]) -> Dict:
    """Pass 1 of two-pass loudnorm over the whole compiled audio."""
    cmd = [get_ffmpeg_path(), "-hide_banner", "-nostats", *cmd_inputs,
           "-filter_complex", ";".join(audio_graph), "-map", "[aout]", "-f", "null", "-"]
    err = subprocess.run(cmd, check=True, capture_output=True).stderr.decode()
    return json.loads(err[err.rindex("{"): err.rindex("}") + 1])


def _overlay_inputs(overlays: List[Dict], file_map: Dict[str, str]) -> List[str]:
    """One accurately seeked input per overlay (picture only; +0.5 s slack, exact length set in-graph)."""
    args = []
    for o in overlays:
        src = file_map.get(o["source_file"])
        if not src or not os.path.exists(src):
            raise FileNotFoundError(f"overlay source {o['source_file']!r} not found at {src!r}")
        dur = o["src_out"] - o["src_in"] if o.get("src_out") else o["rec_out"] - o["rec_in"] + 0.5
        args += ["-ss", f"{o['src_in']:.6f}", "-t", f"{dur:.6f}", "-an", "-i", src]   # never past the shot's end
    return args


def _overlay_graph(overlays: List[Dict], first_input: int, fps: int, size: Tuple[int, int], base: str) -> List[str]:
    """Lay each overlay's picture over [base] from rec_in to rec_out (output clock, whole frames); a shot shorter
    than its slot holds its last frame. Output label [vov]. The audio graph is not touched."""
    parts, cur = [], base
    for k, o in enumerate(overlays):
        a, b = round(o["rec_in"] * fps), round(o["rec_out"] * fps)
        parts.append(f"[{first_input + k}:v]setpts=PTS-STARTPTS,fps={fps},{_render_scale(*size)},"
                     f"tpad=stop_mode=clone:stop_duration={(b - a) / fps + 0.5:.4f},trim=end_frame={b - a},"
                     f"setpts=PTS-STARTPTS+{a}/({fps}*TB)[o{k}]")
        nxt = "vov" if k == len(overlays) - 1 else f"ov{k}"
        parts.append(f"[{cur}][o{k}]overlay=eof_action=pass:enable='between(n,{a},{b - 1})'[{nxt}]")
        cur = nxt
    return parts


def render_compiled(segments: List[Dict], file_map: Dict[str, str], output_path: str, fps: int = 30,
                    join_fade: float = 0.015, head_fade: float = 0.0, tail_fade: float = 0.0,
                    size: Tuple[int, int] = RENDER_SIZE, overlays: Optional[List[Dict]] = None,
                    subtitles: Optional[str] = None) -> Dict:
    """Render compiled segments [{source_file, src_in, src_out}] in one pass.

    Video and audio of every segment have the same exact length (whole frames; the compiler
    snaps segment lengths), so A/V stays in sync across any number of cuts. Joins get short
    non-overlapping audio fades; loudness is normalised once over the whole output (two-pass
    loudnorm, linear gain), not per clip. size: output frame (w, h), each clip fitted with black bars.
    A segment from an audio-only file (voice-over) gets a black picture of its exact length; its sound is
    treated like any other segment's. overlays: [{source_file, src_in, rec_in, rec_out}] B-roll pictures laid
    over the main picture on the OUTPUT clock (rec_*: seconds of the output); the sound is never touched.
    subtitles: path of an ASS file (output clock, PlayRes = size) burned in over picture and B-roll, under the fades.
    Returns {"loudness": measured, "duration_sec": ...}.
    Raises on failure (fail loud; no fallback path).
    """
    if not segments:
        raise ValueError("render_compiled: no segments")
    inputs = _compiled_inputs(segments, file_map)
    pic_args, pic = _picture_inputs(segments, file_map, fps, size)
    measured = _measure_loudness(inputs, _compiled_audio_graph(
        segments, fps, join_fade, head_fade, tail_fade, _LOUDNORM + ":print_format=json"))
    norm = (f"{_LOUDNORM}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
            f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
            f":offset={measured['target_offset']}:linear=true")
    video = []
    total = 0.0
    for i, s in enumerate(segments):
        frames = round((s["src_out"] - s["src_in"]) * fps)
        total += frames / fps
        video.append(f"[{pic.get(i, i)}:v]setpts=PTS-STARTPTS,fps={fps},{_render_scale(*size)},tpad=stop_mode=clone:stop_duration=0.5,"
                     f"trim=end_frame={frames},setpts=PTS-STARTPTS[v{i}]")
    vins = "".join(f"[v{i}]" for i in range(len(segments)))
    overlays = [o for o in overlays or [] if round(o["rec_out"] * fps) > round(o["rec_in"] * fps)]
    fades = f"fade=in:st=0:d=0.5,fade=out:st={max(0.0, total - 1.0):.6f}:d=1.0"
    if subtitles:
        if not os.path.exists(subtitles):
            raise FileNotFoundError(f"subtitles file not found: {subtitles!r}")
        if any(c in subtitles for c in ":'\\,;[]"):     # filtergraph syntax; the caller picks a plain temp path
            raise ValueError(f"subtitles path has filtergraph characters: {subtitles!r}")
        fades = f"ass=filename='{subtitles}',{fades}"
    if overlays:                   # B-roll over the main picture, then the global fades over both
        video.append(f"{vins}concat=n={len(segments)}:v=1:a=0[vbase]")
        video += _overlay_graph(overlays, len(segments) + len(pic), fps, size, "vbase")
        video.append(f"[vov]{fades}[vout]")
    else:
        video.append(f"{vins}concat=n={len(segments)}:v=1:a=0,{fades}[vout]")
    graph = video + _compiled_audio_graph(segments, fps, join_fade, head_fade, tail_fade, norm)
    cmd = [get_ffmpeg_path(), "-y", *inputs, *pic_args, *_overlay_inputs(overlays, file_map),
           "-filter_complex", ";".join(graph), "-map", "[vout]", "-map", "[aout]", "-r", str(fps), "-c:v", get_hw_encoder(), "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", output_path]
    logger.info(f"Compiled render: {len(segments)} segments + {len(overlays)} overlays{' + subtitles' if subtitles else ''} -> {output_path} ({total:.2f}s, {size[0]}x{size[1]}), "
                f"input loudness {measured['input_i']} LUFS")
    run_ffmpeg_with_gpu_fallback(cmd)
    return {"loudness": measured, "duration_sec": round(total, 6)}


def media_durations(path: str) -> Dict[str, float]:
    """{"video": container duration, "audio": DECODED samples / rate}. ffprobe's audio stream
    duration rounds up to whole AAC packets plus priming (+33..67 ms not in the audio)."""
    out = subprocess.run([get_ffprobe_path(), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration",
                          "-of", "json", path], check=True, capture_output=True).stdout
    pcm = subprocess.run([get_ffmpeg_path(), "-v", "error", "-i", path, "-map", "0:a:0", "-ac", "1", "-ar", "16000",
                          "-f", "s16le", "-"], check=True, capture_output=True).stdout
    return {"video": float(json.loads(out)["streams"][0]["duration"]), "audio": len(pcm) / 2 / 16000}


def parse_recording_time(tags: Dict) -> Optional[str]:
    """ISO 8601 recording time from container tags: the QuickTime creation date (local time with its
    UTC offset) first, else creation_time (UTC). None when absent or implausible (a camera clock never
    set). Never guessed from the filename: messaging apps strip these tags and name files by save time."""
    from datetime import datetime
    for key in ("com.apple.quicktime.creationdate", "creation_time"):
        try:
            t = datetime.fromisoformat(str(tags[key]).strip())
        except (KeyError, ValueError):
            continue
        if t.tzinfo is not None and t.year >= 2010:
            return t.isoformat()
    return None


def recording_time(path: str) -> Optional[str]:
    """When the clip was recorded (see parse_recording_time), or None."""
    out = subprocess.run([get_ffprobe_path(), "-v", "error", "-show_entries", "format_tags", "-of", "json", path],
                         capture_output=True, text=True)
    if out.returncode != 0:
        logger.warning(f"recording time: ffprobe failed for {path}: {out.stderr.strip()[:200]}")
        return None
    return parse_recording_time(json.loads(out.stdout or "{}").get("format", {}).get("tags", {}))
