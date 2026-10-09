"""Review-screen extras for a word-grid job: the output's audio waveform, and the last render's
edit decision list exported for other editors (CMX3600 .edl for Premiere/Resolve/Avid, FCPXML 1.9
for Final Cut/Resolve). Both read the stored timeline, so they describe the LAST RENDER.

Exports use the render's frame grid (compiler FPS): source and record ranges are the same whole
frames the render cut. Source timecode starts at 00:00:00:00 unless the file carries a timecode tag."""

import logging
import os
import subprocess
from fractions import Fraction
from typing import Dict, Optional
from urllib.parse import quote
from xml.sax.saxutils import quoteattr

import numpy as np

from app.config import settings
from app.tasks.compiler import FPS
from app.utils import artifacts
from app.utils.ffmpeg import get_ffmpeg_path, get_video_info

logger = logging.getLogger("VlogForge.ReviewExport")

WAVE_RATE = 50              # waveform peaks per second of output
WAVE_SR = 8000              # decode rate for the waveform (peaks only, no need for more)
WAVE_FLOOR_DB = -48.0       # level drawn as zero height
RECORD_START_SEC = 3600     # EDL record timecode starts at 01:00:00:00 (NLE convention)


def output_waveform(job_id: str) -> Optional[Dict]:
    """Peak level of the rendered output, WAVE_RATE values per second scaled 0..1 on a dB scale.
    Cached in the artifact store, recomputed when the output file changes. None = no output."""
    path = os.path.join(settings.output_dir, f"{job_id}.mp4")
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    cached = artifacts.load_job(job_id, "waveform")
    if cached and cached.get("mtime") == mtime:
        return cached
    pcm = subprocess.run([get_ffmpeg_path(), "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(WAVE_SR),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    x = np.abs(np.frombuffer(pcm, dtype=np.int16).astype(np.float32)) / 32768.0
    hop = WAVE_SR // WAVE_RATE
    n = len(x) // hop
    peaks = x[: n * hop].reshape(n, hop).max(axis=1) if n else np.zeros(0)
    db = 20 * np.log10(np.maximum(peaks, 1e-6))
    scaled = np.clip(1 - db / WAVE_FLOOR_DB, 0, 1)
    out = {"mtime": mtime, "rate": WAVE_RATE, "peaks": [round(float(v), 3) for v in scaled]}
    artifacts.save_job(job_id, "waveform", out)
    logger.info(f"[WAVEFORM] job {job_id}: {n} peaks ({n / WAVE_RATE:.1f}s) from {path}")
    return out


def _probe(path: str) -> Dict:
    """Display size, frame rate, duration and start timecode of a source file (ffprobe)."""
    info = get_video_info(path) or {}
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    w, h = int(video.get("width") or 1920), int(video.get("height") or 1080)
    rotation = int(float((video.get("tags") or {}).get("rotate", 0) or 0))
    for sd in video.get("side_data_list") or []:
        rotation = int(sd.get("rotation", rotation) or rotation)
    if abs(rotation) % 180 == 90:
        w, h = h, w
    tc = None
    for s in [info.get("format", {})] + info.get("streams", []):
        tc = tc or (s.get("tags") or {}).get("timecode")
    if not video:
        logger.warning(f"[EXPORT] {path}: no video stream found by ffprobe, assuming {w}x{h}")
    rate = video.get("r_frame_rate") or ""
    if not rate or rate.startswith("0/") or rate.endswith("/0"):
        rate = f"{FPS}/1"
    return {"width": w, "height": h, "rate": rate,
            "duration": float(info.get("format", {}).get("duration") or 0), "timecode": tc}


def _tc_frames(p: Dict) -> int:
    """A source's start timecode (HH:MM:SS:FF at its own rate) in FPS frames; 0 without one."""
    if not p["timecode"]:
        return 0
    h, m, s, f = (int(x) for x in p["timecode"].replace(";", ":").split(":"))
    rate = float(Fraction(p["rate"])) or FPS
    return round(((h * 60 + m) * 60 + s + f / round(rate)) * FPS)


def _tc(frames: int) -> str:
    s, f = divmod(frames, FPS)
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}:{f:02d}"


def _events(job_id: str) -> Optional[Dict]:
    """The last render as whole-frame events: source file, source in, duration, record in (frames)."""
    timeline, files = artifacts.load_job(job_id, "timeline"), artifacts.load_job(job_id, "files")
    if timeline is None or files is None:
        return None
    moments = (artifacts.load_job(job_id, "moments") or {}).get("moments", [])
    rec, events = 0, []
    for s in timeline["segments"]:
        dur = round((s["src_out"] - s["src_in"]) * FPS)
        ids = set(s["word_ids"])
        notes = [m["summary"] for m in moments if ids & set(m["word_ids"])]
        events.append({"file": s["source_file"], "src_in": round(s["src_in"] * FPS), "dur": dur, "rec_in": rec, "notes": notes})
        rec += dur
    return {"events": events, "files": {f["filename"]: f["path"] for f in files}, "duration": rec}


def export_edl(job_id: str, title: str) -> Optional[str]:
    """CMX3600 EDL of the last render (video + audio events, cuts only)."""
    ev = _events(job_id)
    if ev is None:
        return None
    probes = {f: _probe(p) for f, p in ev["files"].items()}
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    for n, e in enumerate(ev["events"], 1):
        reel = "".join(c for c in os.path.splitext(e["file"])[0].upper() if c.isalnum() or c == "_")[:8] or "AX"
        src0 = _tc_frames(probes[e["file"]]) + e["src_in"]
        rec0 = RECORD_START_SEC * FPS + e["rec_in"]
        lines.append(f"{n:03d}  {reel:<8} AA/V  C        {_tc(src0)} {_tc(src0 + e['dur'])} {_tc(rec0)} {_tc(rec0 + e['dur'])}")
        lines.append(f"* FROM CLIP NAME: {e['file']}")
        for note in e["notes"]:
            lines.append(f"* COMMENT: {note}")
        lines.append("")
    return "\n".join(lines)


def _t(frames_or_sec, per_sec: int = FPS) -> str:
    """FCPXML rational time: frames at per_sec -> 'N/Ds' (reduced), '0s' for zero."""
    v = Fraction(frames_or_sec, per_sec)
    return "0s" if v == 0 else (f"{v.numerator}s" if v.denominator == 1 else f"{v.numerator}/{v.denominator}s")


def export_fcpxml(job_id: str, title: str) -> Optional[str]:
    """FCPXML 1.9 project of the last render: one asset per source file, one asset-clip per event."""
    ev = _events(job_id)
    if ev is None:
        return None
    probes = {f: _probe(p) for f, p in ev["files"].items()}
    out = os.path.join(settings.output_dir, f"{job_id}.mp4")       # sequence = the render's frame size
    seq = _probe(out) if os.path.exists(out) else (probes[ev["events"][0]["file"]] if ev["events"] else {"width": 1920, "height": 1080})
    res = [f'    <format id="r0" frameDuration="1/{FPS}s" width="{seq["width"]}" height="{seq["height"]}"/>']
    asset_ids = {}
    for k, (f, p) in enumerate(probes.items(), 1):
        rate = Fraction(p["rate"]) or Fraction(FPS)
        res.append(f'    <format id="f{k}" frameDuration="{_t(rate.denominator, rate.numerator)}" width="{p["width"]}" height="{p["height"]}"/>')
        tc0 = _tc_frames(p)
        start, dur = _t(tc0), _t(round(p["duration"] * FPS))
        url = "file://" + quote(os.path.abspath(ev["files"][f]))
        res.append(f'    <asset id="a{k}" name={quoteattr(os.path.splitext(f)[0])} start="{start}" duration="{dur}" '
                   f'hasVideo="1" hasAudio="1" format="f{k}" audioSources="1" audioChannels="2" audioRate="48000">\n'
                   f'      <media-rep kind="original-media" src={quoteattr(url)}/>\n    </asset>')
        asset_ids[f] = (f"a{k}", f"f{k}", tc0)
    clips = []
    for e in ev["events"]:
        aid, fid, tc0 = asset_ids[e["file"]]
        name = e["notes"][0] if e["notes"] else os.path.splitext(e["file"])[0]
        clips.append(f'            <asset-clip ref="{aid}" name={quoteattr(name)} offset="{_t(e["rec_in"])}" '
                     f'start="{_t(tc0 + e["src_in"])}" duration="{_t(e["dur"])}" format="{fid}" tcFormat="NDF"/>')
    return "\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>', "<!DOCTYPE fcpxml>", '<fcpxml version="1.9">', "  <resources>",
        *res, "  </resources>", "  <library>", '    <event name="VlogForge">', f"      <project name={quoteattr(title)}>",
        f'        <sequence format="r0" duration="{_t(ev["duration"])}" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">',
        "          <spine>", *clips, "          </spine>", "        </sequence>", "      </project>", "    </event>",
        "  </library>", "</fcpxml>", ""])
