"""Export layouts (export panel): the last render's edit in standard frame shapes.

Each layout is its own MP4, rendered on demand from the stored timeline (same cuts, audio and
fades as the main render; only the frame size differs, each clip fitted with black bars).
The main render already is 16:9 1920x1080, so that layout is the main output, not a copy.
A layout file remembers the timeline it was made from: after a re-compile it shows as outdated.
Each layout can also be rendered with burned-in captions in one preset (app.tasks.subtitles.STYLES): a separate
file per (layout, preset), the main render stays clean; it is outdated when the cuts or the caption words change.
Renders run in a background thread, one per (job, layout, captions) at a time; status lives in the artifact
store (ready / failed) plus an in-process set (rendering)."""

import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from typing import Dict, List, Optional, Tuple

from app.config import settings
from app.models import CompiledTimeline
from app.tasks import subtitles
from app.utils import artifacts

logger = logging.getLogger("VlogForge.Layouts")

LAYOUTS = {
    "16x9": {"ratio": "16:9", "name": "Landscape", "size": (1920, 1080), "for": "YouTube"},
    "9x16": {"ratio": "9:16", "name": "Vertical", "size": (1080, 1920), "for": "Reels, Shorts, TikTok"},
    "1x1": {"ratio": "1:1", "name": "Square", "size": (1080, 1080), "for": "Instagram / Facebook feed, LinkedIn"},
    "4x5": {"ratio": "4:5", "name": "Portrait", "size": (1080, 1350), "for": "Instagram feed"},
}
MAIN = "16x9"               # what the main render is (app.utils.ffmpeg.RENDER_SIZE)

_running: Dict[Tuple[str, str, Optional[str]], float] = {}
_lock = threading.Lock()


def main_path(job_id: str) -> str:
    return os.path.join(settings.output_dir, f"{job_id}.mp4")


def layout_path(job_id: str, key: str, captions: Optional[str] = None) -> str:
    if captions:
        return os.path.join(settings.output_dir, f"{job_id}_{key}_cc-{captions}.mp4")
    return main_path(job_id) if key == MAIN else os.path.join(settings.output_dir, f"{job_id}_{key}.mp4")


def _store(key: str, captions: Optional[str]) -> str:
    return f"layout_{key}_cc-{captions}" if captions else f"layout_{key}"


def _check_captions(captions: Optional[str]) -> None:
    if captions is not None and captions not in subtitles.STYLES:
        raise ValueError(f"unknown caption style: {captions!r}")


def _fingerprint(timeline: Dict) -> str:
    return hashlib.sha256(json.dumps([(s["source_file"], s["src_in"], s["src_out"]) for s in timeline["segments"]]).encode()).hexdigest()[:16]


def _variant_fingerprint(job_id: str, timeline: Dict, captions: Optional[str]) -> str:
    fp = _fingerprint(timeline)
    if not captions:
        return fp
    cues = subtitles.job_cues(job_id)
    if cues is None:
        raise ValueError(f"job {job_id}: no word grid to caption")
    return fp + "-" + subtitles.fingerprint(cues, captions, subtitles.job_placement(job_id))


def list_layouts(job_id: str, captions: Optional[str] = None) -> Optional[List[Dict]]:
    """Every layout with its status: ready | rendering | stale (made from an older edit) | failed | none.
    captions: a caption preset = the captioned files in that preset (None = the clean files)."""
    _check_captions(captions)
    timeline = artifacts.load_job(job_id, "timeline")
    if timeline is None:
        return None
    fp, out = _variant_fingerprint(job_id, timeline, captions), []
    for key, L in LAYOUTS.items():
        row = {"key": key, **L, "size": list(L["size"]), "captions": captions, "error": None}
        stored = artifacts.load_job(job_id, _store(key, captions)) or {}
        if key == MAIN and not captions:
            row["status"] = "ready" if os.path.exists(main_path(job_id)) else "none"
        elif (job_id, key, captions) in _running:
            row["status"] = "rendering"
        elif not stored:
            row["status"] = "none"
        elif stored["fingerprint"] != fp:
            row["status"] = "stale"
        elif stored["status"] == "failed":
            row.update(status="failed", error=stored.get("error"))
        else:
            row["status"] = "ready" if os.path.exists(layout_path(job_id, key, captions)) else "none"
        out.append(row)
    return out


def source_dims(job_id: str) -> Dict[str, List[int]]:
    """Display size (rotation applied) of each source file, for the panel's layout preview: where a
    clip sits inside the 16:9 main render. ffprobe once per job, then cached."""
    cached = artifacts.load_job(job_id, "source_dims")
    if cached is not None:
        return cached
    from app.tasks.review_export import _probe
    dims = {}
    for f in artifacts.load_job(job_id, "files") or []:
        if os.path.exists(f["path"]):
            p = _probe(f["path"])
            dims[f["filename"]] = [p["width"], p["height"]]
        else:
            logger.warning(f"[LAYOUT] job {job_id}: source {f['path']} missing, no preview shape for it")
    artifacts.save_job(job_id, "source_dims", dims)
    return dims


def start_render(job_id: str, key: str, captions: Optional[str] = None) -> bool:
    """Start rendering a layout (with captions in that preset, if given) in the background. False when that
    render is already running. The clean 16:9 is the main render and is never rendered here."""
    _check_captions(captions)
    if key not in LAYOUTS or (key == MAIN and not captions):
        raise ValueError(f"no layout to render: {key!r}")
    with _lock:
        if (job_id, key, captions) in _running:
            return False
        _running[(job_id, key, captions)] = time.time()
    threading.Thread(target=_render, args=(job_id, key, captions), daemon=True,
                     name=f"layout-{key}{'-' + captions if captions else ''}-{job_id[:8]}").start()
    return True


def _render(job_id: str, key: str, captions: Optional[str] = None) -> None:
    from app.tasks.validate import make_report, validate_render
    from app.utils.ffmpeg import get_video_info, media_durations, render_compiled
    w, h = LAYOUTS[key]["size"]
    path, t0 = layout_path(job_id, key, captions), time.time()
    tmp = path[:-4] + ".part.mp4"
    timeline_d = artifacts.load_job(job_id, "timeline")
    fp, ass_dir = _fingerprint(timeline_d), None
    try:
        fp = _variant_fingerprint(job_id, timeline_d, captions)
        ass = None
        if captions:
            cues = subtitles.job_cues(job_id)
            if not cues:
                raise ValueError("the edit has no spoken words to caption")
            ass_dir = tempfile.mkdtemp(prefix="vf-subs-")
            ass = os.path.join(ass_dir, "captions.ass")
            with open(ass, "w", encoding="utf-8") as f:
                f.write(subtitles.build_ass(cues, captions, (w, h), subtitles.job_placement(job_id)))
            logger.info(f"[LAYOUT] job {job_id}: {key} captions '{captions}': {len(cues)} blocks of {subtitles.BLOCK_SEC:g}s")
        files = {f["filename"]: f["path"] for f in artifacts.load_job(job_id, "files") or []}
        timeline = CompiledTimeline(**timeline_d)
        render_compiled([s.model_dump() for s in timeline.segments], files, tmp, fps=timeline.fps,
                        join_fade=timeline.join_fade_sec, head_fade=timeline.head_fade_sec,
                        tail_fade=timeline.tail_fade_sec, size=(w, h), overlays=timeline.overlays, subtitles=ass)
        video = next(s for s in get_video_info(tmp)["streams"] if s.get("codec_type") == "video")
        if (video["width"], video["height"]) != (w, h):          # post-condition: the frame we asked for
            raise RuntimeError(f"rendered {video['width']}x{video['height']}, expected {w}x{h}")
        d = media_durations(tmp)
        report = make_report(validate_render(timeline, d["video"], d["audio"]))
        if report.status == "fail":
            raise RuntimeError("render checks failed: " + "; ".join(c.detail for c in report.checks if c.status == "fail"))
        os.replace(tmp, path)
        artifacts.save_job(job_id, _store(key, captions), {"status": "ready", "fingerprint": fp, "size": [w, h],
                                                           "validation": report.model_dump(), "render_sec": round(time.time() - t0, 1)})
        logger.info(f"[LAYOUT] job {job_id}: {key} {w}x{h}{' captions ' + captions if captions else ''} "
                    f"rendered in {time.time() - t0:.1f}s -> {path}")
    except Exception as e:
        logger.error(f"[LAYOUT] job {job_id}: {key}{' captions ' + captions if captions else ''} render failed: {e}", exc_info=True)
        artifacts.save_job(job_id, _store(key, captions), {"status": "failed", "fingerprint": fp, "error": str(e)[:500]})
        if os.path.exists(tmp):
            os.remove(tmp)
    finally:
        if ass_dir:
            shutil.rmtree(ass_dir, ignore_errors=True)
        with _lock:
            _running.pop((job_id, key, captions), None)
