"""Export layouts (export panel): the last render's edit in standard frame shapes.

Each layout is its own MP4, rendered on demand from the stored timeline (same cuts, audio and
fades as the main render; only the frame size differs, each clip fitted with black bars).
The main render already is 16:9 1920x1080, so that layout is the main output, not a copy.
A layout file remembers the timeline it was made from: after a re-compile it shows as outdated.
Renders run in a background thread, one per (job, layout) at a time; status lives in the artifact
store (ready / failed) plus an in-process set (rendering)."""

import hashlib
import json
import logging
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

from app.config import settings
from app.models import CompiledTimeline
from app.utils import artifacts

logger = logging.getLogger("VlogForge.Layouts")

LAYOUTS = {
    "16x9": {"ratio": "16:9", "name": "Landscape", "size": (1920, 1080), "for": "YouTube"},
    "9x16": {"ratio": "9:16", "name": "Vertical", "size": (1080, 1920), "for": "Reels, Shorts, TikTok"},
    "1x1": {"ratio": "1:1", "name": "Square", "size": (1080, 1080), "for": "Instagram / Facebook feed, LinkedIn"},
    "4x5": {"ratio": "4:5", "name": "Portrait", "size": (1080, 1350), "for": "Instagram feed"},
}
MAIN = "16x9"               # what the main render is (app.utils.ffmpeg.RENDER_SIZE)

_running: Dict[Tuple[str, str], float] = {}
_lock = threading.Lock()


def main_path(job_id: str) -> str:
    return os.path.join(settings.output_dir, f"{job_id}.mp4")


def layout_path(job_id: str, key: str) -> str:
    return main_path(job_id) if key == MAIN else os.path.join(settings.output_dir, f"{job_id}_{key}.mp4")


def _fingerprint(timeline: Dict) -> str:
    return hashlib.sha256(json.dumps([(s["source_file"], s["src_in"], s["src_out"]) for s in timeline["segments"]]).encode()).hexdigest()[:16]


def list_layouts(job_id: str) -> Optional[List[Dict]]:
    """Every layout with its status: ready | rendering | stale (made from an older edit) | failed | none."""
    timeline = artifacts.load_job(job_id, "timeline")
    if timeline is None:
        return None
    fp, out = _fingerprint(timeline), []
    for key, L in LAYOUTS.items():
        row = {"key": key, **L, "size": list(L["size"]), "error": None}
        stored = artifacts.load_job(job_id, f"layout_{key}") or {}
        if key == MAIN:
            row["status"] = "ready" if os.path.exists(main_path(job_id)) else "none"
        elif (job_id, key) in _running:
            row["status"] = "rendering"
        elif not stored:
            row["status"] = "none"
        elif stored["fingerprint"] != fp:
            row["status"] = "stale"
        elif stored["status"] == "failed":
            row.update(status="failed", error=stored.get("error"))
        else:
            row["status"] = "ready" if os.path.exists(layout_path(job_id, key)) else "none"
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


def start_render(job_id: str, key: str) -> bool:
    """Start rendering a layout in the background. False when that render is already running."""
    if key not in LAYOUTS or key == MAIN:
        raise ValueError(f"no layout to render: {key!r}")
    with _lock:
        if (job_id, key) in _running:
            return False
        _running[(job_id, key)] = time.time()
    threading.Thread(target=_render, args=(job_id, key), daemon=True, name=f"layout-{key}-{job_id[:8]}").start()
    return True


def _render(job_id: str, key: str) -> None:
    from app.tasks.validate import make_report, validate_render
    from app.utils.ffmpeg import get_video_info, media_durations, render_compiled
    w, h = LAYOUTS[key]["size"]
    path, t0 = layout_path(job_id, key), time.time()
    tmp = path[:-4] + ".part.mp4"
    timeline_d = artifacts.load_job(job_id, "timeline")
    fp = _fingerprint(timeline_d)
    try:
        files = {f["filename"]: f["path"] for f in artifacts.load_job(job_id, "files") or []}
        timeline = CompiledTimeline(**timeline_d)
        render_compiled([s.model_dump() for s in timeline.segments], files, tmp, fps=timeline.fps,
                        join_fade=timeline.join_fade_sec, head_fade=timeline.head_fade_sec,
                        tail_fade=timeline.tail_fade_sec, size=(w, h))
        video = next(s for s in get_video_info(tmp)["streams"] if s.get("codec_type") == "video")
        if (video["width"], video["height"]) != (w, h):          # post-condition: the frame we asked for
            raise RuntimeError(f"rendered {video['width']}x{video['height']}, expected {w}x{h}")
        d = media_durations(tmp)
        report = make_report(validate_render(timeline, d["video"], d["audio"]))
        if report.status == "fail":
            raise RuntimeError("render checks failed: " + "; ".join(c.detail for c in report.checks if c.status == "fail"))
        os.replace(tmp, path)
        artifacts.save_job(job_id, f"layout_{key}", {"status": "ready", "fingerprint": fp, "size": [w, h],
                                                     "validation": report.model_dump(), "render_sec": round(time.time() - t0, 1)})
        logger.info(f"[LAYOUT] job {job_id}: {key} {w}x{h} rendered in {time.time() - t0:.1f}s -> {path}")
    except Exception as e:
        logger.error(f"[LAYOUT] job {job_id}: {key} render failed: {e}", exc_info=True)
        artifacts.save_job(job_id, f"layout_{key}", {"status": "failed", "fingerprint": fp, "error": str(e)[:500]})
        if os.path.exists(tmp):
            os.remove(tmp)
    finally:
        with _lock:
            _running.pop((job_id, key), None)
