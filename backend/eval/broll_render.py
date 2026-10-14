"""Phase 10 V4 check: rebuild a synthetic project through the real compiler with B-roll over its voice-over.

The speech plays in the ORIGINAL order (story order is measured separately), every word kept. Renders twice,
with and without overlays: the speech audio must be byte-identical. Writes, next to the project (local only):
  render_broll.mp4   the rebuilt edit with B-roll
  compare.mp4        original window (left) | rebuilt (right), for review by eye
  broll_report.json  overlays + planner report

Usage (from backend/):
    PYTHONPATH=. python -m eval.broll_render nz_travel_w1
"""

import hashlib
import json
import os
import subprocess
import sys

from app.models import EditPlan, EditSegment
from app.tasks import broll
from app.tasks.compiler import compile_and_render
from app.tasks.moments import build_moments, clip_order
from app.tasks.word_grid import build_word_grid
from app.utils.ffmpeg import get_ffmpeg_path, is_audio_only
from app.utils.speech_activity import envelope, load_audio
from eval.broll_gold import SRC_DIR
from eval.broll_synth import SYNTH_DIR
from eval.voiceover_eval import project_words


def _pcm_md5(path: str) -> str:
    raw = subprocess.run([get_ffmpeg_path(), "-v", "error", "-i", path, "-vn", "-f", "s16le", "-"], check=True,
                         capture_output=True).stdout
    return hashlib.md5(raw).hexdigest()


def main(proj: str) -> None:
    d = os.path.join(SYNTH_DIR, proj)
    g = json.load(open(os.path.join(d, "gold.json")))
    words = project_words(proj)
    grid = build_word_grid([{**w, "video_file": f} for f, ws in words.items() for w in ws])
    files = [{"filename": f, "path": os.path.join(d, "files", f)} for f in g["upload_order"]]
    for f in files:
        f["audio_only"] = is_audio_only(f["path"])
    file_map = {f["filename"]: f["path"] for f in files}
    by_file = {}
    for w in grid.words:
        by_file.setdefault(w.source_file, []).append(w)
    order = [t["file"] for t in g["timeline"] if t["file"] in by_file]
    plan = EditPlan(segments=[EditSegment(word_start=by_file[f][0].id, word_end=by_file[f][-1].id, reason="original")
                              for f in order], grid_fingerprint=grid.fingerprint())
    envs = {f: envelope(load_audio(os.path.join(d, "work", os.path.splitext(f)[0] + ".wav"))) for f in by_file}
    speech_files = [f for f in files if f["filename"] in by_file]
    moments = build_moments(grid, {w.id for w in grid.words}, clip_order(speech_files))
    plain, with_broll = os.path.join(d, "render_plain.mp4"), os.path.join(d, "render_broll.mp4")
    compile_and_render(plan, grid, envs, file_map, plain)
    timeline, info = compile_and_render(plan, grid, envs, file_map, with_broll,
                                        overlay_planner=lambda tl: broll.plan_overlays(tl, grid, files, moments))
    same = _pcm_md5(plain) == _pcm_md5(with_broll)
    rep = info["broll"]
    with open(os.path.join(d, "broll_report.json"), "w") as fh:
        json.dump({"report": rep, "overlays": timeline.overlays, "audio_identical": same}, fh, indent=1)
    vo_sec = sum(s.src_out - s.src_in for s in timeline.segments if g["category"][s.source_file] == "voice_over")
    print(f"{proj}: edit {timeline.duration_sec:.1f}s, voice-over {vo_sec:.1f}s, overlays {len(timeline.overlays)} "
          f"covering {rep.get('covered_sec')}s, needs footage {rep.get('needs_footage')}, library {rep.get('library')} shots, "
          f"speech audio identical with/without B-roll: {same}, validation {info['validation']['status']}")
    for w in rep["warnings"]:
        print("  ", w)
    t0 = g["timeline"][0]["src_start"]
    src = os.path.join(SRC_DIR, f"{g['clip']}.mp4")
    subprocess.run([get_ffmpeg_path(), "-v", "error", "-y", "-ss", f"{t0:.3f}", "-t", f"{g['duration']:.3f}", "-i", src,
                    "-i", with_broll, "-filter_complex",
                    "[0:v]scale=640:360,setsar=1,drawtext=text='original':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5[l];"
                    "[1:v]scale=640:360,setsar=1,drawtext=text='VlogForge':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5[r];"
                    "[l][r]hstack[v]", "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "23", "-c:a", "aac", "-shortest", os.path.join(d, "compare.mp4")], check=True)
    print("  compare:", os.path.join(d, "compare.mp4"))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
