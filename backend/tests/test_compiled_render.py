"""Compiled-timeline render (Archdoc roadmap Phase 2 / S4): A/V length stays equal across many cuts.

Regression for the legacy multi-pass path, whose 75 ms acrossfade per join made audio
0.76 s shorter than video over 12 clips. Needs ffmpeg on PATH (synthetic sources, ~10 s).
"""
import json
import shutil
import subprocess

import pytest

from app.utils.ffmpeg import render_compiled

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


def _source(path, rate, sr, channels):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=640x360:rate={rate}:duration=12",
                    "-f", "lavfi", "-i", f"sine=f=440:duration=12:sample_rate={sr}", "-ac", str(channels),
                    "-c:v", "libx264", "-c:a", "aac", "-shortest", str(path)], check=True)


def _durations(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration,nb_frames,sample_rate",
                          "-of", "json", str(path)], check=True, capture_output=True).stdout
    return {s["codec_type"]: s for s in json.loads(out)["streams"]}


def test_twelve_cuts_keep_audio_and_video_the_same_length(tmp_path):
    _source(tmp_path / "a.mp4", 60, 48000, 1)          # phone-like: 60 fps, mono 48 kHz
    _source(tmp_path / "b.mp4", 25, 44100, 2)          # 25 fps, stereo 44.1 kHz
    files = {"a.mp4": str(tmp_path / "a.mp4"), "b.mp4": str(tmp_path / "b.mp4")}
    segs, expected = [], 0.0
    for i in range(12):
        frames = 10 + 3 * i                              # whole frames, as the compiler emits
        start = 0.37 * i + 0.013
        segs.append({"source_file": "a.mp4" if i % 2 else "b.mp4", "src_in": start, "src_out": start + frames / 30})
        expected += frames / 30
    out = tmp_path / "out.mp4"
    info = render_compiled(segs, files, str(out), head_fade=0.05, tail_fade=0.05)
    d = _durations(out)
    v, a = float(d["video"]["duration"]), float(d["audio"]["duration"])
    assert info["duration_sec"] == pytest.approx(expected)
    assert int(d["video"]["nb_frames"]) == round(expected * 30)
    assert abs(v - expected) < 1 / 30
    assert abs(a - v) < 1 / 30, f"audio {a:.3f}s vs video {v:.3f}s"
    assert d["audio"]["sample_rate"] == "48000"
