"""Compiled-timeline render (Archdoc roadmap Phase 2 / S4): A/V length stays equal across many cuts.

Regression for an earlier multi-pass render, whose 75 ms acrossfade per join made audio
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


def test_voice_over_audio_file_gets_a_black_picture_of_its_exact_length(tmp_path):
    _source(tmp_path / "a.mp4", 30, 48000, 1)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=220:duration=8:sample_rate=44100",
                    "-c:a", "aac", str(tmp_path / "vo.m4a")], check=True)
    files = {"a.mp4": str(tmp_path / "a.mp4"), "vo.m4a": str(tmp_path / "vo.m4a")}
    segs = [{"source_file": "a.mp4", "src_in": 0.5, "src_out": 0.5 + 45 / 30},
            {"source_file": "vo.m4a", "src_in": 1.0, "src_out": 1.0 + 60 / 30},
            {"source_file": "a.mp4", "src_in": 4.0, "src_out": 4.0 + 30 / 30}]
    out = tmp_path / "out.mp4"
    render_compiled(segs, files, str(out))
    d = _durations(out)
    assert int(d["video"]["nb_frames"]) == 135
    assert abs(float(d["audio"]["duration"]) - 4.5) < 1 / 30
    mid = subprocess.run(["ffmpeg", "-v", "error", "-ss", "2.5", "-i", str(out), "-frames:v", "1", "-vf", "scale=8:8",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], check=True, capture_output=True).stdout
    assert max(mid) < 20                                 # inside the voice-over: black picture


def _frame_rgb(path, t):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1", "-vf", "scale=4:4",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    px = [raw[i:i + 3] for i in range(0, len(raw), 3)]
    return tuple(sum(p[c] for p in px) // len(px) for c in range(3))


def _pcm(path):
    return subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-f", "s16le", "-"], check=True,
                          capture_output=True).stdout


def test_broll_overlay_covers_only_its_slot_and_never_touches_the_sound(tmp_path):
    _source(tmp_path / "a.mp4", 30, 48000, 1)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=320x180:r=25:d=3",
                    "-c:v", "libx264", str(tmp_path / "red.mp4")], check=True)       # 3 s shot, shorter than its slot
    files = {"a.mp4": str(tmp_path / "a.mp4"), "red.mp4": str(tmp_path / "red.mp4")}
    segs = [{"source_file": "a.mp4", "src_in": 0.5, "src_out": 0.5 + 90 / 30},
            {"source_file": "a.mp4", "src_in": 5.0, "src_out": 5.0 + 90 / 30}]
    plain, over = tmp_path / "plain.mp4", tmp_path / "over.mp4"
    render_compiled(segs, files, str(plain))
    render_compiled(segs, files, str(over), overlays=[{"source_file": "red.mp4", "src_in": 0.0,
                                                       "rec_in": 2.0, "rec_out": 5.5}])
    assert _pcm(plain) == _pcm(over)                     # speech audio byte-identical
    r, g, b = _frame_rgb(over, 3.0)
    assert r > 180 and g < 60 and b < 60                 # inside the slot: the B-roll
    r, g, b = _frame_rgb(over, 5.2)
    assert r > 180 and g < 60                            # past the shot's end: its last frame held
    assert _frame_rgb(over, 1.0) == _frame_rgb(plain, 1.0)   # before the slot: the main picture
    assert _frame_rgb(over, 5.8) == _frame_rgb(plain, 5.8)   # after the slot
    assert int(_durations(over)["video"]["nb_frames"]) == 180
