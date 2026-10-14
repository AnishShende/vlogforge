"""Burned-in subtitles: words on the output clock, fixed 1 s blocks, ASS per preset and frame, burn-in render."""
import shutil
import subprocess

import pytest

from app.tasks import subtitles
from app.utils.ffmpeg import render_compiled


def _w(i, text, start, end, f="a.mov"):
    return {"id": f"w{i}", "text": text, "source_file": f, "start": start, "end": end}


def test_words_follow_the_cuts_on_the_output_clock():
    words = [_w(0, "hello", 10.0, 10.4), _w(1, "cut", 10.5, 10.8), _w(2, "there", 20.2, 20.6)]
    timeline = {"fps": 30, "segments": [{"source_file": "a.mov", "src_in": 9.9, "src_out": 10.45, "word_ids": ["w0"]},
                                        {"source_file": "a.mov", "src_in": 20.0, "src_out": 21.0, "word_ids": ["w2"]}]}
    out = subtitles.output_words(timeline, words)
    first = round(0.55 * 30) / 30                        # whole-frame length of segment 1
    assert [w["text"] for w in out] == ["hello", "there"]   # a dropped word is never captioned
    assert out[0]["start"] == pytest.approx(0.1) and out[0]["end"] == pytest.approx(0.5)
    assert out[1]["start"] == pytest.approx(first + 0.2) and out[1]["end"] == pytest.approx(first + 0.6)
    with pytest.raises(KeyError):
        subtitles.output_words({"fps": 30, "segments": [{"src_in": 0, "src_out": 1, "word_ids": ["gone"]}]}, words)


def test_fixed_blocks_by_word_midpoint_and_silent_seconds_show_nothing():
    words = [{"text": "so", "start": 0.1, "end": 0.3}, {"text": "i", "start": 0.4, "end": 0.6},
             {"text": "went", "start": 0.8, "end": 1.4},           # midpoint 1.1 -> second block
             {"text": "there", "start": 1.5, "end": 1.9},
             {"text": "again", "start": 3.2, "end": 3.6}]          # 2-3 s: silence
    assert subtitles.blocks(words, 3.5) == [(0.0, 1.0, "so I"), (1.0, 2.0, "went there"), (3.0, 3.5, "again")]
    assert subtitles._tidy("i think i'm fine in it") == "I think I'm fine in it"


def test_ass_scales_with_the_frame_and_each_preset_differs():
    cues = [(0.0, 1.0, "hello {world}"), (61.0, 62.0, "later")]
    land = subtitles.build_ass(cues, "clean", (1920, 1080))
    tall = subtitles.build_ass(cues, "clean", (1080, 1920))
    assert "PlayResX: 1920" in land and "PlayResY: 1920" in tall
    assert "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,hello (world)" in land     # braces never become tags
    assert "0:01:01.00,0:01:02.00" in land
    margin = lambda ass: int(next(l for l in ass.splitlines() if l.startswith("Style:")).split(",")[-2])
    assert margin(tall) == round(subtitles.PORTRAIT_MARGIN_V * 1920)                     # clear of Reels buttons
    bold = subtitles.build_ass(cues, "bold", (1920, 1080))
    assert "HELLO (WORLD)" in bold and r"\t(0,90" in bold and "Arial Black" in bold
    assert ",3," in next(l for l in subtitles.build_ass(cues, "boxed", (1920, 1080)).splitlines() if l.startswith("Style:"))
    assert len({subtitles.fingerprint(cues, k) for k in subtitles.STYLES}) == len(subtitles.STYLES)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_burned_in_captions_show_only_inside_their_block(tmp_path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=640x360:r=30:d=6",
                    "-f", "lavfi", "-i", "sine=f=440:duration=6", "-c:v", "libx264", "-c:a", "aac", "-shortest",
                    str(tmp_path / "a.mp4")], check=True)
    ass = tmp_path / "c.ass"
    ass.write_text(subtitles.build_ass([(2.0, 3.0, "caption here")], "boxed", (640, 360)))
    segs, files = [{"source_file": "a.mp4", "src_in": 0.0, "src_out": 5.0}], {"a.mp4": str(tmp_path / "a.mp4")}
    plain, cap = tmp_path / "plain.mp4", tmp_path / "cap.mp4"
    render_compiled(segs, files, str(plain), size=(640, 360))
    render_compiled(segs, files, str(cap), size=(640, 360), subtitles=str(ass))

    def frame(path, t):
        return subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
                               "-f", "rawvideo", "-pix_fmt", "gray", "-"], check=True, capture_output=True).stdout

    def diff(t):
        a, b = frame(plain, t), frame(cap, t)
        return sum(abs(x - y) > 40 for x, y in zip(a, b))

    assert diff(2.5) > 500                               # inside the block: text drawn
    assert diff(1.5) < 50 and diff(3.5) < 50             # outside it: same picture (encoder noise only)
    bad = tmp_path / "x'y.ass"
    bad.write_text(ass.read_text())
    with pytest.raises(ValueError):                      # filtergraph syntax in the path: refused, not mis-parsed
        render_compiled(segs, files, str(cap), size=(640, 360), subtitles=str(bad))


def test_placement_moves_and_scales_the_caption_and_is_clamped():
    cues = [(0.0, 1.0, "hi")]
    style = lambda ass: next(l for l in ass.splitlines() if l.startswith("Style:")).split(",")
    plain = subtitles.build_ass(cues, "bold", (1920, 1080))
    big = subtitles.build_ass(cues, "bold", (1920, 1080), {"scale": 2.0, "x": 0.25, "y": 0.5})
    assert abs(int(style(big)[2]) - 2 * int(style(plain)[2])) <= 1                         # font size doubled
    assert r"{\an2\pos(480,540)\fscx80" in big and r"\pos" not in plain                    # bottom centre at (x, y)
    assert subtitles.clean_placement({"scale": 99, "x": 1.7, "y": -1}) == {"scale": 3.0, "x": 1.0, "y": 0.0}
    assert subtitles.clean_placement({"scale": 1, "x": 0.5}) == subtitles.DEFAULT_PLACEMENT  # x without y: preset place
    assert subtitles.fingerprint(cues, "bold") != subtitles.fingerprint(cues, "bold", {"scale": 1.2})
