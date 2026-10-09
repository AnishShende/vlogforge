"""EDL / FCPXML export of the last render (review screen). Temp artifact store, no media files."""
import xml.dom.minidom

import pytest

from app.config import settings
from app.tasks import review_export
from app.utils import artifacts


@pytest.fixture(autouse=True)
def job(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "artifact_dir", str(tmp_path / "artifacts"))
    monkeypatch.setattr(settings, "output_dir", str(tmp_path / "outputs"))
    monkeypatch.setattr(review_export, "_probe", lambda path: {"width": 1080, "height": 1920, "rate": "60/1", "duration": 300.0, "timecode": None})
    artifacts.save_job("j", "files", [{"filename": "IMG_1.MOV", "path": "/media/IMG 1.MOV", "audio_path": "a.wav"}])
    seg = lambda ids, a, b: {"source_file": "IMG_1.MOV", "word_ids": ids, "src_in": a, "src_out": b}
    artifacts.save_job("j", "timeline", {"segments": [seg(["w0"], 74.32, 78.3867), seg(["w5"], 10.0, 10.5)], "duration_sec": 4.567})
    artifacts.save_job("j", "moments", {"moments": [{"id": "m1", "function": "hook", "summary": "Opens <big> & bold", "word_ids": ["w0"]}]})


def test_edl_events_are_the_rendered_frames():
    edl = review_export.export_edl("j", "My vlog").splitlines()
    assert edl[:2] == ["TITLE: My vlog", "FCM: NON-DROP FRAME"]
    # 74.32 s = frame 2230; 4.0667 s = 122 frames; record starts at 01:00:00:00 and runs on
    assert edl[3] == "001  IMG_1    AA/V  C        00:01:14:10 00:01:18:12 01:00:00:00 01:00:04:02"
    assert edl[5] == "* COMMENT: Opens <big> & bold"
    assert edl[7] == "002  IMG_1    AA/V  C        00:00:10:00 00:00:10:15 01:00:04:02 01:00:04:17"


def test_fcpxml_is_valid_xml_with_one_clip_per_event():
    doc = xml.dom.minidom.parseString(review_export.export_fcpxml("j", "My vlog"))
    clips = doc.getElementsByTagName("asset-clip")
    assert [(c.getAttribute("offset"), c.getAttribute("start"), c.getAttribute("duration")) for c in clips] == [
        ("0s", "223/3s", "61/15s"), ("61/15s", "10s", "1/2s")]
    assert clips[0].getAttribute("name") == "Opens <big> & bold"
    assert doc.getElementsByTagName("media-rep")[0].getAttribute("src") == "file:///media/IMG%201.MOV"
    assert doc.getElementsByTagName("sequence")[0].getAttribute("duration") == "137/30s"


def test_no_render_no_export():
    assert review_export.export_edl("missing", "x") is None and review_export.export_fcpxml("missing", "x") is None
    assert review_export.output_waveform("j") is None              # no output file
