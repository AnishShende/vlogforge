"""Export layouts: status per layout (no rendering here; render_compiled is covered by test_compiled_render)."""
import os

import pytest

from app.config import settings
from app.tasks import layouts
from app.utils import artifacts


@pytest.fixture(autouse=True)
def job(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "artifact_dir", str(tmp_path / "artifacts"))
    monkeypatch.setattr(settings, "output_dir", str(tmp_path / "outputs"))
    os.makedirs(tmp_path / "outputs")
    artifacts.save_job("j", "timeline", {"segments": [{"source_file": "a.mov", "src_in": 1.0, "src_out": 2.0}]})


def status(job="j"):
    return {r["key"]: r["status"] for r in layouts.list_layouts(job)}


def test_statuses_follow_files_artifacts_and_the_current_edit():
    assert status() == {"16x9": "none", "9x16": "none", "1x1": "none", "4x5": "none"}
    open(layouts.main_path("j"), "w").close()                              # the main render is the 16:9 layout
    fp = layouts._fingerprint(artifacts.load_job("j", "timeline"))
    open(layouts.layout_path("j", "9x16"), "w").close()
    artifacts.save_job("j", "layout_9x16", {"status": "ready", "fingerprint": fp})
    artifacts.save_job("j", "layout_1x1", {"status": "failed", "fingerprint": fp, "error": "boom"})
    artifacts.save_job("j", "layout_4x5", {"status": "ready", "fingerprint": "older-edit"})
    assert status() == {"16x9": "ready", "9x16": "ready", "1x1": "failed", "4x5": "stale"}
    layouts._running[("j", "1x1")] = 0.0
    try:
        assert status()["1x1"] == "rendering" and layouts.start_render("j", "1x1") is False   # one render at a time
    finally:
        layouts._running.clear()
    artifacts.save_job("j", "timeline", {"segments": [{"source_file": "a.mov", "src_in": 1.0, "src_out": 2.5}]})  # re-compiled
    assert status()["9x16"] == "stale"


def test_main_layout_and_unknown_keys_are_not_rendered():
    with pytest.raises(ValueError):
        layouts.start_render("j", "16x9")
    with pytest.raises(ValueError):
        layouts.start_render("j", "21x9")
    assert layouts.list_layouts("missing") is None


def test_source_dims_probe_once_and_skip_missing_files(tmp_path, monkeypatch):
    from app.tasks import review_export
    src = tmp_path / "a.mov"
    src.write_bytes(b"")
    artifacts.save_job("j", "files", [{"filename": "a.mov", "path": str(src)}, {"filename": "gone.mov", "path": str(tmp_path / "gone.mov")}])
    calls = []
    monkeypatch.setattr(review_export, "_probe", lambda p: calls.append(p) or {"width": 1080, "height": 1920})
    assert layouts.source_dims("j") == {"a.mov": [1080, 1920]}
    assert layouts.source_dims("j") == {"a.mov": [1080, 1920]} and len(calls) == 1      # cached per job
