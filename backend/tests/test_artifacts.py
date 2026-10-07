"""Artifact store + transcription reuse (roadmap Phase 5). Temp dir only, no real models."""
import pytest

from app.config import settings
from app.tasks import transcribe
from app.tasks.recompile import can_recompile, load_job_edit, recompile_job
from app.models import EditPlan, WordGrid
from app.utils import artifacts


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "artifact_dir", str(tmp_path / "artifacts"))
    monkeypatch.setattr(settings, "enable_artifact_cache", True)
    monkeypatch.setattr(settings, "enable_mock_whisper", False)
    return tmp_path


def test_get_put_and_job_store():
    assert artifacts.get("s", "k") is None
    artifacts.put("s", "k", {"a": [1, 2]})
    assert artifacts.get("s", "k") == {"a": [1, 2]}
    artifacts.save_job("job1", "plan", {"segments": []})
    assert artifacts.load_job("job1", "plan") == {"segments": []}
    assert artifacts.load_job("job2", "plan") is None


def test_digest_is_content_not_name(store):
    a, b, c = store / "a.wav", store / "renamed.wav", store / "other.wav"
    a.write_bytes(b"same audio"); b.write_bytes(b"same audio"); c.write_bytes(b"different!")
    assert artifacts.file_digest(str(a)) == artifacts.file_digest(str(b)) != artifacts.file_digest(str(c))


def test_transcription_is_reused_by_content(store, monkeypatch):
    calls = []
    monkeypatch.setattr(transcribe, "_transcribe_audio", lambda p, cb=None: calls.append(p) or [{"text": "hi", "start": 0, "end": 1}])
    a, b = store / "a.wav", store / "copy_of_a.wav"
    a.write_bytes(b"audio bytes"); b.write_bytes(b"audio bytes")
    assert transcribe.transcribe_audio(str(a)) == [{"text": "hi", "start": 0, "end": 1}]
    assert transcribe.transcribe_audio(str(b)) == [{"text": "hi", "start": 0, "end": 1}]   # same content: reused
    assert len(calls) == 1
    monkeypatch.setattr(settings, "enable_word_grid", not settings.enable_word_grid)      # other settings: new run
    transcribe.transcribe_audio(str(a))
    assert len(calls) == 2


def test_failed_transcription_is_not_stored(store, monkeypatch):
    calls = []
    monkeypatch.setattr(transcribe, "_transcribe_audio", lambda p, cb=None: calls.append(p) or [])
    a = store / "a.wav"
    a.write_bytes(b"audio")
    transcribe.transcribe_audio(str(a)); transcribe.transcribe_audio(str(a))
    assert len(calls) == 2


def test_recompile_needs_stored_job():
    assert not can_recompile("nope")
    with pytest.raises(LookupError):
        recompile_job("nope", EditPlan(), "/tmp/never.mp4")


def test_edit_view_maps_kept_words_to_output_time():
    words = [{"id": f"w{i}", "text": t, "source_file": "a.mov", "start": s, "end": s + 0.4}
             for i, (t, s) in enumerate([("one", 1.0), ("two", 2.0), ("three", 3.0)])]
    artifacts.save_job("j", "grid", {"words": words})
    artifacts.save_job("j", "files", [{"filename": "a.mov", "path": "a.mov", "audio_path": "a.wav"}])
    artifacts.save_job("j", "cleanup", [{"word_start": "w1", "word_end": "w1", "label": "remove", "reason": "retake"}])
    seg = lambda ids, a, b: {"word_ids": ids, "src_in": a, "src_out": b}
    artifacts.save_job("j", "timeline", {"segments": [seg(["w0"], 0.9, 1.5), seg(["w2"], 2.9, 3.5)], "duration_sec": 1.2})
    view = load_job_edit("j")
    assert [w["id"] for w in view["words"]] == ["w0", "w1", "w2"]
    assert view["word_out"] == {"w0": 0.1, "w2": 0.7}          # 0.6 s first segment, then 0.1 s into the second
    assert view["ranges"][0]["reason"] == "retake" and view["files"] == ["a.mov"]
    assert load_job_edit("missing") is None


def test_rerun_archives_a_different_stored_plan():
    from app.models import CompiledTimeline, EditSegment
    from app.tasks.recompile import save_job_edit
    grid = WordGrid(words=[])
    tl = CompiledTimeline(segments=[], fps=30, join_fade_sec=0, head_fade_sec=0, tail_fade_sec=0, duration_sec=0)
    user = EditPlan(segments=[EditSegment(word_start="a", word_end="b")])
    artifacts.save_job("j2", "plan", user.model_dump())
    save_job_edit("j2", grid, [], EditPlan(segments=[EditSegment(word_start="a", word_end="c")]), tl, {"validation": None})
    import os
    archived = [f for f in os.listdir(os.path.join(settings.artifact_dir, "jobs", "j2")) if f.startswith("plan.prev-")]
    assert len(archived) == 1 and artifacts.load_job("j2", archived[0][:-5]) == user.model_dump()
