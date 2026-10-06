"""Artifact store + transcription reuse (roadmap Phase 5). Temp dir only, no real models."""
import pytest

from app.config import settings
from app.tasks import transcribe
from app.tasks.recompile import can_recompile, recompile_job
from app.models import EditPlan
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
