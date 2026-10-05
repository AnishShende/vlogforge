"""JEV segment-classification mock cache (eval determinism). Fake client, no network."""
from types import SimpleNamespace as NS

import pytest

from app.config import settings
from app.utils import jev


class FakeClient:
    def __init__(self):
        self.calls = 0

    def system_one(self, state, questions):
        self.calls += 1
        q = 0.5 + 0.01 * self.calls          # live answers drift between calls, like the real model
        return NS(
            choices={"segment_type": NS(choice="SPEECH", confidence=0.9)},
            nouls={"is_bad_take": NS(noul=0.1), "is_background_noise": NS(noul=0.2),
                   "has_structural_cue": NS(noul=0.0)},
            scores={"content_quality": NS(score=q * (jev.QUALITY_SCORE_LEVELS - 1), confidence=0.8)},
            model="fake-jev",
        )


SEG = {"clip_id": "c1", "transcript": "hello there", "visual_description": "a kitchen", "duration_sec": 3.0}


@pytest.fixture
def fake(monkeypatch, tmp_path):
    client = FakeClient()
    monkeypatch.setattr(jev, "_get_jev_client", lambda: client)
    monkeypatch.setattr(settings, "mock_llm_dir", str(tmp_path))
    return client


def test_cache_records_then_replays_identically(fake, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "enable_mock_jev", True)
    a = jev.classify_segment_jev(SEG, context_doc="ctx", rolling_window="rw")
    b = jev.classify_segment_jev(SEG, context_doc="ctx", rolling_window="rw")
    assert fake.calls == 1 and a == b
    assert len(list(tmp_path.glob("jev_segment_*.json"))) == 1


def test_cache_key_covers_full_state(fake, monkeypatch):
    monkeypatch.setattr(settings, "enable_mock_jev", True)
    jev.classify_segment_jev(SEG, context_doc="ctx", rolling_window="rw")
    jev.classify_segment_jev(SEG, context_doc="ctx", rolling_window="different window")
    jev.classify_segment_jev({**SEG, "visual_description": "a street"}, context_doc="ctx", rolling_window="rw")
    assert fake.calls == 3


def test_no_cache_when_mock_disabled(fake, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "enable_mock_jev", False)
    a = jev.classify_segment_jev(SEG)
    b = jev.classify_segment_jev(SEG)
    assert fake.calls == 2 and a != b
    assert not list(tmp_path.glob("jev_segment_*.json"))
