"""Skipped-speech recovery (Archdoc roadmap Phase 1 / S5). Synthetic audio, fake ASR + aligner."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from app.tasks import asr_recovery
from app.tasks.asr_recovery import guarded_text, recover_skipped_speech
from app.utils.speech_activity import SR, envelope

rng = np.random.default_rng(0)


def _audio(bursts, total=10.0):
    a = rng.normal(0, 1e-4, int(total * SR)).astype(np.float32)
    for s, e in bursts:
        a[int(s * SR): int(e * SR)] += rng.normal(0, 0.2, int(e * SR) - int(s * SR))
    return a


@pytest.fixture(autouse=True)
def energy_vad(monkeypatch):
    # synthetic noise bursts are "speech" to the energy method only
    monkeypatch.setattr(asr_recovery, "envelope", lambda a: envelope(a, method="energy"))


AUDIO = _audio([(1.0, 2.0), (4.0, 5.0)])           # first pass only covered 1.0-2.0
FIRST = [{"start": 1.0, "end": 1.4, "text": "hello"}, {"start": 1.5, "end": 2.0, "text": "there"}]


def test_recovers_skipped_stretch_and_flags_words():
    seen = {}

    def clips_fn(audio, clips, language):
        seen["clips"], seen["lang"] = clips, language
        return ["again hello"]

    def align(path, segs):
        seen["segs"] = segs
        return [{"start": 4.0, "end": 4.4, "text": "again"}, {"start": 4.5, "end": 5.0, "text": "hello"}]

    out, stats = recover_skipped_speech("x.wav", FIRST, clips_fn, align, language="en", audio=AUDIO)
    assert seen["lang"] == "en" and len(seen["clips"]) == 1
    assert seen["clips"][0] == pytest.approx((3.8, 5.2), abs=0.03)          # stretch + PAD_SEC
    assert [w["text"] for w in out] == ["hello", "there", "again", "hello"]
    assert [bool(w.get("recovered")) for w in out] == [False, False, True, True]
    assert stats["recovered_words"] == 2 and stats["stretches"] == 1


def test_rejected_text_adds_nothing():
    out, stats = recover_skipped_speech("x.wav", FIRST, lambda a, c, l: [""], lambda p, s: pytest.fail("no align"),
                                        audio=AUDIO)
    assert out == FIRST and stats["with_text"] == 0


def test_recovered_words_never_overlap_first_pass_or_land_outside():
    def align(path, segs):
        return [{"start": 1.9, "end": 2.1, "text": "overlap"},     # outside stretch -> filtered
                {"start": 4.0, "end": 4.6, "text": "kept"},
                {"start": 7.0, "end": 7.5, "text": "far"}]          # outside stretch -> filtered
    out, _ = recover_skipped_speech("x.wav", FIRST, lambda a, c, l: ["x"], align, audio=AUDIO)
    assert [w["text"] for w in out] == ["hello", "there", "kept"]


def test_nothing_to_recover_when_fully_covered():
    covered = FIRST + [{"start": 4.0, "end": 5.0, "text": "all"}]
    out, stats = recover_skipped_speech("x.wav", covered, lambda a, c, l: pytest.fail("no asr"),
                                        lambda p, s: pytest.fail("no align"), audio=AUDIO)
    assert out == covered and stats["stretches"] == 0


def test_guarded_text_applies_hallucination_guards():
    segs = [NS(text=" real words ", no_speech_prob=0.1, avg_logprob=-0.3),
            NS(text="Thank you.", no_speech_prob=0.9, avg_logprob=-0.2),
            NS(text="Gracias", no_speech_prob=0.2, avg_logprob=-1.6)]
    assert guarded_text(segs) == "real words"
