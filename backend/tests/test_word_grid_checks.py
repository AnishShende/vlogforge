"""Word grid speech-activity checks (Archdoc roadmap Phase 1 / S3). Synthetic audio."""
import numpy as np
import pytest

from app.models import WORD_FLAG_LONG_SPAN, WORD_FLAG_LOW_CONF, WORD_FLAG_ON_SILENCE
from app.tasks.word_grid import EDGE_REFINE_SEC, build_word_grid, check_word_grid, summarize_word_grid
from app.utils.speech_activity import SR, envelope, possible_speech

rng = np.random.default_rng(0)


def _audio(bursts, total=12.0):
    a = rng.normal(0, 1e-4, int(total * SR)).astype(np.float32)
    for s, e in bursts:
        a[int(s * SR): int(e * SR)] += rng.normal(0, 0.2, int(e * SR) - int(s * SR))
    return a


def _t(text, s, e, **kw):
    return {"text": text, "start": s, "end": e, "video_file": "a.mov", **kw}


# speech: 1.0-1.5 | 2.0-2.5 | 6.0-7.0 (untranscribed) ; silence elsewhere
ENV = envelope(_audio([(1.0, 1.5), (2.0, 2.5), (6.0, 7.0)]), method="energy")


def _check(transcript):
    return check_word_grid(build_word_grid(transcript), {"a.mov": ENV})


def test_edges_refined_within_window_toward_speech():
    g = _check([_t("late", 1.03, 1.5), _t("early", 2.0, 2.47)])
    assert g.words[0].start == pytest.approx(1.0, abs=0.011)      # moved back 30ms to the onset
    assert g.words[1].end == pytest.approx(2.5, abs=0.011)        # moved out 30ms to the offset
    assert g.checks["edges_refined"] == 2


def test_edge_moves_are_capped():
    g = _check([_t("parked", 0.5, 1.5)])                          # start 0.5s before the onset
    assert g.words[0].start == pytest.approx(0.5 + EDGE_REFINE_SEC)  # only 40ms, never the full 0.5s


def test_never_crosses_a_neighbour():
    g = _check([_t("a", 1.0, 1.3), _t("b", 1.3, 1.5)])            # no silence between them
    assert g.words[0].end <= g.words[1].start and g.validate_integrity() == []


def test_flags_on_silence_long_span_low_conf():
    g = _check([_t("ghost", 3.0, 3.4), _t("it", 1.0, 2.5, conf=0.9), _t("um", 2.0, 2.5, conf=0.1)])
    w = {x.text: x for x in g.words}
    assert WORD_FLAG_ON_SILENCE in w["ghost"].flags
    assert WORD_FLAG_LONG_SPAN in w["it"].flags                   # 1.5s for a 2-char word
    assert WORD_FLAG_LOW_CONF in w["um"].flags and WORD_FLAG_LOW_CONF not in w["it"].flags


def test_unaccounted_speech_is_reported():
    g = _check([_t("a", 1.0, 1.5), _t("b", 2.0, 2.5)])
    assert g.checks["unaccounted_speech"]["a.mov"] == [pytest.approx((6.0, 7.0), abs=0.02)]
    assert summarize_word_grid(g)["unaccounted_speech"]["regions"] == 1


def test_possible_speech_is_loud_but_not_vad():
    env = envelope(_audio([(1.0, 2.0)]), method="energy")
    env.speech = np.zeros_like(env.speech)                        # pretend VAD rejected the burst
    assert possible_speech(env, []) == [pytest.approx((1.0, 2.0), abs=0.02)]
    assert possible_speech(env, [(0.9, 2.1)]) == []               # covered by a word: not reported


def test_inter_word_pauses_are_not_unaccounted_speech():
    # continuous speech 1.0-2.5 with tight word edges leaving 0.2s gaps between words
    env = envelope(_audio([(1.0, 2.5), (6.0, 7.0)]), method="energy")
    g = check_word_grid(build_word_grid([_t("a", 1.0, 1.4), _t("b", 1.6, 2.0), _t("c", 2.2, 2.5)]), {"a.mov": env})
    assert g.checks["unaccounted_speech"]["a.mov"] == [pytest.approx((6.0, 7.0), abs=0.02)]
