"""Waveform edge snapping (Archdoc Phase 0 / S2). Synthetic audio, no media files."""
import numpy as np
import pytest

from eval.gold import Gold
from eval.snap_gold import SR, envelope, snap, snap_end, snap_start, unaccounted

rng = np.random.default_rng(0)


def _audio(bursts, total=10.0, noise=1e-4):
    """Low noise floor with loud bursts at [(start, end), ...] seconds."""
    a = rng.normal(0, noise, int(total * SR)).astype(np.float32)
    for s, e in bursts:
        a[int(s * SR): int(e * SR)] += rng.normal(0, 0.2, int(e * SR) - int(s * SR))
    return a


ENV = envelope(_audio([(1.0, 2.0), (3.0, 4.0), (4.03, 4.3)]), method="energy")   # 30ms gap inside the 3-4.3s "word"


def test_start_on_silence_shrinks_inward_to_onset():
    t, flag = snap_start(ENV, 0.7, 2.0)            # word parked on silence before speech
    assert flag == "moved-in" and t == pytest.approx(1.0, abs=0.02)


def test_start_on_silence_far_from_speech_still_shrinks():
    t, flag = snap_start(ENV, 0.1, 2.0)            # 0.9s away: beyond SEARCH_SEC, bounded by span
    assert flag == "moved-in" and t == pytest.approx(1.0, abs=0.02)


def test_start_on_silence_with_no_speech_in_span_is_flagged():
    t, flag = snap_start(ENV, 2.2, 2.8)
    assert flag == "no-speech" and t == 2.2


def test_start_inside_speech_moves_out_to_onset():
    t, flag = snap_start(ENV, 1.3, 2.0)            # aligner started the word late
    assert flag == "moved-out" and t == pytest.approx(1.0, abs=0.02)


def test_start_on_previous_words_tail_moves_in_past_it():
    # speech 1.0-2.0 | silence 2.0-2.4 | speech 2.4-4.0 ; start parked at 1.85 (tail of prior word)
    env = envelope(_audio([(1.0, 2.0), (2.4, 4.0)]), method="energy")
    t, flag = snap_start(env, 1.85, 4.0)
    assert flag == "moved-in-past-speech" and t == pytest.approx(2.4, abs=0.02)


def test_end_stretched_over_silence_shrinks_inward():
    t, flag = snap_end(ENV, 2.4, 1.0)              # word stretched 0.4s into silence
    assert flag == "moved-in" and t == pytest.approx(2.0, abs=0.02)


def test_end_far_into_silence_shrinks_inward():
    t, flag = snap_end(ENV, 2.9, 1.0)              # 0.9s of stretch, beyond SEARCH_SEC
    assert flag == "moved-in" and t == pytest.approx(2.0, abs=0.02)


def test_short_internal_gap_is_not_a_boundary():
    t, _ = snap_end(ENV, 3.9, 3.0)                 # must skip the 30ms gap at 4.0s
    assert t == pytest.approx(4.3, abs=0.02)


def test_no_boundary_within_search_window_is_flagged():
    env = envelope(_audio([(1.0, 9.0)]), method="energy")
    t, flag = snap_start(env, 5.0, 8.0)
    assert flag == "no-boundary" and t == 5.0


def _gold(keep, exclude=()):
    return Gold.model_validate(dict(
        clip_id="c", status="draft", sources=[{"file": "a", "path": "a", "sha256": "0"}],
        keep=[{"id": f"k{i}", "source_file": "a", "start": s, "end": e, "text": "t"} for i, (s, e) in enumerate(keep)],
        exclude=[{"id": f"x{i}", "source_file": "a", "start": s, "end": e, "text": "t", "reason": "retake"}
                 for i, (s, e) in enumerate(exclude)]))


def test_snap_moves_all_edges_and_reports():
    g = _gold([(0.8, 2.3)], [(3.2, 4.6)])
    rows = snap(g, {"a": ENV})
    assert g.keep[0].start == pytest.approx(1.0, abs=0.02) and g.keep[0].end == pytest.approx(2.0, abs=0.02)
    assert g.exclude[0].start == pytest.approx(3.0, abs=0.02) and g.exclude[0].end == pytest.approx(4.3, abs=0.02)
    assert len(rows) == 2


def test_unaccounted_reports_uncovered_speech_only():
    g = _gold([(1.0, 2.0)])
    gaps = unaccounted(g, {"a": ENV})
    assert len(gaps) == 1 and gaps[0][1] == pytest.approx(3.0, abs=0.02) and gaps[0][2] == pytest.approx(4.3, abs=0.02)


def test_vad_method_ignores_loud_non_speech_noise():
    # loud broadband noise (e.g. sizzling) is "speech" to the energy method, not to VAD
    a = _audio([(1.0, 3.0)])
    assert envelope(a, method="energy").speech.mean() > 0.15
    assert envelope(a, method="vad").speech.mean() < 0.02


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="unknown speech detection method"):
        envelope(_audio([]), method="nope")


def test_edge_in_speech_prefers_outward_even_when_inward_silence_is_nearer():
    # span's own word 1.0-1.5, short pause 1.5-1.65, more own speech 1.65-3.0; previous
    # silence ends at 1.0. Edge at 1.4: inward silence (0.1s) is nearer than outward (0.4s)
    # but moving inward would drop the span's first word.
    env = envelope(_audio([(1.0, 1.5), (1.65, 3.0)]), method="energy")
    t, flag = snap_start(env, 1.4, 3.0)
    assert flag == "moved-out" and t == pytest.approx(1.0, abs=0.02)


def test_no_inward_jump_when_outside_speech_is_unlabelled():
    env = envelope(_audio([(1.0, 2.0), (2.4, 4.0)]), method="energy")
    t, flag = snap_start(env, 1.85, 4.0, outside_labelled=False)
    assert flag == "no-boundary" and t == 1.85


def test_snap_jumps_inward_only_past_a_labelled_neighbour():
    env = envelope(_audio([(1.0, 2.0), (2.4, 4.0)]), method="energy")
    g = _gold([(1.85, 4.0)], [(0.5, 1.85)])  # neighbour x0 touches k0 and covers the speech outside it
    snap(g, {"a": env})
    assert g.keep[0].start == pytest.approx(2.4, abs=0.02)


def test_hand_set_edges_are_never_snapped():
    g = _gold([(0.8, 2.3)])
    g.keep[0].edges = "hand"
    rows = snap(g, {"a": ENV})
    assert (g.keep[0].start, g.keep[0].end) == (0.8, 2.3) and rows[0]["start_flag"] == "hand"
