"""Eval scoring (Archdoc Phase 0 / S3). Hermetic: synthetic gold + timelines."""
import numpy as np
import pytest

from eval.gold import Gold
from eval.score import Range, expected_text, score_text, score_timeline

GOLD = Gold.model_validate(dict(
    clip_id="c", status="reviewed", sources=[{"file": "a", "path": "a", "sha256": "0"}],
    keep=[{"id": "k1", "source_file": "a", "start": 10.0, "end": 12.0, "text": "first line"},
          {"id": "k2", "source_file": "a", "start": 20.0, "end": 23.0, "text": "second line here"}],
    exclude=[{"id": "x1", "source_file": "a", "start": 5.0, "end": 9.0, "text": "first li", "reason": "retake"}],
))


def _keep(res, kid):
    return next(r for r in res["keep"] if r["id"] == kid)


def test_perfect_edit():
    res = score_timeline(GOLD, [Range("a", 10.0, 12.0), Range("a", 20.0, 23.0)])
    s = res["summary"]
    assert s["keep_complete"] == 2 and s["exclude_leaked"] == 0 and s["keep_order_inversions"] == 0


def test_handles_around_keep_still_complete():
    res = score_timeline(GOLD, [Range("a", 9.85, 12.3), Range("a", 19.9, 23.2)])
    assert res["summary"]["keep_complete"] == 2
    assert res["exclude"][0]["leaked_sec"] == 0.0     # 9.85 > 9.0: no overlap with x1


def test_clipped_head_and_tail():
    res = score_timeline(GOLD, [Range("a", 10.2, 12.0), Range("a", 20.0, 22.8)])
    assert _keep(res, "k1")["status"] == "clipped" and _keep(res, "k1")["head_clip_sec"] == pytest.approx(0.2)
    assert _keep(res, "k2")["status"] == "clipped" and _keep(res, "k2")["tail_clip_sec"] == pytest.approx(0.2)


def test_interior_gap_is_clipped():
    res = score_timeline(GOLD, [Range("a", 10.0, 10.8), Range("a", 11.2, 12.0)])
    k = _keep(res, "k1")
    assert k["status"] == "clipped" and k["interior_gap_sec"] == pytest.approx(0.4)


def test_missing_and_duplicated():
    res = score_timeline(GOLD, [Range("a", 20.0, 23.0), Range("a", 20.0, 23.0)])
    assert _keep(res, "k1")["status"] == "missing" and _keep(res, "k2")["status"] == "duplicated"


def test_exclude_leak_threshold():
    small = score_timeline(GOLD, [Range("a", 8.95, 12.0)])
    big = score_timeline(GOLD, [Range("a", 7.0, 12.0)])
    assert not small["exclude"][0]["leaked"] and small["exclude"][0]["leaked_sec"] == pytest.approx(0.05)
    assert big["exclude"][0]["leaked"] and big["summary"]["exclude_leaked_sec"] == pytest.approx(2.0)


def test_other_source_file_never_counts():
    res = score_timeline(GOLD, [Range("b", 10.0, 12.0)])
    assert res["summary"]["keep_missing"] == 2


def test_order_inversion():
    res = score_timeline(GOLD, [Range("a", 20.0, 23.0), Range("a", 10.0, 12.0)])
    assert res["summary"]["keep_order_inversions"] == 1
    assert expected_text(GOLD, res) == "second line here first line"


def test_speech_metrics_cut_in_silence_and_unlabelled():
    hop = 0.01
    mask = np.zeros(3000, bool)
    mask[1000:1200] = True     # k1 speech 10-12
    mask[1300:1400] = True     # unlabelled speech 13-14
    res = score_timeline(GOLD, [Range("a", 9.9, 12.1), Range("a", 12.9, 14.0)], {"a": (mask, hop)})
    s = res["summary"]
    assert s["cut_edges"] == 4
    assert s["cut_edges_in_silence_pct"] == 75.0       # the 14.0 end is right at speech end
    assert s["unlabelled_speech_in_output_sec"] == pytest.approx(1.0)


def test_score_text_counts():
    r = score_text("so keep the lights low", "keep the light low low")
    assert r["missing"] == ["so"] and r["substituted"] == ["lights -> light"] and r["extra"] == ["low"]
    assert r["word_error_rate"] == pytest.approx(3 / 5)


def test_score_text_normalises_case_and_punctuation():
    assert score_text("Don't stop, ok?", "don't STOP ok")["word_error_rate"] == 0.0


def test_rendered_ranges_mirrors_assembly_overlap_clamp():
    from eval.run_eval import rendered_ranges
    edl = [{"source_file": "a", "start_sec": 1.0, "end_sec": 5.0},
           {"source_file": "a", "start_sec": 2.0, "end_sec": 4.0},     # contained -> dropped
           {"source_file": "a", "start_sec": 4.5, "end_sec": 8.0},     # overlaps -> trimmed to 5.0
           {"source_file": "a", "start_sec": 7.95, "end_sec": 8.02},   # < 0.1s after trim -> dropped
           {"source_file": "b", "start_sec": 0.0, "end_sec": 1.0}]
    assert [(r.source_file, r.start, r.end) for r in rendered_ranges(edl)] == [
        ("a", 1.0, 5.0), ("a", 5.0, 8.0), ("b", 0.0, 1.0)]


def test_optional_never_penalised_and_imperfect_drop_counted():
    g = Gold.model_validate(dict(
        clip_id="c", status="reviewed", sources=[{"file": "a", "path": "a", "sha256": "0"}],
        keep=[{"id": "k1", "source_file": "a", "start": 1.0, "end": 2.0, "text": "a", "quality": "imperfect"},
              {"id": "k2", "source_file": "a", "start": 3.0, "end": 4.0, "text": "b"}],
        optional=[{"id": "o1", "source_file": "a", "start": 5.0, "end": 6.0, "text": "ha"}]))
    res = score_timeline(g, [Range("a", 3.0, 4.0), Range("a", 5.0, 6.0)])
    s = res["summary"]
    assert s["keep_imperfect_dropped"] == 1 and s["optional_included"] == 1 and s["exclude_leaked"] == 0
    mask = np.ones(1000, bool)
    s2 = score_timeline(g, [Range("a", 5.0, 6.0)], {"a": (mask, 0.01)})["summary"]
    assert s2["unlabelled_speech_in_output_sec"] == pytest.approx(0.0)   # optional counts as labelled


def _masks(speech_ivs, loud_ivs=(), dur=30.0, hop=0.01):
    sp, act = np.zeros(int(dur / hop), bool), np.zeros(int(dur / hop), bool)
    for a, b in speech_ivs:
        sp[int(a / hop): int(b / hop)] = True
    act |= sp
    for a, b in loud_ivs:
        act[int(a / hop): int(b / hop)] = True
    return {"a": (sp, hop, act)}


def test_removed_silence_inside_a_keep_is_not_clipping():
    m = _masks([(10.0, 10.8), (11.2, 12.0), (20.0, 23.0)])            # 10.8-11.2 is silent
    res = score_timeline(GOLD, [Range("a", 10.0, 10.8), Range("a", 11.2, 12.0), Range("a", 20.0, 23.0)], m)
    k = _keep(res, "k1")
    assert k["status"] == "complete" and k["interior_gap_sec"] == pytest.approx(0.4) and k["interior_gap_active_sec"] == 0


def test_removed_speech_inside_a_keep_is_clipping_even_if_vad_missed_it():
    m = _masks([(10.0, 10.8), (11.2, 12.0)], loud_ivs=[(10.8, 11.2)])  # VAD missed it, loudness did not
    res = score_timeline(GOLD, [Range("a", 10.0, 10.8), Range("a", 11.2, 12.0)], m)
    assert _keep(res, "k1")["status"] == "clipped"


def test_short_exclude_overlap_carrying_speech_is_a_leak():
    m = _masks([(8.0, 9.0), (10.0, 12.0)])
    res = score_timeline(GOLD, [Range("a", 8.92, 12.0)], m)           # 0.08 s < LEAK_TOL, but it is speech
    x = res["exclude"][0]
    assert x["leaked_sec"] == pytest.approx(0.08) and x["leaked"]
    quiet = score_timeline(GOLD, [Range("a", 8.92, 12.0)], _masks([(10.0, 12.0)]))
    assert not quiet["exclude"][0]["leaked"]                          # same overlap, silent: not a leak
