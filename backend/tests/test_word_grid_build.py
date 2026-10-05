"""Word grid builder + aligner interpolation (Archdoc roadmap Phase 1 / S2). Synthetic transcripts."""
import pytest

from app.models import WORD_FLAG_INTERPOLATED, WORD_FLAG_ZERO_LENGTH, generate_word_id
from app.tasks.transcribe import _interpolate_untimed
from app.tasks.word_grid import build_word_grid, summarize_word_grid


def _t(text, s, e, f="a.mov", **kw):
    return {"text": text, "start": s, "end": e, "video_file": f, **kw}


def test_builds_ids_gaps_and_order_per_file():
    g = build_word_grid([_t("world", 0.6, 1.0), _t("hello", 0.0, 0.5), _t("other", 3.0, 3.4, "b.mov")])
    a = [w for w in g.words if w.source_file == "a.mov"]
    assert [w.text for w in a] == ["hello", "world"]
    assert [w.id for w in a] == [generate_word_id("a.mov", 0), generate_word_id("a.mov", 1)]
    assert a[1].gap_before == pytest.approx(0.1) and a[0].gap_before == 0.0
    assert g.validate_integrity() == []


def test_same_transcript_gives_identical_grid():
    tr = [_t("a", 0.0, 0.2), _t("b", 0.3, 0.5, conf=0.9, conf_src="aligner_score")]
    assert build_word_grid(tr).model_dump() == build_word_grid(list(reversed(tr))).model_dump()


def test_flags_and_conf_provenance():
    g = build_word_grid([_t("x", 1.0, 1.0), _t("y", 1.1, 1.3, interpolated=True),
                         _t("z", 1.4, 1.6, conf=0.8, conf_src="aligner_score")])
    w = {x.text: x for x in g.words}
    assert WORD_FLAG_ZERO_LENGTH in w["x"].flags and WORD_FLAG_INTERPOLATED in w["y"].flags
    assert w["z"].conf == 0.8 and w["x"].conf is None
    s = summarize_word_grid(g)
    assert s["conf_sources"] == {"aligner_score": 1} and s["words_without_conf"] == 2


def test_overlap_is_clamped_and_counted():
    g = build_word_grid([_t("a", 0.0, 0.6), _t("b", 0.5, 0.9)])
    assert g.words[1].start == 0.6 and g.asr["overlaps_clamped"] == 1 and g.validate_integrity() == []


def test_multiword_entry_is_split_and_flagged():
    g = build_word_grid([_t("three word entry", 0.0, 0.9)])
    assert [w.text for w in g.words] == ["three", "word", "entry"]
    assert all(WORD_FLAG_INTERPOLATED in w.flags for w in g.words)
    assert g.words[1].start == pytest.approx(0.3)


def test_untimed_entry_is_rejected_loudly():
    with pytest.raises(ValueError, match="untimed"):
        build_word_grid([_t("x", None, None)])


def test_interpolate_untimed_between_neighbours():
    ws = [{"text": "it", "start": 1.0, "end": 1.2}, {"text": "costs", "start": None, "end": None},
          {"text": "15", "start": None, "end": None}, {"text": "dollars", "start": 1.8, "end": 2.2}]
    _interpolate_untimed(ws, 1.0, 2.2)
    assert (ws[1]["start"], ws[1]["end"], ws[2]["start"], ws[2]["end"]) == (1.2, 1.5, 1.5, 1.8)
    assert ws[1]["interpolated"] and ws[2]["interpolated"] and "interpolated" not in ws[0]


def test_interpolate_untimed_at_segment_edges():
    ws = [{"text": "15", "start": None, "end": None}, {"text": "dollars", "start": 1.5, "end": 2.0},
          {"text": "ok", "start": None, "end": None}]
    _interpolate_untimed(ws, 1.0, 2.4)
    assert (ws[0]["start"], ws[0]["end"]) == (1.0, 1.5) and (ws[2]["start"], ws[2]["end"]) == (2.0, 2.4)


def test_recovered_flag_passes_through():
    from app.models import WORD_FLAG_RECOVERED
    g = build_word_grid([_t("a", 0.0, 0.2), _t("b", 0.5, 0.7, recovered=True)])
    assert WORD_FLAG_RECOVERED in g.words[1].flags and WORD_FLAG_RECOVERED not in g.words[0].flags
