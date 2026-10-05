import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.tasks.word_timeline_redundancy import (
    detect_redundancy_on_timeline,
    apply_word_timeline_selection,
    clamp_edl_to_word_timeline,
    WORD_TIMELINE_MODEL,
    _maximal_fluent_runs,
    _has_internal_repeat,
    WordTimelineEntry,
)
from app.tasks.egt import build_egt_document
from app.models import EGTDocument, EGTSegment
from app.config import settings

settings.enable_word_timeline_redundancy = True


def _egt_from_takes(takes, gap=2.0):
    """One EGTSegment per take; words spaced 0.1s, `gap` seconds between takes
    (>= PRESPLIT_GAP_SEC so takes become separate candidate chunks)."""
    segments = []
    t = 0.0
    for i, take in enumerate(takes):
        wts = []
        seg_start = t
        for w in take.split():
            wts.append({"text": w, "start": round(t, 3), "end": round(t + 0.3, 3)})
            t += 0.4
        segments.append(EGTSegment(
            clip_id=f"c{i}", source_file="IMG.MOV",
            start_sec=seg_start, end_sec=t, word_timings=wts, quality_score=0.9,
        ))
        t += gap
    return EGTDocument(segments=segments)


# --- Pure candidate-generation unit tests (deterministic, no scorer) --------

def test_maximal_fluent_run_has_no_internal_repeat():
    """A stuttered chunk must yield at least one run free of repeated 4-grams."""
    text = ("we went to the market early we went to the market early and "
            "we went to the market early to buy the freshest fruit before "
            "the crowds arrived and the stalls ran out of mangoes")
    chunk = [WordTimelineEntry(w, i * 0.4, i * 0.4 + 0.3, "IMG.MOV", True, 0.9)
             for i, w in enumerate(text.split())]
    runs = _maximal_fluent_runs(chunk)
    clean = [r for r in runs if not _has_internal_repeat(r)]
    assert clean, "Expected at least one run with no internal repeated 4-gram"
    # The longest clean run should reach the fluent tail (the real delivery).
    longest = max(clean, key=len)
    joined = " ".join(w.word for w in longest)
    assert "out of mangoes" in joined


# --- Selection tests via injected scorer (hermetic; no live JEV) -----------

def test_single_winner_per_group_and_accounting_balances():
    # Three near-identical takes of one line -> one content group.
    egt = _egt_from_takes([
        "I never planned to say this but morning walks changed my whole routine today",
        "I never planned to say this today but morning walks changed my whole routine",
        "I never wanted to say this but morning walks changed my whole routine today okay",
    ])
    # Scorer prefers longer text (stand-in for completeness).
    scorer = lambda text: (min(4.0, len(text.split()) / 4.0), 0.9)
    rep = detect_redundancy_on_timeline(egt, scorer=scorer)

    acc = rep["accounting"]
    assert acc["kept"] + acc["dropped"] == acc["total_candidates"], acc
    assert acc["kept"] == 1, f"3 retakes of one line -> 1 kept, got {acc}"
    assert rep["kept"][0]["flags"] == []  # long clean take -> not flagged


def test_no_clean_take_flag_when_all_candidates_score_low():
    egt = _egt_from_takes([
        "um uh so like um uh the thing is um uh you know",
    ])
    scorer = lambda text: (1.0, 0.4)  # everything judged poor
    rep = detect_redundancy_on_timeline(egt, scorer=scorer)
    assert rep["kept"], "a winner is still emitted (opt B: keep best, flag it)"
    assert any("no-clean-take" in k["flags"] for k in rep["kept"])


def test_clean_take_beats_longer_stuttery_take():
    """Regression for the real failure: the clean complete take must win even
    when a longer stuttery candidate exists (the JEV judgment, mimicked here)."""
    egt = _egt_from_takes([
        # stuttery long attempt + the clean final delivery, same line/opening
        "slow in slow cooking is not a sign of laziness in the kitchen it is a signa",
        "slow cooking is not a sign of laziness in the kitchen but it is a sign that you value flavour",
    ])

    def jev_like(text):
        # High only when fluent (no local 3-gram repeat) and reasonably complete.
        words = [w.lower().strip(".,") for w in text.split()]
        seen, local_rep = {}, False
        for i in range(len(words) - 2):
            g = tuple(words[i:i + 3])
            if g in seen and i - seen[g] <= 10:
                local_rep = True
            seen[g] = i
        if not local_rep and len(words) >= 12:
            return (4.0, 0.95)
        return (1.5, 0.5)

    rep = detect_redundancy_on_timeline(egt, scorer=jev_like)
    kept_texts = " || ".join(k["text"] for k in rep["kept"])
    assert "that you value flavour" in kept_texts, f"clean take not kept: {kept_texts}"
    # the stuttery "slow in slow cooking ... signa" must not be a winner
    assert not any("it is a signa" in k["text"] for k in rep["kept"])
    acc = rep["accounting"]
    assert acc["kept"] + acc["dropped"] == acc["total_candidates"]


# --- Phase 3 integration unit tests ----------------------------------------

def test_has_speech_recomputed_in_build_egt():
    """Sub-segments created post-subdivision don't carry has_speech; build_egt
    must recompute it from the transcript (>=3 words)."""
    segs = [
        EGTSegment(clip_id="a", source_file="IMG.MOV", start_sec=0.0, end_sec=3.0,
                   transcript="this is a real spoken sentence", has_speech=False),
        EGTSegment(clip_id="b", source_file="IMG.MOV", start_sec=3.0, end_sec=4.0,
                   transcript="ok", has_speech=True),  # 1 word -> should flip False
    ]
    doc = build_egt_document(segs, context_summary="", source_file_count=1, total_duration=4.0)
    by_id = {s.clip_id: s for s in doc.segments}
    assert by_id["a"].has_speech is True
    assert by_id["b"].has_speech is False


def test_apply_word_timeline_selection_builds_pseudo_and_keeps_broll():
    """Speech takes become synthetic SPEECH segments; non-speech B-roll is
    carried through; no-clean-take surfaces as a warning."""
    egt = _egt_from_takes([
        "today I want to talk about discipline and how it changes the way you show up",
        "today I want to talk about discipline and how it changes the way you show up now",
    ])
    # Append a non-speech B-roll segment (no words).
    egt.segments.append(EGTSegment(
        clip_id="broll", source_file="IMG.MOV", start_sec=100.0, end_sec=104.0,
        transcript="", has_speech=False, segment_type="B_ROLL",
    ))

    scorer = lambda text: (4.0, 0.95) if len(text.split()) >= 12 else (1.0, 0.5)
    new_doc, warnings = apply_word_timeline_selection(egt, scorer=scorer)

    types = [s.segment_type for s in new_doc.segments]
    assert "B_ROLL" in types, "non-speech B-roll must survive the merge"
    speech = [s for s in new_doc.segments if s.segment_type == "SPEECH"]
    assert speech, "at least one clean speech take kept"
    assert all(s.word_timings for s in speech), "pseudo speech segments carry word_timings"
    # deterministic clip_ids resolve (non-empty, unique)
    ids = [s.clip_id for s in new_doc.segments]
    assert len(ids) == len(set(ids)) and all(ids)


def test_no_clean_take_emits_warning():
    egt = _egt_from_takes(["um uh so like um uh you know the thing um"])
    scorer = lambda text: (1.0, 0.4)
    _new_doc, warnings = apply_word_timeline_selection(egt, scorer=scorer)
    assert any("No clean take" in w for w in warnings)


def test_clamp_restores_word_timeline_bounds_but_not_broll():
    """The LLM reasoner / word-snap may trim bounds; clamp restores the exact
    word-timeline span for pseudo-segments, and leaves non-speech untouched."""
    egt = EGTDocument(segments=[
        EGTSegment(clip_id="wt1", source_file="IMG.MOV", start_sec=133.5, end_sec=153.7,
                   transcript="example spoken line", segment_type="SPEECH",
                   perception_model=WORD_TIMELINE_MODEL),
        EGTSegment(clip_id="br1", source_file="IMG.MOV", start_sec=200.0, end_sec=210.0,
                   transcript="", segment_type="B_ROLL", perception_model="rule-based-v0"),
    ])
    edl = [
        # LLM trimmed the speech clip mid-sentence (142.2 instead of 133.5)
        {"clip_id": "wt1", "start_sec": 142.2, "end_sec": 154.0,
         "core_start_sec": 142.4, "core_end_sec": 153.7},
        # B-roll legitimately trimmed by the reasoner — must stay as-is
        {"clip_id": "br1", "start_sec": 201.0, "end_sec": 205.0,
         "core_start_sec": 201.0, "core_end_sec": 205.0},
    ]
    n = clamp_edl_to_word_timeline(edl, egt)
    assert n == 1
    assert (edl[0]["start_sec"], edl[0]["end_sec"]) == (133.5, 153.7)
    assert (edl[0]["core_start_sec"], edl[0]["core_end_sec"]) == (133.5, 153.7)
    # B-roll untouched
    assert (edl[1]["start_sec"], edl[1]["end_sec"]) == (201.0, 205.0)
