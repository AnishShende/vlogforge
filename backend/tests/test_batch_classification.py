"""Tests for M4 batched classification (classify_egt_segments_batch).

Tests batch grouping, fallback per-segment behaviour on batch failure,
and schema correctness using entirely mocked Gemini calls.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from app.utils.llm import classify_egt_segments_batch
from app.models import EGTSegment, generate_clip_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_seg_dict(idx: int) -> dict:
    source = "video.mp4"
    start = float(idx * 10)
    end = float((idx + 1) * 10)
    return {
        "clip_id": generate_clip_id(source, start, end),
        "source_file": source,
        "start_sec": start,
        "end_sec": end,
        "transcript": f"Segment {idx} transcript.",
        "visual_description": f"Visual for segment {idx}.",
        "segment_type": "SPEECH",
        "quality_score": 0.9,
        "quality_flags": [],
        "is_bad_take": False,
        "tags": [],
        "structural_cue": None,
        "perception_model": "",
    }


def _make_batch_response(segs: list) -> MagicMock:
    """Build a mock Gemini response that classifies all segs as SPEECH."""
    classifications = [
        {"clip_id": s["clip_id"], "segment_type": "SPEECH", "structural_cue": None}
        for s in segs
    ]
    response = MagicMock()
    response.text = json.dumps({"classifications": classifications})
    return response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_batch_classification_groups_correctly():
    """With batch_size=5 and 12 segments, should make 3 LLM calls (5+5+2)."""
    segments = [_make_seg_dict(i) for i in range(12)]
    call_args = []

    def mock_safe_generate(model, contents, config):
        call_args.append(contents)
        import re
        ids_in_prompt = re.findall(r'"clip_id":\s*"([^"]+)"', contents)
        classifications = [
            {"clip_id": cid, "segment_type": "SPEECH", "structural_cue": None}
            for cid in ids_in_prompt
        ]
        resp = MagicMock()
        resp.text = json.dumps({"classifications": classifications})
        return resp

    import sys
    import types as pytypes
    fake_types = MagicMock()
    fake_genai = MagicMock()
    fake_genai.types = fake_types
    with patch.dict(sys.modules, {"google": MagicMock(), "google.genai": fake_genai, "google.genai.types": fake_types}), \
         patch("app.utils.llm.init_gemini", return_value=True), \
         patch("app.utils.llm.safe_generate_content", side_effect=mock_safe_generate), \
         patch("app.utils.llm.build_rolling_window_summary", return_value="..."):
        result = classify_egt_segments_batch(segments, "Test context.", batch_size=5)

    # 12 segs / 5 = 3 batches
    assert len(call_args) == 3
    assert len(result) == 12


def test_batch_classification_updates_segment_type():
    """Batch classification should update segment_type on the dict in-place."""
    segments = [_make_seg_dict(i) for i in range(3)]
    segments[0]["transcript"] = "Hey guys welcome back!"

    def mock_safe_generate(model, contents, config):
        import re
        ids_in_prompt = re.findall(r'"clip_id":\s*"([^"]+)"', contents)
        classifications = []
        for i, cid in enumerate(ids_in_prompt):
            stype = "INTRO" if i == 0 else "SPEECH"
            classifications.append({"clip_id": cid, "segment_type": stype, "structural_cue": None})
        resp = MagicMock()
        resp.text = json.dumps({"classifications": classifications})
        return resp

    import sys
    fake_types = MagicMock()
    fake_genai = MagicMock()
    fake_genai.types = fake_types
    with patch.dict(sys.modules, {"google": MagicMock(), "google.genai": fake_genai, "google.genai.types": fake_types}), \
         patch("app.utils.llm.init_gemini", return_value=True), \
         patch("app.utils.llm.safe_generate_content", side_effect=mock_safe_generate), \
         patch("app.utils.llm.build_rolling_window_summary", return_value="..."):
        result = classify_egt_segments_batch(segments, "Context.", batch_size=10)

    assert result[0]["segment_type"] == "INTRO"
    assert result[1]["segment_type"] == "SPEECH"
    assert result[2]["segment_type"] == "SPEECH"


def test_batch_classification_fallback_on_batch_failure():
    """If a batch call fails, the fallback per-segment path should run."""
    segments = [_make_seg_dict(i) for i in range(4)]
    call_count = [0]

    def mock_safe_generate(model, contents, config):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("Rate limit exceeded")
        # Per-segment fallback calls succeed
        resp = MagicMock()
        resp.text = json.dumps({"segment_type": "SPEECH", "structural_cue": None})
        return resp

    import sys
    fake_types = MagicMock()
    fake_genai = MagicMock()
    fake_genai.types = fake_types
    with patch.dict(sys.modules, {"google": MagicMock(), "google.genai": fake_genai, "google.genai.types": fake_types}), \
         patch("app.utils.llm.init_gemini", return_value=True), \
         patch("app.utils.llm.safe_generate_content", side_effect=mock_safe_generate), \
         patch("app.utils.llm.build_rolling_window_summary", return_value="..."):
        result = classify_egt_segments_batch(segments, "Context.", batch_size=4)

    # 1 batch call fails, then 4 per-segment fallback calls succeed = 5 total
    assert call_count[0] == 5
    assert len(result) == 4
    for seg in result:
        assert seg["segment_type"] == "SPEECH"


def test_batch_classification_no_gemini_returns_segments_unchanged():
    """If Gemini is unavailable, segments should be returned as-is."""
    segments = [_make_seg_dict(i) for i in range(5)]
    original_types = [s["segment_type"] for s in segments]

    with patch("app.utils.llm.init_gemini", return_value=False):
        result = classify_egt_segments_batch(segments, "Context.", batch_size=5)

    assert result is segments
    assert [s["segment_type"] for s in result] == original_types


def test_batch_classification_marks_perception_model():
    """After successful batch classification, perception_model should be set."""
    segments = [_make_seg_dict(i) for i in range(2)]

    def mock_safe_generate(model, contents, config):
        import re
        ids = re.findall(r'"clip_id":\s*"([^"]+)"', contents)
        classifications = [
            {"clip_id": cid, "segment_type": "SPEECH", "structural_cue": None}
            for cid in ids
        ]
        resp = MagicMock()
        resp.text = json.dumps({"classifications": classifications})
        return resp

    import sys
    fake_types = MagicMock()
    fake_genai = MagicMock()
    fake_genai.types = fake_types
    with patch.dict(sys.modules, {"google": MagicMock(), "google.genai": fake_genai, "google.genai.types": fake_types}), \
         patch("app.utils.llm.init_gemini", return_value=True), \
         patch("app.utils.llm.safe_generate_content", side_effect=mock_safe_generate), \
         patch("app.utils.llm.build_rolling_window_summary", return_value="..."):
        result = classify_egt_segments_batch(segments, "Context.", batch_size=10)

    for seg in result:
        assert "gemini-flash-lite-latest (batch)" in seg["perception_model"]
