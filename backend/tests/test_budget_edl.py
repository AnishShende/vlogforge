"""Tests for the budget-aware EDL pipeline changes.

Tests cover:
- Proportional core trimming (prevents 285s→24s cliff drops)
- Tolerance band enforcement (±10%)
- Priority validation (CoT → priority mismatch detection)
- Editorial subdivision (long segments → atomic editorial units)
"""

import pytest

from app.models import EGTSegment, EGTDocument, EDLEntry, generate_clip_id
from app.tasks.edl import (
    _validate_priority_consistency,
    _proportional_core_trim,
    _MIN_CLIP_DURATION_SEC,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_edl_entry(
    clip_id: str = "abc123",
    source_file: str = "test.mp4",
    start: float = 0.0,
    end: float = 10.0,
    core_start: float = None,
    core_end: float = None,
    priority: str = "MEDIUM",
    quality: float = 1.0,
) -> EDLEntry:
    """Create an EDLEntry for testing."""
    return EDLEntry(
        clip_id=clip_id,
        source_file=source_file,
        start_sec=start,
        end_sec=end,
        core_start_sec=core_start if core_start is not None else start,
        core_end_sec=core_end if core_end is not None else end,
        narrative_priority=priority,
        quality_score=quality,
        editorial_type="KEEP",
        sequence_index=0,
    )


# ---------------------------------------------------------------------------
# Proportional Core Trimming
# ---------------------------------------------------------------------------

class TestProportionalCoreTrim:
    """Tests for _proportional_core_trim."""

    def test_no_trim_when_under_budget(self):
        """Should not trim anything when total is already under budget."""
        entries = [
            _make_edl_entry(clip_id="a", start=0, end=50, priority="MEDIUM"),
            _make_edl_entry(clip_id="b", start=50, end=100, priority="MEDIUM"),
        ]
        result = _proportional_core_trim(entries, max_allowed=120.0, priority_filter=["MEDIUM"])
        total = sum(e.end_sec - e.start_sec for e in result)
        assert total == 100.0

    def test_proportional_trim_distributes_excess(self):
        """Total 285s with 198s max_allowed → should trim to ~198s, not drop clips."""
        entries = [
            _make_edl_entry(clip_id="intro", start=0, end=21, priority="CRITICAL"),
            _make_edl_entry(clip_id="main", start=21, end=282, priority="MEDIUM"),
            _make_edl_entry(clip_id="outro", start=282, end=285, priority="CRITICAL"),
        ]
        result = _proportional_core_trim(entries, max_allowed=198.0, priority_filter=["MEDIUM"])
        total = sum(e.end_sec - e.start_sec for e in result)

        # All 3 clips should still be present
        assert len(result) == 3
        # Total should be close to 198s (intro 21 + trimmed main + outro 3)
        # The main clip (261s) should be trimmed down, CRITICAL clips untouched
        assert total <= 198.0 + 1.0  # Allow small floating point margin
        assert total > 24.0  # Must be way more than the old 24s result

    def test_respects_minimum_clip_duration(self):
        """Should never trim a clip below _MIN_CLIP_DURATION_SEC (3s)."""
        entries = [
            _make_edl_entry(clip_id="short", start=0, end=4, priority="MEDIUM"),
            _make_edl_entry(clip_id="long", start=4, end=200, priority="MEDIUM"),
        ]
        result = _proportional_core_trim(entries, max_allowed=50.0, priority_filter=["MEDIUM"])

        for e in result:
            clip_dur = e.end_sec - e.start_sec
            assert clip_dur >= _MIN_CLIP_DURATION_SEC

    def test_only_trims_matching_priority(self):
        """Should only trim clips matching the priority_filter."""
        entries = [
            _make_edl_entry(clip_id="crit", start=0, end=100, priority="CRITICAL"),
            _make_edl_entry(clip_id="med", start=100, end=200, priority="MEDIUM"),
        ]
        result = _proportional_core_trim(entries, max_allowed=150.0, priority_filter=["MEDIUM"])

        # CRITICAL clip should be untouched
        crit = next(e for e in result if e.clip_id == "crit")
        assert crit.end_sec - crit.start_sec == 100.0

        # MEDIUM clip should be trimmed
        med = next(e for e in result if e.clip_id == "med")
        assert med.end_sec - med.start_sec < 100.0


# ---------------------------------------------------------------------------
# Priority Validation
# ---------------------------------------------------------------------------

class TestPriorityValidation:
    """Tests for _validate_priority_consistency."""

    def test_upgrades_medium_to_critical_on_preservation_language(self):
        """If CoT mentions 'keep' + clip_id, upgrade MEDIUM → CRITICAL."""
        entries = [
            _make_edl_entry(clip_id="abc123", priority="MEDIUM"),
        ]
        cot = "I want to keep clip abc123 because it has a funny moment."
        result = _validate_priority_consistency(entries, cot)
        assert result[0].narrative_priority == "CRITICAL"

    def test_no_upgrade_without_preservation_language(self):
        """If CoT mentions clip_id but without preservation language, keep MEDIUM."""
        entries = [
            _make_edl_entry(clip_id="abc123", priority="MEDIUM"),
        ]
        cot = "Clip abc123 is a standard speech segment with the speaker explaining the recipe."
        result = _validate_priority_consistency(entries, cot)
        assert result[0].narrative_priority == "MEDIUM"

    def test_upgrades_with_cross_sentence_preservation(self):
        """If clip_id is in sentence 1 and 'keep' is in sentence 2, it should upgrade."""
        entries = [
            _make_edl_entry(clip_id="abc123", priority="MEDIUM"),
        ]
        # "keep it" is in a separate sentence but within the 3-sentence window.
        cot = "The moment in clip abc123 is very funny. We should definitely keep it for the final cut."
        result = _validate_priority_consistency(entries, cot)
        assert result[0].narrative_priority == "CRITICAL"

    def test_no_upgrade_if_too_far(self):
        """If preservation language is outside the 3-sentence window, do not upgrade."""
        entries = [
            _make_edl_entry(clip_id="abc123", priority="MEDIUM"),
        ]
        cot = (
            "Clip abc123 is a standard segment. "
            "It shows the host walking into the room. "
            "Nothing special happens here. "
            "Later on we have a great moment. "
            "We should definitely keep it."
        )
        result = _validate_priority_consistency(entries, cot)
        assert result[0].narrative_priority == "MEDIUM"

    def test_no_upgrade_for_critical_clips(self):
        """Already CRITICAL clips should not be re-processed."""
        entries = [
            _make_edl_entry(clip_id="abc123", priority="CRITICAL"),
        ]
        cot = "I want to keep clip abc123 for narrative flow."
        result = _validate_priority_consistency(entries, cot)
        assert result[0].narrative_priority == "CRITICAL"

    def test_empty_cot_returns_unchanged(self):
        """Empty chain_of_thought should return entries unchanged."""
        entries = [
            _make_edl_entry(clip_id="abc123", priority="LOW"),
        ]
        result = _validate_priority_consistency(entries, "")
        assert result[0].narrative_priority == "LOW"


# ---------------------------------------------------------------------------
# Tolerance Band
# ---------------------------------------------------------------------------

class TestToleranceBand:
    """Tests verifying that ±10% tolerance prevents over-correction."""

    def test_285s_edl_against_180s_target_not_24s(self):
        """The bug scenario: 285s EDL against 180s target should NOT produce 24s.

        With graduated repair (proportional trimming + tolerance band), the
        result should land near 180s (within ±10% = 162-198s), not the old
        24s catastrophic failure.
        """
        # Simulate the tea video: 21s intro (CRITICAL), 260s main (MEDIUM), 3s outro (CRITICAL)
        entries = [
            _make_edl_entry(clip_id="intro", start=0, end=21, core_start=0.5, core_end=21, priority="CRITICAL"),
            _make_edl_entry(clip_id="main", start=21, end=282, core_start=21.5, core_end=281.5, priority="MEDIUM"),
            _make_edl_entry(clip_id="outro", start=282, end=285, core_start=282, core_end=284.5, priority="CRITICAL"),
        ]

        target_duration = 180.0
        budget_tolerance = 0.10
        max_allowed = target_duration * (1.0 + budget_tolerance)  # 198s

        # Phase A: Trim padding
        for e in entries:
            if e.narrative_priority in ["LOW", "MEDIUM"]:
                e.start_sec = e.core_start_sec
                e.end_sec = e.core_end_sec

        total = sum(e.end_sec - e.start_sec for e in entries)

        # Phase B: Proportional trim
        if total > max_allowed:
            entries = _proportional_core_trim(entries, max_allowed, priority_filter=["LOW", "MEDIUM"])

        total = sum(e.end_sec - e.start_sec for e in entries)

        # All 3 clips should still exist
        assert len(entries) == 3

        # Total should be close to max_allowed, not 24s
        assert total > 100.0, f"Total {total}s is catastrophically low — the old bug reproduced"
        assert total <= max_allowed + 1.0, f"Total {total}s exceeds tolerance band"

    def test_mechanical_fallback_budget_enforcement(self, monkeypatch):
        """Verify that when LLM fails, the Phase 0 fallback still enforces budget."""
        from app.tasks.edl import generate_edl
        
        # Create a mock EGT document with one long scene
        seg = EGTSegment(
            clip_id="abc1234",
            source_file="test.mp4",
            start_sec=0.0,
            end_sec=285.0,
            transcript="test transcript",
            segment_type="SPEECH"
        )
        egt_doc = EGTDocument(segments=[seg], source_files=["test.mp4"])
        
        # Mock LLM to fail/return None
        monkeypatch.setattr("app.tasks.edl.generate_edl_llm", lambda *args, **kwargs: None)
        
        # Run generate_edl with target_duration=180.0
        final_edl_dicts, warning_msg = generate_edl(egt_doc, target_duration=180.0)
        
        # It should have run mechanical filter + budget enforcement
        assert len(final_edl_dicts) == 1
        result_dur = final_edl_dicts[0]["end_sec"] - final_edl_dicts[0]["start_sec"]
        
        # Should be trimmed to max allowed (180 + 10% = 198)
        assert result_dur <= 198.1


# ---------------------------------------------------------------------------
# Editorial Subdivision
# ---------------------------------------------------------------------------

class TestEditorialSubdivide:
    """Tests for editorial_subdivide from scene_detect.py."""

    def test_long_segment_is_subdivided(self):
        """A 260s segment with speech gaps should be split into multiple sub-segments."""
        import sys
        import types

        # Mock scenedetect module so scene_detect.py can import
        if "scenedetect" not in sys.modules:
            mock_sd = types.ModuleType("scenedetect")
            mock_sd.SceneManager = None
            mock_sd.open_video = None
            mock_detectors = types.ModuleType("scenedetect.detectors")
            mock_detectors.ContentDetector = None
            mock_detectors.AdaptiveDetector = None
            sys.modules["scenedetect"] = mock_sd
            sys.modules["scenedetect.detectors"] = mock_detectors

        from app.tasks.scene_detect import editorial_subdivide

        seg = EGTSegment(
            clip_id=generate_clip_id("test.mp4", 0.0, 260.0),
            source_file="test.mp4",
            start_sec=0.0,
            end_sec=260.0,
            transcript="hello world this is a long segment",
        )

        # Create transcript segments with gaps every ~30s
        transcript_segments = []
        for i in range(0, 260, 30):
            transcript_segments.append({
                "video_file": "test.mp4",
                "start": float(i),
                "end": float(i + 25),
                "text": f"word at {i}",
            })

        result = editorial_subdivide(
            segments=[seg],
            transcript_segments=transcript_segments,
            target_duration=180.0,  # max_segment_sec = max(30, 27) = 30s
        )

        # Should have multiple sub-segments
        assert len(result) > 1
        # All sub-segments should be <= max_segment_sec (30s) or close
        for sub in result:
            assert sub.end_sec - sub.start_sec <= 35.0  # Allow small margin
        # Tags should include editorial_split
        for sub in result:
            assert "editorial_split" in sub.tags

    def test_short_segment_unchanged(self):
        """Segments shorter than max_segment_sec should pass through unchanged."""
        import sys
        import types

        if "scenedetect" not in sys.modules:
            mock_sd = types.ModuleType("scenedetect")
            mock_sd.SceneManager = None
            mock_sd.open_video = None
            mock_detectors = types.ModuleType("scenedetect.detectors")
            mock_detectors.ContentDetector = None
            mock_detectors.AdaptiveDetector = None
            sys.modules["scenedetect"] = mock_sd
            sys.modules["scenedetect.detectors"] = mock_detectors

        from app.tasks.scene_detect import editorial_subdivide

        seg = EGTSegment(
            clip_id=generate_clip_id("test.mp4", 0.0, 20.0),
            source_file="test.mp4",
            start_sec=0.0,
            end_sec=20.0,
            transcript="short clip",
        )

        result = editorial_subdivide(
            segments=[seg],
            transcript_segments=[],
            target_duration=180.0,
        )

        assert len(result) == 1
        assert result[0].clip_id == seg.clip_id
