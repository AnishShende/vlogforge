"""Tests for EDL overlap deduplication, merging, and boundary clamping."""

import pytest
from app.models import EDLEntry, EGTDocument, EGTSegment, generate_clip_id
from app.tasks.edl import _deduplicate_and_clamp_edl_overlaps, generate_edl


def _make_edl_entry(clip_id: str, source_file: str, start: float, end: float, ed_type: str = "KEEP", prio: str = "MEDIUM") -> EDLEntry:
    return EDLEntry(
        clip_id=clip_id,
        source_file=source_file,
        start_sec=start,
        end_sec=end,
        core_start_sec=start,
        core_end_sec=end,
        narrative_priority=prio,
        quality_score=0.9,
        editorial_type=ed_type,
        sequence_index=0,
    )


def test_deduplicate_merge_same_type_overlap():
    """Adjacent clips from the same file with overlapping timestamps and same type should merge."""
    entries = [
        _make_edl_entry("c1", "video.mp4", 6.9, 9.4, "INTRO"),
        _make_edl_entry("c2", "video.mp4", 9.0, 11.6, "INTRO"),
    ]
    cleaned = _deduplicate_and_clamp_edl_overlaps(entries)
    assert len(cleaned) == 1
    assert cleaned[0].start_sec == 6.9
    assert cleaned[0].end_sec == 11.6
    assert cleaned[0].editorial_type == "INTRO"
    assert cleaned[0].sequence_index == 0


def test_deduplicate_clamp_different_type_overlap():
    """Adjacent clips from the same file with overlapping timestamps and different types should clamp."""
    entries = [
        _make_edl_entry("c1", "video.mp4", 12.0, 18.5, "INTRO"),
        _make_edl_entry("c2", "video.mp4", 18.1, 22.4, "KEEP"),
    ]
    cleaned = _deduplicate_and_clamp_edl_overlaps(entries)
    assert len(cleaned) == 2
    assert cleaned[0].start_sec == 12.0
    assert cleaned[0].end_sec == 18.5
    # c2 start is clamped to 18.5, eliminating the 0.4s repetition
    assert cleaned[1].start_sec == 18.5
    assert cleaned[1].end_sec == 22.4
    assert cleaned[1].sequence_index == 1


def test_deduplicate_drop_fully_contained_clip():
    """If a clip is fully contained within the preceding clip, it should be dropped."""
    entries = [
        _make_edl_entry("c1", "video.mp4", 3143.6, 3149.2, "OUTRO"),
        _make_edl_entry("c2", "video.mp4", 3147.7, 3149.2, "OUTRO"),
    ]
    cleaned = _deduplicate_and_clamp_edl_overlaps(entries)
    assert len(cleaned) == 1
    assert cleaned[0].start_sec == 3143.6
    assert cleaned[0].end_sec == 3149.2


def test_deduplicate_bridge_micro_gaps():
    """Continuous parts with a micro-gap (<= 0.15s) should bridge to avoid 1-frame black flash."""
    entries = [
        _make_edl_entry("c1", "video.mp4", 10.0, 20.0, "KEEP"),
        _make_edl_entry("c2", "video.mp4", 20.08, 30.0, "KEEP"),
    ]
    cleaned = _deduplicate_and_clamp_edl_overlaps(entries)
    assert len(cleaned) == 1
    assert cleaned[0].start_sec == 10.0
    assert cleaned[0].end_sec == 30.0


def test_deduplicate_different_source_files_preserved():
    """Clips from different files should not be merged or clamped."""
    entries = [
        _make_edl_entry("c1", "file_a.mp4", 0.0, 10.0, "KEEP"),
        _make_edl_entry("c2", "file_b.mp4", 5.0, 15.0, "KEEP"),
    ]
    cleaned = _deduplicate_and_clamp_edl_overlaps(entries)
    assert len(cleaned) == 2
    assert cleaned[0].source_file == "file_a.mp4"
    assert cleaned[0].start_sec == 0.0
    assert cleaned[0].end_sec == 10.0
    assert cleaned[1].source_file == "file_b.mp4"
    assert cleaned[1].start_sec == 5.0
    assert cleaned[1].end_sec == 15.0
