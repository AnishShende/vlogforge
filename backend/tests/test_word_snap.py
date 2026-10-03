"""Tests for word-boundary cut snapping."""
import pytest
from app.utils.word_snap import snap_to_word_boundaries, check_tail_bleed, snap_edl_to_word_boundaries


class TestWordSnap:
    def test_snap_to_words(self):
        # EDL cut was originally 1.5s to 3.5s
        # Words are at 1.8-2.0 and 2.5-3.0
        word_timings = [
            {"word": "hello", "start": 1.8, "end": 2.0},
            {"word": "world", "start": 2.5, "end": 3.0},
        ]
        
        # Snapping with pre_pad=0.1, post_pad=0.2
        start, end = snap_to_word_boundaries(
            1.5, 3.5, word_timings, pre_pad=0.1, post_pad=0.2
        )
        
        assert start == pytest.approx(1.7)  # 1.8 - 0.1
        assert end == pytest.approx(3.2)    # 3.0 + 0.2

    def test_snap_out_of_bounds_clamped(self):
        word_timings = [
            {"word": "hello", "start": 1.8, "end": 2.0},
        ]
        
        # original cut: 1.7 to 2.1
        # snapped: 1.8-0.5=1.3 (but clamped to 1.7 - 0.5 = 1.2), end: 2.0+0.5=2.5 (clamped to 2.1+0.5=2.6)
        # Wait, the logic is:
        # snapped_start = max(1.8 - 0.5, 1.7 - 0.5) = max(1.3, 1.2) = 1.3
        # snapped_end = min(2.0 + 0.5, 2.1 + 0.5) = min(2.5, 2.6) = 2.5
        start, end = snap_to_word_boundaries(
            1.7, 2.1, word_timings, pre_pad=0.5, post_pad=0.5
        )
        
        assert start == pytest.approx(1.3)
        assert end == pytest.approx(2.5)

    def test_no_words_in_range(self):
        word_timings = [
            {"word": "hello", "start": 5.0, "end": 5.5},
        ]
        
        start, end = snap_to_word_boundaries(
            1.0, 2.0, word_timings, pre_pad=0.1, post_pad=0.1
        )
        
        assert start == 1.0
        assert end == 2.0

    def test_empty_word_timings(self):
        start, end = snap_to_word_boundaries(
            1.0, 2.0, [], pre_pad=0.1, post_pad=0.1
        )
        
        assert start == 1.0
        assert end == 2.0


class TestTailBleed:
    def test_tail_bleed_detected(self):
        prev_timings = [
            {"word": "failed", "start": 1.0, "end": 1.5}
        ]
        
        # New clip starts at 1.4, which overlaps with the end of "failed" (1.5)
        new_start = check_tail_bleed(1.4, prev_timings, safety_gap=0.1)
        
        assert new_start == pytest.approx(1.6)  # 1.5 + 0.1

    def test_no_tail_bleed(self):
        prev_timings = [
            {"word": "failed", "start": 1.0, "end": 1.5}
        ]
        
        # New clip starts safely after
        new_start = check_tail_bleed(1.7, prev_timings, safety_gap=0.1)
        
        assert new_start == 1.7

    def test_empty_prev_timings(self):
        assert check_tail_bleed(1.4, [], safety_gap=0.1) == 1.4


class TestSnapEDL:
    def test_snap_edl(self):
        edl = [
            {"clip_id": "clip1", "start_sec": 1.0, "end_sec": 4.0}
        ]
        
        segments = {
            "clip1": {
                "word_timings": [
                    {"word": "hello", "start": 1.5, "end": 2.0},
                    {"word": "world", "start": 2.5, "end": 3.0}
                ]
            }
        }
        
        snapped_edl = snap_edl_to_word_boundaries(edl, segments, pre_pad=0.1, post_pad=0.1)
        
        assert snapped_edl[0]["start_sec"] == pytest.approx(1.4)
        assert snapped_edl[0]["end_sec"] == pytest.approx(3.1)
