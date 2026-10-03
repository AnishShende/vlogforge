"""Tests for the disfluency detection module."""
import pytest
from app.tasks.disfluency import (
    compute_disfluency_score,
    compute_utterance_disfluency,
    count_repeated_ngrams,
    count_abnormal_pauses,
    _normalize_word,
)


class TestNormalizeWord:
    def test_strips_punctuation(self):
        assert _normalize_word("Hello,") == "hello"
        assert _normalize_word("world!") == "world"
        assert _normalize_word('"test"') == "test"

    def test_lowercase(self):
        assert _normalize_word("HELLO") == "hello"


class TestRepeatedNgrams:
    def test_clean_speech(self):
        words = ["today", "we", "are", "going", "to", "talk", "about"]
        assert count_repeated_ngrams(words) == 0

    def test_bigram_restart(self):
        # "because the because the deepest"
        words = ["because", "the", "because", "the", "deepest"]
        assert count_repeated_ngrams(words) >= 1

    def test_trigram_restart(self):
        # "so today we so today we are going"
        words = ["so", "today", "we", "so", "today", "we", "are", "going"]
        assert count_repeated_ngrams(words) >= 1

    def test_filler_only_ngrams_ignored(self):
        # "uh um uh um" should not count — it's all fillers
        words = ["uh", "um", "uh", "um"]
        assert count_repeated_ngrams(words) == 0

    def test_too_short(self):
        words = ["hello"]
        assert count_repeated_ngrams(words) == 0


class TestAbnormalPauses:
    def test_no_pauses(self):
        timings = [
            {"word": "hello", "start": 0.0, "end": 0.3},
            {"word": "world", "start": 0.4, "end": 0.7},
        ]
        assert count_abnormal_pauses(timings) == 0

    def test_mid_phrase_pause(self):
        timings = [
            {"word": "hello", "start": 0.0, "end": 0.3},
            {"word": "world", "start": 2.0, "end": 2.3},  # 1.7s gap
        ]
        assert count_abnormal_pauses(timings) == 1

    def test_sentence_boundary_pause_ignored(self):
        timings = [
            {"word": "done.", "start": 0.0, "end": 0.3},
            {"word": "Now", "start": 2.0, "end": 2.3},  # 1.7s gap but after period
        ]
        assert count_abnormal_pauses(timings) == 0

    def test_single_word(self):
        timings = [{"word": "hello", "start": 0.0, "end": 0.3}]
        assert count_abnormal_pauses(timings) == 0


class TestDisfluencyScore:
    def test_clean_segment(self):
        timings = [
            {"word": "today", "start": 0.0, "end": 0.3},
            {"word": "we", "start": 0.35, "end": 0.5},
            {"word": "are", "start": 0.55, "end": 0.7},
            {"word": "going", "start": 0.75, "end": 1.0},
        ]
        result = compute_disfluency_score(timings)
        assert result["disfluency_score"] == 0.0
        assert result["restart_count"] == 0
        assert result["hesitation_ratio"] == 0.0
        assert result["pause_count"] == 0
        assert result["word_count"] == 4

    def test_disfluent_segment(self):
        # "because the uh because the deepest"
        timings = [
            {"word": "because", "start": 0.0, "end": 0.4},
            {"word": "the", "start": 0.5, "end": 0.6},
            {"word": "uh", "start": 0.7, "end": 0.9},
            {"word": "because", "start": 1.0, "end": 1.4},
            {"word": "the", "start": 1.5, "end": 1.6},
            {"word": "deepest", "start": 1.7, "end": 2.1},
        ]
        result = compute_disfluency_score(timings)
        assert result["restart_count"] >= 1  # "because the" repeated
        assert result["hesitation_ratio"] > 0  # "uh" is a filler
        assert result["disfluency_score"] > 0

    def test_empty_timings(self):
        result = compute_disfluency_score([])
        assert result["disfluency_score"] == 0.0
        assert result["word_count"] == 0


class TestUtteranceDisfluency:
    def test_multi_segment(self):
        seg1_timings = [
            {"word": "hello", "start": 0.0, "end": 0.3},
            {"word": "world", "start": 0.4, "end": 0.7},
        ]
        seg2_timings = [
            {"word": "how", "start": 1.0, "end": 1.2},
            {"word": "are", "start": 1.3, "end": 1.5},
        ]
        result = compute_utterance_disfluency([seg1_timings, seg2_timings])
        assert result["word_count"] == 4
        assert result["disfluency_score"] == 0.0
