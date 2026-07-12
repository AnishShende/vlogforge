"""Tests for the two-pass cascade scene detection module.

Tests the cascade logic, merging, and fallback behavior using mocked
PySceneDetect internals — no video files needed.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.tasks.scene_detect import (
    detect_scenes,
    _fixed_interval_scenes,
    _merge_short_scenes,
    _adaptive_subdivide,
)


# ---------------------------------------------------------------------------
# Unit: _fixed_interval_scenes
# ---------------------------------------------------------------------------

class TestFixedIntervalScenes:
    def test_basic_intervals(self):
        scenes = _fixed_interval_scenes(24.0, interval=8.0)
        assert len(scenes) == 3
        assert scenes[0] == {"start": 0.0, "end": 8.0}
        assert scenes[1] == {"start": 8.0, "end": 16.0}
        assert scenes[2] == {"start": 16.0, "end": 24.0}

    def test_short_video_single_scene(self):
        scenes = _fixed_interval_scenes(5.0, interval=8.0)
        assert len(scenes) == 1
        assert scenes[0] == {"start": 0.0, "end": 5.0}

    def test_remainder_scene_included(self):
        scenes = _fixed_interval_scenes(10.0, interval=8.0)
        assert len(scenes) == 2
        assert scenes[1] == {"start": 8.0, "end": 10.0}

    def test_sub_second_remainder_excluded(self):
        scenes = _fixed_interval_scenes(8.5, interval=8.0)
        # The remainder is 0.5s, which is < 1.0s, so excluded
        assert len(scenes) == 1
        assert scenes[0] == {"start": 0.0, "end": 8.0}

    def test_zero_duration(self):
        scenes = _fixed_interval_scenes(0.0)
        assert scenes == []


# ---------------------------------------------------------------------------
# Unit: _merge_short_scenes
# ---------------------------------------------------------------------------

class TestMergeShortScenes:
    def test_no_short_scenes(self):
        scenes = [
            {"start": 0.0, "end": 5.0, "detection_method": "content"},
            {"start": 5.0, "end": 12.0, "detection_method": "content"},
        ]
        merged = _merge_short_scenes(scenes, min_duration_sec=1.0)
        assert len(merged) == 2

    def test_short_scene_merged_into_previous(self):
        scenes = [
            {"start": 0.0, "end": 5.0, "detection_method": "content"},
            {"start": 5.0, "end": 5.5, "detection_method": "adaptive"},  # 0.5s — too short
            {"start": 5.5, "end": 12.0, "detection_method": "content"},
        ]
        merged = _merge_short_scenes(scenes, min_duration_sec=1.0)
        assert len(merged) == 2
        # First scene absorbed the short one
        assert merged[0]["end"] == 5.5
        assert merged[1]["start"] == 5.5

    def test_multiple_consecutive_short_scenes(self):
        scenes = [
            {"start": 0.0, "end": 5.0, "detection_method": "content"},
            {"start": 5.0, "end": 5.3, "detection_method": "adaptive"},
            {"start": 5.3, "end": 5.6, "detection_method": "adaptive"},
            {"start": 5.6, "end": 12.0, "detection_method": "content"},
        ]
        merged = _merge_short_scenes(scenes, min_duration_sec=1.0)
        assert len(merged) == 2
        assert merged[0]["end"] == 5.6

    def test_empty_input(self):
        assert _merge_short_scenes([], 1.0) == []

    def test_single_scene(self):
        scenes = [{"start": 0.0, "end": 10.0, "detection_method": "content"}]
        merged = _merge_short_scenes(scenes, min_duration_sec=1.0)
        assert len(merged) == 1

    def test_unsorted_input_sorted_before_merge(self):
        scenes = [
            {"start": 5.0, "end": 12.0, "detection_method": "content"},
            {"start": 0.0, "end": 5.0, "detection_method": "content"},
        ]
        merged = _merge_short_scenes(scenes, min_duration_sec=1.0)
        assert merged[0]["start"] == 0.0
        assert merged[1]["start"] == 5.0


# ---------------------------------------------------------------------------
# Integration: detect_scenes (mocked PySceneDetect)
# ---------------------------------------------------------------------------

def _make_mock_timecode(seconds: float):
    """Create a mock FrameTimecode with get_seconds()."""
    tc = MagicMock()
    tc.get_seconds.return_value = seconds
    return tc


class TestDetectScenesCascade:
    """Test the two-pass cascade with mocked PySceneDetect."""

    @patch("app.tasks.scene_detect.open_video")
    @patch("app.tasks.scene_detect.SceneManager")
    def test_short_scenes_skip_pass2(self, MockSceneManager, mock_open_video):
        """When all Pass 1 scenes are shorter than the long-scene threshold,
        Pass 2 should not run at all."""
        mock_manager = MagicMock()
        MockSceneManager.return_value = mock_manager
        mock_manager.get_scene_list.return_value = [
            (_make_mock_timecode(0.0), _make_mock_timecode(5.0)),
            (_make_mock_timecode(5.0), _make_mock_timecode(10.0)),
        ]

        scenes = detect_scenes(
            "fake_video.mp4",
            duration=10.0,
            long_scene_threshold_sec=15.0,
        )

        assert len(scenes) == 2
        assert all(s["detection_method"] == "content" for s in scenes)

    @patch("app.tasks.scene_detect._adaptive_subdivide")
    @patch("app.tasks.scene_detect.open_video")
    @patch("app.tasks.scene_detect.SceneManager")
    def test_long_scene_triggers_pass2(self, MockSceneManager, mock_open_video, mock_adaptive):
        """A scene longer than the threshold should trigger AdaptiveDetector."""
        mock_manager = MagicMock()
        MockSceneManager.return_value = mock_manager
        mock_manager.get_scene_list.return_value = [
            (_make_mock_timecode(0.0), _make_mock_timecode(30.0)),
        ]

        # Simulate AdaptiveDetector finding two sub-scenes
        mock_adaptive.return_value = [
            {"start": 0.0, "end": 14.0, "detection_method": "adaptive"},
            {"start": 14.0, "end": 30.0, "detection_method": "adaptive"},
        ]

        scenes = detect_scenes(
            "fake_video.mp4",
            duration=30.0,
            long_scene_threshold_sec=15.0,
        )

        assert len(scenes) == 2
        assert all(s["detection_method"] == "adaptive" for s in scenes)
        mock_adaptive.assert_called_once()

    @patch("app.tasks.scene_detect._adaptive_subdivide")
    @patch("app.tasks.scene_detect.open_video")
    @patch("app.tasks.scene_detect.SceneManager")
    def test_adaptive_finds_nothing_keeps_original(self, MockSceneManager, mock_open_video, mock_adaptive):
        """If AdaptiveDetector finds no sub-boundaries, the original scene is kept."""
        mock_manager = MagicMock()
        MockSceneManager.return_value = mock_manager
        mock_manager.get_scene_list.return_value = [
            (_make_mock_timecode(0.0), _make_mock_timecode(25.0)),
        ]
        mock_adaptive.return_value = []

        scenes = detect_scenes("fake.mp4", duration=25.0, long_scene_threshold_sec=15.0)

        assert len(scenes) == 1
        assert scenes[0]["detection_method"] == "content"

    @patch("app.tasks.scene_detect.open_video")
    @patch("app.tasks.scene_detect.SceneManager")
    def test_content_detector_failure_falls_through(self, MockSceneManager, mock_open_video):
        """If ContentDetector raises, the full duration is treated as one long scene."""
        mock_manager = MagicMock()
        MockSceneManager.return_value = mock_manager
        mock_manager.get_scene_list.side_effect = RuntimeError("decoder error")

        # With no adaptive detector mocked, the long scene will go through Pass 2
        # which will also fail, resulting in the original single-scene fallback
        with patch("app.tasks.scene_detect._adaptive_subdivide", return_value=[]):
            scenes = detect_scenes("fail.mp4", duration=20.0, long_scene_threshold_sec=15.0)

        # Should have 1 scene (the full-duration fallback, kept as-is after adaptive returns [])
        assert len(scenes) == 1
        assert scenes[0]["start"] == 0.0
        assert scenes[0]["end"] == 20.0

    @patch("app.tasks.scene_detect.open_video")
    @patch("app.tasks.scene_detect.SceneManager")
    def test_both_detectors_fail_keeps_full_duration(self, MockSceneManager, mock_open_video):
        """If both detectors fail, the full-duration scene is kept as a single entry."""
        MockSceneManager.side_effect = RuntimeError("import error")

        scenes = detect_scenes("broken.mp4", duration=20.0)

        # Pass 1 fails → creates full-duration fallback scene (0-20s)
        # Pass 2 also fails (same SceneManager error) → keeps the original scene
        assert len(scenes) == 1
        assert scenes[0]["start"] == 0.0
        assert scenes[0]["end"] == 20.0

    def test_fixed_interval_fallback_when_no_duration(self):
        """If duration is 0 and detection returns nothing, we get an empty list."""
        # With no video file and no duration, there's nothing to work with
        scenes = detect_scenes("nonexistent.mp4", duration=0.0)
        assert scenes == []

    @patch("app.tasks.scene_detect._adaptive_subdivide")
    @patch("app.tasks.scene_detect.open_video")
    @patch("app.tasks.scene_detect.SceneManager")
    def test_mixed_short_and_long_scenes(self, MockSceneManager, mock_open_video, mock_adaptive):
        """Mix of short and long scenes: only the long one triggers Pass 2."""
        mock_manager = MagicMock()
        MockSceneManager.return_value = mock_manager
        mock_manager.get_scene_list.return_value = [
            (_make_mock_timecode(0.0), _make_mock_timecode(5.0)),    # short
            (_make_mock_timecode(5.0), _make_mock_timecode(25.0)),   # long (20s)
            (_make_mock_timecode(25.0), _make_mock_timecode(30.0)),  # short
        ]

        mock_adaptive.return_value = [
            {"start": 5.0, "end": 15.0, "detection_method": "adaptive"},
            {"start": 15.0, "end": 25.0, "detection_method": "adaptive"},
        ]

        scenes = detect_scenes("mixed.mp4", duration=30.0, long_scene_threshold_sec=15.0)

        assert len(scenes) == 4
        # First scene: content, kept as-is
        assert scenes[0]["detection_method"] == "content"
        # Middle two: adaptive subdivisions
        assert scenes[1]["detection_method"] == "adaptive"
        assert scenes[2]["detection_method"] == "adaptive"
        # Last scene: content, kept as-is
        assert scenes[3]["detection_method"] == "content"

    @patch("app.tasks.scene_detect.open_video")
    @patch("app.tasks.scene_detect.SceneManager")
    def test_sub_second_scenes_filtered(self, MockSceneManager, mock_open_video):
        """Scenes shorter than min_scene_duration_sec are filtered."""
        mock_manager = MagicMock()
        MockSceneManager.return_value = mock_manager
        mock_manager.get_scene_list.return_value = [
            (_make_mock_timecode(0.0), _make_mock_timecode(0.3)),   # too short
            (_make_mock_timecode(0.3), _make_mock_timecode(10.0)),  # ok
        ]

        scenes = detect_scenes("short.mp4", duration=10.0, min_scene_duration_sec=1.0)

        # The 0.3s scene is filtered in Pass 1; only the 10s scene remains
        assert len(scenes) == 1
        assert scenes[0]["start"] == 0.3

    @patch("app.tasks.scene_detect.open_video")
    @patch("app.tasks.scene_detect.SceneManager")
    def test_output_sorted_by_start(self, MockSceneManager, mock_open_video):
        """Final output is always sorted by start time."""
        mock_manager = MagicMock()
        MockSceneManager.return_value = mock_manager
        mock_manager.get_scene_list.return_value = [
            (_make_mock_timecode(10.0), _make_mock_timecode(20.0)),
            (_make_mock_timecode(0.0), _make_mock_timecode(10.0)),
        ]

        scenes = detect_scenes("unsorted.mp4", duration=20.0)

        assert scenes[0]["start"] <= scenes[1]["start"]
