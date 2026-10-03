"""Tests for EDL pipeline components.

Tests cover:
- Priority validation (CoT → priority mismatch detection)
- Editorial subdivision (long segments → atomic editorial units)
- Soft duration guidance in LLM prompt
"""

import pytest

from app.models import EGTSegment, EGTDocument, EDLEntry, generate_clip_id
from app.tasks.edl import (
    _validate_priority_consistency,
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
# Editorial Subdivision
# ---------------------------------------------------------------------------

class TestEditorialSubdivide:
    """Tests for editorial_subdivide from scene_detect.py."""

    def test_long_segment_is_subdivided_with_distinct_keyframes(self, monkeypatch, tmp_path):
        """A 260s segment with speech gaps should be split, and each sub-segment gets a distinct keyframe."""
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

        # Mock extract_keyframe to verify it's called with correct midpoints
        extracted_keyframes = {}
        def mock_extract(video_path, t, out_path):
            extracted_keyframes[out_path] = t
            return True
        monkeypatch.setattr("app.utils.ffmpeg.extract_keyframe", mock_extract)

        from app.tasks.scene_detect import editorial_subdivide

        parent_kf = str(tmp_path / "keyframes" / "test_scene_0.jpg")
        seg = EGTSegment(
            clip_id=generate_clip_id("test.mp4", 0.0, 260.0),
            source_file="test.mp4",
            start_sec=0.0,
            end_sec=260.0,
            transcript="hello world this is a long segment",
            keyframe_path=parent_kf,
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

        files_info = [{"filename": "test.mp4", "cfr_path": "/mock/proxy/test_proxy.mp4"}]
        result = editorial_subdivide(
            segments=[seg],
            transcript_segments=transcript_segments,
            target_duration=180.0,
            files_info=files_info,
            job_dir=str(tmp_path),
        )

        # Should have multiple sub-segments
        assert len(result) > 1
        
        # Verify keyframes are distinct and properly extracted
        assert len(extracted_keyframes) == len(result)
        for sub in result:
            assert sub.end_sec - sub.start_sec <= 35.0  # Allow small margin
            assert "editorial_split" in sub.tags
            assert sub.keyframe_path != parent_kf
            assert sub.keyframe_path in extracted_keyframes
            
            # Verify the extraction timestamp was the sub-segment midpoint
            expected_midpoint = sub.start_sec + (sub.end_sec - sub.start_sec) / 2.0
            assert extracted_keyframes[sub.keyframe_path] == expected_midpoint

    def test_short_segment_unchanged_keeps_keyframe(self, monkeypatch):
        """Segments shorter than max_segment_sec should pass through unchanged and keep their keyframe."""
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

        # Track if extract_keyframe is called
        extract_called = False
        def mock_extract(*args, **kwargs):
            nonlocal extract_called
            extract_called = True
            return True
        monkeypatch.setattr("app.utils.ffmpeg.extract_keyframe", mock_extract)

        from app.tasks.scene_detect import editorial_subdivide

        parent_kf = "/mock/job/keyframes/test_scene_0.jpg"
        seg = EGTSegment(
            clip_id=generate_clip_id("test.mp4", 0.0, 20.0),
            source_file="test.mp4",
            start_sec=0.0,
            end_sec=20.0,
            transcript="short clip",
            keyframe_path=parent_kf,
        )

        files_info = [{"filename": "test.mp4", "cfr_path": "/mock/proxy/test_proxy.mp4"}]
        result = editorial_subdivide(
            segments=[seg],
            transcript_segments=[],
            target_duration=180.0,
            files_info=files_info,
            job_dir="/mock/job",
        )

        assert len(result) == 1
        assert result[0].clip_id == seg.clip_id
        assert result[0].keyframe_path == parent_kf
        assert not extract_called


# ---------------------------------------------------------------------------
# Soft Duration Guidance in LLM Prompt
# ---------------------------------------------------------------------------

class TestSoftDurationGuidance:
    """Tests confirming target_duration is present as soft guidance in the LLM prompt."""

    def test_target_duration_appears_as_soft_guidance_in_prompt(self, monkeypatch):
        """The EDL LLM prompt should contain target_duration as soft guidance, not a hard constraint."""
        import sys
        import types as pytypes

        # Mock google.genai.types so the import inside generate_edl_llm succeeds
        if "google" not in sys.modules:
            mock_google = pytypes.ModuleType("google")
            mock_genai = pytypes.ModuleType("google.genai")

            class MockTypes:
                @staticmethod
                def GenerateContentConfig(**kwargs):
                    return kwargs

            mock_genai.types = MockTypes
            mock_google.genai = mock_genai
            sys.modules["google"] = mock_google
            sys.modules["google.genai"] = mock_genai
            sys.modules["google.genai.types"] = MockTypes

        captured_prompts = []

        def mock_safe_generate(*args, **kwargs):
            # Capture the prompt (first positional arg after model is contents)
            contents = kwargs.get("contents") or (args[1] if len(args) > 1 else "")
            captured_prompts.append(contents)
            # Return a mock response
            class MockResponse:
                text = '{"chain_of_thought": "test", "edl": []}'
            return MockResponse()

        monkeypatch.setattr("app.utils.llm.safe_generate_content", mock_safe_generate)
        monkeypatch.setattr("app.utils.llm.init_gemini", lambda: True)

        from app.utils.llm import generate_edl_llm

        egt_json = {
            "segments": [
                {"clip_id": "a", "source_file": "test.mp4", "start_sec": 0, "end_sec": 10,
                 "segment_type": "SPEECH", "transcript": "hello"}
            ],
            "total_duration_sec": 10.0,
        }

        generate_edl_llm(egt_json, target_duration=180.0, user_prompt="")

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]

        # Should contain soft guidance language
        assert "DURATION TARGET (SOFT GUIDANCE)" in prompt
        assert "180" in prompt
        assert "creative target, not a hard limit" in prompt

        # Should NOT contain old hard constraint language
        assert "HARD CONSTRAINT" not in prompt
        assert "MUST total between" not in prompt

    def test_no_duration_section_when_target_is_none(self, monkeypatch):
        """When target_duration is None, the prompt should not contain any duration guidance."""
        import sys
        import types as pytypes

        # Mock google.genai.types so the import inside generate_edl_llm succeeds
        if "google" not in sys.modules:
            mock_google = pytypes.ModuleType("google")
            mock_genai = pytypes.ModuleType("google.genai")

            class MockTypes:
                @staticmethod
                def GenerateContentConfig(**kwargs):
                    return kwargs

            mock_genai.types = MockTypes
            mock_google.genai = mock_genai
            sys.modules["google"] = mock_google
            sys.modules["google.genai"] = mock_genai
            sys.modules["google.genai.types"] = MockTypes

        captured_prompts = []

        def mock_safe_generate(*args, **kwargs):
            contents = kwargs.get("contents") or (args[1] if len(args) > 1 else "")
            captured_prompts.append(contents)
            class MockResponse:
                text = '{"chain_of_thought": "test", "edl": []}'
            return MockResponse()

        monkeypatch.setattr("app.utils.llm.safe_generate_content", mock_safe_generate)
        monkeypatch.setattr("app.utils.llm.init_gemini", lambda: True)

        from app.utils.llm import generate_edl_llm

        egt_json = {
            "segments": [
                {"clip_id": "a", "source_file": "test.mp4", "start_sec": 0, "end_sec": 10,
                 "segment_type": "SPEECH", "transcript": "hello"}
            ],
            "total_duration_sec": 10.0,
        }

        generate_edl_llm(egt_json, target_duration=None, user_prompt="")

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]

        assert "DURATION TARGET" not in prompt
        assert "DURATION BUDGET" not in prompt

