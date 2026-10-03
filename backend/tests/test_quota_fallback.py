import pytest
import time
import threading
from unittest.mock import patch, MagicMock

from app.models import EGTSegment
from app.tasks.edl import generate_edl
from app.tasks.scene_detect import subdivide_by_speech_gaps
from app.utils.llm import PriorityRateLimiter

# ---------------------------------------------------------
# Part A Test: is_superseded_take exclusion
# ---------------------------------------------------------
def test_edl_excludes_superseded_takes():
    # 1. Test Mechanical Filter directly (what we fixed)
    # Import edl directly to access its internal logic.
    from app.tasks.edl import generate_edl
    
    seg1 = EGTSegment(clip_id="c1", source_file="f1.mp4", start_sec=0, end_sec=5, quality_score=0.9, segment_type="SPEECH", transcript="good take")
    
    seg2 = EGTSegment(clip_id="c2", source_file="f1.mp4", start_sec=5, end_sec=10, quality_score=0.9, segment_type="SPEECH", transcript="retake")
    seg2.is_superseded_take = True  # This is the crux! is_bad_take is False.
    
    seg3 = EGTSegment(clip_id="c3", source_file="f1.mp4", start_sec=10, end_sec=15, quality_score=0.2, segment_type="SPEECH", transcript="bad take")
    seg3.is_bad_take = True
    
    from app.models import EGTDocument
    egt_doc = EGTDocument(job_id="test", segments=[seg1, seg2, seg3], final_video_path="")
    
    # Force mechanical fallback by mocking generate_edl_llm to return None
    with patch("app.tasks.edl.generate_edl_llm", return_value=None):
        edl, warning, _ = generate_edl(egt_doc, [], target_duration=60, user_prompt="")
        
        # EDL should only contain 'c1'. 'c2' is superseded. 'c3' is bad.
        assert len(edl) == 1
        assert edl[0]["clip_id"] == "c1"

    # 2. Test primary LLM prompt building (confirming it's hidden from the LLM too)
    # We mock generate_edl_llm and inspect the `egt_json` it receives.
    with patch("app.tasks.edl.generate_edl_llm") as mock_llm:
        mock_llm.return_value = [{"clip_id": "c1", "source_file": "f1.mp4", "start_sec": 0, "end_sec": 5, "core_start_sec": 0, "core_end_sec": 5, "narrative_priority": "CRITICAL", "editorial_type": "KEEP", "sequence_index": 0}]
        generate_edl(egt_doc, [], target_duration=60, user_prompt="")
        
        called_egt_json = mock_llm.call_args[0][0]  # First positional arg is egt_json
        
        # Ensure 'c2' (superseded) and 'c3' (bad) are NOT in the payload
        clip_ids = [s["clip_id"] for s in called_egt_json["segments"]]
        assert "c1" in clip_ids
        assert "c2" not in clip_ids
        assert "c3" not in clip_ids

# ---------------------------------------------------------
# Part B Test: Gap-meaningfulness fallback
# ---------------------------------------------------------
def test_gap_meaningfulness_fallback_defaults_to_false(caplog):
    # Simulate a 20-second segment with a 5-second gap (3.0s to 8.0s)
    seg = EGTSegment(clip_id="c1", source_file="f1.mp4", start_sec=0, end_sec=20, quality_score=0.9, segment_type="SPEECH", transcript="hello world")
    
    transcript_segments = [
        {"video_file": "f1.mp4", "start": 0.0, "end": 3.0, "text": "hello"},
        # 5 second gap here
        {"video_file": "f1.mp4", "start": 8.0, "end": 10.0, "text": "world"},
    ]
    
    # Mock extract_keyframe to succeed, but describe_keyframe to fail (simulating 429 quota error)
    with patch("app.utils.ffmpeg.extract_keyframe", return_value=True):
        with patch("app.utils.llm.describe_keyframe", return_value=("Visual description unavailable due to API error.", False)):
            refined = subdivide_by_speech_gaps([seg], transcript_segments, long_scene_threshold_sec=10, video_path="mock.mp4", keyframes_dir="/tmp")
            
            # Since fallback is has_meaningful=False (split), the segment should be split into 3 sub-segments!
            assert len(refined) == 3
            
            # Check for the explicit fallback log event
            fallback_log_found = any("gap_meaningfulness_fallback_used: true" in record.message for record in caplog.records)
            assert fallback_log_found, "Explicit fallback log event was not found in logs!"


# ---------------------------------------------------------
# Part C Test: Priority Rate Limiter
# ---------------------------------------------------------
def test_priority_rate_limiter_ordering():
    # We set RPM to a very slow rate (e.g., 60 RPM -> 1 call per second) to easily test queue ordering
    limiter = PriorityRateLimiter(rpm=60)
    
    execution_order = []
    
    def worker(priority, name):
        limiter.wait(priority, name)
        execution_order.append(name)
        
    # Manually acquire the lock and pretend a call just happened so everyone queues up
    limiter.last_call = time.time()
    
    # Spawn threads. We spawn them with a slight delay to ensure they hit the wait queue
    # before the rate limit interval expires.
    t1 = threading.Thread(target=worker, args=(1, "Keyframe Analysis (Medium)"))
    t2 = threading.Thread(target=worker, args=(2, "Gap Meaningfulness (Low)"))
    t3 = threading.Thread(target=worker, args=(0, "EDL Generation (High)"))
    
    # Start Medium, then Low, then High. 
    # Because they all hit the limiter within the 1-second interval, they will all queue.
    t1.start()
    time.sleep(0.05)
    t2.start()
    time.sleep(0.05)
    t3.start()
    
    # Wait for all to finish (should take ~3 seconds total)
    t1.join()
    t2.join()
    t3.join()
    
    # The execution order MUST be sorted by priority (0, then 1, then 2), 
    # regardless of the order they were started.
    assert execution_order == [
        "EDL Generation (High)",
        "Keyframe Analysis (Medium)",
        "Gap Meaningfulness (Low)"
    ], f"Priority limiter failed to reorder calls! Got: {execution_order}"
