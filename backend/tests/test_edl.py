import pytest
from app.tasks.edl import generate_edl

def test_generate_edl_rules():
    # 1. Total raw duration = 300s. Target duration should be 30% of 300 = 90s.
    # Total input duration is 300s, consisting of 6 scenes of 50s each.
    classified_scenes = [
        {"video_file": "vid1.mp4", "start": 0.0, "end": 50.0, "label": "INTRO", "score": 0.9},
        {"video_file": "vid1.mp4", "start": 50.0, "end": 100.0, "label": "FILLER", "score": 0.8},
        {"video_file": "vid1.mp4", "start": 100.0, "end": 150.0, "label": "HIGHLIGHT", "score": 0.95},
        {"video_file": "vid2.mp4", "start": 0.0, "end": 50.0, "label": "B_ROLL", "score": 0.85},
        {"video_file": "vid2.mp4", "start": 50.0, "end": 100.0, "label": "HIGHLIGHT", "score": 0.75},
        {"video_file": "vid2.mp4", "start": 100.0, "end": 150.0, "label": "OUTRO", "score": 0.95}
    ]
    
    edl = generate_edl(classified_scenes, total_raw_duration=300.0)
    
    # Assertions
    assert len(edl) > 0
    # First item must be INTRO
    assert edl[0]["type"] == "INTRO"
    assert edl[0]["video_file"] == "vid1.mp4"
    
    # Last item must be OUTRO
    assert edl[-1]["type"] == "OUTRO"
    assert edl[-1]["video_file"] == "vid2.mp4"
    
    # Check that FILLER is not present anywhere
    for item in edl:
        assert item["type"] != "FILLER"
        
    # Check total duration is close to target (90s)
    total_dur = sum(item["end_sec"] - item["start_sec"] for item in edl)
    # Intro (50s) + Outro (50s) = 100s. Since intro + outro already exceed 90s target, the total duration will be 100s.
    assert total_dur == 100.0

def test_generate_edl_pacing_rules():
    # 2. Total raw duration = 1200s. Target duration should be 10 minutes (600s).
    # Provide multiple highlights and B-rolls to test pacing.
    scenes = [
        {"video_file": "vid1.mp4", "start": 0.0, "end": 20.0, "label": "INTRO", "score": 0.95},
        # Consecutive highlights adding up to > 90s
        {"video_file": "vid1.mp4", "start": 20.0, "end": 70.0, "label": "HIGHLIGHT", "score": 0.9}, # 50s
        {"video_file": "vid1.mp4", "start": 70.0, "end": 130.0, "label": "HIGHLIGHT", "score": 0.8}, # 60s
        # B-rolls
        {"video_file": "vid2.mp4", "start": 0.0, "end": 30.0, "label": "B_ROLL", "score": 0.85}, # 30s
        {"video_file": "vid2.mp4", "start": 30.0, "end": 80.0, "label": "HIGHLIGHT", "score": 0.7}, # 50s
        {"video_file": "vid2.mp4", "start": 80.0, "end": 100.0, "label": "OUTRO", "score": 0.95} # 20s
    ]
    
    edl = generate_edl(scenes, total_raw_duration=1200.0)
    
    # Ensure it's ordered correctly
    assert edl[0]["type"] == "INTRO"
    assert edl[-1]["type"] == "OUTRO"
    
    # Let's check that we didn't add too many consecutive talkings without B_roll if we can
    types = [item["type"] for item in edl]
    # Check that fillers aren't there
    assert "FILLER" not in types

def test_generate_edl_duplicate_intros():
    # Simulate the user's mango scenario where two scenes get classified as INTRO
    scenes = [
        {"video_file": "vid1.mp4", "start": 0.0, "end": 8.0, "label": "INTRO", "score": 0.9},
        {"video_file": "vid1.mp4", "start": 8.0, "end": 16.0, "label": "INTRO", "score": 0.9},
        {"video_file": "vid1.mp4", "start": 16.0, "end": 22.0, "label": "HIGHLIGHT", "score": 0.75}
    ]
    # Total raw duration = 22.0s
    edl = generate_edl(scenes, total_raw_duration=22.0, target_duration_sec=600.0)
    
    # We should have all three scenes in the EDL since target duration is 22.0s (capped from 600.0s)
    # The first INTRO should remain INTRO, the second should be demoted to HIGHLIGHT and preserved.
    assert len(edl) == 3
    assert edl[0]["start_sec"] == 0.0
    assert edl[0]["type"] == "INTRO"
    
    # The middle one should be scene 1, demoted to HIGHLIGHT
    assert edl[1]["start_sec"] == 8.0
    assert edl[1]["type"] == "HIGHLIGHT"
    
    assert edl[2]["start_sec"] == 16.0
    assert edl[2]["type"] == "HIGHLIGHT"


def test_generate_edl_reel_option():
    # Simulate a day-in-my-life reel with 3 different video files
    scenes = [
        {"video_file": "vid1.mp4", "start": 0.0, "end": 10.0, "label": "INTRO", "score": 0.9},
        {"video_file": "vid1.mp4", "start": 10.0, "end": 20.0, "label": "HIGHLIGHT", "score": 0.8},
        {"video_file": "vid2.mp4", "start": 0.0, "end": 15.0, "label": "HIGHLIGHT", "score": 0.95},
        {"video_file": "vid2.mp4", "start": 15.0, "end": 30.0, "label": "B_ROLL", "score": 0.85},
        {"video_file": "vid3.mp4", "start": 0.0, "end": 12.0, "label": "HIGHLIGHT", "score": 0.9},
        {"video_file": "vid3.mp4", "start": 12.0, "end": 24.0, "label": "OUTRO", "score": 0.95}
    ]
    # target_duration_sec = 60.0 (reel option).
    # Since we are running in tests (init_gemini will return False since API keys are mock/disabled in standard test runs),
    # the code will fall back to the Python heuristic reel selection.
    # The heuristic should select at least one segment from each of "vid1.mp4", "vid2.mp4", and "vid3.mp4".
    edl = generate_edl(scenes, total_raw_duration=111.0, target_duration_sec=60.0, user_prompt="a day in my life")
    
    assert len(edl) > 0
    selected_files = set(item["video_file"] for item in edl)
    
    # Ensure representation from all three files
    assert "vid1.mp4" in selected_files
    assert "vid2.mp4" in selected_files
    assert "vid3.mp4" in selected_files


def test_generate_edl_gym_genre():
    # Gym genre should prioritize gym B-rolls and restrict continuous talking
    scenes = [
        {"video_file": "vid1.mp4", "start": 0.0, "end": 10.0, "label": "INTRO", "score": 0.9},
        {"video_file": "vid1.mp4", "start": 10.0, "end": 40.0, "label": "HIGHLIGHT", "score": 0.75, "text": "talking to camera about lifting", "visual_description": "vlogger talking"}, # 30s talking
        {"video_file": "vid1.mp4", "start": 40.0, "end": 60.0, "label": "B_ROLL", "score": 0.5, "text": "", "visual_description": "lifting barbell at bench"}, # gym B-roll
        {"video_file": "vid2.mp4", "start": 0.0, "end": 10.0, "label": "OUTRO", "score": 0.9}
    ]
    # For a gym vlog, target 60s, it should boost gym B-roll scores and select them
    edl = generate_edl(scenes, total_raw_duration=120.0, target_duration_sec=60.0, vlog_genre="gym")
    assert len(edl) > 0
    # The B-roll contains gym keyword barbell/bench/lifting, so its score should be boosted and selected
    b_roll_items = [item for item in edl if item["type"] == "B_ROLL"]
    assert len(b_roll_items) > 0

def test_generate_edl_boundary_snapping():
    # Boundary snapping test:
    # Scene starts at 10.0 and ends at 20.0.
    # A speech segment starts at 10.4 and ends at 19.8.
    # The scene should snap to [10.4, 19.8].
    scenes = [
        {"video_file": "vid1.mp4", "start": 0.0, "end": 10.0, "label": "INTRO", "score": 0.9},
        {"video_file": "vid1.mp4", "start": 10.0, "end": 20.0, "label": "HIGHLIGHT", "score": 0.8},
        {"video_file": "vid1.mp4", "start": 20.0, "end": 30.0, "label": "OUTRO", "score": 0.9}
    ]
    transcript_segments = [
        {"video_file": "vid1.mp4", "start": 0.0, "end": 9.5, "text": "intro text"},
        {"video_file": "vid1.mp4", "start": 10.4, "end": 19.8, "text": "highlight speech"},
        {"video_file": "vid1.mp4", "start": 20.5, "end": 29.5, "text": "outro text"}
    ]
    edl = generate_edl(scenes, total_raw_duration=30.0, target_duration_sec=60.0, transcript_segments=transcript_segments)
    assert len(edl) == 3
    # Snapped highlight clip:
    assert edl[1]["start_sec"] == 10.4
    assert edl[1]["end_sec"] == 19.8


def test_generate_edl_daily_genre():
    # Daily vlog genre should prioritize lifestyle highlights over B-rolls
    scenes = [
        {"video_file": "vid1.mp4", "start": 0.0, "end": 10.0, "label": "INTRO", "score": 0.9},
        {"video_file": "vid1.mp4", "start": 10.0, "end": 30.0, "label": "HIGHLIGHT", "score": 0.70, "text": "my daily cooking routine", "visual_description": "cooking in kitchen"},
        {"video_file": "vid2.mp4", "start": 0.0, "end": 20.0, "label": "B_ROLL", "score": 0.85, "text": "", "visual_description": "nature mountains landscape"},
        {"video_file": "vid2.mp4", "start": 20.0, "end": 30.0, "label": "OUTRO", "score": 0.9}
    ]
    edl = generate_edl(scenes, total_raw_duration=70.0, target_duration_sec=40.0, vlog_genre="daily")
    assert len(edl) > 0
    # In daily vlog, "my daily cooking routine" highlight should be boosted because of lifestyle keywords and highlight type
    highlight_files = [item["video_file"] for item in edl if item["type"] == "HIGHLIGHT"]
    assert "vid1.mp4" in highlight_files

def test_generate_edl_noise_reduction():
    scenes = [
        {"video_file": "vid1.mp4", "start": 0.0, "end": 10.0, "label": "INTRO", "score": 0.9},
        # Normal highlight - 20s
        {"video_file": "vid1.mp4", "start": 10.0, "end": 30.0, "label": "HIGHLIGHT", "score": 0.80, "text": "normal talk", "visual_description": "talking"},
        # Noisy announcement - 20s
        {"video_file": "vid2.mp4", "start": 0.0, "end": 20.0, "label": "HIGHLIGHT", "score": 0.85, "text": "tali keledar seat belt warning is active", "visual_description": "flight wing view"},
        {"video_file": "vid2.mp4", "start": 20.0, "end": 30.0, "label": "OUTRO", "score": 0.9}
    ]
    
    # Run with default prompt (non-contributing)
    edl_non_contrib = generate_edl(scenes, total_raw_duration=70.0, target_duration_sec=40.0, user_prompt="some daily vlog")
    non_contrib_files = [item["video_file"] for item in edl_non_contrib if item["type"] == "HIGHLIGHT"]
    # The noisy flight seatbelt warning should be penalized and excluded in favor of normal talk
    assert "vid2.mp4" not in non_contrib_files
    
    # Run with prompt requesting flight info (contributing)
    edl_contrib = generate_edl(scenes, total_raw_duration=70.0, target_duration_sec=45.0, user_prompt="show the flight announcement")
    contrib_files = [item["video_file"] for item in edl_contrib if item["type"] == "HIGHLIGHT"]
    # The noisy flight seatbelt warning should not be penalized and should be included since it is contributing
    assert "vid2.mp4" in contrib_files

def test_generate_edl_duration_window():
    # Target duration: 60s. Allowed window: 60 +/- 10s = 50s to 70s.
    scenes = [
        {"video_file": "vid1.mp4", "start": 0.0, "end": 10.0, "label": "INTRO", "score": 0.9},
        {"video_file": "vid1.mp4", "start": 10.0, "end": 40.0, "label": "HIGHLIGHT", "score": 0.8}, # 30s
        {"video_file": "vid2.mp4", "start": 0.0, "end": 35.0, "label": "B_ROLL", "score": 0.75}, # 35s - would push total past 70s (10s + 30s + 35s + 10s = 85s)
        {"video_file": "vid2.mp4", "start": 35.0, "end": 45.0, "label": "OUTRO", "score": 0.9} # 10s
    ]
    edl = generate_edl(scenes, total_raw_duration=95.0, target_duration_sec=60.0)
    total_dur = sum(item["end_sec"] - item["start_sec"] for item in edl)
    # The B-roll must be skipped since it violates the +10s per minute limit (85s > 70s max).
    # So the total duration should be exactly 10 (intro) + 30 (highlight) + 10 (outro) = 50s.
    assert total_dur == 50.0



