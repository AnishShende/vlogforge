import pytest
from app.tasks.edl import generate_edl
from app.models import EGTDocument, EGTSegment

def get_mock_egt():
    return EGTDocument(
        segments=[
            EGTSegment(clip_id="1", source_file="A.mp4", start_sec=0, end_sec=10, quality_score=0.9),
            EGTSegment(clip_id="2", source_file="B.mp4", start_sec=0, end_sec=10, quality_score=0.8),
            EGTSegment(clip_id="3", source_file="C.mp4", start_sec=0, end_sec=10, quality_score=0.7),
        ]
    )

def test_critical_padding_trim(monkeypatch):
    """
    CRITICAL clips with padding, budget only satisfiable by trimming that padding 
    — assert Phase C.5 recovers the duration before halting.
    """
    llm_mock = [
        {
            "clip_id": "1", "source_file": "A.mp4", 
            "start_sec": 0, "end_sec": 10,
            "core_start_sec": 2, "core_end_sec": 8,
            "narrative_priority": "CRITICAL"
        },
        {
            "clip_id": "2", "source_file": "B.mp4", 
            "start_sec": 0, "end_sec": 10,
            "core_start_sec": 2, "core_end_sec": 8,
            "narrative_priority": "CRITICAL"
        }
    ]
    
    monkeypatch.setattr("app.tasks.edl.generate_edl_llm", lambda *a, **k: llm_mock)
    
    edl, warning = generate_edl(get_mock_egt(), target_duration=15.0)
    
    assert warning is None
    assert edl[0]["start_sec"] == 2
    assert edl[0]["end_sec"] == 8
    assert edl[1]["start_sec"] == 2
    assert edl[1]["end_sec"] == 8

def test_adjacency_upgrade(monkeypatch):
    """
    A LOW clip between two CRITICAL clips from different source_file_ids
    — assert it's treated as MEDIUM in the repair loop, not dropped in Phase B.
    """
    llm_mock = [
        {
            "clip_id": "1", "source_file": "A.mp4", 
            "start_sec": 0, "end_sec": 10,
            "core_start_sec": 0, "core_end_sec": 10,
            "narrative_priority": "CRITICAL"
        },
        {
            "clip_id": "2", "source_file": "B.mp4", 
            "start_sec": 0, "end_sec": 10,
            "core_start_sec": 0, "core_end_sec": 10,
            "narrative_priority": "LOW"
        },
        {
            "clip_id": "3", "source_file": "C.mp4", 
            "start_sec": 0, "end_sec": 10,
            "core_start_sec": 0, "core_end_sec": 10,
            "narrative_priority": "CRITICAL"
        }
    ]
    
    monkeypatch.setattr("app.tasks.edl.generate_edl_llm", lambda *a, **k: llm_mock)
    
    edl, warning = generate_edl(get_mock_egt(), target_duration=35.0)
    assert edl[1]["narrative_priority"] == "MEDIUM"

def test_adjacency_no_false_upgrade(monkeypatch):
    """
    A LOW clip between a CRITICAL and a MEDIUM (not two CRITICALs)
    — assert it is NOT upgraded.
    """
    llm_mock = [
        {
            "clip_id": "1", "source_file": "A.mp4", 
            "start_sec": 0, "end_sec": 10,
            "core_start_sec": 0, "core_end_sec": 10,
            "narrative_priority": "CRITICAL"
        },
        {
            "clip_id": "2", "source_file": "B.mp4", 
            "start_sec": 0, "end_sec": 10,
            "core_start_sec": 0, "core_end_sec": 10,
            "narrative_priority": "LOW"
        },
        {
            "clip_id": "3", "source_file": "C.mp4", 
            "start_sec": 0, "end_sec": 10,
            "core_start_sec": 0, "core_end_sec": 10,
            "narrative_priority": "MEDIUM"
        }
    ]
    monkeypatch.setattr("app.tasks.edl.generate_edl_llm", lambda *a, **k: llm_mock)
    
    edl, warning = generate_edl(get_mock_egt(), target_duration=35.0)
    assert edl[1]["narrative_priority"] == "LOW"

def test_phase_d_halt(monkeypatch):
    """
    CRITICAL clips (even after core trimming) exceed the budget.
    Assert the system halts, returns the core-trimmed clips, and emits the specific warning.
    """
    llm_mock = [
        {
            "clip_id": "1", "source_file": "A.mp4", 
            "start_sec": 0, "end_sec": 10,
            "core_start_sec": 2, "core_end_sec": 8,
            "narrative_priority": "CRITICAL"
        },
        {
            "clip_id": "2", "source_file": "B.mp4", 
            "start_sec": 0, "end_sec": 10,
            "core_start_sec": 2, "core_end_sec": 8,
            "narrative_priority": "CRITICAL"
        }
    ]
    monkeypatch.setattr("app.tasks.edl.generate_edl_llm", lambda *a, **k: llm_mock)
    
    # Target duration is 5s. Total core duration is 12s (6s + 6s).
    # It must halt at Phase D and emit the warning.
    edl, warning = generate_edl(get_mock_egt(), target_duration=5.0)
    
    # Should be fully trimmed to core (6s each)
    assert edl[0]["start_sec"] == 2
    assert edl[0]["end_sec"] == 8
    
    # Ensure the warning was returned and specifies the numbers
    assert warning is not None
    assert "Budget Exceeded" in warning
    assert "5.0s" in warning
    assert "12.0s" in warning
