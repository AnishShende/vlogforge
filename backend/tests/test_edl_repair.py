"""Tests for EDL repair and adjacency safeguards.

Tests cover:
- Adjacency safeguard: LOW clips between CRITICALs from different sources get upgraded
- No false upgrade when neighbors are not both CRITICAL
- generate_edl returns None for the warning field (no budget enforcement)
"""
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

def test_adjacency_upgrade(monkeypatch):
    """
    A LOW clip between two CRITICAL clips from different source_file_ids
    — assert it's upgraded to MEDIUM by the adjacency safeguard.
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
    
    edl, warning, _ = generate_edl(get_mock_egt(), target_duration=35.0)
    assert edl[1]["narrative_priority"] == "MEDIUM"
    # No budget enforcement means no warning
    assert warning is None

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
    
    edl, warning, _ = generate_edl(get_mock_egt(), target_duration=35.0)
    assert edl[1]["narrative_priority"] == "LOW"
    assert warning is None

def test_generate_edl_returns_no_warning(monkeypatch):
    """generate_edl should always return None for the warning field now that
    budget enforcement has been removed."""
    llm_mock = [
        {
            "clip_id": "1", "source_file": "A.mp4",
            "start_sec": 0, "end_sec": 200,
            "core_start_sec": 0, "core_end_sec": 200,
            "narrative_priority": "CRITICAL"
        },
    ]
    monkeypatch.setattr("app.tasks.edl.generate_edl_llm", lambda *a, **k: llm_mock)

    # Even with a very tight target_duration, no budget warning should be produced
    edl, warning, _ = generate_edl(get_mock_egt(), target_duration=5.0)
    assert warning is None
    assert len(edl) >= 1
