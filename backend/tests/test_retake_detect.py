import pytest
from app.models import EGTSegment, EGTDocument
from app.config import settings
from app.tasks.retake_detect import detect_and_resolve_retakes, reconstruct_utterances

def test_reconstruct_utterances():
    segments = [
        # Source file A, two adjacent segments tagged with editorial_split
        EGTSegment(clip_id="c1", source_file="file_A.mp4", start_sec=0.0, end_sec=5.0, tags=["editorial_split"]),
        EGTSegment(clip_id="c2", source_file="file_A.mp4", start_sec=5.0, end_sec=10.0, tags=["editorial_split"]),
        
        # Source file A, non-adjacent segment (separate utterance)
        EGTSegment(clip_id="c3", source_file="file_A.mp4", start_sec=15.0, end_sec=20.0, tags=["editorial_split"]),
        
        # Source file B, single undivided scene
        EGTSegment(clip_id="c4", source_file="file_B.mp4", start_sec=0.0, end_sec=10.0, tags=[])
    ]
    
    utterances = reconstruct_utterances(segments)
    
    assert len(utterances) == 3
    assert [s.clip_id for s in utterances[0].segments] == ["c1", "c2"]
    assert utterances[0].is_subdivided is True
    
    assert [s.clip_id for s in utterances[1].segments] == ["c3"]
    assert utterances[1].is_subdivided is False
    
    assert [s.clip_id for s in utterances[2].segments] == ["c4"]
    assert utterances[2].is_subdivided is False

def test_short_retake_explicit_marker():
    segments = [
        EGTSegment(clip_id="c1", source_file="A.mp4", start_sec=0.0, end_sec=3.0, transcript="so today we are going to", quality_score=0.8),
        # A gap of 2 seconds
        EGTSegment(clip_id="c2", source_file="A.mp4", start_sec=5.0, end_sec=8.0, transcript="sorry, again so today we are going to look at", quality_score=0.9),
    ]
    
    doc = EGTDocument(segments=segments)
    result = detect_and_resolve_retakes(doc)
    
    # c2 should win due to higher quality and recency, c1 should be superseded
    assert getattr(result.segments[0], "is_superseded_take", False) is True
    assert getattr(result.segments[1], "is_superseded_take", False) is False

def test_long_retake_opening_similarity():
    # 20+ second retakes with same opening, different middle
    segments = [
        # First take
        EGTSegment(clip_id="c1", source_file="A.mp4", start_sec=0.0, end_sec=10.0, transcript="this is the opening where I talk about the first thing", quality_score=0.8),
        EGTSegment(clip_id="c2", source_file="A.mp4", start_sec=10.0, end_sec=25.0, transcript="and then I ramble off topic for a while", quality_score=0.7),
        
        # Second take (starts 2 seconds after first ends)
        EGTSegment(clip_id="c3", source_file="A.mp4", start_sec=27.0, end_sec=37.0, transcript="this is the opening where I talk about the first thing", quality_score=0.9),
        EGTSegment(clip_id="c4", source_file="A.mp4", start_sec=37.0, end_sec=50.0, transcript="and then I stay on track this time.", quality_score=0.95),
    ]
    
    # Subdivide them manually by setting boundaries to match exactly
    segments[0].tags.append("editorial_split")
    segments[1].tags.append("editorial_split")
    segments[1].start_sec = 10.0
    segments[0].end_sec = 10.0
    
    segments[2].tags.append("editorial_split")
    segments[3].tags.append("editorial_split")
    segments[3].start_sec = 37.0
    segments[2].end_sec = 37.0
    
    doc = EGTDocument(segments=segments)
    result = detect_and_resolve_retakes(doc)
    
    # The first utterance (c1, c2) should be superseded, the second (c3, c4) should win
    assert getattr(result.segments[0], "is_superseded_take", False) is True
    assert getattr(result.segments[1], "is_superseded_take", False) is True
    assert getattr(result.segments[2], "is_superseded_take", False) is False
    assert getattr(result.segments[3], "is_superseded_take", False) is False

def test_false_positive_shared_opening():
    # Two genuinely different utterances sharing a common phrase
    segments = [
        EGTSegment(clip_id="c1", source_file="A.mp4", start_sec=0.0, end_sec=5.0, transcript="so anyway I went to the store today", quality_score=0.8),
        EGTSegment(clip_id="c2", source_file="A.mp4", start_sec=15.0, end_sec=20.0, transcript="so anyway I was telling you about my dog", quality_score=0.9),
    ]
    
    doc = EGTDocument(segments=segments)
    result = detect_and_resolve_retakes(doc)
    
    # Neither should be superseded because whole utterance similarity will be low, 
    # even if "so anyway I" triggers opening similarity.
    # Actually, if opening similarity is very high, it might trigger. Let's make sure opening similarity doesn't false positive if we tune it.
    # Our test just checks the mechanism.
    
    assert getattr(result.segments[0], "is_superseded_take", False) is False
    assert getattr(result.segments[1], "is_superseded_take", False) is False

def test_cluster_of_three_takes():
    segments = [
        EGTSegment(clip_id="c1", source_file="A.mp4", start_sec=0.0, end_sec=3.0, transcript="hello world", quality_score=0.8),
        EGTSegment(clip_id="c2", source_file="A.mp4", start_sec=5.0, end_sec=8.0, transcript="let me start over hello world", quality_score=0.7),
        EGTSegment(clip_id="c3", source_file="A.mp4", start_sec=10.0, end_sec=13.0, transcript="hello world.", quality_score=0.9), # Winning take: ends in punctuation, highest quality, most recent
    ]
    
    doc = EGTDocument(segments=segments)
    result = detect_and_resolve_retakes(doc)
    
    # c3 wins, others superseded
    assert getattr(result.segments[0], "is_superseded_take", False) is True
    assert getattr(result.segments[1], "is_superseded_take", False) is True
    assert getattr(result.segments[2], "is_superseded_take", False) is False
