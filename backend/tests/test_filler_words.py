import pytest
from app.models import EGTSegment, EGTDocument
from app.config import settings
from app.tasks.retake_detect import detect_and_resolve_retakes, reconstruct_utterances

def test_filler_words_explicit_marker():
    segments = [
        EGTSegment(clip_id="c1", source_file="A.mp4", start_sec=0.0, end_sec=3.0, transcript="so today we are going to", quality_score=0.8),
        # A gap of 2 seconds, transcript has filler words in explicit marker
        EGTSegment(clip_id="c2", source_file="A.mp4", start_sec=5.0, end_sec=8.0, transcript="sorry, uh, again so today we are going to look at", quality_score=0.9),
    ]
    
    doc = EGTDocument(segments=segments)
    result = detect_and_resolve_retakes(doc)
    
    # c2 should win due to higher quality and recency, c1 should be superseded
    assert getattr(result.segments[0], "is_superseded_take", False) is True
    assert getattr(result.segments[1], "is_superseded_take", False) is False
