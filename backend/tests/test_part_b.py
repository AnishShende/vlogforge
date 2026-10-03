import pytest
from app.tasks.retake_detect import detect_and_resolve_retakes, compute_repetition_ratio
from app.models import EGTDocument, EGTSegment, generate_clip_id
from unittest.mock import patch

def test_b1_union_logic_paraphrase():
    """
    Test that a whole-utterance similarity forms an edge even if opening similarity is low,
    proving the waterfall was successfully flattened into a union.
    """
    seg1 = EGTSegment(
        clip_id=generate_clip_id("vid.mp4", 0, 5), source_file="vid.mp4", start_sec=0, end_sec=5,
        transcript="This video is only going to reach you if you really need to hear it right now."
    )
    seg2 = EGTSegment(
        clip_id=generate_clip_id("vid.mp4", 6, 11), source_file="vid.mp4", start_sec=6, end_sec=11,
        transcript="I don't know who needs to hear this today, but you really need to hear it right now."
    )
    doc = EGTDocument(segments=[seg1, seg2], job_id="test")

    with patch("app.tasks.retake_detect.compute_cosine_similarity") as mock_sim:
        # We will mock the similarities. 
        # For opening: return 0.3 (fails)
        # For whole: return 0.9 (passes)
        def side_effect(emb1, emb2):
            if "opening" in str(emb1):
                return 0.3
            return 0.9
        
        # We also need to mock emb_map so it triggers the sim logic
        with patch("app.tasks.retake_detect.get_embedding_model") as mock_model:
            class MockModel:
                def encode(self, texts):
                    # just return strings to differentiate opening vs whole in the mock
                    if texts and "this video" in texts[0].lower() and len(texts[0]) < 30:
                        return ["opening_1", "opening_2"]
                    return ["whole_1", "whole_2"]
            
            mock_model.return_value = MockModel()
            mock_sim.side_effect = side_effect
            
            # The logic should mark one of them as superseded
            res = detect_and_resolve_retakes(doc)
            
            # If they clustered, one of them will be superseded (winner selection logic marks the other)
            assert res.segments[0].is_superseded_take or res.segments[1].is_superseded_take, "Edge was not formed!"

def test_b2_intra_segment_repetition():
    """
    Test that a single segment with internal stuttering is flagged by repetition_ratio.
    """
    text = "It is an energy exchange that is out of your control and the more openly you It is an energy exchange that is out of your control and it is an energy exchange"
    
    ratio = compute_repetition_ratio(text)
    assert ratio > 0.4, f"Ratio {ratio} should be > 0.4 for this heavy stutter"
    
    seg = EGTSegment(
        clip_id=generate_clip_id("vid.mp4", 0, 5), source_file="vid.mp4", start_sec=0, end_sec=5,
        transcript=text
    )
    doc = EGTDocument(segments=[seg], job_id="test")
    
    # We patch out get_embedding_model just to avoid slow loads
    with patch("app.tasks.retake_detect.get_embedding_model") as mock_model:
        res = detect_and_resolve_retakes(doc)
        
    assert res.segments[0].is_stutter_repeat is True, "Segment was not flagged as stutter_repeat"
    assert res.segments[0].repetition_ratio == ratio
