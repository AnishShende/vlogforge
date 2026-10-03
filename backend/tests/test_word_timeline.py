import pytest
import os
import sys

# Ensure app is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.tasks.word_timeline_redundancy import WordTimelineEntry, detect_redundancy_on_timeline
from app.models import EGTDocument, EGTSegment

def create_synthetic_egt_for_words(takes_texts):
    segments = []
    start_time = 0.0
    for i, take in enumerate(takes_texts):
        words = take.split()
        word_timings = []
        seg_start = start_time
        for w in words:
            word_timings.append({
                "text": w,
                "start": start_time,
                "end": start_time + 0.3
            })
            start_time += 0.4
        
        segments.append(EGTSegment(
            clip_id=f"clip_{i}",
            source_file="IMG_1614.MOV",
            start_sec=seg_start,
            end_sec=start_time,
            word_timings=word_timings,
            quality_score=0.9
        ))
        # Add utterance gap between takes
        start_time += 2.0

    return EGTDocument(segments=segments)

def test_energy_exchange_cluster():
    # Synthetic word timeline for "energy exchange"
    takes = [
        "It is an energy exchange that is out of your control and the more people you give it out to openly,",
        "it is an energy exchange that is out of your control and the more you give it out openly the less you have it for the right person in your life it",
        "is an energy exchange that is out of your control and the more you give it out openly the less you have it for the right person that comes in your life."
    ]
    
    egt_doc = create_synthetic_egt_for_words(takes)
    
    # We want to test the "extended-to-complete" behavior.
    # To do this, we need the segment to be cut short artificially, but the words continue in the next segment.
    # So let's create a single utterance that is cut into two segments!
    
    sentence = "It is an energy exchange that is out of your control and the more you give it out openly, the less you have it for the right person in your life."
    words = sentence.split()
    
    # Part 1: cut mid-sentence
    word_timings_1 = []
    start_time = 0.0
    for w in words[:12]:  # up to "openly,"
        word_timings_1.append({"text": w, "start": start_time, "end": start_time + 0.3})
        start_time += 0.4
        
    # Part 2: the rest of the sentence, gap is < 1.5s
    start_time += 0.5
    word_timings_2 = []
    for w in words[12:]:
        word_timings_2.append({"text": w, "start": start_time, "end": start_time + 0.3})
        start_time += 0.4
        
    seg1 = EGTSegment(clip_id="c1", source_file="IMG_1614.MOV", start_sec=0.0, end_sec=word_timings_1[-1]["end"], word_timings=word_timings_1, quality_score=0.9)
    # The EGT document has an artificial cut (not tagged with editorial_split)
    seg2 = EGTSegment(clip_id="c2", source_file="IMG_1614.MOV", start_sec=word_timings_2[0]["start"], end_sec=word_timings_2[-1]["end"], word_timings=word_timings_2, quality_score=0.9)
    
    # Another take (incomplete)
    take2_words = "It is an energy exchange that is out of your control and the more people you give it out to openly,".split()
    word_timings_3 = []
    start_time += 2.0  # new utterance
    for w in take2_words:
        word_timings_3.append({"text": w, "start": start_time, "end": start_time + 0.3})
        start_time += 0.4
    seg3 = EGTSegment(clip_id="c3", source_file="IMG_1614.MOV", start_sec=word_timings_3[0]["start"], end_sec=word_timings_3[-1]["end"], word_timings=word_timings_3, quality_score=0.9)

    egt_doc = EGTDocument(segments=[seg1, seg2, seg3])
    
    # Enable feature flag for the test
    from app.config import settings
    settings.enable_word_timeline_redundancy = True
    
    report = detect_redundancy_on_timeline(egt_doc)
    clusters = report.get("clusters", [])
    
    assert len(clusters) == 1, "Should form exactly 1 redundancy cluster"
    cluster = clusters[0]
    
    # It should have chosen the first candidate and extended it, or chosen it because it was incomplete and extended it.
    assert cluster["winner_reason"] == "extended-to-complete", "Should have extended the mid-sentence cut to complete it"
    assert "right person in your life." in cluster["extended_text"], "Extended text should contain the full sentence"

def test_paraphrased_hook_cluster():
    takes = [
        "I don't know who needs to hear this today but high standards in life...",
        "I don't care if this offends you but high standards...",
        "I don't know who needs to hear this but high standards in life."
    ]
    egt_doc = create_synthetic_egt_for_words(takes)
    from app.config import settings
    settings.enable_word_timeline_redundancy = True
    
    report = detect_redundancy_on_timeline(egt_doc)
    clusters = report.get("clusters", [])
    assert len(clusters) > 0, "Should cluster paraphrased hooks correctly"
