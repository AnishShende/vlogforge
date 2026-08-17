import json
from app.tasks.edl import generate_edl
from app.models import EGTDocument, EGTSegment

egt_doc = EGTDocument(
    job_id="test",
    total_duration_sec=300.0,
    segments=[
        EGTSegment(clip_id="c1", source_file="file1.mp4", start_sec=0.0, end_sec=50.0, quality_score=1.0, segment_type="INTRO", is_bad_take=False, transcript="hello"),
        EGTSegment(clip_id="c2", source_file="file1.mp4", start_sec=50.0, end_sec=150.0, quality_score=1.0, segment_type="SPEECH", is_bad_take=False, transcript="world"),
        EGTSegment(clip_id="c3", source_file="file1.mp4", start_sec=150.0, end_sec=250.0, quality_score=1.0, segment_type="OUTRO", is_bad_take=False, transcript="bye")
    ]
)

import app.utils.llm
def mock_generate_edl_llm(egt_json, target_duration, user_prompt):
    return [
        {"clip_id": "c1", "source_file": "file1.mp4", "start_sec": 0.0, "end_sec": 50.0, "core_start_sec": 0.0, "core_end_sec": 50.0, "narrative_priority": "CRITICAL", "editorial_type": "INTRO"},
        {"clip_id": "c2", "source_file": "file1.mp4", "start_sec": 50.0, "end_sec": 150.0, "core_start_sec": 50.0, "core_end_sec": 150.0, "narrative_priority": "MEDIUM", "editorial_type": "KEEP"},
        {"clip_id": "c3", "source_file": "file1.mp4", "start_sec": 150.0, "end_sec": 250.0, "core_start_sec": 150.0, "core_end_sec": 250.0, "narrative_priority": "CRITICAL", "editorial_type": "OUTRO"},
    ]

app.tasks.edl.generate_edl_llm = mock_generate_edl_llm

import logging
logging.basicConfig(level=logging.INFO)

print("Running generate_edl...")
edl, warning = generate_edl(egt_doc, transcript_segments=[], target_duration=3.0)
print(f"Warning returned: {warning}")
print(f"EDL length: {len(edl)}")

