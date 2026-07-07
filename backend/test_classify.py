import os
import sys
import json

# Add backend to path
sys.path.append(r"d:\VlogForge\backend")

from app.utils.llm import classify_egt_segments, init_gemini
import asyncio

test_segments = [
    {
        "clip_id": "test1",
        "start_sec": 0.0,
        "end_sec": 5.0,
        "transcript": "hey guys welcome back to the channel today we are going on a hike",
        "visual_description": "Vlogger looking at camera outdoors."
    },
    {
        "clip_id": "test2",
        "start_sec": 5.0,
        "end_sec": 10.0,
        "transcript": "thanks so much for watching, don't forget to subscribe and see you next time",
        "visual_description": "Vlogger waving at camera."
    }
]

if init_gemini():
    results = classify_egt_segments(test_segments, "Vlog about hiking.")
    for r in results:
        print(f"{r['clip_id']}: {r.get('segment_type')} - {r.get('structural_cue')}")
else:
    print("Gemini not initialized.")
