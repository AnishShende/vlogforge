"""Run the real perception pipeline on the test video with mocks enabled,
then run the word-timeline redundancy diagnostic and print the full report.

Recreated after the original was removed. It drives the SAME stage functions
the orchestrator calls (orchestrator.py:206-391), so recorded mock cache keys
(prompt-hash based) hit exactly. Whisper + LLM are mocked; nothing hits the
network. Watch the log for `[REAL ...]` / cache-miss lines — those mean a mock
did not hit and the run is no longer fully offline/deterministic.

Usage (from backend/):
    PYTHONPATH=. ~/anaconda3/envs/vlogforge/bin/python run_mock_pipeline.py
"""
import os
import sys
import json

from app.config import settings

# --- Enable mocks + the flag under test BEFORE importing stage modules use them
settings.enable_mock_llm = True
settings.enable_mock_whisper = True
settings.enable_mock_jev = True
settings.enable_word_timeline_redundancy = True

from app.tasks.ingest import ingest_video
from app.tasks.transcribe import transcribe_audio, align_transcript_with_segments
from app.tasks.scene_detect import subdivide_by_speech_gaps, editorial_subdivide
from app.tasks.analyze import analyze_segments
from app.tasks.score import score_segments
from app.tasks.egt import build_egt_document
from app.tasks.retake_detect import detect_and_resolve_retakes
from app.tasks.word_timeline_redundancy import detect_redundancy_on_timeline

JOB_ID = "516a8ef1-85c6-43d8-9817-7ad1a10b9e93"
VIDEO = os.path.join(settings.upload_dir, JOB_ID, "raw", "IMG_1614.MOV")
JOB_DIR = os.path.join(settings.upload_dir, JOB_ID)
TARGET_DURATION = 60.0   # match the real frontend jobs (10s drops most clips)
CONTEXT_TEXT = ""


def build_egt():
    """Run the real perception stages (mocked) and return the EGTDocument.

    Exposed so experiments/tests can reuse the real EGT for this video without
    duplicating the stage wiring.
    """
    assert os.path.exists(VIDEO), f"Test video not found: {VIDEO}"
    print(f"mock_llm_dir={settings.mock_llm_dir}")
    print(f"video={VIDEO}")

    # ---- Stage 1: Ingest (proxy, audio, scene detection, keyframes) ----
    result = ingest_video(VIDEO, JOB_DIR)
    file_info = result["file_info"].model_dump()
    file_info["cfr_path"] = result.get("cfr_path", VIDEO)
    files_info = [file_info]
    all_segments = list(result["segments"])
    total_raw_duration = result["file_info"].duration
    print(f"[ingest] {len(all_segments)} scene segments, duration={total_raw_duration:.1f}s")

    # ---- Stage 2: Transcribe (mocked whisper) + align ----
    full_transcript_segments = []
    audio_path = file_info.get("audio_path")
    transcript = transcribe_audio(audio_path)
    for t in transcript:
        t["video_file"] = file_info["filename"]
        full_transcript_segments.append(t)
    all_segments = align_transcript_with_segments(all_segments, full_transcript_segments)
    print(f"[transcribe] {len(full_transcript_segments)} word/segment timings")

    # ---- Stage 3a/3a.5: speech-gap refinement + editorial subdivision ----
    dynamic_long_scene = max(
        settings.long_scene_floor_sec,
        TARGET_DURATION * settings.long_scene_ratio,
    )
    dynamic_speech_gap = settings.speech_gap_floor_sec
    cfr_path = file_info.get("cfr_path", "")
    keyframes_dir = os.path.join(JOB_DIR, "keyframes")
    if cfr_path and os.path.exists(cfr_path):
        all_segments = subdivide_by_speech_gaps(
            segments=all_segments,
            transcript_segments=full_transcript_segments,
            long_scene_threshold_sec=dynamic_long_scene,
            speech_gap_sec=dynamic_speech_gap,
            video_path=cfr_path,
            keyframes_dir=keyframes_dir,
            context_notes=CONTEXT_TEXT,
        )
    all_segments = editorial_subdivide(
        segments=all_segments,
        transcript_segments=full_transcript_segments,
        target_duration=TARGET_DURATION,
        files_info=files_info,
        job_dir=JOB_DIR,
    )
    print(f"[subdivide] {len(all_segments)} segments after subdivision")

    # ---- Stage 3b: visual analysis (mocked) ----
    analysis_result = analyze_segments(
        segments=all_segments,
        user_context=CONTEXT_TEXT,
        job_dir=JOB_DIR,
        files_info=files_info,
    )
    all_segments = analysis_result["segments"]
    context_summary = analysis_result["context_summary"]

    # ---- Stage 4: scoring/classification (mocked) ----
    all_segments = score_segments(
        all_segments, total_raw_duration, context_summary, settings.quality_threshold
    )

    # ---- Stage 5 + 5.5: EGT build + legacy retake resolution ----
    egt_doc = build_egt_document(
        segments=all_segments,
        context_summary=context_summary,
        source_file_count=1,
        total_duration=total_raw_duration,
    )
    egt_doc = detect_and_resolve_retakes(egt_doc)
    print(f"[egt] {len(egt_doc.segments)} segments; "
          f"speech={sum(1 for s in egt_doc.segments if s.has_speech)}")
    ctx = {
        "files_info": files_info,
        "full_transcript_segments": full_transcript_segments,
        "job_dir": JOB_DIR,
        "total_raw_duration": total_raw_duration,
        "context_summary": context_summary,
        "target_duration": TARGET_DURATION,
        "context_text": CONTEXT_TEXT,
    }
    return egt_doc, ctx


def main():
    egt_doc, _ctx = build_egt()
    # ---- Word-timeline redundancy diagnostic ----
    report = detect_redundancy_on_timeline(egt_doc)
    print_report(report)


def print_report(rep):
    print("\n" + "#" * 78)
    print("# REDUNDANCY REPORT")
    print("#" * 78)
    print("ACCOUNTING:", rep.get("accounting"))
    kept = rep.get("kept", [])
    print(f"\nKEPT ({len(kept)}) — time-ordered final selection:")
    for k in kept:
        flags = (" " + ",".join(k["flags"])) if k["flags"] else ""
        print(f"  [{k['start']:.1f}-{k['end']:.1f}s] {k['word_count']}w "
              f"group_of_{k['group_size']}{flags}")
        print(f"      {k['text']}")
    dropped = rep.get("dropped", [])
    print(f"\nDROPPED ({len(dropped)}):")
    for d in dropped:
        print(f"  [{d['start']:.1f}-{d['end']:.1f}s] {d['reason']}")
        print(f"      {d['text']}")


if __name__ == "__main__":
    main()
