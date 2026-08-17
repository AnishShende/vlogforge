import os
import logging
import threading
from typing import List, Dict, Optional

from app.models import EGTSegment

logger = logging.getLogger("VlogForge.Transcribe")

_whisper_model = None
_whisper_model_lock = threading.Lock()  # Guard for concurrent lazy initialization


def get_whisper_model():
    """Load Whisper model lazily to save startup memory. Uses GPU (CUDA) by default with fallback to CPU.
    Thread-safe: uses a lock to prevent duplicate model loads during parallel ingestion.
    """
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    with _whisper_model_lock:
        # Double-checked locking: re-test after acquiring the lock
        if _whisper_model is not None:
            return _whisper_model

        try:
            from faster_whisper import WhisperModel
            import numpy as np
            logger.info("Initializing faster-whisper Model (turbo)...")
            # Attempt to load on GPU first
            try:
                logger.info("Trying to initialize Whisper automatically (CUDA/CPU)...")
                model = WhisperModel("turbo", device="auto", compute_type="default")

                # Dry run to force-load libraries
                logger.info("Performing dry-run to verify integrity...")
                dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
                list(model.transcribe(dummy_audio))

                _whisper_model = model
                logger.info("Whisper Model loaded successfully.")
            except Exception as auto_err:
                logger.warning(f"Auto initialization or dry-run failed: {auto_err}. Falling back to CPU...")
                # Fallback to CPU with a very fast model
                _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
                logger.info("Whisper Model (base) loaded successfully on CPU.")
            return _whisper_model
        except Exception as e:
            logger.error(f"Failed to load faster-whisper: {e}. Transcription will run in Mock Mode.")
            return None


def transcribe_audio(audio_path: str, status_callback=None) -> List[Dict]:
    """Transcribe an audio file and return segment dictionaries with start, end, text.
    Uses Gemini 2.5 Flash Speech-to-Text with local Whisper fallback.
    """
    from app.utils.llm import transcribe_audio_gemini

    if not audio_path or not os.path.exists(audio_path):
        return []

    # Attempt local faster-whisper FIRST for accurate timestamps
    if status_callback:
        status_callback("using local Whisper STT")
    logger.info("Attempting Speech-to-Text using local faster-whisper...")
    model = get_whisper_model()
    
    if model is not None:
        try:
            segments, info = model.transcribe(
                audio_path,
                beam_size=5,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(
                    min_speech_duration_ms=100,   # Catch short single-syllable trailing words like "Bye!"
                    min_silence_duration_ms=500,  # Tightly strip silences > 0.5s to expose pure gaps to the pipeline
                    speech_pad_ms=400             # Ensure start/end syllables aren't chopped off by VAD borders
                ),
                word_timestamps=True
            )
            transcription_results = []
            for segment in segments:
                if segment.words:
                    for word in segment.words:
                        transcription_results.append({
                            "start": word.start,
                            "end": word.end,
                            "text": word.word.strip()
                        })
                else:
                    transcription_results.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip()
                    })
            logger.info(f"Whisper STT completed with {len(transcription_results)} segments/words.")
            return transcription_results
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}. Falling back to Gemini STT.")
    else:
        logger.warning("Local Whisper model unavailable. Falling back to Gemini STT.")

    # Fallback to Gemini 1.5 Flash Lite STT
    if status_callback:
        status_callback("Fallback: using Gemini 1.5 Flash Lite STT")
    logger.info("Attempting Speech-to-Text using Gemini 1.5 Flash Lite...")
    gemini_result = transcribe_audio_gemini(audio_path)
    if gemini_result:
        logger.info(f"Gemini 1.5 Flash Lite STT completed with {len(gemini_result)} segments.")
        return gemini_result

    # Final fallback
    logger.warning("All STT methods failed. Falling back to mock transcription.")
    return _mock_transcribe(audio_path)


def align_transcript_with_segments(
    egt_segments: List[EGTSegment],
    transcript_segments: List[Dict],
) -> List[EGTSegment]:
    """Align word/segment transcripts with EGT segments using a linear two-pointer sweep.

    O(N+M) Two-Pointer Optimisation:
    Both `egt_segments` (from PySceneDetect) and `transcript_segments` (from Whisper/Gemini)
    are generated in strict chronological order. Instead of a brute-force O(N×M) nested
    loop, this uses a two-pointer sweep per video file.

    Writes aligned text into EGTSegment.transcript and sets language_id = "en" (Phase 0).
    Returns the same list of EGTSegment objects, mutated in place.
    """
    if not egt_segments or not transcript_segments:
        return egt_segments

    # --- Group transcript segments by video_file for O(1) per-file lookup ---
    segments_by_file: Dict[str, List[Dict]] = {}
    for seg in transcript_segments:
        file_key = seg.get("video_file", "")
        segments_by_file.setdefault(file_key, []).append(seg)

    # Each per-file list is already in chronological order (Whisper/Gemini emit sorted output).
    # For safety, sort by start time.
    for file_key in segments_by_file:
        segments_by_file[file_key].sort(key=lambda s: s["start"])

    # --- Two-pointer sweep per video file ---
    file_ptr: Dict[str, int] = {k: 0 for k in segments_by_file}

    for egt_seg in egt_segments:
        scene_start = egt_seg.start_sec
        scene_end = egt_seg.end_sec
        scene_file = egt_seg.source_file

        file_segs = segments_by_file.get(scene_file, [])
        t_ptr = file_ptr.get(scene_file, 0)

        scene_text_pieces = []

        # Advance past segments that end before this scene starts
        while t_ptr < len(file_segs) and file_segs[t_ptr]["end"] <= scene_start:
            t_ptr += 1

        # Collect all segments whose midpoint falls within [scene_start, scene_end)
        collect_ptr = t_ptr
        while collect_ptr < len(file_segs) and file_segs[collect_ptr]["start"] < scene_end:
            seg = file_segs[collect_ptr]
            # Assign transcript to this scene ONLY if its midpoint falls within the scene
            seg_midpoint = seg["start"] + (seg["end"] - seg["start"]) / 2.0
            if scene_start <= seg_midpoint < scene_end:
                scene_text_pieces.append(seg["text"])
            collect_ptr += 1

        # Persist the pointer so the next scene starts its skip from here
        file_ptr[scene_file] = t_ptr

        # Write to EGTSegment fields
        egt_seg.transcript = " ".join(scene_text_pieces).strip()
        egt_seg.language_id = "en"  # Phase 0: English only

    return egt_segments


def _mock_transcribe(audio_path: str) -> List[Dict]:
    """Generate dummy transcript segments for offline testing."""
    filename = os.path.basename(audio_path)
    logger.info(f"Running mock transcription for {filename}")

    # We yield standard greetings, mid-sections, and goodbyes spaced out
    return [
        {"start": 0.0, "end": 4.0, "text": "Hey guys, welcome back to my channel! Today we are exploring some amazing spots."},
        {"start": 4.5, "end": 12.0, "text": "This is going to be an awesome vlog. I'm currently setting up the camera and getting ready for the day."},
        {"start": 15.0, "end": 28.0, "text": "Look at this incredible view! The lighting is absolutely perfect right now."},
        {"start": 30.0, "end": 45.0, "text": "Just walking down the street, showing you guys around. This city is beautiful."},
        {"start": 50.0, "end": 58.0, "text": "Okay, that's pretty much everything for this stop. Let's move on to the next location."},
        {"start": 60.0, "end": 75.0, "text": "Alright, that is it for today's video. If you enjoyed it, make sure to hit that subscribe button, and I'll see you next time. Peace!"}
    ]
