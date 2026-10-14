import os
import logging
import threading
from typing import List, Dict, Optional

logger = logging.getLogger("VlogForge.Transcribe")

_whisper_model = None
_whisper_model_lock = threading.Lock()  # Guard for concurrent lazy initialization

_align_model = None          # (model, metadata) singleton for WhisperX alignment
_align_lock = threading.Lock()


def _get_align_model():
    """Lazily load the WhisperX wav2vec2 alignment model (English, CPU)."""
    global _align_model
    if _align_model is not None:
        return _align_model
    with _align_lock:
        if _align_model is not None:
            return _align_model
        import whisperx  # may raise ImportError -> caller falls back
        model_a, metadata = whisperx.load_align_model(language_code="en", device="cpu")
        _align_model = (model_a, metadata)
        logger.info("WhisperX alignment model loaded (en, cpu).")
        return _align_model


def _forced_align(audio_path: str, segments: List[Dict]) -> Optional[List[Dict]]:
    """Refine word timings with WhisperX forced alignment.

    Takes SENTENCE-LEVEL segments [{start,end,text}] from Whisper and aligns each
    within its own time window, returning a flat word list [{start,end,text}]
    with accurate bounds. Aligning per-segment (not one blob over the whole
    audio) keeps boundaries tight in dense/stutter regions — a single giant
    segment smears word ends by ~1s. Returns None (caller keeps raw words) if
    whisperx is unavailable or anything fails — fail-loud, never silent.
    """
    try:
        import whisperx
    except Exception as e:
        logger.warning(f"[FORCED-ALIGN] whisperx unavailable ({e}); keeping raw Whisper times.")
        return None
    try:
        segs = [s for s in segments if s.get("text", "").strip()]
        if not segs:
            return None
        model_a, metadata = _get_align_model()
        audio = whisperx.load_audio(audio_path)
        aligned = whisperx.align(
            segs, model_a, metadata, audio, device="cpu", return_char_alignments=False,
        )
        out = []
        dropped = 0
        for seg in aligned.get("segments", []):
            seg_words = []
            for w in seg.get("words", []):
                text = str(w.get("word", "")).strip()
                if w.get("start") is None or w.get("end") is None:
                    dropped += 1
                    if text:
                        seg_words.append({"start": None, "end": None, "text": text})
                    continue
                entry = {"start": float(w["start"]), "end": float(w["end"]), "text": text}
                if w.get("score") is not None:
                    entry["conf"] = round(float(w["score"]), 4)
                    entry["conf_src"] = "aligner_score"
                seg_words.append(entry)
            _interpolate_untimed(seg_words, seg.get("start"), seg.get("end"))
            out.extend(seg_words)
        if dropped:
            logger.warning(f"[FORCED-ALIGN] {dropped} word(s) had no aligned time: interpolated + flagged")
        if not out:
            logger.warning("[FORCED-ALIGN] produced no words; keeping raw Whisper times.")
            return None
        logger.info(f"[FORCED-ALIGN] refined {len(out)} word timings via WhisperX "
                    f"({len(segs)} segments).")
        return out
    except Exception as e:
        logger.error(f"[FORCED-ALIGN] failed ({e}); keeping raw Whisper times.")
        return None


def _interpolate_untimed(words: List[Dict], seg_start: Optional[float], seg_end: Optional[float]) -> None:
    """Give words the aligner could not time (numbers, symbols, out-of-dictionary
    tokens) evenly spread times between their timed neighbours, in place, and flag
    them (Archdoc §9.1: interpolate with a low-confidence flag, never drop)."""
    i = 0
    while i < len(words):
        if words[i]["start"] is not None:
            i += 1
            continue
        j = i
        while j < len(words) and words[j]["start"] is None:
            j += 1
        lo = words[i - 1]["end"] if i > 0 else (seg_start if seg_start is not None else None)
        hi = words[j]["start"] if j < len(words) else (seg_end if seg_end is not None else None)
        if lo is None and hi is None:
            lo = hi = 0.0
        elif lo is None:
            lo = hi
        elif hi is None:
            hi = lo
        hi = max(hi, lo)
        step = (hi - lo) / (j - i)
        for k in range(i, j):
            words[k]["start"] = round(lo + step * (k - i), 4)
            words[k]["end"] = round(lo + step * (k - i + 1), 4)
            words[k]["interpolated"] = True
        i = j


def _clip_transcriber(model):
    """Re-transcribe each clip on its own. Measured on 40 gaps of an eval clip:
    one-at-a-time with the file language 129 s; faster-whisper batched clips 343 s
    and merged neighbouring clips' text, so not used."""
    from app.tasks.asr_recovery import guarded_text

    def transcribe_clips(audio, clips, language):
        texts = []
        for a, b in clips:
            segs, _ = model.transcribe(audio[int(a * 16000): int(b * 16000)], beam_size=5, temperature=0.0,
                                       condition_on_previous_text=False, vad_filter=False, language=language)
            texts.append(guarded_text(list(segs)))
        return texts
    return transcribe_clips


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
            logger.error(f"Failed to load faster-whisper: {e}.")
            return None


def transcribe_audio(audio_path: str, status_callback=None) -> List[Dict]:
    """Transcribe an audio file (see _transcribe_audio). Outside mock mode, the result is reused
    from the artifact store when the same audio CONTENT was transcribed with the same settings
    (roadmap Phase 5); mock mode keeps its own replay cache (eval baselines)."""
    from app.config import settings
    from app.utils import artifacts
    if getattr(settings, "enable_mock_whisper", False) or not artifacts.enabled() \
            or not audio_path or not os.path.exists(audio_path):
        return _transcribe_audio(audio_path, status_callback)
    # The True stands for the word-grid flag that used to be part of the key: stored transcripts keep hitting.
    key = artifacts.make_key("transcribe", artifacts.file_digest(audio_path), TRANSCRIBE_VERSION, True,
                             bool(getattr(settings, "enable_forced_alignment", True)))
    cached = artifacts.get("transcribe", key)
    if cached is not None:
        if status_callback:
            status_callback("using stored transcript")
        return cached
    result = _transcribe_audio(audio_path, status_callback)
    if result:                                   # never store an empty / failed transcription
        artifacts.put("transcribe", key, result)
    return result


# Bump when transcription output changes for the same audio + settings (model, decoding, recovery).
TRANSCRIBE_VERSION = "faster-whisper-turbo|whisperx-wav2vec2-en|grid-v3|2026-10-06"


def _transcribe_audio(audio_path: str, status_callback=None) -> List[Dict]:
    """Transcribe an audio file with local faster-whisper + WhisperX alignment and return word
    dictionaries with start, end, text (+ conf, interpolated / recovered flags). Raises if Whisper fails.
    """
    if not audio_path or not os.path.exists(audio_path):
        return []

    from app.config import settings
    if getattr(settings, "enable_mock_whisper", False):
        import hashlib, json
        # hash based on file modification time and path
        try:
            stat = os.stat(audio_path)
            filename = os.path.basename(audio_path)
            # v3 transcripts: temperature 0, conf, interpolated + recovered words.
            version = "v3|"
            cache_key = hashlib.sha256(f"{version}{filename}_{stat.st_size}".encode()).hexdigest()
            print(f"[WHISPER CACHE DEBUG] audio_path={audio_path}, size={stat.st_size}, cache_key={cache_key}")
        except Exception as e:
            cache_key = hashlib.sha256(audio_path.encode()).hexdigest()
            print(f"[WHISPER CACHE DEBUG] stat failed: {e}. Fallback cache_key={cache_key}")
            
        cache_file = os.path.join(settings.mock_llm_dir, f"whisper_{cache_key}.json")
        print(f"[WHISPER CACHE DEBUG] Looking for cache file: {cache_file}")
        
        if os.path.exists(cache_file):
            print(f"[WHISPER CACHE DEBUG] [MOCKED WHISPER CALL] Replaying cached transcription for {audio_path}")
            logger.info(f"[MOCKED WHISPER CALL] Replaying cached transcription for {audio_path}")
            if status_callback:
                status_callback("using cached Whisper STT")
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WHISPER CACHE DEBUG] Failed to load cache: {e}")
                logger.warning(f"Failed to load Whisper cache: {e}")
        else:
            print(f"[WHISPER CACHE DEBUG] [REAL WHISPER CALL] Cache miss for {audio_path}. Calling faster-whisper...")
            logger.info(f"[REAL WHISPER CALL] Cache miss for {audio_path}. Calling faster-whisper...")


    # Attempt local faster-whisper FIRST for accurate timestamps
    if status_callback:
        status_callback("using local Whisper STT")
    logger.info("Attempting Speech-to-Text using local faster-whisper...")
    model = get_whisper_model()
    
    if model is None:
        raise RuntimeError("Local Whisper model unavailable: cannot transcribe (is faster-whisper installed?)")
    try:
        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            # temperature 0 only. Fallback re-decodes repetitive (stutter)
            # segments by random sampling: non-deterministic, drops attempts.
            temperature=0.0,
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
        whisper_segments = []   # sentence-level, for forced alignment windows
        for segment in segments:
            if segment.text and segment.text.strip():
                whisper_segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                })
            if segment.words:
                for word in segment.words:
                    entry = {
                        "start": word.start,
                        "end": word.end,
                        "text": word.word.strip()
                    }
                    entry["conf"] = round(float(word.probability), 4)
                    entry["conf_src"] = "whisper_probability"
                    transcription_results.append(entry)
            else:
                transcription_results.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                })

        logger.info(f"Whisper STT completed with {len(transcription_results)} segments/words.")

        # Forced-alignment refinement (WhisperX / wav2vec2). Whisper's own
        # word timestamps are coarse (words mis-bucketed by ~1.5s, stretched
        # tokens), which breaks clean-cut boundaries downstream. Replace them
        # with forced-aligned times when available. Fail-loud fallback keeps
        # the raw Whisper words on any error or if whisperx is absent.
        if getattr(settings, "enable_forced_alignment", False) and whisper_segments:
            refined = _forced_align(audio_path, whisper_segments)
            if refined is not None:
                transcription_results = refined
                # Re-transcribe speech the first pass skipped, in the language it
                # detected, and merge it back.
                from app.tasks.asr_recovery import recover_skipped_speech
                if status_callback:
                    status_callback("recovering skipped speech")
                transcription_results, _stats = recover_skipped_speech(
                    audio_path, transcription_results, _clip_transcriber(model), _forced_align,
                    language=getattr(info, "language", None))

        if getattr(settings, "enable_mock_whisper", False):
            try:
                os.makedirs(settings.mock_llm_dir, exist_ok=True)
                with open(cache_file, "w") as f:
                    json.dump(transcription_results, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write Whisper cache {cache_file}: {e}")
                
        return transcription_results

    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        raise RuntimeError(f"Whisper transcription failed for {os.path.basename(audio_path)}: {e}") from e
