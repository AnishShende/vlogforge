"""Recover speech the first ASR pass skipped (Archdoc roadmap Phase 1 / S5, candidate E).

Whole-file Whisper silently drops repeated attempts and stutters (IMG_1614: 33% of
labelled speech had no words). Transcribed on its own, a skipped stretch comes back.
So after the first pass: find speech (Silero VAD) that no word covers, re-transcribe
each such stretch alone, keep text that passes Whisper's hallucination guards,
force-align it on the full audio, and merge the words in flagged `recovered`.

Bake-off (2 clips): missing labelled speech 33% -> 2% (IMG_1614), 12% -> 7% (chai),
no extra invented words, boundary p90 5.1 s -> 135 ms / 1.3 s -> 401 ms.
"""

import logging
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from app.tasks.word_grid import word_coverage
from app.utils.speech_activity import envelope, load_audio, uncovered_speech

logger = logging.getLogger("VlogForge.ASRRecovery")

PAD_SEC = 0.2            # context around each stretch when re-transcribing
KEEP_SLACK_SEC = 0.25    # recovered words must land within the stretch +/- this
NO_SPEECH_MAX = 0.6      # Whisper hallucination guards for the re-transcription
AVG_LOGPROB_MIN = -1.0

# (audio 16 kHz mono, [(start, end), ...], language) -> one text per clip ("" = rejected)
ClipTranscriber = Callable[[np.ndarray, List[Tuple[float, float]], Optional[str]], List[str]]
# (audio_path, [{start, end, text}, ...]) -> aligned words [{start, end, text, conf?}] or None
Aligner = Callable[[str, List[Dict]], Optional[List[Dict]]]


def recover_skipped_speech(audio_path: str, words: List[Dict], transcribe_clips: ClipTranscriber,
                           align: Aligner, language: Optional[str] = None,
                           audio: Optional[np.ndarray] = None) -> Tuple[List[Dict], Dict]:
    """words: first-pass aligned words [{start, end, text, ...}] for one file.
    Returns (merged words, stats). Recovered words carry recovered=True and never
    overlap first-pass words."""
    audio = load_audio(audio_path) if audio is None else audio
    env = envelope(audio)
    coverage = word_coverage([SimpleNamespace(start=w["start"], end=w["end"]) for w in words])
    stretches = uncovered_speech(env, coverage)
    stats = {"stretches": len(stretches), "with_text": 0, "recovered_words": 0, "dropped_overlap": 0}
    if not stretches:
        return words, stats

    clips = [(max(0.0, s - PAD_SEC), e + PAD_SEC) for s, e in stretches]
    texts = transcribe_clips(audio, clips, language)
    segments = [{"start": a, "end": b, "text": t} for (a, b), t in zip(clips, texts) if t.strip()]
    stats["with_text"] = len(segments)
    aligned = (align(audio_path, segments) or []) if segments else []
    inside = [w for w in aligned
              if any(s - KEEP_SLACK_SEC <= w["start"] and w["end"] <= e + KEEP_SLACK_SEC for s, e in stretches)]

    merged = sorted(words + [{**w, "recovered": True} for w in inside], key=lambda w: (w["start"], w["end"]))
    out, prev_end = [], -1.0
    for w in merged:
        if w.get("recovered") and w["start"] < prev_end:
            stats["dropped_overlap"] += 1
            continue
        out.append(w)
        prev_end = max(prev_end, w["end"])
    stats["recovered_words"] = sum(1 for w in out if w.get("recovered"))
    logger.info(f"[ASR-RECOVERY] {stats}")
    return out, stats


def guarded_text(segments) -> str:
    """Join Whisper segments that pass the hallucination guards."""
    return " ".join(s.text.strip() for s in segments
                    if s.no_speech_prob <= NO_SPEECH_MAX and s.avg_logprob >= AVG_LOGPROB_MIN).strip()
