"""JEV (TypeSafe System One) integration for VlogForge.

Provides snap-judgment classification and quality scoring for video segments
using the Jev System One model. Replaces heavyweight Gemini text-generation
calls with typed, calibrated probability outputs.

The SDK reads ``TYPESAFE_API_KEY`` from the environment automatically.
Do **not** hardcode or pass the key in code.
"""

import hashlib
import json
import logging
import os
import threading
from typing import Dict, List, Optional, Tuple

from app.config import settings

logger = logging.getLogger("VlogForge.Jev")

# ---------------------------------------------------------------------------
# Singleton client — create once, reuse across threads
# ---------------------------------------------------------------------------

_jev_client = None
_jev_lock = threading.Lock()


def _get_jev_client():
    """Lazily initialize the TypeSafe client singleton.

    Returns the client instance, or None if the SDK is not installed
    or the API key is missing.
    """
    global _jev_client
    if _jev_client is not None:
        return _jev_client

    with _jev_lock:
        if _jev_client is not None:
            return _jev_client
        try:
            from typesafe_sdk import TypeSafeClient

            _jev_client = TypeSafeClient()
            logger.info("JEV TypeSafe client initialized successfully.")
            return _jev_client
        except Exception as e:
            logger.warning(f"JEV TypeSafe client initialization failed: {e}. "
                           "JEV features will be disabled.")
            return None


def jev_available() -> bool:
    """Return True if the JEV client can be initialized."""
    return _get_jev_client() is not None


# ---------------------------------------------------------------------------
# Question definitions — kept in one place per JEV Rule 11
# ---------------------------------------------------------------------------

def _build_segment_questions():
    """Build the questions dict for a single EGT segment classification.

    All questions for one state go in one request (JEV Rule 2).
    """
    from typesafe_sdk import Choice, Noul, Score

    return {
        "segment_type": Choice(
            instructions=(
                "What type of video segment is this in a vlog? "
                "Classify based on `segment.transcript` and `segment.visual_description`."
            ),
            criteria={
                "INTRO": {
                    "what": "Opening greeting, introduction, welcoming viewers",
                    "examples": ["hi guys", "welcome back", "hey everyone", "good morning"],
                },
                "OUTRO": {
                    "what": "Closing thoughts, subscription call-to-action, goodbyes",
                    "examples": ["thanks for watching", "subscribe", "see you next time", "bye"],
                },
                "SPEECH": {
                    "what": "Meaningful spoken content — the vlogger talking to camera or narrating",
                    "not_for": "Ambient noise or dead air",
                },
                "B_ROLL": {
                    "what": "Scenic, contextual, or transitional footage with minimal or no speech",
                    "examples": ["landscape shot", "walking footage", "establishing shot"],
                },
                "SILENCE": {
                    "what": "Dead air, no speech, no meaningful visual content, very short",
                    "examples": ["black screen", "camera setup", "empty room"],
                },
            },
        ),
        "is_bad_take": Noul(
            instructions=(
                "Does `segment.transcript` contain a false start, stumble, "
                "or the speaker restarting a sentence mid-way?"
            ),
            criteria={
                "true": (
                    "Speaker restarts a sentence, says 'wait', 'let me redo', 'oops', 'start over', "
                    "'one more time', 'cut that', 'wrong take', 'messed up', or repeats themselves. "
                    "Also true if the transcript consists entirely of filler words like 'uh', 'um', 'er'."
                ),
                "false": (
                    "Clean, intentional speech — even if imperfect. Natural pauses and occasional "
                    "'um' within fluent speech do NOT make it a bad take."
                ),
            },
        ),
        "is_background_noise": Noul(
            instructions=(
                "Does `segment.transcript` represent ambient background noise "
                "rather than intentional direct speech from the vlogger?"
            ),
            criteria={
                "true": (
                    "PA announcements, crowd murmur, TV/radio audio bleeding in, music-only, "
                    "non-speech sounds, or STT bracket annotations like [music], [applause], [noise]."
                ),
                "false": (
                    "Direct speech from the vlogger or an interview subject, "
                    "even if recorded in a noisy environment."
                ),
            },
        ),
        "content_quality": Score(
            instructions=(
                "How valuable is this segment for inclusion in a published vlog? "
                "Judge based on `segment.transcript` clarity, `segment.visual_description` interest, "
                "and `segment.duration_sec` plausibility."
            ),
            criteria=[
                "Incoherent or pure noise — no usable content at all",
                "Blooper or false start — speaker stumbles and restarts",
                "Low-value filler — repetitive, low-energy, or off-topic",
                "Decent supporting content — contributes but not essential",
                "Key highlight moment — high energy, visually rich, or story-essential",
            ],
        ),
        "has_structural_cue": Noul(
            instructions=(
                "Does the speaker in `segment.transcript` make a narrative transition cue — "
                "explicitly mentioning what they will show next or signalling a topic shift? "
                "Examples: 'let me show you the travel videos', 'now for the house tour', "
                "'first let's check out X, then we'll do Y'."
            ),
        ),
    }


# ---------------------------------------------------------------------------
# JEV quality scoring constants — tune these on labeled data
# ---------------------------------------------------------------------------

# Content quality Score has 5 levels (0-4). Normalize to 0.0-1.0:
QUALITY_SCORE_LEVELS = 5

# Noul thresholds
BAD_TAKE_NOUL_THRESHOLD = 0.55       # Above this → is_bad_take
BACKGROUND_NOISE_NOUL_THRESHOLD = 0.60  # Above this → background_noise flag
STRUCTURAL_CUE_NOUL_THRESHOLD = 0.50  # Above this → has structural cue

# Quality score blending weights (JEV Score + rule-based signals)
JEV_QUALITY_WEIGHT = 0.70   # Weight for JEV's content_quality Score
RULE_QUALITY_WEIGHT = 0.30  # Weight for rule-based heuristic signals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_segment_jev(
    segment_dict: Dict,
    context_doc: str = "",
    rolling_window: str = "",
) -> Optional[Dict]:
    """Classify a single EGT segment using JEV System One.

    Sends one request with all questions for this segment (JEV Rule 2).

    Args:
        segment_dict: A dict with keys like transcript, visual_description,
                      start_sec, end_sec, duration_sec, clip_id.
        context_doc: The global context document for the vlog.
        rolling_window: Recent transcript text (preceding ~3 minutes).

    Returns:
        A dict with keys: segment_type, is_bad_take, is_background_noise,
        content_quality_score (0.0-1.0), has_structural_cue, structural_cue,
        jev_confidence, perception_model.
        Returns None if JEV is unavailable.
    """
    client = _get_jev_client()
    if client is None:
        return None

    from typesafe_sdk import TypeSafeError

    # Build state — only what the questions need (JEV Rule 6)
    state = {
        "segment": {
            "transcript": segment_dict.get("transcript", ""),
            "visual_description": segment_dict.get("visual_description", ""),
            "duration_sec": segment_dict.get("duration_sec", 0.0),
        },
        "context": context_doc[:2000] if context_doc else "",
        "recent_transcript": rolling_window[:1500] if rolling_window else "",
    }

    questions = _build_segment_questions()

    # Mock layer (eval/dev only): same record-on-success contract as the clean-take
    # cache. Keyed on the exact request state; bump the version if questions change.
    cache_file = None
    if getattr(settings, "enable_mock_jev", False):
        key = hashlib.sha256(
            f"segment_v1|{json.dumps(state, sort_keys=True)}".encode()
        ).hexdigest()
        cache_file = os.path.join(settings.mock_llm_dir, f"jev_segment_{key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file) as f:
                    cached = json.load(f)
                logger.info(f"[MOCKED JEV] Replaying cached segment classification for {segment_dict.get('clip_id', '?')}")
                return cached
            except Exception as e:
                logger.warning(f"Failed to load JEV cache {cache_file}: {e}")
        else:
            logger.info(f"[REAL JEV] Cache miss for segment {segment_dict.get('clip_id', '?')} — calling TypeSafe")

    try:
        response = client.system_one(state=state, questions=questions)

        # Extract typed answers
        segment_type = response.choices["segment_type"].choice
        segment_type_confidence = response.choices["segment_type"].confidence

        is_bad_take_noul = response.nouls["is_bad_take"].noul
        is_bg_noise_noul = response.nouls["is_background_noise"].noul
        has_cue_noul = response.nouls["has_structural_cue"].noul

        # Content quality: Score 0-4, normalize to 0.0-1.0
        quality_raw = response.scores["content_quality"].score
        quality_confidence = response.scores["content_quality"].confidence
        quality_normalized = quality_raw / (QUALITY_SCORE_LEVELS - 1)

        model_version = response.model

        logger.debug(
            f"JEV classified {segment_dict.get('clip_id', '?')}: "
            f"type={segment_type}(conf={segment_type_confidence:.2f}), "
            f"quality={quality_normalized:.2f}(conf={quality_confidence:.2f}), "
            f"bad_take={is_bad_take_noul:.2f}, "
            f"bg_noise={is_bg_noise_noul:.2f}, "
            f"cue={has_cue_noul:.2f}"
        )

        result = {
            "segment_type": segment_type,
            "segment_type_confidence": segment_type_confidence,
            "is_bad_take_noul": is_bad_take_noul,
            "is_background_noise_noul": is_bg_noise_noul,
            "content_quality_normalized": quality_normalized,
            "content_quality_confidence": quality_confidence,
            "has_structural_cue_noul": has_cue_noul,
            "structural_cue": None,  # JEV can't generate text; code infers from transcript
            "perception_model": model_version,
        }

        if cache_file is not None:
            try:
                os.makedirs(settings.mock_llm_dir, exist_ok=True)
                with open(cache_file, "w") as f:
                    json.dump(result, f)
            except Exception as e:
                logger.warning(f"Failed to write JEV cache {cache_file}: {e}")

        return result

    except TypeSafeError as e:
        logger.error(f"JEV API error for segment {segment_dict.get('clip_id', '?')}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected JEV error for segment {segment_dict.get('clip_id', '?')}: {e}")
        return None


def classify_segments_jev_batch(
    segments: List[Dict],
    context_doc: str = "",
    max_workers: int = 6,
    progress_callback=None,
) -> List[Optional[Dict]]:
    """Classify a batch of EGT segments using JEV, with bounded concurrency.

    Per JEV rules: one request per segment (each segment has different state),
    all questions bundled per request, bounded to ~8 workers max.

    Args:
        segments: List of EGT segment dicts.
        context_doc: Global context document.
        max_workers: Thread pool size (capped at 8 per JEV guidance).
        progress_callback: Optional callable(done: int, total: int).

    Returns:
        List of result dicts (same length as segments). None entries
        indicate JEV failure for that segment.
    """
    import concurrent.futures

    if not jev_available():
        return [None] * len(segments)

    # Cap workers at 8 per JEV guidance (rate limits are shared across account)
    effective_workers = min(max_workers, 8)
    total = len(segments)
    results: List[Optional[Dict]] = [None] * total
    completed = 0
    progress_lock = threading.Lock()

    def _classify_one(index: int, seg: Dict) -> None:
        nonlocal completed
        rolling = _build_rolling_window(segments, index, window_sec=180.0)
        result = classify_segment_jev(seg, context_doc=context_doc, rolling_window=rolling)
        results[index] = result

        with progress_lock:
            nonlocal completed
            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = [
            executor.submit(_classify_one, i, seg)
            for i, seg in enumerate(segments)
        ]
        concurrent.futures.wait(futures)

    return results


def _build_rolling_window(segments: List[Dict], current_index: int, window_sec: float = 180.0) -> str:
    """Build a lightweight rolling transcript window (same logic as llm.py)."""
    if current_index <= 0 or not segments:
        return ""

    current_start = segments[current_index].get("start_sec", 0.0)
    window_start = current_start - window_sec

    window_texts = []
    for i in range(current_index):
        seg = segments[i]
        seg_end = seg.get("end_sec", 0.0)
        seg_text = seg.get("transcript", "").strip()
        if seg_end >= window_start and seg_text:
            window_texts.append(seg_text)

    return " ".join(window_texts) if window_texts else ""

# ---------------------------------------------------------------------------
# Clean-take scoring for the word-timeline hybrid (with mock cache)
# ---------------------------------------------------------------------------

_CLEAN_TAKE_QUESTION = None


def _clean_take_question():
    global _CLEAN_TAKE_QUESTION
    if _CLEAN_TAKE_QUESTION is None:
        from typesafe_sdk import Score
        _CLEAN_TAKE_QUESTION = {
            "fluent_complete": Score(
                instructions=(
                    "This is a raw speech-to-text transcription of one attempt at speaking "
                    "a sentence for a vlog. Rate it as a candidate CLEAN TAKE, judging whether "
                    "the thought is COMPLETE and the delivery is FLUENT (no stutters, restarts, "
                    "or repeated words/phrases)."
                ),
                criteria=[
                    "Incoherent — repeated or stuttered words, not a real sentence",
                    "False start or restart fragment",
                    "Incomplete — cut off before the thought finishes",
                    "Complete but with minor disfluency",
                    "Clean, complete, and fluent delivery",
                ],
            )
        }
    return _CLEAN_TAKE_QUESTION


# Bump when the clean-take question / scale changes (stored scores become stale).
JEV_CLEAN_TAKE_VERSION = "fluent_complete|0-4|v1"


def score_clean_take_candidate(text: str) -> Optional[Tuple[float, float]]:
    """Score one candidate transcript for completeness + fluency via JEV.

    Returns (score 0..4, confidence 0..1), or None if JEV is unavailable.
    When `enable_mock_jev` is set, results are cached/replayed per the same
    record-on-success contract as the whisper cache (fail-loud logging).
    """
    client = _get_jev_client()
    if client is None:
        return None

    cache_file = None
    if getattr(settings, "enable_mock_jev", False):
        key = hashlib.sha256(f"clean_take|{text.strip().lower()}".encode()).hexdigest()
        cache_file = os.path.join(settings.mock_llm_dir, f"jev_{key}.json")
        if os.path.exists(cache_file):
            logger.info(f"[MOCKED JEV] Replaying cached clean-take score for {text[:40]!r}")
            try:
                with open(cache_file) as f:
                    d = json.load(f)
                return float(d["score"]), float(d["confidence"])
            except Exception as e:
                logger.warning(f"Failed to load JEV cache {cache_file}: {e}")
        else:
            logger.info(f"[REAL JEV] Cache miss for {text[:40]!r} — calling TypeSafe")

    from app.utils import artifacts
    art_key = None
    if cache_file is None and artifacts.enabled():          # Phase 5: reuse scores outside mock mode
        art_key = artifacts.make_key("jev_clean_take", JEV_CLEAN_TAKE_VERSION, text.strip().lower())
        hit = artifacts.get("jev", art_key)
        if hit is not None:
            return float(hit["score"]), float(hit["confidence"])

    try:
        r = client.system_one(state={"candidate": {"transcript": text}},
                              questions=_clean_take_question())
        s = r.scores["fluent_complete"]
        result = (float(s.score), float(s.confidence))
    except Exception as e:
        logger.error(f"JEV clean-take scoring failed for {text[:40]!r}: {e}")
        return None

    if cache_file is not None:
        try:
            os.makedirs(settings.mock_llm_dir, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump({"score": result[0], "confidence": result[1], "text": text}, f)
        except Exception as e:
            logger.warning(f"Failed to write JEV cache {cache_file}: {e}")

    if art_key is not None:
        artifacts.put("jev", art_key, {"score": result[0], "confidence": result[1]})

    return result


def check_is_explicit_retake_jev(transcript: str) -> bool:
    """Use JEV to check if the transcript starts with an explicit self-correction marker.
    
    Handles filler words, variations, and conversational phrasing robustly.
    """
    client = _get_jev_client()
    if not client or not transcript.strip():
        return False
        
    from typesafe_sdk import Noul
    
    questions = {
        "is_explicit_retake": Noul(
            instructions=(
                "Does the speaker explicitly indicate they are restarting, re-doing, "
                "or correcting themselves with a self-correction phrase?"
            ),
            criteria={
                "true": (
                    "Speaker uses phrases like 'let me start over', 'take two', 'sorry, again', "
                    "'let's try that again', 'one more time', 'cut that', 'let me redo', or similar "
                    "self-correction markers. Correctly ignores filler words like 'uh' or 'um'."
                ),
                "false": "Normal continuous speech without an explicit meta-commentary about restarting."
            }
        )
    }
    
    try:
        response = client.system_one(state={"transcript": transcript}, questions=questions)
        is_retake = response.nouls["is_explicit_retake"].noul > 0.50
        return is_retake
    except Exception as e:
        logger.warning(f"JEV explicit retake check failed: {e}")
        return False
