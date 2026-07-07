"""Pass 1.4 — Quality Scoring & Segment Type Classification.

This module replaces the former classify.py which used LLM-based editorial judgment
(HIGHLIGHT vs FILLER). In the Phase 0 architecture, this is strictly perception-layer:

    segment_type: what IS the footage  (INTRO | OUTRO | SPEECH | B_ROLL | SILENCE)
    quality_score: calibrated absolute  (0.0–1.0, not relative ranking)
    is_bad_take:   quality_score < threshold

No LLM calls. All classification is rule-based or signal-based.
"""

import re
import logging
from typing import List

from app.models import EGTSegment
from app.config import settings

logger = logging.getLogger("VlogForge.Score")

# ---------------------------------------------------------------------------
# Intro / Outro keyword sets
# ---------------------------------------------------------------------------

INTRO_KEYWORDS = [
    "hi", "hello", "welcome", "hey guys", "what's up", "good morning",
    "starting", "today we", "welcome back", "hey everyone", "what is up",
    "hey what's going on", "good evening", "hey there",
]
OUTRO_KEYWORDS = [
    "bye", "see you", "subscribe", "thanks for watching", "outro",
    "peace out", "next time", "that's it", "thats it", "until next time",
    "catch you", "signing off", "goodbye", "see ya", "like and subscribe",
    "hit that subscribe", "peace",
]

# Disfluency / bad-take indicators
DISFLUENCY_WORDS = {"uh", "um", "er", "ah", "uhh", "umm", "hmm"}
BAD_TAKE_PHRASES = [
    "wait a minute", "hang on", "let me redo", "re-do", "start over",
    "one more time", "that was bad", "delete that", "cut that",
    "wrong take", "messed up", "oops",
]

# Background noise indicators (bracket annotations from STT)
NOISE_BRACKET_PATTERN = re.compile(
    r'\[(music|laughter|applause|chime|sigh|cough|throat|static|hum|buzz|'
    r'beep|siren|alarm|noise|whisper|murmur|chattering|screech)\]',
    re.IGNORECASE
)


def _contains_keyword(text: str, keywords: List[str]) -> bool:
    """Check if text contains any of the keywords (supports multi-word phrases)."""
    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))
    for kw in keywords:
        kw_lower = kw.lower()
        if " " in kw_lower:
            if kw_lower in text_lower:
                return True
        else:
            if kw_lower in words:
                return True
    return False


def classify_segment_type(
    segment: EGTSegment,
    total_duration: float,
    segment_index: int,
    total_segments: int,
) -> str:
    """Determine segment_type based on rules — no LLM call.

    Decision hierarchy:
    1. SILENCE: no transcript text and no meaningful visual description
    2. INTRO: first 15% of footage + intro keyword detection
    3. OUTRO: last 15% of footage + outro keyword detection
    4. B_ROLL: minimal/no speech with visual content
    5. SPEECH: everything else
    """
    text = segment.transcript.strip()
    visual = segment.visual_description.strip()
    duration = segment.duration_sec

    # Position-based weighting
    position_ratio = segment.start_sec / max(total_duration, 1.0)
    is_early = position_ratio < 0.15
    is_late = position_ratio > 0.85

    # SILENCE: no speech, very short, or only noise annotations
    has_meaningful_text = len(text) > 0 and not NOISE_BRACKET_PATTERN.fullmatch(text.strip())
    word_count = len(re.findall(r'\b\w+\b', text))

    if not has_meaningful_text and duration < 3.0:
        return "SILENCE"

    if word_count == 0 and duration < 2.0:
        return "SILENCE"

    # INTRO: early position + keyword match
    if is_early and _contains_keyword(text, INTRO_KEYWORDS):
        return "INTRO"

    # OUTRO: late position + keyword match
    if is_late and _contains_keyword(text, OUTRO_KEYWORDS):
        return "OUTRO"

    # B_ROLL: visual content but minimal speech
    words_per_second = word_count / max(duration, 0.1)
    if words_per_second < 0.5 and word_count < 5:
        # Very little speech — likely B-roll
        if visual and visual != "Visual description unavailable.":
            return "B_ROLL"
        # If no visual description either but still low speech, classify as B_ROLL
        if word_count == 0:
            return "B_ROLL"

    return "SPEECH"


def compute_quality_score(segment: EGTSegment) -> tuple:
    """Compute an absolute quality score (0.0–1.0) and quality flags.

    Multi-signal heuristic for Phase 0:
    - Audio/speech presence and density
    - Transcript word density (disfluency detection)
    - Duration plausibility (very short = likely bad take)
    - Bad-take phrase detection

    Returns:
        (quality_score: float, quality_flags: List[str])
    """
    score = 1.0
    flags = []
    text = segment.transcript.strip()
    duration = segment.duration_sec
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)

    # --- Signal 1: Duration plausibility ---
    if duration < 1.5:
        score -= 0.30
        flags.append("very_short")
    elif duration < 3.0:
        score -= 0.10
        flags.append("short")

    # --- Signal 2: Disfluency ratio ---
    if word_count > 0:
        disfluency_count = sum(1 for w in words if w in DISFLUENCY_WORDS)
        disfluency_ratio = disfluency_count / word_count

        if disfluency_ratio > 0.5:
            # More than half the words are disfluencies
            score -= 0.35
            flags.append("high_disfluency")
        elif disfluency_ratio > 0.25:
            score -= 0.15
            flags.append("moderate_disfluency")

        # All words are disfluencies — almost certainly a bad take
        if word_count > 0 and set(words).issubset(DISFLUENCY_WORDS):
            score -= 0.25
            flags.append("only_disfluencies")

    # --- Signal 3: Bad-take phrase detection ---
    if _contains_keyword(text, BAD_TAKE_PHRASES):
        score -= 0.30
        flags.append("bad_take_phrase")

    # --- Signal 4: No speech content ---
    # For SPEECH-typed segments, having no transcript is a quality issue
    if segment.segment_type == "SPEECH" and word_count == 0:
        score -= 0.25
        flags.append("low_audio")

    # --- Signal 5: Background noise annotations ---
    if NOISE_BRACKET_PATTERN.search(text):
        score -= 0.15
        flags.append("background_noise")

    # --- Signal 6: Very low word density (for SPEECH segments) ---
    if segment.segment_type == "SPEECH" and duration > 5.0:
        words_per_second = word_count / duration
        if words_per_second < 0.3:
            score -= 0.15
            flags.append("low_speech_density")

    # Clamp to [0.0, 1.0]
    score = max(0.0, min(1.0, score))

    return score, flags


def score_segments(
    segments: List[EGTSegment],
    total_duration: float,
    context_doc: str = "",
    quality_threshold: float = 0.35
) -> List[EGTSegment]:
    """Score and classify all EGT segments.

    Phase 1 Update:
    1. Assign segment_type and structural_cue using Gemini Flash Lite.
    2. Compute absolute quality_score + quality_flags using rule-based SNR heuristics.
    3. Set is_bad_take based on config.quality_threshold.

    Returns the same list of EGTSegment objects, mutated in place.
    """
    threshold = quality_threshold
    logger.info(
        f"Scoring {len(segments)} segments "
        f"(quality_threshold={threshold:.2f})..."
    )

    bad_take_count = 0

    # Step 1: Semantic Classification via LLM
    from app.utils.llm import classify_egt_segments
    # We pass the segments as dicts to the LLM, then merge back the results
    segment_dicts = [seg.model_dump() for seg in segments]
    classified_dicts = classify_egt_segments(segment_dicts, context_doc)
    
    # Merge results
    for idx, (seg, classified) in enumerate(zip(segments, classified_dicts)):
        # Apply semantic classification
        seg.segment_type = classified.get("segment_type", "SPEECH")
        seg.structural_cue = classified.get("structural_cue")
        seg.perception_model = classified.get("perception_model", "rule-based-v0")
        
        # If semantic classification fails, use the old rule-based fallback
        if seg.perception_model == "rule-based-v0":
            seg.segment_type = classify_segment_type(seg, total_duration, idx, len(segments))

        # Step 2: Compute quality score (still rule-based for audio SNR)
        score, flags = compute_quality_score(seg)
        seg.quality_score = round(score, 3)
        seg.quality_flags = flags

        # Step 3: Determine bad-take status
        seg.is_bad_take = seg.quality_score < threshold
        if seg.is_bad_take:
            if "bad_take" not in seg.quality_flags:
                seg.quality_flags.append("bad_take")
            bad_take_count += 1

    logger.info(
        f"Scoring complete. "
        f"Bad takes: {bad_take_count}/{len(segments)} "
        f"(threshold={threshold:.2f})"
    )

    # Log type distribution
    type_counts = {}
    for seg in segments:
        type_counts[seg.segment_type] = type_counts.get(seg.segment_type, 0) + 1
    logger.info(f"Segment type distribution: {type_counts}")

    return segments

def recompute_bad_takes(segments: List[EGTSegment], quality_threshold: float) -> List[EGTSegment]:
    """Re-evaluate the `is_bad_take` flag for all segments based on a new threshold.
    
    This is extremely fast because it doesn't re-run LLMs or text analysis;
    it just compares the existing `quality_score` to the new threshold.
    """
    logger.info(f"Recomputing bad takes with new threshold: {quality_threshold:.2f}")
    bad_take_count = 0
    for seg in segments:
        seg.is_bad_take = seg.quality_score < quality_threshold
        
        # Manage the 'bad_take' flag in the quality_flags list
        if seg.is_bad_take and "bad_take" not in seg.quality_flags:
            seg.quality_flags.append("bad_take")
        elif not seg.is_bad_take and "bad_take" in seg.quality_flags:
            seg.quality_flags.remove("bad_take")
            
        if seg.is_bad_take:
            bad_take_count += 1
            
    logger.info(f"Recompute complete. Bad takes: {bad_take_count}/{len(segments)}")
    return segments
