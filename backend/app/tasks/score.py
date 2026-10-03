"""Pass 1.4 — Quality Scoring & Segment Type Classification.

This module replaces the former classify.py which used LLM-based editorial judgment
(HIGHLIGHT vs FILLER). In the Phase 0 architecture, this is strictly perception-layer:

    segment_type: what IS the footage  (INTRO | OUTRO | SPEECH | B_ROLL | SILENCE)
    quality_score: calibrated absolute  (0.0–1.0, not relative ranking)
    is_bad_take:   quality_score < threshold

Classification priority:
    1. JEV System One (if TYPESAFE_API_KEY is set) — typed probabilities, ~100ms
    2. Gemini Flash Lite batch classification (if JEV unavailable)
    3. Rule-based heuristics (if both AI systems fail)
"""

import re
import logging
from typing import List

from app.models import EGTSegment
from app.config import settings
from app.utils.llm import classify_egt_segments, classify_egt_segments_batch
from app.utils.jev import (
    jev_available,
    classify_segments_jev_batch,
    BAD_TAKE_NOUL_THRESHOLD,
    BACKGROUND_NOISE_NOUL_THRESHOLD,
    STRUCTURAL_CUE_NOUL_THRESHOLD,
    JEV_QUALITY_WEIGHT,
    RULE_QUALITY_WEIGHT,
    QUALITY_SCORE_LEVELS,
)

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

    # INTRO: keyword match
    if _contains_keyword(text, INTRO_KEYWORDS):
        return "INTRO"

    # OUTRO: keyword match
    if _contains_keyword(text, OUTRO_KEYWORDS):
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

    # --- Signal 1.5: Fragmented Short Speech ---
    # 1 to 3 words total on a short segment usually means a stumble or incomplete utterance
    if segment.segment_type == "SPEECH" and 0 < word_count <= 3 and duration < 3.0:
        score -= 0.70  # Aggressively penalize to force below threshold
        flags.append("short_speech_fragment")

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


def _apply_jev_results(
    segments: List[EGTSegment],
    jev_results: list,
    total_duration: float,
    quality_threshold: float,
) -> List[EGTSegment]:
    """Merge JEV classification results into EGT segments.

    For segments where JEV succeeded, blend JEV's content_quality Score
    with rule-based heuristic signals. For segments where JEV returned None,
    fall through to rule-based classification.

    JEV provides:
        - segment_type (Choice) — replaces Gemini classification
        - is_bad_take (Noul) — replaces keyword-list detection
        - is_background_noise (Noul) — replaces regex pattern matching
        - content_quality (Score 0-4, normalized to 0.0-1.0) — blended with heuristics
        - has_structural_cue (Noul) — detects narrative transitions
    """
    bad_take_count = 0

    for idx, (seg, jev) in enumerate(zip(segments, jev_results)):
        if jev is not None:
            # === JEV succeeded: use its typed outputs ===
            seg.segment_type = jev["segment_type"]
            seg.perception_model = jev["perception_model"]

            # Structural cue: JEV can't generate text, but it tells us if one exists
            if jev["has_structural_cue_noul"] > STRUCTURAL_CUE_NOUL_THRESHOLD:
                seg.structural_cue = "detected_by_jev"

            # Quality scoring: blend JEV Score with rule-based heuristics
            _, rule_flags = compute_quality_score(seg)
            rule_score_raw = 1.0
            # Recompute a lightweight rule score from the flags
            for flag in rule_flags:
                if flag == "very_short":
                    rule_score_raw -= 0.30
                elif flag == "short":
                    rule_score_raw -= 0.10
                elif flag == "short_speech_fragment":
                    rule_score_raw -= 0.70
                elif flag == "high_disfluency":
                    rule_score_raw -= 0.35
                elif flag == "moderate_disfluency":
                    rule_score_raw -= 0.15
                elif flag == "only_disfluencies":
                    rule_score_raw -= 0.25
                elif flag == "bad_take_phrase":
                    rule_score_raw -= 0.30
                elif flag == "low_audio":
                    rule_score_raw -= 0.25
                elif flag == "background_noise":
                    rule_score_raw -= 0.15
                elif flag == "low_speech_density":
                    rule_score_raw -= 0.15
            rule_score_raw = max(0.0, min(1.0, rule_score_raw))

            # Blend: 70% JEV, 30% rule-based
            blended_score = (
                JEV_QUALITY_WEIGHT * jev["content_quality_normalized"]
                + RULE_QUALITY_WEIGHT * rule_score_raw
            )

            # Apply JEV-specific flags
            quality_flags = list(rule_flags)  # start with rule-based flags

            if jev["is_bad_take_noul"] > BAD_TAKE_NOUL_THRESHOLD:
                if "jev_bad_take" not in quality_flags:
                    quality_flags.append("jev_bad_take")
                # Bad take penalty on the blended score
                blended_score = min(blended_score, 0.25)

            if jev["is_background_noise_noul"] > BACKGROUND_NOISE_NOUL_THRESHOLD:
                if "jev_background_noise" not in quality_flags:
                    quality_flags.append("jev_background_noise")

            seg.quality_score = round(max(0.0, min(1.0, blended_score)), 3)
            seg.quality_flags = quality_flags

        else:
            # === JEV failed: fallback to rule-based classification ===
            seg.perception_model = "rule-based-v0"
            seg.segment_type = classify_segment_type(seg, total_duration, idx, len(segments))

            score, flags = compute_quality_score(seg)
            seg.quality_score = round(score, 3)
            seg.quality_flags = flags

        # Determine bad-take status
        seg.is_bad_take = seg.quality_score < quality_threshold
        if seg.is_bad_take:
            if "bad_take" not in seg.quality_flags:
                seg.quality_flags.append("bad_take")
            bad_take_count += 1

    return segments, bad_take_count


def score_segments(
    segments: List[EGTSegment],
    total_duration: float,
    context_doc: str = "",
    quality_threshold: float = 0.35,
    progress_callback=None  # Optional callable(done: int, total: int)
) -> List[EGTSegment]:
    """Score and classify all EGT segments.

    Classification priority:
    1. JEV System One (if available) — typed probabilities, ~100ms per segment
    2. Gemini Flash Lite batch classification (if JEV unavailable)
    3. Rule-based heuristics (if both AI systems fail)

    Returns the same list of EGTSegment objects, mutated in place.
    """
    threshold = quality_threshold
    logger.info(
        f"Scoring {len(segments)} segments "
        f"(quality_threshold={threshold:.2f})..."
    )

    # === Path 1: JEV System One (preferred) ===
    if jev_available():
        logger.info("JEV available — using TypeSafe System One for classification.")
        segment_dicts = [seg.model_dump() for seg in segments]
        jev_results = classify_segments_jev_batch(
            segment_dicts, context_doc=context_doc,
            progress_callback=progress_callback,
        )

        # Count how many JEV succeeded
        jev_success = sum(1 for r in jev_results if r is not None)
        logger.info(f"JEV classified {jev_success}/{len(segments)} segments successfully.")

        segments, bad_take_count = _apply_jev_results(
            segments, jev_results, total_duration, threshold
        )
    else:
        # === Path 2: Gemini batch classification (fallback) ===
        logger.info("JEV unavailable — falling back to Gemini batch classification.")
        bad_take_count = 0

        segment_dicts = [seg.model_dump() for seg in segments]
        classified_dicts = classify_egt_segments_batch(
            segment_dicts, context_doc, progress_callback=progress_callback
        )

        # Merge results
        for idx, (seg, classified) in enumerate(zip(segments, classified_dicts)):
            # Apply semantic classification
            seg.segment_type = classified.get("segment_type", "SPEECH")
            seg.structural_cue = classified.get("structural_cue")

            perception_model = classified.get("perception_model", "")
            # If semantic classification wasn't run or explicitly failed, fallback
            if not perception_model or perception_model == "rule-based-v0":
                seg.perception_model = "rule-based-v0"
                seg.segment_type = classify_segment_type(seg, total_duration, idx, len(segments))
            else:
                seg.perception_model = perception_model

            # Compute quality score (rule-based)
            score, flags = compute_quality_score(seg)
            seg.quality_score = round(score, 3)
            seg.quality_flags = flags

            # Determine bad-take status
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
