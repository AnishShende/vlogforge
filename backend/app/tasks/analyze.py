import logging
from typing import List, Dict
from app.utils.llm import describe_keyframe, synthesize_context
from app.models import EGTSegment

logger = logging.getLogger("VlogForge.Analyze")

# Number of scenes to sample for context synthesis.
# Uniform strided sampling ensures global coverage rather than intro-bias.
CONTEXT_SAMPLE_COUNT = 10


def analyze_segments(
    segments: List[EGTSegment],
    user_context: str,
) -> Dict:
    """Run visual description for each EGT segment and generate a synthesized Context Document.

    Strided Sampling Engine:
    The Context Document is built from a uniform stride-sampled subset of segments rather
    than just the first 10. Stride S = len(segments) / CONTEXT_SAMPLE_COUNT, so samples
    are drawn at indices 0, S, 2S, ... covering the full timeline evenly. This prevents
    the AI from being biased toward introductory content on long vlogs.

    Returns:
        dict with:
            - segments: List[EGTSegment] (mutated with visual_description and tags)
            - context_summary: str (synthesized context document)
    """
    logger.info(f"Analyzing {len(segments)} segments...")

    # Step 1: Describe each segment's keyframe and extract tags
    for seg in segments:
        kf_path = seg.keyframe_path
        if kf_path:
            logger.info(f"Analyzing keyframe: {kf_path}")
            description = describe_keyframe(kf_path, user_context)
        else:
            description = "Visual description unavailable."

        seg.visual_description = description

        # Extract basic tags from description (lightweight — no LLM needed)
        seg.tags = _extract_tags_from_description(description, seg.transcript)

    # Step 2: Uniform strided sampling for context synthesis
    n = len(segments)
    if n <= CONTEXT_SAMPLE_COUNT:
        sampled_segments = segments
    else:
        stride = n / CONTEXT_SAMPLE_COUNT
        sampled_indices = [int(round(i * stride)) for i in range(CONTEXT_SAMPLE_COUNT)]
        # Clamp to valid range; dict.fromkeys preserves order and removes duplicates
        sampled_indices = list(dict.fromkeys(min(idx, n - 1) for idx in sampled_indices))
        sampled_segments = [segments[i] for i in sampled_indices]

    logger.info(
        f"Context synthesis sampling: {len(sampled_segments)} segments sampled "
        f"(stride={n / max(1, CONTEXT_SAMPLE_COUNT):.1f}) from {n} total."
    )

    sampled_transcripts = [
        {"video_file": s.source_file, "start": s.start_sec, "end": s.end_sec, "text": s.transcript}
        for s in sampled_segments
    ]
    sampled_visuals = [s.visual_description for s in sampled_segments]

    # Step 3: Synthesize overall context document from strided samples
    logger.info("Synthesizing context document from strided segment samples...")
    context_summary = synthesize_context(sampled_transcripts, sampled_visuals, user_context)

    return {
        "segments": segments,
        "context_summary": context_summary,
    }


def _extract_tags_from_description(description: str, transcript: str) -> List[str]:
    """Extract lightweight subject/action tags from visual description and transcript.

    Rule-based for Phase 0 — no LLM call. Tags are used in Phase 1 for semantic
    matching (e.g. adlib-to-B-roll retrieval in travel vlogs).
    """
    tags = []
    desc_lower = description.lower()
    text_lower = transcript.lower()
    combined = f"{desc_lower} {text_lower}"

    # Person/subject detection
    person_cues = ["person", "vlogger", "speaker", "face", "man", "woman", "talking", "selfie"]
    if any(cue in combined for cue in person_cues):
        tags.append("person_speaking")

    # Environment tags
    if any(w in combined for w in ["outdoor", "outside", "street", "park", "sky", "nature", "mountain", "beach"]):
        tags.append("outdoor")
    if any(w in combined for w in ["indoor", "inside", "room", "home", "kitchen", "living"]):
        tags.append("indoor")

    # Activity tags
    if any(w in combined for w in ["gym", "workout", "exercise", "weight", "dumbbell", "squat"]):
        tags.append("gym_activity")
    if any(w in combined for w in ["travel", "explore", "trip", "landmark", "tourist", "hotel"]):
        tags.append("travel_activity")
    if any(w in combined for w in ["food", "eating", "cooking", "restaurant", "meal"]):
        tags.append("food")

    # Visual quality cues
    if any(w in desc_lower for w in ["scenic", "view", "landscape", "panorama", "sunset", "sunrise"]):
        tags.append("scenic")
    if any(w in desc_lower for w in ["dark", "dim", "low light"]):
        tags.append("low_light")

    return tags
