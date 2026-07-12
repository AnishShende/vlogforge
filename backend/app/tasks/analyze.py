import logging
from typing import List, Dict
from app.utils.llm import describe_keyframe, synthesize_context
from app.models import EGTSegment

logger = logging.getLogger("VlogForge.Analyze")

# Number of scenes to sample for context synthesis.
# Uniform strided sampling ensures global coverage rather than intro-bias.
CONTEXT_SAMPLE_COUNT = 10


import os
from app.utils.ffmpeg import extract_keyframe

def analyze_segments(
    segments: List[EGTSegment],
    user_context: str,
    job_dir: str = "",
    files_info: List[Dict] = None
) -> Dict:
    """Run visual description for each EGT segment and generate a synthesized Context Document.
    """
    logger.info(f"Analyzing {len(segments)} segments...")

    files_info = files_info or []
    cfr_lookup = {f.get("filename"): f.get("cfr_path") for f in files_info}
    keyframes_dir = os.path.join(job_dir, "keyframes") if job_dir else ""

    # Step 1: Describe each segment's keyframe and extract tags
    for seg in segments:
        duration = seg.end_sec - seg.start_sec
        
        # Dense sampling for long scenes
        if duration > 10.0 and job_dir and seg.source_file in cfr_lookup:
            logger.info(f"Dense sampling for long segment ({duration:.1f}s): {seg.clip_id}")
            cfr_path = cfr_lookup[seg.source_file]
            
            # Sample every 5 seconds, starting at 2.5s
            timestamps = []
            t = seg.start_sec + 2.5
            while t < seg.end_sec:
                timestamps.append(t)
                t += 5.0
            
            if not timestamps:
                # Fallback to midpoint if duration math somehow fails
                timestamps = [seg.start_sec + duration / 2.0]
                
            dense_paths = []
            valid_t = []
            for t in timestamps:
                kf_filename = f"{seg.clip_id}_dense_{t:.0f}.jpg"
                kf_path = os.path.join(keyframes_dir, kf_filename)
                
                if extract_keyframe(cfr_path, t, kf_path):
                    dense_paths.append(kf_path)
                    valid_t.append(t)
                    seg.keyframe_paths.append(kf_path)
                else:
                    logger.warning(f"Failed to extract dense keyframe at {t}s for {seg.clip_id}")
            
            if dense_paths:
                from app.utils.llm import describe_keyframes_batch
                batch_descs = describe_keyframes_batch(dense_paths, user_context)
                
                descriptions = []
                for t, desc in zip(valid_t, batch_descs):
                    rel_t = t - seg.start_sec
                    descriptions.append(f"[{rel_t:.1f}s] {desc}")
                    
                description = " Timeline: " + " | ".join(descriptions)
            else:
                description = "Visual description unavailable."
        else:
            # Standard single keyframe logic
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
