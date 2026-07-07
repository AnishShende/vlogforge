import logging
from typing import List, Dict
from app.utils.llm import classify_segments

logger = logging.getLogger("VlogForge.Classify")

def classify_scenes(context_document: str, scenes: List[Dict]) -> List[Dict]:
    """Call LLM classifier to categorize each scene segment.

    Passes the full scene list as `full_scenes` so the classifier can build
    a rolling 180-second context window per segment (Fix 4).
    """
    logger.info("Classifying segments into INTRO, OUTRO, HIGHLIGHT, FILLER, B_ROLL...")

    # Prepare scene list simplified for LLM input
    input_scenes = []
    for s in scenes:
        input_scenes.append({
            "video_file": s["video_file"],
            "start": s["start"],
            "end": s["end"],
            "text": s.get("text", ""),
            "visual_description": s.get("visual_description", "")
        })

    try:
        # Pass full_scenes so the rolling window builder has the complete timeline
        classified = classify_segments(context_document, input_scenes, full_scenes=input_scenes)

        # Merge back any original properties (like keyframe path or full paths)
        merged = []
        for orig, cls in zip(scenes, classified):
            m = orig.copy()
            m["label"] = cls.get("label", "HIGHLIGHT")
            m["score"] = cls.get("score", 1.0)
            merged.append(m)

        return merged
    except Exception as e:
        logger.error(f"Classification failed: {e}. Falling back to default HIGHLIGHT.")
        # Default fallback
        for s in scenes:
            s["label"] = "HIGHLIGHT"
            s["score"] = 0.5
        return scenes
