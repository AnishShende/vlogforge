"""M5 Metadata Generation — YouTube-optimized title, description, tags, chapters.

Single LLM call using the perception-tier model to generate all metadata fields
from the context summary, transcript, and EDL clip sequence.
"""

import json
import logging
import re
from typing import Dict, List, Optional

from app.config import settings
from app.utils.llm import safe_generate_content, init_gemini

logger = logging.getLogger("VlogForge.Metadata")


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to YouTube chapter format (M:SS or H:MM:SS)."""
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _build_edl_summary(edl: List[Dict]) -> str:
    """Build a concise timeline summary from EDL entries for the LLM prompt."""
    lines = []
    cumulative_time = 0.0
    for i, entry in enumerate(edl):
        start = entry.get("start_sec", 0)
        end = entry.get("end_sec", 0)
        duration = end - start
        clip_type = entry.get("editorial_type") or entry.get("type", "KEEP")
        clip_id = entry.get("clip_id", f"clip_{i}")

        timestamp = _format_timestamp(cumulative_time)
        lines.append(
            f"  [{timestamp}] Clip {i+1} ({clip_type}): "
            f"{duration:.1f}s from source at {start:.1f}-{end:.1f}s"
        )
        cumulative_time += duration

    return "\n".join(lines)


def _extract_transcript_text(transcript_segments: List[Dict], max_chars: int = 3000) -> str:
    """Extract plain text from transcript segments, truncated for prompt size."""
    texts = []
    total = 0
    for seg in transcript_segments:
        text = seg.get("text", "").strip()
        if text and total + len(text) < max_chars:
            texts.append(text)
            total += len(text)
    return " ".join(texts)


def generate_metadata(
    context_summary: str,
    transcript_segments: List[Dict],
    edl: List[Dict],
    target_duration: float,
    user_prompt: str = "",
) -> Dict:
    """Generate YouTube-ready metadata using LLM.

    Args:
        context_summary: Synthesized context document from Pass 1.
        transcript_segments: Aligned speech transcript segments.
        edl: Final Edit Decision List (list of dicts with start_sec, end_sec, etc.).
        target_duration: Target video duration in seconds.
        user_prompt: Original user creative direction text.

    Returns:
        Dict with keys: title, description, tags, chapters.
    """
    logger.info("Generating YouTube metadata (M5)...")

    # Build inputs for the prompt
    edl_summary = _build_edl_summary(edl)
    transcript_text = _extract_transcript_text(transcript_segments)
    total_duration = sum(
        (e.get("end_sec", 0) - e.get("start_sec", 0)) for e in edl
    )
    duration_str = _format_timestamp(total_duration)

    prompt = f"""You are a YouTube optimization expert. Based on the following vlog information, generate metadata that will maximize discoverability and engagement.

## Vlog Context
{context_summary[:2000] if context_summary else "No context provided."}

## Creator's Direction
{user_prompt[:500] if user_prompt else "No specific direction."}

## Transcript Excerpt
{transcript_text[:2000] if transcript_text else "No transcript available."}

## Video Timeline ({duration_str} total, {len(edl)} clips)
{edl_summary}

## Instructions
Generate the following in valid JSON format:

1. **title**: A compelling YouTube title (max 100 characters). Use power words, be specific, avoid clickbait. Include relevant keywords naturally.

2. **description**: A YouTube description (300-500 words) with:
   - Opening hook paragraph (first 2 lines are visible before "Show More")
   - Brief content summary paragraph
   - Chapter timestamps section (use the timeline above to create meaningful chapter names)
   - Hashtags line (3-5 relevant hashtags)
   - A placeholder line: "Follow me: [Your Social Links]"

3. **tags**: A JSON array of 15-25 YouTube tags. Mix broad terms (e.g., "vlog", "travel") with specific long-tail keywords. Include variations.

4. **chapters**: A JSON array of chapter objects with "time" (string like "0:00") and "label" (short descriptive name). First chapter MUST be "0:00". Create 4-10 chapters based on major content transitions in the timeline.

Respond with ONLY a valid JSON object:
```json
{{
  "title": "...",
  "description": "...",
  "tags": ["...", "..."],
  "chapters": [{{"time": "0:00", "label": "..."}}, ...]
}}
```"""

    if not init_gemini():
        logger.warning("Gemini not available. Using mechanical fallback for metadata.")
        return _mechanical_fallback(context_summary, transcript_text, edl, total_duration)

    try:
        response = safe_generate_content(
            model=settings.perception_model,
            contents=prompt,
            priority=1,
            call_name="Metadata Generation"
        )
        raw_text = response.text.strip()

        # Extract JSON from markdown code fences if present
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if json_match:
            raw_text = json_match.group(1)

        metadata = json.loads(raw_text)

        # Validate and sanitize
        result = {
            "title": str(metadata.get("title", ""))[:100],
            "description": str(metadata.get("description", "")),
            "tags": [str(t) for t in metadata.get("tags", []) if t][:30],
            "chapters": [],
        }

        # Validate chapters
        for ch in metadata.get("chapters", []):
            if isinstance(ch, dict) and "time" in ch and "label" in ch:
                result["chapters"].append({
                    "time": str(ch["time"]),
                    "label": str(ch["label"])[:80],
                })

        # Ensure first chapter is 0:00
        if result["chapters"] and result["chapters"][0]["time"] != "0:00":
            result["chapters"].insert(0, {"time": "0:00", "label": "Intro"})

        logger.info(
            f"Metadata generated: title='{result['title'][:50]}...', "
            f"{len(result['tags'])} tags, {len(result['chapters'])} chapters"
        )
        return result

    except Exception as e:
        logger.error(f"LLM metadata generation failed: {e}. Using mechanical fallback.")
        return _mechanical_fallback(context_summary, transcript_text, edl, total_duration)


def _mechanical_fallback(
    context_summary: str,
    transcript_text: str,
    edl: List[Dict],
    total_duration: float,
) -> Dict:
    """Generate basic metadata without LLM when API is unavailable."""
    # Title: first meaningful sentence from context or transcript
    title = "My Vlog"
    if context_summary:
        first_line = context_summary.split(".")[0].strip()
        if len(first_line) > 10:
            title = first_line[:100]
    elif transcript_text:
        first_line = transcript_text.split(".")[0].strip()
        if len(first_line) > 10:
            title = first_line[:100]

    # Description
    duration_str = _format_timestamp(total_duration)
    description = (
        f"{title}\n\n"
        f"Duration: {duration_str} | {len(edl)} clips\n\n"
        f"Edited with VlogForge AI.\n\n"
        f"#vlog #youtube #vlogforge"
    )

    # Tags from common vlog categories
    tags = ["vlog", "youtube", "daily vlog", "lifestyle", "vlogforge"]

    # Chapters from EDL sequence
    chapters = [{"time": "0:00", "label": "Intro"}]
    cumulative = 0.0
    for i, entry in enumerate(edl):
        duration = entry.get("end_sec", 0) - entry.get("start_sec", 0)
        cumulative += duration
        clip_type = entry.get("editorial_type") or entry.get("type", "KEEP")
        if clip_type in ("OUTRO",) or (i > 0 and i % max(1, len(edl) // 5) == 0):
            chapters.append({
                "time": _format_timestamp(cumulative),
                "label": f"Part {len(chapters)}" if clip_type == "KEEP" else clip_type.title(),
            })

    logger.info("Mechanical fallback metadata generated.")
    return {
        "title": title,
        "description": description,
        "tags": tags,
        "chapters": chapters,
    }
