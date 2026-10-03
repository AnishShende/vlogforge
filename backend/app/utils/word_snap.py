"""Word-boundary-aware cut-point snapping.

Post-processes EDL start_sec/end_sec values to align with word boundaries,
preventing clipped syllables and tail-bleed from rejected takes.

Usage:
    Called as a post-pass after EDL generation, before assembly.
"""

import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger("VlogForge.WordSnap")


def snap_to_word_boundaries(
    start_sec: float,
    end_sec: float,
    word_timings: List[Dict],
    pre_pad: float = 0.15,
    post_pad: float = 0.35,
) -> Tuple[float, float]:
    """Snap cut points to nearest word boundaries with breathing room.

    Adjusts start_sec to be `pre_pad` seconds before the first word
    that falls within the range, and end_sec to be `post_pad` seconds
    after the last word.

    Args:
        start_sec: Original cut-in point.
        end_sec: Original cut-out point.
        word_timings: List of {word, start, end} dicts for the segment.
        pre_pad: Seconds of padding before the first word.
        post_pad: Seconds of padding after the last word.

    Returns:
        (snapped_start, snapped_end) tuple.  Falls back to original
        values if no words are found in range.
    """
    if not word_timings:
        return start_sec, end_sec

    # Find words whose midpoint falls within [start_sec, end_sec]
    words_in_range = [
        w for w in word_timings
        if w.get("end", 0) > start_sec and w.get("start", 0) < end_sec
    ]

    if not words_in_range:
        return start_sec, end_sec

    first_word = words_in_range[0]
    last_word = words_in_range[-1]

    snapped_start = max(0.0, first_word["start"] - pre_pad)
    snapped_end = last_word["end"] + post_pad

    # Don't expand beyond the original range by more than the padding
    # (safety clamp to avoid pulling in content from outside the segment)
    snapped_start = max(snapped_start, start_sec - pre_pad)
    snapped_end = min(snapped_end, end_sec + post_pad)

    logger.debug(
        f"Word-snap: [{start_sec:.3f}-{end_sec:.3f}] → "
        f"[{snapped_start:.3f}-{snapped_end:.3f}] "
        f"(first='{first_word.get('word', '')}' at {first_word['start']:.3f}, "
        f"last='{last_word.get('word', '')}' ends {last_word['end']:.3f})"
    )

    return snapped_start, snapped_end


def check_tail_bleed(
    clip_start: float,
    prev_segment_word_timings: List[Dict],
    safety_gap: float = 0.05,
) -> float:
    """Check if clip_start overlaps with the last word of the previous segment.

    If the previous (rejected or different) segment's last word bleeds into
    this clip's start time, push the start forward past that word.

    Args:
        clip_start: Proposed start time for the current clip.
        prev_segment_word_timings: word_timings from the previous segment
                                    in the source file timeline.
        safety_gap: Minimum gap (seconds) after the previous word ends.

    Returns:
        Adjusted clip_start (same or later than input).
    """
    if not prev_segment_word_timings:
        return clip_start

    last_prev_word = prev_segment_word_timings[-1]
    last_word_end = last_prev_word.get("end", 0.0)

    if clip_start < last_word_end:
        adjusted = last_word_end + safety_gap
        logger.info(
            f"Tail-bleed detected: clip starts at {clip_start:.3f}s but "
            f"previous word '{last_prev_word.get('word', '')}' ends at "
            f"{last_word_end:.3f}s. Adjusting start to {adjusted:.3f}s."
        )
        return adjusted

    return clip_start


def snap_edl_to_word_boundaries(
    edl_entries: List[Dict],
    segments_by_clip_id: Dict[str, Dict],
    pre_pad: float = 0.15,
    post_pad: float = 0.35,
) -> List[Dict]:
    """Post-process an entire EDL to snap all cut points to word boundaries.

    Args:
        edl_entries: List of EDL entry dicts with start_sec, end_sec, clip_id.
        segments_by_clip_id: Mapping of clip_id → EGTSegment dict (with word_timings).
        pre_pad: Seconds of padding before first word.
        post_pad: Seconds of padding after last word.

    Returns:
        The same edl_entries list, mutated in place with snapped times.
    """
    snapped_count = 0

    for entry in edl_entries:
        clip_id = entry.get("clip_id", "")
        segment = segments_by_clip_id.get(clip_id)

        if not segment:
            continue

        word_timings = segment.get("word_timings", [])
        if not word_timings:
            continue

        old_start = entry["start_sec"]
        old_end = entry["end_sec"]

        new_start, new_end = snap_to_word_boundaries(
            old_start, old_end, word_timings,
            pre_pad=pre_pad, post_pad=post_pad,
        )

        if new_start != old_start or new_end != old_end:
            entry["start_sec"] = new_start
            entry["end_sec"] = new_end
            snapped_count += 1

    if snapped_count > 0:
        logger.info(f"Word-snap post-pass: adjusted {snapped_count}/{len(edl_entries)} EDL entries.")

    return edl_entries
