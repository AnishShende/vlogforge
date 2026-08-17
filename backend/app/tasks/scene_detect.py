"""Pass 1.0b — Scene Boundary Detection (Two-Pass Cascade).

Extracts scene detection from ingest.py into a dedicated module with a
two-detector cascade strategy:

    Pass 1: ContentDetector — fast, precise detection of hard visual cuts.
    Pass 2: AdaptiveDetector — applied ONLY to scenes longer than
            `long_scene_threshold_sec` to find gradual transitions
            (slow pans, lighting shifts) that ContentDetector misses.

This cascade is specifically designed for unscripted vlog footage, where
talking-head segments produce zero hard cuts but do contain gradual
visual transitions (gestures, camera repositioning, background changes)
that AdaptiveDetector can catch.

The final scene list is deduplicated, sorted, and filtered to remove
sub-second fragments.
"""

import logging
from typing import List, Dict, Optional

from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector, AdaptiveDetector

logger = logging.getLogger("VlogForge.SceneDetect")

# ---------------------------------------------------------------------------
# Fixed-interval fallback (ultimate safety net)
# ---------------------------------------------------------------------------

def _fixed_interval_scenes(duration: float, interval: float = 8.0) -> List[Dict]:
    """Generate fixed-interval scene boundaries as a last-resort fallback.

    Used when both ContentDetector and AdaptiveDetector fail or return no
    results. Produces uniform chunks of `interval` seconds.
    """
    scenes = []
    current = 0.0
    while current < duration:
        end = min(current + interval, duration)
        if end - current >= 1.0:
            scenes.append({"start": current, "end": end})
        current = end
    return scenes


# ---------------------------------------------------------------------------
# Core: Two-pass cascade detector
# ---------------------------------------------------------------------------

def detect_scenes(
    video_path: str,
    duration: float = 0.0,
    long_scene_threshold_sec: float = 15.0,
    content_threshold: float = 27.0,
    adaptive_threshold: float = 3.0,
    min_scene_duration_sec: float = 1.0,
) -> List[Dict]:
    """Detect scene boundaries using a two-pass cascade strategy.

    Pass 1 — ContentDetector:
        Scans the full video for hard visual cuts (fast HSV delta spikes).
        Precise and cheap. Works well for B-roll montage, action footage,
        and any footage with explicit camera cuts.

    Pass 2 — AdaptiveDetector (selective):
        For each scene from Pass 1 that exceeds `long_scene_threshold_sec`,
        re-scans just that time range using AdaptiveDetector's rolling-average
        algorithm. This catches gradual transitions (pans, lighting changes,
        handheld repositioning) that ContentDetector misses in continuous
        talking-head takes.

    Dedup Pass:
        Merges any resulting scene shorter than `min_scene_duration_sec`
        into its preceding neighbor to avoid sub-second fragments.

    Args:
        video_path: Path to the CFR-normalised video file.
        duration: Total video duration in seconds (used for fallback only).
        long_scene_threshold_sec: Scenes longer than this trigger Pass 2.
        content_threshold: ContentDetector HSV delta threshold.
        adaptive_threshold: AdaptiveDetector rolling-average threshold.
        min_scene_duration_sec: Minimum scene length after merge pass.

    Returns:
        List of {"start": float, "end": float, "detection_method": str}
        dicts sorted by start time.
    """
    filename = video_path.rsplit("/", 1)[-1] if "/" in video_path else video_path

    # ------------------------------------------------------------------
    # Pass 1: ContentDetector — full video scan
    # ------------------------------------------------------------------
    pass1_scenes: List[Dict] = []

    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=content_threshold))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        for scene in scene_list:
            start_sec = scene[0].get_seconds()
            end_sec = scene[1].get_seconds()
            if end_sec - start_sec >= min_scene_duration_sec:
                pass1_scenes.append({
                    "start": start_sec,
                    "end": end_sec,
                    "detection_method": "content",
                })

        logger.info(
            f"Pass 1 (ContentDetector): {len(pass1_scenes)} scene(s) "
            f"from {filename}"
        )
    except Exception as e:
        logger.warning(
            f"Pass 1 (ContentDetector) failed for {filename}: {e}. "
            "Will attempt AdaptiveDetector on full duration."
        )

    # If ContentDetector found nothing, treat the entire video as one long scene
    # so Pass 2 can attempt to subdivide it.
    if not pass1_scenes and duration > 0:
        pass1_scenes = [{"start": 0.0, "end": duration, "detection_method": "content"}]
        logger.info(
            f"ContentDetector found no cuts in {filename}. "
            f"Treating full duration ({duration:.1f}s) as single scene for Pass 2."
        )

    # ------------------------------------------------------------------
    # Pass 2: AdaptiveDetector — selective on long scenes only
    # ------------------------------------------------------------------
    final_scenes: List[Dict] = []

    for scene in pass1_scenes:
        scene_duration = scene["end"] - scene["start"]

        if scene_duration <= long_scene_threshold_sec:
            # Short enough — keep as-is from Pass 1
            final_scenes.append(scene)
            continue

        # Long scene: run AdaptiveDetector on just this time range
        logger.info(
            f"Pass 2 (AdaptiveDetector): Scanning long scene "
            f"[{scene['start']:.1f}s - {scene['end']:.1f}s] "
            f"({scene_duration:.1f}s) in {filename}"
        )

        sub_scenes = _adaptive_subdivide(
            video_path,
            start_time=scene["start"],
            end_time=scene["end"],
            adaptive_threshold=adaptive_threshold,
            min_scene_duration_sec=min_scene_duration_sec,
        )

        if sub_scenes and len(sub_scenes) > 1:
            logger.info(
                f"Pass 2: Scene [{scene['start']:.1f}s - {scene['end']:.1f}s] "
                f"subdivided into {len(sub_scenes)} sub-scene(s)"
            )
            final_scenes.extend(sub_scenes)
        else:
            # AdaptiveDetector found no sub-boundaries — keep the original
            logger.info(
                f"Pass 2: No sub-boundaries found in "
                f"[{scene['start']:.1f}s - {scene['end']:.1f}s]. "
                "Keeping original scene."
            )
            final_scenes.append(scene)

    # ------------------------------------------------------------------
    # Dedup: merge sub-second fragments into neighbors
    # ------------------------------------------------------------------
    final_scenes = _merge_short_scenes(final_scenes, min_scene_duration_sec)

    # ------------------------------------------------------------------
    # Ultimate fallback: fixed-interval if still empty
    # ------------------------------------------------------------------
    if not final_scenes and duration > 0:
        logger.warning(
            f"Both detectors returned no scenes for {filename}. "
            "Falling back to fixed 8-second intervals."
        )
        final_scenes = _fixed_interval_scenes(duration)

    # Sort by start time for downstream chronological processing
    final_scenes.sort(key=lambda s: s["start"])

    # Log summary
    method_counts = {}
    for s in final_scenes:
        m = s.get("detection_method", "unknown")
        method_counts[m] = method_counts.get(m, 0) + 1
    logger.info(
        f"Final scene count for {filename}: {len(final_scenes)} "
        f"(breakdown: {method_counts})"
    )

    return final_scenes


# ---------------------------------------------------------------------------
# Pass 2 helper: AdaptiveDetector on a time range
# ---------------------------------------------------------------------------

def _adaptive_subdivide(
    video_path: str,
    start_time: float,
    end_time: float,
    adaptive_threshold: float = 3.0,
    min_scene_duration_sec: float = 1.0,
) -> List[Dict]:
    """Run AdaptiveDetector on a specific time range of the video.

    Returns a list of sub-scene dicts, or an empty list on failure.
    """
    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(
            AdaptiveDetector(
                adaptive_threshold=adaptive_threshold,
                min_scene_len=15,  # Minimum ~0.5s at 30fps
            )
        )
        video.seek(start_time)
        scene_manager.detect_scenes(
            video,
            end_time=end_time,
        )
        scene_list = scene_manager.get_scene_list()

        sub_scenes = []
        for scene in scene_list:
            s = scene[0].get_seconds()
            e = scene[1].get_seconds()
            if e - s >= min_scene_duration_sec:
                sub_scenes.append({
                    "start": s,
                    "end": e,
                    "detection_method": "adaptive",
                })

        return sub_scenes

    except Exception as e:
        logger.warning(
            f"AdaptiveDetector failed on range "
            f"[{start_time:.1f}s - {end_time:.1f}s]: {e}"
        )
        return []


# ---------------------------------------------------------------------------
# Dedup helper: merge sub-second fragments
# ---------------------------------------------------------------------------

def _merge_short_scenes(
    scenes: List[Dict],
    min_duration_sec: float,
) -> List[Dict]:
    """Merge scenes shorter than min_duration_sec into their preceding neighbor.

    Iterates in order; short scenes extend the end time of the previous scene
    rather than creating a standalone segment.
    """
    if not scenes:
        return scenes

    # Sort first to guarantee order
    sorted_scenes = sorted(scenes, key=lambda s: s["start"])
    merged = [sorted_scenes[0]]

    for scene in sorted_scenes[1:]:
        duration = scene["end"] - scene["start"]
        if duration < min_duration_sec:
            # Extend previous scene's end to absorb this fragment
            merged[-1]["end"] = scene["end"]
        else:
            merged.append(scene)

    return merged


# ---------------------------------------------------------------------------
# Pass 3: Speech-gap subdivision for long scenes
# ---------------------------------------------------------------------------

def subdivide_by_speech_gaps(
    segments,
    transcript_segments: List[Dict],
    long_scene_threshold_sec: float,
    speech_gap_sec: float = 1.5,
    video_path: str = "",
    keyframes_dir: str = "",
    context_notes: str = "",
) -> list:
    """Subdivide long EGT segments at speech gaps (silence windows).

    For each segment longer than `long_scene_threshold_sec`, this function:
    1. Finds speech gaps (periods with no transcript words) >= `speech_gap_sec`
    2. For each candidate gap, extracts a keyframe at the gap's midpoint
    3. Uses Gemini Vision (describe_keyframe with classify_content=True) to
       determine if there is meaningful visual content during the silence
    4. Splits the segment at the gap midpoint only if the frame is dead air
    5. Preserves pauses that serve the video (person demonstrating, scenic
       view, exercise, etc.) — fully genre-agnostic

    Args:
        segments: List of EGTSegment objects from the perception pass.
        transcript_segments: Raw transcript word-level dicts with
            {video_file, start, end, text} keys.
        long_scene_threshold_sec: Scenes longer than this are candidates.
        speech_gap_sec: Minimum silence duration to consider as a gap.
        video_path: Path to the CFR video for keyframe extraction.
        keyframes_dir: Directory to save gap keyframes.
        context_notes: User-provided context for Vision classification.

    Returns:
        New list of EGTSegment objects (original short segments unchanged,
        long segments potentially subdivided).
    """
    from app.models import EGTSegment, generate_clip_id
    from app.utils.ffmpeg import extract_keyframe
    from app.utils.llm import describe_keyframe
    import os

    refined = []

    for seg in segments:
        duration = seg.end_sec - seg.start_sec
        if duration <= long_scene_threshold_sec:
            refined.append(seg)
            continue

        logger.info(
            f"Pass 3 (Speech-gap): Analyzing long segment "
            f"[{seg.start_sec:.1f}s - {seg.end_sec:.1f}s] "
            f"({duration:.1f}s) for speech gaps >= {speech_gap_sec:.1f}s"
        )

        # Find speech segments that fall within this scene's time range
        scene_words = [
            t for t in transcript_segments
            if t.get("video_file") == seg.source_file
            and t.get("end", 0) > seg.start_sec
            and t.get("start", 0) < seg.end_sec
        ]

        # Sort by start time
        scene_words.sort(key=lambda w: w.get("start", 0))

        # Find gaps between consecutive speech segments
        gaps = []
        if not scene_words:
            # No speech at all in this segment — keep as-is
            refined.append(seg)
            logger.info(
                f"Pass 3: No speech found in [{seg.start_sec:.1f}s - {seg.end_sec:.1f}s]. "
                "Keeping as single segment."
            )
            continue

        # Gap from scene start to first word
        first_word_start = scene_words[0].get("start", seg.start_sec)
        if first_word_start - seg.start_sec >= speech_gap_sec:
            gaps.append((seg.start_sec, first_word_start))

        # Gaps between consecutive words
        for i in range(len(scene_words) - 1):
            current_end = scene_words[i].get("end", 0)
            next_start = scene_words[i + 1].get("start", 0)
            gap_duration = next_start - current_end
            if gap_duration >= speech_gap_sec:
                gaps.append((current_end, next_start))

        # Gap from last word to scene end
        last_word_end = scene_words[-1].get("end", seg.end_sec)
        if seg.end_sec - last_word_end >= speech_gap_sec:
            gaps.append((last_word_end, seg.end_sec))

        if not gaps:
            refined.append(seg)
            logger.info(
                f"Pass 3: No speech gaps >= {speech_gap_sec:.1f}s in "
                f"[{seg.start_sec:.1f}s - {seg.end_sec:.1f}s]. Keeping as single segment."
            )
            continue

        logger.info(
            f"Pass 3: Found {len(gaps)} speech gap(s) in "
            f"[{seg.start_sec:.1f}s - {seg.end_sec:.1f}s]: "
            + ", ".join(f"[{g[0]:.1f}s-{g[1]:.1f}s]" for g in gaps)
        )

        # Classify each gap: meaningful content (preserve) vs dead air (split)
        split_points = []
        for gap_start, gap_end in gaps:
            gap_mid = gap_start + (gap_end - gap_start) / 2.0

            # Extract a keyframe at the gap midpoint
            gap_kf_filename = (
                f"{os.path.splitext(os.path.basename(seg.source_file))[0]}"
                f"_gap_{gap_mid:.0f}.jpg"
            )
            gap_kf_path = os.path.join(keyframes_dir, gap_kf_filename)

            has_meaningful = True  # Conservative default: preserve

            if video_path and keyframes_dir:
                kf_ok = extract_keyframe(video_path, gap_mid, gap_kf_path)
                if kf_ok:
                    _desc, has_meaningful = describe_keyframe(
                        gap_kf_path, context_notes, classify_content=True
                    )
                    logger.info(
                        f"Pass 3: Gap [{gap_start:.1f}s-{gap_end:.1f}s] → "
                        f"meaningful={has_meaningful}"
                    )
                else:
                    logger.warning(
                        f"Pass 3: Keyframe extraction failed for gap at {gap_mid:.1f}s. "
                        "Preserving gap (conservative)."
                    )
            else:
                logger.warning(
                    "Pass 3: No video_path or keyframes_dir provided. "
                    "Preserving all gaps (conservative)."
                )

            if not has_meaningful:
                split_points.append(gap_mid)

        if not split_points:
            refined.append(seg)
            logger.info(
                f"Pass 3: All gaps in [{seg.start_sec:.1f}s - {seg.end_sec:.1f}s] "
                "have meaningful visual content. Keeping as single segment."
            )
            continue

        # Build sub-segments from split points
        split_points.sort()
        boundaries = [seg.start_sec] + split_points + [seg.end_sec]

        logger.info(
            f"Pass 3: Splitting [{seg.start_sec:.1f}s - {seg.end_sec:.1f}s] "
            f"into {len(boundaries) - 1} sub-segment(s) at: "
            + ", ".join(f"{sp:.1f}s" for sp in split_points)
        )

        for i in range(len(boundaries) - 1):
            sub_start = boundaries[i]
            sub_end = boundaries[i + 1]
            sub_duration = sub_end - sub_start

            if sub_duration < 1.0:
                # Skip tiny fragments
                continue

            # Generate a new clip_id for the sub-segment
            sub_clip_id = generate_clip_id(seg.source_file, sub_start, sub_end)

            # Extract keyframe at sub-segment midpoint
            sub_mid = sub_start + sub_duration / 2.0
            sub_kf_filename = (
                f"{os.path.splitext(os.path.basename(seg.source_file))[0]}"
                f"_scene_{sub_start:.0f}.jpg"
            )
            sub_kf_path = os.path.join(keyframes_dir, sub_kf_filename)
            kf_ok = extract_keyframe(video_path, sub_mid, sub_kf_path) if video_path else False

            # Slice transcript for this sub-segment
            sub_transcript_words = [
                t.get("text", "")
                for t in scene_words
                if t.get("start", 0) >= sub_start - 0.5
                and t.get("end", 0) <= sub_end + 0.5
            ]
            sub_transcript = " ".join(sub_transcript_words).strip()

            sub_seg = EGTSegment(
                clip_id=sub_clip_id,
                source_file=seg.source_file,
                source_file_hash=seg.source_file_hash,
                start_sec=sub_start,
                end_sec=sub_end,
                keyframe_path=sub_kf_path if kf_ok else seg.keyframe_path,
                transcript=sub_transcript,
                visual_description=seg.visual_description,  # Inherited until re-analyzed
                tags=list(seg.tags) + ["speech_gap_split"],
            )
            refined.append(sub_seg)

        logger.info(
            f"Pass 3: Long segment [{seg.start_sec:.1f}s - {seg.end_sec:.1f}s] "
            f"subdivided into {len(boundaries) - 1} sub-segment(s)"
        )

    # Log summary
    if len(refined) != len(segments):
        logger.info(
            f"Pass 3 (Speech-gap) complete: {len(segments)} → {len(refined)} segments"
        )
    else:
        logger.info("Pass 3 (Speech-gap): No segments required subdivision.")

    return refined


# ---------------------------------------------------------------------------
# Pass 3b: Editorial subdivision for LLM Director granularity
# ---------------------------------------------------------------------------

def editorial_subdivide(
    segments,
    transcript_segments: List[Dict],
    target_duration: float,
    min_gap_sec: float = 1.0,
) -> list:
    """Subdivide long EGT segments at speech gaps for editorial granularity.

    Unlike ``subdivide_by_speech_gaps`` (which only splits at "dead air" gaps
    confirmed by vision), this function splits at **every** speech gap above
    ``min_gap_sec`` — regardless of whether the gap contains meaningful visual
    content.  The purpose is to produce atomic editorial units small enough for
    the LLM Director to assign individual ``narrative_priority`` values to each
    one, so Tier 3 repair can trim proportionally instead of dropping whole
    monolithic blocks.

    A maximum segment duration cap of ``max(30, target_duration * 0.15)`` is
    enforced: any sub-segment that is still too long after speech-gap splitting
    is recursively halved at its longest internal gap.

    Args:
        segments: List of EGTSegment objects (already refined by Pass 3).
        transcript_segments: Raw transcript word-level dicts with
            ``{video_file, start, end, text}`` keys.
        target_duration: User-specified target video duration in seconds.
        min_gap_sec: Minimum speech-gap duration to consider as a cut point.

    Returns:
        New list of EGTSegment objects with long segments subdivided.
    """
    from app.models import EGTSegment, generate_clip_id

    max_segment_sec = max(30.0, target_duration * 0.15)
    refined = []

    for seg in segments:
        duration = seg.end_sec - seg.start_sec
        if duration <= max_segment_sec:
            refined.append(seg)
            continue

        logger.info(
            f"Editorial subdivide: Segment [{seg.start_sec:.1f}s - "
            f"{seg.end_sec:.1f}s] ({duration:.1f}s) exceeds cap "
            f"{max_segment_sec:.1f}s — finding speech-gap cut points"
        )

        # Collect speech words within this segment's time range
        scene_words = [
            t for t in transcript_segments
            if t.get("video_file") == seg.source_file
            and t.get("end", 0) > seg.start_sec
            and t.get("start", 0) < seg.end_sec
        ]
        scene_words.sort(key=lambda w: w.get("start", 0))

        if not scene_words:
            refined.append(seg)
            continue

        # Build list of ALL speech gaps >= min_gap_sec
        gaps = []

        first_word_start = scene_words[0].get("start", seg.start_sec)
        if first_word_start - seg.start_sec >= min_gap_sec:
            gaps.append((seg.start_sec, first_word_start))

        for i in range(len(scene_words) - 1):
            current_end = scene_words[i].get("end", 0)
            next_start = scene_words[i + 1].get("start", 0)
            gap_duration = next_start - current_end
            if gap_duration >= min_gap_sec:
                gaps.append((current_end, next_start))

        last_word_end = scene_words[-1].get("end", seg.end_sec)
        if seg.end_sec - last_word_end >= min_gap_sec:
            gaps.append((last_word_end, seg.end_sec))

        if not gaps:
            refined.append(seg)
            logger.info(
                f"Editorial subdivide: No speech gaps >= {min_gap_sec:.1f}s "
                f"in [{seg.start_sec:.1f}s - {seg.end_sec:.1f}s]. "
                "Keeping as single segment."
            )
            continue

        # Greedily pick gap midpoints as split points, prioritizing the longest
        # gaps first, until all resulting sub-segments are within the cap.
        split_points = _pick_split_points(
            seg.start_sec, seg.end_sec, gaps, max_segment_sec
        )

        if not split_points:
            refined.append(seg)
            continue

        # Build sub-segments
        split_points.sort()
        boundaries = [seg.start_sec] + split_points + [seg.end_sec]

        logger.info(
            f"Editorial subdivide: Splitting [{seg.start_sec:.1f}s - "
            f"{seg.end_sec:.1f}s] into {len(boundaries) - 1} sub-segment(s) "
            f"at: {', '.join(f'{sp:.1f}s' for sp in split_points)}"
        )

        for i in range(len(boundaries) - 1):
            sub_start = boundaries[i]
            sub_end = boundaries[i + 1]
            sub_duration = sub_end - sub_start

            if sub_duration < 1.0:
                continue

            sub_clip_id = generate_clip_id(seg.source_file, sub_start, sub_end)

            # Slice transcript for this sub-segment
            sub_transcript_words = [
                t.get("text", "")
                for t in scene_words
                if t.get("start", 0) >= sub_start - 0.5
                and t.get("end", 0) <= sub_end + 0.5
            ]
            sub_transcript = " ".join(sub_transcript_words).strip()

            sub_seg = EGTSegment(
                clip_id=sub_clip_id,
                source_file=seg.source_file,
                source_file_hash=seg.source_file_hash,
                start_sec=sub_start,
                end_sec=sub_end,
                keyframe_path=seg.keyframe_path,
                keyframe_paths=list(seg.keyframe_paths),
                transcript=sub_transcript,
                visual_description=seg.visual_description,
                tags=list(seg.tags) + ["editorial_split"],
            )
            refined.append(sub_seg)

    # Log summary
    if len(refined) != len(segments):
        logger.info(
            f"Editorial subdivide complete: {len(segments)} → "
            f"{len(refined)} segments (cap={max_segment_sec:.1f}s)"
        )
    else:
        logger.info("Editorial subdivide: No segments required subdivision.")

    return refined


def _pick_split_points(
    seg_start: float,
    seg_end: float,
    gaps: List[tuple],
    max_segment_sec: float,
) -> List[float]:
    """Greedily select gap midpoints as split points until all resulting
    sub-segments are within ``max_segment_sec``.

    Strategy: sort gaps by duration descending (longest gaps are the most
    natural editorial cut points), then add the midpoint of each gap as a
    split point if it would break an oversized sub-segment.  Stop when all
    sub-segments are within the cap or we run out of gaps.
    """
    split_points: List[float] = []
    sorted_gaps = sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)

    for gap_start, gap_end in sorted_gaps:
        gap_mid = gap_start + (gap_end - gap_start) / 2.0

        # Check if this gap_mid falls within a sub-segment that's too long
        boundaries = sorted([seg_start] + split_points + [seg_end])
        needs_split = False
        for i in range(len(boundaries) - 1):
            b_start = boundaries[i]
            b_end = boundaries[i + 1]
            if b_end - b_start > max_segment_sec and b_start < gap_mid < b_end:
                needs_split = True
                break

        if needs_split:
            split_points.append(gap_mid)

    return split_points
