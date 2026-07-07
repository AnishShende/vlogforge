"""Pass 2 — EDL Generation (Phase 0: Mechanical Chronological Filter).

Phase 0 constraint: ZERO reasoning. This module performs a purely mechanical
transformation from EGT → EDL:

    1. Filter out bad takes (is_bad_take == True)
    2. Filter out SILENCE segments
    3. Keep everything else in STRICT chronological order
    4. Position INTRO first, OUTRO last (if detected)
    5. Snap cut boundaries to speech segment edges
    6. Emit EDLEntry list with sequential sequence_index

No reordering. No narrative analysis. No genre weighting. No duration targeting.
The full reasoning pipeline (edl_v1.py) is preserved in tasks/reasoning/ for Phase 1.
"""

import logging
from typing import List, Dict, Optional

from app.models import EGTSegment, EGTDocument, EDLEntry, generate_clip_id

logger = logging.getLogger("VlogForge.EDL")


def snap_boundary_to_speech(
    start: float,
    end: float,
    source_file: str,
    transcript_segments: List[Dict],
) -> tuple:
    """Snaps the start and end of a clip to aligned speech segment boundaries if they are close,
    preventing clipping mid-sentence or mid-word.
    """
    if not transcript_segments:
        return start, end

    snapped_start = start
    snapped_end = end

    file_segs = [t for t in transcript_segments if t.get("video_file") == source_file]
    if not file_segs:
        return start, end

    # Snap start: find the speech segment that starts close to the scene start
    best_start_diff = 1.5
    for seg in file_segs:
        diff = abs(seg["start"] - start)
        if diff < best_start_diff:
            if seg["start"] < end - 0.5:
                snapped_start = seg["start"]
                best_start_diff = diff

    # Snap end: find the speech segment that ends close to the scene end
    best_end_diff = 1.5
    for seg in file_segs:
        diff = abs(seg["end"] - end)
        if diff < best_end_diff:
            if seg["end"] > snapped_start + 0.5:
                snapped_end = seg["end"]
                best_end_diff = diff

    return snapped_start, snapped_end


def generate_edl(
    egt_doc: EGTDocument,
    transcript_segments: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Generate the Phase 0 EDL from a validated EGTDocument.

    Phase 0 algorithm:
    1. Filter out segments where is_bad_take == True
    2. Filter out segments where segment_type == "SILENCE"
    3. Keep all remaining segments in strict chronological order
    4. Ensure INTRO is first (if one exists) and OUTRO is last
    5. Apply speech-boundary snapping
    6. Emit EDLEntry dicts with sequential sequence_index

    Args:
        egt_doc: Validated EGTDocument from perception pass.
        transcript_segments: Raw transcript dicts for boundary snapping.

    Returns:
        List of EDLEntry dicts (serializable for storage and API).
    """
    segments = egt_doc.segments
    logger.info(
        f"Generating Phase 1 EDL using LLM Reasoning: {len(segments)} EGT segments."
    )

    # 1. Attempt Phase 1 LLM Reasoning
    from app.utils.llm import generate_edl_llm
    
    # We pass a simplified EGT dictionary to the LLM to save tokens
    egt_json = egt_doc.model_dump()
    llm_edl_dicts = generate_edl_llm(egt_json)
    
    if llm_edl_dicts:
        logger.info(f"Phase 1 LLM returned {len(llm_edl_dicts)} EDL entries.")
        # Re-map start and end times to snap to speech boundaries
        final_edl_dicts = []
        for idx, entry in enumerate(llm_edl_dicts):
            start_sec = entry.get("start_sec", 0.0)
            end_sec = entry.get("end_sec", 0.0)
            
            if transcript_segments:
                start_sec, end_sec = snap_boundary_to_speech(
                    start_sec, end_sec, entry.get("source_file", ""), transcript_segments
                )
            
            # Reconstruct dict strictly to the EDLEntry schema
            final_entry = EDLEntry(
                clip_id=entry.get("clip_id"),
                source_file=entry.get("source_file"),
                start_sec=start_sec,
                end_sec=end_sec,
                editorial_type=entry.get("editorial_type", "KEEP"),
                sequence_index=idx
            )
            final_edl_dicts.append(final_entry.model_dump())
            
        return final_edl_dicts

    # 2. Fallback to Phase 0 Mechanical Filter
    logger.warning("Phase 1 Reasoning failed or returned empty. Falling back to Phase 0 mechanical filter.")
    
    kept = []
    removed_bad = 0
    removed_silence = 0

    for seg in segments:
        if seg.is_bad_take:
            removed_bad += 1
            logger.debug(
                f"Filtered (bad_take): {seg.clip_id} "
                f"[{seg.source_file} {seg.start_sec:.1f}-{seg.end_sec:.1f}s] "
                f"score={seg.quality_score:.3f} flags={seg.quality_flags}"
            )
            continue
        if seg.segment_type == "SILENCE":
            removed_silence += 1
            logger.debug(
                f"Filtered (silence): {seg.clip_id} "
                f"[{seg.source_file} {seg.start_sec:.1f}-{seg.end_sec:.1f}s]"
            )
            continue
        kept.append(seg)

    logger.info(
        f"EDL fallback filter results: {len(kept)} kept, "
        f"{removed_bad} bad takes removed, "
        f"{removed_silence} silence segments removed"
    )

    if not kept:
        logger.warning("EDL is empty after filtering — all segments were bad takes or silence.")
        return []

    # Sort by (source_file, start_sec) to preserve multi-file chronology
    kept.sort(key=lambda s: (s.source_file, s.start_sec))

    # INTRO first, OUTRO last
    intros = [s for s in kept if s.segment_type == "INTRO"]
    outros = [s for s in kept if s.segment_type == "OUTRO"]
    middle = [s for s in kept if s.segment_type not in ("INTRO", "OUTRO")]

    ordered = intros + middle + outros

    edl_entries = []
    for idx, seg in enumerate(ordered):
        start_sec = seg.start_sec
        end_sec = seg.end_sec

        if transcript_segments:
            start_sec, end_sec = snap_boundary_to_speech(
                start_sec, end_sec, seg.source_file, transcript_segments
            )

        if seg.segment_type == "INTRO":
            editorial_type = "INTRO"
        elif seg.segment_type == "OUTRO":
            editorial_type = "OUTRO"
        else:
            editorial_type = "KEEP"

        entry = EDLEntry(
            clip_id=seg.clip_id,
            source_file=seg.source_file,
            start_sec=start_sec,
            end_sec=end_sec,
            editorial_type=editorial_type,
            sequence_index=idx,
        )
        edl_entries.append(entry)

    total_duration = sum(e.end_sec - e.start_sec for e in edl_entries)
    logger.info(
        f"Phase 0 Fallback EDL generation complete: {len(edl_entries)} entries, "
        f"total duration: {total_duration:.1f}s"
    )

    return [entry.model_dump() for entry in edl_entries]
