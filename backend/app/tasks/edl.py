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
from typing import List, Dict, Optional, Tuple

from app.models import EGTSegment, EGTDocument, EDLEntry, generate_clip_id
from app.utils.llm import generate_edl_llm

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
    target_duration: Optional[float] = None,
    user_prompt: str = ""
) -> Tuple[List[Dict], Optional[str]]:
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
    
    # We pass a simplified EGT dictionary to the LLM to save tokens
    egt_json = egt_doc.model_dump()
    llm_edl_dicts = generate_edl_llm(egt_json, target_duration, user_prompt)
    
    if llm_edl_dicts:
        logger.info(f"Phase 1 LLM returned {len(llm_edl_dicts)} EDL entries.")
        
        # 1. Parse into EDLEntry objects and enrich with quality_score
        entries = []
        for idx, entry in enumerate(llm_edl_dicts):
            orig_seg = next((s for s in segments if s.clip_id == entry.get("clip_id")), None)
            quality_score = orig_seg.quality_score if orig_seg else 0.0
            
            start_sec = entry.get("start_sec", 0.0)
            end_sec = entry.get("end_sec", 0.0)
            core_start_sec = entry.get("core_start_sec")
            core_end_sec = entry.get("core_end_sec")
            
            if core_start_sec is None or core_start_sec < start_sec:
                core_start_sec = start_sec
            if core_end_sec is None or core_end_sec > end_sec:
                core_end_sec = end_sec
                
            e = EDLEntry(
                clip_id=entry.get("clip_id"),
                source_file=entry.get("source_file"),
                start_sec=start_sec,
                end_sec=end_sec,
                core_start_sec=core_start_sec,
                core_end_sec=core_end_sec,
                narrative_priority=entry.get("narrative_priority", "MEDIUM"),
                quality_score=quality_score,
                editorial_type=entry.get("editorial_type", "KEEP"),
                sequence_index=idx
            )
            entries.append(e)

        # 2. Pre-pass: Adjacency Risk Safeguard
        # Heuristic: Upgrade LOW clips to MEDIUM if sandwiched between CRITICAL clips from different files.
        # Note for v1: This is a fast approximation to prevent jarring jump cuts.
        # It will over-trigger when different files are actually a continuous take (e.g., camera auto-split),
        # and it will under-trigger for same-file time gaps (e.g., jump cuts within a single long recording).
        for i in range(1, len(entries) - 1):
            if entries[i].narrative_priority == "LOW":
                prev_e = entries[i-1]
                next_e = entries[i+1]
                if prev_e.narrative_priority == "CRITICAL" and next_e.narrative_priority == "CRITICAL":
                    if prev_e.source_file != next_e.source_file:
                        logger.info(f"Adjacency safeguard: Upgrading clip {entries[i].clip_id} from LOW to MEDIUM to prevent jump cut.")
                        entries[i].narrative_priority = "MEDIUM"

        # 3. Budget Enforcement (Tier 3 Repair)
        warning_msg = None
        if target_duration:
            def get_total_dur(edl_list):
                return sum(e.end_sec - e.start_sec for e in edl_list)

            # Phase A: Trim padding for LOW and MEDIUM
            if get_total_dur(entries) > target_duration:
                for e in entries:
                    if e.narrative_priority in ["LOW", "MEDIUM"]:
                        e.start_sec = e.core_start_sec
                        e.end_sec = e.core_end_sec

            # Phase B: Drop LOW
            if get_total_dur(entries) > target_duration:
                low_clips = sorted([e for e in entries if e.narrative_priority == "LOW"], key=lambda x: x.quality_score)
                for e in low_clips:
                    if get_total_dur(entries) <= target_duration:
                        break
                    entries.remove(e)

            # Phase C: Drop MEDIUM
            if get_total_dur(entries) > target_duration:
                med_clips = sorted([e for e in entries if e.narrative_priority == "MEDIUM"], key=lambda x: x.quality_score)
                for e in med_clips:
                    if get_total_dur(entries) <= target_duration:
                        break
                    entries.remove(e)
                    
            # Phase C.5: Trim CRITICAL padding
            if get_total_dur(entries) > target_duration:
                for e in entries:
                    if e.narrative_priority == "CRITICAL":
                        e.start_sec = e.core_start_sec
                        e.end_sec = e.core_end_sec

            # Phase D: CRITICAL Exhaustion
            final_dur = get_total_dur(entries)
            if final_dur > target_duration + 0.1:
                warning_msg = f"Budget Exceeded: Target duration is {target_duration}s, but mandatory CRITICAL clips alone total {final_dur:.1f}s. Halted repair to preserve narrative integrity."
                logger.warning(warning_msg)

        # 4. Snap boundaries & Finalize
        final_edl_dicts = []
        for idx, entry in enumerate(entries):
            start_sec = entry.start_sec
            end_sec = entry.end_sec
            
            if transcript_segments:
                orig_seg = next((s for s in segments if s.clip_id == entry.clip_id), None)
                snapped_start, snapped_end = snap_boundary_to_speech(
                    start_sec, end_sec, entry.source_file, transcript_segments
                )
                
                if orig_seg:
                    if abs(start_sec - orig_seg.start_sec) > 0.1:
                        snapped_start = start_sec
                    if abs(end_sec - orig_seg.end_sec) > 0.1:
                        snapped_end = end_sec
                        
                entry.start_sec = snapped_start
                entry.end_sec = snapped_end
            
            entry.sequence_index = idx
            final_edl_dicts.append(entry.model_dump())
            
        return final_edl_dicts, warning_msg

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
        return [], None

    # Sort by (source_file, start_sec) to preserve multi-file chronology
    kept.sort(key=lambda s: (s.source_file, s.start_sec))

    # INTRO first, OUTRO last
    intros = [s for s in kept if s.segment_type == "INTRO"]
    outros = [s for s in kept if s.segment_type == "OUTRO"]
    middle = [s for s in kept if s.segment_type not in ("INTRO", "OUTRO")]

    ordered = intros + middle + outros

    final_edl_dicts = []
    for idx, seg in enumerate(ordered):
        start_sec = seg.start_sec
        end_sec = seg.end_sec

        if transcript_segments:
            start_sec, end_sec = snap_boundary_to_speech(
                start_sec, end_sec, seg.source_file, transcript_segments
            )

        final_entry = EDLEntry(
            clip_id=seg.clip_id,
            source_file=seg.source_file,
            start_sec=seg.start_sec,
            end_sec=seg.end_sec,
            core_start_sec=seg.start_sec,
            core_end_sec=seg.end_sec,
            narrative_priority="MEDIUM",
            quality_score=seg.quality_score,
            editorial_type=seg.segment_type if seg.segment_type in ["INTRO", "OUTRO"] else "KEEP",
            sequence_index=idx
        )
        final_edl_dicts.append(final_entry.model_dump())

    return final_edl_dicts, None
