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
from app.utils.llm import generate_edl_llm, generate_edl_reduce_llm
from app.config import settings

logger = logging.getLogger("VlogForge.EDL")


def _enforce_broll_minimums(entries: List[EDLEntry], segments: List[EGTSegment]):
    """Ensure B_ROLL clips are at least 3 seconds long, clamped to their original raw footage bounds."""
    for entry in entries:
        if entry.editorial_type == "B_ROLL":
            orig_seg = next((s for s in segments if s.clip_id == entry.clip_id), None)
            if not orig_seg:
                continue
            
            dur = entry.end_sec - entry.start_sec
            if dur < 3.0:
                deficit = 3.0 - dur
                # Try to expand equally on both sides, clamped by orig_seg bounds
                new_start = max(orig_seg.start_sec, entry.start_sec - (deficit / 2))
                new_end = min(orig_seg.end_sec, entry.end_sec + (deficit / 2))
                
                # If we still haven't reached 3.0s, try pushing one side further
                new_dur = new_end - new_start
                if new_dur < 3.0:
                    remaining_deficit = 3.0 - new_dur
                    if new_start > orig_seg.start_sec:
                        new_start = max(orig_seg.start_sec, new_start - remaining_deficit)
                    elif new_end < orig_seg.end_sec:
                        new_end = min(orig_seg.end_sec, new_end + remaining_deficit)
                
                entry.start_sec = new_start
                entry.end_sec = new_end
                entry.core_start_sec = min(entry.core_start_sec, new_start)
                entry.core_end_sec = max(entry.core_end_sec, new_end)
    return entries


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


# ---------------------------------------------------------------------------
# Priority Validation — catch reasoning/priority mismatches
# ---------------------------------------------------------------------------

_PRESERVATION_KEYWORDS = [
    "preserve", "keep", "retain", "must include", "essential",
    "important", "don't cut", "do not cut", "should not be dropped",
    "want to keep", "wanted to keep", "not cut",
]


def _validate_priority_consistency(
    entries: List,
    chain_of_thought: str,
) -> List:
    """Detect and fix reasoning/priority mismatches from the LLM.

    If the LLM's chain-of-thought reasoning mentions wanting to preserve a
    clip (using preservation keywords) but assigned it MEDIUM or LOW priority,
    this function upgrades it to CRITICAL. This prevents the Tier 3 enforcer
    from dropping clips the LLM explicitly reasoned about wanting to keep.

    Also warns if a single clip's core duration exceeds 50% of the total EDL
    duration and is tagged MEDIUM — a risk signal for destructive drops.
    """
    if not chain_of_thought:
        return entries

    # Split into sentences roughly
    import re
    sentences = re.split(r'(?<=[.!?])\s+', chain_of_thought.lower())
    total_dur = sum(e.end_sec - e.start_sec for e in entries)

    for entry in entries:
        # Check 1: Does the CoT mention this clip_id with preservation language in a 3-sentence window?
        if entry.narrative_priority in ["MEDIUM", "LOW"]:
            clip_id_lower = entry.clip_id.lower()
            
            has_preservation = False
            for i, sentence in enumerate(sentences):
                if clip_id_lower in sentence:
                    # Check this sentence, the previous, and the next (3-sentence window)
                    window = []
                    if i > 0:
                        window.append(sentences[i-1])
                    window.append(sentence)
                    if i + 1 < len(sentences):
                        window.append(sentences[i+1])
                        
                    combined = " ".join(window)
                    if any(kw in combined for kw in _PRESERVATION_KEYWORDS):
                        has_preservation = True
                        break

            if has_preservation:
                old_priority = entry.narrative_priority
                entry.narrative_priority = "CRITICAL"
                logger.warning(
                    f"Priority validation: Upgraded clip {entry.clip_id} "
                    f"from {old_priority} to CRITICAL — CoT context mentions "
                    f"preservation intent but priority was under-tagged."
                )

        # Check 2: Large clip with weak priority is a risk signal
        if entry.narrative_priority == "MEDIUM" and total_dur > 0:
            core_dur = entry.core_end_sec - entry.core_start_sec
            if core_dur > total_dur * 0.5:
                logger.warning(
                    f"Priority risk: Clip {entry.clip_id} is {core_dur:.1f}s "
                    f"({core_dur/total_dur*100:.0f}% of total) but tagged "
                    f"MEDIUM. Tier 3 may drop it entirely."
                )

    return entries


# ---------------------------------------------------------------------------
# Proportional Core Trimming — surgical budget reduction
# ---------------------------------------------------------------------------

_MIN_CLIP_DURATION_SEC = 3.0  # Never trim a clip below this floor


def _proportional_core_trim(
    entries: List,
    max_allowed: float,
    priority_filter: List[str],
) -> List:
    """Trim eligible clips' core bounds proportionally to their share of the
    budget excess.

    Instead of dropping entire clips (which causes cliff-edge duration drops
    like 285s → 24s), this function distributes the excess evenly:

        excess = total_duration - max_allowed
        per_clip_trim = excess * (clip_duration / total_eligible_duration)

    Each clip loses duration proportional to its size, trimmed from the end
    of its core bounds. No clip is trimmed below ``_MIN_CLIP_DURATION_SEC``.

    Args:
        entries: The current EDL entries (mutated in place).
        max_allowed: Maximum allowed total duration (target + tolerance).
        priority_filter: Only trim clips matching these priority levels.

    Returns:
        The same entries list (mutated).
    """
    total_dur = sum(e.end_sec - e.start_sec for e in entries)
    if total_dur <= max_allowed:
        return entries

    excess = total_dur - max_allowed
    eligible = [e for e in entries if e.narrative_priority in priority_filter]
    eligible_dur = sum(e.end_sec - e.start_sec for e in eligible)

    if eligible_dur <= 0:
        return entries

    for e in eligible:
        clip_dur = e.end_sec - e.start_sec
        if clip_dur <= _MIN_CLIP_DURATION_SEC:
            continue

        # Proportional share of the excess to trim from this clip
        trim_amount = excess * (clip_dur / eligible_dur)

        # Don't trim below the minimum floor
        max_trimmable = clip_dur - _MIN_CLIP_DURATION_SEC
        trim_amount = min(trim_amount, max_trimmable)

        if trim_amount > 0:
            # Trim from the end of the core bounds
            e.core_end_sec = max(
                e.core_start_sec + _MIN_CLIP_DURATION_SEC,
                e.core_end_sec - trim_amount,
            )
            e.end_sec = e.core_end_sec
            logger.info(
                f"Proportional trim: Clip {e.clip_id} trimmed by "
                f"{trim_amount:.1f}s → new duration {e.end_sec - e.start_sec:.1f}s"
            )

    return entries


def _enforce_budget(entries: List[EDLEntry], target_duration: float) -> Tuple[List[EDLEntry], Optional[str]]:
    """Apply Tier 3 graduated budget enforcement to the EDL entries."""
    warning_msg = None
    budget_tolerance = 0.10  # ±10% tolerance band
    max_allowed = target_duration * (1.0 + budget_tolerance)

    def get_total_dur(edl_list):
        return sum(e.end_sec - e.start_sec for e in edl_list)

    def is_over_budget(edl_list):
        return get_total_dur(edl_list) > max_allowed

    # Phase A: Trim padding for LOW and MEDIUM (collapse to core bounds)
    if is_over_budget(entries):
        logger.info(f"Tier 3 Phase A: Trimming padding from LOW/MEDIUM clips. Current: {get_total_dur(entries):.1f}s, target: {max_allowed:.1f}s")
        for e in entries:
            if e.narrative_priority in ["LOW", "MEDIUM"]:
                e.start_sec = e.core_start_sec
                e.end_sec = e.core_end_sec

    # Phase B: Proportional core trimming (LOW and MEDIUM)
    if is_over_budget(entries):
        logger.info(f"Tier 3 Phase B: Proportional core trimming on LOW/MEDIUM clips. Current: {get_total_dur(entries):.1f}s")
        entries = _proportional_core_trim(
            entries, max_allowed,
            priority_filter=["LOW", "MEDIUM"]
        )

    # Phase C: Drop LOW clips (by ascending quality score)
    if is_over_budget(entries):
        logger.info(f"Tier 3 Phase C: Dropping LOW clips. Current: {get_total_dur(entries):.1f}s")
        low_clips = sorted([e for e in entries if e.narrative_priority == "LOW"], key=lambda x: x.quality_score)
        for e in low_clips:
            if not is_over_budget(entries):
                break
            entries.remove(e)

    # Phase D: Proportional core trimming (MEDIUM only)
    if is_over_budget(entries):
        logger.info(f"Tier 3 Phase D: Proportional core trimming on MEDIUM clips. Current: {get_total_dur(entries):.1f}s")
        entries = _proportional_core_trim(
            entries, max_allowed,
            priority_filter=["MEDIUM"]
        )

    # Phase E: Drop MEDIUM clips (last resort before touching CRITICAL)
    if is_over_budget(entries):
        logger.info(f"Tier 3 Phase E: Dropping MEDIUM clips. Current: {get_total_dur(entries):.1f}s")
        med_clips = sorted([e for e in entries if e.narrative_priority == "MEDIUM"], key=lambda x: x.quality_score)
        for e in med_clips:
            if not is_over_budget(entries):
                break
            entries.remove(e)

    # Phase F: Trim CRITICAL padding
    if is_over_budget(entries):
        logger.info(f"Tier 3 Phase F: Trimming CRITICAL padding. Current: {get_total_dur(entries):.1f}s")
        for e in entries:
            if e.narrative_priority == "CRITICAL":
                e.start_sec = e.core_start_sec
                e.end_sec = e.core_end_sec

    # Phase G: CRITICAL Exhaustion — halt and warn
    final_dur = get_total_dur(entries)
    if final_dur > max_allowed:
        warning_msg = (
            f"Budget Exceeded: Target duration is {target_duration}s "
            f"(tolerance band: {max_allowed:.1f}s), but mandatory CRITICAL "
            f"clips alone total {final_dur:.1f}s. Halted repair to preserve "
            f"narrative integrity."
        )
        logger.warning(warning_msg)
    else:
        logger.info(f"Tier 3 repair complete. Final duration: {final_dur:.1f}s (target: {target_duration}s, max allowed: {max_allowed:.1f}s)")
        
    return entries, warning_msg

def _partition_segments_into_chunks(
    segments: List[EGTSegment],
    chunk_size: int,
) -> List[List[EGTSegment]]:
    """Partition EGT segments into fixed-size chunks for Map-Reduce reasoning.

    Chunking is done in strict chronological order (the natural list order after
    perception). We do NOT attempt to honour source-file boundaries because the
    Reduce phase's global ordering will correct any cross-chunk ordering issues.
    """
    chunks = []
    for i in range(0, len(segments), chunk_size):
        chunks.append(segments[i : i + chunk_size])
    return chunks


def generate_edl_chunked(
    egt_doc: EGTDocument,
    transcript_segments: Optional[List[Dict]] = None,
    target_duration: Optional[float] = None,
    user_prompt: str = "",
) -> Tuple[List[Dict], Optional[str]]:
    """M4 Map-Reduce EDL generation for large EGT documents (>= edl_chunk_threshold segments).

    Map Phase:
        Partitions EGT segments into chunks of settings.edl_chunk_size segments.
        Calls generate_edl_llm() on each chunk with a time-proportional sub-budget
        (target_duration * chunk_raw_duration / total_raw_duration).
        Each chunk returns a local EDL list.

    Reduce Phase:
        Merges all per-chunk EDLs into a candidate set.
        Builds lightweight summaries (clip_id, duration, priority, transcript snippet)
        and calls generate_edl_reduce_llm() to produce a globally coherent
        sequence_index ordering + priority overrides.
        Falls back to positional ordering if Reduce fails.

    Budget enforcement and speech-boundary snapping are applied once on the final
    merged EDL, identical to the short-footage path.
    """
    segments = egt_doc.segments
    total_raw_duration = sum(s.duration_sec for s in segments) or 1.0
    chunk_size = settings.edl_chunk_size
    context_doc = egt_doc.context_summary

    chunks = _partition_segments_into_chunks(segments, chunk_size)
    logger.info(
        f"Map-Reduce EDL: {len(segments)} segments split into {len(chunks)} chunks "
        f"of up to {chunk_size} segments each."
    )

    # -----------------------------------------------------------------------
    # Map Phase
    # -----------------------------------------------------------------------
    import concurrent.futures

    def process_map_chunk(chunk_idx, chunk):
        chunk_raw_duration = sum(s.duration_sec for s in chunk)
        chunk_sub_budget = (
            (target_duration * chunk_raw_duration / total_raw_duration)
            if target_duration
            else None
        )

        mini_egt = EGTDocument(
            segments=chunk,
            total_duration_sec=chunk_raw_duration,
            source_file_count=len({s.source_file for s in chunk}),
            context_summary=context_doc,
        )
        mini_egt_json = mini_egt.model_dump()

        logger.info(
            f"Map chunk {chunk_idx + 1}/{len(chunks)}: "
            f"{len(chunk)} segments, sub-budget={chunk_sub_budget:.1f}s"
            if chunk_sub_budget else
            f"Map chunk {chunk_idx + 1}/{len(chunks)}: {len(chunk)} segments, no budget."
        )

        llm_response = generate_edl_llm(mini_egt_json, chunk_sub_budget, user_prompt)
        return chunk_idx, chunk, llm_response

    chunk_results = [None] * len(chunks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_map_chunk, i, c): i for i, c in enumerate(chunks)}
        for future in concurrent.futures.as_completed(futures):
            idx, chunk, llm_response = future.result()
            chunk_results[idx] = (chunk, llm_response)

    all_candidate_entries: List[EDLEntry] = []

    for chunk_idx, (chunk, llm_response) in enumerate(chunk_results):
        llm_edl_dicts: List[Dict] = []
        cot_text = ""
        if isinstance(llm_response, dict):
            llm_edl_dicts = llm_response.get("edl", [])
            cot_text = llm_response.get("chain_of_thought", "")
        elif isinstance(llm_response, list):
            llm_edl_dicts = llm_response

        if not llm_edl_dicts:
            # Map chunk failed — fall back to keeping all non-bad-take, non-silence
            # segments from this chunk in chronological order.
            logger.warning(
                f"Map chunk {chunk_idx + 1} LLM failed. Using mechanical fallback for this chunk."
            )
            for seg in chunk:
                if not seg.is_bad_take and seg.segment_type != "SILENCE":
                    priority = "CRITICAL" if seg.segment_type in ("INTRO", "OUTRO") else "MEDIUM"
                    all_candidate_entries.append(
                        EDLEntry(
                            clip_id=seg.clip_id,
                            source_file=seg.source_file,
                            start_sec=seg.start_sec,
                            end_sec=seg.end_sec,
                            core_start_sec=seg.start_sec,
                            core_end_sec=seg.end_sec,
                            narrative_priority=priority,
                            quality_score=seg.quality_score,
                            editorial_type=seg.segment_type if seg.segment_type in ("INTRO", "OUTRO") else "KEEP",
                            sequence_index=0,
                        )
                    )
            continue

        # Parse chunk LLM output into EDLEntry objects
        for entry in llm_edl_dicts:
            orig_seg = next((s for s in chunk if s.clip_id == entry.get("clip_id")), None)
            quality_score = orig_seg.quality_score if orig_seg else 0.0
            start_sec = entry.get("start_sec", 0.0)
            end_sec = entry.get("end_sec", 0.0)
            core_start = entry.get("core_start_sec") or start_sec
            core_end = entry.get("core_end_sec") or end_sec
            if core_start < start_sec:
                core_start = start_sec
            if core_end > end_sec:
                core_end = end_sec

            all_candidate_entries.append(
                EDLEntry(
                    clip_id=entry.get("clip_id"),
                    source_file=entry.get("source_file", ""),
                    start_sec=start_sec,
                    end_sec=end_sec,
                    core_start_sec=core_start,
                    core_end_sec=core_end,
                    narrative_priority=entry.get("narrative_priority", "MEDIUM"),
                    quality_score=quality_score,
                    editorial_type=entry.get("editorial_type", "KEEP"),
                    sequence_index=0,
                )
            )

        # Priority validation for this chunk
        all_candidate_entries_for_chunk = all_candidate_entries[-len(llm_edl_dicts):]
        validated = _validate_priority_consistency(all_candidate_entries_for_chunk, cot_text)
        all_candidate_entries[-len(llm_edl_dicts):] = validated

    logger.info(f"Map phase complete: {len(all_candidate_entries)} candidate EDL entries across all chunks.")

    # -----------------------------------------------------------------------
    # Reduce Phase
    # -----------------------------------------------------------------------
    # Build lightweight summaries (small payload for the Reduce LLM)
    seg_lookup = {s.clip_id: s for s in segments}
    chunk_summaries = [
        {
            "clip_id": e.clip_id,
            "source_file": e.source_file,
            "start_sec": e.start_sec,
            "end_sec": e.end_sec,
            "duration_sec": round(e.end_sec - e.start_sec, 2),
            "narrative_priority": e.narrative_priority,
            "editorial_type": e.editorial_type,
            "transcript_snippet": (
                seg_lookup[e.clip_id].transcript[:80]
                if e.clip_id in seg_lookup else ""
            ),
        }
        for e in all_candidate_entries
    ]

    reduce_result = generate_edl_reduce_llm(
        chunk_summaries,
        target_duration=target_duration or total_raw_duration,
        context_doc=context_doc,
    )

    if reduce_result:
        # Apply Reduce ordering and priority overrides to the candidate entries
        id_to_entry = {e.clip_id: e for e in all_candidate_entries}
        ordered_entries: List[EDLEntry] = []
        seen_ids = set()
        for item in reduce_result:
            cid = item.get("clip_id", "")
            if cid in id_to_entry and cid not in seen_ids:
                entry = id_to_entry[cid]
                entry.narrative_priority = item.get("narrative_priority", entry.narrative_priority)
                entry.sequence_index = item.get("sequence_index", 0)
                ordered_entries.append(entry)
                seen_ids.add(cid)
        # Any entries the Reduce phase dropped — append at end with LOW priority
        for e in all_candidate_entries:
            if e.clip_id not in seen_ids:
                logger.debug(f"Reduce dropped clip {e.clip_id}; appending as LOW at end.")
                e.narrative_priority = "LOW"
                e.sequence_index = len(ordered_entries)
                ordered_entries.append(e)

        ordered_entries.sort(key=lambda e: e.sequence_index)
        entries = ordered_entries
        logger.info(f"Reduce phase complete: {len(entries)} entries in final ordering.")
    else:
        # Reduce failed — fall back to positional ordering of Map outputs
        logger.warning("Reduce phase failed. Falling back to Map-phase positional order.")
        entries = all_candidate_entries

    # Adjacency risk safeguard (identical to short-footage path)
    for i in range(1, len(entries) - 1):
        if entries[i].narrative_priority == "LOW":
            prev_e = entries[i - 1]
            next_e = entries[i + 1]
            if (
                prev_e.narrative_priority == "CRITICAL"
                and next_e.narrative_priority == "CRITICAL"
                and prev_e.source_file != next_e.source_file
            ):
                logger.info(
                    f"Adjacency safeguard: Upgrading clip {entries[i].clip_id} from LOW to MEDIUM."
                )
                entries[i].narrative_priority = "MEDIUM"

    entries = _enforce_broll_minimums(entries, segments)

    # Budget Enforcement (Tier 3 Graduated Repair) — same as short-footage path
    warning_msg = None
    if target_duration:
        entries, warning_msg = _enforce_budget(entries, target_duration)

    # Snap boundaries & Finalize
    final_edl_dicts = []
    for idx, entry in enumerate(entries):
        if transcript_segments:
            orig_seg = seg_lookup.get(entry.clip_id)
            snapped_start, snapped_end = snap_boundary_to_speech(
                entry.start_sec, entry.end_sec, entry.source_file, transcript_segments
            )
            if orig_seg:
                if abs(entry.start_sec - orig_seg.start_sec) > 0.1:
                    snapped_start = entry.start_sec
                if abs(entry.end_sec - orig_seg.end_sec) > 0.1:
                    snapped_end = entry.end_sec
            entry.start_sec = snapped_start
            entry.end_sec = snapped_end

        entry.sequence_index = idx
        final_edl_dicts.append(entry.model_dump())

    return final_edl_dicts, warning_msg


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
        f"EDL generation: {len(segments)} EGT segments, target={target_duration}s."
    )

    # M4: For large jobs, delegate to the Map-Reduce path.
    # Short footage (below threshold) continues to use the single-shot LLM path.
    if len(segments) > settings.edl_chunk_threshold:
        logger.info(
            f"Segment count {len(segments)} exceeds edl_chunk_threshold "
            f"({settings.edl_chunk_threshold}). Using Map-Reduce EDL reasoning."
        )
        return generate_edl_chunked(egt_doc, transcript_segments, target_duration, user_prompt)

    logger.info(
        f"Generating Phase 1 EDL using single-shot LLM Reasoning: {len(segments)} EGT segments."
    )

    egt_json = egt_doc.model_dump()
    llm_response = generate_edl_llm(egt_json, target_duration, user_prompt)
    
    # generate_edl_llm returns a dict with 'edl' and 'chain_of_thought' keys
    llm_edl_dicts = None
    cot_text = ""
    if isinstance(llm_response, dict):
        llm_edl_dicts = llm_response.get("edl", [])
        cot_text = llm_response.get("chain_of_thought", "")
    elif isinstance(llm_response, list):
        # Backward compatibility: older return format was just a list
        llm_edl_dicts = llm_response
    
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

        # 2. Priority Validation: catch reasoning/priority mismatches
        entries = _validate_priority_consistency(entries, cot_text)

        # 3. Pre-pass: Adjacency Risk Safeguard
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

    else:
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
        
        entries = []
        for seg in ordered:
            priority = "CRITICAL" if seg.segment_type in ["INTRO", "OUTRO"] else "MEDIUM"
            e = EDLEntry(
                clip_id=seg.clip_id,
                source_file=seg.source_file,
                start_sec=seg.start_sec,
                end_sec=seg.end_sec,
                core_start_sec=seg.start_sec,
                core_end_sec=seg.end_sec,
                narrative_priority=priority,
                quality_score=seg.quality_score,
                editorial_type=seg.segment_type if seg.segment_type in ["INTRO", "OUTRO"] else "KEEP",
                sequence_index=0
            )
            entries.append(e)

    entries = _enforce_broll_minimums(entries, segments)

    # 4. Budget Enforcement (Tier 3 Graduated Repair)
    warning_msg = None
    if target_duration:
        entries, warning_msg = _enforce_budget(entries, target_duration)

    # 5. Snap boundaries & Finalize
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
