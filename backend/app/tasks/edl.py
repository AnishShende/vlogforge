"""Pass 2 — EDL Generation (Phase 0: Mechanical Chronological Filter).

Phase 0 constraint: ZERO reasoning. This module performs a purely mechanical
transformation from EGT → EDL:

    1. Filter out bad takes (is_bad_take == True or is_superseded_take == True)
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


def _deduplicate_and_clamp_edl_overlaps(entries: List[EDLEntry]) -> List[EDLEntry]:
    """Clean up and eliminate any time overlaps or redundant repetitions between consecutive clips.

    Ensures that for any consecutive entries referencing the same source video file:
    1. If entry[i+1] is completely contained inside entry[i], drop it.
    2. If entry[i+1] overlaps with entry[i] (start_sec < prev_end_sec):
       - If they share the same editorial category (e.g. both INTRO, both OUTRO, or both KEEP/HIGHLIGHT/B_ROLL),
         merge them into a single seamless continuous clip.
       - Otherwise, clamp entry[i+1].start_sec = entry[i].end_sec.
       - If clamping reduces entry[i+1] to a degenerate sliver (< 0.2s), drop it.
    3. If entry[i+1] and entry[i] share the same source file and editorial category and have a micro-gap (<= 0.15s),
       bridge/merge them to eliminate micro-cuts and audio pops.
    4. Re-assign clean sequential sequence_index.
    """
    if not entries:
        return []

    merged: List[EDLEntry] = []

    for entry in entries:
        if not merged:
            merged.append(entry)
            continue

        prev = merged[-1]

        # Check if they are from the same source video file
        if entry.source_file == prev.source_file:
            # Check for complete containment (duplicate / subset)
            if entry.start_sec >= prev.start_sec and entry.end_sec <= prev.end_sec:
                logger.debug(
                    f"Dropping redundant clip {entry.clip_id} [{entry.start_sec:.2f}-{entry.end_sec:.2f}s] "
                    f"fully contained in prev [{prev.start_sec:.2f}-{prev.end_sec:.2f}s]"
                )
                continue

            is_same_category = (
                (prev.editorial_type == entry.editorial_type)
                or (
                    prev.editorial_type in ("KEEP", "HIGHLIGHT", "B_ROLL")
                    and entry.editorial_type in ("KEEP", "HIGHLIGHT", "B_ROLL")
                )
            )

            # Check for overlap: entry starts before previous ends
            if entry.start_sec < prev.end_sec:
                overlap_dur = prev.end_sec - entry.start_sec
                logger.info(
                    f"Detected {overlap_dur:.2f}s overlap between adjacent clips {prev.clip_id} "
                    f"[{prev.start_sec:.2f}-{prev.end_sec:.2f}s] and {entry.clip_id} [{entry.start_sec:.2f}-{entry.end_sec:.2f}s]."
                )
                if is_same_category:
                    # Merge into one continuous clip
                    prev.end_sec = max(prev.end_sec, entry.end_sec)
                    prev.core_end_sec = max(prev.core_end_sec, entry.core_end_sec)
                    if entry.narrative_priority == "CRITICAL":
                        prev.narrative_priority = "CRITICAL"
                    prev.quality_score = max(prev.quality_score, entry.quality_score)
                    logger.info(f"Merged into continuous clip: [{prev.start_sec:.2f}-{prev.end_sec:.2f}s]")
                    continue
                else:
                    # Clamp start of current entry to end of previous entry
                    entry.start_sec = prev.end_sec
                    if entry.core_start_sec < entry.start_sec:
                        entry.core_start_sec = entry.start_sec
                    # If remaining duration is less than 0.2s, skip it
                    if entry.end_sec - entry.start_sec < 0.2:
                        logger.debug(f"Dropping clamped degenerate clip {entry.clip_id} (<0.2s duration)")
                        continue
            elif is_same_category and (0 < entry.start_sec - prev.end_sec <= 0.15):
                # Bridge micro-gap between continuous parts
                logger.debug(
                    f"Bridging micro-gap ({entry.start_sec - prev.end_sec:.2f}s) between "
                    f"adjacent clips {prev.clip_id} and {entry.clip_id}"
                )
                prev.end_sec = max(prev.end_sec, entry.end_sec)
                prev.core_end_sec = max(prev.core_end_sec, entry.core_end_sec)
                if entry.narrative_priority == "CRITICAL":
                    prev.narrative_priority = "CRITICAL"
                prev.quality_score = max(prev.quality_score, entry.quality_score)
                continue

        merged.append(entry)

    # Re-index sequence_index
    for idx, entry in enumerate(merged):
        entry.sequence_index = idx

    return merged


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

    # Snap start: if the cut falls inside a speech segment, snap to its nearest edge
    inside_start_seg = next((s for s in file_segs if s["start"] < start < s["end"]), None)
    if inside_start_seg:
        if abs(inside_start_seg["start"] - start) <= abs(inside_start_seg["end"] - start) and inside_start_seg["start"] < end - 0.5:
            snapped_start = inside_start_seg["start"]
        elif inside_start_seg["end"] < end - 0.5:
            snapped_start = inside_start_seg["end"]
    else:
        best_start_diff = 3.0
        for seg in file_segs:
            diff = abs(seg["start"] - start)
            if diff < best_start_diff and seg["start"] < end - 0.5:
                snapped_start = seg["start"]
                best_start_diff = diff

    # Snap end: if the cut falls inside a speech segment, snap to its nearest edge
    inside_end_seg = next((s for s in file_segs if s["start"] < end < s["end"]), None)
    if inside_end_seg:
        if abs(inside_end_seg["end"] - end) <= abs(inside_end_seg["start"] - end) and inside_end_seg["end"] > snapped_start + 0.5:
            snapped_end = inside_end_seg["end"]
        elif inside_end_seg["start"] > snapped_start + 0.5:
            snapped_end = inside_end_seg["start"]
    else:
        best_end_diff = 3.0
        for seg in file_segs:
            diff = abs(seg["end"] - end)
            if diff < best_end_diff and seg["end"] > snapped_start + 0.5:
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
    this function upgrades it to CRITICAL. This ensures consistency between
    the LLM's stated editorial intent and the priority tags it assigns.
    """
    if not chain_of_thought:
        return entries

    # Split into sentences roughly
    import re
    sentences = re.split(r'(?<=[.!?])\s+', chain_of_thought.lower())

    for entry in entries:
        # Does the CoT mention this clip_id with preservation language in a 3-sentence window?
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

    return entries




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
        Calls generate_edl_llm() on each chunk with target_duration passed as
        soft editorial guidance (not a hard budget).
        Each chunk returns a local EDL list.

    Reduce Phase:
        Merges all per-chunk EDLs into a candidate set.
        Builds lightweight summaries (clip_id, duration, priority, transcript snippet)
        and calls generate_edl_reduce_llm() to produce a globally coherent
        sequence_index ordering + priority overrides.
        Falls back to positional ordering if Reduce fails.

    Speech-boundary snapping is applied once on the final merged EDL.
    """
    # Anti-hallucination guardrail: exclude bad and superseded takes entirely
    # from the LLM prompt. This guarantees they cannot be selected.
    segments = [
        s for s in egt_doc.segments 
        if not s.is_bad_take and not getattr(s, "is_superseded_take", False)
    ]
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
    from app.utils.llm import current_job_id
    parent_job_id = current_job_id.get()

    def process_map_chunk(chunk_idx, chunk):
        if parent_job_id:
            current_job_id.set(parent_job_id)
            
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
                if not seg.is_bad_take and not getattr(seg, "is_superseded_take", False) and seg.segment_type != "SILENCE":
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

    # Snap boundaries & Finalize
    for entry in entries:
        if transcript_segments:
            snapped_start, snapped_end = snap_boundary_to_speech(
                entry.start_sec, entry.end_sec, entry.source_file, transcript_segments
            )
            entry.start_sec = snapped_start
            entry.end_sec = snapped_end

    # Clean up and eliminate any overlapping cuts from adjacent clips
    entries = _deduplicate_and_clamp_edl_overlaps(entries)
    final_edl_dicts = [entry.model_dump() for entry in entries]

    return final_edl_dicts, None


def generate_edl(
    egt_doc: EGTDocument,
    transcript_segments: Optional[List[Dict]] = None,
    target_duration: Optional[float] = None,
    user_prompt: str = ""
) -> Tuple[List[Dict], Optional[str], str]:
    """Generate the EDL from a validated EGTDocument.

    When a Gemini reasoning model is available, uses LLM-based editorial
    reasoning (single-shot or Map-Reduce for large footage). Falls back
    to a mechanical chronological filter if the LLM is unavailable.

    target_duration is passed as soft editorial guidance to the LLM prompt,
    not enforced as a hard budget constraint.

    Args:
        egt_doc: Validated EGTDocument from perception pass.
        transcript_segments: Raw transcript dicts for boundary snapping.
        target_duration: Soft duration target for editorial guidance.
        user_prompt: User's creative brief / context text.

    Returns:
        Tuple of (EDL dicts, warning string or None, reasoning_mode string).
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
        return generate_edl_chunked(egt_doc, transcript_segments, target_duration, user_prompt) + ("llm_reasoned",)

    logger.info(
        f"Generating Phase 1 EDL using single-shot LLM Reasoning: {len(segments)} EGT segments."
    )

    egt_json = egt_doc.model_dump()
    
    # Anti-hallucination guardrail: exclude bad and superseded takes entirely
    egt_json["segments"] = [
        s for s in egt_json["segments"]
        if not s.get("is_bad_take") and not s.get("is_superseded_take") and not s.get("is_stutter_repeat")
    ]
    
    llm_response = generate_edl_llm(egt_json, target_duration, user_prompt)
    
    # generate_edl_llm returns a dict with 'edl' and 'chain_of_thought' keys
    llm_edl_dicts = None
    cot_text = ""
    reasoning_mode = "fallback_mechanical"
    
    if isinstance(llm_response, dict):
        llm_edl_dicts = llm_response.get("edl", [])
        cot_text = llm_response.get("editorial_rationale", "")
    elif isinstance(llm_response, list):
        # Backward compatibility: older return format was just a list
        llm_edl_dicts = llm_response
    
    if llm_edl_dicts:
        reasoning_mode = "llm_reasoned"
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
            if seg.is_bad_take or getattr(seg, "is_superseded_take", False) or getattr(seg, "is_stutter_repeat", False):
                removed_bad += 1
                logger.debug(
                    f"Filtered (bad/superseded/stutter): {seg.clip_id} "
                    f"[{seg.source_file} {seg.start_sec:.1f}-{seg.end_sec:.1f}s] "
                    f"score={seg.quality_score:.3f}"
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
            return [], None, reasoning_mode
    
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

    # 4. Snap boundaries & Finalize
    for entry in entries:
        start_sec = entry.start_sec
        end_sec = entry.end_sec
        
        if transcript_segments:
            snapped_start, snapped_end = snap_boundary_to_speech(
                start_sec, end_sec, entry.source_file, transcript_segments
            )
            entry.start_sec = snapped_start
            entry.end_sec = snapped_end
        
    # Clean up and eliminate any overlapping cuts from adjacent clips
    entries = _deduplicate_and_clamp_edl_overlaps(entries)
    final_edl_dicts = [entry.model_dump() for entry in entries]
        
    return final_edl_dicts, None, reasoning_mode
