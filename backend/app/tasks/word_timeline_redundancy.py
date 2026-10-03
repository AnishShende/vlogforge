import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import sys
import os

from app.models import EGTDocument, EGTSegment
from app.config import settings
from app.tasks.retake_detect import (
    get_embedding_model, 
    compute_cosine_similarity, 
    _build_clusters,
    reconstruct_utterances
)
from app.utils.jev import check_is_explicit_retake_jev

logger = logging.getLogger("VlogForge.WordTimeline")

@dataclass
class WordTimelineEntry:
    word: str
    start_sec: float
    end_sec: float
    source_file: str
    visual_description: str
    has_speech: bool
    is_bad_take: bool
    is_superseded_take: bool
    is_stutter_repeat: bool
    quality_score: float

def build_word_timeline(egt_doc: EGTDocument) -> List[WordTimelineEntry]:
    timeline = []
    # Segments are inherently ordered by time in EGTDocument
    segments = sorted(egt_doc.segments, key=lambda s: (s.source_file, s.start_sec))
    for seg in segments:
        for wt in seg.word_timings:
            timeline.append(WordTimelineEntry(
                word=wt.get("text", wt.get("word", "")).strip(),
                start_sec=wt.get("start", 0.0),
                end_sec=wt.get("end", 0.0),
                source_file=seg.source_file,
                visual_description=seg.visual_description,
                has_speech=seg.has_speech,
                is_bad_take=seg.is_bad_take,
                is_superseded_take=getattr(seg, "is_superseded_take", False),
                is_stutter_repeat=getattr(seg, "is_stutter_repeat", False),
                quality_score=seg.quality_score
            ))
    # Ensure chronological order
    timeline.sort(key=lambda x: (x.source_file, x.start_sec))
    return timeline

@dataclass
class WordSpan:
    words: List[WordTimelineEntry]
    
    @property
    def source_file(self) -> str:
        return self.words[0].source_file if self.words else ""
    
    @property
    def start_sec(self) -> float:
        return self.words[0].start_sec if self.words else 0.0
        
    @property
    def end_sec(self) -> float:
        return self.words[-1].end_sec if self.words else 0.0
        
    @property
    def full_transcript(self) -> str:
        return " ".join(w.word for w in self.words).strip()
        
    def is_complete(self) -> bool:
        """
        A span is complete if it ends on terminal punctuation.
        """
        if not self.words:
            return False
        last_word = self.words[-1].word
        return any(last_word.endswith(punct) for punct in [".", "!", "?"])

def generate_candidate_spans_from_utterances(egt_doc: EGTDocument, timeline: List[WordTimelineEntry]) -> List[WordSpan]:
    """
    Generate candidate spans over the word timeline that match today's utterance reconstruction.
    """
    utterances = reconstruct_utterances(egt_doc.segments)
    spans = []
    for u in utterances:
        span_words = []
        for seg in u.segments:
            for wt in seg.word_timings:
                w_start = wt.get("start", 0.0)
                # Find matching entry in timeline
                for t in timeline:
                    if t.source_file == seg.source_file and abs(t.start_sec - w_start) < 0.01:
                        span_words.append(t)
                        break
        if span_words:
            spans.append(WordSpan(words=span_words))
    return spans

def detect_redundancy_on_timeline(egt_doc: EGTDocument) -> Dict:
    timeline = build_word_timeline(egt_doc)
    candidate_spans = generate_candidate_spans_from_utterances(egt_doc, timeline)
    
    edges = []
    
    # FIX 2: Trim-to-last-clean-run repetition check
    MIN_CLEAN_WORDS = 8
    NGRAM_SIZE = 4
    # Maximum inter-word gap (seconds) to still consider two words temporally
    # contiguous. Normal speech inter-word gaps are 0.05–0.3s. Whisper's VAD
    # already splits at min_silence_duration_ms=500, so anything >0.5s that
    # survives as a single "word" timing is hesitation/stutter padding.
    # 0.6s gives comfortable headroom above normal pauses while catching
    # the pathological 5–7s gaps left when repeated content is excised.
    MAX_CLEAN_WORD_GAP_SEC = 0.6
    
    import re
    def normalize_word(w: str) -> str:
        return re.sub(r'[^\w\s]', '', w.lower())

    trimmed_spans = []
    for span in candidate_spans:
        if not span.words:
            continue
            
        norm_words = [normalize_word(w.word) for w in span.words]
        marked = [False] * len(span.words)
        
        # Mark every word that is part of a repeated n-gram 
        # (i.e. occurs verbatim at an earlier position in the same span)
        for i in range(len(norm_words) - NGRAM_SIZE + 1):
            ngram = norm_words[i:i+NGRAM_SIZE]
            for j in range(i + 1, len(norm_words) - NGRAM_SIZE + 1):
                if norm_words[j:j+NGRAM_SIZE] == ngram:
                    for k in range(j, j+NGRAM_SIZE):
                        marked[k] = True
                        
        # Build temporally-contiguous runs of UNMARKED words.
        # Two consecutive unmarked words are only in the same run if:
        #   1. They are adjacent in the word list (no marked word between them), AND
        #   2. The temporal gap between them is <= MAX_CLEAN_WORD_GAP_SEC
        all_runs = []
        current_run_indices = []
        for i, is_marked in enumerate(marked):
            if not is_marked:
                if current_run_indices:
                    # Check temporal contiguity with the previous word in this run
                    prev_idx = current_run_indices[-1]
                    gap = span.words[i].start_sec - span.words[prev_idx].end_sec
                    if gap > MAX_CLEAN_WORD_GAP_SEC:
                        # Temporal break — finalize current run, start new one
                        all_runs.append(current_run_indices)
                        current_run_indices = [i]
                    else:
                        current_run_indices.append(i)
                else:
                    current_run_indices = [i]
            else:
                if current_run_indices:
                    all_runs.append(current_run_indices)
                    current_run_indices = []
        if current_run_indices:
            all_runs.append(current_run_indices)
        
        # Pick the longest temporally-contiguous clean run
        best_run_indices = max(all_runs, key=len) if all_runs else []
            
        # If the clean run is empty or below a minimal word-count floor
        if len(best_run_indices) < MIN_CLEAN_WORDS:
            span.has_clean_take = False
            print(f"  [DEBUG RESCORE] Span '{span.full_transcript[:60]}...' -> has_clean_take=False, NO rescore (original word scores retained)")
            print(f"    original_word_scores={[w.quality_score for w in span.words]}")
        else:
            span.has_clean_take = True
            # Trim the span to that clean run's word range
            trimmed_words = [span.words[idx] for idx in best_run_indices]
            span.words = trimmed_words
            
            # Fix 3: Recompute quality score via rule-based heuristic.
            # The original per-word scores were copied homogeneously from the
            # parent EGTSegment (e.g. all 0.25 for a bad-take segment), so
            # averaging them after trimming yields the same stale value.
            # Build a lightweight EGTSegment from the trimmed words and re-score.
            from app.tasks.score import compute_quality_score
            trimmed_text = " ".join(w.word for w in trimmed_words)
            trimmed_duration = trimmed_words[-1].end_sec - trimmed_words[0].start_sec
            dummy_seg = EGTSegment(
                clip_id="trim_rescore",
                source_file=span.source_file,
                start_sec=trimmed_words[0].start_sec,
                end_sec=trimmed_words[-1].end_sec,
                transcript=trimmed_text,
                segment_type="SPEECH",
                has_speech=True,
                word_timings=[
                    {"word": w.word, "start": w.start_sec, "end": w.end_sec}
                    for w in trimmed_words
                ],
            )
            new_score, new_flags = compute_quality_score(dummy_seg)
            new_score = round(max(0.0, min(1.0, new_score)), 3)
            print(f"  [DEBUG RESCORE] Span '{trimmed_text[:60]}...'")
            print(f"    duration={trimmed_duration:.2f}s, word_count={len(trimmed_words)}, marked_count={sum(marked)}")
            print(f"    original_word_scores={[trimmed_words[0].quality_score]}")
            print(f"    compute_quality_score() returned: raw_score={new_score}, flags={new_flags}")
            for w in trimmed_words:
                w.quality_score = new_score
            
        trimmed_spans.append(span)
            
    valid_spans = trimmed_spans
    
    def get_opening_words(text: str, num_words: int = 5) -> str:
        return " ".join(text.split()[:num_words]).strip()
        
    opening_texts = [get_opening_words(u.full_transcript, 5).lower() for u in valid_spans]
    whole_texts = [u.full_transcript.lower() for u in valid_spans]
    
    is_explicit_marker = []
    for u in valid_spans:
        t_opening_jev = get_opening_words(u.full_transcript, 15)
        if t_opening_jev:
            is_explicit_marker.append(check_is_explicit_retake_jev(t_opening_jev))
        else:
            is_explicit_marker.append(False)
            
    try:
        model = get_embedding_model()
        valid_indices = [i for i, t in enumerate(opening_texts) if t]
        if valid_indices:
            opening_embs = model.encode([opening_texts[i] for i in valid_indices])
            whole_embs = model.encode([whole_texts[i] for i in valid_indices])
            emb_map = {idx: (opening_embs[j], whole_embs[j]) for j, idx in enumerate(valid_indices)}
        else:
            emb_map = {}
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        emb_map = {}
        
    for i in range(len(valid_spans)):
        u1 = valid_spans[i]
        t1 = whole_texts[i]
        if not t1: continue
        
        for j in range(i + 1, len(valid_spans)):
            u2 = valid_spans[j]
            t2 = whole_texts[j]
            if not t2: continue
            
            if u1.source_file != u2.source_file:
                continue
                
            gap = u2.start_sec - u1.end_sec
            if gap < 0 or gap > settings.retake_candidate_window_sec:
                continue
                
            is_match = False
            if is_explicit_marker[j]:
                is_match = True
                
            if i in emb_map and j in emb_map:
                o_emb1, w_emb1 = emb_map[i]
                o_emb2, w_emb2 = emb_map[j]
                
                opening_sim = compute_cosine_similarity(o_emb1, o_emb2)
                if opening_sim >= settings.retake_opening_similarity_threshold:
                    is_match = True
                    
                whole_sim = compute_cosine_similarity(w_emb1, w_emb2)
                if whole_sim >= settings.retake_whole_utterance_similarity_threshold:
                    is_match = True
                    
            if is_match:
                edges.append((i, j))
                
    clusters = _build_clusters(edges, len(valid_spans))
    report = []
    
    for cluster_indices in clusters:
        cluster_info = {
            "cluster_start": valid_spans[cluster_indices[0]].start_sec,
            "cluster_end": valid_spans[cluster_indices[-1]].end_sec,
            "candidates": []
        }
        
        # Determine completeness of each candidate
        complete_candidates = []
        valid_candidates = []
        for idx in cluster_indices:
            span = valid_spans[idx]
            is_comp = span.is_complete()
            qs = sum(w.quality_score for w in span.words) / len(span.words) if span.words else 0
            cluster_info["candidates"].append({
                "index": idx,
                "text": span.full_transcript,
                "start": span.start_sec,
                "end": span.end_sec,
                "is_complete": is_comp,
                "quality_score": qs,
                "words": [{"word": w.word, "start_sec": w.start_sec, "end_sec": w.end_sec} for w in span.words]
            })
            if getattr(span, "has_clean_take", True):
                valid_candidates.append(idx)
                if is_comp:
                    complete_candidates.append((idx, qs))
                
        winner_idx = -1
        winner_reason = ""
        extended_text = None
        
        if not valid_candidates:
            winner_idx = -1
            winner_reason = "no-clean-take-found"
        elif complete_candidates:
            complete_candidates.sort(
                key=lambda x: (x[1], len(valid_spans[x[0]].words)),
                reverse=True
            )
            winner_idx = complete_candidates[0][0]
            
            # Check if there is a tie in quality score
            if len(complete_candidates) > 1 and abs(complete_candidates[0][1] - complete_candidates[1][1]) < 0.001:
                winner_reason = "complete (longest among tied candidates)"
            else:
                winner_reason = "complete"
        else:
            best_incomplete = max(valid_candidates, key=lambda idx: sum(w.quality_score for w in valid_spans[idx].words)/len(valid_spans[idx].words))
            best_span = valid_spans[best_incomplete]
            
            # Find the start time of the next candidate in the cluster (if any)
            # to ensure we don't extend INTO another candidate (which means the speaker abandoned this take)
            next_candidate_start = float('inf')
            for idx in cluster_indices:
                if idx > best_incomplete:
                    next_candidate_start = min(next_candidate_start, valid_spans[idx].start_sec)
            
            # Global extension check
            last_word = best_span.words[-1]
            try:
                global_idx = timeline.index(last_word)
                extended_words = list(best_span.words)
                extended = False
                for forward_idx in range(global_idx + 1, len(timeline)):
                    next_word = timeline[forward_idx]
                    
                    if next_word.source_file != last_word.source_file:
                        break
                    
                    # Stop if we hit a speech gap
                    gap = next_word.start_sec - extended_words[-1].end_sec
                    if gap >= 1.5:  # utterance gap
                        break
                        
                    # Stop if we hit the next candidate's start time (take was abandoned)
                    if next_word.start_sec >= next_candidate_start:
                        break
                        
                    extended_words.append(next_word)
                    if any(next_word.word.endswith(punct) for punct in [".", "!", "?"]):
                        extended = True
                        break
                
                if extended:
                    winner_idx = best_incomplete
                    winner_reason = "extended-to-complete"
                    extended_text = " ".join(w.word for w in extended_words).strip()
                    # It returns a single span, so we update the candidate's end_sec
                    for cand in cluster_info["candidates"]:
                        if cand["index"] == winner_idx:
                            cand["end"] = extended_words[-1].end_sec
                else:
                    winner_idx = -1
                    winner_reason = "no-clean-take-found"
            except ValueError:
                winner_idx = -1
                winner_reason = "no-clean-take-found"
                
        cluster_info["winner_index"] = winner_idx
        cluster_info["winner_reason"] = winner_reason
        if extended_text:
            cluster_info["extended_text"] = extended_text
        report.append(cluster_info)
        
    dropped_content = []
    for span in candidate_spans:
        if not getattr(span, "has_clean_take", True):
            dropped_content.append({
                "start": span.words[0].start_sec if hasattr(span, "words") and span.words else span.start_sec,
                "end": span.words[-1].end_sec if hasattr(span, "words") and span.words else span.end_sec,
                "text": span.full_transcript,
                "reason": "below-minimal-word-count-floor-after-trim"
            })
        
    return {"clusters": report, "dropped_content": dropped_content}
