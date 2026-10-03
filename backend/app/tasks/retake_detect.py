import logging
import math
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

from app.models import EGTSegment, EGTDocument
from app.config import settings
from app.utils.jev import check_is_explicit_retake_jev
from app.tasks.disfluency import compute_utterance_disfluency

logger = logging.getLogger("VlogForge.RetakeDetect")


# Lazily loaded embedding model
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformers all-MiniLM-L6-v2...")
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            logger.error("sentence-transformers not installed. Embeddings disabled.")
            raise
    return _embedding_model

@dataclass
class Utterance:
    segments: List[EGTSegment]
    
    @property
    def source_file(self) -> str:
        return self.segments[0].source_file
        
    @property
    def start_sec(self) -> float:
        return self.segments[0].start_sec
        
    @property
    def end_sec(self) -> float:
        return self.segments[-1].end_sec
        
    @property
    def full_transcript(self) -> str:
        return " ".join([s.transcript for s in self.segments if s.transcript]).strip()
        
    @property
    def is_subdivided(self) -> bool:
        return len(self.segments) > 1

def reconstruct_utterances(segments: List[EGTSegment]) -> List[Utterance]:
    """
    Reconstruct full utterances from EGT segments.
    Subdivided scenes are glued back together if they share contiguous boundaries
    and have split tags. Single undivided scenes become single-segment utterances.
    """
    if not segments:
        return []
        
    utterances = []
    current_utt_segments = []
    
    # Sort by file and time
    sorted_segments = sorted(segments, key=lambda s: (s.source_file, s.start_sec))
    
    for seg in sorted_segments:
        if not current_utt_segments:
            current_utt_segments.append(seg)
            continue
            
        prev_seg = current_utt_segments[-1]
        
        same_file = prev_seg.source_file == seg.source_file
        contiguous = abs(prev_seg.end_sec - seg.start_sec) < 0.05
        is_split = (
            any(t in seg.tags for t in ["editorial_split", "speech_gap_split"]) or
            any(t in prev_seg.tags for t in ["editorial_split", "speech_gap_split"])
        )
        
        if same_file and contiguous and is_split:
            current_utt_segments.append(seg)
        else:
            utterances.append(Utterance(segments=current_utt_segments))
            current_utt_segments = [seg]
            
    if current_utt_segments:
        utterances.append(Utterance(segments=current_utt_segments))
        
    logger.info(f"Reconstructed {len(utterances)} utterances from {len(segments)} segments.")
    return utterances

def get_opening_words(text: str, num_words: int = 5) -> str:
    """Get the first N words of a transcript."""
    return " ".join(text.split()[:num_words]).strip()

def compute_cosine_similarity(vec1, vec2) -> float:
    import numpy as np
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return float(np.dot(v1, v2) / norm)

def _build_clusters(pairs: List[Tuple[int, int]], num_nodes: int) -> List[List[int]]:
    """Union-find for connected components."""
    parent = list(range(num_nodes))
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]
        
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            
    for u, v in pairs:
        union(u, v)
        
    clusters = {}
    for i in range(num_nodes):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)
        
    return [c for c in clusters.values() if len(c) > 1]

def compute_repetition_ratio(transcript: str) -> float:
    import re
    words = [w.lower() for w in re.findall(r'\b\w+\b', transcript)]
    if len(words) < 6:
        return 0.0
    
    trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    if not trigrams:
        return 0.0
        
    unique_trigrams = set(trigrams)
    duplicates = len(trigrams) - len(unique_trigrams)
    return float(duplicates) / len(trigrams)

def detect_and_resolve_retakes(egt_doc: EGTDocument) -> EGTDocument:
    """
    Length-agnostic bad-take detection via clustering.
    Identifies retried lines, groups them, and marks all but the best one as superseded.
    """
    segments = egt_doc.segments
    
    # 0. Intra-segment repetition detection (Step B.2)
    for seg in segments:
        ratio = compute_repetition_ratio(seg.transcript)
        seg.repetition_ratio = ratio
        if ratio >= settings.max_intra_segment_repetition_ratio:
            seg.is_stutter_repeat = True
            logger.info(f"Segment {seg.clip_id} marked as stutter repeat (ratio={ratio:.2f})")
            
    utterances = reconstruct_utterances(segments)
    
    if not utterances:
        return egt_doc
        
    edges = []
    
    # Pre-cache embeddings for the opening and whole text to avoid redundant computation
    opening_texts = [get_opening_words(u.full_transcript, 5).lower() for u in utterances]
    whole_texts = [u.full_transcript.lower() for u in utterances]
    
    # Pre-compute explicit markers via JEV
    is_explicit_marker = []
    for u in utterances:
        # Check the first ~15 words to see if it starts with an explicit self-correction
        t_opening_jev = get_opening_words(u.full_transcript, 15)
        if t_opening_jev:
            is_explicit_marker.append(check_is_explicit_retake_jev(t_opening_jev))
        else:
            is_explicit_marker.append(False)
            
    try:
        model = get_embedding_model()
        # Batch encode for speed
        valid_indices = [i for i, t in enumerate(opening_texts) if t]
        if valid_indices:
            opening_embs = model.encode([opening_texts[i] for i in valid_indices])
            whole_embs = model.encode([whole_texts[i] for i in valid_indices])
            
            emb_map = {idx: (opening_embs[j], whole_embs[j]) for j, idx in enumerate(valid_indices)}
        else:
            emb_map = {}
    except Exception as e:
        logger.error(f"Embedding failed: {e}. Falling back to explicit text markers only.")
        emb_map = {}
        
    # Step 2: Candidate pairing & Step 3: Similarity scoring
    for i in range(len(utterances)):
        u1 = utterances[i]
        t1 = whole_texts[i]
        
        if not t1:
            continue
            
        for j in range(i + 1, len(utterances)):
            u2 = utterances[j]
            t2 = whole_texts[j]
            
            if not t2:
                continue
                
            if u1.source_file != u2.source_file:
                continue
                
            # Must start within window of previous end
            gap = u2.start_sec - u1.end_sec
            if gap < 0 or gap > settings.retake_candidate_window_sec:
                continue
                
            # 1. Primary signal: Explicit self-correction phrases (JEV)
            is_match = False
            if is_explicit_marker[j]:
                is_match = True
                logger.info(f"Retake edge (explicit JEV): [{u1.start_sec:.1f}-{u1.end_sec:.1f}] vs [{u2.start_sec:.1f}-{u2.end_sec:.1f}]")
                    
            if i in emb_map and j in emb_map:
                o_emb1, w_emb1 = emb_map[i]
                o_emb2, w_emb2 = emb_map[j]
                
                # 2. Secondary signal: Opening-window similarity
                opening_sim = compute_cosine_similarity(o_emb1, o_emb2)
                if opening_sim >= settings.retake_opening_similarity_threshold:
                    if not is_match:
                        shape1 = f"{len(u1.segments)}-segment"
                        shape2 = f"{len(u2.segments)}-segment"
                        logger.info(f"Retake edge (opening {opening_sim:.2f}): {shape1} vs {shape2} at [{u1.start_sec:.1f}] vs [{u2.start_sec:.1f}]")
                    is_match = True
                    
                # 3. Tertiary signal: Whole-utterance similarity
                whole_sim = compute_cosine_similarity(w_emb1, w_emb2)
                if whole_sim >= settings.retake_whole_utterance_similarity_threshold:
                    if not is_match:
                        logger.info(f"Retake edge (whole {whole_sim:.2f}): [{u1.start_sec:.1f}] vs [{u2.start_sec:.1f}]")
                    is_match = True
            
            if is_match:
                edges.append((i, j))
                
    # Step 4: Clustering
    clusters = _build_clusters(edges, len(utterances))
    logger.info(f"Formed {len(clusters)} retake clusters.")
    
    # Step 5: Winner selection (disfluency-driven)
    for cluster_indices in clusters:
        # Score each utterance — primary signal is delivery cleanness
        best_score = -999.0
        winner_idx = -1
        
        for idx in cluster_indices:
            u = utterances[idx]
            
            # Primary signal: delivery cleanness via word-level disfluency
            all_word_timings = [s.word_timings for s in u.segments]
            disf = compute_utterance_disfluency(all_word_timings)
            # Invert: lower disfluency → higher cleanness (capped at 1.0)
            cleanness = 1.0 - min(disf["disfluency_score"] / 10.0, 1.0)
            
            # Secondary: content quality (averaged over sub-segments)
            avg_quality = sum(s.quality_score for s in u.segments) / len(u.segments)
            
            # Tertiary: recency as tiebreaker only
            recency = cluster_indices.index(idx) * 0.01
            
            score = cleanness * 0.6 + avg_quality * 0.3 + recency * 0.1
            
            logger.debug(
                f"  Take at {u.start_sec:.1f}s: cleanness={cleanness:.2f} "
                f"(restarts={disf['restart_count']}, fillers={disf['hesitation_ratio']:.2f}, "
                f"pauses={disf['pause_count']}), quality={avg_quality:.2f}, "
                f"total_score={score:.3f}"
            )
            
            if score > best_score:
                best_score = score
                winner_idx = idx
                
        # Mark losers as superseded
        for idx in cluster_indices:
            if idx != winner_idx:
                for seg in utterances[idx].segments:
                    seg.is_superseded_take = True
                    logger.info(f"Marked segment {seg.clip_id} as superseded take.")
            else:
                logger.info(f"Winner selected for cluster at {utterances[idx].start_sec:.1f}s (score={best_score:.3f}).")
                
    return egt_doc
