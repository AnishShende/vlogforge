"""Pass 2a — Word-Level Disfluency Detection.

Scans word_timings within a segment or utterance to detect delivery
problems that indicate a bad or unclean take:

    1. Repeated n-grams: "and then the and then the market..."
    2. Hesitation density: ratio of filler words (uh, um, er, like)
    3. Abnormal mid-phrase pauses: long gaps between words within a sentence

The composite disfluency_score is used by retake_detect.py to pick
the cleanest take in a cluster, replacing the old quality + recency heuristic.
"""

import logging
from typing import List, Dict, Tuple

logger = logging.getLogger("VlogForge.Disfluency")

# Common English filler words that indicate hesitation
FILLER_WORDS = frozenset({
    "uh", "um", "er", "erm", "ah", "eh", "hm", "hmm",
    "like",  # only when used as filler, but we count all instances as a proxy
})

# Self-correction markers that indicate the speaker is restarting
RESTART_MARKERS = frozenset({
    "sorry", "wait", "actually", "no",
})


def _normalize_word(w: str) -> str:
    """Strip punctuation and lowercase for comparison."""
    return w.lower().strip(",.!?;:\"'()-–—")


def count_repeated_ngrams(
    words: List[str],
    n_range: Tuple[int, int] = (2, 5),
    max_gap: int = 2,
) -> int:
    """Count instances where the speaker repeats a sequence of words.

    Looks for n-grams (n in n_range) that appear again within max_gap
    words of the original occurrence.  Each repeated instance counts once.

    Example:
        "and then the and then the market" → 1 repeated bigram
        "so today so today we are so today" → 2 repeated bigrams

    Args:
        words: List of normalised word strings.
        n_range: Tuple of (min_n, max_n) inclusive.
        max_gap: Maximum number of words between the end of the first
                 occurrence and the start of the repeat.

    Returns:
        Total number of repeated n-gram instances found.
    """
    if len(words) < n_range[0] * 2:
        return 0

    restart_count = 0

    for n in range(n_range[0], min(n_range[1] + 1, len(words) // 2 + 1)):
        seen_positions: Dict[tuple, int] = {}  # ngram_tuple → last_end_index

        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i + n])

            # Skip ngrams that are entirely filler words
            if all(w in FILLER_WORDS for w in ngram):
                continue

            if ngram in seen_positions:
                prev_end = seen_positions[ngram]
                gap = i - prev_end
                if 0 <= gap <= max_gap:
                    restart_count += 1
                    logger.debug(
                        f"Repeated {n}-gram at word {i}: {' '.join(ngram)} "
                        f"(gap={gap} words)"
                    )

            seen_positions[ngram] = i + n

    return restart_count


def count_abnormal_pauses(
    word_timings: List[Dict],
    threshold: float = 1.5,
) -> int:
    """Count mid-phrase pauses longer than threshold seconds.

    A "mid-phrase" pause is a gap between consecutive words that is
    significantly longer than expected for natural speech.  We exclude
    gaps that likely represent sentence boundaries (where a pause is normal).

    Args:
        word_timings: List of {word, start, end} dicts, sorted by start.
        threshold: Minimum gap duration in seconds to flag.

    Returns:
        Number of abnormal pauses found.
    """
    if len(word_timings) < 2:
        return 0

    count = 0
    for i in range(len(word_timings) - 1):
        current_end = word_timings[i].get("end", 0.0)
        next_start = word_timings[i + 1].get("start", 0.0)
        gap = next_start - current_end

        if gap >= threshold:
            # Check if the previous word ends with sentence-ending punctuation
            # If so, this is likely a natural pause between sentences
            prev_word_raw = word_timings[i].get("word", "")
            if prev_word_raw and prev_word_raw[-1] in ".!?":
                continue

            count += 1
            logger.debug(
                f"Abnormal pause: {gap:.2f}s after "
                f"'{word_timings[i].get('word', '')}' at {current_end:.2f}s"
            )

    return count


def compute_disfluency_score(word_timings: List[Dict]) -> Dict:
    """Compute a composite disfluency score for a segment or utterance.

    Lower score = cleaner delivery.  Zero = perfectly clean.

    Args:
        word_timings: List of {word, start, end} dicts from EGTSegment.word_timings.

    Returns:
        Dict with keys:
            disfluency_score: float (composite, lower = better)
            restart_count: int
            hesitation_ratio: float (0.0–1.0)
            pause_count: int
            word_count: int
    """
    if not word_timings:
        return {
            "disfluency_score": 0.0,
            "restart_count": 0,
            "hesitation_ratio": 0.0,
            "pause_count": 0,
            "word_count": 0,
        }

    words = [_normalize_word(wt.get("word", "")) for wt in word_timings]
    words = [w for w in words if w]  # Remove empty strings

    word_count = len(words)
    if word_count == 0:
        return {
            "disfluency_score": 0.0,
            "restart_count": 0,
            "hesitation_ratio": 0.0,
            "pause_count": 0,
            "word_count": 0,
        }

    # 1. Repeated n-grams
    restart_count = count_repeated_ngrams(words)

    # 2. Hesitation density
    filler_count = sum(1 for w in words if w in FILLER_WORDS)
    hesitation_ratio = filler_count / word_count

    # 3. Abnormal pauses
    pause_count = count_abnormal_pauses(word_timings)

    # Composite score: weighted sum (higher = more disfluent)
    # Restarts are the strongest signal (3x), followed by hesitation (2x),
    # then pauses (1x).
    score = (
        restart_count * 3.0
        + hesitation_ratio * 2.0
        + pause_count * 1.0
    )

    return {
        "disfluency_score": round(score, 3),
        "restart_count": restart_count,
        "hesitation_ratio": round(hesitation_ratio, 3),
        "pause_count": pause_count,
        "word_count": word_count,
    }


def compute_utterance_disfluency(word_timings_list: List[List[Dict]]) -> Dict:
    """Compute disfluency for a multi-segment utterance.

    Concatenates word_timings from all segments in the utterance
    and runs the disfluency scanner on the combined stream.

    Args:
        word_timings_list: List of word_timings lists, one per segment
                           in the utterance.

    Returns:
        Same dict format as compute_disfluency_score.
    """
    combined = []
    for wt_list in word_timings_list:
        combined.extend(wt_list)

    # Sort by start time to handle any ordering edge cases
    combined.sort(key=lambda w: w.get("start", 0.0))

    return compute_disfluency_score(combined)
