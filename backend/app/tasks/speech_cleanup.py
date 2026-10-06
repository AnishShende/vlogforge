"""Speech cleanup on the word grid (Archdoc Stage 5, roadmap Phase 4).

KEEP BY DEFAULT: every grid word is kept unless a better version of the same line exists
(roadmap S8, user decision 2026-10-06). The legacy selector kept only group winners, so
speech that never became a candidate was silently lost (chai: 246 of 311 gold keep words).

Reuses the word-timeline selector unchanged for candidates (maximal fluent runs), grouping
(same chunk OR embedding similarity) and the winner (JEV completeness+fluency, argmax).
A losing candidate's words are removed only if that candidate is the SAME LINE as its
group's winner: it shares grid words with it (an earlier start of the same run, i.e. a
restart / false start) or is similar to it by embedding (a retake). Different content that
merely sits in the same silence-delimited chunk is kept.

Output: per-word labels and ranges [{word_start, word_end, label, reason, conf}], and an
EditPlan of the keep ranges in source order.
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

from app.models import EditPlan, EditSegment, WordGrid
from app.tasks import word_timeline_redundancy as wtr

logger = logging.getLogger("VlogForge.SpeechCleanup")

KEEP, REMOVE, REVIEW = "keep", "remove", "review"   # review = kept, but a person should confirm the take


def _entries(grid: WordGrid) -> Tuple[List[wtr.WordTimelineEntry], Dict[int, int]]:
    """Grid words as the selector's timeline entries; map id(entry) -> grid index."""
    entries, back = [], {}
    for i, w in enumerate(grid.words):
        e = wtr.WordTimelineEntry(word=w.text, start_sec=w.start, end_sec=w.end, source_file=w.source_file,
                                  has_speech=True, quality_score=0.5)
        entries.append(e)
        back[id(e)] = i
    return entries, back


SAME_LINE_BIGRAMS = 0.5   # a stretch is an attempt at a line if >= half its word pairs are in that line's best take
STRETCH_PAUSE_SEC = 0.5   # stretches split at pauses this long: one stretch ~ one attempt, not a run of several


def _bigrams(tokens: List[str]) -> set:
    return set(zip(tokens, tokens[1:])) if len(tokens) > 1 else {(tokens[0],)} if tokens else set()


def label_grid(grid: WordGrid, scorer: Optional[Callable[[str], Tuple[float, float]]] = None) -> List[Dict]:
    """One label per grid word: {id, label, reason, conf, by}.
    1. candidates + groups + JEV winner per group (selector unchanged);
    2. winner = JEV's best score; winners' words: keep, or REVIEW when the line has another
       distinct take that JEV also scores as clean; otherwise ('best-take', or 'no-clean-take' when JEV scored < CLEAN_TAKE_MIN_SCORE);
    3. every other stretch of words (consecutive non-winner words, split at STRETCH_PAUSE_SEC pauses)
       is removed only if it is an attempt at a line that has several attempts: >= SAME_LINE_BIGRAMS
       of its word pairs occur in that line's winning take. Everything else is kept ('default')."""
    entries, back = _entries(grid)
    spans = wtr._generate_candidates(sorted(entries, key=lambda e: (e.source_file, e.start_sec)))
    groups = wtr._group_by_content(spans)
    if scorer is None:
        scorer = wtr._jev_scorer()
        if scorer is None:
            logger.warning("[CLEANUP] JEV unavailable — falling back to the deterministic lexical scorer")
            scorer = wtr._lexical_scorer

    W = grid.words
    labels = [{"id": w.id, "label": KEEP, "reason": "default", "conf": None, "by": None} for w in W]
    idx = lambda span: [back[id(e)] for e in span.words]
    scored_cache: Dict[str, Tuple[float, float]] = {}
    winners = []                                           # (span, score, conf, multi_attempt)
    for members in groups:
        scored = []
        for m in members:
            key = wtr._norm(spans[m].text)
            if key not in scored_cache:
                scored_cache[key] = scorer(spans[m].text)
            s, c = scored_cache[key]
            scored.append((s, c, len(spans[m].words), spans[m].start_sec, m))
        s, c, _, _, win = max(scored)
        winners.append((spans[win], s, c, len(members) > 1))
        # Two or more distinct complete takes: text cannot decide between them (delivery can),
        # so keep the best-scored one and flag the line for review (user decision 2026-10-06).
        win_idx = set(idx(spans[win]))
        alts = [x for x in scored if x[4] != win and x[0] >= wtr.CLEAN_TAKE_MIN_SCORE
                and not (set(idx(spans[x[4]])) & win_idx)]
        for i in win_idx:
            if alts and s >= wtr.CLEAN_TAKE_MIN_SCORE:
                labels[i].update(label=REVIEW, reason="best-of-several-takes", conf=round(c, 3),
                                 by={"alternatives": [{"text": spans[x[4]].text[:80], "start": spans[x[4]].start_sec,
                                                       "end": spans[x[4]].end_sec, "score": round(x[0], 3)} for x in alts]})
            else:
                labels[i].update(reason="best-take" if s >= wtr.CLEAN_TAKE_MIN_SCORE else "no-clean-take", conf=round(c, 3))
    in_winner = {i for sp, *_ in winners for i in idx(sp)}
    lines = [(sp, s, c, _bigrams([wtr._norm(e.word) for e in sp.words])) for sp, s, c, multi in winners if multi]

    stretches, cur = [], []
    for i, w in enumerate(W):
        if i in in_winner or (cur and (W[cur[-1]].source_file != w.source_file
                                       or w.start - W[cur[-1]].end >= STRETCH_PAUSE_SEC)):
            if cur:
                stretches.append(cur)
            cur = []
        if i not in in_winner:
            cur.append(i)
    if cur:
        stretches.append(cur)
    for st in stretches:
        grams = _bigrams([wtr._norm(W[i].text) for i in st])
        best = max(((len(grams & lg) / len(grams), sp, s, c) for sp, s, c, lg in lines
                    if sp.source_file == W[st[0]].source_file), key=lambda x: x[0], default=None)
        if best and best[0] >= SAME_LINE_BIGRAMS:
            frac, sp, s, c = best
            for i in st:
                labels[i].update(label=REMOVE, reason="retake", conf=round(frac, 3),
                                 by={"text": sp.text[:80], "start": sp.start_sec, "end": sp.end_sec, "score": round(s, 3)})
    n_remove = sum(l["label"] == REMOVE for l in labels)
    n_review = len({l["by"]["alternatives"][0]["start"] for l in labels if l["label"] == REVIEW})
    logger.info(f"[CLEANUP] {len(W)} words: keep {len(W) - n_remove}, remove {n_remove} "
                f"({len(spans)} candidates in {len(groups)} groups, {len(lines)} lines with several attempts, "
                f"{len(scored_cache)} scored); lines flagged for review: {n_review}")
    return labels


def label_ranges(grid: WordGrid, labels: List[Dict]) -> List[Dict]:
    """Consecutive words with the same label and reason (same file) as one range."""
    out: List[Dict] = []
    for w, l in zip(grid.words, labels):
        prev = out[-1] if out else None
        if prev and prev["label"] == l["label"] and prev["reason"] == l["reason"] and prev["source_file"] == w.source_file:
            prev["word_end"] = w.id
            prev["end"] = w.end
            prev["text"] += " " + w.text
        else:
            out.append({"word_start": w.id, "word_end": w.id, "label": l["label"], "reason": l["reason"],
                        "conf": l["conf"], "source_file": w.source_file, "start": w.start, "end": w.end, "text": w.text,
                        "by": l["by"]})
    return out


def cleanup_plan(grid: WordGrid, labels: List[Dict]) -> EditPlan:
    """EditPlan of the keep + review ranges, in source order (consecutive keep words are one segment)."""
    index = {w.id: i for i, w in enumerate(grid.words)}
    segs = []
    for r in label_ranges(grid, labels):
        if r["label"] == REMOVE:
            continue
        if segs and index[r["word_start"]] == index[segs[-1].word_end] + 1:
            segs[-1].word_end = r["word_end"]
        else:
            segs.append(EditSegment(word_start=r["word_start"], word_end=r["word_end"], reason=r["reason"]))
    return EditPlan(segments=segs, grid_fingerprint=grid.fingerprint())
