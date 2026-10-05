"""Audio-first, word-level clean-take selection (flag-gated diagnostic).

Design (see docs/SPEECH_FIRST_DESIGN.md):
  The word timeline is the unit of selection for speech. Segments are ignored
  here. The algorithm is punctuation-free and relies only on word text + timing.

  1. Build a flat word timeline from the EGT.
  2. Candidate generation: cheap pre-split at small silences, then within each
     chunk emit MAXIMAL FLUENT RUNS — maximal spans with no internal repeated
     n-gram — seeded at the chunk start and at every restart (a position where
     an n-gram recurs). Retake attempts and the final clean delivery all appear
     as candidates; the final delivery survives intact because it has no
     internal repeat.
  3. Group candidates by CONTENT similarity (opening + whole-text embeddings),
     not by silence — silence cannot separate retakes from distinct ideas.
  4. Per group keep the LONGEST fluent run (most complete; completeness is
     relative, no punctuation), tie-broken by latest start. Singletons kept.
  5. Drop leftover scraps that are substrings of a kept span. A group whose
     winner is still only a short fragment is kept but flagged `no-clean-take`.

  Every candidate ends up kept or dropped; span accounting asserts this.

NOT wired into the orchestrator/EDL/assembly (that is Phase 3, blocked).
"""
import logging
import re
from typing import List, Dict, Callable, Optional, Tuple
from dataclasses import dataclass, field

from app.models import EGTDocument
from app.tasks.retake_detect import get_embedding_model, compute_cosine_similarity

logger = logging.getLogger("VlogForge.WordTimeline")

# --- Tunables (validated against the real IMG_1614 transcript) --------------
PRESPLIT_GAP_SEC = 2.0     # break runs only on genuine idea gaps; small within-
                           # delivery pauses must NOT chop a fluent take. NOT the
                           # grouping rule (grouping is by content).
NGRAM = 4                  # repeated 4-gram marks a stutter/restart
MIN_WORDS = 6              # floor for a candidate to be considered a span
MAX_CLEAN_WORD_GAP_SEC = 1.0   # consecutive words must be ~back-to-back; a bigger
                               # gap means stutter/dead-air between them, so the
                               # FFmpeg cut would play it. Required for the span's
                               # time bounds to match clean audio (handoff rule).
MAX_WORD_DURATION_SEC = 2.0    # a single token longer than this is a Whisper
                               # stretch artifact (dead air / stutter collapsed
                               # into one word); never include it in a clean run.
SIMILARITY = 0.72          # opening/whole cosine sim to call two spans the same line
CLEAN_TAKE_MIN_SCORE = 3.0  # winner completeness+fluency score below this => no-clean-take


@dataclass
class WordTimelineEntry:
    word: str
    start_sec: float
    end_sec: float
    source_file: str
    has_speech: bool
    quality_score: float


@dataclass
class WordSpan:
    words: List[WordTimelineEntry] = field(default_factory=list)
    chunk_id: int = -1   # silence-delimited idea region this span came from

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
    def text(self) -> str:
        return " ".join(w.word for w in self.words).strip()


def build_word_timeline(egt_doc: EGTDocument) -> List[WordTimelineEntry]:
    """Flatten all EGT segments' word timings into one chronological stream."""
    timeline = []
    segments = sorted(egt_doc.segments, key=lambda s: (s.source_file, s.start_sec))
    for seg in segments:
        for wt in seg.word_timings:
            word = wt.get("text", wt.get("word", "")).strip()
            if not word:
                continue
            timeline.append(WordTimelineEntry(
                word=word,
                start_sec=wt.get("start", 0.0),
                end_sec=wt.get("end", 0.0),
                source_file=seg.source_file,
                has_speech=seg.has_speech,
                quality_score=seg.quality_score,
            ))
    timeline.sort(key=lambda x: (x.source_file, x.start_sec))
    return timeline


def _norm(w: str) -> str:
    return re.sub(r"[^\w\s]", "", w.lower())


def _has_internal_repeat(words: List[WordTimelineEntry], n: int = NGRAM) -> bool:
    nw = [_norm(w.word) for w in words]
    seen = set()
    for i in range(len(nw) - n + 1):
        g = tuple(nw[i:i + n])
        if g in seen:
            return True
        seen.add(g)
    return False


def _chunk_by_gap(timeline: List[WordTimelineEntry], gap: float) -> List[List[WordTimelineEntry]]:
    chunks, cur = [], []
    for w in timeline:
        if cur and (w.source_file != cur[-1].source_file
                    or (w.start_sec - cur[-1].end_sec) >= gap):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def _maximal_fluent_runs(chunk: List[WordTimelineEntry]) -> List[List[WordTimelineEntry]]:
    """Emit maximal spans with no internal repeated n-gram.

    Seeds: the chunk start and every restart position (where an n-gram recurs).
    From each seed, extend until adding a word would complete a repeated n-gram
    inside the run. The final clean delivery is seeded at the last restart and
    extends to the chunk end, so it survives as one full candidate.
    """
    nw = [_norm(w.word) for w in chunk]
    n = len(chunk)
    seeds = {0}
    seen = {}
    for i in range(n - NGRAM + 1):
        g = tuple(nw[i:i + NGRAM])
        if g in seen:
            seeds.add(i)           # a restart begins here
        else:
            seen[g] = i

    runs = []
    for s in sorted(seeds):
        local = set()
        e = s
        while e < n:
            # Temporal contiguity: a clean run's words must be back-to-back in
            # time. A larger inter-word gap is stutter/dead-air the cut would
            # otherwise play, so the run ends here.
            if e > s and (chunk[e].start_sec - chunk[e - 1].end_sec) > MAX_CLEAN_WORD_GAP_SEC:
                break
            # A single over-long token is a Whisper stretch artifact (dead air /
            # stutter collapsed into one word) — stop before including it.
            if (chunk[e].end_sec - chunk[e].start_sec) > MAX_WORD_DURATION_SEC:
                break
            if e - s >= NGRAM - 1:
                g = tuple(nw[e - NGRAM + 1:e + 1])
                if g in local:
                    break          # would repeat inside this run -> stop before e
                local.add(g)
            e += 1
        runs.append(chunk[s:e])
    return runs


def _generate_candidates(timeline: List[WordTimelineEntry]) -> List[WordSpan]:
    runs = []  # (chunk_id, run)
    for chunk_id, chunk in enumerate(_chunk_by_gap(timeline, PRESPLIT_GAP_SEC)):
        for run in _maximal_fluent_runs(chunk):
            if len(run) >= MIN_WORDS:
                runs.append((chunk_id, run))

    # Candidate reduction (cuts downstream JEV call volume). The restart-seeded
    # generator emits many overlapping runs that share an end word (left
    # extensions). Keep only the LONGEST run per distinct (source_file, end)
    # position, then drop exact-text duplicates. We deliberately do NOT
    # substring-prune across different ends — that once hid a clean take behind
    # a longer junk superset.
    by_end = {}
    for chunk_id, run in runs:
        key = (run[-1].source_file, round(run[-1].end_sec, 3))
        if key not in by_end or len(run) > len(by_end[key][1]):
            by_end[key] = (chunk_id, run)

    spans, seen_text = [], set()
    for chunk_id, run in sorted(by_end.values(), key=lambda cr: (cr[1][0].source_file, cr[1][0].start_sec)):
        norm_text = " ".join(_norm(w.word) for w in run)
        if norm_text not in seen_text:
            seen_text.add(norm_text)
            spans.append(WordSpan(words=list(run), chunk_id=chunk_id))
    return spans


def _group_by_content(spans: List[WordSpan]) -> List[List[int]]:
    """Union-find groups over opening/whole-text embedding similarity.
    Returns groups of indices, INCLUDING singletons."""
    n = len(spans)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if n > 1:
        whole = [s.text.lower() for s in spans]
        opens = [" ".join(s.text.split()[:6]).lower() for s in spans]
        try:
            model = get_embedding_model()
            we = model.encode(whole)
            oe = model.encode(opens)
        except Exception as e:
            logger.error(f"Embedding failed, grouping on chunk membership only: {e}")
            we = oe = None
        for i in range(n):
            for j in range(i + 1, n):
                if spans[i].source_file != spans[j].source_file:
                    continue
                # Same-idea edge via a UNION of signals:
                #  - same silence-delimited chunk (one idea region fragmented by
                #    Whisper into several contiguous pieces), OR
                #  - opening / whole-text embedding similarity (retakes of one
                #    line spread across chunks, e.g. separated by a long pause).
                same_chunk = spans[i].chunk_id == spans[j].chunk_id
                similar = we is not None and (
                    compute_cosine_similarity(oe[i], oe[j]) >= SIMILARITY or
                    compute_cosine_similarity(we[i], we[j]) >= SIMILARITY
                )
                if same_chunk or similar:
                    parent[find(i)] = find(j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# --- Winner selection: JEV completeness+fluency judgment -------------------
# This is the "small call" of the hybrid: lexical generates/groups candidates
# deterministically; JEV judges which one is the clean complete take (score
# 0..4, confidence 0..1). The JEV call + mock cache live in app.utils.jev.
# Punctuation plays no role.


def _jev_scorer() -> Optional[Callable[[str], Tuple[float, float]]]:
    """Return a JEV-backed scorer callable, or None if JEV is unavailable."""
    from app.utils.jev import jev_available, score_clean_take_candidate
    if not jev_available():
        return None

    def score(text: str) -> Tuple[float, float]:
        result = score_clean_take_candidate(text)
        # jev_available() was true; a None here means a transient call failure —
        # treat as lowest score so a flaky candidate never wins silently.
        return result if result is not None else (0.0, 0.0)

    return score


def _lexical_scorer(text: str) -> Tuple[float, float]:
    """Deterministic fallback when JEV is unavailable. Fails loud via caller log.
    Scores by word count minus a local-repetition penalty (normalized to ~0..4)."""
    words = text.split()
    nw = [_norm(w) for w in words]
    reps = 0
    seen = {}
    for i in range(len(nw) - 2):
        g = tuple(nw[i:i + 3])
        if g in seen and i - seen[g] <= 10:
            reps += 1
        seen[g] = i
    penalty = min(4.0, reps * 0.5)
    return max(0.0, min(4.0, len(words) / 8.0)) - penalty, 0.0


def detect_redundancy_on_timeline(
    egt_doc: EGTDocument,
    scorer: Optional[Callable[[str], Tuple[float, float]]] = None,
) -> Dict:
    timeline = build_word_timeline(egt_doc)
    spans = _generate_candidates(timeline)
    total_candidates = len(spans)

    # Next-word-start lookup, per source file. Used to cap a span's end so the
    # cut can't overrun into the next spoken word: when the speaker leaves no
    # pause (e.g. a restart right after the final word), the aligner stretches
    # that last word's end across the following audio, and cutting at it would
    # play the next word. Capping at the next word's onset trims that bleed
    # without clipping the kept word.
    next_start = {}
    for a, b in zip(timeline, timeline[1:]):
        if a.source_file == b.source_file:
            next_start[id(a)] = b.start_sec

    def _capped_end(span: WordSpan) -> float:
        lw = span.words[-1]
        nxt = next_start.get(id(lw))
        return min(lw.end_sec, nxt) if nxt is not None else lw.end_sec

    groups = _group_by_content(spans)

    # Winner selection (the hybrid's "small call"): JEV judges completeness +
    # fluency per candidate; code picks the argmax. Injectable for hermetic
    # tests. Falls back loudly to a deterministic lexical scorer if JEV is down.
    if scorer is None:
        scorer = _jev_scorer()
        if scorer is None:
            logger.warning("JEV unavailable — falling back to deterministic lexical scorer")
            scorer = _lexical_scorer

    kept = []
    dropped = []
    for members in groups:
        # Dedup exact-text duplicates only (saves scorer calls); never prune by
        # substring — that once hid the clean take behind a longer junk superset.
        seen, distinct = set(), []
        for m in members:
            key = _norm(spans[m].text)
            if key not in seen:
                seen.add(key)
                distinct.append(m)

        scored = []
        for m in distinct:
            s, c = scorer(spans[m].text)
            scored.append((s, c, len(spans[m].words), spans[m].start_sec, m))
        scored.sort(reverse=True)
        win_score, win_conf, _, _, win = scored[0]

        flags = []
        if win_score < CLEAN_TAKE_MIN_SCORE:
            flags.append("no-clean-take")
        w = spans[win]
        end = _capped_end(w)
        wt = [{"word": x.word, "start": x.start_sec, "end": x.end_sec} for x in w.words]
        if wt:
            wt[-1]["end"] = min(wt[-1]["end"], end)  # keep word_timings consistent
        kept.append({
            "source_file": w.source_file,
            "start": w.start_sec,
            "end": end,
            "text": w.text,
            "word_timings": wt,
            "word_count": len(w.words),
            "duration": round(end - w.start_sec, 3),
            "group_size": len(members),
            "score": round(win_score, 3),
            "confidence": round(win_conf, 3),
            "flags": flags,
        })
        for m in members:
            if m != win:
                dropped.append({
                    "start": spans[m].start_sec,
                    "end": spans[m].end_sec,
                    "text": spans[m].text,
                    "reason": "superseded-in-group",
                })

    kept.sort(key=lambda k: k["start"])

    # --- Span conservation accounting (permanent, fail-loud) ---------------
    accounting = {
        "total_candidates": total_candidates,
        "kept": len(kept),
        "dropped": len(dropped),
    }
    logger.info(
        f"SPAN ACCOUNTING: total_candidates={total_candidates} "
        f"kept={len(kept)} dropped={len(dropped)}"
    )
    assert len(kept) + len(dropped) == total_candidates, (
        f"Span accounting leak: {total_candidates} candidates != "
        f"{len(kept)} kept + {len(dropped)} dropped"
    )

    return {"kept": kept, "dropped": dropped, "accounting": accounting}


WORD_TIMELINE_MODEL = "word-timeline-jev"


def clamp_edl_to_word_timeline(edl, egt_doc: EGTDocument) -> int:
    """Restore the exact clean bounds on EDL entries that reference word-timeline
    pseudo-segments, overriding any trim the LLM reasoner or the word-snap pass
    applied. The word-timeline span IS the cut decision; the reasoner may only
    select/order/type these clips, never move their start/end. Returns the count
    of clamped entries. Non-speech/B-roll entries are left untouched.
    """
    bounds = {
        seg.clip_id: (seg.start_sec, seg.end_sec)
        for seg in egt_doc.segments
        if seg.perception_model == WORD_TIMELINE_MODEL
    }
    n = 0
    for entry in edl:
        b = bounds.get(entry.get("clip_id"))
        if b is None:
            continue
        start, end = b
        if entry.get("start_sec") != start or entry.get("end_sec") != end:
            logger.info(
                f"Clamp {entry.get('clip_id')}: EDL "
                f"[{entry.get('start_sec')}-{entry.get('end_sec')}] -> clean [{start}-{end}]"
            )
        entry["start_sec"] = start
        entry["end_sec"] = end
        entry["core_start_sec"] = start
        entry["core_end_sec"] = end
        n += 1
    return n


def apply_word_timeline_selection(egt_doc: EGTDocument, scorer=None):
    """Replace the SPEECH portion of an EGT with word-timeline-selected clean
    takes, keeping non-speech/B-roll segments for the legacy path.

    Returns (new_egt_doc, warnings). Each kept speech span becomes a synthetic
    SPEECH EGTSegment with a deterministic clip_id and its own word_timings, so
    the existing EDL reasoner, word-snap, and clip_id anti-hallucination gate
    work unchanged. `no-clean-take` spans are kept (best fragment) and reported
    as warnings. `scorer` is forwarded for hermetic tests (defaults to JEV).
    """
    from app.models import EGTSegment, generate_clip_id

    report = detect_redundancy_on_timeline(egt_doc, scorer=scorer)

    pseudo_speech = []
    warnings = []
    for k in report["kept"]:
        clip_id = generate_clip_id(k["source_file"], k["start"], k["end"])
        pseudo_speech.append(EGTSegment(
            clip_id=clip_id,
            source_file=k["source_file"],
            start_sec=k["start"],
            end_sec=k["end"],
            transcript=k["text"],
            word_timings=k["word_timings"],
            has_speech=True,
            segment_type="SPEECH",
            quality_score=k.get("score", 1.0),
            perception_model=WORD_TIMELINE_MODEL,
        ))
        if "no-clean-take" in k["flags"]:
            warnings.append(
                f"No clean take for line at {k['start']:.1f}-{k['end']:.1f}s "
                f"(kept best fragment): {k['text'][:80]!r}"
            )

    # Carry over only TRUE B-roll: segments the classifier explicitly typed
    # B_ROLL. We must NOT use `has_speech == False` as the B-roll test — a
    # stutter region Whisper under-transcribed (<3 words, or none) also has
    # has_speech=False, and carrying it resurrects exactly the unintelligible
    # speech the word-timeline dropped. Real B-roll is a visual classification,
    # not a word-count artifact; speech-region leftovers are intentionally
    # discarded (their clean words were already the word-timeline's job).
    non_speech = [
        seg for seg in egt_doc.segments
        if seg.segment_type == "B_ROLL"
        and not seg.is_bad_take
    ]

    merged = sorted(pseudo_speech + non_speech,
                    key=lambda s: (s.source_file, s.start_sec))
    logger.info(
        f"Word-timeline selection: {len(pseudo_speech)} clean speech takes + "
        f"{len(non_speech)} non-speech segments ({len(warnings)} no-clean-take)"
    )

    new_doc = EGTDocument(
        segments=merged,
        total_duration_sec=egt_doc.total_duration_sec,
        source_file_count=egt_doc.source_file_count,
        context_summary=egt_doc.context_summary,
        perception_model_version=egt_doc.perception_model_version,
    )
    return new_doc, warnings
