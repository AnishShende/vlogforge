"""Word grid builder (Archdoc roadmap Phase 1 / S2).

Turns the per-file transcript word list (Whisper + forced alignment) into a
WordGrid: one Word per spoken word, deterministic IDs, per-file time order.
Edits will reference these IDs instead of segment clip_ids.

Built straight from the transcript, not from EGT segments: segments are a
derived view and split words across scene boundaries.
"""

import logging
from collections import Counter
from typing import Dict, List, Optional

from app.models import (WORD_FLAG_INTERPOLATED, WORD_FLAG_LONG_SPAN, WORD_FLAG_LOW_CONF,
                        WORD_FLAG_ON_SILENCE, WORD_FLAG_RECOVERED, WORD_FLAG_ZERO_LENGTH, Word, WordGrid, generate_word_id)
from app.utils.speech_activity import Envelope, possible_speech, snap_end, snap_start, uncovered_speech

logger = logging.getLogger("VlogForge.WordGrid")

# Speech-activity checks (Phase 1 / S3). Heuristics, chosen from the eval set's real
# distributions and documented in docs/ARCHDOC_ROADMAP.md; the ASR bake-off may revise.
EDGE_REFINE_SEC = 0.04          # word edges move at most this far toward the true onset/offset
LOW_CONF = 0.3                  # aligner score; fixed (not per-clip percentile) on purpose
LONG_SPAN_BASE_SEC = 0.6        # long_span if duration > BASE + PER_CHAR * len(text)
LONG_SPAN_PER_CHAR_SEC = 0.15
ON_SILENCE_MAX_SPEECH = 0.2     # on_silence if < 20% of the word's frames are speech or loud
WORD_BRIDGE_SEC = 0.5           # coverage: consecutive words closer than this cover the gap too
                                # (aligner edges are tight; VAD runs straight through inter-word pauses)


def word_coverage(words: List[Word], bridge_sec: float = WORD_BRIDGE_SEC) -> List[tuple]:
    """(start, end) stretches covered by words, bridging inter-word gaps < bridge_sec."""
    out: List[list] = []
    for w in sorted(words, key=lambda x: x.start):
        if out and w.start - out[-1][1] < bridge_sec:
            out[-1][1] = max(out[-1][1], w.end)
        else:
            out.append([w.start, w.end])
    return [tuple(x) for x in out]


def _split_multiword(entry: Dict) -> List[Dict]:
    """A transcript entry holding several words (segment-level fallback, e.g. Gemini
    STT) is spread evenly over its span; the per-word times are interpolated."""
    tokens = entry["text"].split()
    if len(tokens) <= 1:
        return [entry]
    s, e = float(entry["start"]), float(entry["end"])
    step = max(e - s, 0.0) / len(tokens)
    return [{**entry, "text": t, "start": s + step * k, "end": s + step * (k + 1), "interpolated": True}
            for k, t in enumerate(tokens)]


def build_word_grid(transcript: List[Dict], asr: Optional[Dict] = None) -> WordGrid:
    """transcript: [{start, end, text, video_file, conf?, conf_src?, interpolated?}, ...]
    as produced by transcribe_audio() and tagged with video_file by the orchestrator."""
    by_file: Dict[str, List[Dict]] = {}
    for entry in transcript:
        if not str(entry.get("text", "")).strip():
            continue
        if entry.get("start") is None or entry.get("end") is None:
            raise ValueError(f"untimed transcript entry reached the grid builder: {entry!r}")
        by_file.setdefault(entry.get("video_file", ""), []).extend(_split_multiword(entry))

    words: List[Word] = []
    clamped = 0
    conf_sources = Counter()
    for source_file in sorted(by_file):                  # deterministic file order
        entries = sorted(by_file[source_file], key=lambda x: (float(x["start"]), float(x["end"])))
        prev_end = None
        for n, e in enumerate(entries):
            start, end = float(e["start"]), float(e["end"])
            if prev_end is not None and start < prev_end:   # never produce overlaps
                clamped += 1
                start = prev_end
                end = max(end, start)
            flags = []
            if e.get("interpolated"):
                flags.append(WORD_FLAG_INTERPOLATED)
            if e.get("recovered"):
                flags.append(WORD_FLAG_RECOVERED)
            if end == start:
                flags.append(WORD_FLAG_ZERO_LENGTH)
            if e.get("conf") is not None:
                conf_sources[e.get("conf_src", "unknown")] += 1
            words.append(Word(
                id=generate_word_id(source_file, n),
                text=str(e["text"]).strip(),
                source_file=source_file,
                start=round(start, 4),
                end=round(end, 4),
                conf=e.get("conf"),
                gap_before=round(start - prev_end, 4) if prev_end is not None else 0.0,
                flags=flags,
            ))
            prev_end = round(end, 4)

    if clamped:
        logger.warning(f"[WORD-GRID] clamped {clamped} overlapping word(s) to the previous word's end")
    grid = WordGrid(words=words, asr={
        **(asr or {}),
        "conf_sources": dict(conf_sources),
        "words_without_conf": sum(w.conf is None for w in words),
        "overlaps_clamped": clamped,
    })
    errors = grid.validate_integrity()
    if errors:   # builder bug: fail loud, never hand a broken grid downstream
        raise ValueError(f"word grid failed integrity ({len(errors)}): {errors[:5]}")
    return grid


def summarize_word_grid(grid: WordGrid) -> Dict:
    flags = Counter(f for w in grid.words for f in w.flags)
    unacc = [e - s for spans in grid.checks.get("unaccounted_speech", {}).values() for s, e in spans]
    poss = [e - s for spans in grid.checks.get("possible_speech", {}).values() for s, e in spans]
    return {
        "unaccounted_speech": {"regions": len(unacc), "sec": round(sum(unacc), 2)},
        "possible_speech": {"regions": len(poss), "sec": round(sum(poss), 2)},
        "edges_refined": grid.checks.get("edges_refined"),
        "words": len(grid.words),
        "files": len({w.source_file for w in grid.words}),
        "flags": dict(flags),
        "words_without_conf": grid.asr.get("words_without_conf"),
        "conf_sources": grid.asr.get("conf_sources"),
        "overlaps_clamped": grid.asr.get("overlaps_clamped"),
    }


def check_word_grid(grid: WordGrid, envs: Dict[str, Envelope]) -> WordGrid:
    """Refine word edges against the audio and flag suspicious words, in place.

    Detect and flag only: words are never moved more than EDGE_REFINE_SEC, never
    re-ordered, never dropped. Repairs (re-alignment) are a later story.
    envs: {source_file: Envelope} from app.utils.speech_activity.envelope().
    """
    refined = 0
    by_file: Dict[str, List[Word]] = {}
    for w in grid.words:
        by_file.setdefault(w.source_file, []).append(w)

    for source_file, words in by_file.items():
        env = envs[source_file]
        for k, w in enumerate(words):
            lo = words[k - 1].end if k > 0 else 0.0                  # never cross a neighbour
            hi = words[k + 1].start if k + 1 < len(words) else float("inf")
            ns, _ = snap_start(env, w.start, w.end, outside_labelled=False, search_sec=EDGE_REFINE_SEC)
            ne, _ = snap_end(env, w.end, w.start, outside_labelled=False, search_sec=EDGE_REFINE_SEC)
            # an edge on silence may shrink inward without limit in snap_*; cap every move here
            ns = min(max(ns, w.start - EDGE_REFINE_SEC, lo), w.start + EDGE_REFINE_SEC)
            ne = max(min(ne, w.end + EDGE_REFINE_SEC, hi), w.end - EDGE_REFINE_SEC)
            if ne < ns:
                ns, ne = w.start, w.end
            if (round(ns, 4), round(ne, 4)) != (w.start, w.end):
                refined += 1
                w.start, w.end = round(ns, 4), round(ne, 4)

            a, b = env.t2i(w.start), max(env.t2i(w.end), env.t2i(w.start) + 1)
            active = env.speech[a:b] | (env.loud[a:b] if env.loud is not None else False)
            if active.mean() < ON_SILENCE_MAX_SPEECH and WORD_FLAG_ON_SILENCE not in w.flags:
                w.flags.append(WORD_FLAG_ON_SILENCE)
            if (w.end - w.start) > LONG_SPAN_BASE_SEC + LONG_SPAN_PER_CHAR_SEC * len(w.text) \
                    and WORD_FLAG_LONG_SPAN not in w.flags:
                w.flags.append(WORD_FLAG_LONG_SPAN)
            if w.conf is not None and w.conf < LOW_CONF and WORD_FLAG_LOW_CONF not in w.flags:
                w.flags.append(WORD_FLAG_LOW_CONF)
        prev_end = None
        for w in words:                                              # keep gap_before consistent
            w.gap_before = round(w.start - prev_end, 4) if prev_end is not None else 0.0
            prev_end = w.end

    checks = {"edges_refined": refined, "unaccounted_speech": {}, "possible_speech": {}}
    for source_file, words in by_file.items():
        spans = word_coverage(words)
        checks["unaccounted_speech"][source_file] = uncovered_speech(envs[source_file], spans)
        checks["possible_speech"][source_file] = possible_speech(envs[source_file], spans)
    checks["flags"] = dict(Counter(f for w in grid.words for f in w.flags))
    grid.checks = checks
    errors = grid.validate_integrity()
    if errors:
        raise ValueError(f"word grid failed integrity after checks ({len(errors)}): {errors[:5]}")
    return grid
