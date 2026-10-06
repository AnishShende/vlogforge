"""Score an edit against gold (Archdoc Phase 0 / S3). Pure functions, no I/O.

Pipeline-agnostic: the input is just the output timeline, i.e. source ranges in
output order, so the legacy EDL path, the word-timeline path and the future
Phase 2 compiler are all scored the same way.

Two independent views:
  timeline  exact source-range checks against gold keep/exclude spans
  text      re-transcribed output vs the expected keep text (catches what the
            timeline can't: audible clipping, fades, crossfades eating words)
"""

import difflib
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from eval.gold import Gold, all_spans

EDGE_TOL_SEC = 0.05      # a keep edge missing less than this is not "clipped"
LEAK_TOL_SEC = 0.10      # exclude overlap below this is reported but not a leak
CUT_SILENCE_WIN = 0.02   # a cut is "in silence" if +/- this window has no speech
LEAK_SPEECH_SEC = 0.05   # exclude overlap carrying this much activity is a leak, however short


@dataclass
class Range:
    source_file: str
    start: float
    end: float


Interval = Tuple[float, float]


def _union(iv: List[Interval]) -> List[Interval]:
    out: List[Interval] = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _overlaps(ranges: List[Range], f: str, s: float, e: float) -> List[Interval]:
    return [(max(s, r.start), min(e, r.end)) for r in ranges
            if r.source_file == f and r.start < e and s < r.end]


def _active_sec(speech: Optional[Dict], f: str, intervals: List[Interval]) -> Optional[float]:
    """Seconds of activity (3rd mask if given: speech OR loud; else the speech mask) in intervals."""
    if not speech or f not in speech:
        return None
    mask, hop = speech[f][0], speech[f][1]
    if len(speech[f]) > 2 and speech[f][2] is not None:
        mask = speech[f][2]
    return sum(float(mask[int(round(a / hop)): int(round(b / hop))].sum()) * hop for a, b in intervals)


def score_timeline(gold: Gold, ranges: List[Range], speech: Optional[Dict] = None) -> dict:
    """speech: optional {source_file: (speech frames, hop_sec[, activity frames])} masks for the
    cut-in-silence and unlabelled-speech metrics. With them, a gap removed INSIDE a keep is
    clipping only if it holds speech/activity (removing silence is not clipping), and an
    exclude overlap holding >= LEAK_SPEECH_SEC of activity is a leak however short it is.
    Activity = the optional 3rd mask (speech OR loud: VAD misses speech on noisy audio)."""
    keep_rows = []
    for k in gold.keep:
        dur = k.end - k.start
        ov = _overlaps(ranges, k.source_file, k.start, k.end)
        cov = _union(ov)
        covered = sum(e - s for s, e in cov)
        head = (cov[0][0] - k.start) if cov else dur
        tail = (k.end - cov[-1][1]) if cov else dur
        interior = max(0.0, dur - covered - max(head, 0) - max(tail, 0)) if cov else 0.0
        gaps = [(a[1], b[0]) for a, b in zip(cov, cov[1:])]
        interior_active = _active_sec(speech, k.source_file, gaps)
        interior_bad = interior > EDGE_TOL_SEC if interior_active is None else interior_active > EDGE_TOL_SEC
        multiplicity = sum(e - s for s, e in ov) / dur if dur else 0.0
        status = ("missing" if not cov else
                  "duplicated" if multiplicity > 1.5 else
                  "clipped" if head > EDGE_TOL_SEC or tail > EDGE_TOL_SEC or interior_bad else
                  "complete")
        first_pos = min((i for i, r in enumerate(ranges) if r.source_file == k.source_file
                         and r.start < k.end and k.start < r.end), default=None)
        keep_rows.append({"id": k.id, "status": status, "quality": k.quality, "covered_frac": round(covered / dur, 3),
                          "head_clip_sec": round(max(head, 0), 3), "tail_clip_sec": round(max(tail, 0), 3),
                          "interior_gap_sec": round(interior, 3),
                          "interior_gap_active_sec": None if interior_active is None else round(interior_active, 3),
                          "multiplicity": round(multiplicity, 2),
                          "output_pos": first_pos, "text": k.text})

    excl_rows = []
    for x in gold.exclude:
        ov = _union(_overlaps(ranges, x.source_file, x.start, x.end))
        leaked = sum(e - s for s, e in ov)
        active = _active_sec(speech, x.source_file, ov)
        excl_rows.append({"id": x.id, "reason": x.reason, "leaked_sec": round(leaked, 3),
                          "leaked_active_sec": None if active is None else round(active, 3),
                          "leaked": leaked > LEAK_TOL_SEC or (active is not None and active >= LEAK_SPEECH_SEC),
                          "text": x.text})

    opt_rows = []
    for o in gold.optional:
        inc = sum(e - s for s, e in _union(_overlaps(ranges, o.source_file, o.start, o.end)))
        opt_rows.append({"id": o.id, "included_sec": round(inc, 3),
                         "included_frac": round(inc / (o.end - o.start), 3), "text": o.text})

    present = [r for r in keep_rows if r["output_pos"] is not None]
    order = [r["output_pos"] for r in sorted(present, key=lambda r: next(
        k.start for k in gold.keep if k.id == r["id"]))]
    inversions = sum(1 for a, b in zip(order, order[1:]) if b < a)

    result = {
        "keep": keep_rows,
        "exclude": excl_rows,
        "optional": opt_rows,
        "summary": {
            "keep_total": len(keep_rows),
            **{f"keep_{s}": sum(r["status"] == s for r in keep_rows)
               for s in ("complete", "clipped", "missing", "duplicated")},
            "exclude_total": len(excl_rows),
            "exclude_leaked": sum(r["leaked"] for r in excl_rows),
            "exclude_leaked_sec": round(sum(r["leaked_sec"] for r in excl_rows), 3),
            "keep_imperfect_total": sum(r["quality"] == "imperfect" for r in keep_rows),
            "keep_imperfect_dropped": sum(r["quality"] == "imperfect" and r["status"] == "missing"
                                          for r in keep_rows),
            "optional_total": len(opt_rows),
            "optional_included": sum(r["included_frac"] > 0.5 for r in opt_rows),
            "keep_order_inversions": inversions,
            "output_ranges": len(ranges),
            "output_sec": round(sum(r.end - r.start for r in ranges), 3),
        },
    }
    if speech:
        result["summary"].update(_speech_metrics(gold, ranges, speech))
    return result


def _speech_metrics(gold: Gold, ranges: List[Range], speech: Dict) -> dict:
    edges_total = edges_silent = 0
    in_speech = []
    unlabelled = 0.0
    for r in ranges:
        mask, hop = speech[r.source_file][0], speech[r.source_file][1]
        w = max(1, int(round(CUT_SILENCE_WIN / hop)))
        for t, side in ((r.start, "start"), (r.end, "end")):
            i = int(round(t / hop))
            window = mask[max(0, i - w): min(len(mask), i + w + 1)]
            edges_total += 1
            if not window.any():
                edges_silent += 1
            else:
                in_speech.append({"source_file": r.source_file, "t": round(t, 3), "edge": side})
        a, b = int(r.start / hop), int(r.end / hop)
        seg = mask[a:b].copy()
        for s in all_spans(gold):
            if s.source_file == r.source_file:
                lo, hi = max(int(s.start / hop) - a, 0), min(int(s.end / hop) - a, len(seg))
                if lo < hi:
                    seg[lo:hi] = False
        unlabelled += float(seg.sum()) * hop
    return {
        "cut_edges": edges_total,
        "cut_edges_in_silence_pct": round(100 * edges_silent / edges_total, 1) if edges_total else None,
        "cut_edges_in_speech": in_speech,
        "unlabelled_speech_in_output_sec": round(unlabelled, 2),
    }


_TOKEN = re.compile(r"[a-z0-9']+")


def tokens(text: str) -> List[str]:
    return _TOKEN.findall(text.lower().replace("’", "'"))


def expected_text(gold: Gold, timeline: dict) -> str:
    """Keep texts in the order they appear in the output (missing keeps excluded:
    the timeline view already counts those)."""
    rows = [r for r in timeline["keep"] if r["output_pos"] is not None]
    return " ".join(r["text"] for r in sorted(rows, key=lambda r: r["output_pos"]))


def score_text(expected: str, heard: str) -> dict:
    exp, hyp = tokens(expected), tokens(heard)
    sm = difflib.SequenceMatcher(a=exp, b=hyp, autojunk=False)
    missing, extra, substituted = [], [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "delete":
            missing.append(" ".join(exp[i1:i2]))
        elif op == "insert":
            extra.append(" ".join(hyp[j1:j2]))
        elif op == "replace":
            substituted.append(f"{' '.join(exp[i1:i2])} -> {' '.join(hyp[j1:j2])}")
    n_missing = sum(len(m.split()) for m in missing)
    n_extra = sum(len(x.split()) for x in extra)
    n_sub = sum(max(len(s.split(' -> ')[0].split()), len(s.split(' -> ')[1].split())) for s in substituted)
    return {
        "expected_words": len(exp), "heard_words": len(hyp),
        "missing_words": n_missing, "extra_words": n_extra, "substituted_words": n_sub,
        "word_error_rate": round((n_missing + n_extra + n_sub) / len(exp), 3) if exp else None,
        "missing": missing, "extra": extra, "substituted": substituted,
    }
