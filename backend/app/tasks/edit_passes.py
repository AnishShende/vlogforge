"""Speech edit passes (roadmap Phase 6.5): LLM passes over the verbatim word grid, each on the edit
left by the previous one. The model only names word ranges to remove, with a category and a reason;
this code validates them, applies them to the kept-word set and records every cut. Times, cuts and
the render stay with the compiler.

v2 (2026-10-07, after the v1 eval): whole takes first, small cleanups last; tag, then decide.
  1. retake hints   deterministic: local alignment (Smith-Waterman) between attempts finds shared
                    wording, also when one attempt stops halfway or has fillers. Pairs, never chained
                    into groups; a difference in numbers or negation is not a retake.
  2. retakes        (applied) same line said more than once: keep the best take. Hints are evidence,
                    not decisions: redos in different words must be found from meaning.
  3. incomplete     (applied) false starts, self-corrections, restart cues, remarks not meant for
                    the audience.
  4. final_review   (applied) dangling words, references to cut content, a point made twice.
  5. inside_take    (suggestions only) stutters, cut-off words, fillers inside kept takes.
Uncertain cuts become suggestions for the review UI (like every inside_take cut), except retake
choices that the wording detector corroborates: those are applied and flagged for review, since keeping
both takes would repeat the line. (Dead gaps = compiler pause shortening; visual judgement deferred.)

The transcript shown to the model is the CURRENT edit split into numbered attempts (pause >= 0.4 s
or a cut between two words), each word with its index in the attempt. Results are cached by the
exact request (pass, model, prompt version, transcript), so a replay is free and identical.
"""

import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Set, Tuple

import anthropic

from app.config import settings
from app.models import WordGrid
from app.utils import artifacts

logger = logging.getLogger("VlogForge.EditPasses")

PROMPT_VERSION = "3"
ATTEMPT_PAUSE_SEC = 0.4
CHUNK_WORDS = 160          # inside_take: target attempts per call, ~this many words
CONTEXT_ATTEMPTS = 2       # inside_take: read-only attempts shown on each side of a chunk
SMALL_MODEL = "claude-haiku-4-5"
LARGE_MODEL = "claude-sonnet-5-5"

# Retake hints (local alignment over normalised tokens)
SW_MATCH, SW_MISMATCH, SW_GAP, SW_FILLER_GAP = 2, -1, -1, 0
HINT_MIN_MATCHED = 4       # shared tokens in the aligned region ...
HINT_MIN_CONTENT = 2       # ... of which at least this many are not function words
MAX_HINTS = 200
HINT_PARTNERS = 3         # strongest matches kept per attempt
FILLERS = {"um", "uh", "uhm", "umm", "er", "erm", "hmm", "mm", "ah", "eh"}
NEGATIONS = {"not", "no", "never", "dont", "don't", "cant", "can't", "wont", "won't", "isnt", "isn't",
             "didnt", "didn't", "nahi", "nahin", "mat"}
FUNCTION_WORDS = {"i", "you", "we", "he", "she", "it", "they", "a", "an", "the", "and", "or", "but", "so",
                  "to", "of", "in", "on", "at", "for", "is", "are", "was", "be", "that", "this", "if", "my",
                  "your", "me", "do", "just", "like", "with", "as", "it's", "i'm", "what", "there"}

COMMON = """You are editing the transcript of a video in which one creator talks to the camera.
The transcript is verbatim speech recognition: it keeps stutters, repeats, fillers, false starts and
retakes. Punctuation and capitalisation are unreliable and small recognition errors are common. The
speech may be in any language or a mix (for example Hindi and English).
The video is edited only by removing words: you cannot add, reorder or rewrite anything.
The transcript is split into numbered attempts wherever the speaker paused or an earlier edit removed
words; each word carries its index inside its attempt, like [3]word.
Remove only what this pass is about. Never remove information that is said only once and that the
viewer needs. Return word ranges (attempt number, first and last word index, inclusive), each with a
category and a short reason. Set "uncertain" to true whenever you are not confident the creator would
want the cut: the creator reviews uncertain cuts."""

PASSES: Dict[str, Dict] = {
    "retakes": {
        "model": LARGE_MODEL, "chunked": False, "applied": True,
        "categories": ["earlier_take", "partial_take", "worse_take"],
        "task": """This pass handles REPEATED TAKES. The creator sometimes says the same line or makes the same
point more than once because an attempt did not go well; the wording may be the same or different.
A retake is an attempt the speaker ABANDONED and then said again: they stopped, restarted or redid it.
For each line said more than once, keep exactly one take and remove the others, including partial
takes that stop before the end of the line. Choose the take to keep by, in this order:
  1. meaning: it says everything the line needs (facts, qualifiers, corrections), with nothing wrong or
     oddly worded;
  2. completeness: the thought is finished and fits what comes before and after it;
  3. fluency: fewer stumbles, restarts, repeated words or phrases and hesitations.
Judge every take exactly as it is written: repeats and restarts INSIDE a take count against it, because
nothing inside a kept take will be cleaned up later.
The position of a take (earlier or later) is NOT a criterion. When two complete takes are about equally
good, keep the first one and mark the cut uncertain.
A take can start or end inside an attempt: remove word ranges, not only whole attempts.
Not retakes: lines said only once; deliberate repetition for emphasis or humour; a few words restated
immediately inside flowing speech (that is normal talking, not a redo: leave it); statements that look
alike but differ in numbers, names, negation or time (they may be different facts).
"Possible repeats" below were found by matching wording. They are evidence, not decisions: confirm
each from meaning, and also find repeats that use different words.""",
    },
    "incomplete": {
        "model": LARGE_MODEL, "chunked": False, "applied": True,
        "categories": ["false_start", "self_correction", "restart_cue", "off_audience_remark"],
        "task": """This pass removes speech that is not part of the finished video:
1. false_start: a sentence that is abandoned and never completed, where the speaker moves on or
   starts over. A short incomplete phrase that still makes sense in context is NOT a false start.
2. self_correction: a wrong statement that the speaker immediately corrects. Remove the wrong part and
   the correcting words; keep the corrected statement.
3. restart_cue: the speaker signalling a restart ("start again", "one more time", "okay, again"),
   together with the abandoned attempt it refers to.
4. off_audience_remark: speech clearly not meant for the viewer: talking to someone off camera,
   checking the camera or the recording, judging their own take.
Remarks addressed to the viewer are content, even when casual, chatty or off-topic: keep them. Keep
asides, jokes, reactions and imperfect but complete sentences.""",
    },
    "final_review": {
        "model": LARGE_MODEL, "chunked": False, "applied": True,
        "categories": ["dangling", "orphan_reference", "duplicate_point"],
        "task": """This is the FINAL REVIEW. The transcript has already been cleaned; read it as the viewer will hear
it and remove only:
1. dangling: a word or half phrase left behind by earlier edits that now leads nowhere.
2. orphan_reference: a sentence that refers to something that is no longer in the video.
3. duplicate_point: the same point made twice in a row in different words where the second adds
   nothing; keep the better one.
Make very few changes. Words that connect two sentences naturally are not dangling.""",
    },
    "inside_take": {
        "model": SMALL_MODEL, "chunked": True, "applied": False,
        "categories": ["stutter", "cut_off_word", "filler", "filler_phrase"],
        "task": """This pass suggests small cleanups INSIDE sentences (the creator reviews each one):
1. stutter: a word or the start of a word said two or more times in a row by accident. Suggest
   removing the extra copies, keeping the last complete one. Deliberate repetition for emphasis
   ("very, very") is not a stutter.
2. cut_off_word: a word abandoned partway AND restarted right after it. Words in other languages or
   unusual spellings are not cut-off words.
3. filler: hesitation sounds such as um, uh, er, hmm.
4. filler_phrase: stalling words ("you know", "I mean", "basically") only when the sentence reads
   naturally without them. Keep words that connect ideas ("so", "and", "but", "then").
Do not remove whole sentences. Only attempts marked TARGET may be edited; CONTEXT attempts are there
to help you understand.""",
    },
}
ORDER = ["retakes", "incomplete", "final_review", "inside_take"]


def attempts_of(grid: WordGrid, kept: Set[str]) -> List[List[int]]:
    """Kept word indices grouped into attempts: break at a file change, a pause >= ATTEMPT_PAUSE_SEC
    or a removed word between two kept words."""
    W, out = grid.words, []
    prev = None
    for i, w in enumerate(W):
        if w.id not in kept:
            continue
        if (prev is not None and prev == i - 1 and W[prev].source_file == w.source_file
                and w.start - W[prev].end < ATTEMPT_PAUSE_SEC):
            out[-1].append(i)
        else:
            out.append([i])
        prev = i
    return out


def _norm(t: str) -> str:
    return re.sub(r"[^\w']", "", t.lower())


def local_align(a: List[str], b: List[str]) -> Tuple[int, List[Tuple[int, int]]]:
    """Smith-Waterman over tokens. Fillers are skipped at no cost. Returns (score, matched index pairs)."""
    n, m = len(a), len(b)
    H = [[0] * (m + 1) for _ in range(n + 1)]
    best, at = 0, (0, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = H[i - 1][j - 1] + (SW_MATCH if a[i - 1] == b[j - 1] else SW_MISMATCH)
            up = H[i - 1][j] + (SW_FILLER_GAP if a[i - 1] in FILLERS else SW_GAP)
            left = H[i][j - 1] + (SW_FILLER_GAP if b[j - 1] in FILLERS else SW_GAP)
            H[i][j] = max(0, diag, up, left)
            if H[i][j] > best:
                best, at = H[i][j], (i, j)
    pairs, (i, j) = [], at
    while i > 0 and j > 0 and H[i][j] > 0:
        if H[i][j] == H[i - 1][j - 1] + (SW_MATCH if a[i - 1] == b[j - 1] else SW_MISMATCH):
            if a[i - 1] == b[j - 1]:
                pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif H[i][j] == H[i - 1][j] + (SW_FILLER_GAP if a[i - 1] in FILLERS else SW_GAP):
            i -= 1
        else:
            j -= 1
    return best, pairs[::-1]


def retake_hints(grid: WordGrid, atts: List[List[int]]) -> List[Dict]:
    """Attempt pairs sharing wording (local alignment), with the aligned word ranges. Pairs whose
    aligned regions differ in numbers or negation are dropped (different facts, not takes)."""
    toks = [[_norm(grid.words[i].text) for i in a] for a in atts]
    bigrams = [set(zip(t, t[1:])) for t in toks]
    hints = []
    for x in range(len(atts)):
        for y in range(x + 1, len(atts)):
            if grid.words[atts[x][0]].source_file != grid.words[atts[y][0]].source_file or not bigrams[x] & bigrams[y]:
                continue
            score, pairs = local_align(toks[x], toks[y])
            matched = [toks[x][i] for i, _ in pairs]
            if len(matched) < HINT_MIN_MATCHED or sum(t not in FUNCTION_WORDS for t in matched) < HINT_MIN_CONTENT:
                continue
            (a0, b0), (a1, b1) = pairs[0], pairs[-1]
            ra, rb = toks[x][a0:a1 + 1], toks[y][b0:b1 + 1]
            diff = set(ra) ^ set(rb)
            if any(t.replace(".", "").replace("%", "").isdigit() for t in diff) or diff & NEGATIONS:
                continue
            hints.append({"a": x, "a_words": (a0, a1), "b": y, "b_words": (b0, b1), "shared": len(matched),
                          "score": score})
    # each attempt keeps only its strongest partners: evidence for the model, not an exhaustive list
    hints.sort(key=lambda h: -h["score"])
    per, out = {}, []
    for h in hints:
        if per.get(h["a"], 0) < HINT_PARTNERS and per.get(h["b"], 0) < HINT_PARTNERS:
            out.append(h)
            per[h["a"]] = per.get(h["a"], 0) + 1
            per[h["b"]] = per.get(h["b"], 0) + 1
    return out[:MAX_HINTS]


def internal_repeats(tokens: List[str], window: int = 12) -> int:
    """Tokens inside a phrase (>= 2 words) that is said again within `window` tokens: "because every
    because every" -> 2. Fillers ignored. A stumble count for comparing takes of the same line."""
    t = [x for x in tokens if x not in FILLERS]
    rep = set()
    for i in range(len(t)):
        for j in range(i + 2, min(len(t), i + window)):
            k = 0
            while j + k < len(t) and i + k < j and t[i + k] == t[j + k]:
                k += 1
            if k >= 2:
                rep |= set(range(i, i + k))
    return len(rep)


def render(grid: WordGrid, atts: List[List[int]], targets: Optional[range] = None) -> str:
    W, lines = grid.words, []
    for n, a in enumerate(atts):
        tag = "" if targets is None else (" TARGET" if n in targets else " CONTEXT")
        pause = W[a[0]].start - W[atts[n - 1][-1]].end if n else 0.0
        t = W[a[0]].start
        lines.append(f"#{n + 1}{tag} ({int(t // 60)}:{t % 60:04.1f}, pause before {pause:.1f}s): "
                     + " ".join(f"[{k}]{W[i].text}" for k, i in enumerate(a)))
    return "\n".join(lines)


def render_hints(hints: List[Dict]) -> str:
    if not hints:
        return "Possible repeats (matching wording): none found."
    return "Possible repeats (matching wording):\n" + "\n".join(
        f"- #{h['a'] + 1}[{h['a_words'][0]}-{h['a_words'][1]}] ~ #{h['b'] + 1}[{h['b_words'][0]}-{h['b_words'][1]}]"
        f" ({h['shared']} words shared)" for h in sorted(hints, key=lambda h: (h["a"], h["b"])))


def _schema(categories: List[str]) -> Dict:
    cut = {"type": "object", "additionalProperties": False,
           "required": ["attempt", "from_word", "to_word", "category", "reason", "uncertain"],
           "properties": {"attempt": {"type": "integer"}, "from_word": {"type": "integer"},
                          "to_word": {"type": "integer"}, "category": {"type": "string", "enum": categories},
                          "reason": {"type": "string"}, "uncertain": {"type": "boolean"}}}
    return {"type": "object", "additionalProperties": False, "required": ["cuts"],
            "properties": {"cuts": {"type": "array", "items": cut}}}


_client: Optional[anthropic.Anthropic] = None


def _call(model: str, system: str, user: str, categories: List[str]) -> Tuple[List[Dict], Dict]:
    """One structured-output request, cached by its exact content. Returns (cuts, usage)."""
    key = hashlib.sha256(json.dumps([PROMPT_VERSION, model, system, user, categories]).encode()).hexdigest()
    hit = artifacts.get("edit_pass", key)
    if hit is not None:
        return hit["cuts"], {**hit["usage"], "cached": True}
    global _client
    _client = _client or anthropic.Anthropic(api_key=settings.claude_api_key)
    fmt = {"format": {"type": "json_schema", "schema": _schema(categories)}}
    if model == LARGE_MODEL:     # Sonnet 5.5: adaptive thinking (default), server-side refusal fallback
        r = _client.beta.messages.create(model=model, max_tokens=16000, system=system,
                                         messages=[{"role": "user", "content": user}],
                                         output_config={"effort": "medium", **fmt},
                                         betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    else:
        r = _client.messages.create(model=model, max_tokens=8000, system=system,
                                    messages=[{"role": "user", "content": user}], output_config=fmt)
    if r.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"edit pass call on {model} stopped: {r.stop_reason}")
    cuts = json.loads(next(b.text for b in r.content if b.type == "text"))["cuts"]
    usage = {"model": r.model, "input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}
    artifacts.put("edit_pass", key, {"cuts": cuts, "usage": usage})
    return cuts, {**usage, "cached": False}


def _chunks(atts: List[List[int]]) -> List[range]:
    out, start, words = [], 0, 0
    for n, a in enumerate(atts):
        words += len(a)
        if words >= CHUNK_WORDS:
            out.append(range(start, n + 1))
            start, words = n + 1, 0
    if start < len(atts):
        out.append(range(start, len(atts)))
    return out


def run_pass(name: str, grid: WordGrid, kept: Set[str], model: Optional[str] = None) -> Tuple[List[Dict], Dict]:
    """One pass on the current edit. Returns (validated cuts with word ids, stats). Each cut has
    `applied`: False for uncertain cuts and for suggestion-only passes."""
    spec = PASSES[name]
    model = model or spec["model"]
    atts = attempts_of(grid, kept)
    system = COMMON + "\n\n" + spec["task"]
    n_hints, hints = 0, []
    if spec["chunked"]:
        jobs = []
        for ch in _chunks(atts):
            lo, hi = max(0, ch.start - CONTEXT_ATTEMPTS), min(len(atts), ch.stop + CONTEXT_ATTEMPTS)
            targets = range(ch.start - lo, ch.stop - lo)        # numbering inside the request is local
            jobs.append((lo, targets, render(grid, atts[lo:hi], targets)))
        with ThreadPoolExecutor(max_workers=4) as ex:
            res = list(ex.map(lambda j: _call(model, system, j[2], spec["categories"]), jobs))
        raw = [(j[0], j[1], c) for j, (cs, _) in zip(jobs, res) for c in cs]
        usages = [u for _, u in res]
    else:
        user = render(grid, atts)
        if name == "retakes":
            hints = retake_hints(grid, atts)
            n_hints = len(hints)
            user += "\n\n" + render_hints(hints)
        cs, u = _call(model, system, user, spec["categories"])
        raw, usages = [(0, range(len(atts)), c) for c in cs], [u]

    hinted: Set[int] = set()        # grid word indices inside a wording-matched repeat (retakes only)
    if name == "retakes":
        for h in hints:
            hinted |= set(atts[h["a"]][h["a_words"][0]:h["a_words"][1] + 1])
            hinted |= set(atts[h["b"]][h["b_words"][0]:h["b_words"][1] + 1])
    accepted, rejected = [], []
    for offset, allowed, c in raw:
        local = c["attempt"] - 1
        a = atts[offset + local] if local in allowed and 0 <= offset + local < len(atts) else None
        if a is None or not 0 <= c["from_word"] <= c["to_word"] < len(a):
            rejected.append(c)
            continue
        span = a[c["from_word"]:c["to_word"] + 1]
        accepted.append({"pass": name, "category": c["category"], "reason": c["reason"], "uncertain": c["uncertain"],
                         # uncertain cuts become suggestions, except an uncertain retake choice that the
                         # wording detector corroborates: applied (else the line plays twice), flagged for review
                         "applied": spec["applied"] and (not c["uncertain"]
                                                        or (name == "retakes" and len(set(span) & hinted) >= len(span) / 2)),
                         "review": c["uncertain"],
                         "word_ids": [grid.words[i].id for i in span], "text": " ".join(grid.words[i].text for i in span)})
    if name == "retakes":
        accepted = _fluency_tiebreak(grid, atts, hints, accepted)
    if rejected:
        logger.warning(f"[EDIT-PASS] {name}: rejected {len(rejected)} cut(s) with invalid attempt/word numbers: {rejected[:3]}")
    applied = [c for c in accepted if c["applied"]]
    stats = {"pass": name, "model": model, "calls": len(usages), "cached": sum(u["cached"] for u in usages),
             "input_tokens": sum(u["input_tokens"] for u in usages), "output_tokens": sum(u["output_tokens"] for u in usages),
             "hints": n_hints, "cuts": len(applied), "words_cut": len({i for c in applied for i in c["word_ids"]}),
             "suggestions": len(accepted) - len(applied), "rejected": len(rejected)}
    logger.info(f"[EDIT-PASS] {stats}")
    return accepted, stats


def _fluency_tiebreak(grid: WordGrid, atts: List[List[int]], hints: List[Dict], cuts: List[Dict]) -> List[Dict]:
    """For an UNCERTAIN take choice on a wording-matched pair, keep the side with fewer internal
    repeats: the model may judge a stumbling take as if its stumble would be cleaned up."""
    W = grid.words
    pos = {W[i].id: i for a in atts for i in a}
    out = list(cuts)
    for h in hints:
        side_a = atts[h["a"]][h["a_words"][0]:h["a_words"][1] + 1]
        side_b = atts[h["b"]][h["b_words"][0]:h["b_words"][1] + 1]
        for k, c in enumerate(out):
            if not (c["applied"] and c["review"]):
                continue
            span = {pos[i] for i in c["word_ids"] if i in pos}
            cut_side, kept_side = (side_a, side_b) if len(span & set(side_a)) > len(side_a) / 2 else \
                                  (side_b, side_a) if len(span & set(side_b)) > len(side_b) / 2 else (None, None)
            if cut_side is None or any(c2["applied"] and set(c2["word_ids"]) & {W[i].id for i in kept_side} for c2 in out):
                continue
            r_cut = internal_repeats([_norm(W[i].text) for i in atts[h["a"] if cut_side is side_a else h["b"]]])
            r_kept = internal_repeats([_norm(W[i].text) for i in atts[h["b"] if cut_side is side_a else h["a"]]])
            if r_kept > r_cut:
                att = atts[h["b"] if cut_side is side_a else h["a"]]
                kept_side = att[:att.index(kept_side[-1]) + 1]     # from the attempt start: takes the stumble too
                keep_ids = {W[i].id for i in kept_side}
                out[k] = {**c, "word_ids": [W[i].id for i in kept_side], "text": " ".join(W[i].text for i in kept_side),
                          "reason": f"{c['reason']} [swapped by code: the kept take repeated {r_kept} words, this one {r_cut}]"}
                logger.info(f"[EDIT-PASS] retakes: swapped an uncertain take choice ({r_kept} vs {r_cut} repeated words)")
                assert keep_ids
    return out


def run_passes(grid: WordGrid, kept: Optional[Set[str]] = None, passes: List[str] = ORDER,
               models: Optional[Dict[str, str]] = None) -> Tuple[Set[str], List[Dict], List[Dict]]:
    """All passes in order, each on the previous edit. Returns (kept ids, cuts incl. suggestions,
    per-pass stats). Only cuts with applied=True change `kept`."""
    kept = set(kept) if kept is not None else {w.id for w in grid.words}
    cuts, stats = [], []
    for name in passes:
        acc, st = run_pass(name, grid, kept, (models or {}).get(name))
        for c in acc:
            if c["applied"]:
                kept -= set(c["word_ids"])
        cuts += acc
        stats.append({**st, "kept_after": len(kept)})
    return kept, cuts, stats
