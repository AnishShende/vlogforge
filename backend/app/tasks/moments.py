"""Moment index and story graph (roadmap Phase 8, Archdoc Stage 7, small version).

Input: a job's CURRENT edit (kept words, after cleanup / edit passes). Output: a synopsis and a moment
table: one row per story beat, anchored to a range of kept words, with a story function, importance
and the earlier moments it depends on. Data for Phase 9 (duration-constrained plans); nothing here
changes the render.

Two model calls (short talking clips, so global and local passes are one call):
  1. build:  synopsis, creator goal, entities, moments (start word, function, importance, depends_on)
  2. refine: check the table against the transcript: dependencies (setup -> payoff, references such
             as "this"/"it"), functions, importance; return the corrected table
The model only names START words; code derives the ranges, so moments always tile the edit in order
with no gaps or overlaps. Code also drops dependencies that point forward or to unknown moments.
Every request is cached by its content (replays are free and identical).
"""

import hashlib
import json
import logging
from typing import Dict, List, Optional, Set, Tuple

import anthropic

from app.config import settings
from app.models import WordGrid
from app.tasks.edit_passes import attempts_of
from app.utils import artifacts

logger = logging.getLogger("VlogForge.Moments")

PROMPT_VERSION = "2"
MODEL = "claude-sonnet-5-5"
FUNCTIONS = ["hook", "orientation", "goal", "activity", "explanation", "progress", "obstacle", "discovery",
             "evidence", "reaction", "transition", "atmosphere", "payoff", "reflection", "conclusion"]

SYSTEM = """You analyse the story of a video in which one creator talks to the camera. You get the
transcript of the FINISHED edit (retakes and false starts already removed), as numbered lines; every
word carries a global index like [57]word. Speech recognition errors are common; the speech may be in
any language or a mix (for example Hindi and English).

Split the transcript into MOMENTS: consecutive stretches that each do one job in the story (one beat:
a point, a step, a reaction). A moment is usually one to four sentences. For each moment give:
- start_word: the global index of its first word (the first moment starts at the first word; every
  later moment starts after the previous one; together they cover every word in order),
- function: its story function, one of: hook, orientation, goal, activity, explanation, progress,
  obstacle, discovery, evidence, reaction, transition, atmosphere, payoff, reflection, conclusion,
- importance: 0 to 1, how much the video loses if this moment is cut (1 = the video breaks without
  it, 0.2 = nice to have),
- depends_on: numbers of EARLIER moments this one needs (moments are numbered from 1 in order: the
  first moment is 1) to make sense when watched (a payoff needs its
  setup, an answer needs its question, "this"/"that"/"it" needs what it refers to). Empty if it
  stands alone.
- summary: one short line, what the moment says or does.
Also give a synopsis of the whole video (two sentences), the creator's goal, and the main entities."""

REFINE = """Below is a moment table someone made for this transcript (moments numbered from 1). Check it carefully and return the
corrected full table (same format):
- dependencies: every moment that only makes sense after an earlier one must list it (setup -> payoff,
  question -> answer, a reference such as "this"/"it"/"that" -> what it refers to); remove listed
  dependencies that are not real;
- functions and importance must fit the creator's goal; the hook and the payoff of the main thread
  are usually the most important;
- change moment boundaries only when a moment clearly does two unrelated jobs or splits one thought.
Keep everything that is already right."""


def _schema() -> Dict:
    moment = {"type": "object", "additionalProperties": False,
              "required": ["start_word", "function", "importance", "depends_on", "summary"],
              "properties": {"start_word": {"type": "integer"}, "function": {"type": "string", "enum": FUNCTIONS},
                             "importance": {"type": "number"}, "depends_on": {"type": "array", "items": {"type": "integer"}},
                             "summary": {"type": "string"}}}
    return {"type": "object", "additionalProperties": False,
            "required": ["synopsis", "creator_goal", "entities", "moments"],
            "properties": {"synopsis": {"type": "string"}, "creator_goal": {"type": "string"},
                           "entities": {"type": "array", "items": {"type": "string"}},
                           "moments": {"type": "array", "items": moment}}}


_client: Optional[anthropic.Anthropic] = None


def _call(system: str, user: str) -> Tuple[Dict, Dict]:
    key = hashlib.sha256(json.dumps([PROMPT_VERSION, MODEL, system, user]).encode()).hexdigest()
    hit = artifacts.get("moments_call", key)
    if hit is not None:
        return hit["out"], {**hit["usage"], "cached": True}
    global _client
    _client = _client or anthropic.Anthropic(api_key=settings.claude_api_key)
    r = _client.beta.messages.create(model=MODEL, max_tokens=16000, system=system,
                                     messages=[{"role": "user", "content": user}],
                                     output_config={"effort": "medium", "format": {"type": "json_schema", "schema": _schema()}},
                                     betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    if r.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"moments call stopped: {r.stop_reason}")
    out = json.loads(next(b.text for b in r.content if b.type == "text"))
    usage = {"model": r.model, "input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}
    artifacts.put("moments_call", key, {"out": out, "usage": usage})
    return out, {**usage, "cached": False}


def render(grid: WordGrid, order: List[int]) -> str:
    """Kept words as lines (pause / cut breaks), each word with its GLOBAL index in `order`."""
    W, gi = grid.words, {i: n for n, i in enumerate(order)}
    lines = []
    for k, a in enumerate(attempts_of(grid, {W[i].id for i in order})):
        t = W[a[0]].start
        lines.append(f"({int(t // 60)}:{t % 60:04.1f}) " + " ".join(f"[{gi[i]}]{W[i].text}" for i in a))
    return "\n".join(lines)


def _validate(raw: Dict, n_words: int) -> Tuple[Optional[List[Dict]], List[str]]:
    """Moments as [{from, to, ...}] over global indices, or None when the starts are unusable."""
    notes = []
    ms = raw.get("moments") or []
    starts = [m["start_word"] for m in ms]
    if not ms or starts[0] != 0 or any(b <= a for a, b in zip(starts, starts[1:])) or starts[-1] >= n_words:
        return None, [f"unusable moment starts: {starts[:12]}"]
    out = []
    for k, m in enumerate(ms):
        deps = sorted({d for d in m["depends_on"] if 1 <= d <= k})            # earlier moments only
        if len(deps) != len(set(m["depends_on"])):
            notes.append(f"moment {k + 1}: dropped dependencies {sorted(set(m['depends_on']) - set(deps))}")
        out.append({"from": m["start_word"], "to": (starts[k + 1] - 1) if k + 1 < len(ms) else n_words - 1,
                    "function": m["function"], "importance": round(min(1.0, max(0.0, float(m["importance"]))), 2),
                    "depends_on": deps, "summary": m["summary"]})
    return out, notes


def build_moments(grid: WordGrid, kept: Set[str]) -> Dict:
    """Synopsis + moment table for the edit `kept`. Raises if the model's table is unusable twice."""
    W = grid.words
    order = [i for i, w in enumerate(W) if w.id in kept]
    if not order:
        raise ValueError("empty edit: no kept words")
    transcript = render(grid, order)
    out1, u1 = _call(SYSTEM, transcript)
    m1, notes1 = _validate(out1, len(order))
    if m1 is None:
        raise RuntimeError(f"moment table unusable: {notes1}")
    table = "\n".join(f"{k + 1}. start_word {m['from']} | {m['function']} | importance {m['importance']} | "
                      f"depends_on {m['depends_on']} | {m['summary']}" for k, m in enumerate(m1))
    out2, u2 = _call(SYSTEM, f"{transcript}\n\n{REFINE}\n\nSynopsis: {out1['synopsis']}\nCreator goal: "
                             f"{out1['creator_goal']}\n\nMoment table:\n{table}")
    m2, notes2 = _validate(out2, len(order))
    final, used = (m2, "refined") if m2 is not None else (m1, "first pass (refinement unusable)")
    src = out2 if m2 is not None else out1
    for note in notes1 + notes2:
        logger.warning(f"[MOMENTS] {note}")
    moments = []
    for k, m in enumerate(final):
        ws = [W[i] for i in order[m["from"]:m["to"] + 1]]
        moments.append({"id": f"m{k + 1}", "word_start": ws[0].id, "word_end": ws[-1].id,
                        "word_ids": [w.id for w in ws], "source_file": ws[0].source_file,
                        "start": ws[0].start, "end": ws[-1].end, "text": " ".join(w.text for w in ws),
                        "function": m["function"], "importance": m["importance"],
                        "depends_on": [f"m{d}" for d in m["depends_on"]], "summary": m["summary"]})
    stats = {"model": MODEL, "calls": 2, "cached": int(u1["cached"]) + int(u2["cached"]),
             "input_tokens": u1["input_tokens"] + u2["input_tokens"], "output_tokens": u1["output_tokens"] + u2["output_tokens"],
             "moments_first": len(m1), "moments_final": len(final), "table": used}
    logger.info(f"[MOMENTS] {len(order)} words -> {len(moments)} moments ({used}); {stats}")
    return {"synopsis": src["synopsis"], "creator_goal": src["creator_goal"], "entities": src["entities"],
            "moments": moments, "stats": stats, "prompt_version": PROMPT_VERSION}


def moments_for_job(job_id: str) -> Dict:
    """Build (or rebuild) the moment table for a job's stored edit and store it as artifact 'moments'."""
    grid_d, plan = artifacts.load_job(job_id, "grid"), artifacts.load_job(job_id, "plan")
    if grid_d is None or plan is None:
        raise LookupError(f"job {job_id}: no stored word grid / plan")
    grid = WordGrid(**grid_d)
    index = {w.id: i for i, w in enumerate(grid.words)}
    kept = {grid.words[i].id for s in plan["segments"] for i in range(index[s["word_start"]], index[s["word_end"]] + 1)}
    result = build_moments(grid, kept)
    artifacts.save_job(job_id, "moments", result)
    return result
