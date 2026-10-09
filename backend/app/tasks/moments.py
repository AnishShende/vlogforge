"""Moment index and story graph (roadmap Phase 8, Archdoc Stage 7, small version).

Input: a job's CURRENT edit (kept words, after cleanup / edit passes). Output: a synopsis and a moment
table: one row per story beat, anchored to a range of kept words, with a story function, importance,
the earlier moments it depends on, and its place in the structure (role intro / body / outro, the
section it belongs to, the sections it announces), plus the order the sections should play in. Data
for Phase 9 (story order and duration-constrained plans); nothing here changes the render.

Clips are listed by recording time when every clip has a real one (container metadata), otherwise in
upload order; never by filename. Each clip's header tells the model its recording time or "unknown".

Two model calls (short talking clips, so global and local passes are one call):
  1. build:  synopsis, creator goal, entities, section order, moments (start word, function,
             importance, depends_on, role, section, announces)
  2. refine: check the table against the transcript: dependencies (setup -> payoff, references such
             as "this"/"it"), functions, importance, structure; return the corrected table
The model only names START words; code derives the ranges, so moments always tile the edit in order
with no gaps or overlaps. Code also drops dependencies that point forward or to unknown moments.
Every request is cached by its content (replays are free and identical).
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import anthropic

from app.config import settings
from app.models import WordGrid
from app.tasks.edit_passes import attempts_of
from app.utils import artifacts

logger = logging.getLogger("VlogForge.Moments")

PROMPT_VERSION = "3"
MODEL = "claude-sonnet-5-5"
FUNCTIONS = ["hook", "orientation", "goal", "activity", "explanation", "progress", "obstacle", "discovery",
             "evidence", "reaction", "transition", "atmosphere", "payoff", "reflection", "conclusion"]
ROLES = ["intro", "body", "outro"]

SYSTEM = """You analyse the story of a video made from one or more clips in which one creator talks to the
camera. You get the transcript of the FINISHED edit (retakes and false starts already removed), clip by
clip, as numbered lines; every word carries a global index like [57]word. Each clip header gives the
clip's recording time, or "unknown" when the camera stored none. The clips are listed by recording time
when every clip has one, otherwise in upload order; either way this is NOT necessarily the story order
(creators often film a welcome, a sign-off or a link between parts at another time). Speech recognition
errors are common; the speech may be in any language or a mix (for example Hindi and English).

Split the transcript into MOMENTS: consecutive stretches that each do one job in the story (one beat:
a point, a step, a reaction). A moment is usually one to four sentences. List the moments in
TRANSCRIPT order, never in story order (the story order is expressed by role, section, announces and
section_order below; code reorders the moments from those). For each moment give:
- start_word: the global index of its first word (the first moment starts at the first word; every
  later moment starts after the previous one; together they cover every word once, in order),
- function: its story function, one of: hook, orientation, goal, activity, explanation, progress,
  obstacle, discovery, evidence, reaction, transition, atmosphere, payoff, reflection, conclusion,
- importance: 0 to 1, how much the video loses if this moment is cut (1 = the video breaks without
  it, 0.2 = nice to have),
- depends_on: numbers of EARLIER moments this one needs (moments are numbered from 1 in order: the
  first moment is 1) to make sense when watched (a payoff needs its
  setup, an answer needs its question, "this"/"that"/"it" needs what it refers to). Empty if it
  stands alone.
- role: "intro" only for the opening of the whole video (a welcome, what this video is); "outro" only
  for its closing (a sign-off, "see you next time", a call to follow the channel); "body" for everything else,
  including a line that opens one part ("now let's head to the beach") or announces what comes next.
- section: for a body moment, a short name for the part of the video it belongs to (a topic, place or
  activity, e.g. "morning coffee", "beach walk"). Moments about the same thing share one name, also
  across clips, even when the creator words it differently (a market visit the creator later calls
  "the bazaar trip" is one section). When the creator comes back to a topic after a different one within the same
  clip, the later stretch gets its own name. Empty for intro and outro.
- announces: when the moment states what the video will show and in what order ("first the market,
  then the beach"), those parts in the stated order, each with a short name (part) and
  first_moment: the number of the moment where that part's footage starts (it may come later in the
  transcript; footage the creator words differently counts), or 0 when the transcript has no footage
  of it. Creators usually film what they announce: before giving 0, look through the whole transcript,
  also clips filmed earlier, for that part under another name. Empty otherwise.
- summary: one short line, what the moment says or does.
Also give a synopsis of the whole video (two sentences), the creator's goal, the main entities, and
- section_order: every section name of the body moments, in the order the parts should play for a
  viewer. Decide it by, in this priority: (1) an order the creator states; (2) when things happened:
  time cues in the speech ("this morning", "after lunch") and the clips' recording times when known;
  a moment that talks about other footage (a welcome, a link, a sign-off) can be filmed at any time,
  so its recording time says nothing about its place; (3) topic logic (what a viewer needs first).
- section_order_reason: one sentence: which of these decided the order, and how."""

REFINE = """Below is a moment table someone made for this transcript (moments numbered from 1). Check it carefully and return the
corrected full table (same format):
- dependencies: every moment that only makes sense after an earlier one must list it (setup -> payoff,
  question -> answer, a reference such as "this"/"it"/"that" -> what it refers to); remove listed
  dependencies that are not real;
- functions and importance must fit the creator's goal; the hook and the payoff of the main thread
  are usually the most important;
- structure: intro and outro only for the opening and closing of the whole video; footage of one
  thing shares one section name; every announced part points at the moment where its footage
  starts (0 only when no footage in the whole transcript, under any name, matches it); the section order follows a stated order first, then when things happened, then topic logic;
- change moment boundaries only when a moment clearly does two unrelated jobs or splits one thought.
Keep everything that is already right."""


def _schema() -> Dict:
    part = {"type": "object", "additionalProperties": False, "required": ["part", "first_moment"],
            "properties": {"part": {"type": "string"}, "first_moment": {"type": "integer"}}}
    moment = {"type": "object", "additionalProperties": False,
              "required": ["start_word", "function", "importance", "depends_on", "role", "section", "announces", "summary"],
              "properties": {"start_word": {"type": "integer"}, "function": {"type": "string", "enum": FUNCTIONS},
                             "importance": {"type": "number"}, "depends_on": {"type": "array", "items": {"type": "integer"}},
                             "role": {"type": "string", "enum": ROLES}, "section": {"type": "string"},
                             "announces": {"type": "array", "items": part}, "summary": {"type": "string"}}}
    return {"type": "object", "additionalProperties": False,
            "required": ["synopsis", "creator_goal", "entities", "section_order", "section_order_reason", "moments"],
            "properties": {"synopsis": {"type": "string"}, "creator_goal": {"type": "string"},
                           "entities": {"type": "array", "items": {"type": "string"}},
                           "section_order": {"type": "array", "items": {"type": "string"}},
                           "section_order_reason": {"type": "string"},
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


def clip_order(files: List[Dict]) -> List[Dict]:
    """files: [{filename, path}] in upload order. Returns [{filename, recorded}] in listing order: by
    recording time when every clip has a real one, else upload order (never by filename)."""
    from app.utils.ffmpeg import recording_time
    clips = [{"filename": f["filename"], "recorded": recording_time(f["path"])} for f in files]
    if clips and all(c["recorded"] for c in clips):
        clips.sort(key=lambda c: datetime.fromisoformat(c["recorded"]))
    elif len(clips) > 1:
        logger.info(f"[MOMENTS] recording time unknown for {sum(not c['recorded'] for c in clips)}/{len(clips)} "
                    "clip(s): clips listed in upload order")
    return clips


def _clip_header(n: int, total: int, recorded: Optional[str]) -> str:
    when = datetime.fromisoformat(recorded).strftime("%Y-%m-%d %H:%M:%S (UTC%z)") if recorded else "unknown"
    return f"=== Clip {n} of {total}, recorded: {when}"


def render(grid: WordGrid, order: List[int], clips: Optional[List[Dict]] = None) -> str:
    """Kept words as lines (pause / cut breaks), each word with its GLOBAL index in `order`; a header
    line at every clip change (clip number in listing order, recording time) when `clips` is given."""
    W, gi = grid.words, {i: n for n, i in enumerate(order)}
    rec = {c["filename"]: (n + 1, c["recorded"]) for n, c in enumerate(clips or [])}
    lines, prev_file = [], None
    for a in sorted(attempts_of(grid, {W[i].id for i in order}), key=lambda a: gi[a[0]]):
        f = W[a[0]].source_file
        if clips and f != prev_file:
            n, recorded = rec[f]
            lines.append(_clip_header(n, len(clips), recorded))
        prev_file = f
        t = W[a[0]].start
        lines.append(f"({int(t // 60)}:{t % 60:04.1f}) " + " ".join(f"[{gi[i]}]{W[i].text}" for i in a))
    return "\n".join(lines)


def _norm_section(s: str) -> str:
    return " ".join(str(s).lower().split())


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
        role = m.get("role") if m.get("role") in ROLES else "body"
        section = _norm_section(m.get("section", "")) if role == "body" else ""
        if role == "body" and not section:
            section = f"part {k + 1}"
            notes.append(f"moment {k + 1}: body moment without a section; named '{section}'")
        out.append({"from": m["start_word"], "to": (starts[k + 1] - 1) if k + 1 < len(ms) else n_words - 1,
                    "function": m["function"], "importance": round(min(1.0, max(0.0, float(m["importance"]))), 2),
                    "depends_on": deps, "role": role, "section": section, "summary": m["summary"],
                    "refs": [(_norm_section(a.get("part", "")), int(a.get("first_moment") or 0))
                             for a in m.get("announces") or []]})
    for k, m in enumerate(out):                         # announced parts -> the section of their footage
        names = []
        for part, t in m["refs"]:
            if 1 <= t <= len(out) and t != k + 1 and out[t - 1]["section"]:
                names.append(out[t - 1]["section"])
            else:
                if t:
                    notes.append(f"moment {k + 1}: announced part '{part}' points at moment {t}, which has no "
                                 "section; treated as not filmed")
                names.append(part)
        m["announces"] = list(dict.fromkeys(n for n in names if n))
    return out, notes


def build_moments(grid: WordGrid, kept: Set[str], clips: Optional[List[Dict]] = None) -> Dict:
    """Synopsis + moment table for the edit `kept`. clips: clip_order() output; None lists the clips
    in grid order with unknown recording times. Raises if the model's table is unusable twice."""
    W = grid.words
    files = list(dict.fromkeys(w.source_file for w in W))
    clips = clips or [{"filename": f, "recorded": None} for f in files]
    rank = {c["filename"]: n for n, c in enumerate(clips)}
    if set(files) - set(rank):
        raise ValueError(f"clips missing from the clip order: {sorted(set(files) - set(rank))}")
    order = sorted((i for i, w in enumerate(W) if w.id in kept), key=lambda i: (rank[W[i].source_file], i))
    if not order:
        raise ValueError("empty edit: no kept words")
    transcript = render(grid, order, clips)
    out1, u1 = _call(SYSTEM, transcript)
    m1, notes1 = _validate(out1, len(order))
    if m1 is None:
        raise RuntimeError(f"moment table unusable: {notes1}")
    table = "\n".join(f"{k + 1}. start_word {m['from']} | {m['function']} | importance {m['importance']} | "
                      f"depends_on {m['depends_on']} | role {m['role']} | section {m['section'] or '-'} | "
                      f"announces {[f'{p} -> moment {t}' for p, t in m['refs']] or '-'} | {m['summary']}" for k, m in enumerate(m1))
    out2, u2 = _call(SYSTEM, f"{transcript}\n\n{REFINE}\n\nSynopsis: {out1['synopsis']}\nCreator goal: "
                             f"{out1['creator_goal']}\nSection order: {out1['section_order']} "
                             f"({out1['section_order_reason']})\n\nMoment table:\n{table}")
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
                        "clip": rank[ws[0].source_file] + 1,
                        "start": ws[0].start, "end": ws[-1].end, "text": " ".join(w.text for w in ws),
                        "function": m["function"], "importance": m["importance"],
                        "depends_on": [f"m{d}" for d in m["depends_on"]], "role": m["role"],
                        "section": m["section"], "announces": m["announces"], "summary": m["summary"]})
    stats = {"model": MODEL, "calls": 2, "cached": int(u1["cached"]) + int(u2["cached"]),
             "input_tokens": u1["input_tokens"] + u2["input_tokens"], "output_tokens": u1["output_tokens"] + u2["output_tokens"],
             "moments_first": len(m1), "moments_final": len(final), "table": used}
    section_order = list(dict.fromkeys(s for s in map(_norm_section, src["section_order"]) if s))
    logger.info(f"[MOMENTS] {len(order)} words -> {len(moments)} moments ({used}); sections {section_order} "
                f"({src['section_order_reason']}); {stats}")
    return {"synopsis": src["synopsis"], "creator_goal": src["creator_goal"], "entities": src["entities"],
            "section_order": section_order, "section_order_reason": src["section_order_reason"],
            "clips": clips, "moments": moments, "stats": stats, "prompt_version": PROMPT_VERSION}


def moments_for_job(job_id: str) -> Dict:
    """Build (or rebuild) the moment table for a job's stored edit and store it as artifact 'moments'."""
    grid_d, plan = artifacts.load_job(job_id, "grid"), artifacts.load_job(job_id, "plan")
    if grid_d is None or plan is None:
        raise LookupError(f"job {job_id}: no stored word grid / plan")
    grid = WordGrid(**grid_d)
    index = {w.id: i for i, w in enumerate(grid.words)}
    kept = {grid.words[i].id for s in plan["segments"] for i in range(index[s["word_start"]], index[s["word_end"]] + 1)}
    files = artifacts.load_job(job_id, "files")
    result = build_moments(grid, kept, clip_order(files) if files else None)
    artifacts.save_job(job_id, "moments", result)
    return result
