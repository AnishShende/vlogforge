"""Story plans (roadmap Phase 9, Archdoc Stages 8-9, small version): which moments to keep and in what
order, for a target duration.

  1. candidates   story_order is built in code from the moment table's structure (story_order()):
                  intro, then the body sections, then outro. Section order, in priority: the order the
                  creator states (a moment that announces parts plays right before the first of them
                  that has footage), then the moment model's section_order (when things happened, then
                  topic logic). One Sonnet call adds two variants of it: cold_open and tight.
  2. repair       code enforces the hard rules: known moments only, no duplicates; every moment after
                  the moments it depends on (missing ones inserted before it); duration within
                  DURATION_TOLERANCE of the target by dropping the least important moment nothing else
                  needs, or adding back the most important one. When the whole edit already fits the
                  target, every moment is kept: only the ORDER can change.
  3. score        deterministic: importance kept, fit to the target, story shape (intro first, outro
                  last, stated orders kept; without roles: an opening function first, a closing one
                  last), exact durations from the compiler
The best plan becomes the edit; all candidates are stored for the review UI.
"""

import hashlib
import json
import logging
from typing import Dict, List, Optional, Tuple

import anthropic

from app.config import settings
from app.models import EditPlan, EditSegment, WordGrid
from app.tasks.compiler import FPS, compile_plan
from app.utils import artifacts

logger = logging.getLogger("VlogForge.StoryPlan")

PROMPT_VERSION = "2"
MODEL = "claude-sonnet-5-5"
VARIANTS = ["cold_open", "tight"]                      # the model's plans; story_order is built in code
DURATION_TOLERANCE = 0.15
OPENING = {"hook", "orientation", "goal"}
CLOSING = {"conclusion", "reflection", "payoff"}
W_IMPORTANCE, W_FIT, W_SHAPE = 0.5, 0.3, 0.2

SYSTEM = """You plan variants of the final cut of a video in which one creator talks to the camera. You get
a table of its MOMENTS (story beats of the already-cleaned edit): id, source clip and position in it,
role (intro / body / outro), section, story function, importance (0-1), the earlier moments it needs,
duration and summary; and the STORY ORDER: the intro, then the parts in the order the creator stated
or in which they happened, then the outro.

Make two variants of the story order, each an ordered list of moment ids:
- cold_open: first the strongest moment (a striking claim, payoff or reveal), then the story order
  without it;
- tight: only the main thread of the story order, for viewers in a hurry.
Rules for both: apart from what the variant changes, keep the story order; a moment must come after
every moment it needs; if the full edit is longer than the target duration, leave out the least
important moments to get close to the target; if it already fits, keep every moment. Think about the
order first, then give the plans."""


def _schema() -> Dict:
    plan = {"type": "object", "additionalProperties": False, "required": ["strategy", "reasoning", "order"],
            "properties": {"strategy": {"type": "string", "enum": VARIANTS}, "reasoning": {"type": "string"},
                           "order": {"type": "array", "items": {"type": "string"}}}}
    return {"type": "object", "additionalProperties": False, "required": ["plans"],
            "properties": {"plans": {"type": "array", "items": plan}}}


_client: Optional[anthropic.Anthropic] = None


def _call(user: str) -> Tuple[List[Dict], Dict]:
    key = hashlib.sha256(json.dumps([PROMPT_VERSION, MODEL, SYSTEM, user]).encode()).hexdigest()
    hit = artifacts.get("storyplan_call", key)
    if hit is not None:
        return hit["plans"], {**hit["usage"], "cached": True}
    global _client
    _client = _client or anthropic.Anthropic(api_key=settings.claude_api_key)
    r = _client.beta.messages.create(model=MODEL, max_tokens=16000, system=SYSTEM,
                                     messages=[{"role": "user", "content": user}],
                                     output_config={"effort": "medium", "format": {"type": "json_schema", "schema": _schema()}},
                                     betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    if r.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"story plan call stopped: {r.stop_reason}")
    plans = json.loads(next(b.text for b in r.content if b.type == "text"))["plans"]
    usage = {"model": r.model, "input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}
    artifacts.put("storyplan_call", key, {"plans": plans, "usage": usage})
    return plans, {**usage, "cached": False}


def moment_segments(grid: WordGrid, moment: Dict) -> List[EditSegment]:
    """A moment's words as plan segments: runs of grid-consecutive kept words."""
    pos = {w.id: i for i, w in enumerate(grid.words)}
    idx = [pos[i] for i in moment["word_ids"]]
    segs: List[EditSegment] = []
    for k, i in enumerate(idx):
        if k and i == idx[k - 1] + 1:
            segs[-1].word_end = grid.words[i].id
        else:
            segs.append(EditSegment(word_start=grid.words[i].id, word_end=grid.words[i].id, reason=moment["id"]))
    return segs


def edit_plan(grid: WordGrid, moments: List[Dict], order: List[str]) -> EditPlan:
    by_id = {m["id"]: m for m in moments}
    return EditPlan(segments=[s for mid in order for s in moment_segments(grid, by_id[mid])],
                    grid_fingerprint=grid.fingerprint())


def duration(grid: WordGrid, moments: List[Dict], order: List[str], envs, pause_targets) -> float:
    segs = compile_plan(edit_plan(grid, moments, order), grid, envs, pause_targets)
    return round(sum(round((s.src_out - s.src_in) * FPS) for s in segs) / FPS, 3)


def repair(order: List[str], moments: List[Dict], durs: Dict[str, float], target: Optional[float],
           full: float, ref: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    """Enforce the hard rules on a candidate order. Moments added back go to their place in `ref`
    (the story order; default: listing order). Returns (order, notes)."""
    by_id, notes = {m["id"]: m for m in moments}, []
    rank = {mid: k for k, mid in enumerate(ref or [m["id"] for m in moments])}
    seen, out = set(), []
    for mid in order:                                   # known, once
        if mid in by_id and mid not in seen:
            seen.add(mid)
            out.append(mid)
    if len(out) != len(order):
        notes.append(f"dropped {len(order) - len(out)} unknown/duplicate id(s)")

    def with_deps(seq: List[str]) -> List[str]:          # every moment after what it needs
        res: List[str] = []

        def place(mid: str, stack=()):
            if mid in res or mid in stack:
                return
            for d in sorted(by_id[mid]["depends_on"], key=rank.get):
                place(d, stack + (mid,))
            res.append(mid)
        for mid in seq:
            place(mid)
        return res

    fixed = with_deps(out)
    if fixed != out:
        notes.append("moved/added moments so each comes after what it needs")
    out = fixed
    if target is None or full <= target * (1 + DURATION_TOLERANCE):
        missing = [m["id"] for m in moments if m["id"] not in out]   # it all fits: keep every moment
        if missing:
            notes.append(f"added back {len(missing)} moment(s): the full edit fits the target")
            for mid in missing:                          # at its place in the reference order
                prev = [m for m in out if rank[m] < rank[mid]]
                k = out.index(prev[-1]) + 1 if prev else 0
                out = with_deps(out[:k] + [mid] + out[k:])
        return out, notes
    total = lambda seq: sum(durs[m] for m in seq)
    while total(out) > target * (1 + DURATION_TOLERANCE):  # too long: drop least important leaf
        needed = {d for m in out for d in by_id[m]["depends_on"]}
        leaves = [m for m in out if m not in needed]
        if len(leaves) <= 1:
            break
        drop = min(leaves, key=lambda m: (by_id[m]["importance"], -durs[m]))
        out.remove(drop)
        notes.append(f"dropped {drop} (importance {by_id[drop]['importance']}) for duration")
    while total(out) < target * (1 - DURATION_TOLERANCE):  # too short: add back most important
        cand = [m for m in moments if m["id"] not in out]
        if not cand:
            break
        add = max(cand, key=lambda m: m["importance"])["id"]
        prev = [m for m in out if rank[m] < rank[add]]       # at its place in the reference order
        k = out.index(prev[-1]) + 1 if prev else 0
        out = with_deps(out[:k] + [add] + out[k:])
        notes.append(f"added {add} back for duration")
    return out, notes


def story_order(moments: List[Dict], section_order: List[str]) -> Dict:
    """The story order, built in code: intro, then the body by section, then outro (listing order
    within each). Sections: the creator's stated order (moments that announce parts) wins over the
    model's section_order; sections it leaves out follow in listing order. A moment that announces
    parts plays right before the first of them that has footage; one whose parts have no footage at
    all (a teaser for another video) stays in its own section, or ends the body.
    Returns {order, sections, notes, warnings}."""
    notes, warnings = [], []
    body = [m for m in moments if m.get("role", "body") == "body"]
    footage = list(dict.fromkeys(m.get("section", "") for m in body if not m.get("announces")))
    secs = [s for s in section_order if s in footage]
    secs += [s for s in footage if s not in secs]
    stated: Dict[str, List[str]] = {}                   # lead-in id -> its parts with footage, stated order
    for m in body:
        if not m.get("announces"):
            continue
        missing = [s for s in m["announces"] if s not in footage]
        if missing:
            warnings.append(f"{m['id']} announces {', '.join(missing)}, but there is no footage of it")
        parts = [s for s in m["announces"] if s in footage]
        if parts:                                       # the stated order wins: same slots, stated order
            for k, s in zip(sorted(secs.index(s) for s in parts), parts):
                secs[k] = s
            stated[m["id"]] = parts
    for mid, parts in stated.items():
        if [s for s in secs if s in parts] != parts:
            warnings.append(f"{mid}: stated order {' -> '.join(parts)} conflicts with a later statement; "
                            "followed the later one")
    before: Dict[str, List[str]] = {}
    for mid, parts in stated.items():
        before.setdefault(min(parts, key=secs.index), []).append(mid)
    order = [m["id"] for m in moments if m.get("role") == "intro"]
    for s in secs:
        order += before.get(s, []) + [m["id"] for m in body if m.get("section") == s and m["id"] not in stated]
    rest = [m["id"] for m in body if m["id"] not in order]
    if rest:
        notes.append(f"{', '.join(rest)}: no section with footage; placed at the end of the body")
    order += rest + [m["id"] for m in moments if m.get("role") == "outro"]
    pos = {mid: k for k, mid in enumerate(order)}
    for c in dict.fromkeys(m.get("clip") for m in body):   # inside one continuous clip: recording order
        seq = [m["id"] for m in body if m.get("clip") == c and m["id"] not in stated]
        if sorted(seq, key=pos.get) != seq:
            notes.append(f"clip {c}: moments reordered within one continuous clip")
    return {"order": order, "sections": secs, "notes": notes, "warnings": warnings}


def structure(order: List[str], moments: List[Dict]) -> Optional[float]:
    """Share of the structure rules `order` keeps (intro first, outro last, every stated order of the
    parts present, its announcement before them); None when the moments carry no structure."""
    by_id, pos = {m["id"]: m for m in moments}, {mid: k for k, mid in enumerate(order)}
    checks = []
    for role, first in (("intro", True), ("outro", False)):
        mine = [pos[m] for m in order if by_id[m].get("role") == role]
        other = [pos[m] for m in order if by_id[m].get("role") != role]
        if mine and other:
            checks.append(max(mine) < min(other) if first else min(mine) > max(other))
    start: Dict[str, int] = {}
    for m in order:
        if not by_id[m].get("announces") and by_id[m].get("role", "body") == "body":
            start.setdefault(by_id[m].get("section", ""), pos[m])
    for m in order:
        parts = [s for s in by_id[m].get("announces") or [] if s in start]
        if parts:
            firsts = [start[s] for s in parts]
            checks.append(pos[m] < firsts[0] and firsts == sorted(firsts))
    return sum(checks) / len(checks) if checks else None


def score(order: List[str], moments: List[Dict], dur: float, target: Optional[float]) -> Dict:
    by_id = {m["id"]: m for m in moments}
    imp = sum(by_id[m]["importance"] for m in order) / max(1e-9, sum(m["importance"] for m in moments))
    fit = 1.0 if target is None else max(0.0, 1 - abs(dur - target) / target)
    shape = structure(order, moments) if order else 0
    if shape is None:
        shape = 0.5 * (by_id[order[0]]["function"] in OPENING) + 0.5 * (by_id[order[-1]]["function"] in CLOSING)
    shape = round(shape, 3)
    return {"importance_kept": round(imp, 3), "fit": round(fit, 3), "shape": shape,
            "score": round(W_IMPORTANCE * imp + W_FIT * fit + W_SHAPE * shape, 3)}


def plan_story(grid: WordGrid, moments_result: Dict, envs, pause_targets, target: Optional[float]) -> Dict:
    """Candidates -> repair -> score. Returns {"best": {...}, "plans": [...], "full_sec", "target_sec"}."""
    moments = moments_result["moments"]
    ids = [m["id"] for m in moments]
    durs = {m["id"]: duration(grid, moments, [m["id"]], envs, pause_targets) for m in moments}
    full = duration(grid, moments, ids, envs, pause_targets)
    base = story_order(moments, moments_result.get("section_order", []))
    for w in base["warnings"]:
        logger.warning(f"[STORYPLAN] {w}")
    base_reason = (f"Intro, then {' -> '.join(base['sections']) or 'the body'}, then outro. Section order: "
                   f"{moments_result.get('section_order_reason') or 'listing order'}")
    table = "\n".join(
        f"{m['id']} | clip {m.get('clip', '?')} at {m['start']:.1f}s | {m.get('role', 'body')} | section "
        f"{m.get('section') or '-'} | {m['function']} | importance {m['importance']} | needs "
        f"{','.join(m['depends_on']) or '-'} | {durs[m['id']]:.1f}s | {m['summary']}" for m in moments)
    user = (f"Synopsis: {moments_result['synopsis']}\nCreator goal: {moments_result['creator_goal']}\n"
            f"Full edit: {full:.1f}s. Target duration: {f'{target:.0f}s' if target else 'none (keep everything)'}.\n\n"
            f"Moments (listing order):\n{table}\n\nStory order: {' '.join(base['order'])}")
    try:
        raw, usage = _call(user)
    except Exception as e:                                 # the variants are optional; story_order is not
        logger.warning(f"[STORYPLAN] variant plans failed, story order only: {e}")
        raw, usage = [], {"model": MODEL, "input_tokens": 0, "output_tokens": 0, "cached": False, "error": str(e)}
    plans = []
    for p in [{"strategy": "story_order", "reasoning": base_reason, "order": base["order"], "notes": base["notes"]}] + raw:
        order, notes = repair(p["order"], moments, durs, target, full, base["order"])
        if not order:
            continue
        d = duration(grid, moments, order, envs, pause_targets)
        plans.append({"strategy": p["strategy"], "reasoning": p["reasoning"], "order": order,
                      "repairs": p.get("notes", []) + notes, "duration_sec": d, **score(order, moments, d, target)})
    if not plans:                                          # the model gave nothing usable: raw order
        d = duration(grid, moments, ids, envs, pause_targets)
        plans.append({"strategy": "raw_order", "reasoning": "fallback: no usable candidate", "order": ids,
                      "repairs": [], "duration_sec": d, **score(ids, moments, d, target)})
        logger.warning("[STORYPLAN] no usable candidate plan; using the raw order")
    plans.sort(key=lambda p: -p["score"])
    for p in plans:
        logger.info(f"[STORYPLAN] {p['strategy']}: {' '.join(p['order'])} ({len(p['order'])}/{len(moments)} moments), "
                    f"{p['duration_sec']:.1f}s (target {target}), score {p['score']} shape {p['shape']} {p['repairs'] or ''}")
    return {"best": plans[0], "plans": plans, "full_sec": full, "target_sec": target, "warnings": base["warnings"],
            "stats": {**usage, "calls": 1}, "prompt_version": PROMPT_VERSION}
