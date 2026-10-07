"""Story plans (roadmap Phase 9, Archdoc Stages 8-9, small version): which moments to keep and in what
order, for a target duration.

  1. candidates   one Sonnet call: three plans over the moment table (story_order, cold_open, tight),
                  reasoning first, each an ordered list of moment ids
  2. repair       code enforces the hard rules: known moments only, no duplicates; every moment after
                  the moments it depends on (missing ones inserted before it); duration within
                  DURATION_TOLERANCE of the target by dropping the least important moment nothing else
                  needs, or adding back the most important one. When the whole edit already fits the
                  target, every moment is kept: only the ORDER can change.
  3. score        deterministic: importance kept, fit to the target, story shape (an opening function
                  first, a closing function last), exact durations from the compiler
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

PROMPT_VERSION = "1"
MODEL = "claude-sonnet-5-5"
STRATEGIES = ["story_order", "cold_open", "tight"]
DURATION_TOLERANCE = 0.15
OPENING = {"hook", "orientation", "goal"}
CLOSING = {"conclusion", "reflection", "payoff"}
W_IMPORTANCE, W_FIT, W_SHAPE = 0.5, 0.3, 0.2

SYSTEM = """You plan the final cut of a video in which one creator talks to the camera. You get a table of
its MOMENTS (story beats of the already-cleaned edit): id, source clip and position in the recording,
story function, importance (0-1), the earlier moments it needs, duration and summary. The moments are
listed in the order the raw footage happens to be in, which may not be the right story order (for
example a sign-off recorded or filed first).

Make three plans, each an ordered list of moment ids:
- story_order: the most natural order for a viewer: open with a welcome, hook or goal, then the body,
  and end with the conclusion or sign-off;
- cold_open: open with the strongest moment (a striking claim, payoff or reveal), then the story;
- tight: only the main thread, for viewers in a hurry.
Rules for every plan: a moment must come after every moment it needs; if the full edit is longer than
the target duration, leave out the least important moments to get close to the target; if it already
fits, keep every moment and only choose the order. Think about the order first, then give the plans."""


def _schema() -> Dict:
    plan = {"type": "object", "additionalProperties": False, "required": ["strategy", "reasoning", "order"],
            "properties": {"strategy": {"type": "string", "enum": STRATEGIES}, "reasoning": {"type": "string"},
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
           full: float) -> Tuple[List[str], List[str]]:
    """Enforce the hard rules on a candidate order. Returns (order, notes)."""
    by_id, notes = {m["id"]: m for m in moments}, []
    rank = {m["id"]: k for k, m in enumerate(moments)}
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
            for mid in missing:                          # at its recording position
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
        prev = [m for m in out if rank[m] < rank[add]]       # at its recording position
        k = out.index(prev[-1]) + 1 if prev else 0
        out = with_deps(out[:k] + [add] + out[k:])
        notes.append(f"added {add} back for duration")
    return out, notes


def score(order: List[str], moments: List[Dict], dur: float, target: Optional[float]) -> Dict:
    by_id = {m["id"]: m for m in moments}
    imp = sum(by_id[m]["importance"] for m in order) / max(1e-9, sum(m["importance"] for m in moments))
    fit = 1.0 if target is None else max(0.0, 1 - abs(dur - target) / target)
    shape = 0.5 * (by_id[order[0]]["function"] in OPENING) + 0.5 * (by_id[order[-1]]["function"] in CLOSING) if order else 0
    return {"importance_kept": round(imp, 3), "fit": round(fit, 3), "shape": shape,
            "score": round(W_IMPORTANCE * imp + W_FIT * fit + W_SHAPE * shape, 3)}


def plan_story(grid: WordGrid, moments_result: Dict, envs, pause_targets, target: Optional[float]) -> Dict:
    """Candidates -> repair -> score. Returns {"best": {...}, "plans": [...], "full_sec", "target_sec"}."""
    moments = moments_result["moments"]
    ids = [m["id"] for m in moments]
    durs = {m["id"]: duration(grid, moments, [m["id"]], envs, pause_targets) for m in moments}
    full = duration(grid, moments, ids, envs, pause_targets)
    files = sorted({m["source_file"] for m in moments})
    table = "\n".join(
        f"{m['id']} | clip {files.index(m['source_file']) + 1} at {m['start']:.1f}s | {m['function']} | importance "
        f"{m['importance']} | needs {','.join(m['depends_on']) or '-'} | {durs[m['id']]:.1f}s | {m['summary']}" for m in moments)
    user = (f"Synopsis: {moments_result['synopsis']}\nCreator goal: {moments_result['creator_goal']}\n"
            f"Full edit: {full:.1f}s. Target duration: {f'{target:.0f}s' if target else 'none (keep everything)'}.\n\n"
            f"Moments (raw order):\n{table}")
    raw, usage = _call(user)
    plans = []
    for p in raw:
        order, notes = repair(p["order"], moments, durs, target, full)
        if not order:
            continue
        d = duration(grid, moments, order, envs, pause_targets)
        plans.append({"strategy": p["strategy"], "reasoning": p["reasoning"], "order": order, "repairs": notes,
                      "duration_sec": d, **score(order, moments, d, target)})
    if not plans:                                          # the model gave nothing usable: raw order
        d = duration(grid, moments, ids, envs, pause_targets)
        plans.append({"strategy": "raw_order", "reasoning": "fallback: no usable candidate", "order": ids,
                      "repairs": [], "duration_sec": d, **score(ids, moments, d, target)})
        logger.warning("[STORYPLAN] no usable candidate plan; using the raw order")
    plans.sort(key=lambda p: -p["score"])
    for p in plans:
        logger.info(f"[STORYPLAN] {p['strategy']}: {len(p['order'])}/{len(moments)} moments, {p['duration_sec']:.1f}s "
                    f"(target {target}), score {p['score']} {p['repairs'] or ''}")
    return {"best": plans[0], "plans": plans, "full_sec": full, "target_sec": target,
            "stats": {**usage, "calls": 1}, "prompt_version": PROMPT_VERSION}
