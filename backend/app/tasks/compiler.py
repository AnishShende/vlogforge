"""Deterministic compiler: EditPlan (word ranges) -> source cuts (Archdoc Stage 10, roadmap Phase 2).

S2 cut-point selection. For each run of kept words:
  1. Find where the activity past the word edge stops: speech (VAD) OR loud frames,
     followed at most TAIL_SEARCH_SEC. Aligner word ends are early (IMG_1614: speech runs
     100-180 ms past them) and Silero VAD misses real speech on noisy audio (chai review),
     so neither the word edge nor VAD alone protects a word's tail.
  2. The cut window lies in the pause between that activity and the neighbouring word's
     activity, keeping at most MAX_PAUSE_OUT_SEC / MAX_PAUSE_IN_SEC of it.
  3. The cut goes on the lowest-energy 10 ms frame in the window.
  4. Segment length is snapped to whole output frames (audio length == video length).
No pause (activity gap < TIGHT_SEC): cut at the quietest point between the word edges,
flagged `tight`. A cut never lands inside a kept or neighbouring word.

S3 pause normalisation (optional, pause_targets): dead air longer than PAUSE_MAX_SEC between kept
words (no words in the gap: silence OR noise) is shortened to the speaker's own median pause by
splitting the run there. Both cuts lie inside the longest real silence, or else inside the longest
stretch with no VAD speech (noise allowed; flagged in_noise). Sound VAD calls speech is never cut. Changed 2026-10-06
after the user's listening review: noisy pauses between sentences (chai kitchen) had been kept whole.
Values measured on the eval set and approved 2026-10-06 (docs/ARCHDOC_ROADMAP.md, Phase 2).
"""

import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.models import (CUT_FLAG_FRAME_OFF, CUT_FLAG_IN_NOISE, CUT_FLAG_LONG_TAIL, CUT_FLAG_TIGHT, CompiledSegment, CompiledTimeline,
                        CutPoint, EditPlan, WordGrid)
from app.utils.speech_activity import HOP_SEC, MIN_SILENCE_SEC, Envelope

logger = logging.getLogger("VlogForge.Compiler")

FPS = 30                    # output frame rate (render scales every source to 30 fps)
TAIL_SEARCH_SEC = 0.5       # follow activity past a word edge at most this far
MARGIN_SEC = 0.03           # minimum distance between a cut and any activity
MAX_PAUSE_OUT_SEC = 0.25    # pause kept after the last word of a segment
MAX_PAUSE_IN_SEC = 0.15     # pause kept before the first word of a segment
TIGHT_SEC = 0.06            # activity gap below this: no pause to cut in
PAUSE_MAX_SEC = 1.0        # S3: silences longer than this inside a kept run are shortened
PAUSE_NATURAL_SEC = (0.15, 1.0)   # S3: a speaker's natural pauses, for their median
PAUSE_TARGET_SEC = (0.3, 0.8)     # S3: clamp for the shortened pause
PAUSE_TARGET_DEFAULT = 0.5        # S3: when a file has too few natural pauses to measure
PAUSE_OUT_SHARE = 0.6             # S3: share of the shortened pause kept after the speech (rest before)
JOIN_FADE_SEC = 0.015       # S4: audio fade at both ends of every segment (cuts sit in pauses)
MAX_EDGE_FADE_SEC = 0.5     # S4: global audio fade-in/out, capped by the first/last pause


class PlanError(ValueError):
    pass


def resolve_plan(plan: EditPlan, grid: WordGrid) -> List[Tuple[List[int], List[str]]]:
    """Validate the plan and expand it to runs of grid word indices, in output order.
    Plan segments that continue each other on the grid merge into one run (no cut)."""
    errors = plan.validate_against(grid)
    if errors:
        raise PlanError("invalid edit plan:\n  " + "\n  ".join(errors))
    index = {w.id: i for i, w in enumerate(grid.words)}
    runs: List[Tuple[List[int], List[str]]] = []
    for seg in plan.segments:
        a, b = index[seg.word_start], index[seg.word_end]
        prev = runs[-1][0][-1] if runs else None
        if prev is not None and prev + 1 == a and grid.words[prev].source_file == grid.words[a].source_file:
            runs[-1][0].extend(range(a, b + 1))
            runs[-1][1].append(seg.reason)
        else:
            runs.append((list(range(a, b + 1)), [seg.reason]))
    return runs


def _active(env: Envelope) -> np.ndarray:
    return env.speech | env.loud if env.loud is not None else env.speech


def activity_edge(env: Envelope, t: float, step: int, search_sec: float = TAIL_SEARCH_SEC) -> Tuple[float, bool]:
    """Follow active frames from word edge t outward (step=+1 after a word end, -1 before a
    start), at most TAIL_SEARCH_SEC. Quiet gaps shorter than MIN_SILENCE_SEC (stop consonants,
    VAD jitter) do not end the activity. Returns (edge time, limit hit)."""
    act, lim = _active(env), int(round(search_sec / HOP_SEC))
    gap_max = max(1, int(round(MIN_SILENCE_SEC / HOP_SEC)))
    i = int(round(t / HOP_SEC)) if step > 0 else int(round(t / HOP_SEC)) - 1
    last, quiet = None, 0          # last active frame offset; current quiet run length
    for n in range(lim):
        j = i + step * n
        if not 0 <= j < len(act):
            break
        if act[j]:
            last, quiet = n, 0
        else:
            quiet += 1
            if quiet >= gap_max:
                break
    hit = last is not None and last >= lim - gap_max and 0 <= i + step * lim < len(act) and bool(act[i + step * lim])
    if last is None:
        return t, False
    return (max(t, (i + last + 1) * HOP_SEC), hit) if step > 0 else (min(t, (i - last) * HOP_SEC), hit)


def _level(env: Envelope, t: float) -> float:
    return float(env.db[env.t2i(t)])


def _quietest(env: Envelope, lo: float, hi: float, prefer_late: bool) -> float:
    """Lowest-energy frame time in [lo, hi]; ties go to the end nearest the kept word."""
    if hi <= lo:
        return lo
    ts = np.arange(lo, hi + 1e-9, HOP_SEC)
    act = _active(env)
    quiet = [t for t in ts if not act[env.t2i(t)]]
    ts = np.array(quiet) if quiet else ts        # prefer frames with no activity at all
    levels = np.array([_level(env, t) for t in ts])
    k = len(levels) - 1 - int(np.argmin(levels[::-1])) if prefer_late else int(np.argmin(levels))
    return float(ts[k])


def longest_silence(env: Envelope, a_end: float, b_start: float, speech_only: bool = False) -> Tuple[float, float]:
    """Longest run of frames with no activity (no VAD speech, not loud; with speech_only: no VAD
    speech, noise allowed) between two words, as (start, end). (a_end, a_end) when there is none."""
    act = env.speech if speech_only else _active(env)
    i0, i1 = int(np.ceil(a_end / HOP_SEC - 1e-9)), min(len(act), int(np.floor(b_start / HOP_SEC + 1e-9)))
    best, run_start = (i0, i0), None
    for i in range(i0, i1 + 1):
        if i < i1 and not act[i]:
            run_start = i if run_start is None else run_start
        elif run_start is not None:
            if i - run_start > best[1] - best[0]:
                best = (run_start, i)
            run_start = None
    return best[0] * HOP_SEC, best[1] * HOP_SEC


def silence_between(env: Envelope, a_end: float, b_start: float) -> float:
    s0, s1 = longest_silence(env, a_end, b_start)
    return s1 - s0


def speaker_pause_targets(grid: WordGrid, envs: Dict[str, Envelope]) -> Dict[str, float]:
    """Per source file (one speaker per file assumed): median of the natural pauses between
    consecutive words, clamped to PAUSE_TARGET_SEC."""
    out = {}
    for f in sorted({w.source_file for w in grid.words}):
        ws = [w for w in grid.words if w.source_file == f]
        sil = [silence_between(envs[f], a.end, b.start) for a, b in zip(ws, ws[1:])]
        nat = [x for x in sil if PAUSE_NATURAL_SEC[0] <= x <= PAUSE_NATURAL_SEC[1]]
        if len(nat) < 5:
            logger.warning(f"[COMPILE] {f}: only {len(nat)} natural pauses; pause target defaults to {PAUSE_TARGET_DEFAULT}s")
            out[f] = PAUSE_TARGET_DEFAULT
        else:
            out[f] = round(float(np.clip(np.median(nat), *PAUSE_TARGET_SEC)), 3)
    return out


def cut_out(env: Envelope, word_end: float, next_start: Optional[float], file_end: float,
            max_pause: float = MAX_PAUSE_OUT_SEC, search_sec: float = TAIL_SEARCH_SEC) -> CutPoint:
    """Cut after a kept word; next_start is the following (unkept) word, None at file end."""
    edge, long_tail = activity_edge(env, word_end, +1, search_sec)
    if next_start is None:
        bound, margin = file_end, 0.0
    else:
        bound, margin = min(next_start, activity_edge(env, next_start, -1, search_sec)[0]), MARGIN_SEC
    flags = [CUT_FLAG_LONG_TAIL] if long_tail else []
    if bound - edge < TIGHT_SEC and next_start is not None:
        lo, hi = word_end, max(word_end, next_start)
        flags.append(CUT_FLAG_TIGHT)
    else:
        lo = min(edge + MARGIN_SEC, bound)
        hi = max(lo, min(edge + max_pause, bound - margin))
    t = _quietest(env, lo, hi, prefer_late=False)
    return CutPoint(time=t, side="out", word_edge=word_end, activity_edge=edge, window=[lo, hi],
                    level_db=_level(env, t), flags=flags)


def cut_in(env: Envelope, word_start: float, prev_end: Optional[float], max_pause: float = MAX_PAUSE_IN_SEC,
           search_sec: float = TAIL_SEARCH_SEC) -> CutPoint:
    """Cut before a kept word; prev_end is the preceding (unkept) word, None at file start."""
    edge, long_tail = activity_edge(env, word_start, -1, search_sec)
    if prev_end is None:
        bound, margin = 0.0, 0.0
    else:
        bound, margin = max(prev_end, activity_edge(env, prev_end, +1, search_sec)[0]), MARGIN_SEC
    flags = [CUT_FLAG_LONG_TAIL] if long_tail else []
    if edge - bound < TIGHT_SEC and prev_end is not None:
        lo, hi = min(prev_end, word_start), word_start
        flags.append(CUT_FLAG_TIGHT)
    else:
        hi = max(edge - MARGIN_SEC, bound)
        lo = min(hi, max(edge - max_pause, bound + margin))
    t = _quietest(env, lo, hi, prefer_late=True)
    return CutPoint(time=t, side="in", word_edge=word_start, activity_edge=edge, window=[lo, hi],
                    level_db=_level(env, t), flags=flags)


def snap_to_frames(env: Envelope, c_in: CutPoint, c_out: CutPoint) -> None:
    """Make the segment a whole number of output frames by moving a cut inside its window
    (out first, then in). Otherwise flag frame_off on the out cut."""
    frame = 1.0 / FPS
    dur = c_out.time - c_in.time
    k = max(1, round(dur * FPS))
    act = _active(env)
    # a move may not put a cut that was on a quiet frame onto activity (Phase 3 validator caught this)
    ok = lambda c, t: (c.window[0] - 1e-9 <= t <= c.window[1] + 1e-9
                       and (not act[env.t2i(t)] or bool(act[env.t2i(c.time)])))
    for target in sorted({k, max(1, k - 1), k + 1}, key=lambda x: abs(x * frame - dur)):
        new_out = c_in.time + target * frame
        if ok(c_out, new_out):
            c_out.time, c_out.level_db = new_out, _level(env, new_out)
            return
        new_in = c_out.time - target * frame
        if ok(c_in, new_in):
            c_in.time, c_in.level_db = new_in, _level(env, new_in)
            return
    c_out.flags.append(CUT_FLAG_FRAME_OFF)


def pause_cut(env: Envelope, a_end: float, b_start: float, target: float, side: str,
              speech_only: bool = False) -> CutPoint:
    """A cut inside the longest silent run between two words, so that at most `target` of it
    is kept (PAUSE_OUT_SHARE after the run starts, the rest before it ends). Everything
    outside that run, including loud non-speech, is kept untouched."""
    s0, s1 = longest_silence(env, a_end, b_start, speech_only)
    if side == "out":
        lo, hi, prefer_late, edge, word = s0 + MARGIN_SEC, s0 + target * PAUSE_OUT_SHARE, False, s0, a_end
    else:
        lo, hi, prefer_late, edge, word = s1 - target * (1 - PAUSE_OUT_SHARE), s1 - MARGIN_SEC, True, s1, b_start
    t = _quietest(env, lo, hi, prefer_late)
    flags = [CUT_FLAG_IN_NOISE] if speech_only and _active(env)[env.t2i(t)] else []
    return CutPoint(time=t, side=side, word_edge=word, activity_edge=edge, window=[lo, hi], level_db=_level(env, t),
                    flags=flags)


def _dead_air(W, a: int, b: int, env: Envelope) -> Optional[bool]:
    """Dead air (user, 2026-10-06) between consecutive grid words: True = a real silence longer than
    PAUSE_MAX_SEC; False = no real silence, but a stretch longer than PAUSE_MAX_SEC with no VAD speech
    (noise only); None = neither. Any sound VAD calls speech is kept (an earlier whole-gap rule cut
    speech the grid has no words for on chai)."""
    if W[b].start - W[a].end <= PAUSE_MAX_SEC:
        return None
    if silence_between(env, W[a].end, W[b].start) > PAUSE_MAX_SEC:
        return True
    s0, s1 = longest_silence(env, W[a].end, W[b].start, speech_only=True)
    return False if s1 - s0 > PAUSE_MAX_SEC else None


def _split_long_pauses(idxs: List[int], W, env: Envelope) -> List[Tuple[List[int], Optional[float]]]:
    """Split a run at dead air between consecutive words (silent OR noisy, > PAUSE_MAX_SEC).
    Returns [(sub-run, gap shortened before it or None)]."""
    parts: List[Tuple[List[int], Optional[float]]] = [([idxs[0]], None)]
    for a, b in zip(idxs, idxs[1:]):
        if _dead_air(W, a, b, env) is not None:
            parts.append(([b], round(W[b].start - W[a].end, 3)))
        else:
            parts[-1][0].append(b)
    return parts


def cut_segment(W, idxs: List[int], env: Envelope, kept: set, pause_target: Optional[float] = None,
                search_sec: float = TAIL_SEARCH_SEC) -> Tuple[CutPoint, CutPoint]:
    """Both cuts of one segment (grid word indices idxs), by the S2/S3 rules:
    - neighbour not kept (or file edge): normal cut in the pause next to the kept word;
    - neighbour kept elsewhere, silence > PAUSE_MAX_SEC and pause_target set: S3 pause cut;
    - neighbour kept elsewhere otherwise (reordered grid neighbours): the gap is split at its midpoint.
    Also used by the Phase 3 repair to re-cut one segment."""
    first, last = W[idxs[0]], W[idxs[-1]]
    same = lambda j: 0 <= j < len(W) and W[j].source_file == first.source_file
    p, n = idxs[0] - 1, idxs[-1] + 1
    dead_in = _dead_air(W, p, idxs[0], env) if same(p) and p in kept and pause_target is not None else None
    dead_out = _dead_air(W, idxs[-1], n, env) if same(n) and n in kept and pause_target is not None else None
    if dead_in is not None:          # cut inside the silent (True) or speech-free noisy (False) stretch
        c_in = pause_cut(env, W[p].end, first.start, pause_target, "in", speech_only=not dead_in)
    else:
        prev_end = W[p].end if same(p) else None
        if prev_end is not None and p in kept:
            prev_end = (prev_end + first.start) / 2
        c_in = cut_in(env, first.start, prev_end, search_sec=search_sec)
    if dead_out is not None:
        c_out = pause_cut(env, last.end, W[n].start, pause_target, "out", speech_only=not dead_out)
    else:
        next_start = W[n].start if same(n) else None
        if next_start is not None and n in kept:
            next_start = (last.end + next_start) / 2
        c_out = cut_out(env, last.end, next_start, file_end=len(env.speech) * HOP_SEC, search_sec=search_sec)
    snap_to_frames(env, c_in, c_out)
    return c_in, c_out


def compile_plan(plan: EditPlan, grid: WordGrid, envs: Dict[str, Envelope],
                 pause_targets: Optional[Dict[str, float]] = None) -> List[CompiledSegment]:
    """Compile a plan into source ranges in output order. envs: per source file, with db levels.
    pause_targets: {source_file: seconds} turns on S3 pause shortening (None = off)."""
    W = grid.words
    runs = resolve_plan(plan, grid)
    kept = {i for idxs, _ in runs for i in idxs}
    out: List[CompiledSegment] = []
    for idxs, reasons in runs:
        f = W[idxs[0]].source_file
        env = envs.get(f)
        if env is None or env.db is None:
            raise PlanError(f"no audio envelope with levels for {f!r}")
        parts = _split_long_pauses(idxs, W, env) if pause_targets is not None else [(idxs, None)]
        target = pause_targets.get(f, PAUSE_TARGET_DEFAULT) if pause_targets is not None else None
        for sub, shortened in parts:
            c_in, c_out = cut_segment(W, sub, env, kept, pause_target=target)
            out.append(CompiledSegment(source_file=f, src_in=c_in.time, src_out=c_out.time,
                                       word_ids=[W[i].id for i in sub], reasons=reasons, cut_in=c_in, cut_out=c_out,
                                       pause_shortened_before_sec=shortened))
    flags = Counter(f for s in out for c in (s.cut_in, s.cut_out) for f in c.flags)
    shortened = [s.pause_shortened_before_sec for s in out if s.pause_shortened_before_sec]
    logger.info(f"[COMPILE] {len(plan.segments)} plan segments -> {len(out)} compiled segments; "
                f"cut flags {dict(flags) or 'none'}; pauses shortened {len(shortened)} "
                f"({sum(shortened):.1f}s of silence, targets {pause_targets})")
    for s in out:
        for c in (s.cut_in, s.cut_out):
            if c.flags:
                logger.warning(f"[COMPILE] {s.source_file} cut {c.side} @ {c.time:.3f}s {c.flags} "
                               f"(word edge {c.word_edge:.3f}, window {c.window[0]:.3f}-{c.window[1]:.3f})")
    return out


def build_timeline(segments: List[CompiledSegment], grid: WordGrid,
                   pause_targets: Optional[Dict[str, float]] = None) -> CompiledTimeline:
    """Wrap compiled segments with render settings. Global fades only cover the pause before
    the first word / after the last, so they never dim speech."""
    if not segments:
        raise PlanError("nothing to render: no compiled segments")
    head = min(MAX_EDGE_FADE_SEC, max(0.0, segments[0].cut_in.activity_edge - segments[0].src_in))
    tail = min(MAX_EDGE_FADE_SEC, max(0.0, segments[-1].src_out - segments[-1].cut_out.activity_edge))
    frames = sum(round((s.src_out - s.src_in) * FPS) for s in segments)
    return CompiledTimeline(segments=segments, fps=FPS, join_fade_sec=JOIN_FADE_SEC, head_fade_sec=round(head, 4),
                            tail_fade_sec=round(tail, 4), duration_sec=round(frames / FPS, 6),
                            grid_fingerprint=grid.fingerprint(), pause_targets=pause_targets)


def render_timeline(timeline: CompiledTimeline, file_map: Dict[str, str], output_path: str) -> Dict:
    """Render through app.utils.ffmpeg.render_compiled. file_map: source_file -> original media path."""
    from app.utils.ffmpeg import render_compiled
    return render_compiled([s.model_dump() for s in timeline.segments], file_map, output_path, fps=timeline.fps,
                           join_fade=timeline.join_fade_sec, head_fade=timeline.head_fade_sec,
                           tail_fade=timeline.tail_fade_sec)


def compile_and_render(plan: EditPlan, grid: WordGrid, envs: Dict[str, Envelope], file_map: Dict[str, str],
                       output_path: str, pauses: bool = True) -> Tuple[CompiledTimeline, Dict]:
    """Plan -> cuts (S2, + S3 pause shortening unless pauses=False) -> post-condition checks ->
    render -> render checks. A failed structural check raises before rendering; the report
    (info["validation"]) is attached either way it gets that far."""
    from app.tasks.validate import make_report, validate_render, validate_structure
    from app.utils.ffmpeg import media_durations
    targets = speaker_pause_targets(grid, envs) if pauses else None
    timeline = build_timeline(compile_plan(plan, grid, envs, pause_targets=targets), grid, pause_targets=targets)
    checks = validate_structure(timeline, plan, grid, envs)
    if any(c.status == "fail" for c in checks):
        report = make_report(checks)
        raise PlanError("compiled timeline failed validation:\n  " + "\n  ".join(
            f"{c.check} (segment {c.segment}): {c.detail}" for c in report.checks if c.status == "fail"))
    info = {**render_timeline(timeline, file_map, output_path), "pause_targets": targets}
    d = media_durations(output_path)
    info["validation"] = make_report(checks + validate_render(timeline, d["video"], d["audio"])).model_dump()
    return timeline, info
