"""Post-condition checks on compiled output (Archdoc Stage 11, roadmap Phase 3, reduced scope).

Cheap and deterministic, run on every compiled job; a failure means a compiler bug or a bad
hand-written plan, and is raised loudly. Whether the plan itself is right is Phase 4's job.

  structure (timeline, before render): times_valid, words_valid, plan_coverage, overlap,
      cut_inside_word, unintended_word, cut_on_activity, whole_frames, duration;
      warnings: short_segment, flagged_cut (tight / long_tail)
  render (after render): video length == timeline, |audio - video| < 1 frame

Deferred by user decision (2026-10-06, avoid over-engineering / keep the product fast):
repair loop (nothing produces broken timelines until the Phase 6 editor), per-segment
render correlation, forced-alignment word check.
"""

from collections import Counter
import logging
from typing import Dict, List

from app.models import (CHECK_FAIL, CHECK_PASS, CHECK_WARN, CUT_FLAG_FRAME_OFF, CUT_FLAG_IN_NOISE, CUT_FLAG_LONG_TAIL,
                        CUT_FLAG_TIGHT,
                        CompiledTimeline, EditPlan, ValidationCheck, ValidationReport, WordGrid)
from app.tasks.compiler import FPS, PlanError, _active, resolve_plan
from app.utils.speech_activity import HOP_SEC, Envelope

logger = logging.getLogger("VlogForge.Validate")

EPS = 1e-3                     # 1 ms: timing tolerance for overlaps / word edges
MIN_SEGMENT_SEC = 0.5          # warn below (user-approved; shortest gold segment 0.83 s)


def _row(check, status, segment=None, detail="", **evidence) -> ValidationCheck:
    return ValidationCheck(check=check, status=status, segment=segment, detail=detail, evidence=evidence)


# ---------------------------------------------------------------------------- S2

def validate_structure(timeline: CompiledTimeline, plan: EditPlan, grid: WordGrid,
                       envs: Dict[str, Envelope]) -> List[ValidationCheck]:
    W, index = grid.words, {w.id: i for i, w in enumerate(grid.words)}
    by_file: Dict[str, List[int]] = {}
    for i, w in enumerate(W):
        by_file.setdefault(w.source_file, []).append(i)
    rows: List[ValidationCheck] = []
    checked = Counter()

    for k, s in enumerate(timeline.segments):
        env = envs.get(s.source_file)
        dur = len(env.speech) * HOP_SEC if env is not None else None
        checked["times_valid"] += 1
        if env is None or not (-EPS <= s.src_in < s.src_out <= dur + EPS):
            rows.append(_row("times_valid", CHECK_FAIL, k, f"{s.src_in:.3f}-{s.src_out:.3f} outside 0-{dur}",
                             src_in=s.src_in, src_out=s.src_out, file_sec=dur))
            continue
        checked["words_valid"] += 1
        idxs = [index.get(wid) for wid in s.word_ids]
        if not idxs or None in idxs or any(b != a + 1 for a, b in zip(idxs, idxs[1:])) \
                or any(W[i].source_file != s.source_file for i in idxs):
            rows.append(_row("words_valid", CHECK_FAIL, k, "word ids unknown, not consecutive, or from another file",
                             word_ids=s.word_ids[:5]))
            continue
        mine = set(idxs)
        file_words = [W[i] for i in by_file[s.source_file]]
        for c in (s.cut_in, s.cut_out):
            checked["cut_inside_word"] += 1
            inside = [w for w in file_words if w.start + EPS < c.time < w.end - EPS]
            if inside:
                rows.append(_row("cut_inside_word", CHECK_FAIL, k, f"{c.side} cut {c.time:.3f} inside '{inside[0].text}'",
                                 side=c.side, time=c.time, word=inside[0].id, word_span=[inside[0].start, inside[0].end]))
        checked["unintended_word"] += 1
        extra = [W[i] for i in by_file[s.source_file] if i not in mine
                 and min(W[i].end, s.src_out) - max(W[i].start, s.src_in) > EPS]
        if extra:
            rows.append(_row("unintended_word", CHECK_FAIL, k, f"range keeps {len(extra)} word(s) not in the plan: "
                             + " ".join(w.text for w in extra[:5]), words=[w.id for w in extra]))
        act = _active(env)
        for c in (s.cut_in, s.cut_out):
            if c.time <= EPS or c.time >= dur - EPS:            # file edge: not an edit
                continue
            checked["cut_on_activity"] += 1
            on = bool(act[env.t2i(c.time)])
            known = set(c.flags) & {CUT_FLAG_TIGHT, CUT_FLAG_LONG_TAIL, CUT_FLAG_IN_NOISE}
            if on and not known:
                rows.append(_row("cut_on_activity", CHECK_FAIL, k, f"unflagged {c.side} cut {c.time:.3f} on speech/loud audio",
                                 side=c.side, time=c.time))
            if known:
                rows.append(_row("flagged_cut", CHECK_WARN, k, f"{c.side} cut {c.time:.3f} {sorted(known)}"
                                 + (" (on activity)" if on else ""), side=c.side, time=c.time, flags=c.flags))
        checked["whole_frames"] += 1
        frames = (s.src_out - s.src_in) * FPS
        if abs(frames - round(frames)) > 1e-4:
            status = CHECK_WARN if CUT_FLAG_FRAME_OFF in s.cut_out.flags else CHECK_FAIL
            rows.append(_row("whole_frames", status, k, f"{frames:.4f} frames", frames=frames))
        checked["short_segment"] += 1
        if s.src_out - s.src_in < MIN_SEGMENT_SEC:
            rows.append(_row("short_segment", CHECK_WARN, k, f"{s.src_out - s.src_in:.2f}s < {MIN_SEGMENT_SEC}s"))

    checked["overlap"] += 1
    spans = sorted((s.source_file, s.src_in, s.src_out, k) for k, s in enumerate(timeline.segments))
    for a, b in zip(spans, spans[1:]):
        if a[0] == b[0] and a[2] > b[1] + EPS:
            rows.append(_row("overlap", CHECK_FAIL, b[3], f"source {b[1]:.3f} < {a[2]:.3f} (segment {a[3]})",
                             other=a[3], overlap_sec=round(a[2] - b[1], 4)))

    checked["plan_coverage"] += 1
    try:
        planned = [W[i].id for idxs, _ in resolve_plan(plan, grid) for i in idxs]
    except PlanError as e:
        planned = None
        rows.append(_row("plan_coverage", CHECK_FAIL, None, f"plan invalid: {e}"))
    if planned is not None:
        got = Counter(wid for s in timeline.segments for wid in s.word_ids)
        missing = [w for w in planned if got[w] == 0]
        dup = [w for w, n in got.items() if n > 1]
        extra = sorted(set(got) - set(planned))
        if missing or dup or extra:
            rows.append(_row("plan_coverage", CHECK_FAIL, None,
                             f"missing {len(missing)}, duplicated {len(dup)}, not planned {len(extra)}",
                             missing=missing[:20], duplicated=dup[:20], extra=extra[:20]))
    checked["duration"] += 1
    frames = sum(round((s.src_out - s.src_in) * FPS) for s in timeline.segments)
    if abs(frames / FPS - timeline.duration_sec) > EPS:
        rows.append(_row("duration", CHECK_FAIL, None, f"timeline says {timeline.duration_sec}, segments sum {frames / FPS}"))

    failed = {r.check for r in rows if r.status == CHECK_FAIL}
    rows += [_row(name, CHECK_PASS, None, f"{n} checked", checked=n) for name, n in sorted(checked.items())
             if name not in failed]
    return rows


# ---------------------------------------------------------------------------- render

def validate_render(timeline: CompiledTimeline, video_sec: float, audio_sec: float) -> List[ValidationCheck]:
    """video_sec: container duration; audio_sec: decoded sample count / rate."""
    return [
        _row("render_length", CHECK_PASS if abs(video_sec - timeline.duration_sec) < 2 * EPS else CHECK_FAIL, None,
             f"video {video_sec:.3f}s vs timeline {timeline.duration_sec:.3f}s", video_sec=video_sec,
             expected_sec=timeline.duration_sec),
        _row("av_sync", CHECK_PASS if abs(audio_sec - video_sec) < 1 / FPS else CHECK_FAIL, None,
             f"|audio - video| {abs(audio_sec - video_sec) * 1000:.1f} ms", audio_sec=audio_sec, video_sec=video_sec),
    ]


def make_report(checks: List[ValidationCheck]) -> ValidationReport:
    status = (CHECK_FAIL if any(c.status == CHECK_FAIL for c in checks) else
              CHECK_WARN if any(c.status == CHECK_WARN for c in checks) else CHECK_PASS)
    report = ValidationReport(status=status, checks=checks)
    log = logger.error if status == CHECK_FAIL else logger.warning if status == CHECK_WARN else logger.info
    log(f"[VALIDATE] {report.summarize()}")
    for c in checks:
        if c.status == CHECK_FAIL:
            logger.error(f"[VALIDATE] FAIL {c.check} segment {c.segment}: {c.detail}")
    return report
