"""Compiler cut-point selection (Archdoc roadmap Phase 2 / S2). Synthetic envelopes, no audio."""
import random

import numpy as np
import pytest

from app.models import (CUT_FLAG_FRAME_OFF, CUT_FLAG_LONG_TAIL, CUT_FLAG_TIGHT, EditPlan, EditSegment, Word,
                        WordGrid, generate_word_id)
from app.tasks.compiler import (FPS, MARGIN_SEC, MAX_PAUSE_IN_SEC, MAX_PAUSE_OUT_SEC, PlanError, compile_plan,
                                resolve_plan)
from app.utils.speech_activity import HOP_SEC, Envelope

QUIET, SPEECH = -60.0, -20.0


def _env(dur, speech=(), loud=(), dips=()):
    """speech/loud: [(start, end)] intervals; dips: [(t, db)] single quiet frames."""
    n = int(round(dur / HOP_SEC))
    sp, ld, db = np.zeros(n, bool), np.zeros(n, bool), np.full(n, QUIET)
    for mask, ivs in ((sp, speech), (ld, loud)):
        for a, b in ivs:
            mask[int(round(a / HOP_SEC)): int(round(b / HOP_SEC))] = True
    db[sp | ld] = SPEECH
    for t, v in dips:
        db[int(round(t / HOP_SEC))] = v
    return Envelope(speech=sp, threshold_db=-40, floor_db=QUIET, peak_db=SPEECH, method="vad", loud=ld, db=db)


def _grid(times, f="a.mov"):
    words, prev = [], None
    for i, (s, e) in enumerate(times):
        words.append(Word(id=generate_word_id(f, i), text=f"w{i}", source_file=f, start=s, end=e,
                          gap_before=(s - prev) if prev is not None else 0.0))
        prev = e
    return WordGrid(words=words)


def _plan(grid, *ranges):
    return EditPlan(segments=[EditSegment(word_start=grid.words[a].id, word_end=grid.words[b].id) for a, b in ranges])


def _frames(x):
    return abs(x * FPS - round(x * FPS)) < 1e-6


def test_adjacent_plan_segments_merge_into_one_run():
    g = _grid([(1, 1.4), (1.5, 1.9), (2.0, 2.4), (3, 3.4), (5, 5.4)])
    runs = resolve_plan(_plan(g, (0, 0), (1, 2), (4, 4)), g)
    assert [r[0] for r in runs] == [[0, 1, 2], [4]]


def test_reordered_grid_neighbours_split_their_shared_gap():
    g = _grid([(1.0, 1.5), (2.5, 3.0)])
    env = _env(4, speech=[(1.0, 1.5), (2.5, 3.0)])
    later, earlier = compile_plan(_plan(g, (1, 1), (0, 0)), g, {"a.mov": env})
    assert earlier.src_out <= 2.0 + 1e-9 <= later.src_in + 2e-9


def test_invalid_plan_raises():
    g = _grid([(1, 1.4)])
    with pytest.raises(PlanError):
        compile_plan(EditPlan(), g, {"a.mov": _env(3)})


def test_out_cut_waits_for_speech_past_the_word_end():
    g = _grid([(1.0, 1.5), (4.0, 4.5)])
    env = _env(6, speech=[(1.0, 1.65), (4.0, 4.5)])          # aligner end 1.5, speech runs to 1.65
    seg = compile_plan(_plan(g, (0, 0)), g, {"a.mov": env})[0]
    assert seg.cut_out.activity_edge == pytest.approx(1.65)
    assert seg.src_out >= 1.65 + MARGIN_SEC - 1e-9
    assert seg.src_out <= 1.65 + MAX_PAUSE_OUT_SEC + 1e-9
    assert seg.cut_out.flags == []


def test_loud_frames_protect_speech_vad_missed():
    g = _grid([(1.0, 1.5), (4.0, 4.5)])
    env = _env(6, speech=[(1.0, 1.4)], loud=[(1.0, 1.8)])    # VAD stops early, loudness continues
    seg = compile_plan(_plan(g, (0, 0)), g, {"a.mov": env})[0]
    assert seg.src_out >= 1.8 + MARGIN_SEC - 1e-9


def test_in_cut_keeps_at_most_the_incoming_pause():
    g = _grid([(0.2, 0.6), (3.0, 3.5)])
    env = _env(6, speech=[(0.2, 0.6), (2.95, 3.5)])
    seg = compile_plan(_plan(g, (1, 1)), g, {"a.mov": env})[0]
    assert 2.95 - MAX_PAUSE_IN_SEC - 1e-9 <= seg.src_in <= 2.95 - MARGIN_SEC + 1e-9


def test_cut_goes_to_the_quietest_frame():
    g = _grid([(1.0, 1.5), (4.0, 4.5)])
    env = _env(6, speech=[(1.0, 1.5), (4.0, 4.5)], dips=[(1.62, -75.0)])
    seg = compile_plan(_plan(g, (0, 0)), g, {"a.mov": env})[0]
    # the dip wins unless frame snapping had to move the out cut; then the in cut absorbed it
    assert seg.cut_out.time == pytest.approx(1.62) or CUT_FLAG_FRAME_OFF in seg.cut_out.flags \
        or seg.cut_out.level_db == QUIET


def test_no_pause_is_flagged_tight_and_cuts_between_word_edges():
    g = _grid([(1.0, 1.5), (1.52, 2.0)])
    env = _env(3, speech=[(1.0, 2.0)])
    seg = compile_plan(_plan(g, (0, 0)), g, {"a.mov": env})[0]
    assert CUT_FLAG_TIGHT in seg.cut_out.flags
    assert 1.5 - 1e-9 <= seg.src_out <= 1.52 + 1e-9


def test_endless_activity_is_flagged_long_tail():
    g = _grid([(1.0, 1.5), (5.0, 5.5)])
    env = _env(7, loud=[(1.0, 3.0)], speech=[(5.0, 5.5)])   # kitchen noise after the word
    seg = compile_plan(_plan(g, (0, 0)), g, {"a.mov": env})[0]
    assert CUT_FLAG_LONG_TAIL in seg.cut_out.flags


def _random_case(rng):
    t, times, speech, loud = rng.uniform(0, 1), [], [], []
    for _ in range(rng.randint(3, 25)):
        dur = rng.uniform(0.0, 0.6) if rng.random() > 0.05 else 0.0          # some zero-length words
        times.append((round(t, 3), round(t + dur, 3)))
        speech.append((t + rng.uniform(-0.05, 0.05), t + dur + rng.uniform(-0.05, 0.3)))
        if rng.random() < 0.2:
            loud.append((t + dur, t + dur + rng.uniform(0, 1.2)))            # noise tails, some > limit
        t += dur + rng.choice([0.0, 0.02, 0.05, 0.1, 0.3, 0.8, 2.0])
    env = _env(t + 2, speech=speech, loud=loud, dips=[(rng.uniform(0, t), rng.uniform(-90, -40)) for _ in range(10)])
    g = _grid(times)
    idx, ranges = 0, []
    while idx < len(times):
        a = idx + rng.randint(0, 2)
        b = a + rng.randint(0, 3)
        if b >= len(times):
            break
        ranges.append((a, b))
        idx = b + 1
    return g, env, ranges


def test_properties_on_random_grids_and_plans():
    rng = random.Random(1234)
    checked = 0
    for _ in range(300):
        g, env, ranges = _random_case(rng)
        if not ranges:
            continue
        if rng.random() < 0.3:
            rng.shuffle(ranges)                                             # plans may reorder
        plan = _plan(g, *ranges)
        segs = compile_plan(plan, g, {"a.mov": env})
        assert segs == compile_plan(plan, g, {"a.mov": env})                 # deterministic
        W = g.words
        for s in segs:
            first, last = g.by_id()[s.word_ids[0]], g.by_id()[s.word_ids[-1]]
            assert s.src_in <= first.start + 1e-9 and s.src_out >= last.end - 1e-9
            for c in (s.cut_in, s.cut_out):                                 # never inside any word
                assert not any(w.start + 1e-9 < c.time < w.end - 1e-9 for w in W)
                assert c.window[0] - 1e-9 <= c.time <= c.window[1] + 1e-9
                at_file_edge = c.time <= 1e-9 or c.time >= len(env.speech) * HOP_SEC - 1e-9  # not an edit
                if not set(c.flags) & {CUT_FLAG_TIGHT, CUT_FLAG_LONG_TAIL, 'valley'} and not at_file_edge:  # in a pause
                    i = env.t2i(c.time)
                    assert not (env.speech[i] or env.loud[i]), (c, s)
            if CUT_FLAG_FRAME_OFF not in s.cut_out.flags:
                assert _frames(s.src_out - s.src_in)
        spans = sorted((s.src_in, s.src_out) for s in segs)
        assert all(a[1] <= b[0] + 1e-9 for a, b in zip(spans, spans[1:]))  # no overlaps in the source
        checked += 1
    assert checked > 200


# --- S3 pause normalisation ---------------------------------------------------------------

from app.tasks.compiler import PAUSE_OUT_SHARE, PAUSE_TARGET_DEFAULT, speaker_pause_targets  # noqa: E402


def _kept_pause(a_seg, b_seg):
    """Silence the listener hears across a split: after a's activity + before b's activity."""
    return (a_seg.src_out - a_seg.cut_out.activity_edge) + (b_seg.cut_in.activity_edge - b_seg.src_in)


def test_long_silence_inside_a_run_is_shortened_to_the_target():
    g = _grid([(1.0, 1.5), (1.6, 2.0), (8.0, 8.5), (8.6, 9.0)])
    env = _env(10, speech=[(1.0, 2.0), (8.0, 9.0)])                  # 6 s of silence inside the run
    segs = compile_plan(_plan(g, (0, 3)), g, {"a.mov": env}, pause_targets={"a.mov": 0.4})
    assert [len(s.word_ids) for s in segs] == [2, 2]
    assert segs[1].pause_shortened_before_sec == pytest.approx(6.0, abs=0.02)
    assert 2 * MARGIN_SEC - 1e-9 <= _kept_pause(*segs) <= 0.4 + 1e-9  # shortened, never to zero


def test_pause_shortening_off_by_default_and_short_pauses_untouched():
    g = _grid([(1.0, 1.5), (8.0, 8.5), (9.3, 9.8)])
    env = _env(11, speech=[(1.0, 1.5), (8.0, 8.5), (9.3, 9.8)])
    assert len(compile_plan(_plan(g, (0, 2)), g, {"a.mov": env})) == 1                  # off: one segment
    segs = compile_plan(_plan(g, (0, 2)), g, {"a.mov": env}, pause_targets={"a.mov": 0.5})
    assert [len(s.word_ids) for s in segs] == [1, 2]                                    # 0.8 s pause kept


def test_noisy_dead_air_is_shortened_too():
    g = _grid([(1.0, 1.5), (5.0, 5.5)])
    env = _env(7, speech=[(1.0, 1.5), (5.0, 5.5)], loud=[(1.5, 5.0)])  # kitchen noise, no words
    segs = compile_plan(_plan(g, (0, 1)), g, {"a.mov": env}, pause_targets={"a.mov": 0.5})
    assert len(segs) == 2 and segs[1].pause_shortened_before_sec == pytest.approx(3.5)
    kept_gap = (segs[0].src_out - 1.5) + (5.0 - segs[1].src_in)
    assert 2 * MARGIN_SEC - 1e-9 <= kept_gap <= 0.5 + 1e-9                # shortened to the target, never to zero
    assert segs[0].src_out <= segs[1].src_in                            # cuts never cross


def test_speaker_pause_target_is_the_median_natural_pause_clamped():
    times, speech, t = [], [], 0.5
    for gap in [0.2, 0.4, 0.6, 0.6, 0.6, 0.9, 3.0, 0.01]:          # 3.0 and 0.01 are not natural pauses
        times.append((t, t + 0.3)); speech.append((t, t + 0.3)); t += 0.3 + gap
    times.append((t, t + 0.3)); speech.append((t, t + 0.3))
    g, env = _grid(times), _env(t + 1, speech=speech)
    assert speaker_pause_targets(g, {"a.mov": env})["a.mov"] == pytest.approx(0.6, abs=0.02)
    few = _grid(times[:3])
    assert speaker_pause_targets(few, {"a.mov": env})["a.mov"] == PAUSE_TARGET_DEFAULT


def test_properties_hold_with_pause_shortening():
    rng = random.Random(99)
    for _ in range(300):
        g, env, ranges = _random_case(rng)
        if not ranges:
            continue
        segs = compile_plan(_plan(g, *ranges), g, {"a.mov": env}, pause_targets={"a.mov": 0.5})
        W = g.words
        for s in segs:
            for c in (s.cut_in, s.cut_out):
                assert not any(w.start + 1e-9 < c.time < w.end - 1e-9 for w in W)
                at_file_edge = c.time <= 1e-9 or c.time >= len(env.speech) * HOP_SEC - 1e-9
                if not set(c.flags) & {CUT_FLAG_TIGHT, CUT_FLAG_LONG_TAIL, 'in_noise', 'valley'} and not at_file_edge:
                    i = env.t2i(c.time)
                    assert not (env.speech[i] or env.loud[i])
            if CUT_FLAG_FRAME_OFF not in s.cut_out.flags:
                assert _frames(s.src_out - s.src_in)
        for a, b in zip(segs, segs[1:]):
            if b.pause_shortened_before_sec:
                assert _kept_pause(a, b) <= 0.5 + 1e-9
        spans = sorted((s.src_in, s.src_out) for s in segs)
        assert all(x[1] <= y[0] + 1e-9 for x, y in zip(spans, spans[1:]))


def test_untranscribed_burst_before_word_is_split_off_at_the_dip():
    """A stray sound (stutter) runs into the kept word's onset with only a deep dip, no silence,
    between them: the cut goes in the dip, not before the stray sound (user report 2026-10-07)."""
    speech = [(0.5, 0.8), (1.48, 1.60), (1.60, 1.75), (1.75, 2.2)]   # w0 | stray burst, dip, w1 (aligned late)
    env = _env(3.0, speech=speech, dips=[(1.65 + k * HOP_SEC, -58.0) for k in range(5)])
    g = _grid([(0.5, 0.8), (1.80, 2.2)])
    seg = compile_plan(_plan(g, (1, 1)), g, {"a.mov": env})[0]
    assert "valley" in seg.cut_in.flags
    assert 1.65 - 1e-9 <= seg.src_in <= 1.70 + 1e-9                 # in the dip: stray burst left out
    seg = compile_plan(_plan(g, (0, 0)), g, {"a.mov": env})[0]       # w0's cut out is a normal pause cut
    assert "valley" not in seg.cut_out.flags
