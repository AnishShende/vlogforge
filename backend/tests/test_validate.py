"""Post-condition checks on compiled output (roadmap Phase 3, reduced scope). Synthetic, no audio."""
from app.tasks.compiler import build_timeline, compile_plan
from app.tasks.validate import make_report, validate_render, validate_structure
from test_compiler_cuts import _env, _grid, _plan

G = _grid([(1.0, 1.5), (1.6, 2.0), (4.0, 4.5), (4.6, 5.0), (8.0, 8.5)])
ENV = {"a.mov": _env(10, speech=[(1.0, 2.0), (4.0, 4.5), (4.6, 5.0), (8.0, 8.5)])}   # 100 ms pause after the cut word
PLAN = _plan(G, (0, 1), (3, 4))            # word 2 is cut


def _fails(tl, plan=PLAN):
    return {c.check for c in validate_structure(tl, plan, G, ENV) if c.status == "fail"}


def _timeline():
    return build_timeline(compile_plan(PLAN, G, ENV), G)


def _with(tl, k, **upd):
    segs = list(tl.segments)
    s = segs[k]
    cuts = {}
    if "src_in" in upd:
        cuts["cut_in"] = s.cut_in.model_copy(update={"time": upd["src_in"]})
    if "src_out" in upd:
        cuts["cut_out"] = s.cut_out.model_copy(update={"time": upd["src_out"]})
    segs[k] = s.model_copy(update={**upd, **cuts})
    return build_timeline(segs, G)


def test_compiled_timeline_passes():
    assert _fails(_timeline()) == set()
    assert make_report(validate_structure(_timeline(), PLAN, G, ENV)).status == "pass"


def test_cut_inside_a_kept_word_fails():
    tl = _timeline()
    assert "cut_inside_word" in _fails(_with(tl, 0, src_in=1.2))


def test_range_reaching_an_unplanned_word_fails():
    # segment 1 (words 3-4) pulled back to 3.9 s swallows the cut word 2 (4.0-4.5) whole
    assert "unintended_word" in _fails(_with(_timeline(), 1, src_in=3.9))


def test_overlap_duplicate_and_missing_words_fail():
    tl = _timeline()
    dup = build_timeline([tl.segments[0], tl.segments[0], tl.segments[1]], G)
    assert {"overlap", "plan_coverage"} <= _fails(dup)
    assert "plan_coverage" in _fails(build_timeline([tl.segments[0]], G))


def test_unflagged_cut_on_speech_fails():
    tl = _timeline()                                        # compiled where speech ends at 2.0 ...
    later = {"a.mov": _env(10, speech=[(1.0, 2.4), (4.0, 5.0), (8.0, 8.5)])}   # ... but it runs to 2.4
    fails = {c.check for c in validate_structure(tl, PLAN, G, later) if c.status == "fail"}
    assert fails == {"cut_on_activity"}


def test_render_length_and_sync():
    tl = _timeline()
    ok = validate_render(tl, tl.duration_sec, tl.duration_sec + 0.008)
    assert all(c.status == "pass" for c in ok)
    bad = {c.check for c in validate_render(tl, tl.duration_sec - 0.1, tl.duration_sec + 0.2) if c.status == "fail"}
    assert bad == {"render_length", "av_sync"}
