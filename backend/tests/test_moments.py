"""Phase 8 moment table: structural validation (no API calls)."""
from app.tasks.moments import _validate


def _m(start, deps=(), imp=0.5):
    return {"start_word": start, "function": "explanation", "importance": imp, "depends_on": list(deps), "summary": "s"}


def test_moments_tile_the_edit_and_dependencies_point_backward():
    ms, notes = _validate({"moments": [_m(0), _m(5, [1]), _m(9, [1, 2, 3, 7], imp=1.4)]}, 12)
    assert [(m["from"], m["to"]) for m in ms] == [(0, 4), (5, 8), (9, 11)]
    assert ms[2]["depends_on"] == [1, 2] and ms[2]["importance"] == 1.0      # self / forward deps dropped, clamped
    assert notes


def test_unusable_starts_are_rejected():
    assert _validate({"moments": [_m(2), _m(5)]}, 10)[0] is None             # does not start at the first word
    assert _validate({"moments": [_m(0), _m(5), _m(5)]}, 10)[0] is None      # not increasing
    assert _validate({"moments": [_m(0), _m(12)]}, 10)[0] is None            # past the end
