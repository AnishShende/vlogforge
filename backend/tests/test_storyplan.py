"""Phase 9 story plans: the hard rules enforced in code (no API calls)."""
from app.tasks.storyplan import repair, score


def _m(mid, imp=0.5, deps=(), fn="explanation"):
    return {"id": mid, "importance": imp, "depends_on": list(deps), "function": fn}


MOMENTS = [_m("m1", 0.3, fn="transition"), _m("m2", 0.9, fn="hook"), _m("m3", 0.4, ["m2"]),
           _m("m4", 0.2), _m("m5", 0.8, ["m3"], fn="conclusion")]
DURS = {"m1": 5, "m2": 10, "m3": 10, "m4": 10, "m5": 10}


def test_moments_come_after_what_they_need():
    order, notes = repair(["m5", "m2"], MOMENTS, DURS, target=30, full=45)
    assert order.index("m2") < order.index("m3") < order.index("m5") and notes


def test_full_edit_that_fits_keeps_every_moment_and_only_reorders():
    order, _ = repair(["m2", "m5"], MOMENTS, DURS, target=60, full=45)
    assert sorted(order) == ["m1", "m2", "m3", "m4", "m5"]
    assert order.index("m2") < order.index("m3") < order.index("m5")


def test_too_long_drops_least_important_moment_nothing_needs():
    order, notes = repair(["m2", "m3", "m4", "m5", "m1"], MOMENTS, DURS, target=30, full=45)
    assert "m4" not in order and "m1" not in order and order == ["m2", "m3", "m5"]


def test_score_prefers_an_opening_first_and_a_closing_last():
    good = score(["m2", "m3", "m5"], MOMENTS, 30, 30)
    bad = score(["m3", "m5", "m2"], MOMENTS, 30, 30)
    assert good["shape"] == 1.0 and good["score"] > bad["score"]
