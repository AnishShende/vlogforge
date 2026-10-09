"""Phase 9 story plans: the hard rules and the story order, enforced in code (no API calls)."""
from app.tasks.storyplan import repair, score, story_order, structure


def _m(mid, imp=0.5, deps=(), fn="explanation", role=None, section="", announces=(), clip=1):
    m = {"id": mid, "importance": imp, "depends_on": list(deps), "function": fn, "clip": clip}
    if role:
        m.update(role=role, section=section, announces=list(announces))
    return m


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


# Listing order: sign-off, the beach footage (filmed first), the welcome, then a line filmed later that
# announces "first the market, then the beach" (no market footage).
STRUCTURED = [_m("m1", role="outro", clip=1), _m("m2", role="body", section="beach", clip=2),
              _m("m3", ["m2"], role="body", section="beach", clip=2), _m("m4", role="intro", clip=3),
              _m("m5", role="body", section="market", announces=["market", "beach"], clip=4)]


def test_intro_first_outro_last_and_the_announcement_before_what_it_announces():
    r = story_order(STRUCTURED, ["beach"])
    assert r["order"] == ["m4", "m5", "m2", "m3", "m1"]
    assert r["warnings"] == ["m5 announces market, but there is no footage of it"]


def test_stated_order_wins_over_the_model_section_order():
    ms = [_m("a", role="body", section="cafe", clip=1), _m("b", role="body", section="beach", clip=2),
          _m("c", role="body", section="hike", clip=3),
          _m("d", role="body", section="cafe", announces=["hike", "cafe"], clip=4)]
    r = story_order(ms, ["cafe", "beach", "hike"])
    assert r["order"] == ["d", "c", "b", "a"] and r["sections"] == ["hike", "beach", "cafe"]


def test_unstated_sections_follow_the_model_order_and_a_clip_keeps_its_recording_order():
    ms = [_m("a", role="body", section="evening", clip=1), _m("b", role="body", section="morning", clip=2),
          _m("c", role="body", section="morning", clip=2)]
    r = story_order(ms, ["morning", "evening"])
    assert r["order"] == ["b", "c", "a"] and not r["notes"]
    r = story_order([_m("x", role="body", section="b", clip=1), _m("y", role="body", section="a", clip=1)], ["a", "b"])
    assert r["order"] == ["y", "x"] and r["notes"] == ["clip 1: moments reordered within one continuous clip"]


def test_a_teaser_with_no_footage_ends_the_body():
    ms = [_m("a", role="intro"), _m("b", role="body", section="cooking"),
          _m("c", role="body", section="next video", announces=["paris trip"]), _m("d", role="body", section="cooking"),
          _m("e", role="outro")]
    r = story_order(ms, ["cooking"])
    assert r["order"] == ["a", "b", "d", "c", "e"] and r["warnings"]


def test_structure_scores_the_middle_too():
    assert structure(["m4", "m5", "m2", "m3", "m1"], STRUCTURED) == 1.0
    assert structure(["m4", "m2", "m3", "m5", "m1"], STRUCTURED) == 2 / 3     # the bug: announcement after the tour
    assert structure(["m2", "m3"], STRUCTURED) is None                         # nothing to check


def test_moments_added_back_go_to_their_place_in_the_story_order():
    order, _ = repair(["m4", "m5"], STRUCTURED, {m["id"]: 5 for m in STRUCTURED}, target=60, full=25,
                      ref=["m4", "m5", "m2", "m3", "m1"])
    assert order == ["m4", "m5", "m2", "m3", "m1"]                             # the sign-off goes back last
