"""Phase 10 E1: splitting a finished cut into raw pieces (invented timings)."""

from eval.broll_synth import assign_words, build_plan, runs_of


def _w(s, e, t="w"):
    return {"start": s, "end": e, "text": t}


SHOTS = [{"i": 0, "start": 0.0, "end": 4.0, "label": "a_roll"},
         {"i": 1, "start": 4.0, "end": 6.0, "label": "b_roll"},
         {"i": 2, "start": 6.0, "end": 8.0, "label": "b_roll"},
         {"i": 3, "start": 8.0, "end": 10.0, "label": "silent"},
         {"i": 4, "start": 10.0, "end": 14.0, "label": "a_roll"}]


def test_runs_merge_same_labels():
    assert [(r["label"], r["start"], r["end"]) for r in runs_of(SHOTS)] == \
        [("a_roll", 0, 4), ("b_roll", 4, 8), ("silent", 8, 10), ("a_roll", 10, 14)]


def test_word_hanging_over_an_aroll_edge_goes_to_the_voice_over():
    runs = runs_of(SHOTS)
    words = [_w(1.0, 1.5), _w(3.8, 4.3), _w(5.0, 5.4), _w(10.5, 11.0)]
    assert assign_words(words, runs) == [(0, "aroll"), (1, "vo"), (1, "vo"), (3, "aroll")]


def test_every_word_has_exactly_one_owner_and_parts_elsewhere_are_muted():
    words = [_w(1.0, 1.5), _w(3.8, 4.3), _w(5.0, 5.4), _w(6.5, 7.0), _w(9.0, 9.3), _w(10.5, 11.0)]
    plan = build_plan(words, SHOTS)
    owned = sorted(i for p in plan["pieces"] if p["kind"] in ("aroll", "vo", "silent") for i in p["words"])
    assert owned == list(range(len(words)))
    first_aroll = plan["pieces"][0]
    assert first_aroll["kind"] == "aroll" and first_aroll["mute"] == [[3.8, 4.0]]


def test_one_muted_video_per_broll_shot_and_one_voice_over_per_run():
    words = [_w(4.5, 5.0), _w(6.5, 7.0)]
    plan = build_plan(words, SHOTS)
    kinds = [p["kind"] for p in plan["pieces"]]
    assert kinds.count("broll") == 2 and kinds.count("vo") == 1
    vo = next(p for p in plan["pieces"] if p["kind"] == "vo")
    assert vo["start"] == 4.25 and vo["end"] == 7.25 and vo["shots"] == [1, 2]


def test_voice_over_never_reaches_into_a_neighbours_word():
    words = [_w(3.0, 3.9), _w(4.0, 4.5)]          # A-roll word ends 0.1 s before the voice-over word
    vo = next(p for p in build_plan(words, SHOTS)["pieces"] if p["kind"] == "vo")
    assert vo["start"] >= 3.9
