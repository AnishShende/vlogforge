"""Phase 10 V4: voice-over stretches on the output clock, slots and shot fitting (invented timelines)."""

from types import SimpleNamespace as NS

from app.tasks.broll import _slots, fit_shot, voice_over_stretches


def _seg(f, a, b, ids):
    return NS(source_file=f, src_in=a, src_out=b, word_ids=ids)


def _w(f, i, a, b):
    return NS(source_file=f, id=f"{f}{i}", start=a, end=b, text=f"w{i}")


def test_stretches_follow_the_output_clock_and_join_adjacent_voice_over():
    words = {"a.mp4": [_w("a.mp4", 0, 1.0, 1.5)], "vo.m4a": [_w("vo.m4a", 0, 10.2, 10.6), _w("vo.m4a", 1, 20.1, 20.5)]}
    tl = NS(fps=30, segments=[_seg("a.mp4", 0.9, 1.9, ["a.mp40"]),           # out 0.0-1.0
                              _seg("vo.m4a", 10.0, 11.0, ["vo.m4a0"]),       # out 1.0-2.0
                              _seg("vo.m4a", 20.0, 21.0, ["vo.m4a1"])])      # out 2.0-3.0
    st = voice_over_stretches(tl, words, {"vo.m4a": [(0.0, 1e9)]})
    assert len(st) == 1
    assert abs(st[0]["rec_in"] - 1.0) < 1e-6 and abs(st[0]["rec_out"] - 3.0) < 1e-6
    assert [round(w["start"], 3) for w in st[0]["words"]] == [1.2, 2.1]


def test_words_cut_from_the_edit_are_not_in_the_stretch():
    words = {"vo.m4a": [_w("vo.m4a", 0, 0.2, 0.5), _w("vo.m4a", 1, 0.6, 0.9)]}
    tl = NS(fps=30, segments=[_seg("vo.m4a", 0.0, 1.0, ["vo.m4a1"])])
    assert [w["id"] for w in voice_over_stretches(tl, words, {"vo.m4a": [(0.0, 1e9)]})[0]["words"]] == ["vo.m4a1"]


def test_slots_tile_the_stretch_with_cuts_in_the_pauses():
    st = {"rec_in": 1.0, "rec_out": 9.0}
    beats = [{"start": 1.2, "end": 3.8}, {"start": 4.2, "end": 8.7}]
    assert _slots(st, beats) == [(1.0, 4.0), (4.0, 9.0)]


def test_fit_shot_takes_the_middle_of_a_long_shot_and_all_of_a_short_one():
    long = fit_shot({"file": "b.mp4", "start": 10.0, "end": 20.0}, 2.0, 6.0)
    assert (long["src_in"], long["src_out"]) == (13.0, 17.0)
    short = fit_shot({"file": "b.mp4", "start": 10.0, "end": 12.0}, 2.0, 6.0)
    assert (short["src_in"], short["src_out"]) == (10.0, 12.0) and short["rec_out"] == 6.0
