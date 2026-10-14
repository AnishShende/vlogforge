"""Phase 10 S0 helpers: speech overlap per shot, pre-labels, review windows (invented examples)."""

from eval.broll_gold import pick_windows, prelabel, speech_in


def _w(s, e, t="x"):
    return {"start": s, "end": e, "text": t}


def test_speech_in_counts_only_the_overlap():
    words = [_w(0.0, 1.0, "hello"), _w(1.5, 2.5, "there"), _w(9.0, 9.5, "later")]
    share, text = speech_in(words, 1.0, 3.0)
    assert abs(share - 0.5) < 1e-6          # 1.0 s of "there" inside a 2 s shot
    assert text == "there"


def test_prelabel_needs_speech_then_uses_to_camera():
    assert prelabel({"to_camera": True}, 0.1) == "silent"
    assert prelabel({"to_camera": True}, 0.8) == "a_roll"
    assert prelabel({"to_camera": False}, 0.8) == "b_roll"
    assert prelabel({"to_camera": False, "speaking": True}, 0.8) == "a_roll"


def _rows(labels, dur=10.0):
    return [{"i": i, "start": i * dur, "end": (i + 1) * dur, "pre": l} for i, l in enumerate(labels)]


def test_pick_windows_prefers_mixed_stretches_and_never_overlaps():
    # first half all A-roll, second half alternates A/B: the mixed half must win
    rows = _rows(["a_roll"] * 10 + ["a_roll", "b_roll"] * 10)
    wins = pick_windows(rows, n=2, length=40.0)
    assert len(wins) == 2
    (a1, b1), (a2, b2) = wins
    assert b1 <= a2                          # sorted, no overlap
    for a, b in wins:
        assert a >= 10                       # both inside the mixed half
        assert {"a_roll", "b_roll"} <= {r["pre"] for r in rows[a:b]}


def test_pick_windows_short_video_returns_what_fits():
    rows = _rows(["a_roll", "b_roll", "silent"])
    wins = pick_windows(rows, n=2, length=240.0)
    assert len(wins) <= 1


def test_pick_windows_one_shot_longer_than_the_window_is_kept_whole():
    rows = [{"i": 0, "start": 0.0, "end": 500.0, "pre": "a_roll"},
            {"i": 1, "start": 500.0, "end": 900.0, "pre": "b_roll"}]
    for a, b in pick_windows(rows, n=2, length=240.0):
        assert b > a                         # never an empty window
