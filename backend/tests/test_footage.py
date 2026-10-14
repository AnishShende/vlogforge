"""Phase 10 V1: footage categories (invented shots)."""

from app.tasks.footage import classify, shot_category, speech_share


def _w(s, e):
    return {"start": s, "end": e, "text": "w"}


def test_shot_category_rules():
    assert shot_category({"to_camera": True}, 0.8) == "a_roll"
    assert shot_category({"to_camera": False, "speaking": True}, 0.8) == "a_roll"
    assert shot_category({"to_camera": False, "speaking": False}, 0.8) == "voice_over"
    assert shot_category({"usable": True}, 0.0) == "b_roll"
    assert shot_category({"usable": False}, 0.1) == "silent"


def test_speech_share_clips_words_to_the_shot():
    assert speech_share([_w(0.5, 1.5)], 1.0, 2.0) == 0.5


def test_audio_only_file_is_one_voice_over_stretch():
    r = classify([_w(0.2, 0.8)], None, 3.0)
    assert r["category"] == "voice_over" and r["audio_only"] and len(r["shots"]) == 1


def test_file_category_is_the_longest_by_duration():
    index = {"shots": [{"i": 0, "start": 0.0, "end": 2.0, "to_camera": True},
                       {"i": 1, "start": 2.0, "end": 8.0, "to_camera": False, "speaking": False}]}
    words = [_w(0.1, 1.9), _w(2.5, 7.5)]
    r = classify(words, index, 8.0)
    assert [s["category"] for s in r["shots"]] == ["a_roll", "voice_over"]
    assert r["category"] == "voice_over"


def test_beats_merge_short_lines_and_keep_word_ranges():
    from app.tasks.broll import build_beats
    lines = [{"n": k + 1, "a": 2 * k, "b": 2 * k + 1, "start": float(k), "end": k + 0.8, "text": f"l{k}"} for k in range(7)]
    beats = build_beats(lines, min_sec=2.5)
    assert all(b["end"] - b["start"] >= 2.5 for b in beats)
    assert beats[0]["a"] == 0 and beats[-1]["b"] == 13 and " ".join(b["text"] for b in beats) == "l0 l1 l2 l3 l4 l5 l6"
