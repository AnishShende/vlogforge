"""Phase 10 measures: placement scoring and shot-choice queries (invented examples)."""

from eval.broll_eval import placement_score, query_text, word_targets


def _w(s, t="w"):
    return {"start": s, "end": s + 0.4, "text": t}


def test_word_targets_skip_silent_shots_and_mark_broll_words():
    words = [_w(0.1), _w(1.1), _w(2.1)]
    gold = {"shots": [{"start": 0, "end": 1, "label": "a_roll"}, {"start": 1, "end": 2, "label": "b_roll"},
                      {"start": 2, "end": 3, "label": "silent"}]}
    assert word_targets(words, gold) == [(0, False), (1, True)]


def test_placement_score_precision_recall():
    targets = [(0, True), (1, True), (2, False), (3, False)]
    assert placement_score(targets, {0, 2}) == {"precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert placement_score(targets, set())["recall"] == 0.0


def test_query_text_widens_short_queries_with_neighbours():
    words = [_w(i, f"w{i}") for i in range(20)]
    q = query_text(words, 10.0, 10.5)            # one word inside the shot
    assert len(q.split()) >= 8 and "w10" in q
    assert query_text(words, 100.0, 101.0) == ""
