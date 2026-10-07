"""Phase 6.5 edit passes: deterministic parts (alignment, hints, validation, apply rules). No API calls."""
from app.models import Word, WordGrid, generate_word_id
from app.tasks import edit_passes as ep


def _grid(lines, gap=1.0):
    """lines: list of sentences; each becomes one attempt (pauses between them)."""
    words, t, k = [], 0.0, 0
    for line in lines:
        for n, tok in enumerate(line.split()):
            words.append(Word(id=generate_word_id("a.mov", k), text=tok, source_file="a.mov", start=t, end=t + 0.2,
                              gap_before=(gap if n == 0 and k else 0.05) if k else 0.0))
            t += 0.25
            k += 1
        t += gap
    # gap_before must match times exactly for the grid's own checks; recompute
    for i in range(1, len(words)):
        words[i].gap_before = round(words[i].start - words[i - 1].end, 6)
    return WordGrid(words=words)


def test_local_alignment_skips_fillers_and_finds_partial_takes():
    a = "today we're going to um test the new".split()
    b = "today we're going to test the new camera outdoors".split()
    score, pairs = ep.local_align(a, b)
    assert [a[i] for i, _ in pairs] == ["today", "we're", "going", "to", "test", "the", "new"]


def test_hints_pair_retakes_but_not_different_numbers():
    g = _grid(["the bakery opens early on most weekday mornings",
               "the bakery opens really early on weekday mornings",
               "the train leaves from platform two every evening",
               "the train leaves from platform 14 every evening"])
    atts = ep.attempts_of(g, {w.id for w in g.words})
    hints = ep.retake_hints(g, atts)
    assert [(h["a"], h["b"]) for h in hints] == [(0, 1)]     # 2 vs 3 differ in a number: not a retake


def test_invalid_ranges_rejected_and_uncertain_cuts_become_suggestions(monkeypatch):
    g = _grid(["so we start here", "this line is said once", "this line is said once again"])
    calls = []

    def fake(model, system, user, categories, extra=None):
        calls.append(user)
        return [{"attempt": 2, "from_word": 0, "to_word": 4, "category": "other_take", "reason": "r", "uncertain": True, "kept_attempt": 3},
                {"attempt": 9, "from_word": 0, "to_word": 1, "category": "other_take", "reason": "bad", "uncertain": False, "kept_attempt": 3},
                {"attempt": 1, "from_word": 0, "to_word": 0, "category": "other_take", "reason": "r", "uncertain": True, "kept_attempt": 2},
                {"attempt": 3, "from_word": 0, "to_word": 5, "category": "other_take", "reason": "slip", "uncertain": False, "kept_attempt": 3}], \
               {"model": model, "input_tokens": 0, "output_tokens": 0, "cached": False}
    monkeypatch.setattr(ep, "_call", fake)
    cuts, st = ep.run_pass("retakes", g, {w.id for w in g.words})
    assert st["rejected"] == 1 and "Possible repeats" in calls[0]
    by_attempt = {c["text"].split()[0]: c for c in cuts if not c["reason"].startswith("slip")}
    assert by_attempt["this"]["applied"]                                      # uncertain but wording-corroborated
    assert not by_attempt["so"]["applied"]                                    # uncertain, no repeat found: suggestion
    slip = [c for c in cuts if c["reason"].startswith("slip")][0]
    assert not slip["applied"] and st["contradictions"] == 1                 # says it keeps the take it cuts


def test_filler_phrases_stay_suggestions(monkeypatch):
    g = _grid(["you know I think so"])
    monkeypatch.setattr(ep, "_call", lambda *a, **k: ([{"attempt": 1, "from_word": 0, "to_word": 1, "category": "filler_phrase",
                                                  "reason": "r", "uncertain": False}],
                                                 {"model": "m", "input_tokens": 0, "output_tokens": 0, "cached": False}))
    kept, cuts, _ = ep.run_passes(g, passes=["inside_take"])
    assert kept == {w.id for w in g.words} and len(cuts) == 1 and not cuts[0]["applied"]


def test_phrase_said_twice_in_a_row_loses_its_first_copy():
    g = _grid(["because every because every partner counts", "it was very very good", "I I think so"])
    kept = {w.id for w in g.words}
    cuts = ep.no_repeated_phrases(g, kept)
    assert [c["text"] for c in cuts] == ["because every", "I"]          # "very very" is emphasis: kept


def test_plan_and_ui_ranges_from_cuts():
    g = _grid(["uh so we start here", "this line is said once"])
    ids = [w.id for w in g.words]
    cuts = [{"pass": "retakes", "category": "other_take", "reason": "r", "applied": True, "review": True,
             "word_ids": ids[0:2], "text": "uh so"},
            {"pass": "inside_take", "category": "filler", "reason": "f", "applied": False, "review": False,
             "word_ids": ids[6:7], "text": "line"}]
    kept = set(ids) - set(ids[0:2])
    plan = ep.plan_from_kept(g, kept)
    assert [(s.word_start, s.word_end) for s in plan.segments] == [(ids[2], ids[-1])] and not plan.validate_against(g)
    ranges = ep.cut_ranges(g, cuts)
    assert [(r["label"], r["review"]) for r in ranges] == [("remove", True), ("suggest", False)]


def test_no_line_is_ever_kept_twice():
    g = _grid(["so the bakery opens early on most weekday mornings", "then we walk to the river",
               "the bakery opens early on most weekday mornings", "and that is the end"])
    kept = {w.id for w in g.words}
    cuts = ep.no_duplicate_lines(g, kept)
    assert len(cuts) == 1 and cuts[0]["applied"] and cuts[0]["review"]
    assert cuts[0]["text"].startswith("so the bakery")                   # the earlier copy goes (redo wins)
    assert ep.no_duplicate_lines(g, kept) == []                          # nothing left to remove
