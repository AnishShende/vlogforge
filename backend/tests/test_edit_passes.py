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


def test_suggestion_only_pass_never_changes_the_edit(monkeypatch):
    g = _grid(["I I think so"])
    monkeypatch.setattr(ep, "_call", lambda *a, **k: ([{"attempt": 1, "from_word": 0, "to_word": 0, "category": "stutter",
                                                  "reason": "r", "uncertain": False}],
                                                 {"model": "m", "input_tokens": 0, "output_tokens": 0, "cached": False}))
    kept, cuts, _ = ep.run_passes(g, passes=["inside_take"])
    assert kept == {w.id for w in g.words} and len(cuts) == 1 and not cuts[0]["applied"]
