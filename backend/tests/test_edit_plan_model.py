"""EditPlan contract (Archdoc roadmap Phase 2 / S1)."""
from app.models import EditPlan, EditSegment, Word, WordGrid, generate_word_id


def _grid():
    words, prev = [], {}
    for f, n in (("a.mov", 6), ("b.mov", 3)):
        for i in range(n):
            s = 1.0 * i
            words.append(Word(id=generate_word_id(f, i), text=f"w{i}", source_file=f, start=s, end=s + 0.5,
                              gap_before=(s - prev[f]) if f in prev else 0.0))
            prev[f] = s + 0.5
    return WordGrid(words=words)


def _id(f, i):
    return generate_word_id(f, i)


def _plan(*ranges, **kw):
    return EditPlan(segments=[EditSegment(word_start=a, word_end=b) for a, b in ranges], **kw)


def test_valid_plan_has_no_errors():
    g = _grid()
    p = _plan((_id("a.mov", 0), _id("a.mov", 2)), (_id("b.mov", 1), _id("b.mov", 1)), (_id("a.mov", 4), _id("a.mov", 5)),
              grid_fingerprint=g.fingerprint())
    assert p.validate_against(g) == []


def test_empty_plan_is_an_error():
    assert any("no segments" in e for e in EditPlan().validate_against(_grid()))


def test_unknown_reversed_and_cross_file_segments():
    g = _grid()
    assert any("unknown word id" in e for e in _plan(("w_nope_00000", _id("a.mov", 1))).validate_against(g))
    assert any("comes after" in e for e in _plan((_id("a.mov", 3), _id("a.mov", 1))).validate_against(g))
    assert any("spans files" in e for e in _plan((_id("a.mov", 5), _id("b.mov", 0))).validate_against(g))


def test_word_used_twice_is_an_error():
    g = _grid()
    p = _plan((_id("a.mov", 0), _id("a.mov", 3)), (_id("a.mov", 3), _id("a.mov", 4)))
    assert any("already used by segment 0" in e for e in p.validate_against(g))


def test_fingerprint_ignores_times_but_not_words():
    g = _grid()
    shifted = g.model_copy(deep=True)
    shifted.words[0].end = 0.45                       # edge refinement keeps the plan valid
    assert shifted.fingerprint() == g.fingerprint()
    retext = g.model_copy(deep=True)
    retext.words[0].text = "other"                    # a different transcript does not
    assert retext.fingerprint() != g.fingerprint()
    p = _plan((_id("a.mov", 0), _id("a.mov", 1)), grid_fingerprint=retext.fingerprint())
    assert any("not this grid" in e for e in p.validate_against(g))
