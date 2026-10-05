"""Word grid model (Archdoc roadmap Phase 1 / S1)."""
from app.models import Word, WordGrid, generate_word_id


def _w(i, start, end, f="a.mov", gap=None, prev_end=None):
    return Word(id=generate_word_id(f, i), text=f"w{i}", source_file=f, start=start, end=end,
                gap_before=gap if gap is not None else (start - prev_end if prev_end is not None else 0.0))


def test_word_id_is_deterministic_and_file_scoped():
    assert generate_word_id("a.mov", 7) == generate_word_id("a.mov", 7)
    assert generate_word_id("a.mov", 7) != generate_word_id("b.mov", 7)
    assert generate_word_id("a.mov", 7) != generate_word_id("a.mov", 8)
    wid = generate_word_id("a.mov", 7)
    assert wid.startswith("w_") and wid.endswith("_00007") and len(wid) == 2 + 8 + 1 + 5


def test_valid_grid_has_no_errors():
    g = WordGrid(words=[_w(0, 0.0, 0.4), _w(1, 0.5, 0.9, prev_end=0.4), _w(2, 0.9, 0.9, prev_end=0.9)])
    assert g.validate_integrity() == []          # zero-length word is allowed (builder flags it)


def test_files_are_independent_timelines():
    g = WordGrid(words=[_w(0, 5.0, 6.0, "a.mov"), _w(0, 1.0, 2.0, "b.mov"), _w(1, 6.5, 7.0, "a.mov", prev_end=6.0)])
    assert g.validate_integrity() == []


def test_integrity_errors_are_reported():
    dup = WordGrid(words=[_w(0, 0.0, 0.4), _w(0, 0.5, 0.9, prev_end=0.4)])
    assert any("Duplicate" in e for e in dup.validate_integrity())
    backwards = WordGrid(words=[_w(0, 1.0, 0.5)])
    assert any("Invalid timestamps" in e for e in backwards.validate_integrity())
    overlap = WordGrid(words=[_w(0, 0.0, 0.6), _w(1, 0.5, 0.9, gap=-0.1)])
    assert any("Overlap" in e for e in overlap.validate_integrity())
    bad_gap = WordGrid(words=[_w(0, 0.0, 0.4), _w(1, 0.5, 0.9, gap=0.3)])
    assert any("gap_before mismatch" in e for e in bad_gap.validate_integrity())


def test_by_id_lookup():
    g = WordGrid(words=[_w(0, 0.0, 0.4), _w(1, 0.5, 0.9, prev_end=0.4)])
    assert g.by_id()[generate_word_id("a.mov", 1)].start == 0.5
