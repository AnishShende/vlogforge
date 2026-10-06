"""Speech cleanup on the grid (roadmap Phase 4): keep by default, remove only a worse attempt
at a line that has a better one. Injected scorer; invented sentences (no user footage)."""
from app.models import Word, WordGrid, generate_word_id
from app.tasks.speech_cleanup import KEEP, REMOVE, REVIEW, cleanup_plan, label_grid


def _grid(phrases):
    """phrases: [(start_sec, "text", pause_after_sec)] -> one grid word per token, 0.25 s each."""
    words, t, prev = [], None, None
    for start, text, _ in phrases:
        t = start
        for tok in text.split():
            words.append(Word(id=generate_word_id("a.mov", len(words)), text=tok, source_file="a.mov", start=t,
                              end=t + 0.25, gap_before=(t - prev) if prev is not None else 0.0))
            prev = t + 0.25
            t += 0.3
    return WordGrid(words=words)


def _by_phrase(grid, labels, phrases):
    out, i = [], 0
    for _, text, _ in phrases:
        n = len(text.split())
        out.append({l["label"] for l in labels[i:i + n]})
        i += n
    return out


def score_by(table):
    return lambda text: next((v for k, v in table.items() if text.startswith(k)), (1.0, 0.5))


def test_unique_narration_is_never_removed():
    phrases = [(0.0, "today we are making a fresh pot of ginger tea at home", 0),
               (4.5, "first we boil the water with a few crushed cardamom pods", 0),
               (9.0, "then the tea leaves go in along with plenty of milk", 0)]
    g = _grid(phrases)
    labels = label_grid(g, scorer=score_by({}))
    assert _by_phrase(g, labels, phrases) == [{KEEP}, {KEEP}, {KEEP}]


def test_worse_attempt_of_a_line_is_removed_and_better_kept():
    line = "the small habits you repeat every day shape who you become"
    phrases = [(0.0, "the small habits you repeat every day um shape who", 0), (6.0, line, 0)]
    g = _grid(phrases)
    labels = label_grid(g, scorer=score_by({line: (3.9, 0.9)}))
    assert _by_phrase(g, labels, phrases) == [{REMOVE}, {KEEP}]
    plan = cleanup_plan(g, labels)
    assert [s.word_start for s in plan.segments] == [g.words[10].id]   # 10 words in the first attempt


def test_two_clean_takes_keep_the_best_and_flag_review():
    a = "the small habits you repeat every day shape who you become"
    b = "the small habits you repeat each day shape who you become"
    phrases = [(0.0, a, 0), (6.0, b, 0)]
    g = _grid(phrases)
    labels = label_grid(g, scorer=score_by({a: (3.9, 0.9), b: (3.5, 0.8)}))
    assert _by_phrase(g, labels, phrases) == [{REVIEW}, {REMOVE}]
    review = next(l for l in labels if l["label"] == REVIEW)
    assert review["by"]["alternatives"][0]["start"] == 6.0
    assert len(cleanup_plan(g, labels).segments) == 1          # review is kept in the plan
