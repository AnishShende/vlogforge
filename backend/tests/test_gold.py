"""Gold schema + draft helper tests (Archdoc Phase 0 / S2). Hermetic: no media, no pipeline."""
from dataclasses import dataclass

import pytest

from eval.gold import Gold
from eval.draft_gold import EXCLUDE_SPLIT_GAP_SEC, _exclude_runs


def _gold(**over):
    base = dict(
        clip_id="c1", status="draft",
        sources=[{"file": "a.mov", "path": "x/a.mov", "sha256": "0" * 64}],
        keep=[{"id": "k1", "source_file": "a.mov", "start": 10.0, "end": 14.0, "text": "hello there"}],
        exclude=[{"id": "x1", "source_file": "a.mov", "start": 2.0, "end": 5.0, "text": "hello th",
                  "reason": "retake"}],
    )
    base.update(over)
    return base


def test_valid_gold_parses():
    g = Gold.model_validate(_gold())
    assert g.keep[0].id == "k1" and g.exclude[0].reason == "retake"


@pytest.mark.parametrize("over, msg", [
    ({"exclude": [{"id": "x1", "source_file": "a.mov", "start": 13.0, "end": 16.0, "text": "t",
                   "reason": "retake"}]}, "overlaps"),
    ({"exclude": [{"id": "k1", "source_file": "a.mov", "start": 1.0, "end": 2.0, "text": "t",
                   "reason": "retake"}]}, "duplicate span ids"),
    ({"keep": [{"id": "k1", "source_file": "b.mov", "start": 10.0, "end": 14.0, "text": "t"}]}, "not in sources"),
    ({"keep": [{"id": "k1", "source_file": "a.mov", "start": 14.0, "end": 10.0, "text": "t"}]}, "invalid range"),
    ({"keep": [{"id": "k1", "source_file": "a.mov", "start": 10.0, "end": 14.0, "text": "  "}]}, "empty text"),
    ({"status": "reviewed",
      "exclude": [{"id": "x1", "source_file": "a.mov", "start": 2.0, "end": 5.0, "text": "t",
                   "reason": "unreviewed"}]}, "reason=unreviewed"),
    ({"schema_version": 99}, "schema_version"),
])
def test_invalid_gold_rejected(over, msg):
    with pytest.raises(ValueError, match=msg):
        Gold.model_validate(_gold(**over))


def test_overlap_only_checked_within_same_source():
    g = _gold(sources=[{"file": "a.mov", "path": "a", "sha256": "0" * 64},
                       {"file": "b.mov", "path": "b", "sha256": "0" * 64}],
              exclude=[{"id": "x1", "source_file": "b.mov", "start": 11.0, "end": 13.0, "text": "t",
                        "reason": "retake"}])
    Gold.model_validate(g)   # same times, different file: fine


@dataclass
class W:
    word: str
    start_sec: float
    end_sec: float
    source_file: str = "a.mov"


@dataclass
class K:
    source_file: str
    start: float
    end: float


def test_exclude_runs_skip_kept_words_and_split_on_gap():
    tl = [W("so", 0.0, 0.2), W("I", 0.3, 0.4),                               # run 1
          W("then", 0.4 + EXCLUDE_SPLIT_GAP_SEC + 0.5, 2.2),                   # run 2 (gap split)
          W("kept", 5.0, 5.4), W("words", 5.5, 5.9),                           # inside keep
          W("tail", 7.0, 7.3)]                                                 # run 3
    runs, dropped = _exclude_runs(tl, [K("a.mov", 5.0, 5.9)])
    assert [r[3] for r in runs] == ["so I", "then", "tail"] and dropped == 0


def test_exclude_runs_dedupe_and_clamp_to_keep_edge():
    # word straddles the keep start (midpoint outside) -> run end clamped to keep start
    tl = [W("a", 1.0, 1.2), W("a", 1.0, 1.2), W("b", 1.3, 2.4)]
    runs, _ = _exclude_runs(tl, [K("a.mov", 2.0, 4.0)])
    assert runs == [("a.mov", 1.0, 2.0, "a b")]


def test_exclude_runs_split_on_source_change():
    tl = [W("a", 1.0, 1.2, "a.mov"), W("b", 1.3, 1.5, "b.mov")]
    runs, _ = _exclude_runs(tl, [])
    assert [r[0] for r in runs] == ["a.mov", "b.mov"]


def test_optional_spans_and_quality():
    g = Gold.model_validate(_gold(
        keep=[{"id": "k1", "source_file": "a.mov", "start": 10.0, "end": 14.0, "text": "t", "quality": "imperfect"}],
        optional=[{"id": "o1", "source_file": "a.mov", "start": 15.0, "end": 16.0, "text": "ha"}]))
    assert g.keep[0].quality == "imperfect" and g.optional[0].id == "o1"


def test_optional_overlap_rejected():
    with pytest.raises(ValueError, match="overlaps"):
        Gold.model_validate(_gold(optional=[{"id": "o1", "source_file": "a.mov", "start": 13.0, "end": 16.0,
                                             "text": "t"}]))
