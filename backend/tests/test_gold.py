"""Gold schema tests. Hermetic: no media, no pipeline."""
import pytest

from eval.gold import Gold


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


def test_optional_spans_and_quality():
    g = Gold.model_validate(_gold(
        keep=[{"id": "k1", "source_file": "a.mov", "start": 10.0, "end": 14.0, "text": "t", "quality": "imperfect"}],
        optional=[{"id": "o1", "source_file": "a.mov", "start": 15.0, "end": 16.0, "text": "ha"}]))
    assert g.keep[0].quality == "imperfect" and g.optional[0].id == "o1"


def test_optional_overlap_rejected():
    with pytest.raises(ValueError, match="overlaps"):
        Gold.model_validate(_gold(optional=[{"id": "o1", "source_file": "a.mov", "start": 13.0, "end": 16.0,
                                             "text": "t"}]))
