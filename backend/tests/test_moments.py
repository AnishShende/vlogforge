"""Phase 8 moment table: structural validation and clip order (no API calls)."""
from app.tasks.moments import _validate, clip_order
from app.utils.ffmpeg import parse_recording_time


def _m(start, deps=(), imp=0.5, role="body", section="s", announces=()):
    return {"start_word": start, "function": "explanation", "importance": imp, "depends_on": list(deps),
            "role": role, "section": section, "announces": list(announces), "summary": "s"}


def test_moments_tile_the_edit_and_dependencies_point_backward():
    ms, notes = _validate({"moments": [_m(0), _m(5, [1]), _m(9, [1, 2, 3, 7], imp=1.4)]}, 12)
    assert [(m["from"], m["to"]) for m in ms] == [(0, 4), (5, 8), (9, 11)]
    assert ms[2]["depends_on"] == [1, 2] and ms[2]["importance"] == 1.0      # self / forward deps dropped, clamped
    assert notes


def test_unusable_starts_are_rejected():
    assert _validate({"moments": [_m(2), _m(5)]}, 10)[0] is None             # does not start at the first word
    assert _validate({"moments": [_m(0), _m(5), _m(5)]}, 10)[0] is None      # not increasing
    assert _validate({"moments": [_m(0), _m(12)]}, 10)[0] is None            # past the end


def test_structure_fields_are_normalised():
    ms, notes = _validate({"moments": [_m(0, role="intro", section="ignored"), _m(3, section="  Beach  Walk "),
                                       _m(6, section=""), _m(8, role="weird")]}, 10)
    assert [m["role"] for m in ms] == ["intro", "body", "body", "body"]
    assert ms[0]["section"] == "" and ms[1]["section"] == "beach walk" and ms[2]["section"] == "part 3" and notes


def test_announced_parts_resolve_to_the_section_of_the_footage_they_point_at():
    ann = [{"part": "Market", "first_moment": 0}, {"part": "beach", "first_moment": 2},
           {"part": "kitchen", "first_moment": 1}]                    # no footage / named differently / intro
    ms, notes = _validate({"moments": [_m(0, role="intro"), _m(3, section="seaside"), _m(6, section="seaside"),
                                       _m(8, section="market", announces=ann)]}, 10)
    assert ms[3]["announces"] == ["market", "seaside", "kitchen"]           # later footage, forward pointer
    assert any("kitchen" in n for n in notes)


def test_recording_time_only_from_real_metadata():
    assert parse_recording_time({"com.apple.quicktime.creationdate": "2026-10-07T11:07:40+0530",
                                 "creation_time": "2026-10-07T05:37:40.000000Z"}) == "2026-10-07T11:07:40+05:30"
    assert parse_recording_time({"creation_time": "2026-10-07T05:37:40.000000Z"}) == "2026-10-07T05:37:40+00:00"
    assert parse_recording_time({}) is None                                  # stripped (messaging apps)
    assert parse_recording_time({"creation_time": "1970-01-01T00:00:00Z"}) is None   # clock never set
    assert parse_recording_time({"creation_time": "garbage"}) is None


def test_clips_sort_by_recording_time_only_when_every_clip_has_one(monkeypatch):
    times = {"b.mov": "2026-10-07T11:00:00+05:30", "a.mov": "2026-10-07T12:00:00+05:30", "c.mp4": None}
    monkeypatch.setattr("app.utils.ffmpeg.recording_time", lambda p: times[p])
    files = [{"filename": f, "path": f} for f in ("a.mov", "b.mov")]
    assert [c["filename"] for c in clip_order(files)] == ["b.mov", "a.mov"]          # by recording time
    files.append({"filename": "c.mp4", "path": "c.mp4"})
    assert [c["filename"] for c in clip_order(files)] == ["a.mov", "b.mov", "c.mp4"]  # one unknown: upload order
