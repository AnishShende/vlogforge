"""Post copy for the export panel: format by length, chapter rules, hashtag cleanup. Model stubbed."""
import pytest

from app.config import settings
from app.tasks import publish_copy as pc
from app.utils import artifacts


def _job(job_id, seg_secs, moments=True):
    """A stored render: one word per segment, segments of the given lengths."""
    words = [{"id": f"w{i}", "text": f"word{i}", "source_file": "a.mov", "start": 10.0 * i, "end": 10.0 * i + 0.4} for i in range(len(seg_secs))]
    artifacts.save_job(job_id, "grid", {"words": words})
    artifacts.save_job(job_id, "files", [{"filename": "a.mov", "path": "a.mov", "audio_path": "a.wav"}])
    segs = [{"source_file": "a.mov", "word_ids": [f"w{i}"], "src_in": 10.0 * i, "src_out": 10.0 * i + d} for i, d in enumerate(seg_secs)]
    artifacts.save_job(job_id, "timeline", {"segments": segs, "duration_sec": sum(seg_secs)})
    if moments:
        artifacts.save_job(job_id, "moments", {"synopsis": "s", "creator_goal": "g", "moments": [
            {"id": f"m{i}", "function": "explanation", "summary": f"Point {i}", "word_ids": [f"w{i}"]} for i in range(len(seg_secs))]})


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "artifact_dir", str(tmp_path / "artifacts"))
    monkeypatch.setattr(settings, "enable_artifact_cache", True)
    calls = []

    def fake_call(fmt, user, variant):
        calls.append((fmt, user, variant))
        if fmt == "short":
            v = lambda n: {"hook": f"Hook {n} v{variant}", "body": f"Body {n}", "cta": f"CTA {n}"}
            return {"audience": "young men", "tone": "Witty & casual", "variations": [v(1), v(2), v(3)],
                    "hashtags": {"niche": ["#BroCode", "male friendship", "brocode"], "content": ["friendship", "Bros!"], "broad": ["lifestyle"]}}, {}
        return {"audience": "a", "keyword": "k", "titles": {"curiosity": "C title", "seo": "S title", "stakes": "H title"},
                "description_hook": "Hook.", "description_outline": "Outline.", "pinned_comments": ["P1", " ", "P2", "P3"],
                "hashtags": ["vlog"],
                "chapters": [{"moment": 1, "label": "Intro"}, {"moment": 2, "label": "Too soon"}, {"moment": 3, "label": "Middle"},
                             {"moment": 5, "label": "End"}, {"moment": 99, "label": "Unknown moment"}]}, {}
    monkeypatch.setattr(pc, "_call", fake_call)
    return calls


def test_short_video_gets_three_variations_and_grouped_hashtags(store):
    _job("s", [20, 20, 15])                                      # 55 s
    c = pc.publish_copy("s")
    assert c["format"] == "short" and c["auto_format"] == "short"
    assert [(v["key"], v["hook"]) for v in c["variations"]] == [("value_drop", "Hook 1 v0"), ("micro_story", "Hook 2 v0"), ("short_punchy", "Hook 3 v0")]
    assert "Topic/Core Message: s" in store[-1][1] and "Content Format: Reel" in store[-1][1]   # the creator's brief, filled
    # lowercased, no spaces/punctuation, de-duplicated across groups
    assert c["hashtag_groups"] == {"niche": ["#brocode", "#malefriendship"], "content": ["#friendship", "#bros"], "broad": ["#lifestyle"]}
    assert c["hashtags"] == ["#brocode", "#malefriendship", "#friendship", "#bros", "#lifestyle"]


def test_long_video_chapters_follow_youtube_rules(store):
    _job("l", [12, 5, 20, 20, 12])                              # 69 s; moment starts 0, 12, 17, 37, 57
    c = pc.publish_copy("l")
    assert c["format"] == "long" and [t["text"] for t in c["titles"]] == ["C title", "S title", "H title"]
    assert c["pinned_comments"] == ["P1", "P2"] and c["links"].startswith("- Lead Magnet") and c["hashtags"] == ["#vlog"]
    # moment 2 at 12 s: kept; moment 3 at 17 s is 5 s after it: dropped; moment 5 at 57 s ends 12 s later: kept; 99 unknown
    assert [(x["time"], x["label"]) for x in c["chapters"]] == [("0:00", "Intro"), ("0:12", "Too soon"), ("0:57", "End")]
    assert c["chapter_note"] is None


def test_too_few_long_sections_means_no_chapters(store):
    _job("f", [55, 6])                                          # 61 s, second section < 10 s
    c = pc.publish_copy("f")
    assert c["format"] == "long" and c["chapters"] == [] and "at least 3" in c["chapter_note"]


def test_cached_per_render_and_regenerate_makes_a_new_variant(store):
    _job("s", [20, 20])
    pc.publish_copy("s"); pc.publish_copy("s")
    assert len(store) == 1                                       # second open is free
    c = pc.publish_copy("s", regenerate=True)
    assert c["variant"] == 1 and len(store) == 2 and store[-1][2] == 1 and c["variations"][0]["hook"] == "Hook 1 v1"
    _job("s", [20, 21])                                          # a re-compile changes the render
    assert pc.publish_copy("s")["variant"] == 0 and len(store) == 3


def test_format_override_and_no_moment_table(store):
    _job("n", [30, 40], moments=False)                           # 70 s, no moments: clips are the sections
    c = pc.publish_copy("n", "short")
    assert c["format"] == "short" and c["auto_format"] == "long"
    assert "1. (0:00) clip: word0" in store[-1][1]


def test_no_render():
    assert pc.publish_copy("missing") is None


def test_background_write_waits_for_the_last_render(store, monkeypatch):
    import time
    monkeypatch.setattr(settings, "post_copy_delay_sec", 0.2)
    _job("b", [20, 20])
    pc.schedule_background("b"); time.sleep(0.1); pc.schedule_background("b")    # a re-compile restarts the wait
    time.sleep(0.15)
    assert store == []                                                     # first timer was cancelled
    time.sleep(0.3)
    assert len(store) == 1 and artifacts.load_job("b", "publish_short")["variations"]
    assert pc.publish_copy("b") and len(store) == 1                        # the panel then opens from the cache


def test_two_callers_share_one_model_call(store, monkeypatch):
    import threading, time
    slow = pc._call
    monkeypatch.setattr(pc, "_call", lambda *a: (time.sleep(0.2), slow(*a))[1])
    _job("c", [20, 20])
    out = []
    ts = [threading.Thread(target=lambda: out.append(pc.publish_copy("c"))) for _ in range(2)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(store) == 1 and out[0]["variations"] == out[1]["variations"]


def test_background_off(store, monkeypatch):
    monkeypatch.setattr(settings, "enable_background_post_copy", False)
    pc.schedule_background("x")
    assert "x" not in pc._timers
