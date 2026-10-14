"""Footage categories (roadmap Phase 10 v2, V1; user design 2026-10-12).

Every stretch of uploaded material is one of four categories; the edit follows from the combination:
  a_roll      the creator is seen and talks (speech recorded with that picture)
  voice_over  the creator's speech with no matching picture: an audio-only upload, or talk while the camera
              points away (nobody speaking is seen)
  b_roll      no speech, a picture worth showing
  silent      no speech, nothing worth showing (black, smeared, floor / pocket, lens covered)

Unit = one shot of the shot index (app.tasks.shots); an audio-only file is one voice_over stretch. Inputs per
shot: the share of the shot covered by recognised words (file-local ASR times) and the caption model's
`to_camera` / `speaking` / `usable` flags. A file's category is its longest category by duration (files
from a camera often mix: a talk to camera that turns into a pan of the room is a_roll then voice_over).
"""

from typing import Dict, List, Optional

from app.utils.ffmpeg import is_audio_only  # noqa: F401  (re-exported for callers)

CATEGORIES = ["a_roll", "voice_over", "b_roll", "silent"]
SPEECH_SHARE = 0.3          # a shot is "over speech" when words cover >= 30% of it (same rule as the gold)


def speech_share(words: List[Dict], s: float, e: float) -> float:
    covered = sum(min(w["end"], e) - max(w["start"], s) for w in words if w["end"] > s and w["start"] < e)
    return covered / max(e - s, 1e-6)


def shot_category(shot: Dict, share: float) -> str:
    if share >= SPEECH_SHARE:
        return "a_roll" if shot.get("to_camera") or shot.get("speaking") else "voice_over"
    return "b_roll" if shot.get("usable", True) else "silent"


def classify(words: List[Dict], index: Optional[Dict], duration: float) -> Dict:
    """words: the file's ASR words (file-local times); index: its shot index, None for an audio-only file.
    -> {category, audio_only, seconds: {category: s}, shots: [{i, start, end, category, speech_share}]}"""
    if index is None:
        return {"category": "voice_over", "audio_only": True, "seconds": {"voice_over": round(duration, 3)},
                "shots": [{"i": 0, "start": 0.0, "end": round(duration, 3), "category": "voice_over",
                           "speech_share": round(speech_share(words, 0.0, duration), 3)}]}
    shots, seconds = [], {c: 0.0 for c in CATEGORIES}
    for s in index["shots"]:
        share = speech_share(words, s["start"], s["end"])
        cat = shot_category(s, share)
        seconds[cat] += s["end"] - s["start"]
        shots.append({"i": s["i"], "start": s["start"], "end": s["end"], "category": cat,
                      "speech_share": round(share, 3)})
    return {"category": max(CATEGORIES, key=lambda c: seconds[c]), "audio_only": False,
            "seconds": {c: round(v, 3) for c, v in seconds.items() if v > 0}, "shots": shots}
