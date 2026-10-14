"""B-roll for voice-over, chosen for its place in the story (roadmap Phase 10 v2, V3 + V3b; user 2026-10-12).

Voice-over (footage.py) has no picture of its own, so B-roll always covers it. What to show is decided by the
story, not by the words alone:
  1. sections: one call places every library shot in a section of the moment table (or "other")
  2. visuals:  one call writes, per voice-over line, the footage that would show it, knowing the synopsis,
               the line's moment (summary, section) and the lines around it
  3. ranking:  SigLIP text embedding of the visual vs shot embeddings; shots of the line's section first
  4. pick:     one call per batch of beats (lines merged to >= BEAT_MIN_SEC: one shot per beat) ranks up to 3
               candidates; code gives each beat its first choice not used yet in the video (no shot twice),
               else the best unused shot of its own section; None only when the section has no footage
               ("needs footage")
The model only names line / candidate numbers; code checks them. Every call is cached by its content.
"""

import hashlib
import json
import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.utils import artifacts
from app.utils.llm import claude_client

logger = logging.getLogger("VlogForge.BRoll")

MODEL = "claude-sonnet-5-5"
PROMPT_VERSION = "3"
LINE_PAUSE_SEC = 0.7
LINE_MAX_WORDS = 25
TOP_K = 12
BEAT_MIN_SEC = 2.5          # one shot covers at least this much voice-over (editors: ~2 s montage .. ~6 s)
PICKS_PER_CALL = 15
OTHER = "other"

SECTIONS_SYSTEM = """You sort the shots of a creator's footage library into the sections of their video. You get the
video's synopsis, its sections (each with the moments in it) and the shots (number + what the shot shows).
For every shot return the section whose content it shows (the same place, activity or topic), or "other"
when it fits no section or plainly belongs to a different video. Answer every shot."""

VISUALS_SYSTEM = """You are the editor of a vlog. The numbered lines below are VOICE-OVER: the creator's narration with
no picture of its own, so footage will play under every line. For each line write `visual`: what the camera
should show while it plays, as a short concrete description of a shot (place, objects, people, action, camera
view), as if searching a footage library. Use the story: the line's moment and section, and the lines around
it, tell you WHERE and WHEN in the video it is ("this place" means the place of this section). Never describe
the creator talking to the camera. Speech recognition errors are common; judge by meaning."""

PICK_SYSTEM = """You choose B-roll for voice-over in a vlog. The voice-over has no picture of its own, so a shot MUST
play under every request. For each request you get the spoken words, their place in the story (section, moment
summary), the footage wanted, and candidate shots from the creator's library (number + what the shot shows +
its section). Rank up to 3 candidates, best first: shots that show what is said, or failing that, shots from the
same part of the story (same place, trip or activity) that make a natural cutaway. A shot from another section
or another video is wrong even when its description matches some words. Return an empty list only when no
candidate belongs to this part of the story at all. Answer every request."""


# ---------------------------------------------------------------- lines

def build_lines(words: Sequence[Dict]) -> List[Dict]:
    """Words [{start, end, text}] -> lines [{n, a, b, start, end, text}] (a..b inclusive word indices),
    split at sentence ends, pauses >= LINE_PAUSE_SEC, or LINE_MAX_WORDS."""
    lines, a = [], 0
    for i, w in enumerate(words):
        last = i == len(words) - 1
        end_sentence = bool(re.search(r"[.?!]['\"]?$", w["text"].strip()))
        pause = not last and words[i + 1]["start"] - w["end"] >= LINE_PAUSE_SEC
        if last or end_sentence or pause or i - a + 1 >= LINE_MAX_WORDS:
            lines.append({"n": len(lines) + 1, "a": a, "b": i, "start": words[a]["start"], "end": w["end"],
                          "text": " ".join(x["text"].strip() for x in words[a:i + 1])})
            a = i + 1
    return lines


def build_beats(lines: List[Dict], min_sec: float = BEAT_MIN_SEC) -> List[Dict]:
    """Consecutive lines merged until each beat lasts >= min_sec (a last short beat joins the one before).
    One shot plays under each beat. Beats keep the a..b word range and text of their lines."""
    beats: List[Dict] = []
    for l in lines:
        if beats and beats[-1]["end"] - beats[-1]["start"] < min_sec:
            b = beats[-1]
            b.update(end=l["end"], b=l["b"], text=f"{b['text']} {l['text']}")
        else:
            beats.append(dict(l))
    if len(beats) > 1 and beats[-1]["end"] - beats[-1]["start"] < min_sec:
        last = beats.pop()
        beats[-1].update(end=last["end"], b=last["b"], text=f"{beats[-1]['text']} {last['text']}")
    for n, b in enumerate(beats, 1):
        b["n"] = n
    return beats


# ---------------------------------------------------------------- model calls

def _call(stage: str, system: str, user: str, schema: Dict) -> Tuple[Dict, Dict]:
    key = hashlib.sha256(json.dumps([PROMPT_VERSION, MODEL, system, user, schema]).encode()).hexdigest()
    hit = artifacts.get(stage, key)
    if hit is not None:
        return hit["out"], {**hit["usage"], "cached": True}
    r = claude_client().beta.messages.create(
        model=MODEL, max_tokens=16000, system=system, messages=[{"role": "user", "content": user}],
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": schema}},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    if r.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"{stage} call stopped: {r.stop_reason}")
    out = json.loads(next(b.text for b in r.content if b.type == "text"))
    usage = {"model": r.model, "input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}
    artifacts.put(stage, key, {"out": out, "usage": usage})
    return out, {**usage, "cached": False}


def _list_schema(item_key: str, props: Dict) -> Dict:
    item = {"type": "object", "additionalProperties": False, "required": [item_key, *props],
            "properties": {item_key: {"type": "integer"}, **props}}
    return {"type": "object", "additionalProperties": False, "required": ["items"],
            "properties": {"items": {"type": "array", "items": item}}}


def _story_outline(moments: Dict) -> str:
    secs: Dict[str, List[str]] = {}
    for m in moments["moments"]:
        secs.setdefault(m.get("section") or OTHER, []).append(f"{m['id']}: {m['summary']}")
    return (f"Synopsis: {moments['synopsis']}\n\nSections:\n"
            + "\n".join(f"- {s}: " + " | ".join(v) for s, v in secs.items()))


def assign_sections(moments: Dict, captions: List[str]) -> Tuple[List[str], Dict]:
    """Section (or OTHER) per library shot, in the given order."""
    names = sorted({m.get("section") for m in moments["moments"] if m.get("section")})
    user = _story_outline(moments) + "\n\nShots:\n" + "\n".join(f"{k}. {c}" for k, c in enumerate(captions, 1))
    schema = _list_schema("shot", {"section": {"type": "string", "enum": names + [OTHER]}})
    out, usage = _call("broll_sections_call", SECTIONS_SYSTEM, user, schema)
    got = {int(x["shot"]): x["section"] for x in out["items"]}
    return [got.get(k, OTHER) for k in range(1, len(captions) + 1)], usage


def write_visuals(moments: Dict, lines: List[Dict]) -> Tuple[List[str], Dict]:
    """lines: [{text, moment: moment dict or None}] in story order -> a visual per line."""
    rows = []
    for k, l in enumerate(lines, 1):
        m = l.get("moment") or {}
        rows.append(f"{k}. [{m.get('section') or '-'} | {m.get('summary', '-')}] {l['text']}")
    user = f"{_story_outline(moments)}\n\nVoice-over lines in story order:\n" + "\n".join(rows)
    out, usage = _call("broll_visuals_call", VISUALS_SYSTEM, user, _list_schema("line", {"visual": {"type": "string"}}))
    got = {int(x["line"]): x["visual"].strip() for x in out["items"]}
    return [got.get(k) or lines[k - 1]["text"] for k in range(1, len(lines) + 1)], usage


def rank_candidates(query_emb: np.ndarray, pool_emb: np.ndarray, pool_sections: List[str],
                    line_sections: List[Optional[str]], k: int = TOP_K) -> List[List[int]]:
    """Per query: up to k pool indices, shots of the line's section first (each group by cosine)."""
    sims = query_emb @ pool_emb.T
    out = []
    for q, sec in enumerate(line_sections):
        order = list(np.argsort(-sims[q]))
        if sec:
            order = [j for j in order if pool_sections[j] == sec] + [j for j in order if pool_sections[j] != sec]
        out.append([int(j) for j in order[:k]])
    return out


def pick_shots(lines: List[Dict], visuals: List[str], candidates: List[List[int]], captions: List[str],
               sections: List[str]) -> Tuple[List[Optional[int]], Dict]:
    """Ranked choices per line from its candidates -> one pool index per line (first not used yet) or None."""
    ranked: List[List[int]] = []
    usage = {"calls": 0, "cached": 0, "input_tokens": 0, "output_tokens": 0}
    for s in range(0, len(lines), PICKS_PER_CALL):
        parts = []
        for r, q in enumerate(range(s, min(s + PICKS_PER_CALL, len(lines))), 1):
            m = lines[q].get("moment") or {}
            cands = "\n".join(f"  {j}. [{sections[c]}] {captions[c]}" for j, c in enumerate(candidates[q], 1))
            parts.append(f"Request {r}\nLine: {lines[q]['text']}\nStory: section {m.get('section') or '-'}; "
                         f"{m.get('summary', '-')}\nFootage wanted: {visuals[q]}\nCandidates:\n{cands}")
        schema = _list_schema("request", {"shots": {"type": "array", "items": {"type": "integer"}}})
        out, u = _call("broll_pick_call", PICK_SYSTEM, "\n\n".join(parts), schema)
        got = {int(x["request"]): x["shots"] for x in out["items"]}
        for r, q in enumerate(range(s, min(s + PICKS_PER_CALL, len(lines))), 1):
            ranked.append([candidates[q][j - 1] for j in got.get(r, []) if 1 <= j <= len(candidates[q])][:3])
        usage["calls"] += 1
        usage["cached"] += int(u["cached"])
        usage["input_tokens"] += u["input_tokens"]
        usage["output_tokens"] += u["output_tokens"]
    used, chosen = set(), []
    for q, choices in enumerate(ranked):
        c = next((j for j in choices if j not in used), None)
        if c is None:                    # something must play: best unused shot of the line's own section
            sec = (lines[q].get("moment") or {}).get("section")
            c = next((j for j in candidates[q] if j not in used and sec and sections[j] == sec), None)
        if c is not None:
            used.add(c)
        chosen.append(c)
    return chosen, usage


# ---------------------------------------------------------------- job: overlays on the compiled timeline

PICK_WITH_MODEL = False     # eval 2026-10-12: section-first top shot >= model pick on average, 0 decoys, no gaps
JOIN_GAP_FRAMES = 1.5       # voice-over pieces closer than this on the output clock form one stretch


def _slots(stretch: Dict, beats: List[Dict]) -> List[Tuple[float, float]]:
    """Tile the stretch with one slot per beat: boundaries halfway through the pauses between beats."""
    if not beats:
        return [(stretch["rec_in"], stretch["rec_out"])]
    cuts = [stretch["rec_in"]] + [(beats[k]["end"] + beats[k + 1]["start"]) / 2 for k in range(len(beats) - 1)] \
        + [stretch["rec_out"]]
    return list(zip(cuts[:-1], cuts[1:]))


def voice_over_stretches(timeline, words_by: Dict[str, list], needs: Dict[str, List[Tuple[float, float]]]) -> List[Dict]:
    """Parts of the compiled timeline whose sound is voice-over without a usable picture, on the OUTPUT clock:
    [{rec_in, rec_out, words: [{start, end, text, id}] (output clock)}]. needs: file -> source intervals."""
    fps, rec, out = timeline.fps, 0.0, []
    for seg in timeline.segments:
        dur = round((seg.src_out - seg.src_in) * fps) / fps
        kept = set(seg.word_ids)
        for a, b in needs.get(seg.source_file, []):
            lo, hi = max(a, seg.src_in), min(b, seg.src_in + dur)
            if hi - lo < 1 / fps:
                continue
            r0, r1 = rec + lo - seg.src_in, rec + hi - seg.src_in
            ws = [{"start": rec + w.start - seg.src_in, "end": rec + w.end - seg.src_in, "text": w.text, "id": w.id}
                  for w in words_by.get(seg.source_file, []) if w.id in kept and lo <= (w.start + w.end) / 2 < hi]
            if out and r0 - out[-1]["rec_out"] < JOIN_GAP_FRAMES / fps:
                out[-1]["rec_out"] = r1
                out[-1]["words"] += ws
            else:
                out.append({"rec_in": r0, "rec_out": r1, "words": ws})
        rec += dur
    return out


def fit_shot(shot: Dict, rec_in: float, rec_out: float) -> Dict:
    """Source range of a library shot for a slot: its middle when longer than the slot, else all of it
    (the render holds its last frame)."""
    length, slot = shot["end"] - shot["start"], rec_out - rec_in
    src_in = shot["start"] + max(0.0, (length - slot) / 2)
    return {"source_file": shot["file"], "src_in": round(src_in, 4), "src_out": round(min(shot["end"], src_in + slot), 4),
            "rec_in": round(rec_in, 4), "rec_out": round(rec_out, 4)}


def plan_overlays(timeline, grid, files: List[Dict], moments: Optional[Dict] = None) -> Tuple[List[Dict], Dict]:
    """B-roll over every part of the edit whose sound is voice-over without a usable picture (audio-only files;
    talk over an unusable picture). Library = every shot classified b_roll (footage.py) in the job's video
    files. Returns (overlays for CompiledTimeline.overlays, report {warnings, needs_footage, ...})."""
    from app.tasks import footage
    from app.tasks import shots as shot_index
    from app.utils.ffmpeg import get_video_duration, is_audio_only
    words_by: Dict[str, list] = {}
    for w in grid.words:
        words_by.setdefault(w.source_file, []).append(w)
    report: Dict = {"warnings": [], "needs_footage": [], "library": 0, "stretches": 0, "beats": 0,
                    "covered_sec": 0.0, "categories": {}}
    needs: Dict[str, List[Tuple[float, float]]] = {}
    lib: List[Dict] = []
    for f in files:
        name, path = f["filename"], f["path"]
        if f.get("audio_only", is_audio_only(path)):
            needs[name] = [(0.0, 1e9)]
            report["categories"][name] = "voice_over (audio only)"
            continue
        ix = shot_index.index_file(path)
        ws = [{"start": w.start, "end": w.end, "text": w.text} for w in words_by.get(name, [])]
        cat = footage.classify(ws, ix, get_video_duration(path))
        report["categories"][name] = cat["seconds"]
        emb = shot_index.load_embeddings(ix)
        for s, c in zip(ix["shots"], cat["shots"]):
            if c["category"] == "b_roll":
                lib.append({"file": name, "start": s["start"], "end": s["end"], "caption": s["caption"], "emb": emb[s["i"]]})
            elif c["category"] == "voice_over" and not s.get("usable", True):
                needs.setdefault(name, []).append((s["start"], s["end"]))
    report["library"] = len(lib)
    stretches = voice_over_stretches(timeline, words_by, needs)
    report["stretches"] = len(stretches)
    if not stretches:
        return [], report
    moments = moments or {"synopsis": "unknown", "moments": []}
    moment_of = {wid: m for m in moments["moments"] for wid in m["word_ids"]}
    beats, slots = [], []
    for st in stretches:
        bs = build_beats(build_lines(st["words"])) if st["words"] else []
        for b in bs:
            b["moment"] = moment_of.get(st["words"][b["a"]]["id"])
        slots += _slots(st, bs)
        beats += bs or [{"text": "", "moment": None}]
    report["beats"] = len(beats)
    chosen: List[Optional[int]] = [None] * len(beats)
    if lib:
        P = np.stack([e["emb"] for e in lib])
        captions = [e["caption"] for e in lib]
        visuals, _ = write_visuals(moments, beats)
        if moments["moments"]:
            sections, _ = assign_sections(moments, captions)
        else:
            sections = [OTHER] * len(lib)
        line_secs = [(b.get("moment") or {}).get("section") or None for b in beats]
        cands = rank_candidates(shot_index.embed_texts(visuals), P, sections, line_secs)
        if PICK_WITH_MODEL:
            chosen, _ = pick_shots(beats, visuals, cands, captions, sections)
        else:
            used = set()
            for q, c in enumerate(cands):
                chosen[q] = next((j for j in c if j not in used), None)
                if chosen[q] is not None:
                    used.add(chosen[q])
    overlays = []
    for b, (r0, r1), j in zip(beats, slots, chosen):
        if j is None:
            report["needs_footage"].append([round(r0, 2), round(r1, 2)])
            continue
        overlays.append({**fit_shot(lib[j], r0, r1), "caption": lib[j]["caption"], "beat": b["text"][:200],
                         "section": (b.get("moment") or {}).get("section") or ""})
        report["covered_sec"] += r1 - r0
    report["covered_sec"] = round(report["covered_sec"], 2)
    if overlays:
        report["warnings"].append(f"B-roll: {len(overlays)} shot(s) over {report['covered_sec']:.1f}s of voice-over.")
    if report["needs_footage"]:
        report["warnings"].append("Voice-over without footage (black picture) at " + ", ".join(
            f"{int(a // 60)}:{a % 60:04.1f}-{int(b // 60)}:{b % 60:04.1f}" for a, b in report["needs_footage"]))
    return overlays, report
