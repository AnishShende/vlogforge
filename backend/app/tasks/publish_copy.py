"""Post copy for the finished video (export panel): what the creator pastes when publishing.

Built from the LAST RENDER (the final cut's words in output order + the moment table), so it
describes what viewers will actually see. Format by length unless the creator overrides it:
  short (< 60 s): Instagram Reel - 3 caption variations + a grouped hashtag block
  long  (>= 60 s): YouTube - 3 title styles, 3-part description, pinned comments, chapters, hashtags
The task prompts are the creator's own (2026-10-08), with their [placeholders] filled from the final
cut (audience / tone / keyword are inferred by the model and returned, so the panel can show them).
One model call. Chapter TIMES are code: the model picks which moments open a chapter and names
them; code takes each moment's output time and enforces YouTube's chapter rules (first at 0:00,
>= 3 chapters, each >= 10 s), else it returns no chapters and says why. The description's link
placeholders are fixed text written by code.
Cached per render + format + variant + prompt version, so a re-compile gets fresh copy.
Background: schedule_background() writes the auto-format copy post_copy_delay_sec after the last
render of a job (a newer render restarts the wait); one call per (job, format) at a time, so the
panel opening mid-call waits for that call instead of making a second one."""

import hashlib
import json
import logging
import re
import threading
from typing import Dict, List, Optional, Tuple

import anthropic

from app.config import settings
from app.tasks.recompile import load_job_edit
from app.utils import artifacts

logger = logging.getLogger("VlogForge.PublishCopy")

PROMPT_VERSION = "2"
MODEL = "claude-sonnet-5-5"
SHORT_MAX_SEC = 60.0
CHAPTER_MIN_SEC = 10.0      # YouTube: each chapter >= 10 s, at least 3, first at 0:00
CHAPTER_MIN_COUNT = 3
HASHTAG_GROUPS = {"niche": 5, "content": 5, "broad": 3}     # Instagram block: 10-15 tags
YT_HASHTAGS = 5
VARIATIONS = [("value_drop", "The Direct Value-Drop", "Best for Carousels/Graphics"),
              ("micro_story", "The Micro-Story", "Best for Reels/Lifestyle Photos"),
              ("short_punchy", "Short & Punchy", "Best for Fast-Paced Reels")]
TITLE_STYLES = [("curiosity", "The Curiosity Gap", "High CTR, under 50 characters"),
                ("seo", "SEO-Heavy", "Long-tail search, keyword first"),
                ("stakes", "The High Stakes / Urgency", "Mistake to avoid or transformation")]
LINKS = """- Lead Magnet/Newsletter Opt-in: [Link]
- Related Video/Playlist to keep them on the channel: [Link]
- Affiliate Links/Gear Used: [Link]
- Subscribe Nudge: [Link]"""

# Grounding for every call: the creator's prompts below say what to write; this says what it may claim.
SYSTEM = """You write the post copy a creator pastes when they publish their video. You get the final
cut's transcript (what viewers will hear, in order, with timestamps), its synopsis and its story moments.
- Write in the creator's own voice and language: match the transcript (if they speak Hinglish, write
  natural Hinglish in Latin script).
- Only promise what the video delivers: no facts, numbers or claims that are not in the transcript.
- Hashtags: lowercase, no spaces; never generic filler such as viral, fyp, trending, explore.
- Fill the JSON fields; where the brief below says "format your response exactly like this", that
  layout maps onto the fields (the app lays them out)."""

# The creator's Instagram prompt, placeholders filled.
SHORT_BRIEF = """Act as an expert Instagram social media strategist and copywriter. I need you to write 3 distinct caption variations and a strategic hashtag block for an upcoming Instagram post.

Here is the context of the post:
- Topic/Core Message: {topic}
- Content Format: Reel
- Target Audience: infer it from the transcript and the creator's goal ({goal}); return it as `audience`
- Tone of Voice: choose 1-2 that match how the creator speaks in the transcript; return them as `tone`

Please format your response exactly like this (as the JSON fields `variations` in this order, then `hashtags`):

### VARIATION 1: The Direct Value-Drop (Best for Carousels/Graphics)
• Hook: A scroll-stopping first line under 10 words. Must create a curiosity gap or call out a major pain point. Do NOT start with "Are you tired of..." or generic cliches.
• Body: 2–3 short, punchy sentences explaining the core tip or value. Use line breaks for readability.
• CTA: A specific, low-friction action (e.g., "Save this for your next session" or "Comment 'GUIDE' and I'll DM you the link").

### VARIATION 2: The Micro-Story (Best for Reels/Lifestyle Photos)
• Hook: An emotional, relatable, or intriguing statement that sets up a quick story or transformation.
• Body: A short, narrative-driven paragraph (under 4 sentences) detailing a quick realization, lesson, or behind-the-scenes moment.
• CTA: A conversation-starting question to drive comments (e.g., "Have you ever experienced this? Drop your thoughts below").

### VARIATION 3: Short & Punchy (Best for Fast-Paced Reels)
• Hook: An ultra-short statement that forces them to finish watching the video to understand it.
• Body: 1 single, high-impact sentence reinforcing the video's message.
• CTA: A clear instruction (e.g., "Read that again and hit follow for daily tips").

### HASHTAG STRATEGY
Provide a block of 10-15 highly relevant hashtags, grouped together at the very bottom. Do not mix them into the caption text. Structure the mix as follows:
- 4-5 Niche/Target Audience tags (highly specific to your ideal viewer)
- 4-5 Content Specific tags (exactly what the post topic is about)
- 2-3 Industry/Broad tags (larger categories for SEO context)"""

# The creator's YouTube prompt, placeholders filled; timestamps / links / chapters adapted (see notes).
LONG_BRIEF = """Act as an expert YouTube growth strategist and SEO specialist. I need you to create the metadata for an upcoming video.

Here is the context of the video:
- Video Topic/Core Subject: {topic}
- Target Audience: infer it from the transcript; return it as `audience`
- Main Value Proposition: {goal}
- Primary Focus Keyword: the main phrase people would type into YouTube search to find this; infer it and return it as `keyword`

Please format your response exactly like this (as the JSON fields):

### TITLE OPTIMIZATION (Provide 3 Styles) -> `titles`
- Style 1 (The Curiosity Gap - High CTR): A title under 50 characters that creates intense curiosity or challenges a common belief.
- Style 2 (SEO-Heavy - Long-tail Search): A title that naturally prioritizes the primary keyword at the very beginning.
- Style 3 (The High Stakes/Urgency): A title focusing on a major mistake to avoid or an immediate transformation.

### THE 3-PART DESCRIPTION FLOW
Part 1: The Above-The-Fold Hook (First 2 sentences) -> `description_hook`
Write a compelling 2-sentence summary that repeats the primary keyword naturally. This must entice users clicking from Google or YouTube search results before they hit "Show More".

Part 2: Deep SEO Outline & Context -> `description_outline`
Write a comprehensive 2-3 paragraph overview of what the video covers. Naturally weave in secondary, highly related search terms. Do not write timestamps or [00:00] placeholders: the real chapter timeline is inserted after this part by the app.

Part 3: Multi-Link Channel Ecosystem
The app adds the link placeholders itself; do not write them.

### ENGAGEMENT ACCELERATOR (Pinned Comment) -> `pinned_comments`
Provide 2 options for a pinned comment designed to drive high-retention comments immediately after publishing (e.g., asking an polarizing question or offering an exclusive bonus resource).

### CHAPTERS -> `chapters`
The moments where a chapter starts, by moment number, with a plain label under 40 characters each. Start with moment 1. Group small moments: a chapter should be a real section.

### HASHTAGS -> `hashtags`
3-5 hashtags for the end of the description: one or two broad, the rest specific to this video."""


def _schema(fmt: str) -> Dict:
    strs = {"type": "array", "items": {"type": "string"}}
    obj = lambda props: {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}
    text = {"type": "string"}
    if fmt == "short":
        return obj({"audience": text, "tone": text,
                    "variations": {"type": "array", "items": obj({"hook": text, "body": text, "cta": text})},
                    "hashtags": obj({"niche": strs, "content": strs, "broad": strs})})
    return obj({"audience": text, "keyword": text,
                "titles": obj({k: text for k, _, _ in TITLE_STYLES}),
                "description_hook": text, "description_outline": text, "pinned_comments": strs,
                "chapters": {"type": "array", "items": obj({"moment": {"type": "integer"}, "label": text})},
                "hashtags": strs})


_client: Optional[anthropic.Anthropic] = None
_guard = threading.Lock()
_locks: Dict[Tuple[str, str], threading.Lock] = {}    # one copy call per (job, format) at a time
_timers: Dict[str, threading.Timer] = {}             # pending background writes, per job


def _call(fmt: str, user: str, variant: int) -> Tuple[Dict, Dict]:
    system = SYSTEM
    if variant:
        system += f"\n\nThis is alternative #{variant + 1}: take a clearly different angle from the obvious one."
    key = hashlib.sha256(json.dumps([PROMPT_VERSION, MODEL, system, user]).encode()).hexdigest()
    hit = artifacts.get("publish_call", key)
    if hit is not None:
        return hit["out"], {**hit["usage"], "cached": True}
    global _client
    _client = _client or anthropic.Anthropic(api_key=settings.claude_api_key)
    r = _client.beta.messages.create(model=MODEL, max_tokens=8000, system=system,
                                     messages=[{"role": "user", "content": user}],
                                     output_config={"effort": "low", "format": {"type": "json_schema", "schema": _schema(fmt)}},
                                     betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    if r.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"post copy call stopped: {r.stop_reason}")
    out = json.loads(next(b.text for b in r.content if b.type == "text"))
    usage = {"model": r.model, "input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}
    artifacts.put("publish_call", key, {"out": out, "usage": usage})
    return out, {**usage, "cached": False}


def stamp(sec: float) -> str:
    """YouTube timestamp: M:SS, or H:MM:SS from an hour."""
    s = int(sec)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def _final_cut(job_id: str) -> Optional[Dict]:
    """The last render as the model sees it: transcript lines and moments, both with output times."""
    view = load_job_edit(job_id)
    if view is None or not view.get("segments"):
        return None
    by_id = {w["id"]: w for w in view["words"]}
    lines = [f"[{stamp(s['rec_in'])}] " + " ".join(by_id[w]["text"] for w in s["word_ids"]) for s in view["segments"]]
    mo = artifacts.load_job(job_id, "moments") or {}
    moments = []
    for m in mo.get("moments", []):
        times = [view["word_out"][w] for w in m["word_ids"] if w in view["word_out"]]
        if times:
            moments.append({"start": min(times), "function": m["function"], "summary": m["summary"]})
    moments.sort(key=lambda m: m["start"])
    if not moments:                                   # no moment table: each clip is a candidate section
        moments = [{"start": s["rec_in"], "function": "clip", "summary": short_text(" ".join(by_id[w]["text"] for w in s["word_ids"]))}
                   for s in view["segments"]]
    return {"duration": view["duration_sec"], "lines": lines, "moments": moments,
            "synopsis": mo.get("synopsis", ""), "goal": mo.get("creator_goal", ""),
            "fingerprint": hashlib.sha256(json.dumps([[(s["word_ids"], s["rec_in"], s["rec_out"]) for s in view["segments"]],
                                                      [m["summary"] for m in moments]]).encode()).hexdigest()[:16]}


def short_text(t: str, n: int = 80) -> str:
    return t if len(t) <= n else t[: n - 1] + "…"


def _hashtags(raw: List[str], cap: int, seen: Optional[set] = None) -> List[str]:
    """'#tag' list: lowercase, letters/digits/_ only, de-duplicated (also against `seen`), at most cap."""
    seen = set() if seen is None else seen
    out = []
    for h in raw:
        tag = re.sub(r"[^\w]", "", str(h).lower().lstrip("#"))
        if tag and tag not in seen and len(out) < cap:
            seen.add(tag); out.append("#" + tag)
    return out


def _chapters(raw: List[Dict], moments: List[Dict], duration: float) -> Tuple[List[Dict], Optional[str]]:
    """Model-picked moments -> timed chapters that YouTube accepts, or [] and the reason."""
    picks = {}
    for c in raw:
        k = c.get("moment")
        if isinstance(k, int) and 1 <= k <= len(moments) and k not in picks:
            picks[k] = short_text(str(c.get("label", "")).strip(), 40)
    if not picks:
        return [], "no chapters suggested"
    out = []
    for k in sorted(picks):
        t = 0.0 if not out else moments[k - 1]["start"]
        if out and (t - out[-1]["sec"] < CHAPTER_MIN_SEC or duration - t < CHAPTER_MIN_SEC):
            continue                                   # too close to the previous chapter or the end
        out.append({"sec": round(t, 2), "time": stamp(t), "label": picks[k]})
    if len(out) < CHAPTER_MIN_COUNT:
        return [], f"only {len(out)} section(s) of 10 s or more; YouTube needs at least {CHAPTER_MIN_COUNT} for chapters"
    return out, None


def publish_copy(job_id: str, fmt: str = "auto", regenerate: bool = False) -> Optional[Dict]:
    """Post copy for the last render. fmt: auto | short | long. None = no render stored."""
    cut = _final_cut(job_id)
    if cut is None:
        return None
    auto = "short" if cut["duration"] < SHORT_MAX_SEC else "long"
    fmt = auto if fmt == "auto" else fmt
    with _guard:
        lock = _locks.setdefault((job_id, fmt), threading.Lock())
    with lock:                                        # a caller arriving mid-call gets that call's result
        return {**_write(job_id, fmt, cut, regenerate), "auto_format": auto}


def _write(job_id: str, fmt: str, cut: Dict, regenerate: bool) -> Dict:
    stored = artifacts.load_job(job_id, f"publish_{fmt}")
    if stored and stored["fingerprint"] == cut["fingerprint"] and stored.get("prompt_version") == PROMPT_VERSION and not regenerate:
        return stored
    variant = (stored["variant"] + 1) if (stored and regenerate) else 0
    numbered = "\n".join(f"{k}. ({stamp(m['start'])}) {m['function']}: {m['summary']}" for k, m in enumerate(cut["moments"], 1))
    brief = (SHORT_BRIEF if fmt == "short" else LONG_BRIEF).format(
        topic=cut["synopsis"] or "see the transcript below", goal=cut["goal"] or "infer it from the transcript")
    user = (f"{brief}\n\n---\n## The video\nLength: {stamp(cut['duration'])} ({cut['duration']:.0f} s)\n"
            f"Synopsis: {cut['synopsis'] or '(none)'}\nCreator's goal: {cut['goal'] or '(none)'}\n\n"
            f"Moments (number, start time in the video, function: summary):\n{numbered}\n\n"
            f"Final cut transcript:\n" + "\n".join(cut["lines"]))
    raw, usage = _call(fmt, user, variant)
    result = {"format": fmt, "duration_sec": cut["duration"], "fingerprint": cut["fingerprint"], "variant": variant,
              "prompt_version": PROMPT_VERSION, "audience": raw.get("audience", "").strip(), "usage": usage}
    if fmt == "short":
        vs = [v for v in raw.get("variations", []) if any(str(v.get(k, "")).strip() for k in ("hook", "body", "cta"))]
        if len(vs) < len(VARIATIONS):
            logger.warning(f"[PUBLISH] job {job_id}: model returned {len(vs)} caption variations, expected {len(VARIATIONS)}")
        result["tone"] = raw.get("tone", "").strip()
        result["variations"] = [{"key": k, "name": n, "best_for": b, **{f: str(v.get(f, "")).strip() for f in ("hook", "body", "cta")}}
                                for (k, n, b), v in zip(VARIATIONS, vs)]
        seen: set = set()
        result["hashtag_groups"] = {g: _hashtags(raw.get("hashtags", {}).get(g, []), cap, seen) for g, cap in HASHTAG_GROUPS.items()}
        result["hashtags"] = [t for g in HASHTAG_GROUPS for t in result["hashtag_groups"][g]]
    else:
        chapters, note = _chapters(raw.get("chapters", []), cut["moments"], cut["duration"])
        result.update(keyword=raw.get("keyword", "").strip(),
                      titles=[{"key": k, "name": n, "hint": h, "text": str(raw.get("titles", {}).get(k, "")).strip()[:100]} for k, n, h in TITLE_STYLES],
                      description_hook=raw.get("description_hook", "").strip(), description_outline=raw.get("description_outline", "").strip(),
                      links=LINKS, pinned_comments=[c.strip() for c in raw.get("pinned_comments", []) if c.strip()][:2],
                      chapters=chapters, chapter_note=note, hashtags=_hashtags(raw.get("hashtags", []), YT_HASHTAGS))
    artifacts.save_job(job_id, f"publish_{fmt}", result)
    detail = (f", {len(result['variations'])} variations" if fmt == "short"
              else f", {len(result['chapters'])} chapters {result['chapter_note'] or ''}")
    logger.info(f"[PUBLISH] job {job_id}: {fmt} copy v{variant} for {cut['duration']:.1f}s "
                f"({len(result['hashtags'])} hashtags{detail}); usage {usage}")
    return result


def schedule_background(job_id: str) -> None:
    """Write the auto-format post copy after the last render settles (restarts on every render)."""
    if not settings.enable_background_post_copy:
        return
    with _guard:
        old = _timers.pop(job_id, None)
        if old:
            old.cancel()
        t = threading.Timer(settings.post_copy_delay_sec, _background, args=(job_id,))
        t.daemon = True
        _timers[job_id] = t
        t.start()
    logger.info(f"[PUBLISH] job {job_id}: post copy scheduled in {settings.post_copy_delay_sec:.0f}s")


def _background(job_id: str) -> None:
    with _guard:
        _timers.pop(job_id, None)
    try:
        c = publish_copy(job_id)
        logger.info(f"[PUBLISH] job {job_id}: background post copy ready ({c['format']}, cached={c['usage'].get('cached', False)})")
    except Exception as e:                            # the panel retries on open and shows the error
        logger.warning(f"[PUBLISH] job {job_id}: background post copy failed: {e}")
