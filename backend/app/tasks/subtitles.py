"""Burned-in subtitles (export panel): the kept words of the last render, in fixed blocks on the output clock.

Block k covers [k * BLOCK_SEC, (k + 1) * BLOCK_SEC) of the output and shows every kept word whose midpoint
falls inside it; a block without words shows nothing. Words are placed with the same mapping as the editor
(output time = segment's whole-frame output offset + offset in the source), so captions follow the cuts.
One style preset per render (STYLES). The ASS file is sized to the output frame (PlayRes = frame size), so
a preset reads the same in every layout; app.utils.ffmpeg.render_compiled burns it in with libass.
A job's placement (moved / resized in the editor's caption overlay) applies to every preset and layout:
scale multiplies the preset's size, x / y put the bottom centre of the caption at that fraction of the frame
(None = the preset's own place)."""

import hashlib
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from app.utils import artifacts

logger = logging.getLogger("VlogForge.Subtitles")

BLOCK_SEC = 1.0
VERSION = "1"                 # bump when the ASS output changes for the same words and style

# size / outline / shadow: fraction of the frame's short side; margin_v: fraction of its height.
# Colours are RGB hex; box_alpha 0 = opaque .. 255 = invisible. em_height: the font's ascent + descent in em
# (OS/2 win metrics). libass makes that the font size, a browser makes the em the font size, so the editor's
# live overlay divides by it to draw the same size as the burned-in text.
STYLES: Dict[str, Dict] = {
    "clean": {"name": "Clean", "description": "White, black outline, bottom",
              "font": "Arial", "em_height": 1.117, "bold": True, "italic": False, "upper": False, "pop": False,
              "size": 0.052, "colour": "FFFFFF", "outline_colour": "000000", "outline": 0.0045, "shadow": 0.0015,
              "box": False, "box_alpha": 0, "margin_v": 0.07},
    "bold": {"name": "Bold", "description": "Big yellow caps that pop in, Reels style",
             "font": "Arial Black", "em_height": 1.410, "bold": False, "italic": False, "upper": True, "pop": True,
             "size": 0.066, "colour": "FFE14D", "outline_colour": "000000", "outline": 0.0075, "shadow": 0.003,
             "box": False, "box_alpha": 0, "margin_v": 0.16},
    "boxed": {"name": "Boxed", "description": "White on a dark box, bottom",
              "font": "Helvetica Neue", "em_height": 1.192, "bold": True, "italic": False, "upper": False, "pop": False,
              "size": 0.048, "colour": "FFFFFF", "outline_colour": "000000", "outline": 0.008, "shadow": 0.0,
              "box": True, "box_alpha": 0x50, "margin_v": 0.07},
    "cinema": {"name": "Cinema", "description": "Soft serif italic, low and small",
               "font": "Georgia", "em_height": 1.136, "bold": False, "italic": True, "upper": False, "pop": False,
               "size": 0.044, "colour": "F5F0E6", "outline_colour": "000000", "outline": 0.002, "shadow": 0.003,
               "box": False, "box_alpha": 0, "margin_v": 0.05},
}
PORTRAIT_MARGIN_V = 0.2       # tall frames: keep clear of the Reels / Shorts / TikTok buttons at the bottom
SIDE_MARGIN = 0.08            # fraction of the frame width on each side
SCALE_RANGE = (0.4, 3.0)
DEFAULT_PLACEMENT = {"scale": 1.0, "x": None, "y": None}


def clean_placement(p: Optional[Dict]) -> Dict:
    """Placement within bounds: scale in SCALE_RANGE, x / y both set inside the frame or both None."""
    p = {**DEFAULT_PLACEMENT, **(p or {})}
    out = {"scale": round(min(max(float(p["scale"]), SCALE_RANGE[0]), SCALE_RANGE[1]), 3), "x": None, "y": None}
    if p["x"] is not None and p["y"] is not None:
        out["x"], out["y"] = round(min(max(float(p["x"]), 0.0), 1.0), 4), round(min(max(float(p["y"]), 0.0), 1.0), 4)
    return out


def job_placement(job_id: str) -> Dict:
    return clean_placement(artifacts.load_job(job_id, "caption_placement"))


def save_placement(job_id: str, p: Dict) -> Dict:
    p = clean_placement(p)
    artifacts.save_job(job_id, "caption_placement", p)
    return p


def output_words(timeline: Dict, grid_words: List[Dict]) -> List[Dict]:
    """Kept words on the output clock: [{text, start, end}], clipped to their segment."""
    by_id = {w["id"]: w for w in grid_words}
    fps, t, out = timeline["fps"], 0.0, []
    for s in timeline["segments"]:
        dur = round((s["src_out"] - s["src_in"]) * fps) / fps
        for wid in s["word_ids"]:
            w = by_id.get(wid)
            if w is None:
                raise KeyError(f"timeline word {wid!r} is not in the grid")
            a = min(max(w["start"] - s["src_in"], 0.0), dur)
            b = min(max(w["end"] - s["src_in"], a), dur)
            out.append({"text": w["text"], "start": t + a, "end": t + b})
        t += dur
    return out


def _tidy(text: str) -> str:
    """Aligner words are lowercase without punctuation: only the pronoun I is capitalised."""
    return re.sub(r"\bi(?=$|\s|')", "I", text)


def blocks(words: List[Dict], duration: float, block_sec: float = BLOCK_SEC) -> List[Tuple[float, float, str]]:
    """Fixed blocks with words: [(start, end, text)] on the output clock, the last one ending at duration."""
    texts: Dict[int, List[str]] = {}
    for w in words:
        k = int(((w["start"] + w["end"]) / 2) // block_sec)
        texts.setdefault(k, []).append(w["text"].strip())
    out = []
    for k in sorted(texts):
        a, b = k * block_sec, min((k + 1) * block_sec, duration)
        text = _tidy(" ".join(x for x in texts[k] if x))
        if b > a and text:
            out.append((a, b, text))
    return out


def _colour(rgb: str, alpha: int = 0) -> str:
    return f"&H{alpha:02X}{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}".upper()


def _ts(t: float) -> str:
    cs = round(t * 100)
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _escape(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


def build_ass(cues: List[Tuple[float, float, str]], style: str, size: Tuple[int, int],
              placement: Optional[Dict] = None) -> str:
    """ASS file for the cues in one preset, laid out for a w x h frame, at the job's placement."""
    st, pl = STYLES[style], clean_placement(placement)
    w, h = size
    short = min(w, h) * pl["scale"]
    margin_v = max(st["margin_v"], PORTRAIT_MARGIN_V) if h > w else st["margin_v"]
    back = _colour("000000", st["box_alpha"]) if st["box"] else _colour("000000", 0x80)
    style_line = ",".join(str(x) for x in [
        "Default", st["font"], round(st["size"] * short), _colour(st["colour"]), _colour(st["colour"]),
        _colour(st["outline_colour"]), back, -1 if st["bold"] else 0, -1 if st["italic"] else 0, 0, 0, 100, 100, 0, 0,
        3 if st["box"] else 1, round(st["outline"] * short, 1), round(st["shadow"] * short, 1), 2,
        round(SIDE_MARGIN * w), round(SIDE_MARGIN * w), round(margin_v * h), 1])
    pop = r"\fscx80\fscy80\t(0,90,\fscx100\fscy100)" if st["pop"] else ""
    pos = rf"\an2\pos({round(pl['x'] * w)},{round(pl['y'] * h)})" if pl["x"] is not None else ""
    tags = "{" + pos + pop + "}" if pos or pop else ""
    events = [f"Dialogue: 0,{_ts(a)},{_ts(b)},Default,,0,0,0,,{tags}{_escape(text.upper() if st['upper'] else text)}"
              for a, b, text in cues]
    return "\n".join([
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {w}", f"PlayResY: {h}", "WrapStyle: 0",
        "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: {style_line}", "",
        "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        *events, ""])


def job_cues(job_id: str) -> Optional[List[Tuple[float, float, str]]]:
    """Caption blocks of a job's last render. None = no stored render."""
    timeline, grid = artifacts.load_job(job_id, "timeline"), artifacts.load_job(job_id, "grid")
    if timeline is None or grid is None:
        return None
    return blocks(output_words(timeline, grid["words"]), timeline["duration_sec"])


def fingerprint(cues: List[Tuple[float, float, str]], style: str, placement: Optional[Dict] = None) -> str:
    """Changes when the words, their timing, the preset or the placement change (a captioned file then shows
    as outdated)."""
    return hashlib.sha256(json.dumps([VERSION, BLOCK_SEC, STYLES[style], cues, clean_placement(placement)]).encode()).hexdigest()[:16]
