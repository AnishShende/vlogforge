"""Shot index (roadmap Phase 10 S1): what each stretch of a video file SHOWS, for B-roll.

Per source file:
  1. shots: hard cuts (PySceneDetect ContentDetector); one continuous take = one shot
  2. frames: 3 small JPEGs per shot (start, middle, end, with a margin)
  3. captions: Claude Haiku 4.5 looks at the frames of 10 shots per call and returns, per shot, a
     one-sentence caption, a kind (person to camera / people / place / object / action / text),
     whether someone talks to the camera, whether someone in the shot is talking at all (A-roll), and
     whether the picture is usable at all (B-roll vs Silent) — structured output, cached by the images
  4. embeddings: SigLIP 2 image embedding per frame, averaged per shot (L2-normalised), so a text
     ("a mountain lake at sunrise") can be matched to shots by cosine similarity (embed_texts)

Speech is NOT part of the index (it comes from the word grid / ASR of the job). Results are cached in
the artifact store by the file's CONTENT hash + VERSION; frames and embeddings live next to it under
artifact_dir/shot_index/<digest>/. Nothing here changes a job's edit or render.
"""

import base64
import hashlib
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.config import settings
from app.utils import artifacts
from app.utils.llm import claude_client

logger = logging.getLogger("VlogForge.Shots")

VERSION = "1"
CAPTION_MODEL = "claude-haiku-4-5"
CAPTION_PROMPT_VERSION = "3"
EMBED_MODEL = "google/siglip2-base-patch16-224"
FRAMES_PER_SHOT = 3
FRAME_HEIGHT = 252                 # 448x252 for 16:9: enough to see what a shot shows, cheap as an image
SHOTS_PER_CALL = 10
CUT_THRESHOLD = 27.0               # ContentDetector default
MIN_SHOT_SEC = 0.4
KINDS = ["person_to_camera", "people", "place", "object", "action", "text_or_graphic", "other"]

CAPTION_SYSTEM = """You describe shots of a video for a video editor who searches them later by text.
For every shot you get 3 frames (start, middle, end). For each shot return:
- caption: one plain sentence saying what the viewer SEES: the place, the main objects or people, what
  happens, the camera view (close-up, wide, aerial, moving). Concrete nouns, no guesses about story,
  no judgement words ("beautiful", "amazing").
- kind: the main subject, one of: person_to_camera (a person looks at and talks to the camera),
  people (people seen doing something, not addressing the camera), place (landscape, street, room,
  building), object (a thing in close view: food, product, gear), action (movement is the point:
  riding, lifting, driving, walking shot from behind), text_or_graphic (titles, maps, screens), other.
- to_camera: true when someone in the shot faces the camera and appears to be speaking to it.
- speaking: true when someone in the shot appears to be talking (mouth open mid-word across frames,
  talking gestures), whichever way they face: to the camera, to someone else, while driving or riding.
- usable: false only when the shot shows nothing a viewer would want to see: black or nearly black, fully
  out of focus or smeared, lens covered, the camera pointing at the floor, the sky or the inside of a
  pocket or bag by accident, extreme shake where nothing can be recognised. Plain or ordinary subjects
  (a wall, a road, a room) are still usable.
Return exactly one entry per shot, with the shot number you were given."""


# ---------------------------------------------------------------- shots + frames

def detect_shots(path: str) -> Tuple[List[Tuple[float, float]], float]:
    """Hard cuts -> [(start_sec, end_sec)], covering the whole file; and the file's fps."""
    from scenedetect import ContentDetector, detect, open_video
    fps = float(open_video(path).frame_rate)
    scenes = detect(path, ContentDetector(threshold=CUT_THRESHOLD), start_in_scene=True)
    shots = [(s.get_seconds(), e.get_seconds()) for s, e in scenes]
    merged: List[List[float]] = []
    for s, e in shots:                       # fold flash-frame fragments into the previous shot
        if merged and e - s < MIN_SHOT_SEC:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(round(s, 3), round(e, 3)) for s, e in merged], fps


def extract_frames(path: str, shots: List[Tuple[float, float]], fps: float, out_dir: str) -> List[List[str]]:
    """FRAMES_PER_SHOT JPEGs per shot (file names relative to out_dir), in shot order."""
    from scenedetect import FrameTimecode, open_video
    from scenedetect.scene_manager import save_images
    os.makedirs(out_dir, exist_ok=True)
    video = open_video(path)
    scene_list = [(FrameTimecode(s, fps), FrameTimecode(e, fps)) for s, e in shots]
    saved = save_images(scene_list, video, num_images=FRAMES_PER_SHOT, frame_margin=2, output_dir=out_dir,
                        image_name_template="s$SCENE_NUMBER-$IMAGE_NUMBER", height=FRAME_HEIGHT,
                        encoder_param=85)
    return [list(saved[i]) for i in sorted(saved)]


# ---------------------------------------------------------------- embeddings

_embed = None
_embed_lock = threading.RLock()     # also serialises inference (index_file may run in threads)


def _embedder():
    global _embed
    with _embed_lock:
        if _embed is None:
            import torch
            from transformers import AutoModel, AutoProcessor
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model = AutoModel.from_pretrained(EMBED_MODEL).to(device).eval()
            _embed = (model, AutoProcessor.from_pretrained(EMBED_MODEL), device)
    return _embed


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def embed_images(paths: List[str], batch: int = 32) -> np.ndarray:
    import torch
    from PIL import Image
    model, proc, device = _embedder()
    out = []
    for i in range(0, len(paths), batch):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
        with _embed_lock, torch.no_grad():
            inputs = proc(images=imgs, return_tensors="pt").to(device)
            out.append(model.get_image_features(**inputs).float().cpu().numpy())
    return _unit(np.concatenate(out)) if out else np.zeros((0, 768), np.float32)


def embed_texts(texts: List[str]) -> np.ndarray:
    import torch
    model, proc, device = _embedder()
    with _embed_lock, torch.no_grad():
        inputs = proc(text=texts, padding="max_length", max_length=64, truncation=True,
                      return_tensors="pt").to(device)
        return _unit(model.get_text_features(**inputs).float().cpu().numpy())


# ---------------------------------------------------------------- captions

def _caption_schema() -> Dict:
    shot = {"type": "object", "additionalProperties": False, "required": ["shot", "caption", "kind", "to_camera", "speaking", "usable"],
            "properties": {"shot": {"type": "integer"}, "caption": {"type": "string"},
                           "kind": {"type": "string", "enum": KINDS}, "to_camera": {"type": "boolean"},
                           "speaking": {"type": "boolean"}, "usable": {"type": "boolean"}}}
    return {"type": "object", "additionalProperties": False, "required": ["shots"],
            "properties": {"shots": {"type": "array", "items": shot}}}


def _caption_call(group: List[Tuple[int, List[str]]]) -> Tuple[Dict[int, Dict], Dict]:
    """group: [(shot number, [frame paths])] -> {shot number: {caption, kind, to_camera}}, usage."""
    content, h = [], hashlib.sha256()
    for n, frames in group:
        content.append({"type": "text", "text": f"Shot {n}:"})
        for p in frames:
            data = open(p, "rb").read()
            h.update(data)
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                        "data": base64.standard_b64encode(data).decode()}})
    numbers = [n for n, _ in group]
    content.append({"type": "text", "text": f"Describe shots {', '.join(map(str, numbers))}."})
    key = artifacts.make_key(CAPTION_PROMPT_VERSION, CAPTION_MODEL, CAPTION_SYSTEM, numbers, h.hexdigest())
    hit = artifacts.get("shots_caption_call", key)
    if hit is not None:
        return {int(k): v for k, v in hit["out"].items()}, {**hit["usage"], "cached": True}
    r = claude_client().messages.create(model=CAPTION_MODEL, max_tokens=4000, system=CAPTION_SYSTEM,
                                        messages=[{"role": "user", "content": content}],
                                        output_config={"format": {"type": "json_schema", "schema": _caption_schema()}})
    if r.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"caption call stopped: {r.stop_reason}")
    raw = json.loads(next(b.text for b in r.content if b.type == "text"))
    out = {int(s["shot"]): {"caption": s["caption"], "kind": s["kind"], "to_camera": bool(s["to_camera"]),
                            "speaking": bool(s["speaking"]), "usable": bool(s["usable"])}
           for s in raw["shots"] if s["shot"] in numbers}
    missing = set(numbers) - set(out)
    if missing:
        raise RuntimeError(f"caption call returned no entry for shots {sorted(missing)}")
    usage = {"model": r.model, "input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}
    artifacts.put("shots_caption_call", key, {"out": out, "usage": usage})
    return out, {**usage, "cached": False}


def caption_shots(frames: List[List[str]], workers: int = 4) -> Tuple[List[Dict], Dict]:
    """One caption entry per shot (shots numbered from 1 inside each call's prompt)."""
    groups = [[(i + 1, frames[i]) for i in range(s, min(s + SHOTS_PER_CALL, len(frames)))]
              for s in range(0, len(frames), SHOTS_PER_CALL)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_caption_call, groups))
    merged: Dict[int, Dict] = {}
    stats = {"calls": len(groups), "cached": 0, "input_tokens": 0, "output_tokens": 0}
    for out, usage in results:
        merged.update(out)
        stats["cached"] += int(usage["cached"])
        stats["input_tokens"] += usage["input_tokens"]
        stats["output_tokens"] += usage["output_tokens"]
    return [merged[i + 1] for i in range(len(frames))], stats


# ---------------------------------------------------------------- index

def index_dir(digest: str) -> str:
    return os.path.join(settings.artifact_dir, "shot_index", digest)


def load_embeddings(index: Dict) -> np.ndarray:
    return np.load(os.path.join(index_dir(index["digest"]), "emb.npy"))


def _base(path: str, digest: str) -> Dict:
    """Shots, frames and embeddings (cached by content; independent of the caption prompt)."""
    key = artifacts.make_key("shots_base", VERSION, digest, EMBED_MODEL)
    out_dir = index_dir(digest)
    hit = artifacts.get("shots_base", key)
    if hit is not None and os.path.exists(os.path.join(out_dir, "emb.npy")):
        return hit
    shots, fps = detect_shots(path)
    logger.info(f"[SHOTS] {os.path.basename(path)}: {len(shots)} shots")
    frames = extract_frames(path, shots, fps, out_dir)
    if len(frames) != len(shots):
        raise RuntimeError(f"frames for {len(frames)} of {len(shots)} shots")
    emb = embed_images([os.path.join(out_dir, f) for fs in frames for f in fs])
    shot_emb = _unit(emb.reshape(len(shots), FRAMES_PER_SHOT, -1).mean(axis=1))
    np.save(os.path.join(out_dir, "emb.npy"), shot_emb.astype(np.float32))
    base = {"fps": fps, "shots": [{"i": i, "start": s, "end": e, "frames": frames[i]}
                                  for i, (s, e) in enumerate(shots)]}
    artifacts.put("shots_base", key, base)
    return base


def index_file(path: str) -> Dict:
    """Shot index of one video file (cached by content). Shots: {i, start, end, frames, caption,
    kind, to_camera, speaking, usable}; frame paths are relative to index_dir(digest); embeddings: load_embeddings()."""
    digest = artifacts.file_digest(path)
    key = artifacts.make_key("shots", VERSION, digest, EMBED_MODEL, CAPTION_MODEL, CAPTION_PROMPT_VERSION)
    hit = artifacts.get("shots", key)
    if hit is not None and os.path.exists(os.path.join(index_dir(digest), "emb.npy")):
        return hit
    base = _base(path, digest)
    captions, stats = caption_shots([[os.path.join(index_dir(digest), f) for f in s["frames"]] for s in base["shots"]])
    index = {"version": VERSION, "digest": digest, "file": os.path.basename(path), "fps": base["fps"],
             "embed_model": EMBED_MODEL, "caption_model": CAPTION_MODEL, "caption_stats": stats,
             "shots": [{**s, **captions[s["i"]]} for s in base["shots"]]}
    artifacts.put("shots", key, index)
    logger.info(f"[SHOTS] {os.path.basename(path)}: indexed ({stats})")
    return index
