"""ASR bake-off (Archdoc roadmap Phase 1 / S4).

Every candidate yields a transcript for each eval clip; each goes through the
same word-grid build, speech-activity checks and gold scoring (eval.grid_eval).

Candidates (local only, user decision 2026-10-05):
  A  turbo, production settings (temperature fallback ON) + WhisperX align
  B  turbo, temperature 0 (no fallback)                    + align
  C  B + disfluent initial_prompt                          + align
  D  large-v3, temperature 0                               + align
  E  B, then re-transcribe each uncovered speech run alone and merge (recovered words)
  H1 smallest.ai Pulse Pro (English-only), its own word timestamps      [hosted]
  H1a H1 text, re-timed by our WhisperX forced alignment                [hosted]
  H2 smallest.ai Pulse, per-clip language (manifest `asr_language`)    [hosted]
  F_turbo / F_large  CrisperWhisper 2.0 (turbo / large), precomputed in its own venv by
     eval/external/crisperwhisper_runner.py -> eval-set/_work/asr/F_<model>.json
     [evaluation only: weights under Nyra Health Non-Commercial Research License]
Hosted candidates upload the clip audio (user OK 2026-10-05, key SMALLEST_API_KEY).

Scored per clip: gold speech without words (primary), boundary error, flags,
words outside every gold span (hallucination proxy), runtime, determinism
(--repeat 2: identical words across runs).

Usage (from backend/, ffmpeg on PATH):
    PYTHONPATH=. python -m eval.asr_bakeoff [--cand A --cand B ...] [--repeat 2]
"""

import argparse
import copy
import json
import os
import time
from datetime import datetime
from typing import Dict, List

from app.config import settings
from app.models import WORD_FLAG_RECOVERED
from app.tasks.transcribe import _forced_align
from app.tasks.word_grid import build_word_grid, check_word_grid, summarize_word_grid, word_coverage
from app.utils.speech_activity import SR, envelope, load_audio, uncovered_speech
from eval.gold import EVAL_SET_DIR, REPO_ROOT, all_spans, load_gold, load_manifest
from eval.grid_eval import clip_audio, grid_vs_gold

DISFLUENT_PROMPT = "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking. So so the the, I- I mean, uh, yeah."
# Production decode settings (app/tasks/transcribe.py) shared by every candidate.
BASE = dict(beam_size=5, condition_on_previous_text=False, word_timestamps=True, vad_filter=True,
            vad_parameters=dict(min_speech_duration_ms=100, min_silence_duration_ms=500, speech_pad_ms=400))
# Standard faster-whisper hallucination guards, applied to E's snippet re-transcriptions.
NO_SPEECH_MAX = 0.6
AVG_LOGPROB_MIN = -1.0

CANDIDATES = {
    "A": dict(model="turbo", decode={}),
    "B": dict(model="turbo", decode={"temperature": 0.0}),
    "C": dict(model="turbo", decode={"temperature": 0.0, "initial_prompt": DISFLUENT_PROMPT}),
    "D": dict(model="large-v3", decode={"temperature": 0.0}),
    "E": dict(model="turbo", decode={"temperature": 0.0}, recover=True),
    "H1": dict(api="smallest", model="pulse-pro", decode={}),
    "H1a": dict(api="smallest", model="pulse-pro", decode={}, realign=True),
    "H2": dict(api="smallest", model="pulse", decode={}),
    "F_turbo": dict(file="F_turbo", model="crisperwhisper2.0-turbo", decode={}),
    "F_large": dict(file="F_large", model="crisperwhisper2.0-large", decode={}),
}
ASR_WORK = os.path.join(EVAL_SET_DIR, "_work", "asr")


def _precomputed(wav: str, cand: dict, run_idx: int) -> List[Dict]:
    """Words from an external runner's JSON, matched by wav file name."""
    data = json.load(open(os.path.join(ASR_WORK, cand["file"] + ".json")))["results"]
    runs = next((v for k, v in data.items() if os.path.basename(k) == os.path.basename(wav)), None)
    if runs is None:
        raise RuntimeError(f"{cand['file']}: no precomputed transcript for {os.path.basename(wav)}")
    r = runs[run_idx % len(runs)]
    cand["_runtime"] = cand.get("_runtime", 0.0) + r["runtime_sec"]
    return r["words"]
SMALLEST_URL = "https://api.smallest.ai/waves/v1/stt/"
_models: Dict[str, object] = {}


def _model(name: str):
    from faster_whisper import WhisperModel
    if name not in _models:
        _models[name] = WhisperModel(name, device="auto", compute_type="default")
    return _models[name]


def _smallest(wav: str, cand: dict, language: str) -> List[Dict]:
    """smallest.ai pre-recorded STT. Returns [{start, end, text, conf?}] as delivered."""
    import requests
    from dotenv import dotenv_values
    key = os.environ.get("SMALLEST_API_KEY") or dotenv_values(os.path.join(REPO_ROOT, ".env")).get("SMALLEST_API_KEY")
    if not key:
        raise RuntimeError("SMALLEST_API_KEY not set")
    lang = "en" if cand["model"] == "pulse-pro" else language
    with open(wav, "rb") as f:
        r = requests.post(SMALLEST_URL, params={"model": cand["model"], "language": lang, "word_timestamps": "true"},
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/octet-stream"},
                          data=f.read(), timeout=600)
    r.raise_for_status()
    body = r.json()
    if body.get("status") != "success":
        raise RuntimeError(f"smallest.ai error: {str(body)[:300]}")
    out = []
    for w in body.get("words", []):
        text = str(w.get("word", w.get("text", ""))).strip()
        if not text or w.get("start") is None:
            continue
        e = {"start": float(w["start"]), "end": float(w["end"]), "text": text}
        if w.get("confidence") is not None:
            e["conf"], e["conf_src"] = round(float(w["confidence"]), 4), "smallest_confidence"
        out.append(e)
    return out


def _segments_from_words(words: List[Dict], gap: float = 0.5) -> List[Dict]:
    segs: List[Dict] = []
    for w in words:
        if segs and w["start"] - segs[-1]["end"] < gap:
            segs[-1]["end"], segs[-1]["text"] = w["end"], segs[-1]["text"] + " " + w["text"]
        else:
            segs.append({"start": w["start"], "end": w["end"], "text": w["text"]})
    return segs


def transcribe(wav: str, cand: dict, language: str = "en", run_idx: int = 0) -> List[Dict]:
    """Whisper pass + WhisperX forced alignment (word grid semantics: conf kept, untimed interpolated)."""
    if cand.get("file"):
        return _precomputed(wav, cand, run_idx)
    if cand.get("api") == "smallest":
        words = _smallest(wav, cand, language)
        if not cand.get("realign"):
            return words
        aligned = _forced_align(wav, [{**sg, "start": max(0.0, sg["start"] - 0.2), "end": sg["end"] + 0.2}
                                      for sg in _segments_from_words(words)])
        if aligned is None:
            raise RuntimeError(f"forced alignment failed for {wav}")
        return aligned
    segs, _ = _model(cand["model"]).transcribe(wav, **{**BASE, **cand["decode"]})
    segs = list(segs)
    sentence = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segs if s.text.strip()]
    aligned = _forced_align(wav, sentence)
    if aligned is None:
        raise RuntimeError(f"forced alignment failed for {wav}")
    return aligned


def recover(wav: str, audio, env, transcript: List[Dict], cand: dict) -> List[Dict]:
    """Re-transcribe each uncovered speech run on its own; keep words that pass the
    hallucination guards, align them on the full audio, flag them `recovered`."""
    grid = build_word_grid([{**t, "video_file": "x"} for t in transcript])
    model, added = _model(cand["model"]), []
    for s, e in uncovered_speech(env, word_coverage(grid.words)):
        a, b = max(0.0, s - 0.2), e + 0.2
        segs, _ = model.transcribe(audio[int(a * SR): int(b * SR)], beam_size=5, temperature=0.0,
                                   condition_on_previous_text=False, vad_filter=False)
        text = " ".join(x.text.strip() for x in segs
                        if x.no_speech_prob <= NO_SPEECH_MAX and x.avg_logprob >= AVG_LOGPROB_MIN).strip()
        if not text:
            continue
        words = _forced_align(wav, [{"start": a, "end": b, "text": text}]) or []
        added += [{**w, "recovered": True} for w in words if s - 0.25 <= w["start"] and w["end"] <= e + 0.25]
    merged = sorted(transcript + added, key=lambda w: (w["start"], w["end"]))
    out, prev_end = [], -1.0
    for w in merged:                                   # recovered words never overlap first-pass words
        if w.get("recovered") and w["start"] < prev_end:
            continue
        out.append(w)
        prev_end = max(prev_end, w["end"])
    return out


def run_candidate(name: str, clip_id: str, envs: dict, audios: dict, language: str = "en", run_idx: int = 0) -> dict:
    cand = CANDIDATES[name]
    cand.pop("_runtime", None)
    t0 = time.time()
    transcript = []
    for source_file, wav in clip_audio(clip_id).items():
        tr = transcribe(wav, cand, language, run_idx)
        if cand.get("recover"):
            tr = recover(wav, audios[source_file], envs[source_file], tr, cand)
        transcript += [{**t, "video_file": source_file} for t in tr]
    runtime = cand.pop("_runtime", None) if cand.get("file") else time.time() - t0   # external: runner's own timing
    grid = build_word_grid(transcript, asr={"candidate": name, **{k: v for k, v in cand.items() if k != "decode"},
                                            "decode": {k: v for k, v in cand["decode"].items() if k != "initial_prompt"}})
    grid = check_word_grid(grid, envs)
    gold = load_gold(clip_id)
    spans = [(s.source_file, s.start, s.end) for s in all_spans(gold)]
    outside = [w.text for w in grid.words
               if not any(f == w.source_file and a - 0.1 <= (w.start + w.end) / 2 <= b + 0.1 for f, a, b in spans)]
    return {"candidate": name, "clip": clip_id, "runtime_sec": round(runtime, 1),
            "score": grid_vs_gold(grid, gold, envs), "summary": summarize_word_grid(grid),
            "recovered_words": sum(WORD_FLAG_RECOVERED in w.flags for w in grid.words),
            "words_outside_gold": len(outside), "outside_sample": outside[:15],
            "word_texts": [w.text for w in grid.words]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cand", action="append", choices=sorted(CANDIDATES))
    ap.add_argument("--clip", action="append")
    ap.add_argument("--repeat", type=int, default=2, help="runs per candidate (determinism)")
    a = ap.parse_args()
    settings.enable_word_grid = True              # aligner keeps conf + interpolates untimed words
    cands = a.cand or [c for c in sorted(CANDIDATES) if not ({"api", "file"} & set(CANDIDATES[c]))]   # hosted/external: opt-in
    entries = [c for c in load_manifest()["clips"] if not a.clip or c["clip_id"] in a.clip]
    clips = [c["clip_id"] for c in entries]
    langs = {c["clip_id"]: c.get("asr_language", "en") for c in entries}
    run_dir = os.path.join(EVAL_SET_DIR, "results", f"asr-bakeoff-{datetime.now():%Y%m%d-%H%M%S}")
    os.makedirs(run_dir, exist_ok=True)

    rows = []
    for clip_id in clips:
        gold = load_gold(clip_id)
        audios = {s.file: load_audio(os.path.join(REPO_ROOT, s.path)) for s in gold.sources}
        envs = {f: envelope(x) for f, x in audios.items()}
        for name in cands:
            runs = [run_candidate(name, clip_id, envs, audios, langs[clip_id], i) for i in range(a.repeat)]
            r = runs[0]
            r["deterministic"] = all(x["word_texts"] == r["word_texts"] for x in runs[1:]) if a.repeat > 1 else None
            r["runtime_sec_all"] = [x["runtime_sec"] for x in runs]
            rows.append(r)
            with open(os.path.join(run_dir, f"{clip_id}__{name}.json"), "w") as f:
                json.dump({**r, "runs": [{k: v for k, v in x.items() if k != "word_texts"} for x in runs]}, f, indent=2)
            s, sm = r["score"], r["summary"]
            print(f"[{clip_id} {name}] words {sm['words']:4} | no-word gold speech {s['gold_speech_without_words_sec']:5.1f}s"
                  f" of {s['gold_speech_sec']} | bnd med/p90 {s['boundary_err_ms']['median']}/{s['boundary_err_ms']['p90']}ms"
                  f" >100ms {s['boundary_err_ms']['over_100ms']} | flags {sm['flags']} | outside-gold {r['words_outside_gold']}"
                  f" | recovered {r['recovered_words']} | det {r['deterministic']} | {r['runtime_sec_all']}s", flush=True)

    lines = [f"ASR BAKE-OFF {os.path.basename(run_dir)}", "",
             f"{'clip':<20}{'cand':<5}{'words':>6}{'no-word s':>10}{'of':>7}{'bnd med':>8}{'p90':>6}{'>100':>5}"
             f"{'low_conf':>9}{'on_sil':>7}{'long':>5}{'outside':>8}{'recov':>6}{'det':>6}{'sec':>7}"]
    for r in rows:
        s, sm, fl = r["score"], r["summary"], r["summary"]["flags"]
        lines.append(f"{r['clip']:<20}{r['candidate']:<5}{sm['words']:>6}{s['gold_speech_without_words_sec']:>10}"
                     f"{s['gold_speech_sec']:>7}{s['boundary_err_ms']['median']:>8}{s['boundary_err_ms']['p90']:>6}"
                     f"{s['boundary_err_ms']['over_100ms']:>5}{fl.get('low_conf', 0):>9}{fl.get('on_silence', 0):>7}"
                     f"{fl.get('long_span', 0):>5}{r['words_outside_gold']:>8}{r['recovered_words']:>6}"
                     f"{str(r['deterministic']):>6}{r['runtime_sec']:>7}")
    open(os.path.join(run_dir, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines) + f"\n\nresults: {os.path.relpath(run_dir, REPO_ROOT)}")


if __name__ == "__main__":
    main()
