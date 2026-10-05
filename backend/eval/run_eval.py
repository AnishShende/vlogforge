"""Run the eval set (Archdoc Phase 0 / S3).

For each clip in eval-set/manifest.json (or --clip), for each pipeline variant:
  1. run the pipeline stages (mocked perception) -> EDL -> rendered MP4
  2. score the rendered timeline against gold (eval.score.score_timeline)
  3. re-transcribe the rendered MP4 and diff against the expected keep text
Writes eval-set/results/<run_id>/<clip>__<variant>.json and summary.txt.

Usage (from backend/, ffmpeg on PATH):
    PYTHONPATH=. python -m eval.run_eval                       # all clips, both variants
    PYTHONPATH=. python -m eval.run_eval --clip IMG_1614 --variant word_timeline

Variants
  legacy         enable_word_timeline_redundancy = False
  word_timeline  enable_word_timeline_redundancy = True
The runner is an adapter: a later phase adds its own variant here; the scorer
does not change.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import run_mock_pipeline as rmp   # importing enables the whisper/LLM/JEV mocks
from app.config import settings
from app.tasks.assemble import assemble_vlog
from app.tasks.edl import generate_edl
from app.tasks.word_timeline_redundancy import apply_word_timeline_selection, clamp_edl_to_word_timeline
from app.utils.word_snap import snap_edl_to_word_boundaries
from eval.draft_gold import _default_job_dir, _git_rev
from eval.gold import EVAL_SET_DIR, REPO_ROOT, load_gold, load_manifest
from eval.score import Range, expected_text, score_text, score_timeline
from eval.snap_gold import HOP_SEC, envelope, load_audio

logger = logging.getLogger("VlogForge.Eval")
VARIANTS = {"legacy": False, "word_timeline": True}
DEFAULT_TARGET_SEC = 60.0


def _mock_files() -> set:
    d = settings.mock_llm_dir
    return set(os.listdir(d)) if os.path.isdir(d) else set()


def rendered_ranges(edl: List[Dict]) -> List[Range]:
    """The source ranges assembly actually renders, in output order.
    MIRRORS app/tasks/assemble.py assemble_vlog() overlap clamp (consecutive
    same-file entries: drop contained, trim overlapping, drop < 0.1s). Keep in sync."""
    out: List[Range] = []
    for e in edl:
        r = Range(e.get("source_file") or e.get("video_file", ""), float(e["start_sec"]), float(e["end_sec"]))
        if out and r.source_file == out[-1].source_file:
            prev = out[-1]
            if r.start >= prev.start and r.end <= prev.end:
                continue
            if r.start < prev.end:
                r.start = prev.end
                if r.end - r.start < 0.1:
                    continue
        out.append(r)
    return out


def run_pipeline(video: str, job_dir: str, work_dir: str, clip_id: str, variant: str,
                 target: float, context: str) -> Tuple[List[Dict], str, dict]:
    """Mirror of orchestrator.run_pipeline_sync stages 1-7 (mocked perception)."""
    settings.enable_word_timeline_redundancy = VARIANTS[variant]
    egt_doc, ctx = rmp.build_egt(video=video, job_dir=job_dir, target_duration=target, context_text=context)
    warnings = []
    if settings.enable_word_timeline_redundancy:
        egt_doc, warnings = apply_word_timeline_selection(egt_doc)
    all_segments = egt_doc.segments
    edl, _w, reasoning_mode = generate_edl(egt_doc, ctx["full_transcript_segments"], target, context)
    edl = snap_edl_to_word_boundaries(edl, {s.clip_id: s.model_dump() for s in all_segments})
    clamped = clamp_edl_to_word_timeline(edl, egt_doc) if settings.enable_word_timeline_redundancy else 0
    egt_clip_ids = {s.clip_id for s in all_segments
                    if not s.is_bad_take and not getattr(s, "is_superseded_take", False)
                    and not getattr(s, "is_stutter_repeat", False)}
    os.makedirs(work_dir, exist_ok=True)
    out = os.path.join(work_dir, f"{clip_id}__{variant}.mp4")
    if not assemble_vlog(edl, ctx["files_info"], work_dir, out, egt_clip_ids=egt_clip_ids):
        raise RuntimeError(f"assembly failed for {clip_id}/{variant}")
    info = {"reasoning_mode": reasoning_mode, "edl_entries": len(edl), "word_timeline_clamped": clamped,
            "pipeline_warnings": warnings}
    return edl, out, info


def transcribe_output(mp4: str) -> str:
    """Re-transcribe the rendered output with the local Whisper model directly
    (bypassing the whisper mock layer, which would replay the SOURCE transcript)."""
    from app.tasks.transcribe import get_whisper_model
    model = get_whisper_model()
    if model is None:
        raise RuntimeError("local Whisper model unavailable; cannot re-transcribe output")
    segs, _ = model.transcribe(load_audio(mp4), beam_size=5, condition_on_previous_text=False, vad_filter=True)
    return " ".join(s.text.strip() for s in segs)


def eval_clip(entry: dict, variant: str, run_dir: str, work_dir: str, skip_text: bool) -> dict:
    clip_id = entry["clip_id"]
    gold = load_gold(clip_id)
    if gold.status != "reviewed":
        logger.warning(f"[EVAL] {clip_id}: gold status is {gold.status!r}; results are NOT gate-eligible")
    if len(gold.sources) != 1:
        raise NotImplementedError(f"{clip_id}: multi-source clips need a multi-file runner (not built yet)")
    video = os.path.join(REPO_ROOT, gold.sources[0].path)
    target = float(entry.get("target_duration_sec", DEFAULT_TARGET_SEC))

    before = _mock_files()
    edl, mp4, info = run_pipeline(video, _default_job_dir(video, clip_id), work_dir, clip_id, variant,
                                  target, entry.get("context_text", ""))
    new_mocks = sorted(_mock_files() - before)
    if new_mocks:
        logger.warning(f"[EVAL] {clip_id}/{variant}: {len(new_mocks)} LIVE model call(s) (mock cache misses)")

    env = envelope(load_audio(video))
    ranges = rendered_ranges(edl)
    timeline = score_timeline(gold, ranges, {gold.sources[0].file: (env.speech, HOP_SEC)})
    text = None
    if not skip_text:
        heard = transcribe_output(mp4)
        text = {**score_text(expected_text(gold, timeline), heard), "heard": heard}

    result = {
        "clip_id": clip_id, "variant": variant, "gold_status": gold.status,
        "target_duration_sec": target, "output_mp4": os.path.relpath(mp4, REPO_ROOT),
        "pipeline": {**info, "live_model_calls": len(new_mocks), "new_mock_files": new_mocks},
        "rendered_ranges": [[r.source_file, round(r.start, 3), round(r.end, 3)] for r in ranges],
        "timeline": timeline, "text": text,
    }
    with open(os.path.join(run_dir, f"{clip_id}__{variant}.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def format_report(results: List[dict], meta: dict) -> str:
    L = [f"EVAL RUN {meta['run_id']}  git={meta['git']}  at={meta['at']}", ""]
    hdr = (f"{'clip':<12} {'variant':<14} {'keep ok':>7} {'clip':>4} {'miss':>4} {'dup':>3} "
           f"{'leak':>4} {'leak_s':>6} {'imp_drop':>8} {'opt_in':>6} {'cuts_sil%':>9} {'unlab_s':>7} {'WER':>5} {'out_s':>6} {'live':>4} gold")
    L += [hdr, "-" * len(hdr)]
    for r in results:
        s, t = r["timeline"]["summary"], r["text"]
        L.append(f"{r['clip_id']:<12} {r['variant']:<14} {s['keep_complete']:>3}/{s['keep_total']:<3} "
                 f"{s['keep_clipped']:>4} {s['keep_missing']:>4} {s['keep_duplicated']:>3} "
                 f"{s['exclude_leaked']:>4} {s['exclude_leaked_sec']:>6.2f} "
                 f"{s['keep_imperfect_dropped']:>3}/{s['keep_imperfect_total']:<4} "
                 f"{s['optional_included']:>2}/{s['optional_total']:<3} "
                 f"{s.get('cut_edges_in_silence_pct') or 0:>9.1f} {s.get('unlabelled_speech_in_output_sec', 0):>7.2f} "
                 f"{(t['word_error_rate'] if t else float('nan')):>5.2f} {s['output_sec']:>6.1f} "
                 f"{r['pipeline']['live_model_calls']:>4} {r['gold_status']}")
    for r in results:
        L += ["", f"=== {r['clip_id']} / {r['variant']} ===",
              f"pipeline: {json.dumps({k: v for k, v in r['pipeline'].items() if k != 'new_mock_files'})}",
              "keep spans:"]
        for k in r["timeline"]["keep"]:
            L.append(f"  {k['id']:>4} {k['status']:<10} {k['quality'][:4]:<4} cov={k['covered_frac']:.2f} head={k['head_clip_sec']:.2f} "
                     f"tail={k['tail_clip_sec']:.2f} gap={k['interior_gap_sec']:.2f} x{k['multiplicity']} "
                     f"pos={k['output_pos']}  {k['text'][:70]}")
        opts = r["timeline"].get("optional", [])
        if opts:
            L.append("optional spans (not scored): " + ", ".join(
                f"{o['id']}={'in' if o['included_frac'] > 0.5 else 'out'}" for o in opts))
        leaks = [x for x in r["timeline"]["exclude"] if x["leaked_sec"] > 0]
        L.append(f"exclude spans with any overlap ({len(leaks)}):")
        for x in leaks:
            L.append(f"  {x['id']:>4} {'LEAK' if x['leaked'] else 'edge':<5} {x['leaked_sec']:6.2f}s "
                     f"[{x['reason']}] {x['text'][:70]}")
        ins = r["timeline"]["summary"].get("cut_edges_in_speech", [])
        L.append(f"cut edges in speech ({len(ins)}): "
                 + ", ".join(f"{c['edge']}@{c['t']}" for c in ins))
        L.append("rendered ranges: " + ", ".join(f"[{a:.2f}-{b:.2f}]" for _, a, b in r["rendered_ranges"]))
        t = r["text"]
        if t:
            L.append(f"text: expected {t['expected_words']} words, heard {t['heard_words']}; "
                     f"missing {t['missing_words']}, extra {t['extra_words']}, substituted {t['substituted_words']}")
            for k in ("missing", "extra", "substituted"):
                if t[k]:
                    L.append(f"  {k}: " + " | ".join(t[k]))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", action="append", help="clip_id from the manifest (repeatable; default: all)")
    ap.add_argument("--variant", action="append", choices=sorted(VARIANTS), help="default: all variants")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--skip-text", action="store_true", help="skip output re-transcription")
    a = ap.parse_args()

    manifest = load_manifest()
    entries = [c for c in manifest["clips"] if not a.clip or c["clip_id"] in a.clip]
    unknown = set(a.clip or []) - {c["clip_id"] for c in entries}
    if unknown:
        sys.exit(f"unknown clip(s): {sorted(unknown)}")
    variants = a.variant or sorted(VARIANTS)

    meta = {"run_id": a.run_id or datetime.now().strftime("%Y%m%d-%H%M%S"), "git": _git_rev(),
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "clips": [c["clip_id"] for c in entries], "variants": variants,
            "coverage_gaps": manifest.get("coverage_gaps", [])}
    run_dir = os.path.join(EVAL_SET_DIR, "results", meta["run_id"])
    work_dir = os.path.join(EVAL_SET_DIR, "_work", meta["run_id"])
    os.makedirs(run_dir, exist_ok=True)

    results = []
    for c in entries:
        for v in variants:
            print(f"[eval] {c['clip_id']} / {v} ...", flush=True)
            results.append(eval_clip(c, v, run_dir, work_dir, a.skip_text))

    report = format_report(results, meta)
    if meta["coverage_gaps"]:
        report += "\nNOT COVERED by this eval set: " + "; ".join(meta["coverage_gaps"]) + "\n"
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write(report)
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\n" + report)
    print(f"results: {os.path.relpath(run_dir, REPO_ROOT)}")


if __name__ == "__main__":
    main()
