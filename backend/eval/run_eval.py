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
  compiled_gold  Phase 2: the gold keeps as an EditPlan -> compiler (S2 cuts, S3 pauses) -> S4 render.
                 Upper bound for any planner; measures the compiler alone. No LLM.
  grid_cleanup   Phase 4: word grid -> speech cleanup (keep by default, retakes removed, JEV) ->
                 EditPlan -> compiler -> render. The real planner, no gold input.
The runner is an adapter: a later phase adds its own variant here; the scorer
does not change.

    PYTHONPATH=. python -m eval.run_eval --variant compiled_gold --gate     # Phase 2 exit gate
    PYTHONPATH=. python -m eval.run_eval --rescore baseline-2026-10-05-b    # stored run, current gold + scorer
    PYTHONPATH=. python -m eval.run_eval --compare RUN_A RUN_B              # determinism
"""

import argparse
import hashlib
import json
import subprocess
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
VARIANTS = {"legacy": False, "word_timeline": True, "compiled_gold": None, "grid_cleanup": None}
LEAK_BUDGET = 0.10       # Phase 4 gate: leaked exclude seconds <= 10% of output (user, 2026-10-06)
AV_TOL_SEC = 1 / 30      # gate: |audio - video| below one output frame
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


def _pcm_md5(mp4: str) -> str:
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", mp4, "-map", "0:a:0", "-f", "s16le", "-"],
                         check=True, capture_output=True).stdout
    return hashlib.md5(pcm).hexdigest()


def run_compiled_gold(clip_id: str, work_dir: str) -> Tuple[List[Range], str, dict]:
    from eval.gold_plan import compile_gold
    saved = (settings.enable_word_grid, settings.enable_mock_whisper)
    try:   # compile_gold turns the word grid on; other variants in this run must not inherit it
        r = compile_gold(clip_id, render_dir=work_dir, transcribe=False)
    finally:
        settings.enable_word_grid, settings.enable_mock_whisper = saved
    segs = r["segments"]
    ranges = [Range(s["source_file"], float(s["src_in"]), float(s["src_out"])) for s in segs]
    tight = [round(c["time"], 3) for s in segs for c in (s["cut_in"], s["cut_out"]) if "tight" in c["flags"]]
    flags = {}
    for s in segs:
        for c in (s["cut_in"], s["cut_out"]):
            for f in c["flags"]:
                flags[f] = flags.get(f, 0) + 1
    R = r["render"]
    info = {"plan_segments": r["plan_segments"], "compiled_segments": len(segs),
            "keeps_without_words": r["keeps_without_words"], "pause_targets": r["pause_targets"],
            "pauses_shortened": [s["pause_shortened_before_sec"] for s in segs if s.get("pause_shortened_before_sec")],
            "cut_flags": flags, "tight_cut_times": tight, "expected_sec": R["expected_sec"],
            "video_sec": round(R["streams"]["video"], 4), "audio_sec": round(R["streams"]["audio"], 4),
            "audio_pcm_md5": _pcm_md5(R["mp4"]), "validation": R["validation"]["status"],
            "validation_not_pass": {f"{c['status']}:{c['check']}": 1 for c in R["validation"]["checks"]
                                    if c["status"] != "pass"}}
    return ranges, R["mp4"], info


def run_grid_cleanup(clip_id: str, work_dir: str) -> Tuple[List[Range], str, dict]:
    from app.tasks.compiler import compile_and_render
    from app.tasks.speech_cleanup import REMOVE, REVIEW, cleanup_plan, label_grid, label_ranges
    from app.tasks.word_grid import check_word_grid
    from eval.gold_plan import gold_plan
    from eval.grid_eval import build_clip_grid
    saved = (settings.enable_word_grid, settings.enable_mock_whisper, settings.enable_mock_jev)
    try:
        settings.enable_word_grid = settings.enable_mock_whisper = settings.enable_mock_jev = True
        gold = load_gold(clip_id)
        envs = {s.file: envelope(load_audio(os.path.join(REPO_ROOT, s.path))) for s in gold.sources}
        grid = check_word_grid(build_clip_grid(clip_id), envs)
        labels = label_grid(grid)
        plan = cleanup_plan(grid, labels)
        os.makedirs(work_dir, exist_ok=True)
        mp4 = os.path.join(work_dir, f"{clip_id}__grid_cleanup.mp4")
        timeline, info = compile_and_render(plan, grid, envs, {s.file: os.path.join(REPO_ROOT, s.path)
                                                                for s in gold.sources}, mp4)
    finally:
        settings.enable_word_grid, settings.enable_mock_whisper, settings.enable_mock_jev = saved
    ranges = [Range(s.source_file, s.src_in, s.src_out) for s in timeline.segments]
    rr = label_ranges(grid, labels)
    v = info["validation"]
    lengths = {c["check"]: c["evidence"] for c in v["checks"] if c["check"] in ("render_length", "av_sync")}
    info = {"words": len(grid.words), "removed_words": sum(l["label"] == REMOVE for l in labels),
            "review_lines": [{"start": round(r["start"], 2), "end": round(r["end"], 2), "text": r["text"][:60],
                              "alternatives": len(r["by"]["alternatives"])} for r in rr if r["label"] == REVIEW],
            "plan_segments": len(plan.segments), "compiled_segments": len(timeline.segments),
            "keeps_without_words": gold_plan(gold, grid)[1], "pause_targets": info["pause_targets"],
            "video_sec": lengths["render_length"]["video_sec"], "audio_sec": lengths["av_sync"]["audio_sec"],
            "expected_sec": timeline.duration_sec, "validation": v["status"],
            "validation_not_pass": {f"{c['status']}:{c['check']}": 1 for c in v["checks"] if c["status"] != "pass"},
            "audio_pcm_md5": _pcm_md5(mp4)}
    return ranges, mp4, info


def run_pipeline(video: str, job_dir: str, work_dir: str, clip_id: str, variant: str,
                 target: float, context: str) -> Tuple[List[Range], str, dict]:
    """Mirror of orchestrator.run_pipeline_sync stages 1-7 (mocked perception)."""
    if variant == "compiled_gold":
        return run_compiled_gold(clip_id, work_dir)
    if variant == "grid_cleanup":
        return run_grid_cleanup(clip_id, work_dir)
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
    return rendered_ranges(edl), out, info


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
    ranges, mp4, info = run_pipeline(video, _default_job_dir(video, clip_id), work_dir, clip_id, variant,
                                     target, entry.get("context_text", ""))
    new_mocks = sorted(_mock_files() - before)
    if new_mocks:
        logger.warning(f"[EVAL] {clip_id}/{variant}: {len(new_mocks)} LIVE model call(s) (mock cache misses)")

    env = envelope(load_audio(video))
    timeline = score_timeline(gold, ranges, {gold.sources[0].file: (env.speech, HOP_SEC, env.speech | env.loud)})
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


def rescore(run_id: str) -> Tuple[List[dict], dict]:
    """Re-score a stored run (its rendered ranges + heard text) on the CURRENT gold and scorer."""
    src = os.path.join(EVAL_SET_DIR, "results", run_id)
    results = []
    for name in sorted(os.listdir(src)):
        if not name.endswith(".json") or name == "meta.json":
            continue
        old = json.load(open(os.path.join(src, name)))
        gold = load_gold(old["clip_id"])
        video = os.path.join(REPO_ROOT, gold.sources[0].path)
        env = envelope(load_audio(video))
        ranges = [Range(f, a, b) for f, a, b in old["rendered_ranges"]]
        timeline = score_timeline(gold, ranges, {gold.sources[0].file: (env.speech, HOP_SEC, env.speech | env.loud)})
        text = None
        if old.get("text"):
            heard = old["text"]["heard"]
            text = {**score_text(expected_text(gold, timeline), heard), "heard": heard}
        results.append({**old, "gold_status": gold.status, "timeline": timeline, "text": text,
                        "rescored_from": run_id})
    return results, json.load(open(os.path.join(src, "meta.json")))


def gate(results: List[dict]) -> List[str]:
    """Phase 2 exit gate, per compiled_gold result. Returns report lines; 'FAIL' marks a failure."""
    L = ["", "PHASE 2 GATE (compiled_gold)"]
    for r in results:
        if r["variant"] != "compiled_gold":
            continue
        s, p = r["timeline"]["summary"], r["pipeline"]
        no_words = set(p["keeps_without_words"])
        missing = [k["id"] for k in r["timeline"]["keep"] if k["status"] == "missing" and k["id"] not in no_words]
        in_speech = [c["t"] for c in s.get("cut_edges_in_speech", [])]
        unflagged = [t for t in in_speech if not any(abs(t - x) < 0.002 for x in p["tight_cut_times"])]
        av = abs(p["audio_sec"] - p["video_sec"])
        checks = [("0 clipped", s["keep_clipped"] == 0, s["keep_clipped"]),
                  ("0 missing (excl. no-word keeps " + (",".join(sorted(no_words)) or "none") + ")", not missing, missing),
                  ("0 duplicated", s["keep_duplicated"] == 0, s["keep_duplicated"]),
                  ("0 leaked", s["exclude_leaked"] == 0, s["exclude_leaked"]),
                  ("cuts in speech all flagged tight", not unflagged, f"{len(in_speech)} in speech, unflagged {unflagged}"),
                  ("video length == plan", abs(p["video_sec"] - p["expected_sec"]) < 1e-3, f"{p['video_sec']} vs {p['expected_sec']}"),
                  ("|audio - video| < 1 frame", av < AV_TOL_SEC, f"{av * 1000:.1f} ms"),
                  ("validation report, no fail", p.get("validation") in ("pass", "warn"), p.get("validation"))]
        for name, ok, detail in checks:
            L.append(f"  {r['clip_id']:<20} {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    return L


def gate_cleanup(results: List[dict]) -> List[str]:
    """Phase 4 exit gate (user, 2026-10-06), per grid_cleanup result."""
    L = ["", "PHASE 4 GATE (grid_cleanup)"]
    for r in results:
        if r["variant"] != "grid_cleanup":
            continue
        s, p = r["timeline"]["summary"], r["pipeline"]
        no_words = set(p["keeps_without_words"])
        incomplete = [k["id"] for k in r["timeline"]["keep"] if k["status"] != "complete" and k["id"] not in no_words]
        leak_frac = s["exclude_leaked_sec"] / s["output_sec"] if s["output_sec"] else 0.0
        leaks = [f"{x['id']} {x['leaked_sec']:.1f}s" for x in r["timeline"]["exclude"] if x["leaked"]]
        checks = [("all keeps complete (excl. no-word keeps " + (",".join(sorted(no_words)) or "none") + ")",
                   not incomplete, incomplete),
                  ("0 clipped", s["keep_clipped"] == 0, s["keep_clipped"]),
                  ("0 unique speech dropped", s["keep_imperfect_dropped"] == 0,
                   f"{s['keep_imperfect_dropped']}/{s['keep_imperfect_total']}"),
                  (f"leaks <= {LEAK_BUDGET:.0%} of output", leak_frac <= LEAK_BUDGET,
                   f"{s['exclude_leaked_sec']:.1f}s of {s['output_sec']:.1f}s = {leak_frac:.0%}: " + ", ".join(leaks)),
                  ("validation report, no fail", p["validation"] in ("pass", "warn"), p["validation"])]
        for name, ok, detail in checks:
            L.append(f"  {r['clip_id']:<20} {'PASS' if ok else 'FAIL'}  {name}: {detail}")
        L.append(f"  {r['clip_id']:<20} INFO  review lines {len(p['review_lines'])}: "
                 + "; ".join(f"{x['start']}-{x['end']} ({x['alternatives']} alt)" for x in p["review_lines"]))
    return L


def compare_runs(a: str, b: str) -> List[str]:
    L = [f"DETERMINISM {a} vs {b}"]
    for name in sorted(os.listdir(os.path.join(EVAL_SET_DIR, "results", a))):
        if not name.endswith(".json") or name == "meta.json":
            continue
        x = json.load(open(os.path.join(EVAL_SET_DIR, "results", a, name)))
        pb = os.path.join(EVAL_SET_DIR, "results", b, name)
        if not os.path.exists(pb):
            L.append(f"  {name}: missing in {b}")
            continue
        y = json.load(open(pb))
        same = {"ranges": x["rendered_ranges"] == y["rendered_ranges"],
                "timeline": x["timeline"] == y["timeline"],
                "audio_pcm": x["pipeline"].get("audio_pcm_md5") == y["pipeline"].get("audio_pcm_md5"),
                "heard_text": (x["text"] or {}).get("heard") == (y["text"] or {}).get("heard")}
        L.append(f"  {name}: " + "  ".join(f"{k} {'same' if v else 'DIFFERENT'}" for k, v in same.items()))
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", action="append", help="clip_id from the manifest (repeatable; default: all)")
    ap.add_argument("--variant", action="append", choices=sorted(VARIANTS), help="default: all variants")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--skip-text", action="store_true", help="skip output re-transcription")
    ap.add_argument("--gate", action="store_true", help="append the Phase 2 gate check (compiled_gold rows)")
    ap.add_argument("--rescore", metavar="RUN_ID", help="re-score a stored run on the current gold + scorer")
    ap.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="determinism check of two runs")
    a = ap.parse_args()
    if a.compare:
        print("\n".join(compare_runs(*a.compare)))
        return
    if a.rescore:
        results, old_meta = rescore(a.rescore)
        meta = {**old_meta, "run_id": a.run_id or f"{a.rescore}__rescored-{datetime.now():%Y%m%d}",
                "rescored_from": a.rescore, "rescored_git": _git_rev(),
                "rescored_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        run_dir = os.path.join(EVAL_SET_DIR, "results", meta["run_id"])
        os.makedirs(run_dir, exist_ok=True)
        for r in results:
            with open(os.path.join(run_dir, f"{r['clip_id']}__{r['variant']}.json"), "w") as f:
                json.dump(r, f, indent=2, ensure_ascii=False)
        report = format_report(results, meta)
        with open(os.path.join(run_dir, "summary.txt"), "w") as f:
            f.write(report)
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(report)
        print(f"results: {os.path.relpath(run_dir, REPO_ROOT)}")
        return

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
    if a.gate:
        report += "\n".join(gate(results) + gate_cleanup(results)) + "\n"
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
