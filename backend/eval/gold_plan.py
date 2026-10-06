"""Gold -> EditPlan (Archdoc roadmap Phase 2): the plan a perfect planner would emit.

Each grid word goes to the gold span it overlaps most (a zero-length word to the span
containing it). Midpoints are not used: aligners stretch words, so a word's midpoint can
fall outside the span it belongs to. The plan is every keep span's words, in source order.
Keeps with no grid words cannot be expressed as word ranges: reported, never hidden.

Usage (from backend/, ffmpeg on PATH):
    PYTHONPATH=. python -m eval.gold_plan [--clip ID ...]     # compile + score the gold plans, no render
    PYTHONPATH=. python -m eval.gold_plan --render DIR         # + render, A/V lengths, re-transcribed text
"""

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

from app.config import settings
from app.models import EditPlan, EditSegment, WordGrid
import subprocess

from app.tasks.compiler import build_timeline, compile_plan, render_timeline, speaker_pause_targets
from app.tasks.word_grid import check_word_grid
from app.utils.speech_activity import HOP_SEC, envelope, load_audio
from eval.gold import REPO_ROOT, Gold, all_spans, load_gold, load_manifest
from eval.score import Range, expected_text, score_text, score_timeline


def word_owner(word, spans) -> Optional[object]:
    ov = [(min(word.end, s.end) - max(word.start, s.start), s) for s in spans if s.source_file == word.source_file]
    ov = [x for x in ov if x[0] > 0]
    if ov:
        return max(ov, key=lambda x: x[0])[1]
    return next((s for s in spans if s.source_file == word.source_file and s.start <= word.start <= s.end), None)


def gold_plan(gold: Gold, grid: WordGrid) -> Tuple[EditPlan, List[str]]:
    """Returns (plan, keep ids with no grid words)."""
    spans = all_spans(gold)
    keep_ids = {k.id for k in gold.keep}
    owner = [word_owner(w, spans) for w in grid.words]
    segments, seen = [], set()
    i = 0
    while i < len(grid.words):
        s = owner[i]
        if s is None or s.id not in keep_ids:
            i += 1
            continue
        j = i
        while j + 1 < len(grid.words) and owner[j + 1] is s:
            j += 1
        segments.append(EditSegment(word_start=grid.words[i].id, word_end=grid.words[j].id, reason=f"gold {s.id}"))
        seen.add(s.id)
        i = j + 1
    return EditPlan(segments=segments, grid_fingerprint=grid.fingerprint()), sorted(keep_ids - seen,
                                                                                    key=lambda x: int(x[1:]))


def stream_durations(path: str) -> Dict[str, float]:
    from app.utils.ffmpeg import media_durations
    return media_durations(path)


def compile_gold(clip_id: str, render_dir: Optional[str] = None, pauses: bool = True, transcribe: bool = True) -> Dict:
    from eval.grid_eval import build_clip_grid
    settings.enable_word_grid = True
    settings.enable_mock_whisper = True
    gold = load_gold(clip_id)
    envs = {s.file: envelope(load_audio(os.path.join(REPO_ROOT, s.path))) for s in gold.sources}
    grid = check_word_grid(build_clip_grid(clip_id), envs)
    plan, no_words = gold_plan(gold, grid)
    targets = speaker_pause_targets(grid, envs) if pauses else None
    segs = compile_plan(plan, grid, envs, pause_targets=targets)
    ranges = [Range(s.source_file, s.src_in, s.src_out) for s in segs]
    timeline = score_timeline(gold, ranges, {f: (e.speech, HOP_SEC, e.speech | e.loud) for f, e in envs.items()})
    result = {"clip_id": clip_id, "plan_segments": len(plan.segments), "keeps_without_words": no_words,
              "pause_targets": targets, "segments": [s.model_dump() for s in segs], "timeline": timeline}
    if render_dir:
        from eval.run_eval import transcribe_output
        from app.tasks.validate import make_report, validate_render, validate_structure
        tl = build_timeline(segs, grid, pause_targets=targets)
        os.makedirs(render_dir, exist_ok=True)
        mp4 = os.path.join(render_dir, f"{clip_id}__compiled_gold.mp4")
        info = render_timeline(tl, {s.file: os.path.join(REPO_ROOT, s.path) for s in gold.sources}, mp4)
        result["render"] = {"mp4": mp4, "expected_sec": tl.duration_sec, "streams": stream_durations(mp4),
                            "head_fade_sec": tl.head_fade_sec, "tail_fade_sec": tl.tail_fade_sec,
                            "input_lufs": info["loudness"]["input_i"]}
        streams = result["render"]["streams"]
        result["render"]["validation"] = make_report(validate_structure(tl, plan, grid, envs) + validate_render(
            tl, streams["video"], streams["audio"])).model_dump()
        if transcribe:
            heard = transcribe_output(mp4)
            result["render"]["text"] = {**score_text(expected_text(gold, timeline), heard), "heard": heard}
    return result


def report(r: Dict) -> str:
    s, segs = r["timeline"]["summary"], r["segments"]
    cuts = [c for x in segs for c in (x["cut_in"], x["cut_out"])]
    flags = Counter(f for c in cuts for f in c["flags"])
    L = [f"=== {r['clip_id']}: {r['plan_segments']} plan segments -> {len(segs)} compiled ===",
         f"keep complete {s['keep_complete']}/{s['keep_total']}  clipped {s['keep_clipped']}  missing {s['keep_missing']}  "
         f"dup {s['keep_duplicated']}  leaked {s['exclude_leaked']} ({s['exclude_leaked_sec']:.2f}s)  "
         f"cuts in silence {s.get('cut_edges_in_silence_pct')}%  output {s['output_sec']:.1f}s",
         f"keeps with no grid words (known grid gap, not gated): {r['keeps_without_words'] or 'none'}",
         f"cut flags: {dict(flags) or 'none'}"]
    sh = [x["pause_shortened_before_sec"] for x in segs if x.get("pause_shortened_before_sec")]
    L.append(f"pause targets {r.get('pause_targets')}; pauses shortened {len(sh)}: "
             + (", ".join(f"{v:.2f}s" for v in sh) or "none"))
    bad = [k for k in r["timeline"]["keep"] if k["status"] != "complete" and k["id"] not in r["keeps_without_words"]]
    for k in bad:
        L.append(f"  KEEP {k['id']} {k['status']} cov={k['covered_frac']:.2f} head={k['head_clip_sec']:.2f} "
                 f"tail={k['tail_clip_sec']:.2f} gap={k['interior_gap_sec']:.2f}")
    for x in [x for x in r["timeline"]["exclude"] if x["leaked_sec"] > 0]:
        L.append(f"  EXCL {x['id']} {'LEAK' if x['leaked'] else 'edge'} {x['leaked_sec']:.2f}s [{x['reason']}]")
    for c in s.get("cut_edges_in_speech", []):
        L.append(f"  cut in speech: {c['edge']} @ {c['t']}")
    if "render" in r:
        R, t = r["render"], r["render"].get("text")
        v, a = R["streams"]["video"], R["streams"]["audio"]
        L.append(f"render: expected {R['expected_sec']:.3f}s  video {v:.3f}s  audio {a:.3f}s  |a-v| {abs(a - v) * 1000:.0f} ms  "
                 f"fades head {R['head_fade_sec']}s tail {R['tail_fade_sec']}s  input {R['input_lufs']} LUFS")
        if t:
            L.append(f"text: expected {t['expected_words']} words, heard {t['heard_words']}; missing {t['missing_words']}, "
                     f"extra {t['extra_words']}, substituted {t['substituted_words']}, WER {t['word_error_rate']:.2f}")
            for k in ("missing", "extra", "substituted"):
                if t[k]:
                    L.append(f"  {k}: " + " | ".join(t[k]))
    for x in segs:
        for c in (x["cut_in"], x["cut_out"]):
            if c["flags"]:
                L.append(f"  flagged cut {c['side']:<3} @ {c['time']:8.3f}  word edge {c['word_edge']:8.3f}  "
                         f"activity {c['activity_edge']:8.3f}  window {c['window'][0]:.3f}-{c['window'][1]:.3f}  "
                         f"{c['level_db']:.1f} dB  {c['flags']}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", action="append")
    ap.add_argument("--out", help="write full results JSON here")
    ap.add_argument("--render", metavar="DIR", help="also render each gold plan into DIR and check it")
    ap.add_argument("--no-pauses", action="store_true", help="S3 pause shortening off")
    a = ap.parse_args()
    clips = [c["clip_id"] for c in load_manifest()["clips"] if not a.clip or c["clip_id"] in a.clip]
    results = [compile_gold(c, a.render, not a.no_pauses) for c in clips]
    print("\n\n".join(report(r) for r in results))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
