"""Phase 6.5 S1: score an automatic speech edit against the user's choices, per clip, in words.

Reference per clip (first found):
  eval-set/<clip>/answer_key.json  user's take picks on the verbatim page: kept word ids of one job's grid
  eval-set/<clip>/gold.json        gold spans mapped to the job's words by midpoint: keep -> kept,
                                   exclude -> cut, optional / unlabelled -> not scored
Edit under test ("predictor"):
  stored-cleanup  the job's original automatic cleanup (cleanup.json, everything not labelled remove)
  edit-passes     roadmap Phase 6.5 passes (added with S2)

Metrics (scored words only):
  cut recall    share of the words the user cut that the edit cuts            (higher is better)
  over-cut      share of the words the user kept that the edit cuts            (lower is better)
  imperfect dropped  gold keep spans marked imperfect (unique speech) that lose most of their words: must be 0
Also lists every attempt (0.4 s pause split, as on the verbatim page) where edit and user disagree.

Usage (from backend/):
    PYTHONPATH=. python -m eval.edit_passes_eval                       # all clips with a known job
    PYTHONPATH=. python -m eval.edit_passes_eval --job WA_20260710_182656=review-gate-WA_20260710_182656
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Set, Tuple

from app.models import WordGrid
from app.utils import artifacts
from eval.gold import EVAL_SET_DIR, load_gold, load_manifest
from eval.verbatim_page import lines_of


def default_jobs() -> Dict[str, str]:
    """clip -> job: the answer key's job, else the Phase 6 review-gate job when stored."""
    out = {}
    for c in load_manifest()["clips"] + [{"clip_id": d} for d in sorted(os.listdir(EVAL_SET_DIR))]:
        clip = c["clip_id"]
        key = os.path.join(EVAL_SET_DIR, clip, "answer_key.json")
        if os.path.exists(key):
            out[clip] = json.load(open(key))["job"]
        elif artifacts.load_job(f"review-gate-{clip}", "grid") is not None:
            out[clip] = f"review-gate-{clip}"
    return out


def reference(clip: str, grid: WordGrid) -> Tuple[Set[str], Set[str], List[Set[str]], str]:
    """(kept, cut, imperfect keep spans as word sets, source). Unscored words are in neither set."""
    key_path = os.path.join(EVAL_SET_DIR, clip, "answer_key.json")
    ids = {w.id for w in grid.words}
    if os.path.exists(key_path):
        key = json.load(open(key_path))
        if key["grid_fingerprint"] != grid.fingerprint():
            raise SystemExit(f"{clip}: answer key made on grid {key['grid_fingerprint']}, job grid is {grid.fingerprint()}")
        kept = set(key["kept"])
        return kept, ids - kept, [], "answer_key"
    gold = load_gold(clip, verify_sources=False)
    kept, cut, imperfect = set(), set(), []
    for w in grid.words:
        m = (w.start + w.end) / 2
        inside = lambda spans: next((s for s in spans if s.source_file == w.source_file and s.start <= m <= s.end), None)
        if inside(gold.keep):
            kept.add(w.id)
        elif inside(gold.exclude):
            cut.add(w.id)
    for s in gold.keep:
        if getattr(s, "quality", None) == "imperfect":
            imperfect.append({w.id for w in grid.words if w.source_file == s.source_file
                              and s.start <= (w.start + w.end) / 2 <= s.end})
    return kept, cut, imperfect, "gold"


def predict_stored_cleanup(job: str, grid: WordGrid) -> Set[str]:
    ranges = artifacts.load_job(job, "cleanup")
    if ranges is None:
        raise SystemExit(f"job {job}: no stored cleanup.json")
    index = {w.id: i for i, w in enumerate(grid.words)}
    kept = set()
    for r in ranges:
        if r["label"] != "remove":
            kept |= {grid.words[i].id for i in range(index[r["word_start"]], index[r["word_end"]] + 1)}
    return kept


def predict_edit_passes(job: str, grid: WordGrid, trace: Optional[List] = None) -> Set[str]:
    """All Phase 6.5 passes from the full transcript; `trace` gets (pass, kept after it, cuts, stats)."""
    from app.tasks.edit_passes import ORDER, run_pass
    kept = {w.id for w in grid.words}
    for name in ORDER:
        cuts, st = run_pass(name, grid, kept)
        for c in cuts:
            if c["applied"]:
                kept -= set(c["word_ids"])
        if trace is not None:
            trace.append((name, set(kept), cuts, st))
    return kept


PREDICTORS = {"stored-cleanup": predict_stored_cleanup, "edit-passes": predict_edit_passes}


def score(grid: WordGrid, pred_kept: Set[str], kept: Set[str], cut: Set[str], imperfect: List[Set[str]]) -> Dict:
    dur = {w.id: w.end - w.start for w in grid.words}
    sec = lambda s: round(sum(dur[i] for i in s), 1)
    pred_cut = {w.id for w in grid.words} - pred_kept
    hit, over = pred_cut & cut, pred_cut & kept
    attempts = []
    for n, a in enumerate(lines_of([w.model_dump() for w in grid.words], 0.4)):
        ids = [w["id"] for w in a if w["id"] in kept | cut]
        miss = [i for i in ids if i in cut and i in pred_kept]
        wrong = [i for i in ids if i in kept and i not in pred_kept]
        if miss or wrong:
            attempts.append({"attempt": n + 1, "start": round(a[0]["start"], 1), "words": len(a),
                             "user_cut_edit_kept": len(miss), "user_kept_edit_cut": len(wrong)})
    return {
        "scored_words": len(kept | cut), "user_cut": len(cut), "user_cut_sec": sec(cut),
        "edit_cut": len(pred_cut & (kept | cut)),
        "cut_recall": round(len(hit) / len(cut), 3) if cut else None, "cut_recall_sec": sec(hit),
        "over_cut": round(len(over) / len(kept), 3) if kept else None, "over_cut_words": len(over), "over_cut_sec": sec(over),
        "imperfect_dropped": sum(1 for s in imperfect if s and len(s - pred_kept) > len(s) / 2),
        "imperfect_total": len(imperfect),
        "disagreements": attempts,
    }


def run(jobs: Dict[str, str], predictor: str) -> Dict[str, Dict]:
    out = {}
    for clip, job in sorted(jobs.items()):
        grid = WordGrid(**artifacts.load_job(job, "grid"))
        kept, cut, imperfect, src = reference(clip, grid)
        trace: List = []
        pred = PREDICTORS[predictor](job, grid, trace) if predictor == "edit-passes" else PREDICTORS[predictor](job, grid)
        r = score(grid, pred, kept, cut, imperfect)
        out[clip] = {"job": job, "reference": src, **r}
        if trace:      # cumulative score after each pass, plus what each pass cut
            out[clip]["passes"] = [{"pass": name, **{k: v for k, v in score(grid, k_after, kept, cut, imperfect).items()
                                                      if k != "disagreements"}, "stats": st,
                                    "cuts": [{**c, "user": "cut" if set(c["word_ids"]) <= cut else
                                              "kept" if set(c["word_ids"]) <= kept else "mixed"} for c in cuts]}
                                   for name, k_after, cuts, st in trace]
            sug = {i for _, _, cuts, _ in trace for c in cuts if not c["applied"] and c.get("action") != "restore"
                   for i in c["word_ids"]} & pred
            out[clip]["suggestions"] = {"words": len(sug), "user_cut": len(sug & cut), "user_kept": len(sug & kept)}
            ver = [c for _, _, cuts, _ in trace for c in cuts if c["pass"] == "verify"]
            rm = {i for c in ver if c["action"] == "remove" for i in c["word_ids"]}
            rs = {i for c in ver if c["action"] == "restore" for i in c["word_ids"]}
            out[clip]["verify"] = {"remove_words": len(rm), "remove_user_cut": len(rm & cut), "remove_user_kept": len(rm & kept),
                                   "missed_before": len(cut & pred), "restore_words": len(rs),
                                   "restore_user_kept": len(rs & kept), "restore_user_cut": len(rs & cut),
                                   "wrong_before": len(kept - pred)}
    return out


def report(results: Dict[str, Dict], predictor: str) -> str:
    lines = [f"edit under test: {predictor}",
             f"{'clip':20} {'ref':10} {'scored':>6} {'user cut':>12} {'cut recall':>16} {'over-cut':>16} {'imperfect dropped':>18}"]
    for clip, r in results.items():
        rec = "-" if r["cut_recall"] is None else f"{r['cut_recall']:.0%} ({r['cut_recall_sec']}s)"
        lines.append(f"{clip:20} {r['reference']:10} {r['scored_words']:6} {r['user_cut']:5} w {r['user_cut_sec']:5}s "
                     f"{rec:>16} {r['over_cut']:>7.1%} ({r['over_cut_words']} w) {r['imperfect_dropped']:>8}/{r['imperfect_total']}")
    for clip, r in results.items():
        for p in r.get("passes", []):
            st = p["stats"]
            lines.append(f"  {clip:18} after {p['pass']:13} cut recall {p['cut_recall'] if p['cut_recall'] is None else format(p['cut_recall'], '.0%'):>5}"
                         f"  over-cut {p['over_cut']:6.1%} ({p['over_cut_words']} w)  imperfect dropped {p['imperfect_dropped']}"
                         f"  | {st['model']} {st['calls']} call(s) ({st['cached']} cached), {st['input_tokens']}+{st['output_tokens']} tok,"
                         f" {st['cuts']} cuts, {st.get('suggestions', 0)} suggestions, {st['rejected']} rejected")
        if "verify" in r:
            v = r["verify"]
            lines.append(f"  {clip:18} VERIFIER: remove {v['remove_words']} w (user cut {v['remove_user_cut']} of the "
                         f"{v['missed_before']} still missed, user kept {v['remove_user_kept']}); restore {v['restore_words']} w "
                         f"(user kept {v['restore_user_kept']} of the {v['wrong_before']} wrongly cut, user cut {v['restore_user_cut']})")
        if "suggestions" in r:
            sg = r["suggestions"]
            lines.append(f"  {clip:18} suggestions (not applied): {sg['words']} words, user cut {sg['user_cut']}, user kept {sg['user_kept']}")
    for clip, r in results.items():
        if r["disagreements"]:
            lines.append(f"\n{clip}: attempts where the edit and the user disagree (#, start, user-cut-but-kept, user-kept-but-cut)")
            lines += [f"  #{d['attempt']:<3} {d['start']:7.1f}s  {d['words']:3} w   missed cut {d['user_cut_edit_kept']:3}   wrong cut {d['user_kept_edit_cut']:3}"
                      for d in r["disagreements"]]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", default=[], metavar="CLIP=JOB", help="override / add a clip's job")
    ap.add_argument("--predictor", default="stored-cleanup", choices=sorted(PREDICTORS))
    ap.add_argument("--out", help="write the results JSON here")
    a = ap.parse_args()
    jobs = default_jobs()
    jobs.update(dict(x.split("=", 1) for x in a.job))
    results = run(jobs, a.predictor)
    print(report(results, a.predictor))
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
