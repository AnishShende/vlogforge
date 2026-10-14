"""Phase 10 measures: B-roll placement and shot choice vs the editors of finished vlogs (eval-set/broll/).

Placement  per review window, the words spoken over the editor's B-roll shots are "covered", the words
           over A-roll shots are "face"; words over silent shots are not scored. Score = precision /
           recall of covered words. Baselines: cover every word ("cover all"), cover none.
Shot choice  for every B-roll shot of a window, the query is what is said over it (at least
           MIN_QUERY_WORDS words, widened with the neighbouring words); the pool is every non-A-roll shot
           of ALL indexed vlogs (other vlogs = wrong-vlog negatives). Hit@k when the editor's shot, or a
           near-duplicate of it from the same vlog (SigLIP cosine >= NEAR_DUP), is in the top k.
           Baselines: random pick (k / pool size), SigLIP on the spoken words alone.
(The v1 --predict path, per-line face/B-roll placement, was retired 2026-10-12; numbers in the roadmap.
Voice-over B-roll is measured by eval/voiceover_eval.py.)

Usage (from backend/):
    PYTHONPATH=. python -m eval.broll_eval
"""

import json
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from app.tasks import shots as shot_index
from eval.broll_gold import ANSWER_KEY_CLIPS, CLIPS, OUT_DIR, SRC_DIR, words_of

MIN_QUERY_WORDS = 8
NEAR_DUP = 0.9
KS = (1, 5, 20)


def load(clip: str) -> Tuple[Dict, Optional[Dict], Dict, np.ndarray]:
    pre = json.load(open(os.path.join(OUT_DIR, clip, "prelabels.json")))
    gp = os.path.join(OUT_DIR, clip, "gold.json")
    gold = json.load(open(gp)) if os.path.exists(gp) else None
    index = shot_index.index_file(os.path.join(SRC_DIR, f"{clip}.mp4"))
    return pre, gold, index, shot_index.load_embeddings(index)


def labels_of(pre: Dict, gold: Optional[Dict]) -> Dict[int, str]:
    """Gold label inside the review windows, pre-label elsewhere."""
    lab = {s["i"]: s["pre"] for s in pre["shots"]}
    for s in (gold or {}).get("shots", []):
        lab[s["i"]] = s["label"]
    return lab


def word_targets(words: List[Dict], gold: Dict) -> List[Tuple[int, bool]]:
    """(word index, covered) for every word whose middle falls in a window shot labelled a_roll/b_roll."""
    out = []
    shots = gold["shots"]
    for wi, w in enumerate(words):
        mid = (w["start"] + w["end"]) / 2
        for s in shots:
            if s["start"] <= mid < s["end"]:
                if s["label"] != "silent":
                    out.append((wi, s["label"] == "b_roll"))
                break
    return out


def placement_score(targets: List[Tuple[int, bool]], covered: Set[int]) -> Dict:
    tp = sum(1 for wi, c in targets if c and wi in covered)
    fp = sum(1 for wi, c in targets if not c and wi in covered)
    fn = sum(1 for wi, c in targets if c and wi not in covered)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}


def query_text(words: List[Dict], s: float, e: float) -> str:
    idx = [i for i, w in enumerate(words) if w["end"] > s and w["start"] < e]
    if not idx:
        return ""
    a, b = idx[0], idx[-1]
    while b - a + 1 < MIN_QUERY_WORDS and (a > 0 or b < len(words) - 1):
        a, b = max(0, a - 1), min(len(words) - 1, b + 1)
    return " ".join(words[i]["text"].strip() for i in range(a, b + 1))


def window_words(words: List[Dict], gold: Dict, a: int, b: int) -> List[int]:
    by_i = {s["i"]: s for s in gold["shots"]}
    s, e = by_i[a]["start"], by_i[b]["end"]
    return [wi for wi, w in enumerate(words) if s <= (w["start"] + w["end"]) / 2 < e]


def main() -> None:
    data = {c: load(c) for c in CLIPS}
    # pool: every non-A-roll shot of every indexed vlog
    pool_clip, pool_i, pool_emb, pool_caption = [], [], [], []
    for c, (pre, gold, index, emb) in data.items():
        lab = labels_of(pre, gold)
        for i, l in lab.items():
            if l != "a_roll":
                pool_clip.append(c), pool_i.append(i), pool_emb.append(emb[i])
                pool_caption.append(index["shots"][i]["caption"])
    P = np.stack(pool_emb)
    print(f"pool: {len(pool_i)} non-A-roll shots from {', '.join(CLIPS)}")
    for c in ANSWER_KEY_CLIPS:
        pre, gold, index, emb = data[c]
        words = words_of(c)
        targets = word_targets(words, gold)
        n_cov = sum(1 for _, cov in targets if cov)

        def near_of(i: int) -> Set[int]:
            return {j for j, (pc, pi) in enumerate(zip(pool_clip, pool_i))
                    if pc == c and (pi == i or float(emb[pi] @ emb[i]) >= NEAR_DUP)}

        print(f"\n== {c}: {len(gold['shots'])} window shots, {len(targets)} scored words, {n_cov} covered by the editor "
              f"({n_cov / max(len(targets), 1):.0%})")
        print("  placement  cover all :", placement_score(targets, {wi for wi, _ in targets}))
        print("  placement  cover none:", placement_score(targets, set()))
        b_shots = [s for s in gold["shots"] if s["label"] == "b_roll"]
        queries = [(s, query_text(words, s["start"], s["end"])) for s in b_shots]
        queries = [(s, q) for s, q in queries if q]
        Q = shot_index.embed_texts([q for _, q in queries])
        sims = Q @ P.T
        hits = {k: 0 for k in KS}
        wrong_vlog_top1 = 0
        for qi, (s, q) in enumerate(queries):
            near = near_of(s["i"])
            order = np.argsort(-sims[qi])
            rank = next(r for r, j in enumerate(order) if j in near)
            for k in KS:
                hits[k] += rank < k
            wrong_vlog_top1 += pool_clip[order[0]] != c
        n = len(queries)
        print(f"  shot choice ({n} B-roll shots, query = spoken words):")
        print("    random     : " + ", ".join(f"@{k} {k / len(pool_i):.1%}" for k in KS))
        print("    SigLIP text: " + ", ".join(f"@{k} {hits[k] / n:.1%}" for k in KS)
              + f"; top-1 from the wrong vlog {wrong_vlog_top1 / n:.0%}")


if __name__ == "__main__":
    main()
