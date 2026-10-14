"""Phase 10 V2b + V3b on the synthetic raw projects (eval/broll_synth.py), scored against the editor's cut.

V2b story: word grid of every file with speech (all words kept: finished speech is already clean), moments
(app.tasks.moments, VOICE-OVER headers), story_order (code). Scored on the order of speech files: pairwise order
vs the original, and voice-over files "in place" (same speech neighbours before and after as in the original).
Run on the original upload order and on the shuffled one (no recording times: listing order is all there is).
V3b B-roll (original order run): every voice-over line gets a shot from the library = the project's B-roll and
silent clips + decoys from other videos (gym, the other travel vlog). Hit = the shot is one of the editor's shots
under that line. Variants: words-only SigLIP (no model), visual SigLIP, story-aware (sections + visuals + pick).

Usage (from backend/):
    PYTHONPATH=. python -m eval.voiceover_eval [--project nz_travel_w1] [--no-shuffled]
"""

import argparse
import json
import os
import random
import sys
from itertools import combinations
from typing import Dict, List, Optional, Set

import numpy as np

from app.tasks import broll
from app.tasks import shots as shot_index
from app.tasks.moments import build_moments, clip_order
from app.tasks.storyplan import story_order
from app.tasks.transcribe import transcribe_audio
from app.tasks.word_grid import build_word_grid
from eval.broll_gold import OUT_DIR
from eval.broll_synth import SYNTH_DIR

N_DECOYS_OTHER = 40


def project_words(proj: str) -> Dict[str, List[Dict]]:
    work = os.path.join(SYNTH_DIR, proj, "work")
    out = {}
    for f in json.load(open(os.path.join(SYNTH_DIR, proj, "gold.json")))["upload_order"]:
        wav = os.path.join(work, os.path.splitext(f)[0] + ".wav")
        ws = [w for w in transcribe_audio(wav) if w.get("start") is not None and str(w.get("text", "")).strip()]
        if ws:
            out[f] = ws
    return out


def story(proj: str, order_key: str, words: Dict[str, List[Dict]]):
    g = json.load(open(os.path.join(SYNTH_DIR, proj, "gold.json")))
    files_dir = os.path.join(SYNTH_DIR, proj, "files")
    grid = build_word_grid([{**w, "video_file": f} for f, ws in words.items() for w in ws])
    upload = [f for f in g[order_key] if f in words]
    clips = clip_order([{"filename": f, "path": os.path.join(files_dir, f)} for f in upload])
    moments = build_moments(grid, {w.id for w in grid.words}, clips)
    so = story_order(moments["moments"], moments["section_order"])
    by_id = {m["id"]: m for m in moments["moments"]}
    src = {w.id: w.source_file for w in grid.words}
    seq: List[str] = []
    for mid in so["order"]:
        for wid in by_id[mid]["word_ids"]:
            if src[wid] not in seq:
                seq.append(src[wid])
    orig = [t["file"] for t in g["timeline"] if t["file"] in words]
    pos = {f: k for k, f in enumerate(seq)}
    pairs = list(combinations(orig, 2))
    pair_ok = sum(pos[a] < pos[b] for a, b in pairs) / max(len(pairs), 1)
    vo = [f for f in orig if g["category"][f] == "voice_over"]
    def nb(lst, f):
        k = lst.index(f)
        return (lst[k - 1] if k else None, lst[k + 1] if k + 1 < len(lst) else None)
    in_place = sum(nb(seq, f) == nb(orig, f) for f in vo)
    vo_roles = {}
    for m in moments["moments"]:
        if g["category"].get(m["source_file"]) == "voice_over":
            vo_roles[m["role"]] = vo_roles.get(m["role"], 0) + 1
    print(f"  story [{order_key}]: pairwise order {pair_ok:.0%}, voice-over in place {in_place}/{len(vo)}, "
          f"moments {len(moments['moments'])}, sections {moments['section_order']}, voice-over moment roles {vo_roles}",
          flush=True)
    return grid, moments, so, seq


def library(proj: str, g: Dict) -> Dict:
    """Project B-roll + silent clips (every shot) + decoys from other videos."""
    files_dir = os.path.join(SYNTH_DIR, proj, "files")
    ents = []
    for f, cat in g["category"].items():
        if cat in ("b_roll", "silent"):
            ix = shot_index.index_file(os.path.join(files_dir, f))
            emb = shot_index.load_embeddings(ix)
            for s in ix["shots"]:
                ents.append({"file": f, "src": proj, "caption": s["caption"], "emb": emb[s["i"]]})
    other = "cycling_europe" if g["clip"] == "nz_travel" else "nz_travel"
    rng = random.Random(f"decoys|{proj}")
    for clip, n in (("gym_chest", None), (other, N_DECOYS_OTHER)):
        pre = json.load(open(os.path.join(OUT_DIR, clip, "prelabels.json")))
        ix = shot_index.index_file(pre["video"])
        emb = shot_index.load_embeddings(ix)
        pool = [s for s in pre["shots"] if s["pre"] != "a_roll"]
        for s in (pool if n is None else rng.sample(pool, n)):
            ents.append({"file": f"{clip}#{s['i']}", "src": clip, "caption": ix["shots"][s["i"]]["caption"],
                         "emb": emb[s["i"]]})
    return {"ents": ents, "emb": np.stack([e["emb"] for e in ents])}


def broll_choice(proj: str, grid, moments: Dict, seq: List[str]) -> Dict:
    g = json.load(open(os.path.join(SYNTH_DIR, proj, "gold.json")))
    lib = library(proj, g)
    ents, P = lib["ents"], lib["emb"]
    moment_of = {wid: m for m in moments["moments"] for wid in m["word_ids"]}
    lines: List[Dict] = []
    for f in seq:
        if g["category"][f] != "voice_over":
            continue
        ws = sorted((w for w in grid.words if w.source_file == f), key=lambda w: w.start)
        wd = [{"start": w.start, "end": w.end, "text": w.text, "id": w.id} for w in ws]
        for l in broll.build_beats(broll.build_lines(wd)):
            under = [v for v in g["vo_words"] if v["vo_file"] == f and v["shot_file"]
                     and v["end"] > l["start"] and v["start"] < l["end"]]
            lines.append({"text": l["text"], "moment": moment_of.get(wd[l["a"]]["id"]),
                          "gold": {v["shot_file"] for v in under}, "under": [v["shot_file"] for v in under]})
    lines = [l for l in lines if l["gold"]]
    n = len(lines)
    def hit(j: Optional[int], l: Dict) -> bool:
        return j is not None and ents[j]["file"] in l["gold"]
    def greedy(order_rows) -> List[int]:
        used, out = set(), []
        for row in order_rows:
            j = next(int(x) for x in row if int(x) not in used)
            used.add(j)
            out.append(j)
        return out
    res = {}
    Qw = shot_index.embed_texts([l["text"] for l in lines])
    words_pick = greedy(np.argsort(-(Qw @ P.T), axis=1))
    res["words SigLIP"] = (words_pick, None)
    visuals, uv = broll.write_visuals(moments, lines)
    Qv = shot_index.embed_texts(visuals)
    res["visual SigLIP"] = (greedy(np.argsort(-(Qv @ P.T), axis=1)), None)
    sections, us = broll.assign_sections(moments, [e["caption"] for e in ents])
    line_secs = [(l["moment"] or {}).get("section") or None for l in lines]
    cands = broll.rank_candidates(Qv, P, sections, line_secs)
    res["visual + section first"] = ([c[0] for c in cands], cands)
    picks, up = broll.pick_shots(lines, visuals, cands, [e["caption"] for e in ents], sections)
    res["story-aware pick"] = (picks, cands)
    print(f"  B-roll for {n} voice-over beats, library {len(ents)} shots "
          f"({sum(e['src'] == proj for e in ents)} own, {sum(e['src'] != proj for e in ents)} decoys):", flush=True)
    out = {}
    for name, (choice, cl) in res.items():
        h = sum(hit(j, l) for j, l in zip(choice, lines))
        tw = sum(len(l["under"]) for l in lines)
        wh = sum(sum(1 for x in l["under"] if j is not None and ents[j]["file"] == x) for j, l in zip(choice, lines)) / tw
        decoy = sum(j is not None and ents[j]["src"] != proj for j in choice)
        none = sum(j is None for j in choice)
        extra = ""
        if cl is not None:
            r5 = sum(any(ents[j]["file"] in l["gold"] for j in c[:5]) for c, l in zip(cl, lines))
            extra = f", editor's shot in top-5 candidates {r5 / n:.0%}"
        print(f"    {name:24}: beat hit {h / n:.0%}, voice-over words under the editor's shot {wh:.0%}, "
              f"decoy {decoy / n:.0%}, none {none / n:.0%}{extra}", flush=True)
        out[name] = {"hit": h / n, "word_hit": wh, "decoy": decoy / n, "none": none / n}
    own_secs = {sections[k] for k, e in enumerate(ents) if e["src"] == proj}
    decoy_other = sum(sections[k] == broll.OTHER for k, e in enumerate(ents) if e["src"] != proj)
    print(f"    sections: decoys put in 'other' {decoy_other}/{sum(e['src'] != proj for e in ents)}; "
          f"own shots in sections {sorted(own_secs)}", flush=True)
    return {"lines": n, **out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", action="append")
    ap.add_argument("--no-shuffled", action="store_true")
    a = ap.parse_args()
    projs = a.project or sorted(p for p in os.listdir(SYNTH_DIR) if os.path.exists(os.path.join(SYNTH_DIR, p, "gold.json")))
    for proj in projs:
        print(f"== {proj}", flush=True)
        words = project_words(proj)
        grid, moments, so, seq = story(proj, "upload_order", words)
        if not a.no_shuffled:
            try:
                story(proj, "shuffled_order", words)
            except RuntimeError as e:                    # the job would fall back to recording order
                print(f"  story [shuffled_order]: FAILED ({str(e)[:120]})", flush=True)
        broll_choice(proj, grid, moments, seq)


if __name__ == "__main__":
    sys.exit(main())
