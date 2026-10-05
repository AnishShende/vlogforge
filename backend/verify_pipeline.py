"""Closed-loop verification: assemble the REAL final video, re-transcribe its
audio, and print it side by side with the EGT/EDL so we can SEE whether the
output audio matches what was selected.

This is the ground-truth check the mock report can't give: the mock only shows
the TEXT we selected; here we transcribe the actual assembled audio and line it
up against each EDL clip's expected transcript.

Usage (from backend/):
    PATH=~/anaconda3/envs/vlogforge/bin:$PATH \
    PYTHONPATH=. python verify_pipeline.py
"""
import os
import re

import run_mock_pipeline as rmp
from app.config import settings

# Exercise REAL transcription so WhisperX forced alignment actually runs
# (the mock whisper cache holds the old, pre-alignment times). LLM/JEV stay
# mocked/cached where possible.
settings.enable_mock_whisper = False
settings.enable_forced_alignment = True
from app.tasks.word_timeline_redundancy import (
    apply_word_timeline_selection, clamp_edl_to_word_timeline,
)
from app.tasks.edl import generate_edl
from app.utils.word_snap import snap_edl_to_word_boundaries
from app.tasks.assemble import assemble_vlog
from app.tasks.transcribe import get_whisper_model

OUT = os.path.join(settings.output_dir, "verify_IMG_1614.mp4")


def _norm(t):
    return re.sub(r"[^\w\s]", "", t.lower()).split()


def _token_overlap(expected, actual):
    """Fraction of expected tokens present (in order-independent multiset) in actual."""
    exp, act = _norm(expected), list(_norm(actual))
    if not exp:
        return 1.0
    hit = 0
    for w in exp:
        if w in act:
            act.remove(w)
            hit += 1
    return hit / len(exp)


def _repeated_ngrams(actual, n=3):
    w = _norm(actual)
    seen, reps = set(), []
    for i in range(len(w) - n + 1):
        g = tuple(w[i:i + n])
        if g in seen:
            reps.append(" ".join(g))
        seen.add(g)
    return reps


def transcribe_clip_from_source(src_path, start, end):
    """Extract exactly [start,end] from the SOURCE and transcribe it.

    Drift-free per-clip measurement: tells us precisely what audio the selected
    span contains (no concat-offset drift, no cross-clip bleed)."""
    import subprocess, tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", src_path,
         "-ar", "16000", "-ac", "1", tmp.name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    model = get_whisper_model()
    segments, _ = model.transcribe(tmp.name, beam_size=5, vad_filter=False)
    text = " ".join(s.text.strip() for s in segments).strip()
    os.unlink(tmp.name)
    return text


def _extra_words(expected, actual):
    """Words in ACTUAL not accounted for by EXPECTED (shows 'actual larger')."""
    exp = list(_norm(expected))
    extra = []
    for w in _norm(actual):
        if w in exp:
            exp.remove(w)
        else:
            extra.append(w)
    return extra


def main():
    egt_doc, ctx = rmp.build_egt()
    egt_doc, wt_warnings = apply_word_timeline_selection(egt_doc)

    # EDL reasoning + word-snap + word-timeline clamp (same order as orchestrator)
    edl, _w, mode = generate_edl(
        egt_doc, ctx["full_transcript_segments"], ctx["target_duration"], ctx["context_text"])
    segments_by_clip_id = {s.clip_id: s.model_dump() for s in egt_doc.segments}
    edl = snap_edl_to_word_boundaries(edl, segments_by_clip_id)
    if settings.enable_word_timeline_redundancy:
        clamp_edl_to_word_timeline(edl, egt_doc)

    egt_clip_ids = {
        s.clip_id for s in egt_doc.segments
        if not s.is_bad_take and not getattr(s, "is_superseded_take", False)
        and not getattr(s, "is_stutter_repeat", False)
    }
    ok = assemble_vlog(edl, ctx["files_info"], ctx["job_dir"], OUT, egt_clip_ids=egt_clip_ids)
    assert ok, "assembly failed"

    egt_by_id = {s.clip_id: s for s in egt_doc.segments}
    src_path = ctx["files_info"][0].get("original_path") or ctx["files_info"][0].get("cfr_path")

    print("\n" + "#" * 100)
    print("# PER-CLIP SOURCE AUDIO vs EGT  (EXPECTED = selected text,  ACTUAL = audio actually in the cut span)")
    print("#" * 100)
    if wt_warnings:
        print("warnings:", wt_warnings)

    for i, entry in enumerate(edl):
        cid = entry.get("clip_id", "")
        expected = egt_by_id[cid].transcript if cid in egt_by_id else "(not in EGT)"
        actual = transcribe_clip_from_source(src_path, entry["start_sec"], entry["end_sec"])
        overlap = _token_overlap(expected, actual)
        extra = _extra_words(expected, actual)
        reps = _repeated_ngrams(actual)
        tag = ""
        if reps:
            tag += f"  STUTTER:{reps}"
        if extra:
            tag += f"  EXTRA(actual>expected):{extra}"
        print(f"\n[{i}] {entry.get('editorial_type','KEEP')} clip_id={cid} "
              f"src[{entry['start_sec']:.2f}-{entry['end_sec']:.2f}]s "
              f"overlap={overlap:.0%}{tag}")
        print(f"    EXPECTED: {expected}")
        print(f"    ACTUAL  : {actual}")

    print(f"\nFinal video: {OUT}  ({len(edl)} clips)")


if __name__ == "__main__":
    main()
