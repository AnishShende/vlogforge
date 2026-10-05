"""Gold annotations for the eval set (Archdoc roadmap Phase 0 / S2).

Clip-agnostic. One file per clip at ``eval-set/<clip_id>/gold.json``, listed in
``eval-set/manifest.json``. Spans are SOURCE time ranges plus their text, so the
format does not depend on word IDs (Phase 1) or on any one pipeline version.

Semantics (what the S3 eval checks):
  keep      must appear in the output exactly once, with no clipped words.
            quality="imperfect" marks unique speech with no clean take: it must
            still be kept (Archdoc §2.5: classify before excluding).
  exclude   must not appear in the output (retake attempts, stutters, ...).
  optional  either choice is fine (humour, asides, texture; Archdoc §2.5):
            reported, never pass/fail.
  Speech in no list is unlabelled: reported, never pass/fail.

Retakes repeat the same words, so "must not appear" cannot be a text check;
it is a source-range check, which is why exclude carries times.
"""

import hashlib
import json
import os
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = 1
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVAL_SET_DIR = os.path.join(REPO_ROOT, "eval-set")
MANIFEST_PATH = os.path.join(EVAL_SET_DIR, "manifest.json")

ExcludeReason = Literal[
    "retake",            # an earlier/worse attempt of a kept line
    "stutter",           # repeated words/restart inside one attempt
    "false-start",       # abandoned fragment, no completed version here
    "production-talk",   # "let me do that again", "is it recording"
    "unreviewed",        # draft only: a human has not classified it yet
    "other",
]


class Source(BaseModel):
    file: str            # basename used as source_file throughout the pipeline
    path: str            # relative to repo root
    sha256: str          # identity check: catches stale / swapped paths


class KeepSpan(BaseModel):
    id: str
    source_file: str
    start: float
    end: float
    text: str
    quality: Literal["clean", "imperfect"] = "clean"
    edges: Literal["snapped", "hand"] = "snapped"   # "hand": verified by ear/level, never auto-snapped
    note: str = ""


class OptionalSpan(BaseModel):
    id: str
    source_file: str
    start: float
    end: float
    text: str
    edges: Literal["snapped", "hand"] = "snapped"   # "hand": verified by ear/level, never auto-snapped
    note: str = ""


class ExcludeSpan(BaseModel):
    id: str
    source_file: str
    start: float
    end: float
    text: str
    reason: ExcludeReason
    edges: Literal["snapped", "hand"] = "snapped"   # "hand": verified by ear/level, never auto-snapped
    note: str = ""


class Gold(BaseModel):
    schema_version: int = SCHEMA_VERSION
    clip_id: str
    status: Literal["draft", "reviewed"]
    sources: List[Source]
    covers: List[str] = Field(default_factory=list)
    keep: List[KeepSpan]
    exclude: List[ExcludeSpan] = Field(default_factory=list)
    optional: List[OptionalSpan] = Field(default_factory=list)
    drafted_from: Optional[dict] = None   # provenance of the draft, if any
    review: Optional[dict] = None         # who/when/what changed at human review

    @model_validator(mode="after")
    def _check(self):
        errs = []
        if self.schema_version != SCHEMA_VERSION:
            errs.append(f"schema_version {self.schema_version} != {SCHEMA_VERSION}")
        files = {s.file for s in self.sources}
        spans = ([("keep", s) for s in self.keep] + [("exclude", s) for s in self.exclude]
                 + [("optional", s) for s in self.optional])
        ids = [s.id for _, s in spans]
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            errs.append(f"duplicate span ids: {sorted(dup)}")
        for kind, s in spans:
            if s.source_file not in files:
                errs.append(f"{kind} {s.id}: source_file {s.source_file!r} not in sources")
            if not (0 <= s.start < s.end):
                errs.append(f"{kind} {s.id}: invalid range {s.start}-{s.end}")
            if not s.text.strip():
                errs.append(f"{kind} {s.id}: empty text")
        # No overlap anywhere within a source: one label per moment (or unlabelled).
        by_file = {}
        for kind, s in spans:
            by_file.setdefault(s.source_file, []).append((s.start, s.end, f"{kind}:{s.id}"))
        for f, rs in by_file.items():
            rs.sort()
            for (a0, a1, an), (b0, b1, bn) in zip(rs, rs[1:]):
                if b0 < a1:
                    errs.append(f"{f}: {an} [{a0}-{a1}] overlaps {bn} [{b0}-{b1}]")
        if self.status == "reviewed" and any(s.reason == "unreviewed" for s in self.exclude):
            errs.append("status=reviewed but some exclude spans still have reason=unreviewed")
        if errs:
            raise ValueError("invalid gold:\n  " + "\n  ".join(errs))
        return self


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def gold_path(clip_id: str) -> str:
    return os.path.join(EVAL_SET_DIR, clip_id, "gold.json")


def load_gold(clip_id: str, verify_sources: bool = True) -> Gold:
    with open(gold_path(clip_id)) as f:
        gold = Gold.model_validate(json.load(f))
    if gold.clip_id != clip_id:
        raise ValueError(f"gold.clip_id {gold.clip_id!r} != directory {clip_id!r}")
    if verify_sources:
        for s in gold.sources:
            p = os.path.join(REPO_ROOT, s.path)
            if not os.path.exists(p):
                raise FileNotFoundError(f"{clip_id}: source missing: {s.path}")
            if sha256_file(p) != s.sha256:
                raise ValueError(f"{clip_id}: source hash mismatch: {s.path}")
    return gold


def save_gold(gold: Gold) -> str:
    Gold.model_validate(gold.model_dump())   # re-run validation before writing
    path = gold_path(gold.clip_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(gold.model_dump(), f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def all_spans(gold: Gold):
    """Every labelled span, any kind."""
    return list(gold.keep) + list(gold.exclude) + list(gold.optional)
