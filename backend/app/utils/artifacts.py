"""Artifact store (roadmap Phase 5): stage results saved as JSON files, so a re-run or an
edit does not repeat expensive stages.

Two kinds:
  content-keyed   get(stage, key) / put(stage, key, payload): key = hash of the inputs + versions
                  (e.g. transcription: audio CONTENT hash + model + settings). Shared across jobs.
  per job         save_job / load_job(job_id, name): the job's grid, plan, timeline, ... so the
                  job can be re-compiled after a restart (jobs_data_db is in memory only).

Files on disk under settings.artifact_dir (user decision 2026-10-06), behind this small
interface so it can move to Postgres later. Writes are atomic (tmp + rename). Every hit and
miss is logged.
"""

import hashlib
import json
import logging
import os
import tempfile
from typing import Any, Optional

from app.config import settings

logger = logging.getLogger("VlogForge.Artifacts")

_digests = {}   # (path, size, mtime) -> sha256: avoid re-hashing the same file within a process


def enabled() -> bool:
    return bool(getattr(settings, "enable_artifact_cache", True))


def make_key(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def file_digest(path: str) -> str:
    """sha256 of the file's content."""
    st = os.stat(path)
    memo = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if memo not in _digests:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _digests[memo] = h.hexdigest()
    return _digests[memo]


def _write(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, path)


def _read(path: str) -> Optional[Any]:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:                       # corrupt file: treat as a miss, loudly
        logger.error(f"[ARTIFACT] unreadable {path}: {e}")
        return None


def get(stage: str, key: str) -> Optional[Any]:
    payload = _read(os.path.join(settings.artifact_dir, stage, f"{key}.json"))
    logger.info(f"[ARTIFACT] {stage} {'HIT' if payload is not None else 'MISS'} {key[:12]}")
    return payload


def put(stage: str, key: str, payload: Any) -> None:
    _write(os.path.join(settings.artifact_dir, stage, f"{key}.json"), payload)


def save_job(job_id: str, name: str, payload: Any) -> None:
    _write(os.path.join(settings.artifact_dir, "jobs", job_id, f"{name}.json"), payload)


def load_job(job_id: str, name: str) -> Optional[Any]:
    return _read(os.path.join(settings.artifact_dir, "jobs", job_id, f"{name}.json"))
