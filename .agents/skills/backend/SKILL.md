---
name: "vlogforge-backend"
description: "The Python FastAPI backend for VlogForge — provides the REST+WebSocket API server, the full AI video editing pipeline (perception, reasoning, assembly), debug scripts, and a pytest test suite. Consult this folder as the top-level entry point for all backend changes."
---

# Module: backend

## 📌 Purpose & Responsibility
- The **backend monorepo root** for VlogForge. Runs as a FastAPI server on port 8000 (default).
- Houses the complete Python application under `app/`, all tests under `tests/`, and several one-off diagnostic scripts at root level.
- Entry point: `uvicorn app.main:app` launched from this directory.
- Diagnostic scripts: `check_egt.py`, `check_ffprobe.py`, `dump_egt.py`, `summarize_egt.py` — standalone tools for inspecting EGT/EDL outputs and verifying environment setup without running the full server.
- `test_classify.py` / `test_phase1.py` at root level are older standalone test scripts (not part of the `pytest` suite in `tests/`).
- `.env` file in this directory is the single source of `GEMINI_API_KEY` loaded by `config.py`.

## 🔄 Integration & Data Flow
- **Inputs**: Raw video files via the frontend's HTTP multipart upload to `POST /api/jobs`.
- **Outputs**: Processed `.mp4` files in `../outputs/`, EGT/EDL JSON in memory (`jobs_data_db`), per-job log files in `../logs/`, and interaction logs in `../logs/interactions_{date}.log`.
- **Interactions**:
  - `app/main.py` is the FastAPI application; started by `uvicorn`.
  - `app/tasks/orchestrator.py` manages the in-memory job store and background pipeline thread.
  - All uploaded files are stored under `../uploads/{job_id}/`.
  - Final rendered videos go to `../outputs/{job_id}.mp4`.
  - Conda environment defined by `../environment.yml` provides all Python dependencies (PySceneDetect, faster-whisper, google-genai, FFmpeg, etc.).

## 📂 Code Symbols & Key Files

- [check_egt.py](backend/check_egt.py): CLI tool to validate and pretty-print an EGT JSON file — useful for debugging perception output from a completed job.
- [check_ffprobe.py](backend/check_ffprobe.py): Verifies ffprobe binary discovery and video metadata parsing from the active conda environment.
- [dump_egt.py](backend/dump_egt.py): Dumps an EGT JSON from `docs/` to stdout in a human-readable table format.
- [summarize_egt.py](backend/summarize_egt.py): Prints aggregate statistics (segment count, type distribution, bad-take ratio) from an EGT file.
- [.env](backend/.env): Environment variable file. Must contain `GEMINI_API_KEY=...`. Loaded by `app/config.py` via pydantic-settings.

## 🌿 Subdirectories & Child Skills
- [app](.agents/skills/backend_app/SKILL.md): FastAPI application — API routes, data models, config, and all pipeline task + utility modules.
- [tests](.agents/skills/backend_tests/SKILL.md): Pytest suite for scene detection and EDL generation correctness.
