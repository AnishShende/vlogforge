---
name: "vlogforge-backend"
description: "The Python FastAPI backend for VlogForge — provides the REST+WebSocket API server, the full AI video editing pipeline (perception, reasoning, assembly), and a pytest test suite. Consult this folder as the top-level entry point for all backend changes."
---

# Module: backend

## 📌 Purpose & Responsibility
- The **backend monorepo root** for VlogForge. Runs as a FastAPI server on port 8000 (default).
- Houses the complete Python application under `app/` and all tests under `tests/`.
- Entry point: `uvicorn app.main:app` launched from this directory.
- Database: Postgres initialized via Alembic migrations (`alembic.ini` + `alembic/`).
- `.env` file in this directory is the single source of `GEMINI_API_KEY` loaded by `config.py`.
- Keep this directory clean: throwaway debug/scratch scripts (`scratch_*.py`, `dump*.log`, `patch_*.py`, loose non-pytest `test_*.py`) do not belong here — put experiments in the scratchpad, and real tests under `tests/`.

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

- [alembic.ini](backend/alembic.ini) + [alembic/](backend/alembic/): Alembic configuration and migration versions for the Postgres schema. See the [app/database](.agents/skills/backend_app_db/SKILL.md) skill for the migration workflow.
- [.env](backend/.env): Environment variable file. Must contain `GEMINI_API_KEY=...`. Loaded by `app/config.py` via pydantic-settings.

## 🌿 Subdirectories & Child Skills
- [app](.agents/skills/backend_app/SKILL.md): FastAPI application — API routes, data models, config, and all pipeline task + utility modules.
- [tests](.agents/skills/backend_tests/SKILL.md): Pytest suite for scene detection and EDL generation correctness.
