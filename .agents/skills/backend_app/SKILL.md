---
name: "vlogforge-backend-app"
description: "The FastAPI application package that defines the HTTP API, data models, configuration, and orchestrates the full AI editing pipeline. Consult this folder when modifying API endpoints, Pydantic schemas, application configuration, or the top-level pipeline trigger."
---

# Module: backend/app

## 📌 Purpose & Responsibility
- The **FastAPI application root** — defines all HTTP REST and WebSocket endpoints that the React frontend consumes.
- `main.py`: All route definitions (job creation, status, EGT/EDL/transcript retrieval, download, re-render, re-reason, raw video serving, and WebSocket connection).
- `models.py`: All Pydantic schemas shared across the entire backend — `EGTSegment`, `EGTDocument`, `EDLEntry`, `EDLItem` (legacy), `VideoFileInfo`, `JobStatus`, `WSProgressEvent`, `JobCreate`.
- `config.py`: Application settings loaded from `.env` via `pydantic-settings`. Controls model selection, quality threshold, scene detection parameters, and directory paths with automatic cross-platform path correction.

## 🔄 Integration & Data Flow
- **Inputs**: HTTP requests from the React frontend (`POST /api/jobs` with multipart video uploads, WebSocket connections on `/ws/{job_id}`).
- **Outputs**: `JobStatus` JSON responses, WebSocket progress events (`WSProgressEvent`), `FileResponse` for video downloads.
- **Interactions**:
  1. `POST /api/jobs` → saves uploaded files → calls `start_pipeline()` from `orchestrator.py` (spawns background thread).
  2. WebSocket `/ws/{job_id}` → registers connection with `orchestrator.py` → receives progress broadcasts during pipeline execution.
  3. `POST /api/jobs/{id}/re-render` → calls `assemble_vlog()` directly (skips Pass 1 and Pass 2).
  4. `POST /api/jobs/{id}/re-reason` → calls `start_re_reasoning()` (re-runs Pass 2 + Pass 3 only).
  5. `GET /api/jobs/{id}/egt|edl|transcript` → reads from `jobs_data_db` in `orchestrator.py`.
  6. All schemas in `models.py` are imported by both `main.py` and all task/utility modules.

## 📂 Code Symbols & Key Files

- [main.py](backend/app/main.py): FastAPI application entry point. Declares the `app` instance with CORS enabled. Key endpoints:
  - [create_job](backend/app/main.py#L56-L126): Validates file extensions, saves uploads, creates `JobStatus`, and fires `start_pipeline()`.
  - [re_render_job](backend/app/main.py#L231-L306): Accepts a modified EDL and re-runs assembly only (fast iteration for human editors).
  - [re_reason_job_endpoint](backend/app/main.py#L308-L327): Re-runs reasoning+assembly with a new quality threshold.
  - [websocket_progress_endpoint](backend/app/main.py#L329-L360): Maintains real-time progress channel; sends current job state immediately on (re)connect.

- [models.py](backend/app/models.py): Canonical data schemas. Key classes:
  - [EGTSegment](backend/app/models.py#L38-L88): The atomic unit of perception output. Carries clip_id, temporal bounds, transcript, visual_description, quality_score, quality_flags, segment_type, is_bad_take, tags, and structural cues.
  - [EGTDocument](backend/app/models.py#L91-L113): Collection of EGTSegments plus aggregate metadata. Has [validate_integrity](backend/app/models.py#L100-L113) to check duplicate clip_ids and invalid timestamps.
  - [EDLEntry](backend/app/models.py#L120-L133): Single edit decision with clip_id, source_file, start/end times, editorial_type (KEEP/INTRO/OUTRO), and human_modified metadata.
  - [generate_clip_id](backend/app/models.py#L11-L19): Deterministic SHA-256-based ID from (source_file, start_sec, end_sec) — ensures the same segment always gets the same ID.
  - [JobStatus](backend/app/models.py#L154-L166): Full job state broadcast over REST and WebSocket.

- [config.py](backend/app/config.py): `Settings` class (pydantic-settings). Controls:
  - `gemini_api_key` — loaded from `.env`.
  - `perception_model` / `reasoning_model` — model tier selection (Flash Lite vs Flash).
  - `quality_threshold`, `content_detector_threshold`, `adaptive_detector_threshold`, `long_scene_threshold_sec`, `min_scene_duration_sec` — tunable pipeline parameters.
  - Auto-translates Windows `D:` drive defaults to local workspace paths on non-Windows systems.
  - `settings` singleton instantiated at module level; consumed everywhere.

## 🌿 Subdirectories & Child Skills
- [tasks](.agents/skills/backend_app_tasks/SKILL.md): All pipeline stage implementations (ingest, transcribe, analyze, score, EGT build, EDL generate, assemble, orchestrate).
- [utils](.agents/skills/backend_app_utils/SKILL.md): FFmpeg subprocess wrappers, Gemini LLM clients, and interaction logging.
