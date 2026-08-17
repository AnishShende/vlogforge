# VlogForge Architecture Audit Notes

After a comprehensive independent crawl of the codebase (`backend/` and `frontend/`), here are the factual findings compared against the initial outline:

## 1. INGESTION (Stage 1)
- **Status in Code:** DONE / GREEN
- **Details:** Fully implemented in `backend/app/tasks/ingest.py` and `backend/app/routers/upload.py`. The system uses Tus resumable upload and triggers FFmpeg for CFR transcode and audio extraction.

## 2. PERCEPTION (Pass 1)
- **Status in Code:** DONE / GREEN
- **Details:** 
  - **Transcription:** Implemented in `backend/app/tasks/transcribe.py` (Gemini STT / Whisper).
  - **Scene Detection & Subdivision:** Implemented in `backend/app/tasks/scene_detect.py`. *Note: It uses dynamic thresholds for speech gaps based on `target_duration`, which wasn't mentioned in the outline.*
  - **Visual Analysis:** Implemented in `backend/app/tasks/analyze.py` (Gemini Vision).
  - **Quality Scoring:** Implemented in `backend/app/tasks/score.py` (yields `quality_score` and `is_bad_take`).
  - **EGT Assembly:** Validated and produced via `backend/app/tasks/egt.py`.

## 3. REASONING (Pass 2)
- **Status in Code:** PARTIALLY WIRED UP / YELLOW
- **Correction on Hypothesis:** The outline hypothesized a "size-based branch between a single-pass approach and a hierarchical/chaptered approach for long footage." 
  - **Finding:** *This branch does NOT exist in code.* 
  - `backend/app/tasks/edl.py` implements `generate_edl`, which calls `generate_edl_llm` for Pass 1 LLM Reasoning. If the LLM reasoning fails, it falls back to a Phase 0 mechanical chronological filter. There is no hierarchical logic for long footage at this time.
- **Additional Findings:** The reasoning phase also includes a robust `_validate_priority_consistency` function and an Adjacency Risk Safeguard to sanitize the LLM's output.

## 4. MECHANICAL BUDGET ENFORCEMENT
- **Status in Code:** DONE / GREEN
- **Details:** Implemented natively within `backend/app/tasks/edl.py` as `_enforce_budget`. It performs "Tier 3 graduated budget enforcement" (Phases A through G), which trims padding, applies proportional core trimming, and drops LOW/MEDIUM priority clips to gracefully hit the target duration without violating narrative constraints.

## 5. ASSEMBLY (Pass 3)
- **Status in Code:** DONE / GREEN
- **Details:** Implemented in `backend/app/tasks/assemble.py`. It reads the final EDL and executes FFmpeg for the final render.

## 6. FRONTEND / REVIEW UI
- **Status in Code:** DONE / GREEN
- **Details:** A full React SPA (`frontend/src/App.jsx`) is in place. Components like `UploadPanel.jsx`, `ProcessingMonitor.jsx`, and `VideoPreview.jsx` exist. 
- Real-time updates are driven by WebSockets (connected via `backend/app/tasks/orchestrator.py`).

## 7. AUTH / PROJECT MANAGEMENT
- **Status in Code:** DONE / GREEN
- **Details:** Fully implemented. 
  - Backend uses JWT for authentication (`backend/app/routers/auth.py`), and manages isolated projects via Postgres (`Project` and `VideoFile` models).
  - Frontend has explicit `Login.jsx` and `Register.jsx` components for user accounts.

## 8. Missing from Outline (Unmentioned Features Found in Code)
- **Re-reason Pipeline:** `backend/app/main.py` exposes `/api/jobs/{job_id}/re-reason`. This allows users to adjust the `quality_threshold` on the frontend, triggering a fast recalculation of the `is_bad_take` flags and a new EDL generation *without* re-running the expensive Pass 1 Perception.
- **Re-render Pipeline:** `backend/app/main.py` exposes `/api/jobs/{job_id}/re-render`. This allows the UI Timeline Editor to send a mutated EDL to trigger a rapid re-assembly (Pass 3 only) via FFmpeg.
- **Dynamic Pacing Adjustments:** The `target_duration` affects not only the budget enforcement but also the scene subdivision parameters (long scene thresholds and speech gap ratios) dynamically during Phase 1.
