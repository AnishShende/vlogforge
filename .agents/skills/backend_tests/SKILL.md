---
name: "vlogforge-backend-tests"
description: "Pytest test suite for VlogForge backend pipeline components — specifically scene detection correctness and EDL/EGT generation validation. Consult this folder when writing new tests, debugging pipeline regression issues, or verifying scene detection behavior."
---

# Module: backend/tests

## 📌 Purpose & Responsibility
- Contains **integration and unit tests** for the backend pipeline. Run with `pytest` from the `backend/` directory.
- Scene detection: `test_scene_detect.py` — two-pass cascade (ContentDetector, AdaptiveDetector subdivision, short-scene merging, fallback, and the `ingest_video()` path).
- EDL generation/repair: `test_edl.py`, `test_edl_repair.py`, `test_edl_overlap.py`, `test_budget_edl.py`, `test_chunked_edl.py` — Tier 3 deterministic fallback (padding trimming, adjacency pre-pass, LOW/MEDIUM drops, CRITICAL Phase D halting, overlap/budget handling).
- Perception & reasoning: `test_batch_classification.py`, `test_e2e_transcript.py`, `test_orchestrator_warnings.py`, `test_quota_fallback.py` (LLM quota → fallback).
- Editing-quality features: `test_disfluency.py`, `test_retake_detect.py`, `test_filler_words.py`, `test_word_snap.py`, `test_word_timeline.py`, `test_part_b.py`.
- Tests use synthetic fixtures and real file-system paths but **mock LLM/FFmpeg** — unit tests must stay fast and must never hit real Gemini APIs or render real video.

## 🔄 Integration & Data Flow
- **Inputs**: Test fixtures with synthetic `EGTSegment` / `EGTDocument` objects or real video file references from `test-videos/`.
- **Outputs**: Pytest pass/fail results. No side effects on production data stores.
- **Interactions**:
  - `test_scene_detect.py` imports from `app.tasks.scene_detect` and (optionally) `app.tasks.ingest`.
  - `test_edl.py` imports from `app.models`, `app.tasks.edl`, `app.tasks.egt`, and `app.tasks.score`.
  - Both test files may call into `app.config.settings` for threshold defaults.

## 📂 Code Symbols & Key Files

- [test_scene_detect.py](backend/tests/test_scene_detect.py): Scene detection test suite (~300 lines). Tests:
  - `detect_scenes()` on video files from `test-videos/` directory.
  - That hard-cut heavy footage produces many ContentDetector segments.
  - That long talking-head footage triggers AdaptiveDetector subdivision.
  - Short scene merging correctness.
  - Fixed-interval fallback when both detectors return empty.

- [test_edl.py](backend/tests/test_edl.py): EDL generation test suite (~280 lines). Tests:
  - Tier 3 `generate_edl()` repair algorithm with synthetic LLM EDL documents.
  - Bad-take filtering, SILENCE filtering.
  - INTRO-first and OUTRO-last ordering.
  - `snap_boundary_to_speech()` correctness with transcript edge cases.
  - `build_egt_document()` integrity validation (duplicate clip_ids, invalid timestamps).

## 🛠️ Mocking Conventions & Anti-patterns
- Patch the LLM at its import site in the module under test, e.g. `@patch('app.tasks.score.safe_generate_content')`, and set `mock.return_value.text = '...'`. Never let a test reach the real Gemini client.
- Stub FFmpeg by patching `app.utils.ffmpeg.subprocess.run` (or the specific wrapper) — unit tests must not spawn real FFmpeg.
- Put shared fixtures (mock EGT docs, sample transcripts, DB sessions) in `conftest.py`.
- There is **no** `/api/health` endpoint — for a liveness `TestClient` check use the root route `GET /` (see `main.py`). Don't assert against endpoints that don't exist.
- Keep unit tests fast (sub-second) and never commit real API keys into test files.
