---
name: "vlogforge-backend-tests"
description: "Pytest test suite for VlogForge backend pipeline components — specifically scene detection correctness and EDL/EGT generation validation. Consult this folder when writing new tests, debugging pipeline regression issues, or verifying scene detection behavior."
---

# Module: backend/tests

## 📌 Purpose & Responsibility
- Contains **integration and unit tests** for the backend pipeline.
- `test_scene_detect.py`: Tests the two-pass cascade scene detection pipeline — verifying ContentDetector, AdaptiveDetector subdivision, short scene merging, fallback behavior, and the integration path through `ingest_video()`.
- `test_edl_repair.py` / `test_edl.py`: Tests Tier 3 EDL generation logic — validates the deterministic fallback algorithm (padding trimming, adjacency pre-pass, LOW/MEDIUM drops, and CRITICAL Phase D halting).
- Tests are run with `pytest` from the `backend/` directory and use real file system paths but mock or stub LLM/FFmpeg calls where appropriate.

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
