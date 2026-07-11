---
name: "vlogforge-backend-app-tasks-reasoning"
description: "Contains the legacy Phase 1 EDL generation engine (edl_v1.py) that implements genre-aware, narrative-sequenced edit decision list construction. Consult this folder when working on advanced EDL algorithms, genre scoring, reel selection, or narrative LLM-sequencing."
---

# Module: tasks/reasoning

## 📌 Purpose & Responsibility
- Houses the **legacy / Phase 1 EDL reasoning pipeline** (`edl_v1.py`) — a more sophisticated successor to the Phase 0 mechanical filter.
- Implements **genre-specific score boosting** (gym, travel, makeup, daily), **reel/short-form selection** via LLM, **greedy middle-fill** with pacing constraints, and **LLM narrative sequencing** via `sequence_edl_segments`.
- Intentionally preserved separately from `tasks/edl.py` so that the Phase 0 chronological filter can remain simple while Phase 1 reasoning logic is ready to be promoted.
- Also provides `is_background_noise()` for detecting PA announcements, ambient noise, and non-speech audio in transcripts.

## 🔄 Integration & Data Flow
- **Inputs**: Classified scene dicts (with `label`, `score`, `video_file`, `start`, `end`), `total_raw_duration`, `target_duration_sec`, `context_doc`, `user_prompt`, `vlog_genre`, and optionally `transcript_segments` for boundary snapping.
- **Outputs**: A formatted EDL list of `{video_file, start_sec, end_sec, type}` dicts ready for FFmpeg assembly.
- **Interactions**: Called from `tasks/edl.py` (and historically from the orchestrator). Internally calls `app/utils/llm.py` functions `select_reel_segments_llm` and `sequence_edl_segments` for LLM-powered decisions.
- Not invoked in the current Phase 0 pipeline path; `tasks/edl.py` calls `generate_edl_llm()` from `utils/llm.py` instead, but `edl_v1.py` remains as the production-ready Phase 1 implementation.

## 📂 Code Symbols & Key Files
- [edl_v1.py](backend/app/tasks/reasoning/edl_v1.py): The full Phase 1 EDL generation engine. Contains:
  - [is_background_noise](backend/app/tasks/reasoning/edl_v1.py#L9-L42): Rule-based detector for PA announcements, static, ambient noise. Respects user context to avoid false positives.
  - [apply_genre_scoring_boost](backend/app/tasks/reasoning/edl_v1.py#L44-L98): Adds +0.15–0.35 score boosts based on genre keyword matches; penalizes background noise.
  - [snap_boundary_to_speech](backend/app/tasks/reasoning/edl_v1.py#L100-L132): Snaps clip start/end to nearest speech segment boundary (±1.5s tolerance).
  - [generate_edl](backend/app/tasks/reasoning/edl_v1.py#L134-L391): Top-level EDL builder. Branches into reel (≤60s) vs. long-form paths. Reel path uses LLM selection with Python greedy fallback. Long-form path uses greedy INTRO+MIDDLE+OUTRO fill with talking-time pacing limits.
- [__init__.py](backend/app/tasks/reasoning/__init__.py): Empty package marker.
