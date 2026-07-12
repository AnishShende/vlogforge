---
name: "vlogforge-backend-app-tasks"
description: "The core AI pipeline task modules for VlogForge — each file implements one stage of the 3-pass perception/reasoning/assembly pipeline. Consult this folder when modifying ingestion, scene detection, transcription, visual analysis, quality scoring, EGT/EDL generation, or video assembly logic."
---

# Module: backend/app/tasks

## 📌 Purpose & Responsibility
- Implements the **8-stage AI pipeline** that transforms raw uploaded video files into an edited vlog.
- The pipeline is divided into 3 passes:
  - **Pass 1 (Perception/Tier 1)**: `ingest.py` → `scene_detect.py` → `transcribe.py` → `analyze.py` → `score.py` → `egt.py`
  - **Pass 2 (Reasoning/Tier 2 & 3)**: `edl.py` implements the **Neurosymbolic Propose-and-Repair Architecture**. It calls `generate_edl_llm()` for semantic narrative judgment (Tier 2), and then applies a strict 5-phase deterministic Python constraint solver (Tier 3) to enforce exact duration budgets and handle padding/drops.
  - **Pass 3 (Assembly)**: `assemble.py` (FFmpeg video rendering)
- `orchestrator.py` is the top-level coordinator that sequences all stages, manages job state, handles concurrency, and broadcasts WebSocket progress updates.
- `classify.py` is a legacy shim wrapping the older LLM classification path (pre-Phase 0).

## 🔄 Integration & Data Flow
- **Inputs**: Raw video file paths from `main.py` after upload; `context_text`, `vlog_genre`, `target_duration`, and `quality_threshold` from the API request.
- **Outputs**: An `EGTDocument` (stored in `jobs_data_db`), an EDL list (stored in `jobs_data_db`), and a final `.mp4` file on disk at `settings.output_dir/{job_id}.mp4`.
- **Interactions**:
  1. `orchestrator.py:start_pipeline()` is called from `main.py`.
  2. The orchestrator runs `ingest_video()` → `transcribe_audio()` → `subdivide_by_speech_gaps()` → `analyze_segments()` → `score_segments()` → `build_egt_document()` → `generate_edl()` → `assemble_vlog()` in sequence. Note the re-ordering where subdivision happens before visual analysis.
  3. At each stage, `broadcast_progress()` pushes WebSocket updates to the frontend via `main.py`'s WebSocket endpoint.
  4. `assemble.py` calls FFmpeg utilities from `app/utils/ffmpeg.py`.
  5. `edl.py` and `score.py` call LLM functions from `app/utils/llm.py`.

## 📂 Code Symbols & Key Files

- [ingest.py](backend/app/tasks/ingest.py): Entry point for per-video pre-processing. Computes SHA-256 hash, transcodes to CFR 30fps, extracts WAV audio, runs scene boundary detection, and generates initial `EGTSegment` objects with keyframe paths. Returns `{file_info, segments, cfr_path, source_file_hash}`.

- [scene_detect.py](backend/app/tasks/scene_detect.py): Two-pass cascade scene boundary detector. Pass 1 uses PySceneDetect `ContentDetector` for hard cuts; Pass 2 applies `AdaptiveDetector` only on scenes longer than `long_scene_threshold_sec`. Falls back to fixed 8-second intervals if both detectors fail. Key function: [detect_scenes](backend/app/tasks/scene_detect.py#L52-L204).

- [transcribe.py](backend/app/tasks/transcribe.py): Speech-to-text with Gemini Flash Lite STT as primary and local `faster-whisper` (large-v3) as fallback. [align_transcript_with_segments](backend/app/tasks/transcribe.py#L95-L157) uses an O(N+M) two-pointer sweep to map transcript words to EGT scene boundaries. Thread-safe lazy Whisper model loading via [get_whisper_model](backend/app/tasks/transcribe.py#L14-L51).

- [analyze.py](backend/app/tasks/analyze.py): Runs Gemini Vision `describe_keyframe()` on segments. For segments > 10s, performs **dense sampling** (extracting a keyframe every 5s) to generate a chronological timeline description, allowing the LLM to track when subjects enter/leave the frame. Synthesizes a Context Document via strided uniform sampling to avoid intro-bias. Key function: [analyze_segments](backend/app/tasks/analyze.py#L13-L75).

- [score.py](backend/app/tasks/score.py): Phase 1.4 — Quality scoring and segment-type classification. Calls `classify_egt_segments()` (Gemini Flash Lite) for semantic segment_type assignment, then applies 6-signal rule-based heuristics for `quality_score` (disfluency, duration, bad-take phrases, noise, speech density). [recompute_bad_takes](backend/app/tasks/score.py#L268-L289) enables fast threshold re-evaluation without re-running LLMs.

- [egt.py](backend/app/tasks/egt.py): Phase 1.5 — Assembles a validated `EGTDocument` from all perception sub-stages. Runs integrity validation (duplicate clip_ids, invalid timestamps). Provides serialization helpers `egt_to_serializable()` and `egt_from_serializable()`.

- [edl.py](backend/app/tasks/edl.py): Pass 2 — EDL generation. Implements the 3-Tier Neurosymbolic architecture:
  - **Tier 2 (Propose)**: Calls `generate_edl_llm()` to obtain an ordered EDL with `narrative_priority` (LOW/MEDIUM/CRITICAL) and `core` bounds.
  - **Tier 3 (Repair)**: Executes a 5-phase deterministic fallback algorithm to enforce the `target_duration`:
    - *Adjacency Pre-pass*: Upgrades `LOW` clips between `CRITICAL` clips to prevent jump cuts.
    - *Phase A*: Trims padding from `LOW`/`MEDIUM` clips.
    - *Phase B & C*: Drops `LOW` and `MEDIUM` clips based on quality score.
    - *Phase C.5*: Trims `CRITICAL` padding.
    - *Phase D*: Halts and warns if budget exceeded (never cuts into `CRITICAL` core payloads). Both paths apply [snap_boundary_to_speech](backend/app/tasks/edl.py#L26-L63) for clean cut boundaries.

- [assemble.py](backend/app/tasks/assemble.py): Pass 3 — Executes the EDL via FFmpeg. Primary path: `assemble_single_pass()` (single filtergraph). Fallback: multi-pass `process_clip` → `concatenate_clips_with_crossfade` → `apply_fade_effects`. Includes anti-hallucination [validate_edl_against_egt](backend/app/tasks/assemble.py#L28-L44) that blocks assembly if any clip_id doesn't resolve to the EGT.

- [orchestrator.py](backend/app/tasks/orchestrator.py): The pipeline coordinator. Manages global `jobs_db`, `jobs_data_db`, `websockets_db`. Runs the 8-stage pipeline in a background thread via `asyncio.to_thread`. Has a heartbeat thread to keep WebSocket connections alive during long Gemini waits. Also handles re-reasoning (`start_re_reasoning`) which re-runs only Pass 2 and Pass 3.

- [classify.py](backend/app/tasks/classify.py): Legacy shim. Calls the older `classify_segments()` LLM function for INTRO/OUTRO/HIGHLIGHT/FILLER/B_ROLL labeling. Superseded by `score.py`'s integration with `classify_egt_segments()`.

## 🌿 Subdirectories & Child Skills
- [reasoning](.agents/skills/backend_app_tasks_reasoning/SKILL.md): Preserves the full Phase 1 EDL generation algorithm with genre-aware scoring, reel selection, and LLM narrative sequencing — ready for promotion to replace the current Phase 0 `edl.py` fallback.
