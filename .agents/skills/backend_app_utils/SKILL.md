---
name: "vlogforge-backend-app-utils"
description: "Low-level utility modules providing FFmpeg video processing, Google Gemini LLM integration, and structured interaction logging for the VlogForge pipeline. Consult this folder when working on video I/O operations, AI API calls, model fallback logic, or pipeline audit logging."
---

# Module: backend/app/utils

## 📌 Purpose & Responsibility
- Provides **reusable, stateless utilities** that all task modules depend on.
- `ffmpeg.py`: All video I/O operations — probing, audio extraction, keyframe extraction, CFR transcoding, clip trimming/scaling/normalization, crossfade concatenation, single-pass filtergraph assembly, and fade effects.
- `llm.py`: All Gemini API interactions — initialization, retry/fallback logic, keyframe description, context synthesis, segment classification (both legacy and EGT-aware), rolling window context building, EDL LLM generation, STT, and reel selection.
- `interaction_logger.py`: Structured audit logging to daily log files. Records API-level user actions (create_job, cancel_job, download) and full pipeline completion summaries with EGT+EDL data.

## 🔄 Integration & Data Flow
- **Inputs**:
  - `ffmpeg.py` receives file paths and time boundaries from `tasks/ingest.py`, `tasks/assemble.py`, and `tasks/edl.py`.
  - `llm.py` receives image paths, transcript dicts, EGT segment dicts, and context strings from `tasks/analyze.py`, `tasks/score.py`, `tasks/edl.py`, and `tasks/transcribe.py`.
  - `interaction_logger.py` is called from `main.py` and `tasks/orchestrator.py`.
- **Outputs**:
  - `ffmpeg.py` writes files to disk (WAV audio, JPEG keyframes, MP4 clips, final video).
  - `llm.py` returns strings, lists of classified dicts, or EDL dicts to the calling task modules.
  - `interaction_logger.py` writes to `logs/interactions_{date}.log`.
- **Interactions**:
  - `safe_generate_content()` in `llm.py` wraps all Gemini API calls with exponential retry and automatic model fallback (gemini-2.5-flash → gemini-flash-latest → gemini-flash-lite-latest).
  - `run_ffmpeg_with_gpu_fallback()` in `ffmpeg.py` transparently retries h264_nvenc failures with CPU libx264.

## 📂 Code Symbols & Key Files

- [ffmpeg.py](backend/app/utils/ffmpeg.py): Core FFmpeg subprocess wrapper. Key functions:
  - [get_ffmpeg_path](backend/app/utils/ffmpeg.py#L11-L27) / [get_ffprobe_path](backend/app/utils/ffmpeg.py#L29-L45): Resolve conda-env or system-path binary locations.
  - [get_video_info](backend/app/utils/ffmpeg.py#L47-L63), [get_video_duration](backend/app/utils/ffmpeg.py#L65-L71), [has_audio_stream](backend/app/utils/ffmpeg.py#L73-L82): Video metadata inspection via ffprobe.
  - [extract_audio](backend/app/utils/ffmpeg.py#L84-L101): Extracts 16kHz mono WAV for Whisper/Gemini STT.
  - [extract_keyframe](backend/app/utils/ffmpeg.py#L103-L119): Seeks to a timestamp and saves a JPEG keyframe.
  - [transcode_to_cfr](backend/app/utils/ffmpeg.py#L138-L160): Re-encodes VFR video to strict 30fps CFR H.264 (prevents PySceneDetect/Whisper timestamp drift).
  - [process_clip](backend/app/utils/ffmpeg.py#L162-L212): Trims, scales to 1920x1080, normalizes audio to -14 LUFS (loudnorm). GPU-accelerated with CPU fallback.
  - [concatenate_clips_with_crossfade](backend/app/utils/ffmpeg.py#L254-L300+): Builds a dynamic `filter_complex` graph for sequential audio crossfades (75ms acrossfade). Falls back to hard-concat for >30 clips.
  - `apply_fade_effects` / `assemble_single_pass`: Final rendering passes (single-pass filtergraph preferred).

- [llm.py](backend/app/utils/llm.py): Gemini API integration hub. Key functions:
  - [init_gemini](backend/app/utils/llm.py#L17-L34): Lazy singleton initialization of the `genai.Client`. Returns False if API key missing (triggers mock mode).
  - [with_gemini_retry](backend/app/utils/llm.py#L36-L66): Decorator that parses `retryDelay` from 429 responses and implements exponential backoff up to 5 retries. Distinguishes daily quota exhaustion (immediate fail) from rate limiting (retry).
  - [safe_generate_content](backend/app/utils/llm.py#L68-L116): Wraps `_gemini_client.models.generate_content` with automatic model fallback chain (flash → flash-latest → flash-lite-latest).
  - [describe_keyframe](backend/app/utils/llm.py#L118-L144): Calls Gemini Flash Lite with a PIL image for 1-2 sentence visual descriptions.
  - [synthesize_context](backend/app/utils/llm.py#L146-L173): Builds the Context Document from strided transcript + visual samples.
  - [build_rolling_window_summary](backend/app/utils/llm.py#L175-L205): Constructs a preceding 180-second transcript window string for per-segment classification context.
  - `classify_segments`, `classify_egt_segments`: Per-segment LLM classification calls with rolling context.
  - `generate_edl_llm`, `select_reel_segments_llm`, `sequence_edl_segments`: LLM-powered EDL reasoning functions.
  - `transcribe_audio_gemini`: Gemini-based STT (used as primary before Whisper fallback).

- [interaction_logger.py](backend/app/utils/interaction_logger.py): Singleton `InteractionLogger` class.
  - [log_interaction](backend/app/utils/interaction_logger.py#L26-L32): Logs API action events as structured JSON.
  - [log_pipeline_completion](backend/app/utils/interaction_logger.py#L34-L66): Writes a detailed EGT+EDL summary to the interactions log at pipeline end.
  - `interaction_logger` singleton: module-level instance imported by `main.py` and `orchestrator.py`.
