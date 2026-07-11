---
name: "vlogforge-frontend-src-components"
description: "React UI components for VlogForge — upload panel, real-time processing monitor, full-featured video timeline editor, and download panel. Consult this folder when modifying the user interface, the video review/editing workflow, or WebSocket progress display."
---

# Module: frontend/src/components

## 📌 Purpose & Responsibility
- Contains all **leaf-level React components** that build the VlogForge UI.
- `UploadPanel.jsx`: Drag-and-drop video file picker with duplicate deduplication, per-file duration calculation, and file size display.
- `ProcessingMonitor.jsx`: Real-time WebSocket progress display with a circular SVG progress ring and a stage-by-stage status list. Exposes cancel functionality.
- `VideoPreview.jsx`: The **primary post-processing interface** — a full NLE-style review editor. Contains the assembled video player, original footage switcher, drag-reorderable EDL timeline, per-segment quality inspector (transcript, quality score, flags, tags), and quality threshold re-reasoning controls.
- `DownloadPanel.jsx`: Final download UI shown after assembly completes.
- `ContextInput.jsx`: Optional freeform context text input for the user to describe their vlog to the AI.

## 🔄 Integration & Data Flow
- **Inputs**:
  - `UploadPanel.jsx`: `files` (state array), `setFiles` callback, `isSubmitting` flag — all from `App.jsx`.
  - `ProcessingMonitor.jsx`: `jobId` string, `onComplete`, `onFailed`, `onReset`, `onCancel` callbacks — from `App.jsx`. Uses the `useWebSocket` hook internally.
  - `VideoPreview.jsx`: `jobId`, `downloadUrl`, `onReset`, `onReEdit` — fetches EGT, EDL, transcript from backend REST endpoints on mount.
  - `DownloadPanel.jsx`: `downloadUrl`, `onReset`.
  - `ContextInput.jsx`: `contextText`, `setContextText`.
- **Outputs**:
  - `UploadPanel.jsx`: Updates parent `files` state.
  - `ProcessingMonitor.jsx`: Fires `onComplete(downloadUrl)` or `onFailed()` based on WebSocket messages.
  - `VideoPreview.jsx`: Sends `POST /api/jobs/{jobId}/re-render` with modified EDL; sends `POST /api/jobs/{jobId}/re-reason` with new quality threshold.
- **Interactions**:
  - `ProcessingMonitor` ↔ `useWebSocket` hook: Subscribes to `/ws/{jobId}` for live stage/progress updates.
  - `VideoPreview` fetches `/api/jobs/{jobId}/egt`, `/api/jobs/{jobId}/edl`, `/api/jobs/{jobId}/transcript` on mount.
  - `VideoPreview` implements `segmentsMatch()` to correlate EDL entries with EGT transcript segments (supports both legacy and new EDL field name conventions).

## 📂 Code Symbols & Key Files

- [UploadPanel.jsx](frontend/src/components/UploadPanel.jsx): Drag-and-drop zone built with `react-dropzone`. Reads video duration client-side via a hidden `<video>` element. Deduplications by `(name, size)` key. Accepts `.mp4 .mov .avi .mkv`.

- [ProcessingMonitor.jsx](frontend/src/components/ProcessingMonitor.jsx): Ordered STAGES list drives the stage-progress UI. SVG circular progress ring (r=65, circumference computed). `getStageStatus()` maps current stage to `completed/active/waiting/failed` for each stage item. Heartbeat messages (`stage === 'heartbeat'`) are silently ignored.

- [VideoPreview.jsx](frontend/src/components/VideoPreview.jsx): The most complex component (~1172 lines). Key behaviors:
  - [segmentsMatch](frontend/src/components/VideoPreview.jsx#L8-L24): Temporal overlap matching between EDL entries and EGT transcript segments (>0.5s overlap or start within 1.6s).
  - [QUALITY_FLAG_ICONS](frontend/src/components/VideoPreview.jsx#L29-L42): Emoji+label mapping for quality flag display in segment inspector.
  - Layout: Three resizable panes (left sidebar: transcript/EGT, center: video player, right: segment inspector) + resizable bottom timeline.
  - View modes: `assembled` (plays final rendered MP4) and `original` (plays raw uploaded footage files).
  - EDL timeline: drag-reorderable clip strips with trim handles.
  - Re-reason panel: quality threshold slider → triggers `POST /api/jobs/{jobId}/re-reason`.

- [DownloadPanel.jsx](frontend/src/components/DownloadPanel.jsx): Minimal panel showing a download button linking to `/api/jobs/{jobId}/download`.

- [ContextInput.jsx](frontend/src/components/ContextInput.jsx): Controlled textarea for the user-provided vlog context note (e.g., "Travel vlog in Tokyo, focus on street food").
