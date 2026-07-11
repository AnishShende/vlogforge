---
name: "vlogforge-frontend-src"
description: "The React source root for VlogForge's frontend — defines the application shell, global CSS design system, and assembles all components into a multi-step workflow. Consult this folder when modifying the app layout, step-based navigation, global styling, or job submission flow."
---

# Module: frontend/src

## 📌 Purpose & Responsibility
- The **React application root**. `main.jsx` bootstraps the React tree. `App.jsx` owns the top-level application state and orchestrates the multi-step workflow:
  1. **Step 1**: Upload Panel + context/genre/duration/threshold configuration.
  2. **Step 2**: Processing Monitor (real-time WebSocket progress).
  3. **Step 3**: Video Preview / Editor (post-processing review, EDL editing, re-render/re-reason).
- `index.css`: The **global design system** — custom CSS variables, typography (Google Fonts: Inter, Outfit), glassmorphism cards, dark/light theme tokens, animation keyframes, component-specific styles, and the full NLE-style editor layout (sidebar, canvas, timeline).

## 🔄 Integration & Data Flow
- **Inputs**: User file selections and configuration choices (vlog genre, target duration, quality threshold, context text).
- **Outputs**: Multipart `POST /api/jobs` request (via `fetch`) that initiates the backend pipeline. Navigates forward to Step 2 on success.
- **Interactions**:
  1. Step 1 → User configures and submits → `App.jsx` builds `FormData` and POSTs to `/api/jobs` → receives `job_id` → transitions to Step 2.
  2. Step 2 → `ProcessingMonitor` subscribes to `/ws/{jobId}` → on `complete` event with `download_url` → `App.jsx` transitions to Step 3.
  3. Step 3 → `VideoPreview` enables EDL timeline editing and re-render/re-reason → on reset → returns to Step 1.
  4. `App.jsx` also manages `theme` (`dark|light`) with localStorage persistence and a CSS `data-theme` attribute on `<html>`.

## 📂 Code Symbols & Key Files

- [App.jsx](frontend/src/App.jsx): Top-level application component. Key logic:
  - State: `step` (1/2/3), `files`, `contextText`, `jobId`, `downloadUrl`, `targetDuration`, `vlogGenre`, `qualityThreshold`, `theme`.
  - [CustomSelect](frontend/src/App.jsx#L7-L86): Inline dropdown component used for genre selection with glassmorphism menu styling.
  - Step 1 form includes `UploadPanel`, `ContextInput`, genre/duration/threshold controls, and a submit handler that builds `FormData` and POSTs to `/api/jobs`.
  - Theme toggle: persisted to `localStorage`, applied via `document.documentElement.setAttribute('data-theme', ...)`.

- [index.css](frontend/src/index.css): Full global design system (~960 lines). Key sections:
  - CSS custom properties (`:root`): Color palette (purple primary `#8b5cf6`, backgrounds, card glows), border radii, font sizes, transitions.
  - Dark/light theme token overrides via `[data-theme="light"]`.
  - Google Fonts import: Inter (body), Outfit (headings).
  - Animation keyframes: `fadeIn`, `slideUp`, `pulse-ring`, `shimmer`.
  - Layout: `.studio-layout` (3-pane grid), `.studio-canvas`, `.studio-sidebar`, `.studio-timeline`.
  - Component styles: `.card`, `.btn-primary`, `.btn-ghost`, `.input-field`, `.processing-ring-svg`, `.timeline-track`, `.segment-chip`.

- [main.jsx](frontend/src/main.jsx): Minimal React entry point — renders `<App />` into `#root`.

## 🌿 Subdirectories & Child Skills
- [components](.agents/skills/frontend_src_components/SKILL.md): UploadPanel, ProcessingMonitor, VideoPreview (NLE editor), DownloadPanel, ContextInput.
- [hooks](.agents/skills/frontend_src_hooks/SKILL.md): `useWebSocket` — WebSocket connection manager with exponential backoff reconnection.
