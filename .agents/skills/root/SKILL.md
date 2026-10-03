---
name: "vlogforge-root"
description: "The root directory of the VlogForge monorepo. Consult this entry point to understand the overall architecture, workspace setup, and how to navigate to the backend API or frontend React application. Deep dive hierarchically through all skill files starting from this one and understand the full context."
---

# Module: vlogforge

## 📌 Purpose & Responsibility
- This is the **workspace root** for VlogForge, an AI-powered vlog creation tool.
- It contains both the FastAPI Python backend and the React + Vite frontend in a monorepo structure.
- Also holds project-level documentation, reference files, and environment definitions (like the Conda environment).
- You would consult this directory to understand the highest-level split of the codebase or to manage workspace-wide dependencies.

## 🔄 Integration & Data Flow
- **Inputs**: None directly. This is a structural container.
- **Outputs**: None directly.
- **Interactions**:
  - Developers will typically split their terminal into two: one running `uvicorn` in `backend/` and one running `npm run dev` in `frontend/`.
  - Global IDE configurations and custom AI agent skills (`.agents/skills`) reside here.

## 📂 Code Symbols & Key Files
- **[docker-compose.yml](docker-compose.yml)**: Defines the Postgres database service for the backend.
- **[environment.yml](environment.yml)**: Conda environment specification. Defines the Python version and major dependencies (FastAPI, PySceneDetect, faster-whisper, google-genai) required for the backend.
- **[.gitignore](.gitignore)**: Global git ignore rules preventing uploads, outputs, and node_modules from being committed.
- **[Excalidraw Diagrams](*.excalidraw)**: Various architecture and planning diagrams (`vlogforge-architecture.excalidraw`, `vlogforge-phase-plan.excalidraw`, etc.).

## 🛠️ Workspace Conventions
- **Code quality**: clean, readable, maintainable; follow Python and React best practices (see `.agents/AGENTS.md`).
- **Branches**: `feat/…`, `fix/…`, `chore/…`. Do not commit directly to `main`.
- **Commits**: Conventional Commits — `<type>(<scope>): <subject>` (`feat`, `fix`, `docs`, `refactor`, `test`, `chore`), imperative mood, keep commits atomic.
- **Hygiene**: never commit scratch/debug artifacts (`*.log`, `*.db`, `.DS_Store`, `patch_*`/`scratch_*`/`dump_*` scripts) or the large local data dirs (`uploads/`, `outputs/`, `logs/`, `test-videos/`, `mocks/`) — all are gitignored.

## 🌿 Subdirectories & Child Skills
- **[backend](backend/SKILL.md)**: The Python FastAPI backend — handles video perception, AI reasoning, and FFmpeg assembly.
- **[frontend](frontend/SKILL.md)**: The React + Vite frontend — provides the drag-and-drop UI and the timeline review editor.
- **[docs](docs/SKILL.md)**: Contains sample EGT/EDL JSON files and diagnostic outputs for reference.
