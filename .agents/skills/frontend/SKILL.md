---
name: "vlogforge-frontend"
description: "The React + Vite frontend workspace for VlogForge. Consult this folder for the top-level SPA configuration, package dependencies, Vite setup, and as the entry point to the UI source code."
---

# Module: frontend

## 📌 Purpose & Responsibility
- This is the **top-level frontend directory** for VlogForge.
- It is a Single Page Application (SPA) built with React 18 and bundled via Vite.
- Defines all Node dependencies (`package.json`), Vite configuration (`vite.config.js`), and the root HTML template (`index.html`).
- The actual React application code lives inside `src/`.

## 🔄 Integration & Data Flow
- **Inputs**: Code modifications in `src/`, dependency additions via `npm install`.
- **Outputs**: Development server running on port 5173 (default Vite port) or static production bundles inside `dist/`.
- **Interactions**:
  - Run `npm run dev` in this directory to start the local development server.
  - The frontend makes HTTP REST and WebSocket requests to the backend server (typically running on port 8000).
  - `vite.config.js` may contain proxy rules to route `/api` and `/ws` traffic to the local backend during development.

## 📂 Code Symbols & Key Files
- **[package.json](frontend/package.json)**: Defines dependencies (React, Lucide icons, react-dropzone) and npm scripts (`dev`, `build`, `preview`).
- **[vite.config.js](frontend/vite.config.js)**: Vite bundler configuration. Handles React plugin integration and development server proxy settings.
- **[index.html](frontend/index.html)**: The root HTML template. Mounts the React app into `<div id="root"></div>`.

## 🌿 Subdirectories & Child Skills
- **[src](frontend/src/SKILL.md)**: The React source code root, containing the application shell, global CSS, components, and hooks.
