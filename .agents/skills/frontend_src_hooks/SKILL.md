---
name: "vlogforge-frontend-src-hooks"
description: "Custom React hooks for VlogForge frontend — currently contains the WebSocket connection manager with exponential backoff reconnection. Consult this folder when modifying real-time communication, WebSocket lifecycle, or adding new shared stateful logic."
---

# Module: frontend/src/hooks

## 📌 Purpose & Responsibility
- Houses **reusable React hooks** that encapsulate cross-cutting state and side-effect logic.
- `useWebSocket.js`: The sole hook — manages a WebSocket connection to the VlogForge backend's `/ws/{jobId}` endpoint. Handles initial connection, message parsing, terminal state detection, and exponential backoff reconnection.

## 🔄 Integration & Data Flow
- **Inputs**: `jobId` (string), `onMessage` callback.
- **Outputs**: `{ status, ws }` — current connection status string (`connecting | connected | disconnected | error`) and the raw WebSocket reference.
- **Interactions**:
  - Used exclusively by `components/ProcessingMonitor.jsx`.
  - In development mode connects directly to `ws://localhost:8000` (bypasses Vite proxy, which can drop during long Gemini rate-limit silences).
  - In production, derives the WebSocket host from `window.location.host` (supports both `ws:` and `wss:`).
  - Heartbeat messages (`stage === 'heartbeat'`) are filtered and not forwarded to `onMessage`.
  - Once `stage` is `complete | failed | cancelled`, sets `isTerminalRef = true` to prevent reconnection attempts.

## 📂 Code Symbols & Key Files

- [useWebSocket.js](frontend/src/hooks/useWebSocket.js): Custom hook. Key implementation details:
  - [WS_BASE](frontend/src/hooks/useWebSocket.js#L6-L9): Conditional base URL — dev uses direct backend URL, prod uses current host.
  - [connect](frontend/src/hooks/useWebSocket.js#L25-L79): Creates a new WebSocket, wires `onopen/onmessage/onerror/onclose` handlers. On close (if non-terminal), schedules reconnect with exponential backoff: `min(1000 × 2^attempt, 15000)ms`.
  - Reconnect guard refs: `isMountedRef` (prevents reconnect after component unmount), `isTerminalRef` (prevents reconnect after job completion).
  - Cleanup: On unmount, clears reconnect timers and closes the WebSocket if still open.
