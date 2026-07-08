import { useEffect, useRef, useState, useCallback } from 'react';

// In development, connect DIRECTLY to the backend WebSocket bypassing the Vite proxy.
// The Vite WS proxy crashes with EPIPE when the backend is silent for 50+ seconds
// (e.g. during Gemini rate-limit retries). Direct connection has no intermediary to fail.
const WS_BASE =
  import.meta.env.DEV
    ? 'ws://localhost:8000'
    : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`;

export function useWebSocket(jobId, onMessage) {
  const [status, setStatus] = useState('connecting');
  const wsRef = useRef(null);
  const onMessageRef = useRef(onMessage);
  const reconnectTimerRef = useRef(null);
  const reconnectAttemptsRef = useRef(0);
  const isMountedRef = useRef(true);
  const isTerminalRef = useRef(false); // true once job is complete/failed/cancelled

  // Keep the callback ref up-to-date
  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  const connect = useCallback(() => {
    if (!jobId || !isMountedRef.current || isTerminalRef.current) return;

    const wsUrl = `${WS_BASE}/ws/${jobId}`;
    console.log(`[WS] Connecting (attempt ${reconnectAttemptsRef.current + 1}): ${wsUrl}`);
    setStatus('connecting');

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[WS] Connected');
      reconnectAttemptsRef.current = 0;
      setStatus('connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        // Ignore heartbeat pings — they only exist to keep the connection alive
        if (data.stage === 'heartbeat') return;

        if (onMessageRef.current) {
          onMessageRef.current(data);
        }

        // Stop reconnecting once job has reached a terminal state
        if (data.stage === 'complete' || data.stage === 'failed' || data.stage === 'cancelled') {
          isTerminalRef.current = true;
        }
      } catch (err) {
        console.error('[WS] Failed to parse message:', err);
      }
    };

    ws.onerror = (error) => {
      console.error('[WS] Error:', error);
      setStatus('error');
    };

    ws.onclose = (event) => {
      console.log(`[WS] Closed: Code ${event.code}, Reason: ${event.reason || 'none'}`);
      setStatus('disconnected');

      // If the job is done, don't reconnect
      if (isTerminalRef.current || !isMountedRef.current) return;

      // Exponential backoff: 1s, 2s, 4s, 8s, max 15s
      const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 15000);
      reconnectAttemptsRef.current += 1;
      console.log(`[WS] Reconnecting in ${delay}ms...`);
      reconnectTimerRef.current = setTimeout(connect, delay);
    };
  }, [jobId]);

  useEffect(() => {
    if (!jobId) return;
    isMountedRef.current = true;
    isTerminalRef.current = false;
    reconnectAttemptsRef.current = 0;

    connect();

    return () => {
      isMountedRef.current = false;
      clearTimeout(reconnectTimerRef.current);
      const ws = wsRef.current;
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        ws.close();
      }
    };
  }, [jobId, connect]);

  return { status, ws: wsRef.current };
}
