import { useEffect, useRef, useState } from 'react';

export function useWebSocket(jobId, onMessage) {
  const [status, setStatus] = useState('connecting');
  const wsRef = useRef(null);
  const onMessageRef = useRef(onMessage);

  // Keep the callback ref up-to-date
  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  useEffect(() => {
    if (!jobId) return;

    setStatus('connecting');

    // Dynamically build the WebSocket URL based on current host/protocol
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/${jobId}`;

    console.log(`Connecting to WebSocket: ${wsUrl}`);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      setStatus('connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (onMessageRef.current) {
          onMessageRef.current(data);
        }
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setStatus('error');
    };

    ws.onclose = (event) => {
      console.log(`WebSocket closed: Code ${event.code}, Reason ${event.reason}`);
      setStatus('disconnected');
    };

    return () => {
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
      }
    };
  }, [jobId]);

  return { status, ws: wsRef.current };
}
