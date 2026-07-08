import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
      },
      '/ws': {
        target: 'ws://127.0.0.1:8000',
        ws: true,
        changeOrigin: true,
        // Don't crash Vite when the backend drops the WS connection during long Gemini retries
        configure: (proxy) => {
          proxy.on('error', (err) => {
            // Suppress EPIPE/connection reset errors - frontend will auto-reconnect
            if (err.code === 'EPIPE' || err.code === 'ECONNRESET' || err.code === 'ECONNREFUSED') {
              return;
            }
            console.error('[vite-proxy] WS error:', err.message);
          });
        },
      }
    }
  }
})
