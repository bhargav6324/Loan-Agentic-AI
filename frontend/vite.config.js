import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/auth': {
        target: 'http://127.0.0.1:8000',
        bypass: function(req) {
          if (req.headers.accept?.includes('text/html')) {
            return false;
          }
        }
      },
      '/applications': {
        target: 'http://127.0.0.1:8000',
        bypass: function(req) {
          if (req.headers.accept?.includes('text/html')) {
            return false;
          }
        }
      },
      '/admin': {
        target: 'http://127.0.0.1:8000',
        bypass: function(req) {
          if (req.headers.accept?.includes('text/html')) {
            return false;
          }
        }
      },
    }
  }
})
