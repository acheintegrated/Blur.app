// vite.config.js — tightened for Electron + local venv/models nearby
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig(({ command }) => ({
  content: ["./index.html", "./src/**/*.{ts,tsx,js,jsx}"],
  theme: { extend: {} },
  plugins: [],

  base: command === 'serve' ? '/' : './',

  server: {
    port: 25329,
    strictPort: true,
    host: true,
    open: false,
    hmr: { overlay: true },
    // don't watch the world
    watch: {
      ignored: [
        '**/blur_env*/**',
        '**/models/**',
        '**/core/ouinet/**',
        '**/core/bin/**'
      ]
    },
    fs: {
      strict: true,
      // only allow project root
      allow: [process.cwd()]
    }
  },

  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'chrome120',
    rollupOptions: {
      // lock dep scan / inputs to ONLY our app HTML
      input: 'index.html'
    }
  },

  optimizeDeps: {
    // force scan to just index.html so it doesn’t dive into random HTML in your tree
    entries: ['index.html'],
    exclude: ['electron'], // renderer doesn’t need electron prebundled
  },

  clearScreen: false,
  envPrefix: 'VITE_',
}));
