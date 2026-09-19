/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The frontend is a standalone SPA. It talks to the Phase 8 FastAPI backend at
// VITE_API_BASE_URL (see .env.example); there is no dev proxy, so the backend's
// explicit CORS allow-list (configs/backend.yaml -> api.cors_allowed_origins)
// is exercised exactly as in a real deployment.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, strictPort: true },
  preview: { port: 5173, strictPort: true },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
    restoreMocks: true,
  },
});
