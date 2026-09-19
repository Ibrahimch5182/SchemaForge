/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

// The frontend is a standalone SPA. It talks to the Phase 8 FastAPI backend at
// VITE_API_BASE_URL (see .env.example); there is no dev proxy, so the backend's
// explicit CORS allow-list (configs/backend.yaml -> api.cors_allowed_origins)
// is exercised exactly as in a real deployment.
export default defineConfig(({ mode }) => {
  // Phase 11: a production bundle must never silently fall back to the localhost dev default.
  if (mode === "production") {
    const apiUrl = (loadEnv(mode, process.cwd(), "VITE_").VITE_API_BASE_URL ?? process.env.VITE_API_BASE_URL ?? "").trim();
    if (!apiUrl) throw new Error("VITE_API_BASE_URL must be set for a production build (public https URL of the SchemaForge backend).");
  }
  return {
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
};
});
