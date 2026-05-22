import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Lean single-bundle build for embedding inside an iframe.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
