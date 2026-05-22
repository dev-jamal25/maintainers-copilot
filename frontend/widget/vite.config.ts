import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Lean build for embedding inside an iframe. Stable asset filenames (no content hash) so the API
// embed route can reference assets/widget.js + assets/widget.css without reading a manifest.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    sourcemap: false,
    rollupOptions: {
      output: {
        entryFileNames: "assets/widget.js",
        chunkFileNames: "assets/[name].js",
        assetFileNames: "assets/widget.[ext]",
      },
    },
  },
});
