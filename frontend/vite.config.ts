import { defineConfig } from "vite";

export default defineConfig({
  server: {
    port: 3001,
    proxy: {
      "/api": "http://localhost:8000",
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
});
