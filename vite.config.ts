import solid from 'vite-plugin-solid';
import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => ({
  base: loadEnv(mode, process.cwd(), 'BASE_PATH').BASE_PATH ?? '/',
  plugins: [solid()],
  build: { target: 'es2022', chunkSizeWarningLimit: 700 },
}));
