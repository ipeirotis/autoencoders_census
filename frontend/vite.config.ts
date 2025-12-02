import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import path from 'path'
import { fileURLToPath } from 'url'
import expressPlugin from './express-plugin.ts' 


const __dirname = path.dirname(fileURLToPath(import.meta.url))

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    expressPlugin() // Custom plugin to integrate Express.js with Vite
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './client'),
      '@shared': path.resolve(__dirname, './shared'),
    },
  },
  server: {
    port: 8080,
    fs: {
      strict: false,
    },
  },
})
