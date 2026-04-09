import { type Plugin, type ViteDevServer } from 'vite';

/**
 * Vite plugin that integrates Express server into the dev server.
 * This allows API routes to work during development.
 *
 * IMPORTANT: The server module is loaded via dynamic `import()` inside
 * `configureServer`, NOT via a top-level static import. Vite loads plugin
 * modules at config-parse time for both `vite dev` AND `vite build`, but
 * `configureServer` is only called for `vite dev`. A static import would
 * pull in `server/index.ts` → `config/env.ts` → `cleanEnv(...)` at build
 * time, hard-failing `npm run build` unless every backend secret
 * (GOOGLE_CLOUD_PROJECT, SESSION_SECRET, etc.) is present — turning the
 * frontend build into a backend-env gate (Codex P1 r(defer-cleanenv)).
 */
export default function expressPlugin(): Plugin {
  return {
    name: 'express-plugin',

    // Called only when Vite's DEV server starts (not during `vite build`).
    async configureServer(server: ViteDevServer) {
      // Dynamic import: server module (and its env validation) is loaded
      // only now, when we actually need it.
      const { createServer: createExpressServer } = await import('./server/index');

      const app = createExpressServer();

      server.middlewares.use(app);

      console.log('Express server integrated with Vite');
    },
  };
}
