import { type Plugin, type ViteDevServer } from 'vite';
import { createServer as createExpressServer } from './server/index';

/**
 * Vite plugin that integrates Express server into the dev server
 * This allows API routes to work during development
 */

// returns a vite plugin object 
// plugins extend vite's functionality
export default function expressPlugin(): Plugin {
  return {
    name: 'express-plugin',
    
    // called when Vite server starts 
    // Gives access to the server instance
    configureServer(server: ViteDevServer) {
      // Create an Express app
      const app = createExpressServer();
      
      // Add Express middleware to Vite's dev server
      // (integrate with Vite)
      server.middlewares.use(app);
      // now both vite (FE) and Express (API) run on same port (8080)
      
      console.log('Express server integrated with Vite');
    },
  };
}