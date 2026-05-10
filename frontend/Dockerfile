# Stage 1: build the React SPA
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
# vite build outputs to dist/; node-build.ts expects ./spa relative to cwd
RUN npm run build && mv dist spa

# Stage 2: runtime
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci
# Set NODE_ENV after npm ci so devDeps (tsx) are installed
ENV NODE_ENV=production
COPY --from=builder /app/spa ./spa
COPY server ./server
COPY shared ./shared
# Cloud Run injects PORT=8080 via env; node-build.ts reads process.env.PORT
EXPOSE 8080
CMD ["npx", "tsx", "server/node-build.ts"]
