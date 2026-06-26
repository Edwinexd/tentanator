# syntax=docker/dockerfile:1
FROM node:22-bookworm-slim AS builder
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --include=dev
COPY . .
RUN npm run build

# --- runtime stage ---
FROM node:22-bookworm-slim
WORKDIR /app

# Nitro emits a self-contained server bundle under .output (server + client
# assets), so no node_modules or source are needed at runtime.
COPY --from=builder /app/.output ./.output

EXPOSE 3000
ENV NODE_ENV=production
ENV PORT=3000
CMD ["node", ".output/server/index.mjs"]
