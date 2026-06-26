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

COPY --from=builder /app/package.json /app/package-lock.json ./
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/public ./public
COPY --from=builder /app/.env.example ./

EXPOSE 3000
ENV NODE_ENV=production
CMD ["node", "dist/server/server.js"]
