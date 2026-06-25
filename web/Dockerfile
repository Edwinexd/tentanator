# syntax=docker/dockerfile:1
FROM node:22-bookworm-slim
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

EXPOSE 3000
# Dev server (SSR), reachable from the host. VITE_API_BASE is read from the env.
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
