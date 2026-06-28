# syntax=docker/dockerfile:1

# --- bindings stage: emit the DTO TypeScript from the Rust structs ---
# The web type-check imports web/src/lib/generated/*.ts, which ts-rs writes when
# the backend's export_bindings_* tests run. They are not committed, so generate
# them here from the backend source (passed in as the `backend` build context)
# instead of depending on a host-side `cargo test` having run first. This keeps
# `docker compose up --build` self-contained on a fresh clone.
FROM rust:1-bookworm AS bindings
WORKDIR /app
# Prime the dependency cache with a dummy crate so backend source edits don't
# rebuild the whole graph (mirrors backend/Dockerfile).
COPY --from=backend Cargo.toml Cargo.lock ./
RUN mkdir src \
 && echo 'fn main() {}' > src/main.rs \
 && cargo build \
 && rm -rf src
COPY --from=backend src ./src
# ts-rs writes to ../../web/src/lib/generated relative to the crate (= /web/...).
RUN touch src/main.rs && cargo test export_bindings

FROM node:22-bookworm-slim AS builder
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --include=dev
COPY . .
# Overlay the freshly generated DTOs (absent from the build context on a clean
# tree since they are gitignored).
COPY --from=bindings /web/src/lib/generated ./src/lib/generated
RUN npm run build

# --- runtime stage ---
FROM node:22-bookworm-slim
WORKDIR /app

# Nitro emits a self-contained server bundle under .output (server + client
# assets), so no node_modules or source are needed at runtime.
COPY --from=builder /app/.output ./.output

# Run as the non-root user that ships with the node image. The Nitro server
# needs no host-writable mounts, so dropping root is safe; the COPY'd .output
# is world-readable.
USER node

EXPOSE 3000
ENV NODE_ENV=production
ENV PORT=3000
CMD ["node", ".output/server/index.mjs"]
