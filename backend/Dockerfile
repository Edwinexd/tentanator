# syntax=docker/dockerfile:1
# --- build stage: compile the release binary (deps cached separately) ---
FROM rust:1-bookworm AS builder
WORKDIR /app

# Prime the dependency cache with a dummy crate so source edits don't rebuild
# the whole dependency graph (Turso/Axum are heavy).
COPY Cargo.toml Cargo.lock ./
RUN mkdir src \
 && echo 'fn main() {}' > src/main.rs \
 && cargo build --release \
 && rm -rf src

COPY src ./src
RUN touch src/main.rs && cargo build --release

# --- runtime stage: slim image with just the binary ---
FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/tentanator-backend /usr/local/bin/tentanator-backend

# Runs as root deliberately. docker-compose.yml bind-mounts the host repo root
# at /workspace (= TENTANATOR_DATA_DIR), and the backend writes its Turso DB and
# exports there. The mount keeps the host's ownership/permissions, so a fixed
# non-root container uid usually cannot write to it and the stack breaks. Unlike
# the results renderer, the backend executes no untrusted/student content, so the
# risk traded off here is low. Revisit if the data dir moves off a host bind mount.
ENV TENTANATOR_DATA_DIR=/data \
    TENTANATOR_BIND=0.0.0.0:8787
WORKDIR /data
EXPOSE 8787
CMD ["tentanator-backend"]
