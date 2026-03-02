# Multi-stage build: compile Rust, then copy binary to slim runtime.
# Target: linux/amd64 (server node)

FROM rust:1.85-bookworm AS builder

# ALSA dev headers needed for cpal to compile.
RUN apt-get update && apt-get install -y \
    libasound2-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY Cargo.toml Cargo.lock* ./
COPY src/ src/

RUN cargo build --release

# --- Runtime stage ---
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    libasound2 \
    alsa-utils \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/jarvis-orchestrator /usr/local/bin/

# Unmute PCH audio on startup (card 1, from Phase 1 findings).
# This runs before the orchestrator binary.
COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 9090

ENTRYPOINT ["entrypoint.sh"]
