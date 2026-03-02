#!/bin/sh
set -e

echo "[entrypoint] Configuring ALSA mixer..."

# Unmute PCH (card 1) — starts muted by default.
# These may fail if card numbering differs; non-fatal.
amixer -c 1 set Master unmute 80% 2>/dev/null || true
amixer -c 1 set Headphone unmute 80% 2>/dev/null || true
amixer -c 1 set Front unmute 80% 2>/dev/null || true

echo "[entrypoint] ALSA devices:"
arecord -l 2>/dev/null || true
aplay -l 2>/dev/null || true

echo "[entrypoint] Starting jarvis-orchestrator..."
exec jarvis-orchestrator "$@"
