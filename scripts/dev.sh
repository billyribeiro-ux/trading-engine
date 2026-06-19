#!/usr/bin/env bash
# Start the whole dashboard with one command: API (:8000) + frontend (:5173).
#
# One-time setup first:
#   uv sync --extra api --extra ml
#   cd frontend && pnpm install && cd ..
#
# Then:  ./scripts/dev.sh      (Ctrl-C stops both)
# Open http://localhost:5173 and set your FMP key via the gear icon.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "→ API     http://127.0.0.1:8000"
uv run engine-api &
API_PID=$!
# Stop the API when this script (and the frontend) exits.
trap 'kill "$API_PID" 2>/dev/null || true' EXIT INT TERM

echo "→ Dashboard http://localhost:5173  (Ctrl-C to stop both)"
cd frontend && pnpm run dev
