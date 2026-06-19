#!/usr/bin/env bash
# Institutional verify gate. Run before every commit / in CI.
#   - ruff lint (3.9-targeted; never rewrites code into 3.10 syntax)
#   - offline pytest with coverage (real-data tests are excluded; they need a key)
#   - the slots=True invariant (the brief's locked Python-3.9 constraint: must be 0)
#
# Usage: ./scripts/verify.sh   (from the repo root)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> ruff check"
uv run ruff check .

echo "==> ruff format --check"
uv run ruff format --check .

echo "==> pytest (offline; real-data excluded)"
uv run pytest -m "not realdata" --cov=engine --cov-report=term-missing

echo "==> slots=True invariant (must be 0)"
if grep -rnq "slots=True" engine/; then
  echo "FAIL: slots=True found in engine/ (Python 3.9 forbids dataclass slots here)"
  grep -rn "slots=True" engine/ || true
  exit 1
fi

echo "All gates green."
