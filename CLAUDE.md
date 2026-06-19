# trading-engine — project instructions for Claude Code

This file loads every session in this repo. It extends the global
`~/.claude/CLAUDE.md` (the floor) with what's specific to THIS project.
**Read `docs/AI_LANDMINES.md` before writing any code** — it lists the runtime
bugs already caught here; re-introducing one wastes the operator's time.

## What this is

A production-grade **Python trading engine on FMP data** + a thin **SvelteKit
dashboard**. It dissects trading sessions into named events, learns signals, and —
above all — **validates edges honestly**. The prime directive: *truth through
adversarial testing*. "No edge exists" is a valid, valued output. "100% accurate"
is **not a goal — it's the signature of a leak** (see AI_LANDMINES).

## How to run it

```bash
# deps (one-time)
uv sync --extra api --extra ml            # Python engine + API + research models
cd frontend && pnpm install && cd ..      # frontend (pnpm only — npm is blocked)

# start everything (API :8000 + dashboard :5173); Ctrl-C stops both
./scripts/dev.sh
# (or run the two halves manually: `uv run engine-api` and `cd frontend && pnpm run dev`)
```

Open the dashboard → **gear icon → paste FMP key → Validate & save** (no env var
needed; the key persists to `~/.config/fmp_engine/settings.json`, chmod 600).
`FMP_API_KEY` in the env still works as a fallback. The saved key is used by the
API **and** every CLI via `engine.settings.resolve_api_key()`.

**Everything is in the dashboard:** run screens, click a symbol to dissect it,
**Scan / Resolve / Export(CSV·XLSX)** the live journal, and Settings — all
point-and-click. Only the heavy **gauntlet** is CLI-only for now (runs minutes;
needs a background-job runner to not block the UI — a tracked next step).

CLIs (same actions, scriptable): `engine-session`, `engine-intraday`,
`engine-gaps`, `engine-forward` (`scan`/`resolve`/`report`/`gauntlet`, with
`--export FILE.{csv,xlsx}`).

## Verify gate (run before every commit)

```bash
uv run pytest -q -m "not realdata"        # offline suite (deterministic)
uv run ruff check .                        # lint
grep -rh "slots=True" engine/ | wc -l      # MUST be 0 (Python 3.9)
cd frontend && pnpm run check && pnpm run build   # svelte-check 0/0 + build
```
Real-data tests are network-gated: `uv run pytest -m realdata` (needs a key).

## Hard constraints (project-specific)

- **Python 3.9 only.** No `match`, no `dataclass(slots=True)`, no runtime PEP-604
  unions (`X | None` hints need `from __future__ import annotations`; FastAPI
  request models must use `typing.Optional`, not `X | None`).
- **pnpm is the only package manager.** npm/yarn are blocked by a preinstall guard.
- **Money is i64 / BIGINT end-to-end** (global rule; no `*_cents` in i32).
- **Svelte 5 runes only**, use the svelte MCP (list-sections → get-documentation →
  svelte-autofixer) on every `.svelte` edit. `.rs` work (none here) uses the
  rust-analyzer MCP.

## The validation gauntlet (the moat)

Every candidate edge MUST pass, in order — each stage killed a real false positive
(see AI_LANDMINES). Code in `engine/forward/`:
pooling (`pooled.py`) → **day-guard** (distinct holdout DAYS, not signal count) →
FDR across versions (`bakeoff.py`) → **rolling** multi-window (`runner.py`) →
**fresh-symbol** confirm → **survivorship** (`tradeable_delisted`) →
interpretability (`interpret.py`) → full stats battery (`stats.py`: DSR, PBO, …).
`run_gauntlet()` / `engine-forward gauntlet` bundles it into one PASS verdict.
Features must be **causal** (≤ event bar); labels forward; edge measured
**over baseline**, never raw bracket profit.

## Current state (2026-06-19)

One validated edge: **GBT swing-long, pooled daily, ~2-week 2:1 bracket** —
de-biased ~**+0.5–0.6R**, model-agnostic, survivorship- & cost-resistant, PBO 0.19,
stronger in bear/high-vol. **HISTORICAL out-of-time only — NOT live-confirmed.**
The live journal (7 open signals) is accumulating; run `engine-forward resolve`
daily. Caveats: regime-dependent (bull-biased window), 2026 softening to watch.

## Where to look

- `docs/AI_LANDMINES.md` — bugs caught; read first.
- `docs/FORWARD_TESTING.md` — the gauntlet + the edge in detail.
- `docs/DECISIONS.md` — ADRs (3.9 pin, FastAPI, one-pipeline, deferred work).
- Memory at `~/.claude/projects/-Users-billyribeiro-trading-engine/memory/`
  (`first-validated-edge`, `three-scanner-architecture`, `causal-boundary-discipline`)
  — auto-loaded each session.
- `git log` — clean, rule-citing commits document the whole build.

## Deferred / next (not yet built)

More scanners + supervised/unsupervised learners (each through the gauntlet);
point-in-time universe reconstitution; analyst-estimate features; the rich D3/
Threlte dashboard viz (deferred by the operator — engine first). Settings will
grow (the panel is built to extend).
