# Trading Engine — Dashboard

SvelteKit 2 + Svelte 5 (runes) frontend for the scanner. Talks to the FastAPI
backend (`engine.api`).

## Run

1. Start the engine API (from the repo root, with `FMP_API_KEY` set):

   ```bash
   uv sync --extra api
   uv run engine-api          # serves http://127.0.0.1:8000
   ```

2. Start the dashboard:

   ```bash
   cd frontend
   cp .env.example .env        # PUBLIC_API_BASE -> the API URL
   pnpm install
   pnpm run dev                # http://localhost:5173
   ```

## What it does

- **Lookback control + watchlist** drive the validated batch-rank scanner.
- **Results table**: only signals whose config survived walk-forward + cost +
  FDR (entry/stop/target, edge R, p(fdr), decay). "No signals" is a correct,
  honest result.
- Click a symbol → the **session dissection** (STRUCTURE / LEG ROLES / VWAP map /
  key levels / plain-English read).

## Gates

```bash
pnpm run check     # svelte-check (0 errors)
pnpm run build     # production build
```

The default `adapter-auto` picks a host at deploy time; swap in
`@sveltejs/adapter-vercel` (or node) for production.
