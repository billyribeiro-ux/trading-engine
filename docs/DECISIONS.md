# Architecture decisions

Short ADRs for choices the brief flagged as needing a decision, plus what is
deliberately deferred. Keep this current as the system grows.

## 1. API stack: FastAPI (Python), not Rust/Axum — decided 2026-06-19

The brief left this open ("FastAPI is fine, or have the Rust/Axum backend shell
out / port the hot path — discuss with Billy which"). **Decision: FastAPI**, in
`engine/api/`.

- The hot path (dissection, multi-scale pivots, walk-forward, the numpy logistic)
  is Python and tightly coupled to pandas/numpy. Re-porting it to Rust would be a
  large rewrite for no current performance need; shelling out from Axum to Python
  per request adds latency and a second process to operate.
- FastAPI wraps the existing engine with zero logic duplication and is installed
  as an optional extra (`uv sync --extra api`), so the core engine stays lean.
- **Reversible**: if the standard Rust/Axum stack is wanted later, Axum can
  reverse-proxy the FastAPI service (no engine change), or the hot path can be
  ported once profiling justifies it. The dashboard talks to a URL
  (`PUBLIC_API_BASE`), so the backend behind it can change without touching the UI.

## 2. Python 3.9, pinned — decided 2026-06-18

The engine targets the operator's Mac system Python (3.9.6). No `match`,
no `dataclass(slots=True)`, no PEP 604 runtime unions. `requires-python =
">=3.9,<3.10"`, ruff `target-version = py39`, a `test_compat` suite, and CI on 3.9
keep it honest. (One consequence: the FastAPI layer uses `typing.Optional`, since
FastAPI evaluates annotations at runtime and `X | None` can't be eval'd on 3.9.)

## 3. One pipeline, three scanners — decided 2026-06-18

`engine.ml.signals.batch_rank` (validate → FDR → survivors → signals) and the
`StructuralUnit` abstraction are written once. Each scanner is a `ScannerConfig`
(intraday / swing / portfolio). No `if intraday: … else: …` deep in the pipeline.

## Deferred (known, not done)

- **Point-in-time index reconstitution.** Survivorship is handled by *including*
  delisted names (`engine.core.universe.build_universe(include_delisted=True)`),
  but reconstructing exact historical index membership as-of each backtest date is
  a deeper data project, not yet built.
- **Gradient-boosted model.** `SignalModel` is a protocol; the numpy logistic can
  be swapped for a GBT behind it. The validation/leak gates already constrain any
  model.
- **Forward testing loop.** A paper/live harness over `evaluate_live` to compare
  realized vs. validated edge is the next phase.
- **Analyst-estimate features.** Skipped for now: the endpoint exposes the period
  being estimated, not a clean publish date, so an as-of join can't be guaranteed
  causal yet.
