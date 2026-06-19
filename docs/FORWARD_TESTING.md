# Forward testing & the validated edge

This engine's prime directive is honesty: an edge counts only if it survives
**out-of-time, out-of-sample, at breadth** — never in-sample fit. This doc records
the validation recipe (the "gauntlet"), the one edge that has passed it so far, and
how to keep it honest going forward.

## The gauntlet (`engine/forward/`)

Every edge candidate runs the same recipe. Each stage killed a real false positive
during discovery, so none is optional.

| Stage | Module | What it enforces | What it killed |
|---|---|---|---|
| **Pooling** | `pooled.py` | cross-symbol training for statistical power | single-name nulls (too few signals) |
| **Day-guard** | `runner.py` | edge must span enough distinct **calendar days**, not just signal count | intraday "p=0.000 on 68 signals" that was really **3 correlated days** |
| **FDR across versions** | `bakeoff.py` | Benjamini-Hochberg over every model/feature tried | inflated significance from searching |
| **Rolling windows** | `runner.py` | persistence across N sequential out-of-time blocks | logistic that worked in one block but **2/5 windows** overall |
| **Fresh symbols** | (run) | confirm on a symbol set never used in discovery | symbol-overfit / meta-search |
| **Survivorship** | `tradeable_delisted` | pool in delisted names (the losers that died) | survivorship inflation of a mean-reversion edge |
| **Interpretability** | `interpret.py` | drivers must be sensible & diversified | leak signatures (one feature at AUC ≈ 1.0) |

`run_gauntlet(symbols, config, ...)` (`gauntlet.py`) bundles bake-off + rolling
into one `PASSED` verdict: a model must be **promoted AND robust**. `"not passed"`
is the common, honest result.

## The edge that passed

**GBT (and, at breadth, logistic) swing-long, pooled daily, ~2-week 2:1 bracket.**

- Wide universe (40 names, 6,131 events, 1,197 distinct days): **PASSED**. GBT
  rolling 6/6 windows, +0.60R over 337 distinct days, p=0.000; logistic 6/6,
  +0.56R, 224 days, p=0.000.
- Generalizes to fresh symbols (5/5 windows on a set never used in discovery).
- **Survives survivorship correction**: on a delisted-only pool (names that died)
  the edge is still +0.45R (p=0.000); survivorship-free (90 names incl. delisted,
  13,543 events) it PASSES both models 6/6. There is measurable inflation
  (survivors +0.70 vs delisted +0.45) so the **de-biased estimate is ~+0.5–0.6R**.
- Drivers (permutation importance on the holdout): `leg_bars`, `range_pos_60`,
  `leg_is_up`, `ma20_vs_ma50`, `dist_from_runlow`, `dist_from_ma50` — a coherent
  **mean-reversion + range-position + trend-regime** strategy, spread across ~8
  features (the opposite of a leak).
- Model-agnostic: logistic is not robust on 20 names (2/5) but is on 40 (6/6) —
  the edge sharpens with data, the signature of a real effect.

### Honest caveats (do not overstate)
- This is **historical out-of-time** evidence, **not live**. Live confirmation is
  accruing via the journal (below).
- Magnitude is **modest**: ~+0.5–0.6R over baseline (de-biased), holdout AUC ~0.61.
- Costs are assumed (~0.02–0.05R); slippage on thin names matters.
- "100% accurate" is impossible — a model near AUC 1.0 is a **leak**, not skill.
- Edges decay. Re-run `run_gauntlet` periodically.

## Live loop (`engine/forward/live.py`, `journal.py`)

Turn historical evidence into live proof. Signals are emitted **only if the pooled
config still passes the forward gate now**; each carries its current forward
backing; resolution is closed-bars-only, conservative stop-first.

```bash
engine-forward scan    --scanner swing --model gbt --direction long   # after close
engine-forward resolve --scanner swing                                # daily
engine-forward report                                                 # realized vs validated
```

Schedule `scan` + `resolve` daily (cron/launchd; make `FMP_API_KEY` available to
the job). Watch realized R converge toward the carried validated edge — if it
drifts to ~0 over a meaningful number of resolved trades, the edge isn't holding
live and should be retired.

The dashboard surfaces the journal at `GET /journal` and the **Live forward
journal** panel (`JournalView.svelte`).
