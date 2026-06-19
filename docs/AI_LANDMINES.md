# AI_LANDMINES — runtime bugs caught here (do not reintroduce)

Each of these passed compilation/tests but was wrong. They were caught by
adversarial testing or the operator's gut, not by the type checker. Read before
writing code. The fix is the rule.

## ML / validation (the dangerous ones)

1. **"100% accurate looking back" is the bug, not the goal.** Memorizing past bars
   (deep tree, 1-NN) or any lookahead feature trivially scores AUC ~1.000 in-sample
   and collapses to ~0.5 forward. Markets cap real forward accuracy well below 100%.
   The asset is an honest ~0.6 that *persists out-of-sample*. Never chase accuracy;
   chase forward persistence.

2. **Lookahead feature leak (`reached_lod`/pivot-entry).** A feature read at the
   pivot EXTREME (with hindsight) gave AUC 1.000. **Fix:** features are causal
   (bars ≤ the event bar); events decide at the pivot **CONFIRMATION** bar, not the
   extreme. Run the univariate-AUC leak detector on every new feature set
   (threshold ~0.45 large-sample; a single feature near 1.0 = leak).

3. **Bracket-geometry false positive.** A profitable 2:1 bracket yields positive
   *raw* expectancy on pure noise by geometry. **Fix:** measure edge **over
   baseline** (taking all events), never raw bracket profit. The noise test must
   return p > 0.10.

4. **NaN-zeroing silently took zero signals.** Event feature vectors are
   HETEROGENEOUS (a vwap event has no `leg_*`, a leg event has no `level_*`), so the
   stacked frame is full of NaN. A plain StandardScaler trained on NaN → all-NaN
   `predict_proba` → harness took 0 signals while reporting a garbage AUC (~0.83
   from sorting NaN). **Fix:** NaN-aware scaler (`nanmean`/`nanstd` + mean-impute);
   `predict_proba` must always be finite. Synthetic frames hide this — test with NaN.

5. **The few-days trap (day-guard).** Pooled same-day signals across symbols are ONE
   correlated bet, so "p=0.000 on 68 signals" was really **3 correlated days**. The
   iid bootstrap massively overstates significance. **Fix:** count distinct holdout
   DAYS; `persisted` requires `min_holdout_days`. (FMP 5-min history is ~7 days, so
   intraday can't be forward-tested over calendar time — use daily/swing.)

6. **Survivorship inflation.** A survivors-only universe inflates a mean-reversion
   ("buy the dip") edge — the dips you'd have bought on names that died and delisted
   are missing. **Fix:** pool in recently-delisted names (`tradeable_delisted`);
   `engine-forward gauntlet --include-delisted N`. (Delisted-only edge was still
   +0.45R, so it survived — but survivors-only was ~0.25R too high.)

7. **Swing/portfolio event-date bug.** Events stamped `date = window.date` (the
   constant as-of date) instead of the event's own bar date → pooled time-ordering
   and the day-count collapsed to 1 distinct day. **Fix:** `date =
   pd.Timestamp(event_time).normalize()`. (Intraday was already correct: session
   date.)

8. **edge-over-baseline is cost-invariant.** Baseline pays cost too, so sweeping
   `cost_r` on edge-over-baseline shows nothing. **Fix:** for tradeability, report
   the **absolute** taken-trade expectancy (`mean(taken_gross) - cost_r`) and its
   break-even cost — not the edge.

9. **More model ≠ more edge.** On weak features the GBT overfit HARDER than the
   logistic (worse forward decay). The bottleneck is signal, judged by the forward
   test — not model capacity. A real edge is found by MANY model classes (the zoo),
   not one.

10. **Embargo was a no-op** in walk-forward purging. **Fix:** `is_end = oos_start -
    purge_sessions - embargo_sessions`.

## Stack / tooling

11. **FastAPI + Python 3.9 + `X | None`.** With `from __future__ import
    annotations`, FastAPI still runtime-evaluates request-model / query annotations
    → `str | None` fails. **Fix:** use `typing.Optional` in `engine/api/*`
    (ruff per-file-ignore UP007/UP045).

12. **npm 11 arborist crash + `pnpm dlx` user-agent reset.** `npm install <pkg>` into
    an existing tree crashed ("Cannot read properties of null (reading 'matches')");
    a clean reinstall with the dep already in package.json fixed it. And
    `pnpm dlx only-allow pnpm` is NOT a valid npm guard — `pnpm dlx` resets
    `npm_config_user_agent` to pnpm, so only-allow never sees the npm caller.
    **Fix:** a zero-dep inline node check of `npm_config_user_agent` in `preinstall`.
    pnpm is the only package manager; npm/yarn are blocked.

## The meta-rule

Compilation + green tests are necessary, not sufficient. Demand runtime evidence
(EXPLAIN, `curl /metrics`, Playwright, forward-test numbers). When the operator says
"X is broken" and the gates pass, the gates are insufficient — find the discrepancy.
