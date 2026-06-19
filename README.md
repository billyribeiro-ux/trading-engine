# Trading Attribution Engine

A pure-Python engine that explains *why* a stock moved and quantifies tradeable
edges with honest statistics. Built on Financial Modeling Prep (FMP) data.

This is the foundation: a tier-aware data layer and a complete **gap-statistics
module**. Feature core, regime classifier, attribution, ML, and walk-forward
validation layer onto the same data foundation.

## Design principles

- **No fabricated certainty.** Every probability ships with a Wilson confidence
  interval and a Bayesian-shrunk estimate. Thin buckets are flagged
  `[INSUFFICIENT EVIDENCE]`, never presented as fact. A bucket with n=3 reading
  100% is shown as exactly that — unreliable — not as a signal.
- **Tier-honest.** The data layer detects your FMP tier and refuses to spend a
  request on a gated endpoint, marking that data UNAVAILABLE rather than faking
  an empty result. Upgrade to Ultimate and 13F / 1-min intraday light up with
  zero code changes.
- **No lookahead.** All conditioning features use only information available at
  or before the gap day's open.

## Setup

```bash
pip install -r requirements.txt
export FMP_API_KEY=your_key        # never hard-code; read from env only
```

Your key is read **only** from the environment — never from the command line
(argv leaks into process listings and shell history) and never from source.

## Usage

```bash
python -m engine.gaps AAPL --lookback 10
python -m engine.gaps TSLA --lookback 15 --min-gap-atr 0.25 --verbose
```

Output: gap continuation and fill probabilities for the symbol, conditioned on
direction, ATR-size tier, prior trend, weekday, and earnings proximity — each
with confidence intervals, plus same-session Kaplan-Meier fill probabilities.

## What FMP Premium ($69/mo) provides

30 years daily OHLCV, 5/15/30/60-min intraday, news + sentiment, earnings
calendar, analyst estimates, ratings, insider trades. **Ultimate-only:** 13F
institutional holdings, 1-min intraday full depth, earnings transcripts. The
engine adapts automatically.

## Module map

```
engine/
  data/        FMP client (tier-aware, cached, rate-limited) + endpoint registry
  gaps/        gap classification, conditional stats, Wilson/Bayes/KM, CLI
  features/    (next) VWAP/POC/VAH/VAL, ATR regimes, ADX/efficiency/Hurst
  regime/      (next) trend / range / chop classifier
  catalyst/    (next) news/earnings/ratings/insider join by date
  attribution/ (next) rank probable drivers per move (measured vs inferred)
  ml/          (next) calibrated setup classifier + feature importance
  validation/  (next) purged/embargoed walk-forward, OOS holdout, cost model
  report/      (next) per-ticker "why it moved" report
```
