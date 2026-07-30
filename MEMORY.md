# SwingScope — Project Memory

*Single source of truth. Update this file whenever a number changes.*

*Last updated: 30 July 2026*

---

## Paste this into any new AI session

```
SwingScope handover — read this first.

WHAT IT IS
  Streamlit research app for NSE swing trading. Ranks stocks by 12-1 momentum,
  produces a weekly bucket with stops and targets. Never places orders.
  Repo: github.com/compukidsdgp-creator/Swing-trading

CURRENT CONFIG
  Model: 12-1 momentum (momentum.py)
  Horizon: 30 days
  Universe: 400 stocks, Nifty 500, liquidity-trimmed
  Bucket: 10 picks (top 2.5%)
  Experiment started: 28 July 2026

BENCHMARK — use this one
  Forward IC benchmark: 0.0262
  Derived: 20-year IC 0.0311 x (1 - 0.156 survivorship correction)
  An inference, not a direct measurement. See below.

VALIDATION — 20 year, 201 windows
  Mean IC: 0.0311  (survivorship-biased)
  Newey-West t: 2.74
  Permutation p: 0.00
  Positive windows: 61.7%
  Gross quintile spread: 0.46%
  Long-only edge at 30d: 2.16%
  Net after costs and 20% STCG: +5.95% annualised

POINT-IN-TIME — 30 windows, 2022-08 to 2026-05, run 30 Jul 2026
  Standard IC: 0.0686 (t=3.36) on the same dates
  Point-in-time IC: 0.0579 (t=2.21)
  Survivorship inflation: 15.6%
  Gross spread: 2.284% (PIT) vs 2.742% (standard)
  Universe churn: 44.6% between observations

SUPERSEDED — do not use
  5-year: IC 0.076, spread 1.42%. Did not generalise.
  v1 composite: residual IC 0.0041, t=0.17. Failed. Retired in the registry.

SUB-PERIOD IC
  2010-13: 0.034 | 2014-17: 0.049 | 2018-21: 0.027 | 2022-26: 0.023
  Declining. Cause unresolved: arbitrage, regime, or noise.

UNIVERSE DEPENDENCE — the key finding
  150 symbols (15y+ history): spread 0.22%
  400 symbols (5y+ history):  spread 0.87%
  The edge lives in the broader, smaller-cap end. But that is also where
  slippage is 0.70% vs 0.10% for large caps. Edge and cost sit in the same place.

COSTS
  Statutory 0.26% | Slippage 0.10/0.30/0.70% by tier | STCG 20%

CAVEATS THAT MATTER
  24 configurations tested, best reported — some margin is selection
  Survivorship bias present (dataset is today's constituents)
  ZERO live forward evidence

STATE
  Forward log: accumulating from 28 Jul 2026
  Next decision point: 8 weeks, compare forward IC against 0.0311
```

Attach `forward_log.csv` and this file.

---

## The numbers

| Metric | Value | Source |
|---|---|---|
| **Benchmark IC** | **0.0262** | 20-year 0.0311 less 15.6% survivorship |
| 20-year IC (biased) | 0.0311 | 201 windows, 2010–2026 |
| Point-in-time IC | 0.0579 | 30 windows, 2022–2026 |
| Survivorship inflation | 15.6% | measured 30 Jul 2026 |
| IC t-statistic | 2.74 | Newey-West |
| Permutation p | 0.00 | 300 shuffles |
| Gross spread | 0.46% | quintile, 30-day |
| Long-only edge | 2.16% | top slice over universe mean |
| Net annualised | +5.95% | after charges, slippage, 20% STCG |
| Horizon | 30 days | revised from 15 |
| Universe | 400 | raised from 150 |
| Bucket | 10 | top 2.5% |

**Do not use IC 0.076.** That was the 5-year figure, superseded.

### Why 0.0262 and not 0.0579

The two validated figures measure different things and neither is simply better.

| | Period | Universe | Windows | IC |
|---|---|---|---|---|
| 20-year | 2010–2026 | 150 long-history symbols | 201 | 0.0311 |
| Point-in-time | 2022–2026 | bhavcopy, 44.6% churn | 30 | 0.0579 |

The point-in-time figure is survivorship-corrected but rests on 30 windows over
3.6 years. The 20-year figure has 201 windows but carries survivorship bias.

**0.0262 applies the measured 15.6% correction to the long-run number.** It is an
inference rather than a direct measurement, and should be labelled as such — but
it is more defensible than either raw figure, because it combines the longer
sample with the bias correction.

If forward IC lands near 0.0579 that is a good outcome, not merely an adequate
one. If it lands near 0.026 it is in line with expectation.

---

## Decisions and why

| Decision | Reason |
|---|---|
| 12-1 momentum, not the v1 composite | Composite residual IC 0.0041, t=0.17 — all five components measured the same thing |
| 30-day horizon, not 15 | At 15 days long-only edge is 0.77%, netting −1.15% annualised. At 30 it is 2.16%, netting +5.95% |
| 400-stock universe, not 150 | Gross spread 0.87% vs 0.22%. Filtering for long history selects large, efficiently-priced companies |
| Small caps excluded | 0.96% round-trip cost against a ~2% edge |
| No order execution | The judgement between a ranked list and a purchase is doing real work while there is no forward evidence |
| No ML | ~60 windows and one signal. Tree ensembles would overfit spectacularly |
| No RSI band in momentum mode | v1 leftover, never validated, made the app disagree with the pipeline |
| Sector cap at 2 | Untested choice, not a finding |

---

## Repo layout

**Live path:** `pipeline.py` (10 stages) → `momentum.py` → `bucket.py` → `notify.py`

**Risk gates, in order:** model permission → health → macro overlay → crash protection → earnings window → exit liquidity → exposure cap

**Research only:** `institutional.py`, `local_data.py`, `decay_monitor.py`, `pit_validation.py`, `outcomes.py`, `horizon.py`

**Workflows:** `ci.yml` (every push), `main.yml` (Mon), `daily.yml` (Tue–Fri), `decay.yml` (quarterly), `fundamentals.yml` (monthly)

---

## Bugs that recurred — check for these first

Three classes bit repeatedly during development. All three now have CI checks.

**1. Silent patch failure.** `str.replace()` returns the string unchanged when
the anchor does not match, and reports nothing. Five patches failed this way in
one session; five modules ended up used but never imported.
**Always read the file back after patching.**

**2. DataFrame truthiness.** `a.get(x) or b.get(y)` raises when the value is a
DataFrame. Appeared three times. CI check: `no DataFrame truthiness`.

**3. Module alias shadowing.** `tr = something` makes `tr` local to the whole
function, breaking `import tiers as tr` used earlier in it. Caused an
`UnboundLocalError` in production. CI check: `shadowed module aliases`.

**Also fixed:** float equality on `std()` (never exactly 0.0), yfinance partial
bars poisoning every last-row read, pre-listing padding creating a fake +5,575%
return.

---

## Outstanding

| Item | Severity |
|---|---|
| ~~PIT validation~~ | **DONE** 30 Jul 2026 — 15.6% inflation |
| Trade attribution not built | medium |
| Stress testing not built | medium |
| Two stale branches in repo | cosmetic |
| README and MANUAL describe v1 | cosmetic |
| F&O list hand-maintained (`circuit.py`) | watch — goes stale silently |
| Repo-rate table hand-maintained (`macro.py`) | watch — same |

---

## The decision point

**Week 8 — compare forward IC against 0.0262.**

| Forward IC | Retention | Action |
|---|---|---|
| Above 0.016 | 60%+ | Consider small live size |
| 0.010–0.016 | 40–60% | Continue paper trading |
| Below 0.010 | Under 40% | Do not size up |
| Zero or negative | — | **Stop** |

Forty to sixty percent retention is normal and healthy. Backtests are optimistic
by construction because the signal was selected on the same data.

**A result that decays to nothing is exactly what this apparatus exists to
catch** — cheaply, on paper, rather than expensively with capital.

---

## Standing reminders

**Do not change parameters mid-experiment.** Every change restarts the clock on
clean evidence, and eight weeks is the scarcest resource here.

**Do not add signals.** Twelve were tested, ten failed. More candidates means
more chances for noise to clear the bar.

**Zero picks in a risk-off regime is the design working**, not a fault.

**Any single trade is close to a coin flip.** Top and bottom decile outcome
distributions overlap 87%. The edge is in average magnitude across many trades.

---

*Research infrastructure. Not investment advice.*
