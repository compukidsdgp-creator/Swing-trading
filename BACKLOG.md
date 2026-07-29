# SwingScope — Backlog and Handover

*Last updated 29 July 2026. Keep this file. Paste the top section into any new
session — I retain nothing between conversations.*

---

## Paste this when you return

```
SwingScope handover.

CURRENT CONFIG
  Model: 12-1 momentum
  Horizon: 30 days
  Universe: 400 stocks, Nifty 500, liquidity-trimmed
  Bucket: 10 picks (top 2.5%)
  Started: ~28 July 2026

VALIDATION (20-year, local dataset, 201 windows)
  Mean IC: 0.0311
  Newey-West t: 2.74
  Permutation p: 0.00
  Positive windows: 61.7%
  Gross quintile spread: 0.46%
  Long-only edge at 30d: 2.16%
  Net after costs and 20% STCG: +5.95% annualised

SUPERSEDED (5-year — do not use)
  IC 0.076, spread 1.42%. Did not generalise.

SUB-PERIOD IC
  2010-13: 0.034 | 2014-17: 0.049 | 2018-21: 0.027 | 2022-26: 0.023

COSTS
  Statutory 0.26% | Slippage 0.10/0.30/0.70% by tier | STCG 20%

CAVEATS
  24 configurations tested, best reported — some margin is selection
  Survivorship bias remains (dataset is today's constituents)
  Zero live forward evidence
```

Attach `forward_log.csv` and this file.

---

## Backlog

Ranked by value to *this* project, not by how impressive they sound.

### 1. Position monitoring and alerts — DO FIRST

**The obvious hole.** The system generates picks and then loses interest. No
open-position view, no alert when something hits its stop or a target, no
running P&L against the levels already recorded.

A screener without a portfolio is half a system. Needed the day real money is
committed.

- Open positions with live P&L against recorded stop and targets
- Telegram alert on stop hit, 1R, 2R
- Days-held counter against the 30-day horizon
- Exit prompt at horizon end

*Build when: before the first live trade.*

### 2. Trade attribution — HIGH VALUE

When a trade works, was it the signal, the sector, the market, or luck?
Decomposing each outcome into those components turns the forward log from
"did it work" into "why".

Institutional desks treat this as basic. Almost no retail tool has it, and it
would make eight weeks of forward data far more informative than a bare IC.

- Decompose each trade: market beta + sector + idiosyncratic
- Report how much of the return the signal actually explains
- Flag when results are driven by market drift rather than selection

*Build when: once ~20 evaluated forward picks exist.*

### 3. Scenario stress testing — CLOSES AN AUDIT GAP

Marked NOT MET in the compliance audit. Crash protection is implemented but its
behaviour in a real crisis has never been tested.

- Replay 2008, 2013 taper, 2020 COVID against the current portfolio
- "What if the Nifty falls 10% tomorrow" on today's holdings
- Correlation breakdown under stress — correlations converge to 1 in crashes,
  which is exactly when diversification is needed

*Build when: convenient. The 20-year dataset makes this possible now.*

### 4. Hierarchical Risk Parity — MODEST IMPROVEMENT

Better than the risk parity already built. No matrix inversion, far more stable
with few observations — directly relevant since covariance is estimated from
roughly 60 windows.

*Build when: after position monitoring. Incremental, not transformative.*

### 5. Meta-labeling — POWERFUL, NEEDS DATA FIRST

A second model predicting whether the *primary* signal will work on this
occasion, rather than trying to improve the primary signal (López de Prado).
Often raises precision substantially.

**Requires forward data to train on.** Building it before then would fit
historical noise.

*Build when: 6+ months of forward log.*

### 6. Regime detection via Hidden Markov Model — LOWER PRIORITY

The current gate is a 200-DMA rule: transparent but crude. HMMs infer regimes
from the data rather than a fixed threshold.

More powerful, considerably less interpretable. The current gate's virtue is
that you can explain exactly why it fired.

*Build when: never, unless the price-based gate demonstrably fails.*

### 7. Walk-forward optimisation

Re-fit periodically, test forward, roll. More honest than a single split.
Relevant only if parameters ever need re-fitting — momentum has none.

*Build when: probably not needed.*

---

## Deliberately not building

**More signals.** Twelve were tested; ten failed. More candidates means more
chances for noise to clear the bar. This is the single most likely way to
destroy what has been built.

**Machine learning on the signal.** With ~60 windows and one signal, tree
ensembles would overfit spectacularly. The most common way retail quant projects
die.

**Pattern recognition, community sentiment, strategy marketplaces.** Standard
retail platform features. None survive factor neutralisation.

**Order execution.** Deliberate. The judgement step between a ranked list and an
actual purchase is doing real work while there is no forward evidence.

**Shorter horizons.** Costs scale with frequency; the edge does not.

---

## Known issues outstanding

| Issue | Severity | Note |
|---|---|---|
| Two stale branches in repo | cosmetic | `patch-1`, `patch-2` — delete them |
| README and MANUAL describe v1 | cosmetic | Content is superseded |
| `automate.py` superseded | cosmetic | `pipeline.py` replaces it; carries a warning header |
| F&O list hand-maintained | watch | In `circuit.py`, will silently go stale |
| Repo-rate table hand-maintained | watch | In `macro.py`, same |
| Tier look-ahead in backtest | low | Does not affect IC; affects backtest expectancy only |

---

## The thing that actually matters

Everything above is optional. **The binding constraint is eight weeks of
forward evidence**, and no feature substitutes for it.

At week 8, compare forward IC against **0.0311**:

| Retention | Reading | Action |
|---|---|---|
| 60%+ | Holding up | Consider small live size |
| 40–60% | Normal decay | Continue paper trading |
| Under 40% | Substantial decay | Do not size up |
| Zero or negative | Backtest did not hold | **Stop** |

Forty to sixty percent retention is normal and healthy, not a disappointment.
Backtests are optimistic by construction. A result that decays to nothing is
precisely what this apparatus exists to catch — cheaply, on paper, rather than
expensively with capital.

---

*Research infrastructure. Not investment advice.*
