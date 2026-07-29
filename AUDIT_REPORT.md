# SwingScope — Audit, Research and Optimisation Report

*Generated 27 July 2026, following a session of substantial changes.*

---

## Scope note

You asked me to review "all theses published by PhD holders, bloggers and YouTube
videos" and accumulate "all hacks." That is not a bounded task and I cannot do it
honestly. What I have done instead:

- Run the full regression suite against tonight's changes
- Traced the process flow end to end for misalignment
- Searched the academic literature on momentum implementation specifically
- Implemented the single best-supported enhancement found

Where I could not verify something, it is marked as unverified rather than
presented as fact.

---

## Part 1 — Regression check

**14 of 14 invariant properties held** after the evening's changes, across
roughly 560 randomised and adversarial cases.

| Property | Result |
|---|---|
| No lookahead: indicators & composite | PASS |
| No lookahead: all 12 signals | PASS |
| No lookahead: regime | PASS |
| Bounds: v1 score components | PASS |
| Bounds: percentile ranking | PASS |
| Bounds: correlation | PASS |
| Risk: regime gate never leaks tiers | PASS |
| Risk: sector cap never breached | PASS |
| Risk: position limits respected | PASS |
| Backtest: accounting consistency | PASS |
| Forward log: integrity | PASS |
| Sentiment: bounded | PASS |
| Data quality: detects defects | PASS |
| Universe: trim unbiased | PASS |

No regressions introduced by the horizon, cost, bhavcopy or optional-import work.

---

## Part 2 — Process-flow audit: findings

### 🔴 F1. The two cost models disagree, and the live one is wrong

`tiers.py` drives position sizing and the small-cap exclusion. It assumes
0.25% / 0.60% / 1.50% round-trip by tier.

`costs.py` computes from statutory rates: **0.258% charges** for a large cap —
so the charge estimate is well calibrated. But add 20% STCG and the true all-in
figure is **0.546%**.

**`tiers.py` has no tax component at all.** Every position-sizing decision and
the entire small/mid/large viability judgement rests on a number that understates
true cost by roughly 0.30 percentage points — more than half the edge.

*Impact: high. Fix: add a tax term to `est_cost_pct`, or reference `costs.py`
directly.*

### 🔴 F2. The system defaults to the one horizon that does not work

`horizon=15` is hardcoded across `validate.py`, `signals.py`, `forward_log.py`,
`automate.py` and the workflow. `backtest.py` defaults to `hold_bars=18`.

Tonight's analysis established that at 15 days the edge nets to approximately
**−0.004% per cycle** — break-even to slightly negative. Twenty days breaks
even; sixty is optimal on the modelled curve.

**The default is the one setting shown not to clear its hurdle.**

*Impact: high. Fix: run the horizon sweep on real data, then change the default
to whatever the plateau supports.*

### 🟡 F3. The live path never mentions tax

`pipeline.py`, `bucket.py` and `report.py` contain no reference to capital gains
tax or break-even economics. The Telegram message and weekly report present picks
with no indication that the strategy is marginal after tax at the current horizon.

Only `momentum.py` docstrings mention breakeven, and only regarding mid-cap
*charges*, not tax.

*Impact: medium — it is a communication gap, not a computational one. But it means
the output looks more actionable than the evidence supports.*

### 🟡 F4. `Cost_viable` is displayed but only partially enforced

The flag is computed in `momentum.py` and shown in the app, and `pipeline.py`
does filter tiers via `RECOMMENDED_TIERS`. But in the app's screener it is a
caption only — small caps still appear in the table and can enter the bucket.

*Impact: medium. Fix: apply the filter, or make the caption a hard exclusion.*

### 🟡 F5. No momentum-specific crash protection

The regime gate watches the **index** — Nifty against its 200 DMA. Momentum
crashes are predicted by **momentum's own realised volatility**, and the two
diverge precisely when it matters. This is addressed in Part 4.

*Impact: high — this is a tail-risk exposure, not a return drag.*

### 🟢 F6. Sector cap is untested

`max_per_sector=2` was my choice, not a finding. Momentum is often
sector-concentrated — that may be where the effect lives, in which case the cap
discards signal rather than managing risk.

*Impact: unknown, which is the point. Testable by re-running validation with the
cap varied.*

### 🟢 F7. Tier classification look-ahead (known, deferred)

`classify_by_turnover` uses today's turnover applied to historical bars. It does
**not** affect the momentum IC, because `_momentum_at` ignores tier. It does
affect backtest expectancy via ATR multiples and cost assumptions.

*Impact: low for the current model. Deferred by prior agreement.*

---

## Part 3 — Literature review

Searched for published work on momentum implementation. The clearest and most
replicated finding concerns crash risk.

### Momentum crashes and volatility scaling

Barroso and Santa-Clara (2015), *"Momentum Has Its Moments"*: the risk of
momentum is highly variable over time and **predictable by its own realised
variance**. Managing that risk by scaling to constant volatility **virtually
eliminated the crashes and nearly doubled the Sharpe ratio — 0.97 against 0.53
unmanaged.** They targeted 12% annualised volatility.

Daniel and Moskowitz (2016), *"Momentum Crashes"*, identified the mechanism:
following major market declines, betas for the past-loser decile can rise above
3 while past winners fall below 0.5. A momentum book therefore carries a large
negative conditional beta, and **crashes when the market rebounds sharply.**

Later work: Bongaerts, Kang and van Dijk (2020) found that adjusting exposure
only in extreme volatility states — leaving the middle unscaled — reduced
drawdowns and tail risk across all major equity markets while also **reducing
turnover**, which matters directly given that costs are the binding constraint here.

Comparative evidence from a study across 55 global futures contracts found the
**constant** volatility scaling approach produced statistically superior alphas
to the dynamic approach in most specifications.

### One critique worth carrying

The 12% target has been criticised as unjustified — it implies every investor
shares the same risk preference. The implementation below exposes it as a
parameter rather than hardcoding it.

### What I did not find

No credible source suggesting a "hack" that materially improves momentum returns
without a corresponding cost or risk. The literature is consistent that momentum
is a real but modest effect whose main practical challenges are **crash risk and
transaction costs** — precisely the two things this session identified
independently.

---

## Part 4 — Implemented: crash protection

New module `crash_protection.py`.

**Volatility scaling.** Exposure is scaled by target ÷ realised volatility,
capped between 10% and 150% so it can neither go fully flat on a single signal
nor lever beyond a sane bound. Tested across regimes:

| Realised volatility | Exposure |
|---|---|
| 7.9% (calm) | 150% (capped) |
| 12.2% (normal) | 98% |
| 24.1% (elevated) | 50% |
| 52.7% (crisis) | 23% |

**Crash-risk indicator.** Flags the specific configuration Daniel and Moskowitz
identified: a sharp rebound out of a deep drawdown, when past losers carry
extreme beta. Correctly identified a synthetic drawdown scenario as `elevated`.

**Fallback behaviour.** With insufficient return history it applies no scaling
and says so explicitly — that being the correct default rather than a licence to
size up.

**Known limitation:** the conditional mode still misclassifies mid-range
volatility in testing and needs calibration on real data. Constant mode is the
default, and is what the comparative literature supports.

---

## Part 5 — Ranked recommendations

| # | Action | Impact | Effort |
|---|---|---|---|
| 1 | **Run the horizon sweep on real data** and change the default to the plateau | Very high | 10 min |
| 2 | **Add tax to `tiers.py` cost model** | High | 30 min |
| 3 | **Wire crash protection into position sizing** | High (tail risk) | 1 hour |
| 4 | Surface net-of-tax economics in the Telegram/report output | Medium | 30 min |
| 5 | Enforce `Cost_viable` as a filter | Medium | 10 min |
| 6 | Test the sector cap at 1, 2, 3, unlimited | Unknown | 20 min |
| 7 | Fix tier look-ahead | Low | 30 min |

**Item 1 dominates everything else.** The difference between a 15-day and a
30–60 day hold is roughly 30–70× larger than any other change available, and it
requires no new data.

---

## Part 6 — Honest position

**What is established:** momentum ranks forward returns in the Nifty 500
(IC 0.048, Newey-West t = 3.73, clearing the Harvey-Liu-Zhu bar). The
implementation is free of lookahead across every component tested. The universe
is unbiased after tonight's fix. Data is 95.8% clean.

**What is established and unwelcome:** at a 15-day horizon the edge is consumed
entirely by charges and 20% short-term capital gains tax. A typical momentum
basket showed 10 positions behaving as **1.8 independent bets**, meaning real
risk is roughly 2.4× what per-position sizing assumes.

**What remains unknown:** whether any horizon clears the hurdle on *real* data —
the sweep has only been validated on synthetic series. And whether the edge
survives live, which needs the forward log and time.

**The largest remaining gap is not technical.** It is that you have zero live
evidence. Every number in this report is measured on historical data that the
signal was also selected on.

---

*Research tool. Not investment advice. Tax treatment varies by individual
circumstance — confirm with a chartered accountant. Consider consulting a
SEBI-registered adviser before committing capital.*
