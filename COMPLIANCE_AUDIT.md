# SwingScope — Compliance Audit Against Institutional Standards

*Assessed 28 July 2026 against the supplied international standards document.*

**Status key:** ✅ Met · 🟡 Partial · ❌ Not met · ⬜ Not applicable

---

## 1. International Standards & Protocols

| Standard | Status | Assessment |
|---|---|---|
| **FIX Protocol** | ⬜ | No order transmission. The system is deliberately research-only and places nothing. FIX becomes relevant only if execution is added, which is currently a *prohibited use* in the model registry. |
| **ISO 20022 / 15022** | ⬜ | No clearing or settlement. Not applicable to a research tool. |
| **REST APIs** | ✅ | Used throughout — yfinance, NSE archives, Google News RSS, Telegram, Paytm Money. |
| **WebSockets** | ❌ | Not implemented. Paytm and Angel One both offer streaming market data; unused because a 15-day horizon has no need for sub-second data. **Genuine gap only if intraday execution is added.** |
| **FAST Protocol** | ⬜ | High-frequency feed compression. Irrelevant at this horizon. |
| **ISIN (ISO 6166)** | 🟡 | Present in bhavcopy data and parsed by `bhavcopy.py`, but not used as the primary key. Tickers are used instead, which breaks on renames — **a real if minor risk**. |
| **FIGI** | ❌ | Not implemented. Would matter for cross-venue or multi-vendor reconciliation; currently single-venue. |
| **CFI (ISO 10962) / MIC (ISO 10383)** | ❌ | Not implemented. Single exchange, single instrument type. |

**Section verdict:** Protocol gaps are mostly *appropriate* absences. The one worth fixing is **ISIN as primary key** — ticker renames are a genuine source of silent data corruption.

---

## 2. Algorithms & Mathematical Models

### Signal generation

| Model | Status | Assessment |
|---|---|---|
| **Statistical arbitrage / pairs** | ❌ | Not implemented. Would need cointegration testing (Engle-Granger) and is a fundamentally different strategy, not an addition. |
| **Bollinger / RSI mean reversion** | 🟡 | Both indicators computed. `reversal_1m` was **tested and failed** in the signal laboratory — residual IC negative. Correctly excluded rather than merely absent. |
| **Adaptive moving averages** | 🟡 | EMA crossovers implemented. **KAMA not implemented.** |
| **Donchian / volatility breakout** | 🟡 | `52w_high_proximity` tested (residual IC +0.0187, t=1.30 — failed). Bollinger squeeze implemented in the v1 setup score. No explicit N-day Donchian channel. |
| **Kalman filters / state space** | ❌ | Not implemented. Defensible: with ~60 windows of evidence, added model complexity fits noise. |
| **Supervised ML (RF, LightGBM)** | ❌ | **Deliberately excluded.** With one validated signal and 62 windows, tree ensembles would overfit spectacularly. This is the most common way retail quant projects fail. |

### Execution

| Model | Status | Assessment |
|---|---|---|
| **Bracket orders** | ❌ | No execution layer. Stops are *calculated* and reported; placing them is manual. |
| **TWAP** | ❌ | Not implemented. |
| **VWAP** | ❌ | Not implemented. |

### Risk & position sizing

| Model | Status | Assessment |
|---|---|---|
| **ATR / vol-adjusted sizing** | ✅ | Fully implemented, tier-aware, with liquidity caps as a share of ADV. Matches the formula in the standards document. |
| **Fixed fractional** | ✅ | Risk-per-trade percentage, default 1%. |
| **Kelly criterion** | ❌ | Not implemented. Arguably correct — full Kelly is far too aggressive for a thin edge, and fractional Kelly needs a reliable win-rate estimate that does not yet exist. |
| **VaR / CVaR** | ✅ | **Now implemented** in `institutional.py`, with the note that CVaR is the figure to size against given fat tails. |

**Section verdict:** Execution algorithms are absent because there is no execution. Signal-side gaps are largely *tested and rejected* rather than untried, which is the stronger position.

---

## 3. Infrastructure & Technicalities

| Concept | Status | Assessment |
|---|---|---|
| **Market microstructure / order book** | ❌ | No Level 2 data. ADV is used as a liquidity proxy. **Real gap** — impact cost is estimated rather than measured. |
| **Time series handling** | ✅ | Split and dividend adjusted via yfinance; data-quality audit catches corporate-action errors; 95.8% clean. |
| **Survivorship bias** | ✅ | Bhavcopy point-in-time universe implemented. Index-membership bias removed; delisting bias documented as residual. |
| **Overnight gap risk** | 🟡 | Liquidity caps and ATR stops address it indirectly. **No explicit gap analysis** — the backtest assumes stops fill at the stop price. |
| **Corporate actions** | 🟡 | yfinance auto-adjusts. Data audit flags implausible moves. **No explicit corporate-action calendar.** |
| **Data ingestion** | 🟡 | pandas. No polars or KDB+, which is proportionate — the full universe scores in ~6 seconds. |
| **Event-driven backtester** | 🟡 | The backtester is loop-based over bars with next-bar-open fills and pessimistic stop ordering. **Not strictly event-driven**, but no lookahead — verified by property test across ~1,700 randomised cases. |
| **OMS / EMS** | ❌ | None. No orders. |
| **Pre-trade risk controls** | ✅ | Regime gate, tier permissions, sector caps, liquidity caps, position limits, health gate, model permission gate. |
| **Circuit breakers** | 🟡 | Health gate aborts on bad data; decay monitor alerts. **No single deliberate kill switch.** |

---

## 4. Standard Quantitative Workflow

| Stage | Status | Assessment |
|---|---|---|
| **1. Hypothesis formulation** | ✅ | Explicit and testable. Momentum grounded in Jegadeesh-Titman with documented rationale for every parameter. |
| **2. Event-driven backtest, PIT data** | 🟡 | No lookahead (property-tested). Point-in-time data **available via bhavcopy but not yet wired into the backtest** — this is the largest remaining methodological gap. |
| **3a. Out-of-sample split** | ✅ | **Now implemented.** Chronological split with an evaluation ledger, because a held-out period stops being held-out once it informs a decision. |
| **3b. Monte Carlo** | ✅ | **Now implemented.** Trade-sequence scrambling with drawdown percentiles, probability of loss and probability of ruin. |
| **4. Paper trading / incubation** | ✅ | Forward log running weekly. Currently empty — only time closes this. |
| **5. Live deployment & risk audit** | 🟡 | Not deployed live. Monitoring infrastructure exists: decay monitor, health checks, CI, failure alerts. |
| **Sharpe / Sortino / MDD tracking** | ✅ | **Now implemented**, with interpretation. |

---

## The gap list, ranked

### 🔴 Material

1. **Point-in-time data not wired into validation.** Bhavcopy gives a genuine PIT universe; the backtest still uses today's constituents. Every published figure — IC 0.076, spread 1.42% — carries residual survivorship bias.
2. **No out-of-sample result yet.** The capability now exists but has not been run. Until it is, the headline numbers are entirely in-sample.
3. **Impact cost estimated, not measured.** No Level 2 data. Against a 1.42% gross spread, an error here matters.

### 🟡 Worth fixing

4. **ISIN not used as primary key** — ticker renames silently corrupt history.
5. **No explicit gap analysis** — backtest assumes stops fill at the stop price.
6. **No corporate-action calendar** — splits and demergers inside a holding window distort positions.
7. **No deliberate kill switch.**

### ⬜ Appropriately absent

FIX, ISO 20022, FAST, OMS/EMS, TWAP/VWAP, bracket orders — all require an execution layer that does not exist and should not until forward evidence justifies it.

ML models, Kalman filters, Kelly sizing — excluded on the grounds that added complexity fits noise at this sample size. **This is a defensible position, not an oversight.**

---

## Honest summary

Against the institutional checklist the system scores **strongly on validation methodology and risk control**, and **weakly on execution infrastructure** — which is exactly right for a research tool that deliberately places no orders.

The three material gaps are all about *confidence in the result* rather than the result itself. Two are now closable in an afternoon; the third needs data that costs money.

**The binding constraint remains unchanged and unaffected by any of this: zero live forward evidence.** Every metric in the system is measured on historical data the signal was also selected on.

---

*Self-assessment, not independent audit. The same person built and graded this — precisely the independence problem SR 11-7's three-lines structure exists to solve.*
