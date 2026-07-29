# SwingScope — Competitive Gap Analysis

*29 July 2026. Compared against five platforms chosen to span the range: global
leader, AI scanner, India specialist, no-code algo, and institutional-grade.*

---

## The five

| Platform | URL | Positioning | Price |
|---|---|---|---|
| **TradingView** | tradingview.com | Global charting and screening leader, Pine Script | From $14.95/mo |
| **Trade Ideas** | trade-ideas.com | AI-powered real-time scanning, US | $127–254/mo |
| **Trendlyne** | trendlyne.com | India-first research, DVM scores, SEBI-registered | ₹310/mo |
| **Streak** | streak.tech | No-code algo by Zerodha, India | Bundled/low |
| **QuantConnect** | quantconnect.com | Open-source LEAN engine, institutional-grade | Free tier + paid |

---

## The headline finding

**SwingScope is far ahead on validation rigour and far behind on product.**

That asymmetry is unusual and worth understanding before deciding what to build.
The commercial platforms have spent years on charting, mobile, execution and
community. SwingScope has spent one intense period on the question none of them
seriously answer: *does the signal actually work, and does it survive costs?*

Both halves matter. Neither substitutes for the other.

---

## Where SwingScope leads

These are not features the commercial platforms have chosen not to build. In
most cases they are things the category does not do at all.

| Capability | SwingScope | TradingView | Trade Ideas | Trendlyne | Streak | QuantConnect |
|---|---|---|---|---|---|---|
| **Factor neutralisation** | ✅ Fama-MacBeth vs 6 factors | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Permutation testing** | ✅ 400 shuffles | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Newey-West correction** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Harvey-Liu-Zhu t>3 bar** | ✅ reported | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Forward log (pre-committed)** | ✅ | ❌ | ❌ | ❌ | ❌ | Partial |
| **Model registry / governance** | ✅ SR 11-7 aligned | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Append-only audit trail** | ✅ | ❌ | ❌ | ❌ | ❌ | Partial |
| **Full cost model incl. STCG** | ✅ 20% tax modelled | ❌ | ❌ | ❌ | ❌ | Fees only |
| **Correlation-adjusted sizing** | ✅ risk parity | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Alpha decay monitoring** | ✅ quarterly | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Property-based test suite** | ✅ 18 invariants | n/a | n/a | n/a | n/a | Partial |
| **Circuit-breaker awareness** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Point-in-time universe** | ✅ bhavcopy | ❌ | ❌ | ❌ | ❌ | ✅ |

**Only QuantConnect comes close**, and only on point-in-time data and backtest
realism. Nothing in the retail category does factor neutralisation — which is
why most retail strategies are repackaged momentum without their authors
knowing it. SwingScope discovered exactly that about its own v1 composite.

---

## Where SwingScope is behind

### Tier 1 — genuine gaps that limit usability

| Gap | Who has it | Why it matters here |
|---|---|---|
| **Position monitoring** | All five | Picks are generated, then abandoned. No open-position view, no P&L against recorded stops. **The most obvious hole.** |
| **Price alerts** | All five | No notification when a holding hits its stop or a target. Levels are computed and then never watched. |
| **Charting** | TradingView, Trade Ideas, Trendlyne | Detail tab has basic candles. TradingView is the category benchmark and is not realistically matchable. |
| **Mobile access** | All five | Streamlit works on a phone but is not an app. Trendlyne rates 4.2–4.3 stars on mobile. |
| **Broker execution** | Streak, QuantConnect, TradingView | Deliberate absence — see below — but a real product gap. |

### Tier 2 — worth considering

| Gap | Who has it | Assessment |
|---|---|---|
| **Event-driven backtester** | QuantConnect (LEAN) | LEAN models T+3 settlement, slippage, spread. SwingScope's is loop-based with next-bar fills — no lookahead, but less realistic. |
| **Parameter optimisation** | QuantConnect, TrendSpider | Deliberately avoided. Optimising across thousands of iterations is how backtests become fiction. |
| **Fundamental depth** | Trendlyne (1,000+ params) | SwingScope has yfinance basics plus a forward archive. Trendlyne's DVM scoring is genuinely deep. |
| **Analyst estimates** | Trendlyne | Consensus forecasts for revenue, EPS, cash flow. Not available free elsewhere. |
| **Real-time scanning** | Trade Ideas, Trendlyne premium | Irrelevant at a 30-day horizon. |
| **Multi-asset** | All five | Equity-only by design. |

### Tier 3 — deliberately absent

| Feature | Who has it | Why not here |
|---|---|---|
| **Strategy marketplace** | Tradetron, QuantConnect | Buying someone else's edge is how retail loses money. |
| **Copy trading** | Several | Same. |
| **Pattern recognition** | TrendSpider, Trade Ideas | Would not survive the factor neutralisation already applied. |
| **Community sentiment** | TradingView | Same. |
| **Hundreds of scan criteria** | Trendlyne (1,400+) | Twelve signals were tested; ten failed. More criteria means more false positives. |

---

## The asymmetry, stated plainly

**What the commercial platforms sell:** the ability to find setups quickly,
chart them beautifully, and act on them immediately.

**What none of them tell you:** whether the setup has any predictive content
once known factors are controlled for, and whether the edge survives costs and
tax.

Trendlyne's DVM backtesting shows "consistent outperformance for high DVM-scored
stocks" — but with no factor neutralisation, no permutation test, and no cost
model, that claim carries roughly the weight SwingScope's own v1 composite did
before it was tested. That composite scored an apparently respectable IC of
0.0506 and turned out to have residual IC of 0.0041.

**This is the genuine competitive position:** SwingScope cannot compete on
product polish and should not try. It can answer a question the category
largely ignores.

---

## Recommended priorities

### Build

**1. Position monitoring and alerts.** Every competitor has this. It is the
obvious missing half, and it is needed the day real money is committed. Open
positions, live P&L against recorded stops and targets, Telegram alert on stop
hit or target reached, days-held against the 30-day horizon.

**2. Trade attribution.** Nobody in the category has it. Decomposing each
outcome into market, sector and idiosyncratic components would make eight weeks
of forward data far more informative than a bare IC — and it plays to the
existing strength rather than chasing a feature race.

**3. Scenario stress testing.** Marked NOT MET in the compliance audit. The
20-year dataset makes it possible now.

### Consider later

**4. Mobile-friendly view.** A simplified read-only layout for the bucket, stops
and open positions. Not an app — a phone-shaped page.

**5. Fundamental screening depth.** Only if a fundamental signal ever clears the
signal laboratory. Novel data is not predictive data.

### Do not build

Execution, marketplace, copy trading, pattern recognition, real-time scanning,
parameter optimisation, multi-asset. Each either contradicts the validation
discipline or solves a problem a 30-day horizon does not have.

---

## The honest closing point

None of these gaps is the binding constraint.

**Zero live forward evidence is.** A platform with perfect charting, mobile
apps and broker integration wrapped around an unvalidated signal is a faster way
to lose money. The reverse — rigorous validation with a plain interface — is
merely inconvenient.

Eight weeks. Then decide what to build.

---

*Comparison based on public product information as of July 2026. Pricing and
features change; verify before relying on any specific figure.*
