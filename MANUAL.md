# SwingScope — User Manual

A complete guide to what every part of this app does, why it exists, and how to
read what it tells you.

Written assuming you know how to buy and sell a stock, but not assuming you know
what RSI, ATR, or expectancy mean. Every term is explained the first time it
appears.

---

## Table of contents

1. [What this app is (and isn't)](#1-what-this-app-is-and-isnt)
2. [The five-minute version](#2-the-five-minute-version)
3. [How the whole thing flows](#3-how-the-whole-thing-flows)
4. [The sidebar, control by control](#4-the-sidebar-control-by-control)
5. [Tab 1 — Screener](#5-tab-1--screener)
6. [Tab 2 — Detail](#6-tab-2--detail)
7. [Tab 3 — Backtest](#7-tab-3--backtest)
8. [Tab 4 — News](#8-tab-4--news)
9. [Tab 5 — Journal](#9-tab-5--journal)
10. [The score, explained properly](#10-the-score-explained-properly)
11. [Tiers — why one model can't fit all stocks](#11-tiers--why-one-model-cant-fit-all-stocks)
12. [Regime — when not to trade](#12-regime--when-not-to-trade)
13. [Position sizing and risk](#13-position-sizing-and-risk)
14. [A worked example, start to finish](#14-a-worked-example-start-to-finish)
15. [Glossary](#15-glossary)
16. [Common mistakes](#16-common-mistakes)
17. [Troubleshooting](#17-troubleshooting)

---

## 1. What this app is (and isn't)

**What it is:** a filter. It takes 50–250 stocks and ranks them by how closely
each one currently resembles a setup that often precedes a 15–20 day move. It
then helps you size the position sensibly and record what happened.

**What it is not:**

- **Not a prediction engine.** A high score does not mean the stock will go up.
  It means the chart currently has certain measurable properties. Many stocks
  with those properties go nowhere or fall.
- **Not live data.** Prices are end-of-day, at least a day old. Always confirm
  on your broker terminal before acting.
- **Not news-driven.** News never affects the ranking. See section 8 for why.
- **Not investment advice.** It's a research tool. You make every decision.

**The honest framing:** think of it as a way to look at 15 charts a week instead
of 200, with a risk framework attached. That's genuinely useful. It is not an
edge by itself.

---

## 2. The five-minute version

If you read nothing else:

1. Open the **Screener** tab. Look at the coloured banner at the top — that's
   the market regime. If it's red, seriously consider not trading this week.
2. Look at the ranked table. Higher score = closer match to the setup profile.
3. Pick a few names, open the **Detail** tab, look at the actual chart. The
   score reduces what you examine; it doesn't replace examining it.
4. Note the suggested **stop** and **quantity**. Don't exceed the quantity.
5. Check the **News** tab for those names — mainly to catch earnings dates
   falling inside your holding window.
6. If you trade it, log it in the **Journal** tab.
7. Once a month, look at your **expectancy** in the Journal. If it's negative
   after 30+ trades, stop and rethink. That number is the whole game.

And before trusting any of it: run the **Backtest** tab on real data.

---

## 3. How the whole thing flows

```
   Universe          →   Price data     →   Indicators    →    Score
(live from NSE)         (yfinance)         (RSI, MACD…)      (0–100 rank)
                                                                  ↓
                                                            Your shortlist
                                                                  ↓
   News lookup  ──────────────────────────────────────→   Sanity check
   (manual, after)                                              ↓
                                                          Position sizing
                                                                  ↓
                                                             Journal
```

The important thing to notice: **news enters at the end, manually.** It has no
influence on the ranking at all.

---

## 4. The sidebar, control by control

### Universe section

**Build universe from** — three modes:

| Mode | What it does | When to use |
|---|---|---|
| Index constituents (live) | Fetches the actual current members of an NSE index | Normal weekly scanning |
| Live market screen | Builds from what's moving today — most active, top gainers, near 52-week highs | When you want to see where activity is concentrated right now |
| Custom list only | Just your typed tickers | Focused work on specific names |

**Index** — 26 available: broad (Nifty 50, 100, 200, 500), size-based (Midcap
150, Smallcap 250), and sector (Bank, IT, Pharma, Auto, Metal, Energy, Realty,
and more). Sector indices are useful when you have a view on a theme.

**The green/orange badge** — green "Live" means it fetched successfully from
NSE, with a timestamp. Orange "Cached" means NSE was unreachable and you're
looking at a bundled snapshot that may be out of date. Always check this.

**Changed since last fetch** — an expander showing which symbols entered and
left. Catches index rebalances.

**Add / define tickers** — type extra symbols, comma-separated. Use NSE symbols
exactly as they appear (`M&M`, not `MAHINDRA`). Wrong symbols fail silently —
they just don't appear.

**Cap universe size** — limits how many stocks get downloaded. The free Streamlit
tier struggles past about 130. Start smaller.

### Filters section

**Min avg daily turnover (₹ Cr)** — liquidity floor. Turnover is price × volume,
averaged over 20 days, in crores. A stock trading ₹5 Cr/day cannot absorb your
order without moving the price against you. Defaults: 50 for large caps, 25 for
mid, 10 for small.

**RSI(14) band** — filters to stocks whose RSI sits between the two values. See
the glossary for what RSI means. The default 45–70 excludes both dead stocks and
severely overheated ones.

**Require price > 50 EMA** — only show stocks trading above their 50-day
exponential moving average. This is a crude "is it in an uptrend" check. Leave
it on unless you're deliberately hunting reversals.

**Min composite score** — hides anything below this score. Raising it shows
fewer, higher-quality matches.

### Risk section

**Trading capital (₹)** — your total swing trading capital. Used for position
sizing. Be honest here; inflating it inflates every position.

**Risk per trade (%)** — how much of your capital you're willing to lose if a
trade hits its stop. 1% is a common starting point. At 1% of ₹5,00,000 you risk
₹5,000 per trade, meaning 20 consecutive losses would cost you 20% — survivable.
At 5% per trade, the same losing streak would nearly wipe you out.

**Stop = ATR ×** — how far below entry to place your stop, measured in ATR units.
See glossary. The app can override this with tier-appropriate defaults.

---

## 5. Tab 1 — Screener

### The regime banner

The coloured bar at the top. Three states:

| Colour | State | Meaning | Tiers allowed |
|---|---|---|---|
| 🟢 Green | RISK ON | Nifty above its 200-day average, and the 50-day average is rising | Large, mid, small |
| 🟠 Amber | NEUTRAL | Above the 200-day, but the 50-day is flat or falling | Large, mid |
| 🔴 Red | RISK OFF | Below the 200-day average | Large only — or nothing |

It also shows **breadth** — the percentage of your universe trading above its own
50-day average. This matters because an index can look healthy while most stocks
are already broken; five large companies can hold the Nifty up. If breadth falls
below 35%, the app downgrades the regime regardless of what the index says.

**Respect regime filter** checkbox — when ticked, tiers the regime doesn't permit
are hidden. Turning it off in a downtrend is how accounts get destroyed.

### The results table

| Column | Meaning |
|---|---|
| Ticker | Stock symbol |
| Tier | large / mid / small — auto-classified |
| Close ₹ | Last closing price |
| Score | Composite 0–100, shown as a progress bar |
| Trend | Sub-score: is it in an uptrend? |
| Mom | Sub-score: momentum |
| Vol | Sub-score: volume confirmation |
| RS | Sub-score: relative strength vs Nifty |
| Setup | Sub-score: chart structure quality |
| RSI | Raw RSI value |
| ATR % | Daily volatility as % of price |
| 20d % | Return over the last 20 days |
| ₹Cr/day | Average daily turnover |

The five sub-scores are shown deliberately. A stock scoring 75 because of pure
momentum is a very different proposition from one scoring 75 on setup quality.
Look at *how* it got its score, not just the total.

**Download results (CSV)** exports everything for your own analysis.

---

## 6. Tab 2 — Detail

Pick a stock and get:

**Header metrics** — close, RSI, ATR (with % of price), distance from the 50 EMA,
distance from the 52-week high.

**The chart** — three panels:
- **Top:** candlesticks with three moving averages overlaid. Orange = 20-day,
  blue = 50-day, grey = 200-day.
- **Middle:** volume bars, green on up-days, red on down-days.
- **Bottom:** RSI, with dotted lines at 70 and 30.

**Position sizing** — the most important part of this tab.

It shows the tier and the tier's defaults, then calculates:

- **Stop** — where you'd exit if wrong, and how far that is in percent
- **Qty** — how many shares to buy so that hitting the stop costs exactly your
  chosen risk amount
- **Exposure** — total rupees deployed, and what share of capital that represents
- **Risk** — rupees at stake

**The liquidity warning.** If ATR sizing suggests more shares than the tier's
volume cap allows, the app reduces the quantity and tells you. This is the
single most important guard for small caps, where the danger isn't being wrong —
it's being right and unable to exit because you're 10% of the day's volume.

**Cost drag line** — shows what a 2R gross win actually nets after estimated
transaction costs. On a small cap with a wide stop this can be a meaningful
haircut, and it's the reason small-cap setups need to be better than large-cap
ones to be worth the same.

**R-multiple targets table** — entry, stop, and the price levels representing
1R, 2R, 3R profit. "R" is your risk unit; see glossary.

---

## 7. Tab 3 — Backtest

**This is the most important tab, and most people will skip it. Don't.**

It answers one question: *does the score actually predict forward returns, or
does it just describe charts that already went up?*

### How it works

For every stock and every historical bar, it computes the score **using only data
available at that moment** — no peeking ahead. When the score clears your
threshold, it enters at the *next* bar's open (never the signal bar's close,
which would be cheating). It holds until either the stop is hit or the holding
period expires, then records the result.

If a bar both hits the stop and would have hit the target, it's recorded as a
loss — the pessimistic assumption, which is the honest one.

### Settings

- **Signal threshold** — minimum score to trigger a trade
- **Hold (bars)** — how many trading days to hold. 18 ≈ your 15–20 day horizon
- **Check every N bars** — how often signals are checked. 5 = weekly
- **Stocks to test** — more is better statistically, slower practically
- **Apply regime filter** — test with and without to see if it helps
- **Subtract transaction costs** — leave on; without it results are fantasy

### How to read the results

**The headline metrics:**

- **Trades** — sample size. Under 100 and you cannot distinguish edge from luck.
- **Win rate** — percentage profitable. **This matters less than you think.** A
  35% win rate with big winners beats a 65% win rate with small ones.
- **Expectancy** — average R per trade. **This is the only number that matters.**
  Positive means the strategy makes money over time. Negative means it loses
  money, and no amount of position sizing fixes that.
- **Total R** — cumulative risk units gained
- **Stopped out %** — how often the stop was hit

**By tier** — does the model work equally on large, mid, and small? Often it
doesn't, which tells you where to focus.

**By regime** — does the regime filter actually help? Compare expectancy in
risk-on versus risk-off.

**By score bucket — the real test.** If expectancy *rises* as score rises, the
score is genuinely ranking something. If expectancy is flat or random across
buckets, **the score is not working** and everything else in this app is
decoration. Check this table before you trust anything.

**Equity curve** — what your capital would have done, and the maximum drawdown.
Ask yourself honestly whether you could have sat through that drawdown without
abandoning the system.

### What "good" looks like

- 200+ trades
- Positive expectancy after costs
- Expectancy rising with score bucket
- Works in more than one regime, or the regime filter correctly excludes the bad one
- Max drawdown you could actually tolerate

If you don't have all five, you have a hypothesis, not a strategy.

---

## 8. Tab 4 — News

Pick stocks, get recent headlines, colour-coded by a crude keyword sentiment
score: 🟢 positive, 🔴 negative, ⚪ neutral.

**Why news is deliberately kept out of the ranking:** headlines are already
priced in. By the time Google indexes a story, it's minutes to hours old, and
institutional algorithms traded it in milliseconds. Feeding stale sentiment into
a score would add noise while *feeling* sophisticated — which is worse than
leaving it out.

**What to actually use this tab for:** checking for scheduled events inside your
holding window. Earnings especially. A technical setup with earnings in ten days
isn't a technical trade — it's a coin flip with extra steps. Also worth catching:
regulatory action, block deals, credit rating changes.

**Don't trust the sentiment colours.** "Shares fall despite strong profit" will
confuse a keyword matcher. Use them to decide what to read, never as a signal.

---

## 9. Tab 5 — Journal

Log every trade: date, ticker, side, entry, stop, target, quantity, exit, notes.
Leave exit as 0 for open positions.

Once you have closed trades it computes:

- **Win rate**
- **Net P&L**
- **Avg R** — average risk-multiple across trades
- **Expectancy** — the number that decides whether you're actually making money

**Why this matters more than the screener.** Most traders leak money through
execution, not selection — cutting winners early, letting losers run past the
stop, oversizing after a win, revenge-trading after a loss. None of that shows
up anywhere except a journal.

**Note:** the journal lives in session state and resets when the app restarts.
**Download the CSV regularly.** The README explains how to wire it to Google
Sheets for permanent storage.

---

## 10. The score, explained properly

The composite is a weighted blend of five components, each scored 0–100.

### Trend (weight varies by tier)

*Is this stock actually going up?*

Checks price against three moving averages (20, 50, 200-day), whether the
averages are stacked in the right order, and ADX for trend strength.

Full marks: price above all three averages, 20-day above 50-day, ADX between
20 and 40.

**Why ADX above 40 scores lower than 20–40:** extreme trend strength usually
marks exhaustion rather than continuation over a 15–20 day window. By the time
everyone can see the trend, much of it has happened.

### Momentum

*Is there force behind the move, without being overheated?*

Uses RSI and MACD. The RSI sweet spot varies by tier — roughly 52–62 for large
caps, 60–75 for small caps.

**Why very high RSI is penalised:** buying extended momentum on a 15–20 day hold
is the classic way to get mean-reverted. You arrive just as the move ends.

**Why the band differs by tier:** small-cap trends persist much longer. Penalising
a small cap at RSI 72 would systematically exclude exactly the moves you're
trying to catch.

### Volume

*Are people actually participating?*

Compares recent volume to the 20-day average, and up-day volume against down-day
volume. A breakout on thin volume usually fails — nobody's behind it.

### Relative strength

*Is it beating the index?*

Compares 20-day and 60-day returns against the Nifty 50.

**Why this is weighted heavily for large caps (35%):** large caps are mostly index
beta. A large cap rising slower than the index isn't strong — it's a slower index
fund. **Why it drops to 10% for small caps:** small-cap moves are driven by
stock-specific flows, so comparison to the Nifty is mostly noise.

### Setup

*Is the chart structure favourable?*

Three things: how tight the Bollinger bands are relative to their own history (a
"squeeze" means the stock is coiling), distance from the 52-week high, and how
deep the recent pullback is.

The ideal: a stock that has consolidated tightly just below its 52-week high.
That's the classic launchpad.

---

## 11. Tiers — why one model can't fit all stocks

Every stock is classified **large**, **mid**, or **small** based on market cap
(or turnover as a fallback). They behave differently enough that one set of rules
can't serve all three.

| | Large | Mid | Small |
|---|---|---|---|
| Typical daily volatility (ATR) | 1.2–2% | 1.8–3% | 2.5–5% |
| Typical 20-day move | 3–6% | 5–10% | 8–20% |
| Momentum behaviour | Mean-reverts fast | Moderate | Trends persist |
| Main driver | Index/sector beta | Sector rotation | Stock-specific flows |
| Stop distance (ATR ×) | 2.0 | 2.5 | 3.0 |
| Max % of capital | 25% | 15% | 8% |
| Max % of daily volume | 5% | 3% | 2% |
| Est. round-trip cost | 0.25% | 0.60% | 1.50% |
| **Main risk** | Opportunity cost | Both | **Can't exit** |

That last row drives everything. In large caps your risk is being wrong. In small
caps your risk is being *right and still trapped* — circuit limits, thin order
books, gap-downs. Hence the tight caps on position size relative to daily volume.

**Important:** these parameters were set by reasoning about *why* each tier
behaves as it does — not by tuning until backtest returns looked good. That
distinction is what separates a model from curve-fitting.

---

## 12. Regime — when not to trade

Small caps fall two to three times as hard as the index in drawdowns. No chart
pattern survives that. The most valuable thing this app does is sometimes tell
you to sit out.

**How the regime is determined:**

- Is the Nifty above its 200-day moving average?
- Is the 50-day moving average rising?
- What percentage of stocks are above their own 50-day average (breadth)?

**Why breadth matters:** indices are weighted by size. A handful of huge companies
can keep the Nifty above its 200-day average while most stocks are already in
downtrends. Breadth catches this. Below 35%, the regime downgrades regardless of
what the index shows.

**In risk-off:** don't just lower your expectations — the app suppresses small and
mid caps entirely. The right position size in a hostile regime is often zero.

---

## 13. Position sizing and risk

### The core idea

You don't decide position size by conviction. You decide it by arithmetic:

```
Quantity = (Capital × Risk%) ÷ (Entry − Stop)
```

If your stop is far away (volatile stock), you buy fewer shares. If it's close,
you buy more. Either way, **being wrong costs the same amount** — which is the
entire point.

### Worked example

- Capital: ₹5,00,000
- Risk per trade: 1% → ₹5,000
- Stock price: ₹800
- ATR: ₹20, tier multiple 2.5 → stop distance ₹50
- Stop price: ₹750

Quantity = ₹5,000 ÷ ₹50 = **100 shares**
Exposure = 100 × ₹800 = **₹80,000** (16% of capital)

If the stop hits, you lose ₹5,000 — exactly 1%. If the stock runs to ₹900, that's
₹100 profit per share = ₹10,000 = **2R**.

### The two caps that override this

1. **Capital cap** — no more than 25% / 15% / 8% of capital in one position
   (large / mid / small)
2. **Liquidity cap** — no more than 5% / 3% / 2% of average daily volume

Whichever is smaller wins. The app tells you which one is binding.

### Why the liquidity cap exists

If you own 10% of a small cap's daily volume and it gaps down, there is nobody
to sell to. Your stop becomes theoretical. This cap is the difference between a
bad trade and a catastrophic one.

---

## 14. A worked example, start to finish

**Monday evening.**

1. Open the app. Sidebar: Universe = "Nifty Midcap 150", capital ₹5,00,000,
   risk 1%.
2. Screener tab. Banner is **amber — NEUTRAL**. Large and mid permitted, small
   suppressed. Fine, we're scanning midcaps anyway.
3. 150 stocks scanned, 11 pass the filters. Top name scores 78 — strong on Setup
   (88) and Trend (85), moderate on Momentum (62). That profile suggests a
   consolidation near highs rather than an extended runner. Good.
4. Detail tab. Chart confirms: six weeks of tight sideways action just under the
   52-week high, volume drying up during the consolidation. Textbook coil.
5. Position sizing says: stop ₹1,142 (4.2% away), quantity 118 shares, exposure
   ₹1,42,000 (28% of capital) — **warning appears**, exceeds the 15% mid-cap cap.
   App reduces quantity to 63 shares. Accept that.
6. News tab. Headlines are routine. **But** one mentions Q2 results scheduled in
   eight days — inside the holding window. That's a coin flip, not a setup.
   **Skip this trade.**
7. Go to the number two name instead. Score 74, no earnings scheduled. Take it.
8. Journal tab: log entry ₹1,192, stop ₹1,142, quantity 63.

**Three weeks later.** Stock hits ₹1,290. Update the journal with the exit. That's
₹98 profit per share on ₹50 risk = **1.96R**.

**End of month.** Journal shows 8 closed trades, expectancy +0.31R. Positive —
keep going, keep logging.

---

## 15. Glossary

**ADV (Average Daily Volume)** — average shares traded per day, usually over 20
days. Used to cap position size so you can actually exit.

**ADX (Average Directional Index)** — measures trend *strength*, not direction.
Below 20 = no real trend. 20–40 = healthy trend. Above 40 = very strong, often
near exhaustion.

**ATR (Average True Range)** — the average distance a stock moves in a day, in
rupees. A stock with ATR ₹20 typically ranges ₹20 between high and low. Used to
place stops at a distance the stock won't hit by accident.

**Bollinger Bands / Band Width** — lines drawn two standard deviations above and
below a moving average. When they narrow ("squeeze"), the stock is coiling —
volatility is compressed and often expands afterwards.

**Breadth** — the share of stocks in a universe above their own 50-day moving
average. Tells you whether a rally is broad or just a few big names.

**Drawdown** — the percentage fall from a peak in your equity. Max drawdown is
the worst such fall. The number that decides whether you can psychologically
survive a system.

**EMA (Exponential Moving Average)** — an average of recent prices that weights
recent days more heavily than older ones. Reacts faster than a simple average.

**Expectancy** — the average profit per trade measured in R. **The single most
important number in trading.** Expectancy of +0.3R means you make 0.3 risk units
per trade on average. Negative expectancy means you lose money no matter how you
size positions.

**MACD** — the difference between two moving averages (12 and 26-day), plus a
signal line. When the MACD is above its signal line, short-term momentum is
stronger than medium-term. The "histogram" is the gap between them.

**R / R-multiple** — profit or loss measured in units of your initial risk. If
you risk ₹5,000 and make ₹10,000, that's 2R. Lets you compare trades of different
sizes on one scale.

**Regime** — the overall market environment (risk-on, neutral, risk-off).

**Relative strength (RS)** — how a stock performs compared to the index. Not the
same as RSI.

**RSI (Relative Strength Index)** — a 0–100 measure of how hard price has been
pushed recently. Above 70 is conventionally "overbought", below 30 "oversold" —
but these thresholds are much less reliable than commonly believed, especially in
small caps.

**Slippage** — the difference between the price you expected and the price you
got. Worse in illiquid stocks.

**Stop / stop-loss** — the price at which you exit a losing trade. Non-negotiable
once set.

**Swing trading** — holding positions for days to weeks. Here, 15–20 trading days.

**Turnover** — price × volume, i.e. rupee value traded. Better liquidity measure
than share volume alone.

**Win rate** — percentage of trades that made money. Much less important than
expectancy — a 35% win rate with large winners easily beats 65% with small ones.

---

## 16. Common mistakes

**Trading the score without looking at the chart.** The score is a filter to
reduce what you examine from 200 names to 15. It is not a substitute for
examining them.

**Ignoring the regime banner.** Every system looks good in a bull market. The
regime filter exists for the other times.

**Overriding the liquidity cap.** "It's only a small position" — until you try to
exit on a gap-down and discover there's no bid.

**Running the screener daily.** Overtrading is the most reliable way to lose money
in swing trading. Weekly is the intended cadence.

**Judging by win rate.** You can be right 70% of the time and still lose money if
your losses are bigger than your wins. Watch expectancy.

**Not logging trades.** Without a journal you have no idea whether you're actually
making money or just remembering the good trades.

**Tuning parameters until the backtest looks good.** With this many knobs you can
fit noise perfectly. Set parameters from reasoning, then validate — don't search
for the best-performing values.

**Trading through earnings.** A technical setup with results due inside the
holding window is a coin flip wearing a costume.

---

## 17. Troubleshooting

**"No data returned"** — yfinance rate-limits aggressively. Wait a minute, reduce
the universe size, and try again. Results cache for 30 minutes.

**Orange "Cached" badge on the universe** — NSE was unreachable. The app fell
back to a bundled list that may be stale. Try the refresh button; if it persists,
NSE may have changed its bot defences.

**A stock I added doesn't appear** — almost always a wrong symbol. It must match
NSE exactly: `M&M`, `BAJAJ-AUTO`, `M&MFIN`. Wrong symbols fail silently.

**Nothing passes the filters** — either the regime is suppressing tiers, or your
thresholds are too tight. Lower the minimum score first.

**The app is slow / times out** — reduce the universe cap. On the free Streamlit
tier, 130 tickers is roughly the practical ceiling; cold starts after inactivity
take about 30 seconds.

**Backtest produces no trades** — lower the signal threshold, or turn off the
regime filter to check whether it's suppressing everything.

**Journal disappeared** — session state resets on restart. Download the CSV
regularly, or wire it to persistent storage (see README).

---

## Final note

The most valuable output of this app is not a stock pick. It's the expectancy
number in your journal after 30 trades, and the regime banner telling you to stay
out.

If your expectancy is negative, the answer is never to size up or trade more
often. It's to change the rules — or to stop.

Educational and research use only. Not investment advice. Verify everything
independently, and consider consulting a SEBI-registered adviser.
