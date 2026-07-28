# SwingScope — 20-Year Validation Report

*Run 28 July 2026 on 451 NSE symbols, 2006–2026, 1.9 million daily bars.*

---

## Headline

**The five-year result did not survive a longer test.**

| Measure | 5-year (yfinance, Nifty 500) | 16-year (local dataset, 150 symbols) |
|---|---|---|
| Mean IC | +0.0760 | **+0.0311** |
| t-statistic | 4.22 | **2.74** |
| Windows | 62 | **201** |
| Permutation p | 0.02 | 0.00 |
| Positive windows | 59.7% | 61.7% |
| **Gross quintile spread** | **1.42%** | **0.46%** |

The signal is still statistically real — 201 windows, t = 2.74, permutation
p = 0.00, positive in every sub-period tested. That part holds.

**What collapsed is the magnitude.** And magnitude is the only thing that pays.

---

## The economics

Gross spread is what covers costs. At 0.46% it does not.

| Spread | Tier | Net per cycle | Annualised | Viable |
|---|---|---|---|---|
| 1.42% (5y) | large | +0.475% | +7.91% | ✅ |
| 1.42% (5y) | mid | +0.297% | +4.95% | ✅ |
| **0.46% (16y)** | **large** | **−0.063%** | **−1.05%** | ❌ |
| **0.46% (16y)** | **mid** | **−0.241%** | **−4.01%** | ❌ |

Charges 0.26%, slippage 0.10–0.30%, tax 20% on gains. A 0.46% gross spread
leaves nothing.

---

## Sub-period detail

Positive in **4 of 4** periods — the direction is consistent. The *size* is not.

| Period | IC | t | Gross spread | Windows |
|---|---|---|---|---|
| 2010–2013 | +0.0338 | 0.93 | 0.30% | 29 |
| 2014–2017 | +0.0492 | 2.33 | **0.96%** | 40 |
| 2018–2021 | +0.0270 | 0.75 | 0.08% | 40 |
| 2022–2026 | +0.0231 | 1.28 | 0.22% | 48 |

Spread ranges from **0.08% to 0.96%** — an order of magnitude. Only 2014–2017
produced anything close to tradeable.

---

## The awkward detail

**The 2022–2026 sub-period here shows a 0.22% spread. The earlier test on the
same calendar period showed 1.42%.**

Same years. Six times the difference. The variable is not time — it is the
universe.

- **Earlier test:** Nifty 500, trimmed to the 100 most liquid, yfinance data
- **This test:** 150 symbols filtered to those with 15+ years of history

Filtering for long history selects **older, larger, more established companies**.
Momentum is documented to work less well in large caps, where pricing is more
efficient. The 1.42% appears to have been specific to a broader, more
mid-cap-weighted universe — not a property of the signal.

**Which figure is right?** Both, for their respective universes. The honest
reading is that the edge is **highly universe-dependent**, and the earlier
number was not the general case.

---

## Data quality findings

The dataset needed repair before it could be used.

**Pre-listing padding.** yfinance pads periods before a stock traded with a
repeated placeholder price and zero volume. When real data begins, the jump
registers as an enormous return:

| Symbol | Apparent move | After trimming |
|---|---|---|
| HINDZINC | **+5,575%** | +25.6% |
| ABBOTINDIA | +188% | +17.5% |

A momentum screen would have ranked those first. Four series in the 150-symbol
sample required trimming; the loader now detects and removes padding
automatically.

**Coverage is not constant.** 261 symbols have 2006 data against 451 in 2026 —
so earlier periods are tested on a smaller, more established sample. This is
itself a form of survivorship bias.

**Survivorship confirmed.** Zero files end early. Every symbol still trades
today, so twenty years of failures are invisible. Absolute returns are therefore
optimistic; regime behaviour is the more reliable conclusion.

---

## What this changes

**Before:** momentum validated at IC 0.076, spread 1.42%, annualising to roughly
+8% net after all costs. Viable.

**After:** the signal is real but far weaker over a longer window, universe-
dependent to a degree that undermines the earlier point estimate, and at
0.46% spread it does not cover costs.

**The 5-year result was not wrong.** It measured what it measured. But it was a
favourable universe over a favourable period, and treating it as the general
case would have been a mistake — one this test caught before any capital was
committed.

---

## Universe hypothesis — tested

The suspicion that universe composition, not time, explained the gap was
testable. Holding the period fixed at 2022–2026 and varying only which symbols
are included:

| Universe | Symbols | IC | t | Gross spread |
|---|---|---|---|---|
| 15y+ history | 150 | +0.0231 | 1.28 | 0.22% |
| 10y+ history | 250 | +0.0265 | 1.53 | 0.19% |
| **5y+ history** | **400** | **+0.0445** | **2.67** | **0.87%** |

**Confirmed.** Broadening the universe from 150 to 400 symbols lifts the spread
from 0.22% to 0.87% — a fourfold increase, over the identical period.

Filtering for long history selects large, established, efficiently-priced
companies. Momentum works less well there, which is exactly what the literature
predicts. The effect lives in the newer, smaller, less-covered end of the market.

This resolves the apparent contradiction. The 1.42% and the 0.46% were both
real; they measured different universes.

**And it points somewhere specific:** if the edge concentrates in the broader
universe, the strategy should target that segment explicitly rather than
defaulting to the most liquid names. But that runs directly into the cost
constraint — smaller companies carry 0.70% slippage against 0.10% for large
caps, which is precisely where a 0.87% gross spread gets consumed.

That tension is the central problem, and it is not resolved by more data.

---

## Recommended position

**Do not trade on the 5-year number.** It is not representative.

**The universe question is now answered.** The edge concentrates in the broader,
smaller-cap end: 0.87% spread across 400 symbols against 0.22% across the most
established 150.

**But that creates the real problem.** The segment where the edge lives is the
segment where costs are highest — 0.70% round-trip slippage for small caps
against 0.10% for large. A 0.87% gross spread minus 0.70% slippage minus 0.26%
charges minus tax is negative.

**The edge and the cost sit in the same place.** That is the finding, and no
further historical analysis changes it.

**The forward log now matters more, not less.** It is universe-agnostic and
measures what actually happens. Eight weeks of it is worth more than any further
historical slicing.

---

## Method note

Momentum (12-1) does not use the benchmark, so the equal-weight composite used
here affects only the regime gate, not the IC. Prices are `Adj Close` (total
return) with OHLC rescaled by the adjustment factor for internal consistency.
Windows are non-overlapping at horizon + 3 days to avoid weekday clustering.

---

*Research output. Not investment advice.*
