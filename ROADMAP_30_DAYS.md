# SwingScope — 30-Day Sharpening Plan

## Read this first

You asked me to accumulate ideas daily and improve the app over 30 days. I can't
do that — I have no memory between conversations, and no ability to work while
you're away. Every session with me starts blank.

But the honest reframe is better than the original request:

**This app will not get sharper from more ideas. It will get sharper from
evidence.** I could add fifty more features and most would make it worse — more
parameters means more ways to fit noise and more false confidence. What actually
sharpens it is the one thing only elapsed time can produce: **forward picks,
recorded before the outcome is known, then scored.**

That is the entire point of the next 30 days. Not cleverness. Data.

**Your total time commitment: about 30 minutes a week, plus four longer
sessions.** Most days on this plan say "do nothing," and that is deliberate —
overtrading and over-tinkering are the two most reliable ways to destroy a
trading system.

---

## Week 1 — Establish the truth

The goal this week is to find out whether you have anything at all. Do not skip
to trading.

### Day 1 (60–90 min) — Baseline validation

1. Deploy the app if you haven't.
2. Open **🔬 Validation**. Settings: horizon 15, period 5y, stocks 40,
   permutations 400, overlapping windows **off**.
3. Run it. Record in a notebook:
   - Mean IC: ______
   - t-statistic: ______
   - Permutation p-value: ______
   - Quintiles monotonic? ______

**Decision gate:**

| Result | What it means | Action |
|---|---|---|
| IC ≤ 0 | Score doesn't rank returns | **Stop.** Don't paper trade. Go to Day 3. |
| p > 0.10 | Indistinguishable from random | **Stop.** Same as above. |
| IC 0.02–0.06, p < 0.05 | Weak but real | Proceed to Day 2 |
| IC > 0.15 | Too good — suspect a bug | Investigate before believing it |

Write the number down. Everything in week 4 compares against it.

### Day 2 (45 min) — Backtest baseline

Open **🧪 Backtest**. Threshold 65, hold 18, check every 5, 30 stocks, regime
filter on, costs on. Record: trades, win rate, expectancy, max drawdown, and
whether expectancy rises across score buckets.

Then run it again with the regime filter **off**. Does the filter help? Write
both down.

### Day 3 (30 min) — Sensitivity check

Re-run validation at horizons 10, 15, 20, and 25 days.

**What you want to see:** IC roughly stable across horizons. **What kills the
model:** IC strongly positive at 15 and negative at 20. That means you found a
horizon-specific artifact, not a signal.

### Day 4 (20 min) — First forward snapshot

1. Run the Screener. Note the regime.
2. Open **📋 Forward log** → Snapshot top 10 picks.
3. **Download the CSV.** Do this every single time.

This is the most important twenty minutes of the month. The clock starts now.

### Days 5–7 — Do nothing

Genuinely nothing. No re-running, no tinkering. If you want to be useful, read
the manual's glossary and sections 10–13 properly.

---

## Week 2 — Accumulate, don't optimise

### Day 8 (20 min) — Snapshot 2

Screener → Forward log → snapshot → download. Note the regime state.

### Day 9 (60 min) — Add earnings exclusion

The highest-value code change available, and it needs no new data source.

```python
# Add to a new file, earnings.py
import yfinance as yf
import datetime as dt
import streamlit as st

@st.cache_data(ttl=60*60*24, show_spinner=False)
def next_earnings(ticker: str) -> dt.date | None:
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is None:
            return None
        val = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if isinstance(val, list) and val:
            val = val[0]
        return val.date() if hasattr(val, "date") else val
    except Exception:
        return None

def reports_within(ticker: str, days: int = 20) -> bool:
    d = next_earnings(ticker)
    if d is None:
        return False
    return 0 <= (d - dt.date.today()).days <= days
```

Then filter the screener results, or just add an "Earnings soon" warning column.
A technical setup with results inside the holding window is a coin flip wearing
a costume.

### Days 10–14 — Do nothing

---

## Week 3 — First signal, first refinements

### Day 15 (30 min) — Snapshot 3 + first evaluation

Snapshot as usual. Then hit **Evaluate** — your Day 4 picks have matured.

You now have your first forward data point. **Ignore what it says.** Ten picks
is pure noise. You are looking for the process to work, not the result.

### Day 16 (60 min) — Sector concentration guard

Real risk you currently carry: five financials in your top ten isn't five
positions, it's one leveraged bet on interest rates. Add a cap of two names per
sector to the screener output, using `yf.Ticker(t).info.get("sector")` cached
daily.

### Day 18 (45 min) — Parameter stability test

Re-run validation with the score threshold at 60, 65, 70, and 75.

**Healthy:** IC changes gradually. **Fatal:** IC is 0.06 at threshold 65 and
0.01 at 70. That's a knife-edge, which means you found noise. A real edge is
robust to small parameter changes.

### Days 17, 19–21 — Do nothing

---

## Week 4 — The verdict

### Day 22 (20 min) — Snapshot 4 + evaluate

### Day 24 (60 min) — Delivery percentage (optional, higher effort)

NSE's bhavcopy carries delivery percentage — the share of volume that resulted in
actual delivery rather than intraday churn. Above 50% on an up-move suggests
genuine accumulation. It's the best free proxy for "is this real money."

Fragile to scrape; skip it if the earnings and sector work isn't finished.

### Day 29 (20 min) — Snapshot 5 + evaluate

### Day 30 (90 min) — The reckoning

This is the session everything else was building toward.

1. Open **📋 Forward log**. Evaluate everything mature.
2. Note forward IC and the score-bucket table.
3. Enter your Day 1 backtest IC in the comparison box.
4. Read what it tells you.

**How to interpret:**

| Forward vs backtest IC | Meaning | Action |
|---|---|---|
| Forward ≤ 0, backtest > 0 | Classic overfitting signature | **Do not trade.** Rebuild or abandon. |
| Forward < 40% of backtest | Substantial decay | Paper trade 3 more months |
| Forward 40–80% of backtest | Normal, healthy decay | Consider small live size |
| Forward > 80% of backtest | Holding up | Still only 5 snapshots — keep going |

**The honest caveat:** five snapshots of ten picks is fifty observations. That is
not enough to conclude much. Thirty days tells you whether the *process* works
and catches gross overfitting. It does not confirm an edge. Twelve weeks starts
to mean something.

---

## What to do on day 31

**If forward IC is negative:** you just saved yourself real money. Most people
discover this with capital instead of a spreadsheet. Either rebuild the score
from a different premise, or accept that discretionary chart reading with strict
risk management may serve you better than a systematic model.

**If forward IC is weakly positive:** keep paper trading for another eight weeks
before risking anything. Add one improvement at a time and re-measure — never two
at once, or you won't know which one mattered.

**If forward IC is solidly positive:** start live at **quarter size**. Your first
live trades should be small enough that being wrong about everything costs you
tuition, not capital.

---

## The discipline that makes this work

**Change one thing at a time.** Two simultaneous changes and you learn nothing
about either.

**Record before you know.** Every snapshot must be committed before the outcome
exists. The moment you start filtering picks after the fact, the log becomes
worthless.

**Don't re-run validation after every tweak.** That's how you overfit to your own
test set. Set parameters from reasoning, then measure once.

**Download the CSV every session.** Session state resets. Losing four weeks of
forward data because you closed a tab would be genuinely painful.

**Expect boredom.** Most days here say do nothing. That is the plan working, not
the plan failing.

---

## Working with me over these 30 days

Since I don't remember previous sessions, when you come back paste in:

1. The numbers you recorded (validation IC, backtest expectancy, forward IC)
2. Your current forward log CSV
3. What you changed since last time

With that I can pick up properly — analyse the results, debug what broke,
implement the next enhancement. What I can't do is remember it myself.

---

*Educational and research use only. Not investment advice. The plan above is a
research methodology, not a recommendation to trade. Consider consulting a
SEBI-registered adviser.*
