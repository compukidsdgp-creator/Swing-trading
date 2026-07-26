# SwingScope

An NSE swing-trading research dashboard built for **15–20 trading day holds**.

Screens a live universe on a transparent composite score, sizes positions from
volatility, analyses news sentiment, and — most importantly — includes the tools
to find out whether any of it actually works.

> **Research tool, not a signal service.** Data is end-of-day and delayed.
> A high score is not a prediction. Swing trading carries real risk of capital
> loss, and SEBI's own studies find most active retail traders lose money net of
> costs. Nothing here is investment advice.

---

## What it does

| Tab | Purpose |
|---|---|
| 🔍 **Screener** | Ranks a live NSE universe 0–100. Regime banner gates which cap tiers are permitted. |
| 📊 **Detail** | Candlestick chart with indicators, ATR position sizing, liquidity caps, R-multiple targets. |
| 🧪 **Backtest** | Walk-forward simulation. No lookahead: entries fill at the next bar's open. |
| 🔬 **Validation** | Information Coefficient with a permutation test — does the score rank forward returns? |
| 📋 **Forward log** | Records picks *before* outcomes exist. The one evidence type that can't be overfitted. |
| 📰 **News** | Finance-tuned sentiment with a summary panel. Handles negation and contrast clauses. |
| 📓 **Journal** | Trade log with win rate, avg R, and expectancy. |
| ❓ **Method** | Full documentation of the scoring logic. |

Plus **automated weekly and month-end reports** via GitHub Actions, delivered by
email as HTML and PDF.

---

## Quick start (local)

```bash
git clone https://github.com/<you>/swingscope.git
cd swingscope
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501.

**Windows:**
```cmd
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy to Streamlit Cloud

1. Push this repo to GitHub (**public**, for the free tier).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Repo: your repo · Branch: `main` · Main file: `app.py`
4. Deploy. First build takes 2–3 minutes.

**Optional** — for one-click report links, add under *Settings → Secrets*:

```toml
[reports]
github_user   = "yourname"
github_repo   = "swingscope"
pages_enabled = false
```

No API keys are needed. yfinance, NSE and Google News RSS are all keyless.

### Free-tier notes

- Start with Nifty 50; ~130 tickers is the practical ceiling on a cold start.
- yfinance rate-limits — results cache for 30 minutes. If you see empty
  results, wait rather than hammering refresh.
- The app sleeps after inactivity; first load takes ~30s.
- **Session state resets on restart.** Download the forward log and journal
  every session, or rely on the GitHub automation for persistence.

---

## Set up the automation (optional but recommended)

The workflow runs every Monday, records picks to `forward_log.csv`, evaluates
matured ones, generates reports, and commits everything back.

**1. Enable write permissions**
Repo → Settings → Actions → General → Workflow permissions → **Read and write**.
Without this the commit step fails silently and your forward log never grows.

**2. Enable email delivery** (optional)
Repo → Settings → Secrets and variables → Actions → New repository secret:

| Secret | Example |
|---|---|
| `SMTP_HOST` | `smtp.gmail.com` |
| `SMTP_PORT` | `465` |
| `SMTP_USER` | `you@gmail.com` |
| `SMTP_PASS` | Gmail **App Password**, not your account password |
| `REPORT_TO` | `you@gmail.com` |

**3. Test it** — Actions tab → *SwingScope weekly run* → **Run workflow**.

**Schedules:** Mondays 03:30 UTC (09:00 IST) for the weekly run; the 1st of each
month 04:00 UTC for the month-end review.

**Run manually:**
```bash
python automate.py --universe "Nifty 500" --top 10 --horizon 15
python automate.py --evaluate-only              # score matured picks only
python automate.py --monthly --backtest-ic 0.045
python automate.py --no-pdf                     # skip WeasyPrint
```

---

## Where reports are saved

```
reports/
├── latest.html / .pdf / .txt        always the most recent weekly
├── latest_monthly.html / .pdf       most recent month-end review
├── report_YYYYMMDD.*                dated weekly archive
└── month_end_YYYYMM.*               dated monthly archive
forward_log.csv                      accumulating pick record
```

Access them via email, the GitHub `reports/` folder, an Actions artifact
(90-day retention), or `git pull`. Enable GitHub Pages for one-click browser
viewing at `https://<user>.github.io/<repo>/reports/latest.html`.

---

## Project structure

```
swingscope/
├── app.py                  Streamlit UI — 8 tabs
│
│   ── Data & scoring ──
├── config.py               Benchmark, report links, method docs
├── universe.py             Live NSE constituents + live market screens
├── fallback_universe.py    Bundled snapshot for when NSE is unreachable
├── indicators.py           RSI, MACD, ATR, ADX, Bollinger — pure pandas
├── scoring.py              Composite score (tier-aware)
├── tiers.py                Large/mid/small classification and parameters
├── regime.py               Risk-on / neutral / risk-off gate
│
│   ── Evidence ──
├── backtest.py             Walk-forward simulation, no lookahead
├── validate.py             Information Coefficient + permutation test
├── forward_log.py          Forward paper-trading record
│
│   ── News ──
├── newsfeed.py             Google News RSS retrieval
├── sentiment.py            Finance lexicon, negation, contrast clauses
│
│   ── Reporting ──
├── report.py               Weekly HTML/PDF report + email
├── monthly.py              Month-end review with confidence calibration
├── automate.py             Headless runner for CI
│
├── requirements.txt
├── .github/workflows/weekly.yml
├── .streamlit/secrets.toml.example
├── .gitignore
│
│   ── Docs ──
├── MANUAL.md               Full user manual (also as PDF)
├── ROADMAP_30_DAYS.md      Day-by-day validation plan (also as PDF)
└── README.md
```

No TA-Lib dependency — every indicator is implemented in pandas, because TA-Lib
needs a C build step Streamlit Cloud doesn't handle.

---

## The score

| Component | Large | Mid | Small | Measures |
|---|---|---|---|---|
| Trend | 20% | 25% | 30% | Price vs 20/50/200 EMA, alignment, ADX 20–40 |
| Momentum | 15% | 20% | 28% | RSI in tier-specific band, MACD posture |
| Volume | 10% | 15% | 22% | Volume vs 20d average, up vs down day volume |
| Relative strength | 35% | 20% | 10% | 20d and 60d return vs Nifty 50 |
| Setup | 20% | 20% | 10% | Bollinger squeeze, distance from 52w high, pullback depth |

Two design choices worth understanding before trusting the output:

**RSI above the tier band is penalised, not rewarded.** Buying extended momentum
on a 15–20 day hold is the classic way to get mean-reverted. The band widens for
small caps (peak 60–75) because small-cap trends genuinely persist longer.

**Relative strength dominates for large caps (35%).** Large caps are mostly index
beta — one rising slower than the Nifty isn't strong, it's a slower index fund.
It falls to 10% for small caps, where stock-specific flows dominate and
comparison to the index is mostly noise.

Disagree? Edit `TIER_WEIGHTS` in `tiers.py`. The transparency is the point.

---

## Before you trust it

Run **🔬 Validation** first: horizon 15, period 5y, 40 stocks, permutations 400,
overlapping windows off.

| Mean IC | Interpretation |
|---|---|
| ≤ 0 | No signal. Stop. |
| < 0.02, or p > 0.10 | Indistinguishable from random. Stop. |
| 0.02–0.06 with p < 0.05 | Weak but real — proceed to paper trading |
| > 0.15 | Implausibly high. Look for a bug before celebrating. |

Then follow `ROADMAP_30_DAYS.md`. It's a day-by-day plan, most days of which say
"do nothing" — deliberately.

---

## Known limitations

- **Not predictive.** It measures whether a chart currently resembles a
  trend-continuation setup, not whether this instance will work.
- **EOD data only.** Confirm every level on your broker terminal.
- **Survivorship bias.** Universes are today's constituents; delisted failures
  are invisible, which flatters any backtest.
- **Bull-market bias.** Indian equities trended up through most of 2020–2025.
- **No fundamental screen.** A stock can score 90 while the business rots.
- **Sentiment is a lexicon, not comprehension.** Use it to decide what to read,
  never as a trading input.
- **NSE fetching is fragile.** NSE blocks bot traffic; the app falls back to a
  bundled snapshot and labels itself non-live when that happens.

---

## Disclaimer

Educational and research use only. Not investment advice. The author is not a
SEBI-registered investment adviser. Verify all data independently before acting,
and consider consulting a registered adviser.
