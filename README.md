# SwingScope

An NSE swing-trading research dashboard built for **15–20 trading day holds**.

Screens a configurable universe on a transparent composite score, charts the
setups, pulls headlines, sizes positions from ATR, and tracks expectancy in a
trade journal.

> **This is a research tool, not a signal service.** Data is end-of-day and
> delayed. Nothing here predicts prices. Swing trading carries real risk of
> capital loss — SEBI's own studies find the majority of active retail traders
> lose money net of costs.

---

## Quick start

```bash
git clone <your-repo-url>
cd swingscope
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501.

## Deploying to Streamlit Community Cloud

1. Push this folder to a **public** GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Point it at your repo, branch `main`, main file `app.py`.
4. Deploy. First build takes 2–3 minutes.

No secrets or API keys are required — Google News RSS and yfinance are both
keyless. If you later add a paid news API, put the key in
**App settings → Secrets** and read it with `st.secrets["KEY_NAME"]`.

### Deployment notes

- **Free tier resource limits are real.** Scanning "Everything" (~130 tickers)
  can approach the memory ceiling on a cold start. Start with Nifty 50.
- **yfinance rate-limits aggressively.** Results are cached 30 minutes
  (`@st.cache_data(ttl=1800)`). If you see empty results, wait a minute rather
  than hammering refresh.
- **Cold starts are slow.** The app sleeps after inactivity on the free tier;
  first load after sleep takes ~30s.

---

## Project layout

```
swingscope/
├── app.py            # Streamlit UI — tabs, sidebar, rendering
├── scoring.py        # Composite score. Edit WEIGHTS here.
├── indicators.py     # RSI, MACD, ATR, ADX, Bollinger — pure pandas
├── newsfeed.py       # Google News RSS + keyword sentiment
├── config.py         # Stock universes + methodology docs
└── requirements.txt
```

No TA-Lib dependency — every indicator is implemented in pandas, because
TA-Lib needs a C build step that Streamlit Cloud doesn't handle cleanly.

---

## The score

| Component | Weight | Measures |
|---|---|---|
| Trend | 25% | Price vs 20/50/200 EMAs, EMA alignment, ADX 20–40 |
| Momentum | 20% | RSI(14) 55–65 sweet spot, MACD posture |
| Volume | 15% | Volume vs 20d average, up-day vs down-day volume |
| Relative Strength | 20% | 20d and 60d return vs Nifty 50 |
| Setup | 20% | Bollinger squeeze percentile, distance from 52w high, pullback depth |

Two choices worth understanding before you trust the output:

**RSI > 72 is penalised.** The score peaks at RSI 55–65. Buying extended
momentum on a 15–20 day horizon is the classic way to get mean-reverted.

**ADX > 40 scores lower than ADX 20–40.** Extreme trend strength more often
marks exhaustion than continuation over this window.

Disagree with either? Edit `WEIGHTS` and the band logic in `scoring.py`. The
whole point of the transparent design is that you can.

---

## Known limitations

- **Not predictive.** It measures whether a chart *currently resembles* a
  trend-continuation setup. It says nothing about whether this instance works.
- **EOD data only.** Confirm every level on your broker terminal.
- **No transaction costs modelled.** STT, brokerage, stamp duty, exchange
  charges and GST materially erode short-hold returns in India. A setup needing
  a 3% move may need 3.5%+ after costs.
- **Survivorship bias.** Universes are today's constituents; delisted names are
  invisible.
- **No fundamental screen.** A stock can score 90 while the business rots.
- **Journal is ephemeral.** Session state resets on restart — download it, or
  wire it to Google Sheets / Supabase.

---

## Extending it

**FII/DII flows** — NSE publishes daily participant-wise data at
`nsearchives.nseindia.com`. Scraping it is fragile (needs cookie/header
priming); a paid feed is more reliable.

**Delivery percentage** — high delivery on an up day implies genuine
accumulation rather than intraday churn. It's in the NSE bhavcopy.

**Bulk & block deals** — published daily by NSE. Large institutional entries
are among the few genuinely tradeable catalysts.

**Earnings calendar** — `yf.Ticker(x).calendar` returns the next earnings date.
Worth excluding anything reporting inside your holding window; earnings turn a
technical setup into a coin flip.

**Sector rotation** — score sector indices with the same model, then prefer
stocks in the top 2–3 sectors. Over 15–20 days, sector beta often dominates
single-stock factors.

**Backtesting** — the current build has no backtest. Before trusting the score,
walk it forward: compute it on historical bars, hold 15–20 days, measure the
distribution of outcomes. If the edge isn't visible over a few hundred trades,
it isn't there.

---

## Suggested workflow

1. Run the screener **weekly**, not daily. Overtrading is the most reliable way
   to lose money in this game.
2. Take the top 10–15 and look at every chart manually. The score reduces what
   you examine; it doesn't replace examining it.
3. Check the news tab for anything scheduled inside your window.
4. Size from ATR, never from conviction.
5. Log every trade. Review expectancy monthly. **If expectancy is negative over
   30+ trades, the edge isn't there** — and no position-sizing tweak fixes that.

---

## Disclaimer

Educational and research use only. Not investment advice. The author is not a
SEBI-registered investment adviser. Verify all data independently before acting
on it, and consider consulting a registered adviser.
