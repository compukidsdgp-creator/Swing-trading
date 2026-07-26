"""Stock universes and documentation strings."""

BENCHMARK = "^NSEI"  # Nifty 50

# --------------------------------------------------------------------------
# Report hosting links
# --------------------------------------------------------------------------
# Set these to your own repo and the buttons appear automatically in the app.
# You can override any of them without editing code by adding a [reports]
# section to .streamlit/secrets.toml (or Streamlit Cloud -> Settings -> Secrets):
#
#   [reports]
#   github_user = "yourname"
#   github_repo = "swingscope"
#   pages_enabled = true
#
GITHUB_USER = ""          # e.g. "yourname"
GITHUB_REPO = "swingscope"
PAGES_ENABLED = False     # True once GitHub Pages is switched on for the repo


def report_links() -> dict[str, str]:
    """Build report URLs from config, overridden by st.secrets if present."""
    user, repo, pages = GITHUB_USER, GITHUB_REPO, PAGES_ENABLED
    try:
        import streamlit as st
        sec = st.secrets.get("reports", {}) if hasattr(st, "secrets") else {}
        user = sec.get("github_user", user)
        repo = sec.get("github_repo", repo)
        pages = bool(sec.get("pages_enabled", pages))
    except Exception:
        pass

    if not user:
        return {}

    base = f"https://github.com/{user}/{repo}"
    links = {
        "folder": f"{base}/tree/main/reports",
        "actions": f"{base}/actions",
        "log_csv": f"{base}/blob/main/forward_log.csv",
        "weekly_pdf": f"{base}/raw/main/reports/latest.pdf",
        "monthly_pdf": f"{base}/raw/main/reports/latest_monthly.pdf",
    }
    if pages:
        pbase = f"https://{user}.github.io/{repo}/reports"
        links["weekly_html"] = f"{pbase}/latest.html"
        links["monthly_html"] = f"{pbase}/latest_monthly.html"
    return links

_NIFTY50 = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "BHARTIARTL", "SBIN",
    "LT", "ITC", "HINDUNILVR", "AXISBANK", "KOTAKBANK", "BAJFINANCE", "MARUTI",
    "M&M", "SUNPHARMA", "NTPC", "TITAN", "ULTRACEMCO", "ASIANPAINT", "TATAMOTORS",
    "POWERGRID", "ONGC", "COALINDIA", "TATASTEEL", "JSWSTEEL", "HINDALCO",
    "ADANIENT", "ADANIPORTS", "GRASIM", "CIPLA", "DRREDDY", "APOLLOHOSP",
    "BAJAJFINSV", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "NESTLEIND",
    "BRITANNIA", "TATACONSUM", "HCLTECH", "WIPRO", "TECHM", "INDUSINDBK",
    "SHRIRAMFIN", "SBILIFE", "HDFCLIFE", "TRENT", "JIOFIN", "ETERNAL",
]

_MIDCAP = [
    "PERSISTENT", "COFORGE", "LTIM", "MPHASIS", "OFSS", "KPITTECH", "TATAELXSI",
    "POLYCAB", "KEI", "HAVELLS", "CUMMINSIND", "ABB", "SIEMENS", "THERMAX",
    "BEL", "HAL", "BDL", "MAZDOCK", "SOLARINDS", "ASTRAL",
    "LUPIN", "AUROPHARMA", "TORNTPHARM", "ZYDUSLIFE", "ALKEM", "LAURUSLABS",
    "MAXHEALTH", "FORTIS", "METROPOLIS", "LALPATHLAB",
    "FEDERALBNK", "IDFCFIRSTB", "BANDHANBNK", "AUBANK", "CHOLAFIN",
    "MUTHOOTFIN", "LICHSGFIN", "PFC", "RECLTD", "CANBK",
    "DIXON", "AMBER", "VOLTAS", "BLUESTARCO", "CROMPTON",
    "PIIND", "SRF", "DEEPAKNTR", "AARTIIND", "NAVINFLUOR",
]

_SMALLCAP = [
    "CAPLIPOINT", "NATCOPHARM", "JBCHEPHARM", "GRANULES", "AJANTPHARM",
    "TANLA", "ROUTE", "HAPPSTMNDS", "NEWGEN", "INTELLECT", "SONATSOFT",
    "ZENSARTECH", "BIRLASOFT", "CYIENT", "MASTEK", "NUCLEUS",
    "CDSL", "MCX", "BSE", "ANGELONE", "IIFL", "MOTILALOFS",
    "TRIVENI", "KIRLOSENG", "TDPOWERSYS", "SHILCHAR", "VOLTAMP",
    "ELECON", "GRINDWELL", "CARBORUNIV", "TIMKEN", "SCHAEFFLER",
]


def _ns(symbols: list[str]) -> tuple[str, ...]:
    return tuple(f"{s}.NS" for s in symbols)


UNIVERSES: dict[str, tuple[str, ...]] = {
    "Nifty 50": _ns(_NIFTY50),
    "Midcap focus": _ns(_MIDCAP),
    "Smallcap focus": _ns(_SMALLCAP),
    "Nifty 50 + Midcap": _ns(_NIFTY50 + _MIDCAP),
    "Everything": _ns(_NIFTY50 + _MIDCAP + _SMALLCAP),
}


METHOD_DOC = """
## How the score is built

The composite is a weighted blend of five bounded sub-scores. Nothing is hidden —
if you disagree with a weighting, edit `WEIGHTS` in `scoring.py`.

| Component | Weight | What it measures |
|---|---|---|
| **Trend** | 25% | Price above 20/50/200 EMAs, EMA alignment, ADX between 20–40 |
| **Momentum** | 20% | RSI(14) in the 55–65 sweet spot, MACD histogram positive and above signal |
| **Volume** | 15% | Recent volume vs 20-day average; up-day volume vs down-day volume |
| **Relative Strength** | 20% | 20-day and 60-day return versus Nifty 50 |
| **Setup** | 20% | Bollinger squeeze percentile, distance from 52-week high, pullback depth |

### Deliberate design choices

**RSI above 72 is penalised, not rewarded.** On a 15–20 day hold, buying extended
momentum is how swing traders get caught in mean reversion. The score peaks at
RSI 55–65.

**ADX above 40 scores lower than ADX 20–40.** Extreme trend strength more often
marks exhaustion than continuation over this horizon.

**Volume is a confirmation filter, not a driver.** It carries only 15% because
volume without trend is noise, but a breakout on thin volume fails often enough
that it needs to cost something.

**Relative strength is weighted heavily (20%).** Over 15–20 days, sector and market
beta dominate single-stock factors. A stock rising less than the index is not
actually strong.

---

## What this tool cannot do

- **It is not predictive.** It measures whether a chart currently resembles a
  trend-continuation setup. Base rates for such setups are modest, and this
  says nothing about whether *this* instance works.
- **Data is end-of-day, delayed.** yfinance is not a live feed. Confirm every
  level on your broker terminal.
- **No fundamental screening.** A stock can score 90 while the business
  deteriorates. Pair this with fundamentals.
- **Survivorship bias in the universe.** The lists are today's constituents.
  Anything delisted or removed is invisible.
- **No transaction costs modelled.** In Indian markets, STT, brokerage, stamp
  duty, exchange and GST charges materially erode short-hold returns. A setup
  needing a 3% move to be worthwhile may need 3.5% after costs.

## Suggested workflow

1. Run the screener weekly, not daily — overtrading is the most reliable way
   to lose money in swing trading.
2. Take the top 10–15 and check each chart manually. The score is a filter to
   reduce what you look at, not a substitute for looking.
3. Read the news tab for anything scheduled — earnings inside your holding
   window turn a technical trade into a coin flip.
4. Size positions from ATR, not conviction.
5. Log every trade. Review expectancy monthly. If expectancy is negative over
   30+ trades, the edge is not there and no amount of position sizing fixes it.

## Extending this

- **FII/DII flows**: NSE publishes daily participant-wise data. Scrape
  `nsearchives.nseindia.com` (fragile, needs headers/cookies) or use a paid feed.
- **Delivery percentage**: high delivery on an up move implies genuine
  accumulation rather than intraday churn. NSE bhavcopy has this.
- **Bulk/block deals**: NSE publishes these daily; large institutional entries
  are a real catalyst.
- **Earnings calendar**: `yf.Ticker(x).calendar` gives next earnings date —
  worth excluding stocks reporting inside your window.
- **Sector rotation**: score sector indices with the same model and prefer
  stocks in the top 2–3 sectors.
- **Persistence**: journal resets on restart. Wire it to Google Sheets
  (`gspread`) or Supabase for durable storage.
"""
