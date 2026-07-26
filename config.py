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
## The signal

**12-1 momentum.** Return over the past 12 months, excluding the most recent
month. Stocks are ranked against each other; the score is a percentile within
today's universe.

That is the whole model.

## How it got here

The original version blended five components — Trend, Momentum, Volume,
Relative Strength, Setup — with weights chosen by reasoning about what should
matter. Factor analysis showed all five measured the same underlying thing:

| Test | Result |
|---|---|
| Raw IC | +0.0506 |
| Residual IC after neutralising six known factors | **+0.0041** |
| IC retention | **13.9%** |
| Fama-MacBeth t on the score | **0.17** |
| Correlation with one-month return | 0.76 |

The composite predicted nothing once momentum, reversal, size, volatility, beta
and liquidity were controlled for. It was momentum wearing five costumes.

Twelve candidate signals were then tested individually. Ten showed no
incremental content. Of the two survivors, one was rejected as redundant
(`idiosyncratic_mom`, correlation 0.971 with 12-1 momentum). One remained.

| Evidence for 12-1 momentum | |
|---|---|
| Residual IC | +0.0553 |
| Newey-West t | **3.73** (clears the Harvey-Liu-Zhu t>3 bar) |
| Positive windows | 59.7% of 62 |

## What this is not

Momentum was published by Jegadeesh & Titman in 1993 and replicated across
almost every market since. This is not a discovery. What has been established is
that the effect is present and measurable in the Nifty 500 at a 15-day horizon,
and that elaborate scoring added nothing to it.

## Cost reality

IC 0.048 implies roughly a 0.5-0.6pp spread between top and bottom quintile per
15 days. Against modelled round-trip costs:

| Tier | Cost | Verdict |
|---|---|---|
| Large | 0.25% | edge survives |
| Mid | 0.60% | roughly breakeven |
| Small | 1.50% | **consumed entirely** |

Small caps are excluded regardless of rank. Being right about a stock you cannot
trade profitably is not an edge.

## Two things to watch

**Percentile ranking is relative.** A score of 100 means "strongest momentum in
today's universe", not "strong". In a falling market that could be a stock down
1% while everything else is down 30%. The absolute momentum floor exists to stop
this, and should stay on.

**Everything above is measured on selection data**, and is optimistic by
construction. The forward log is the only un-overfittable evidence. Give it
eight weeks before trusting any of it.

## What this still cannot fix

- **Delisting survivorship bias.** Companies wound up have no retrievable
  history. Point-in-time liquidity filtering removes index-membership bias but
  not this. Assume results remain modestly inflated.
- **Data quality.** yfinance carries known defects. The audit catches the worst.
- **History depth.** Five years, one market. Serious factor work wants twenty
  years across several.
"""
