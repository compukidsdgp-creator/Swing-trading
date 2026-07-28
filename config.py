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

**12-1 momentum.** Return over the past twelve months, excluding the most recent
month. Stocks are ranked against each other; the score is a percentile within
today's universe.

**Configuration: top 10 from a 400-stock universe, held ~30 days.**

## How it got here

The original five-component composite failed factor neutralisation — residual IC
+0.0041, t = 0.17. All five components measured the same thing. Twelve candidate
signals were then tested individually; ten showed no incremental content, and of
the two survivors one was rejected as 97% correlated with the other.

That left 12-1 momentum, validated on five years at IC 0.076 with a 1.42% gross
spread.

**A twenty-year test then revised that substantially.**

| | 5-year | 20-year |
|---|---|---|
| Windows | 62 | 201 |
| Mean IC | 0.0760 | **0.0311** |
| t-statistic | 4.22 | **2.74** |
| Gross spread | 1.42% | **0.46%** |

The signal is still real — positive in every sub-period, permutation p = 0.00.
But the magnitude collapsed, and magnitude is what pays.

## Two findings that fixed it

**Universe breadth.** Holding the period fixed and varying only which symbols
are included:

| Universe | Gross spread |
|---|---|
| 150 symbols (15y+ history) | 0.22% |
| 250 symbols (10y+ history) | 0.19% |
| **400 symbols (5y+ history)** | **0.87%** |

Filtering for long history selects large, established, efficiently-priced
companies — where momentum works least well. Screen broadly, select narrowly.

**Holding period.** What a long-only book actually captures is the top slice's
excess over the universe mean, not the long-short spread:

| Hold | Long-only edge | Net annualised |
|---|---|---|
| 15 days | 0.77% | **−1.15%** |
| **30 days** | **2.16%** | **+5.95%** |
| 45 days | 1.67% | +2.41% |
| 60 days | 2.79% | +4.44% |

At fifteen days the strategy loses money after charges, slippage and 20%
short-term capital gains tax. At thirty it does not. That single change is the
difference.

## What this is not

Momentum was published by Jegadeesh & Titman in 1993 and replicated across
almost every market since. This is not a discovery — it is confirmation that a
known effect is present in this universe, and measurement of the conditions
under which it survives costs.

## Honest caveats

**Twenty-four configurations were tested and the best is reported.** Some of the
margin above is selection, not signal.

**Window counts are thin.** A 30-day hold with non-overlapping windows gives
roughly 45 observations across the full sample, fewer per sub-period.

**Survivorship bias remains.** Every symbol in the dataset still trades today,
so twenty years of failures are invisible. Absolute returns are optimistic;
regime behaviour is the more reliable conclusion.

**Thirty days is not swing trading.** It is position trading, with different
drawdown behaviour and a different psychological demand.

**Forward evidence is the only check that cannot be gamed**, and there is none
yet.
"""
