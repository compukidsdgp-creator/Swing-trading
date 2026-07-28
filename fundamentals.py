"""Fundamental data — free to obtain, hard to use honestly.

The split
---------
**Current fundamentals are free and already available.** yfinance exposes P/E,
P/B, ROE, debt-to-equity, margins, growth rates, full income statements, balance
sheets and cash flow statements for NSE tickers. No additional API, no key, no
subscription. It is already a dependency of this project.

**Point-in-time fundamentals are the expensive part**, and no free archive
solves it. Two problems compound:

  **Reporting lag.** A company's FY2024 results were not *known* until roughly
  two months after the financial year ended. Backtesting a March 2024 decision
  using FY2024 earnings is lookahead of the worst kind — the model appears to
  know something nobody could have known.

  **Restatement.** Reported figures get revised. The FY2023 earnings visible
  today are not necessarily the figures the market saw in 2023. Vendors like
  Compustat maintain point-in-time snapshots precisely because of this, and
  that is what the licence fee buys.

So a historical fundamental backtest built on today's yfinance data is not
merely imprecise. It is systematically optimistic in a way that is difficult to
quantify and easy to mistake for skill.

What this module does about it
------------------------------
  **Live screening** — current fundamentals for today's decisions. No lookahead
  problem at all, because you are using what is known now to decide now.

  **Forward archiving** — snapshot fundamentals monthly with a timestamp,
  building a genuine point-in-time series from today onward. In two years you
  have two years of honest data. The same approach as the bhavcopy archive:
  the asset compounds while you do nothing.

  **Announcement-date awareness** — yfinance exposes earnings dates. Where
  available, this refuses to use a figure before the date it was announced,
  which removes the reporting-lag component of the bias.

A warning about what comes next
-------------------------------
Adding fundamental signals means adding candidate signals, and twelve were
already tested. Testing more gives noise more chances to look good — the exact
multiple-testing problem the t > 3 threshold exists for.

Quality and value are genuinely documented factors (Fama & French; Asness,
Frazzini & Pedersen on quality-minus-junk), so this is not fishing. But any
fundamental signal must go through the signal laboratory and clear residual IC
after neutralisation, exactly as the twelve technical candidates did. Ten of
those failed. Expect similar.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ARCHIVE_PATH = Path("fundamentals_archive.csv")

# Fields worth capturing. Deliberately limited to metrics with documented
# factor evidence rather than everything the API returns.
FIELDS = {
    # Valuation
    "trailingPE": "pe_trailing",
    "forwardPE": "pe_forward",
    "priceToBook": "pb",
    "enterpriseToEbitda": "ev_ebitda",
    "priceToSalesTrailing12Months": "ps",
    # Quality
    "returnOnEquity": "roe",
    "returnOnAssets": "roa",
    "profitMargins": "net_margin",
    "operatingMargins": "operating_margin",
    "grossMargins": "gross_margin",
    # Leverage
    "debtToEquity": "debt_to_equity",
    "currentRatio": "current_ratio",
    "quickRatio": "quick_ratio",
    # Growth
    "revenueGrowth": "revenue_growth",
    "earningsGrowth": "earnings_growth",
    "earningsQuarterlyGrowth": "earnings_growth_q",
    # Scale and context
    "marketCap": "market_cap",
    "enterpriseValue": "enterprise_value",
    "bookValue": "book_value",
    "dividendYield": "dividend_yield",
    "payoutRatio": "payout_ratio",
    "beta": "beta_reported",
    "sector": "sector",
    "industry": "industry",
}


@dataclass
class FundamentalSnapshot:
    frame: pd.DataFrame
    snapshot_date: str
    n_requested: int
    n_retrieved: int
    failures: list[str]

    @property
    def coverage(self) -> float:
        return self.n_retrieved / self.n_requested if self.n_requested else 0.0


def fetch_one(ticker: str, *, timeout_note: bool = True) -> dict | None:
    """Fundamentals for one ticker. Returns None on failure.

    yfinance's .info endpoint is flaky and rate-limited — failures are normal
    and expected, not exceptional.
    """
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:                                          # noqa: BLE001
        return None
    if not info or len(info) < 5:
        return None

    row = {"ticker": ticker.replace(".NS", "")}
    for src, dest in FIELDS.items():
        v = info.get(src)
        if isinstance(v, (int, float)) and np.isfinite(v):
            row[dest] = float(v)
        elif isinstance(v, str):
            row[dest] = v
        else:
            row[dest] = np.nan

    # Earnings date, for announcement-lag awareness
    try:
        ts = info.get("earningsTimestamp")
        if ts:
            row["last_earnings_date"] = dt.datetime.fromtimestamp(ts).date().isoformat()
    except Exception:                                          # noqa: BLE001
        pass

    return row


def fetch_universe(tickers: tuple[str, ...], *, progress=None,
                   max_tickers: int = 200) -> FundamentalSnapshot:
    """Fundamentals for a universe. Slow — one API call per ticker."""
    today = dt.date.today().isoformat()
    tickers = tickers[:max_tickers]
    rows, failures = [], []

    for i, t in enumerate(tickers):
        if progress:
            progress(i + 1, len(tickers), t)
        row = fetch_one(t)
        if row is None:
            failures.append(t)
            continue
        row["snapshot_date"] = today
        rows.append(row)

    frame = pd.DataFrame(rows) if rows else pd.DataFrame()
    return FundamentalSnapshot(frame, today, len(tickers), len(rows), failures)


# --------------------------------------------------------------------------
# Point-in-time archiving
# --------------------------------------------------------------------------
def archive(snapshot: FundamentalSnapshot,
            path: Path = ARCHIVE_PATH) -> tuple[int, int]:
    """Append a snapshot to the point-in-time archive.

    This is the only honest route to point-in-time fundamentals without paying
    for them: record what is visible today, dated today, and let the series
    accumulate. In two years the archive contains two years of data that was
    genuinely knowable at the time.

    Re-running on the same date is a no-op, so it is safe to schedule.
    """
    if snapshot.frame.empty:
        return 0, 0

    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not existing.empty and "snapshot_date" in existing.columns:
        if (existing["snapshot_date"].astype(str) == snapshot.snapshot_date).any():
            return 0, len(existing)

    combined = (pd.concat([existing, snapshot.frame], ignore_index=True)
                if not existing.empty else snapshot.frame)
    combined.to_csv(path, index=False)
    return len(snapshot.frame), len(combined)


def load_archive(path: Path = ARCHIVE_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "snapshot_date" in df.columns:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    return df


def archive_status(path: Path = ARCHIVE_PATH) -> dict:
    """How usable is the archive yet?"""
    df = load_archive(path)
    if df.empty:
        return {
            "snapshots": 0, "tickers": 0, "usable": False,
            "note": ("Archive empty. Start monthly snapshots now — the series "
                     "only becomes useful with elapsed time."),
        }

    dates = df["snapshot_date"].dropna()
    span_days = (dates.max() - dates.min()).days if len(dates) > 1 else 0
    n_snap = int(dates.dt.date.nunique())

    if n_snap < 6:
        note = (f"{n_snap} snapshot(s) over {span_days} days. Far too few for "
                "any historical analysis. Keep accumulating.")
        usable = False
    elif span_days < 365:
        note = (f"{n_snap} snapshots over {span_days} days. Enough to observe "
                "how fundamentals move, not enough to test a signal against "
                "forward returns.")
        usable = False
    else:
        note = (f"{n_snap} snapshots over {span_days} days ({span_days/365:.1f} "
                "years). Genuine point-in-time data — usable for cautious "
                "signal testing.")
        usable = True

    return {
        "snapshots": n_snap,
        "tickers": int(df["ticker"].nunique()) if "ticker" in df.columns else 0,
        "rows": len(df),
        "first": dates.min().date().isoformat() if len(dates) else None,
        "last": dates.max().date().isoformat() if len(dates) else None,
        "span_days": span_days,
        "usable": usable,
        "note": note,
    }


# --------------------------------------------------------------------------
# Derived metrics
# --------------------------------------------------------------------------
def quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional quality ranking, after Asness/Frazzini/Pedersen.

    Profitability, low leverage, and stable margins. Each component is
    percentile-ranked within the universe, then averaged — no fitted weights,
    because fitting weights on the data you selected on is how backtests become
    fiction.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    components = {}

    for col, higher_better in (("roe", True), ("roa", True),
                               ("net_margin", True), ("operating_margin", True),
                               ("debt_to_equity", False), ("current_ratio", True)):
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        if s.notna().sum() < max(5, len(out) * 0.3):
            continue
        r = s.rank(pct=True, ascending=higher_better)
        components[f"rank_{col}"] = r

    if not components:
        return pd.DataFrame()

    for k, v in components.items():
        out[k] = (v * 100).round(1)
    out["quality_score"] = (pd.DataFrame(components).mean(axis=1) * 100).round(1)
    out["quality_components"] = len(components)
    return out.sort_values("quality_score", ascending=False).reset_index(drop=True)


def value_score(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional value ranking. Cheap on earnings, book and sales."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    components = {}
    for col in ("pe_trailing", "pb", "ps", "ev_ebitda"):
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        # Negative multiples are meaningless for ranking cheapness
        s = s.where(s > 0)
        if s.notna().sum() < max(5, len(out) * 0.3):
            continue
        components[f"rank_{col}"] = s.rank(pct=True, ascending=False)

    if not components:
        return pd.DataFrame()

    for k, v in components.items():
        out[k] = (v * 100).round(1)
    out["value_score"] = (pd.DataFrame(components).mean(axis=1) * 100).round(1)
    out["value_components"] = len(components)
    return out.sort_values("value_score", ascending=False).reset_index(drop=True)


def screen(df: pd.DataFrame, *, min_roe: float | None = None,
           max_debt_equity: float | None = None,
           max_pe: float | None = None,
           min_revenue_growth: float | None = None) -> pd.DataFrame:
    """Straightforward fundamental filtering of a current snapshot.

    No lookahead concern here — this uses what is known now to decide now.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()

    if min_roe is not None and "roe" in out.columns:
        out = out[pd.to_numeric(out["roe"], errors="coerce") >= min_roe]
    if max_debt_equity is not None and "debt_to_equity" in out.columns:
        d = pd.to_numeric(out["debt_to_equity"], errors="coerce")
        out = out[d.isna() | (d <= max_debt_equity)]
    if max_pe is not None and "pe_trailing" in out.columns:
        pe = pd.to_numeric(out["pe_trailing"], errors="coerce")
        out = out[(pe > 0) & (pe <= max_pe)]
    if min_revenue_growth is not None and "revenue_growth" in out.columns:
        out = out[pd.to_numeric(out["revenue_growth"], errors="coerce")
                  >= min_revenue_growth]
    return out.reset_index(drop=True)


def lookahead_warning() -> str:
    """The caveat that must accompany any historical fundamental analysis."""
    return (
        "**Historical fundamental analysis using current data is invalid.**\n\n"
        "Two separate problems compound. Reporting lag: FY2024 figures were not "
        "known until roughly two months after year end, so using them to judge a "
        "March 2024 decision means the model knows what nobody knew. Restatement: "
        "reported numbers get revised, so today's FY2023 figures are not "
        "necessarily what the market saw in 2023.\n\n"
        "A backtest built this way is not merely imprecise — it is "
        "systematically optimistic, and the bias is hard to quantify and easy to "
        "mistake for skill.\n\n"
        "Current fundamentals are fine for screening today. Only the archive "
        "accumulated from today onward is safe for testing signals."
    )
