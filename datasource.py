"""Data source abstraction — removing the single point of failure.

The problem
-----------
Thirteen modules import yfinance directly. It is an unofficial scraper of a
service Yahoo does not support for this purpose, it breaks whenever Yahoo
changes its endpoints, and it rate-limits without warning. When it fails,
everything fails: the screener, the pipeline, the health check, the decay
monitor, the forward log evaluation.

That is a single vendor dependency on an unsupported source, flagged in the
compliance audit as unmitigated. This closes it.

The solution uses data already in hand
--------------------------------------
The fallback is NSE bhavcopy. It is the exchange's own published data, free,
archived to 1994, and the module to fetch it already exists. Reconstructing
OHLCV series from cached bhavcopies gives a genuinely independent source — not
another scraper of the same upstream.

Order of preference:

  1. **yfinance** — fast, one call for many tickers, adjusted for splits and
     dividends.
  2. **Bhavcopy reconstruction** — exchange-official, unadjusted. Slower and
     needs a populated cache, but survives any yfinance outage.
  3. **Explicit failure** — better than silently returning partial data.

An important caveat on the fallback
-----------------------------------
Bhavcopy prices are NOT adjusted for splits or dividends. A 1:5 split appears
as an 80% single-day fall. Momentum computed on unadjusted prices across a
split is badly wrong.

The reconstruction therefore applies the data-quality audit and flags any
series with an implausible move rather than passing it through. Fallback data
is usable for continuity, not as a silent equal to the primary source — and
the source used is always recorded.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Moves beyond this in a single session are almost certainly unadjusted
# corporate actions rather than real price changes.
SPLIT_SUSPECT_THRESHOLD = 0.35


@dataclass
class FetchResult:
    frames: dict[str, pd.DataFrame]
    source: str
    requested: int
    retrieved: int
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fallback_used: bool = False

    @property
    def coverage(self) -> float:
        return self.retrieved / self.requested if self.requested else 0.0

    def summary(self) -> str:
        s = (f"{self.retrieved}/{self.requested} tickers ({self.coverage:.0%}) "
             f"from {self.source}")
        if self.fallback_used:
            s += " [FALLBACK — primary source unavailable]"
        return s


# --------------------------------------------------------------------------
# Primary: yfinance
# --------------------------------------------------------------------------
def _fetch_yfinance(tickers: tuple[str, ...], period: str,
                    min_bars: int) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    raw = yf.download(list(tickers), period=period, interval="1d",
                      auto_adjust=True, progress=False,
                      group_by="ticker", threads=True)
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out

    for t in tickers:
        try:
            df = raw[t].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
        except (KeyError, IndexError):
            continue
        df = df.dropna(how="all")
        if df.empty or len(df) < min_bars:
            continue
        df.columns = [str(c).title() for c in df.columns]
        if {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
            out[t] = df
    return out


# --------------------------------------------------------------------------
# Fallback: reconstruct from cached bhavcopies
# --------------------------------------------------------------------------
def _fetch_bhavcopy(tickers: tuple[str, ...], period: str,
                    min_bars: int) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Rebuild OHLCV from the local bhavcopy cache.

    Exchange-official and completely independent of Yahoo. The cost is that
    prices are unadjusted, so any series spanning a corporate action is flagged
    rather than returned silently.
    """
    import bhavcopy as bc

    warnings: list[str] = []
    cache_dir = bc.CACHE_DIR
    if not cache_dir.exists():
        return {}, ["Bhavcopy cache is empty — download history first via the "
                    "app's NSE bhavcopy section or the daily workflow."]

    years = {"1y": 1, "2y": 2, "5y": 5, "10y": 10}.get(period, 2)
    cutoff = dt.date.today() - dt.timedelta(days=int(365.25 * years))

    files = sorted(cache_dir.rglob("bhav_*.parquet"))
    if not files:
        return {}, ["No cached bhavcopy files found."]

    wanted = {t.replace(".NS", "").upper() for t in tickers}
    rows = []
    for f in files:
        try:
            d = dt.datetime.strptime(f.stem.replace("bhav_", ""), "%Y%m%d").date()
        except ValueError:
            continue
        if d < cutoff:
            continue
        try:
            day = pd.read_parquet(f)
        except Exception:                                      # noqa: BLE001
            continue
        if day.empty or "symbol" not in day.columns:
            continue
        sub = day[day["symbol"].isin(wanted)].copy()
        if sub.empty:
            continue
        sub["date"] = pd.Timestamp(d)
        rows.append(sub)

    if not rows:
        return {}, ["Cached bhavcopies contain none of the requested tickers."]

    allday = pd.concat(rows, ignore_index=True)
    out: dict[str, pd.DataFrame] = {}
    suspect = []

    for sym, g in allday.groupby("symbol"):
        g = g.sort_values("date").set_index("date")
        cols = {"open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume"}
        have = {k: v for k, v in cols.items() if k in g.columns}
        if len(have) < 5:
            continue
        df = g[list(have)].rename(columns=have)
        if len(df) < min_bars:
            continue

        # Unadjusted prices: flag anything that looks like a corporate action
        ret = df["Close"].pct_change().abs()
        if (ret > SPLIT_SUSPECT_THRESHOLD).any():
            suspect.append(sym)
            continue

        out[f"{sym}.NS"] = df

    if suspect:
        warnings.append(
            f"{len(suspect)} ticker(s) excluded from fallback data — a move over "
            f"{SPLIT_SUSPECT_THRESHOLD:.0%} suggests an unadjusted corporate "
            f"action: {', '.join(suspect[:5])}"
            + (" …" if len(suspect) > 5 else "")
        )
    warnings.append(
        "Bhavcopy prices are NOT split or dividend adjusted. Suitable for "
        "continuity during a yfinance outage; verify before acting on it."
    )
    return out, warnings


# --------------------------------------------------------------------------
# Public interface
# --------------------------------------------------------------------------
def fetch(tickers: tuple[str, ...], *, period: str = "2y",
          min_bars: int = 60, allow_fallback: bool = True,
          min_coverage: float = 0.30) -> FetchResult:
    """Fetch OHLCV, falling back to bhavcopy if the primary source fails.

    Args:
        min_coverage: if yfinance returns less than this fraction, treat it as
                      a failure and try the fallback. Partial success is often
                      worse than none — a bucket built on 20% of the universe
                      is confidently wrong.
    """
    n = len(tickers)
    if n == 0:
        return FetchResult({}, "none", 0, 0)

    # --- Primary ---
    try:
        frames = _fetch_yfinance(tickers, period, min_bars)
    except Exception as exc:                                   # noqa: BLE001
        frames = {}
        primary_error = f"{type(exc).__name__}: {exc}"
    else:
        primary_error = None

    coverage = len(frames) / n
    if coverage >= min_coverage:
        return FetchResult(frames, "yfinance", n, len(frames),
                           failures=[t for t in tickers if t not in frames])

    # --- Fallback ---
    if not allow_fallback:
        return FetchResult(
            frames, "yfinance (degraded)", n, len(frames),
            failures=[t for t in tickers if t not in frames],
            warnings=[f"Coverage {coverage:.0%} below the {min_coverage:.0%} "
                      "threshold and fallback is disabled."
                      + (f" Primary error: {primary_error}" if primary_error else "")],
        )

    fb_frames, fb_warnings = _fetch_bhavcopy(tickers, period, min_bars)
    if len(fb_frames) > len(frames):
        return FetchResult(
            fb_frames, "bhavcopy (fallback)", n, len(fb_frames),
            failures=[t for t in tickers if t not in fb_frames],
            warnings=([f"yfinance returned {coverage:.0%} coverage"
                       + (f" ({primary_error})" if primary_error else "")]
                      + fb_warnings),
            fallback_used=True,
        )

    return FetchResult(
        frames, "yfinance (degraded)", n, len(frames),
        failures=[t for t in tickers if t not in frames],
        warnings=([f"yfinance coverage {coverage:.0%}; bhavcopy fallback "
                   f"returned only {len(fb_frames)}"] + fb_warnings),
    )


def source_health() -> dict:
    """Is each source currently usable? Useful before a long run."""
    out = {}

    try:
        import yfinance as yf
        test = yf.download("RELIANCE.NS", period="5d", interval="1d",
                           auto_adjust=True, progress=False)
        ok = test is not None and not test.empty
        out["yfinance"] = {
            "available": bool(ok),
            "bars_returned": len(test) if ok else 0,
            "note": "OK" if ok else "Returned no data — rate limit or API change.",
        }
    except Exception as exc:                                   # noqa: BLE001
        out["yfinance"] = {"available": False,
                           "note": f"{type(exc).__name__}: {exc}"}

    try:
        import bhavcopy as bc
        stats = bc.cache_stats()
        out["bhavcopy_cache"] = {
            "available": stats["days"] > 0,
            "days_cached": stats["days"],
            "earliest": stats.get("earliest"),
            "latest": stats.get("latest"),
            "note": ("Usable as fallback." if stats["days"] > 60 else
                     f"Only {stats['days']} days cached — accumulate more before "
                     "relying on this as a fallback."),
        }
    except Exception as exc:                                   # noqa: BLE001
        out["bhavcopy_cache"] = {"available": False,
                                 "note": f"{type(exc).__name__}: {exc}"}

    primary = out.get("yfinance", {}).get("available", False)
    backup = out.get("bhavcopy_cache", {}).get("available", False)
    out["_verdict"] = (
        "Both sources available." if primary and backup else
        "Primary only — build the bhavcopy cache to remove the single point of failure."
        if primary else
        "PRIMARY DOWN, fallback available." if backup else
        "BOTH SOURCES UNAVAILABLE."
    )
    return out
