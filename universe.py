"""Dynamic universe construction.

Universes are fetched live rather than hardcoded, so index rebalances and
changing market conditions are picked up automatically.

Sources, in order of reliability:
  1. NSE index constituent CSVs  — official, updated on rebalance
  2. NSE live JSON endpoints     — most active / gainers / losers, intraday
  3. Bundled fallback snapshot   — used when NSE is unreachable

NSE blocks unprimed requests. Every call goes through a session that first
visits the homepage to collect cookies, then retries with browser-like headers.
This works but is inherently fragile: NSE changes its bot defences periodically.
If fetching starts failing, the fallback snapshot keeps the app usable.
"""

from __future__ import annotations

import datetime as dt
import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests

import resilience as resil
import streamlit as st

import fallback_universe

NSE_HOME = "https://www.nseindia.com"
ARCHIVE = "https://nsearchives.nseindia.com/content/indices/{file}"
API_INDEX = "https://www.nseindia.com/api/equity-stockIndices?index={index}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}

# Index name -> NSE archive CSV filename
INDEX_FILES = {
    "Nifty 50": "ind_nifty50list.csv",
    "Nifty Next 50": "ind_niftynext50list.csv",
    "Nifty 100": "ind_nifty100list.csv",
    "Nifty 200": "ind_nifty200list.csv",
    "Nifty 500": "ind_nifty500list.csv",
    "Nifty Midcap 150": "ind_niftymidcap150list.csv",
    "Nifty Smallcap 250": "ind_niftysmallcap250list.csv",
    "Nifty Midcap 100": "ind_niftymidcap100list.csv",
    "Nifty Smallcap 100": "ind_niftysmallcap100list.csv",
    "Nifty Bank": "ind_niftybanklist.csv",
    "Nifty IT": "ind_niftyitlist.csv",
    "Nifty Pharma": "ind_niftypharmalist.csv",
    "Nifty Auto": "ind_niftyautolist.csv",
    "Nifty FMCG": "ind_niftyfmcglist.csv",
    "Nifty Metal": "ind_niftymetallist.csv",
    "Nifty Energy": "ind_niftyenergylist.csv",
    "Nifty Realty": "ind_niftyrealtylist.csv",
    "Nifty Financial Services": "ind_niftyfinancelist.csv",
    "Nifty Healthcare": "ind_niftyhealthcarelist.csv",
    "Nifty Consumer Durables": "ind_niftyconsumerdurableslist.csv",
    "Nifty Oil & Gas": "ind_niftyoilgaslist.csv",
    "Nifty PSU Bank": "ind_niftypsubanklist.csv",
    "Nifty Infrastructure": "ind_niftyinfralist.csv",
    "Nifty Commodities": "ind_niftycommoditieslist.csv",
    "Nifty India Digital": "ind_niftyindiadigitallist.csv",
    "Nifty India Manufacturing": "ind_niftyindiamanufacturinglist.csv",
}

# Live screens via NSE JSON — these change intraday
LIVE_SCREENS = {
    "Most active (by value)": "NIFTY 500",
    "Top gainers (Nifty 500)": "NIFTY 500",
    "Top losers (Nifty 500)": "NIFTY 500",
    "Near 52-week high": "NIFTY 500",
}


@dataclass
class UniverseResult:
    """A fetched universe plus provenance, so the UI can be honest about it."""
    tickers: tuple[str, ...]
    source: str
    fetched_at: dt.datetime
    is_live: bool
    note: str = ""
    meta: pd.DataFrame | None = field(default=None, repr=False)


def _session() -> requests.Session:
    """Cookie-primed session. NSE 403s anything that skips the homepage."""
    s = requests.Session()
    s.headers.update(HEADERS)
    try:
        s.get(NSE_HOME, timeout=8)
    except requests.RequestException:
        pass
    return s


def _to_ns(symbols) -> tuple[str, ...]:
    """Normalise NSE symbols to Yahoo tickers, de-duplicated, order preserved."""
    seen, out = set(), []
    for raw in symbols:
        sym = str(raw).strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym if sym.endswith((".NS", ".BO")) else f"{sym}.NS")
    return tuple(out)


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fetch_index_constituents(index_name: str) -> UniverseResult:
    """Pull an index's live constituent list from the NSE archive CSV.

    Cached 12 hours — index composition changes on rebalance, not intraday.
    """
    now = dt.datetime.now()
    filename = INDEX_FILES.get(index_name)
    if not filename:
        return _fallback(index_name, now, "Unknown index name.")

    def _get():
        sess = _session()
        resp = sess.get(ARCHIVE.format(file=filename), timeout=12)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))

    # Retry transient failures (rate limits, resets) before falling back to
    # the cached universe. This is what actually protects the button that
    # calls this function directly — a bare request here previously meant
    # one flaky NSE response fell straight through to the stale fallback.
    try:
        df, stats = resil.call_with_retry(_get, max_attempts=2, base_delay=3.0)
        if df is None:
            raise RuntimeError(stats.last_error or "fetch failed")
    except Exception as exc:                                   # noqa: BLE001
        return _fallback(index_name, now, f"NSE fetch failed ({type(exc).__name__}).")

    col = next((c for c in df.columns if c.strip().lower() == "symbol"), None)
    if col is None:
        return _fallback(index_name, now, "CSV schema changed — no Symbol column.")

    if "Series" in df.columns:
        df = df[df["Series"].astype(str).str.strip().eq("EQ")]

    tickers = _to_ns(df[col].tolist())
    if not tickers:
        return _fallback(index_name, now, "CSV parsed but contained no symbols.")

    return UniverseResult(
        tickers=tickers,
        source=f"NSE archive · {index_name}",
        fetched_at=now,
        is_live=True,
        note=f"{len(tickers)} constituents",
        meta=df,
    )


@st.cache_data(ttl=60 * 15, show_spinner=False)
def fetch_live_screen(screen: str, top_n: int = 50) -> UniverseResult:
    """Build a universe from live market action rather than index membership.

    Cached only 15 minutes because these genuinely move intraday.
    """
    now = dt.datetime.now()
    index = LIVE_SCREENS.get(screen, "NIFTY 500")

    try:
        sess = _session()
        url = API_INDEX.format(index=index.replace(" ", "%20"))
        resp = sess.get(url, timeout=12, headers={**HEADERS, "Accept": "application/json"})
        resp.raise_for_status()
        payload = resp.json()
        rows = payload.get("data", [])
    except (requests.RequestException, ValueError) as exc:
        return _fallback(screen, now, f"NSE live API failed ({type(exc).__name__}).")

    df = pd.DataFrame(rows)
    if df.empty or "symbol" not in df.columns:
        return _fallback(screen, now, "Live API returned no usable rows.")

    df = df[df["symbol"].astype(str).str.upper() != index.upper()]

    for c in ("pChange", "totalTradedValue", "lastPrice", "yearHigh"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if screen.startswith("Most active"):
        if "totalTradedValue" not in df.columns:
            return _fallback(screen, now, "No turnover field in response.")
        df = df.sort_values("totalTradedValue", ascending=False)
    elif screen.startswith("Top gainers"):
        df = df.sort_values("pChange", ascending=False)
    elif screen.startswith("Top losers"):
        df = df.sort_values("pChange", ascending=True)
    elif screen.startswith("Near 52"):
        if {"lastPrice", "yearHigh"}.issubset(df.columns):
            df["_prox"] = df["lastPrice"] / df["yearHigh"].replace(0, pd.NA)
            df = df[df["_prox"].notna()].sort_values("_prox", ascending=False)
        else:
            return _fallback(screen, now, "No 52-week high field in response.")

    df = df.head(top_n)
    tickers = _to_ns(df["symbol"].tolist())
    if not tickers:
        return _fallback(screen, now, "Screen produced no symbols.")

    return UniverseResult(
        tickers=tickers,
        source=f"NSE live · {screen}",
        fetched_at=now,
        is_live=True,
        note=f"top {len(tickers)} from {index}, refreshes every 15 min",
        meta=df,
    )


def _fallback(label: str, now: dt.datetime, reason: str) -> UniverseResult:
    """Bundled snapshot so the app degrades instead of dying."""
    tickers = fallback_universe.get(label)
    return UniverseResult(
        tickers=tickers,
        source=f"Fallback snapshot ({fallback_universe.SNAPSHOT_DATE})",
        fetched_at=now,
        is_live=False,
        note=f"{reason} Using bundled list — may be stale.",
    )


def trim_universe(
    tickers: tuple[str, ...],
    max_n: int,
    *,
    method: str = "liquidity",
    frames: dict[str, pd.DataFrame] | None = None,
    seed: int = 0,
) -> tuple[tuple[str, ...], str]:
    """Reduce a universe to `max_n` WITHOUT introducing selection bias.

    Why this exists
    ---------------
    The original code did `tickers[:max_n]`. NSE returns constituent lists
    alphabetically by company name, so truncating positionally kept roughly
    A-G and silently discarded H-Z. Every pick the system produced came from
    the early alphabet. That is a pure artefact with no economic meaning.

    Methods
    -------
    liquidity : keep the most-traded names. Defensible — these are the ones
                you can actually trade without moving the price. Needs price
                data; falls back to random if unavailable.
    random    : unbiased sample. Statistically clean, but you may end up
                holding illiquid names.
    none      : no truncation. Correct when you can afford the fetch time.

    Never positional. There is no ordering of an index constituent list that
    makes "the first 150" a meaningful selection.
    """
    if max_n <= 0 or len(tickers) <= max_n:
        return tickers, "no truncation needed"

    if method == "none":
        return tickers, "no truncation"

    if method == "liquidity" and frames:
        scored = []
        for t in tickers:
            df = frames.get(t)
            if df is None or df.empty or len(df) < 20:
                continue
            try:
                turn = float((df["Close"] * df["Volume"]).tail(20).mean())
            except Exception:                                  # noqa: BLE001
                continue
            if turn > 0:
                scored.append((t, turn))
        if len(scored) >= max_n:
            scored.sort(key=lambda x: x[1], reverse=True)
            return tuple(t for t, _ in scored[:max_n]), \
                f"top {max_n} by 20-day traded value"

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tickers), size=max_n, replace=False)
    return tuple(tickers[i] for i in sorted(idx)), \
        f"random sample of {max_n} (seed {seed})"


def diff_universes(old: tuple[str, ...], new: tuple[str, ...]) -> dict[str, list[str]]:
    """What entered and left since the last fetch."""
    o, n = set(old), set(new)
    return {
        "added": sorted(s.replace(".NS", "") for s in n - o),
        "removed": sorted(s.replace(".NS", "") for s in o - n),
    }


def apply_liquidity_filter(
    result: UniverseResult, min_turnover_cr: float, price_range: tuple[float, float]
) -> tuple[str, ...]:
    """Trim a universe using metadata already returned by the live API.

    Only applies when NSE gave us price/turnover fields; otherwise the screener
    applies its own liquidity floor from yfinance data downstream.
    """
    df = result.meta
    if df is None or df.empty or "symbol" not in df.columns:
        return result.tickers

    work = df.copy()
    if "lastPrice" in work.columns:
        work["lastPrice"] = pd.to_numeric(work["lastPrice"], errors="coerce")
        lo, hi = price_range
        work = work[work["lastPrice"].between(lo, hi)]
    if "totalTradedValue" in work.columns and min_turnover_cr > 0:
        work["totalTradedValue"] = pd.to_numeric(work["totalTradedValue"], errors="coerce")
        # NSE reports turnover in lakhs on this endpoint
        work = work[work["totalTradedValue"] / 100 >= min_turnover_cr]

    return _to_ns(work["symbol"].tolist()) or result.tickers
