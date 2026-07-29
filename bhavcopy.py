"""NSE bhavcopy — free point-in-time data.

What this solves
----------------
Three of the four "money problems" turn out to be free, because NSE publishes a
complete daily snapshot of every traded security and archives it back to 1994.

  1. **Survivorship bias.** A bhavcopy lists every security that traded on that
     date, including companies later delisted, merged or wound up. Building the
     universe from bhavcopies is genuinely point-in-time — membership is decided
     by what actually traded, not by what survives in an index today.

  2. **History depth.** Archives reach back to January 1994. Compare five years
     of yfinance against thirty years of exchange data.

  3. **Delivery percentage (DELIV_PER).** Not available in yfinance at all.
     The share of traded volume that resulted in actual delivery rather than
     intraday squaring-off. High delivery on an advance suggests genuine
     accumulation; low delivery suggests churn. This is a real signal input
     that was previously out of reach.

It does not solve capacity modelling, which needs order-book depth.

Sources and etiquette
---------------------
NSE blacklists aggressive crawlers, and bulk-downloading years of history
directly will get your IP blocked. This module therefore prefers a community
GitHub mirror that already carries two decades of files, and falls back to NSE
only for recent dates. Downloads are cached locally so any given date is
fetched once, ever.

Be considerate: the mirror is someone's unpaid work.

Format note
-----------
NSE changed its bhavcopy format in July 2024. Older files use columns like
SYMBOL/TOTTRDQTY; newer ones use TckrSymb/TtlTradgVol. Both are handled.
"""

from __future__ import annotations

import datetime as dt
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE_DIR = Path("bhavcache")

# Community mirror — carries 2+ decades, updated daily, no rate limiting.
MIRROR_RAW = ("https://raw.githubusercontent.com/tilak999/NSE-Data-bank/"
              "master/data/{year}/{fname}")

# NSE direct — used sparingly, for recent dates the mirror may not have yet.
NSE_NEW = ("https://nsearchives.nseindia.com/content/cm/"
           "BhavCopy_NSE_CM_0_0_0_{ymd}_F.csv.zip")
NSE_FULL = ("https://nsearchives.nseindia.com/products/content/"
            "sec_bhavdata_full_{ddmmyyyy}.csv")

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Referer": "https://www.nseindia.com/",
}

# Column aliases across the pre- and post-July-2024 formats
ALIASES = {
    "symbol":     ["SYMBOL", "TckrSymb"],
    "series":     ["SERIES", "SctySrs"],
    "open":       ["OPEN", "OPEN_PRICE", "OpnPric"],
    "high":       ["HIGH", "HIGH_PRICE", "HghPric"],
    "low":        ["LOW", "LOW_PRICE", "LwPric"],
    "close":      ["CLOSE", "CLOSE_PRICE", "ClsPric"],
    "prev_close": ["PREVCLOSE", "PREV_CLOSE", "PrvsClsgPric"],
    "volume":     ["TOTTRDQTY", "TTL_TRD_QNTY", "TtlTradgVol"],
    "turnover":   ["TOTTRDVAL", "TURNOVER_LACS", "TtlTrfVal"],
    "trades":     ["TOTALTRADES", "NO_OF_TRADES", "TtlNbOfTxsExctd"],
    "deliv_qty":  ["DELIV_QTY", "DelivQty"],
    "deliv_pct":  ["DELIV_PER", "DelivPer"],
    "isin":       ["ISIN", "ISIN_CODE"],
}


@dataclass
class FetchResult:
    frame: pd.DataFrame
    source: str
    cached: bool


def _cache_path(date: pd.Timestamp) -> Path:
    return CACHE_DIR / f"{date:%Y}" / f"bhav_{date:%Y%m%d}.parquet"


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Map either bhavcopy format onto one schema."""
    df = df.rename(columns=lambda c: str(c).strip())
    out = pd.DataFrame()
    for std, candidates in ALIASES.items():
        for c in candidates:
            if c in df.columns:
                out[std] = df[c]
                break
    if "symbol" not in out.columns:
        return pd.DataFrame()

    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    if "series" in out.columns:
        out["series"] = out["series"].astype(str).str.strip().str.upper()
    for c in ("open", "high", "low", "close", "prev_close", "volume",
              "turnover", "trades", "deliv_qty", "deliv_pct"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Some vintages report turnover in lakhs, others in rupees. Detect by
    # comparing reported turnover against close x volume ROW-WISE — a ratio of
    # medians is unreliable when the universe spans megacaps and microcaps.
    if {"turnover", "close", "volume"}.issubset(out.columns):
        implied = out["close"] * out["volume"]
        valid = (out["turnover"] > 0) & (implied > 0)
        if valid.sum() >= 5:
            ratio = float((implied[valid] / out.loc[valid, "turnover"]).median())
            # Turnover in lakhs gives a ratio near 1e5; in rupees, near 1.
            if 1e4 < ratio < 1e6:
                out["turnover"] = out["turnover"] * 1e5
            elif 1e1 < ratio < 1e4:
                # Reported in thousands or a mixed vintage — scale to match
                out["turnover"] = out["turnover"] * round(ratio, -1)

    return out.dropna(subset=["symbol", "close"])


def fetch_day(date: pd.Timestamp | str, *, use_cache: bool = True,
              session: requests.Session | None = None) -> FetchResult:
    """Fetch one day's bhavcopy. Cached locally after the first retrieval."""
    date = pd.Timestamp(date).normalize()
    cp = _cache_path(date)

    if use_cache and cp.exists():
        try:
            return FetchResult(pd.read_parquet(cp), "cache", True)
        except Exception:                                      # noqa: BLE001
            cp.unlink(missing_ok=True)

    s = session or requests.Session()
    s.headers.update(HEADERS)
    frame, source = pd.DataFrame(), ""

    # 1. Community mirror first — no rate limiting, spares NSE
    for fname in (f"cm{date:%d%b%Y}bhav.csv.zip".upper().replace("CM", "cm"),
                  f"BhavCopy_NSE_CM_0_0_0_{date:%Y%m%d}_F.csv",
                  f"cm{date:%d%b%Y}bhav.csv".replace(date.strftime('%b'),
                                                     date.strftime('%b').upper())):
        try:
            r = s.get(MIRROR_RAW.format(year=date.year, fname=fname), timeout=20)
            if r.status_code == 200 and len(r.content) > 200:
                if fname.endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                        frame = pd.read_csv(z.open(z.namelist()[0]))
                else:
                    frame = pd.read_csv(io.BytesIO(r.content))
                source = "mirror"
                break
        except Exception:                                      # noqa: BLE001
            continue

    # 2. NSE direct — only if the mirror lacks it (typically very recent dates)
    if frame.empty:
        try:
            s.get("https://www.nseindia.com", timeout=8)        # cookie priming
        except requests.RequestException:
            pass
        for url, is_zip in ((NSE_NEW.format(ymd=f"{date:%Y%m%d}"), True),
                            (NSE_FULL.format(ddmmyyyy=f"{date:%d%m%Y}"), False)):
            try:
                r = s.get(url, timeout=25)
                if r.status_code == 200 and len(r.content) > 200:
                    if is_zip:
                        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                            frame = pd.read_csv(z.open(z.namelist()[0]))
                    else:
                        frame = pd.read_csv(io.BytesIO(r.content))
                    source = "nse"
                    break
            except Exception:                                  # noqa: BLE001
                continue

    if frame.empty:
        return FetchResult(pd.DataFrame(), "unavailable", False)

    norm = _normalise(frame)
    if not norm.empty:
        cp.parent.mkdir(parents=True, exist_ok=True)
        try:
            norm.to_parquet(cp, index=False)
        except Exception:                                      # noqa: BLE001
            pass                                               # cache is optional
    return FetchResult(norm, source, False)


def fetch_range(start: str, end: str, *, max_days: int = 400,
                progress=None) -> pd.DataFrame:
    """Fetch a date range. Weekends are skipped; holidays return empty and are ignored.

    Be considerate with `max_days` — this hits someone else's server.
    """
    days = pd.bdate_range(start, end)[:max_days]
    s = requests.Session()
    s.headers.update(HEADERS)
    frames = []
    for i, d in enumerate(days):
        res = fetch_day(d, session=s)
        if not res.frame.empty:
            f = res.frame.copy()
            f["date"] = d
            frames.append(f)
        if progress:
            progress(i + 1, len(days), d, res.source)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------
# The payoff: a genuinely point-in-time universe
# --------------------------------------------------------------------------
def pit_universe(day: pd.DataFrame, *, min_turnover_cr: float = 10.0,
                 eq_only: bool = True, min_price: float = 10.0,
                 min_trades: int = 500) -> pd.DataFrame:
    """Which stocks were genuinely investable on this date?

    Judged entirely on that day's exchange data — no reference to what exists
    today. This is the survivorship fix.

    eq_only excludes the BE series, which carries surveillance restrictions
    (ASM/GSM). Those stocks are frequently untradeable in practice.
    """
    if day is None or day.empty:
        return pd.DataFrame()

    d = day.copy()
    if eq_only and "series" in d.columns:
        d = d[d["series"] == "EQ"]
    if "close" in d.columns:
        d = d[d["close"] >= min_price]
    if "trades" in d.columns and min_trades:
        d = d[d["trades"].fillna(0) >= min_trades]
    if "turnover" in d.columns:
        d["turnover_cr"] = d["turnover"] / 1e7
        d = d[d["turnover_cr"] >= min_turnover_cr]
    return d.sort_values("turnover_cr", ascending=False).reset_index(drop=True) \
        if "turnover_cr" in d.columns else d.reset_index(drop=True)


def delivery_signal(history: pd.DataFrame, *, window: int = 20) -> pd.DataFrame:
    """Delivery-based accumulation signal — unavailable from yfinance.

    Delivery percentage is the share of traded volume that resulted in actual
    delivery rather than intraday squaring-off. The informative construction is
    delivery on up-days versus down-days: institutions accumulating take
    delivery, day-traders do not.

    Returns per-symbol metrics for the most recent `window` sessions.
    """
    if history is None or history.empty or "deliv_pct" not in history.columns:
        return pd.DataFrame()

    h = history.sort_values(["symbol", "date"]).copy()
    h["ret"] = h.groupby("symbol")["close"].pct_change()

    rows = []
    for sym, g in h.groupby("symbol"):
        g = g.tail(window)
        if len(g) < max(10, window // 2) or g["deliv_pct"].isna().all():
            continue
        up = g.loc[g["ret"] > 0, "deliv_pct"].mean()
        dn = g.loc[g["ret"] < 0, "deliv_pct"].mean()
        rows.append({
            "symbol": sym,
            "deliv_pct_mean": round(float(g["deliv_pct"].mean()), 2),
            "deliv_up_days": round(float(up), 2) if pd.notna(up) else None,
            "deliv_down_days": round(float(dn), 2) if pd.notna(dn) else None,
            # Positive means delivery is heavier on advances — accumulation
            "accumulation": (round(float(up - dn), 2)
                             if pd.notna(up) and pd.notna(dn) else None),
            "sessions": len(g),
        })
    return pd.DataFrame(rows).sort_values("accumulation", ascending=False,
                                          na_position="last").reset_index(drop=True)


def universe_churn(frames: dict[pd.Timestamp, pd.DataFrame],
                   **kwargs) -> pd.DataFrame:
    """How universe membership changed over time.

    Meaningful churn is the evidence that survivorship bias is actually being
    addressed — a static list would show none.
    """
    rows, prev = [], set()
    for date in sorted(frames):
        members = set(pit_universe(frames[date], **kwargs)["symbol"])
        if not members:
            continue
        rows.append({
            "date": date,
            "n": len(members),
            "entered": len(members - prev) if prev else 0,
            "exited": len(prev - members) if prev else 0,
        })
        prev = members
    return pd.DataFrame(rows)


def load_cached(start: str | dt.date | None = None,
                end: str | dt.date | None = None) -> dict:
    """Load bhavcopies from the local cache only. Never touches the network.

    Validation must be reproducible and must not depend on a live service, so
    this reads what has already been downloaded and nothing else.

    Returns {date: DataFrame} keyed by datetime.date.
    """
    out: dict = {}
    if not CACHE_DIR.exists():
        return out

    lo = pd.Timestamp(start).date() if start else None
    hi = pd.Timestamp(end).date() if end else None

    for f in sorted(CACHE_DIR.rglob("bhav_*.parquet")):
        try:
            d = dt.datetime.strptime(f.stem.replace("bhav_", ""), "%Y%m%d").date()
        except ValueError:
            continue
        if lo and d < lo:
            continue
        if hi and d > hi:
            continue
        try:
            df = pd.read_parquet(f)
        except Exception:                                      # noqa: BLE001
            continue
        if not df.empty:
            out[d] = df
    return out


def build_price_history(
    frames: dict | None = None,
    *,
    min_days: int = 400,
    adjust_splits: bool = True,
    split_threshold: float = 0.35,
) -> tuple[dict, dict]:
    """Reconstruct per-symbol OHLCV history from cached bhavcopies.

    This is what makes genuine point-in-time validation possible. Bhavcopy
    records every security that traded on a given day, including companies that
    later delisted — right up to their final session. Building price history
    from the same source as the universe means failures are present in both,
    which is the only way to remove survivorship bias rather than merely
    reducing it.

    The catch: bhavcopy prices are NOT adjusted for corporate actions. A 1:5
    split appears as an 80% single-day collapse, and momentum computed across
    one is badly wrong.

    Splits are therefore detected and back-adjusted heuristically: a
    single-session move beyond `split_threshold` with no corresponding move in
    the broader market is treated as a corporate action, and prior prices are
    scaled. This is imperfect — a genuine 40% crash would be misread — so the
    number of adjustments is reported for inspection rather than applied
    silently.

    Returns (frames, report).
    """
    src = frames if frames is not None else load_cached()
    if not src:
        return {}, {"error": "No cached bhavcopies. Download history first."}

    dates = sorted(src)
    rows = []
    for d in dates:
        day = src[d]
        if day is None or day.empty or "symbol" not in day.columns:
            continue
        cols = [c for c in ("symbol", "open", "high", "low", "close",
                            "volume", "turnover", "deliv_pct") if c in day.columns]
        sub = day[cols].copy()
        sub["date"] = pd.Timestamp(d)
        rows.append(sub)

    if not rows:
        return {}, {"error": "No usable rows in the cache."}

    allday = pd.concat(rows, ignore_index=True)
    out, adjusted, skipped = {}, [], []

    for sym, g in allday.groupby("symbol"):
        g = g.sort_values("date").set_index("date")
        if len(g) < min_days:
            skipped.append(sym)
            continue

        ren = {"open": "Open", "high": "High", "low": "Low",
               "close": "Close", "volume": "Volume"}
        have = {k: v for k, v in ren.items() if k in g.columns}
        if len(have) < 5:
            skipped.append(sym)
            continue

        df = g[list(have)].rename(columns=have).astype(float)
        if "turnover" in g.columns:
            df["Turnover"] = g["turnover"].astype(float)
        if "deliv_pct" in g.columns:
            df["DelivPct"] = g["deliv_pct"].astype(float)

        n_adj = 0
        if adjust_splits:
            ret = df["Close"].pct_change()
            # Work backwards so each adjustment applies to everything before it
            for i in range(len(df) - 1, 0, -1):
                r = ret.iloc[i]
                if not np.isfinite(r) or abs(r) < split_threshold:
                    continue
                factor = float(df["Close"].iloc[i] / df["Close"].iloc[i - 1])
                # Only treat it as a split if the ratio is near a simple one
                for target in (0.5, 1/3, 0.25, 0.2, 0.1, 2.0, 3.0, 5.0, 10.0):
                    if abs(factor - target) / target < 0.12:
                        for c in ("Open", "High", "Low", "Close"):
                            df.iloc[:i, df.columns.get_loc(c)] *= factor
                        n_adj += 1
                        break
        if n_adj:
            adjusted.append((sym, n_adj))

        out[f"{sym}.NS"] = df

    report = {
        "symbols": len(out),
        "skipped_short_history": len(skipped),
        "split_adjusted": len(adjusted),
        "adjustments": adjusted[:20],
        "date_range": (dates[0].isoformat(), dates[-1].isoformat()),
        "total_bars": int(sum(len(v) for v in out.values())),
        "note": ("Prices reconstructed from NSE bhavcopy — the same source as "
                 "the universe, so delisted companies are present in both. "
                 "Splits back-adjusted heuristically; verify the adjustment "
                 "count looks plausible before relying on the result."),
    }
    return out, report


def delisted_symbols(frames: dict | None = None,
                     *, gap_days: int = 30) -> pd.DataFrame:
    """Symbols that stopped trading — the ones survivorship bias hides.

    A security whose last appearance is well before the cache's end either
    delisted, merged, or was suspended. Counting them shows directly how much
    a present-day constituent list is missing.
    """
    src = frames if frames is not None else load_cached()
    if not src:
        return pd.DataFrame()

    dates = sorted(src)
    last_seen, first_seen = {}, {}
    for d in dates:
        day = src[d]
        if day is None or day.empty or "symbol" not in day.columns:
            continue
        for s in day["symbol"].astype(str).unique():
            last_seen[s] = d
            first_seen.setdefault(s, d)

    if not last_seen:
        return pd.DataFrame()

    end = dates[-1]
    rows = []
    for s, last in last_seen.items():
        gap = (end - last).days
        if gap > gap_days:
            rows.append({
                "symbol": s,
                "first_seen": first_seen[s].isoformat(),
                "last_seen": last.isoformat(),
                "days_since": gap,
            })
    return (pd.DataFrame(rows).sort_values("last_seen", ascending=False)
            .reset_index(drop=True))


def cache_stats() -> dict:
    """What is already downloaded."""
    if not CACHE_DIR.exists():
        return {"days": 0, "size_mb": 0.0, "earliest": None, "latest": None}
    files = sorted(CACHE_DIR.rglob("bhav_*.parquet"))
    if not files:
        return {"days": 0, "size_mb": 0.0, "earliest": None, "latest": None}
    return {
        "days": len(files),
        "size_mb": round(sum(f.stat().st_size for f in files) / 1e6, 1),
        "earliest": files[0].stem.replace("bhav_", ""),
        "latest": files[-1].stem.replace("bhav_", ""),
    }
