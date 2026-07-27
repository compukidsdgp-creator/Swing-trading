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

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

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
