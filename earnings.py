"""Earnings dates — the check the News tab cannot make.

The gap this closes
-------------------
The News tab detects earnings-*related news*: a headline mentioning results,
margins or a board meeting. That is a hint, not an answer. What matters is
whether a company **reports** inside your holding window, and a headline about
last quarter tells you nothing about the next one.

Every pick in a recent bucket carried an earnings flag, and the system still
presented all of them. A flag that never stops anything is decoration.

Why it matters at a 30-day horizon
----------------------------------
A results announcement is an information event that dwarfs the signal. Momentum
produces a long-only edge of roughly 2.16% over 30 days. A single earnings
surprise routinely moves an Indian mid-cap 8-15% in a session.

Holding through results is not a momentum trade with extra variance. It is an
earnings bet that happens to have been selected by a momentum screen — and the
momentum edge is far too small to survive that noise. The position outcome will
be decided by something the model has no view on.

Data source and its limits
--------------------------
yfinance exposes earnings dates via `Ticker.calendar` and `Ticker.earnings_dates`.
Coverage for Indian equities is decent but not complete, and dates are sometimes
estimated rather than confirmed. Three consequences, handled explicitly:

  * Missing data is reported as unknown, never as "safe". Absence of a date is
    not evidence of absence of an announcement.
  * Estimated dates get a wider buffer than confirmed ones.
  * A cache avoids hammering the API, since this runs across a whole bucket.

Indian reporting seasons cluster in mid-January, mid-April, mid-July and
mid-October. If a pick is made in one of those windows, treat unknown dates as
risky rather than clear.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_PATH = Path("earnings_cache.csv")
CACHE_TTL_DAYS = 3

# Buffer either side of the announcement. Two days before because positioning
# and leaks move price ahead of the print; three days after because the
# reaction frequently continues into subsequent sessions.
BUFFER_BEFORE = 2
BUFFER_AFTER = 3

# Indian results seasons. Roughly six weeks after each quarter end.
RESULT_SEASONS = [(1, 10, 2, 15), (4, 10, 5, 31), (7, 10, 8, 15), (10, 10, 11, 15)]


@dataclass
class EarningsCheck:
    ticker: str
    next_date: dt.date | None
    days_away: int | None
    in_window: bool
    confidence: str            # confirmed | estimated | unknown
    in_results_season: bool
    verdict: str               # clear | avoid | unknown
    reason: str


@dataclass
class ScreenResult:
    clear: pd.DataFrame
    excluded: pd.DataFrame
    unknown: pd.DataFrame
    checked: int
    notes: list[str] = field(default_factory=list)


def in_results_season(date: dt.date | None = None) -> bool:
    """Is this date inside a typical Indian reporting cluster?"""
    d = date or dt.date.today()
    for m1, d1, m2, d2 in RESULT_SEASONS:
        start = dt.date(d.year, m1, d1)
        end = dt.date(d.year, m2, d2)
        if start <= d <= end:
            return True
    return False


def _load_cache() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        return pd.DataFrame(columns=["ticker", "next_date", "confidence", "fetched"])
    try:
        df = pd.read_csv(CACHE_PATH)
        df["fetched"] = pd.to_datetime(df["fetched"], errors="coerce")
        cutoff = pd.Timestamp(dt.date.today() - dt.timedelta(days=CACHE_TTL_DAYS))
        return df[df["fetched"] >= cutoff]
    except Exception:                                          # noqa: BLE001
        return pd.DataFrame(columns=["ticker", "next_date", "confidence", "fetched"])


def _save_cache(rows: list[dict]) -> None:
    if not rows:
        return
    existing = _load_cache()
    new = pd.DataFrame(rows)
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["ticker"], keep="last")
    try:
        combined.to_csv(CACHE_PATH, index=False)
    except Exception:                                          # noqa: BLE001
        pass


def fetch_earnings_date(ticker: str, *, use_cache: bool = True
                        ) -> tuple[dt.date | None, str]:
    """Next earnings date for one ticker. Returns (date, confidence)."""
    sym = ticker if ticker.endswith(".NS") else f"{ticker}.NS"
    key = sym.replace(".NS", "")

    if use_cache:
        cache = _load_cache()
        hit = cache[cache["ticker"] == key]
        if not hit.empty:
            row = hit.iloc[-1]
            d = pd.to_datetime(row["next_date"], errors="coerce")
            return (d.date() if pd.notna(d) else None,
                    str(row.get("confidence", "unknown")))

    import yfinance as yf
    today = dt.date.today()

    # Preferred: the earnings_dates table, which distinguishes future from past
    try:
        t = yf.Ticker(sym)
        ed = t.earnings_dates
        if ed is not None and not ed.empty:
            idx = pd.DatetimeIndex(ed.index).tz_localize(None)
            future = idx[idx >= pd.Timestamp(today)]
            if len(future):
                return future[0].date(), "confirmed"
    except Exception:                                          # noqa: BLE001
        pass

    # Fallback: the calendar field
    try:
        cal = yf.Ticker(sym).calendar
        val = None
        if isinstance(cal, dict):
            val = cal.get("Earnings Date")
        elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
            val = cal.loc["Earnings Date"].iloc[0]
        if isinstance(val, list) and val:
            val = val[0]
        if val is not None:
            d = pd.to_datetime(val, errors="coerce")
            if pd.notna(d) and d.date() >= today:
                return d.date(), "estimated"
    except Exception:                                          # noqa: BLE001
        pass

    return None, "unknown"


def check(ticker: str, *, horizon_days: int = 30,
          as_of: dt.date | None = None, use_cache: bool = True) -> EarningsCheck:
    """Does this stock report inside the holding window?"""
    today = as_of or dt.date.today()
    d, conf = fetch_earnings_date(ticker, use_cache=use_cache)
    season = in_results_season(today)
    sym = ticker.replace(".NS", "")

    if d is None:
        # Unknown is not the same as clear, and during results season the
        # distinction matters a great deal.
        if season:
            return EarningsCheck(
                sym, None, None, False, "unknown", True, "unknown",
                "No earnings date available, and today falls inside a typical "
                "Indian results season. Absence of data is not evidence of "
                "absence — verify on the exchange website before entering.")
        return EarningsCheck(
            sym, None, None, False, "unknown", False, "unknown",
            "No earnings date available. Outside the main reporting cluster, so "
            "lower risk, but unverified.")

    days = (d - today).days
    # Estimated dates get a wider buffer, since they can move by several days
    extra = 3 if conf == "estimated" else 0
    window_end = horizon_days + BUFFER_AFTER
    inside = -BUFFER_BEFORE - extra <= days <= window_end + extra

    if inside:
        return EarningsCheck(
            sym, d, days, True, conf, season, "avoid",
            f"Reports {d} — {days} days away, inside a {horizon_days}-day hold. "
            f"A results surprise routinely moves a stock 8-15% in a session; the "
            f"momentum edge over the whole period is about 2%. The outcome would "
            f"be decided by the announcement, not the signal."
            + (" Date is estimated, so treat the window as wider."
               if conf == "estimated" else ""))

    return EarningsCheck(
        sym, d, days, False, conf, season, "clear",
        f"Reports {d}, {days} days away — outside the holding window."
        + (" Estimated date; re-check nearer the time."
           if conf == "estimated" else ""))


def screen_bucket(picks: pd.DataFrame, *, horizon_days: int = 30,
                  exclude_unknown: bool = False,
                  progress=None) -> ScreenResult:
    """Split a bucket by earnings risk.

    exclude_unknown: treat missing dates as unsafe. Sensible during results
    season, over-cautious outside it.
    """
    if picks is None or picks.empty:
        return ScreenResult(picks, pd.DataFrame(), pd.DataFrame(), 0)

    rows = []
    for i, (_, r) in enumerate(picks.iterrows()):
        tkr = str(r.get("Ticker", ""))
        if progress:
            progress(i + 1, len(picks), tkr)
        c = check(tkr, horizon_days=horizon_days)
        d = r.to_dict()
        d.update({
            "earnings_date": c.next_date.isoformat() if c.next_date else None,
            "days_to_earnings": c.days_away,
            "earnings_confidence": c.confidence,
            "earnings_verdict": c.verdict,
            "earnings_reason": c.reason,
        })
        rows.append(d)

    df = pd.DataFrame(rows)
    clear = df[df["earnings_verdict"] == "clear"].reset_index(drop=True)
    avoid = df[df["earnings_verdict"] == "avoid"].reset_index(drop=True)
    unknown = df[df["earnings_verdict"] == "unknown"].reset_index(drop=True)

    if exclude_unknown and not unknown.empty:
        avoid = pd.concat([avoid, unknown], ignore_index=True)
        unknown = unknown.iloc[0:0]

    notes = []
    season = in_results_season()
    if season:
        notes.append(
            "Today falls inside a typical Indian results season (mid-Jan, "
            "mid-Apr, mid-Jul, mid-Oct). Expect more exclusions, and treat "
            "unknown dates as risky rather than clear.")
    if not avoid.empty:
        notes.append(
            f"{len(avoid)} pick(s) report inside the {horizon_days}-day window. "
            "Holding through results converts a momentum trade into an earnings "
            "bet the model has no view on.")
    if not unknown.empty:
        notes.append(
            f"{len(unknown)} pick(s) have no retrievable date. yfinance coverage "
            "of Indian earnings is incomplete — verify on nseindia.com or the "
            "company's investor relations page before entering.")
    if clear.empty and not df.empty:
        notes.append(
            "No picks are clear of earnings. During results season that is "
            "normal, and the honest response is usually to wait rather than to "
            "relax the check.")

    return ScreenResult(clear, avoid, unknown, len(df), notes)


def next_season_gap() -> dict:
    """When the next clear window opens — useful for planning entries."""
    today = dt.date.today()
    seasons = []
    for m1, d1, m2, d2 in RESULT_SEASONS:
        for yr in (today.year, today.year + 1):
            s, e = dt.date(yr, m1, d1), dt.date(yr, m2, d2)
            if e >= today:
                seasons.append((s, e))
    seasons.sort()

    if not seasons:
        return {"in_season": False}
    s, e = seasons[0]

    if s <= today <= e:
        nxt = seasons[1] if len(seasons) > 1 else None
        return {
            "in_season": True,
            "season_ends": e.isoformat(),
            "days_until_clear": (e - today).days,
            "next_season_starts": nxt[0].isoformat() if nxt else None,
            "note": (f"Results season until {e}. Entries made now are likely to "
                     "span an announcement for at least some picks."),
        }
    return {
        "in_season": False,
        "next_season_starts": s.isoformat(),
        "days_until_season": (s - today).days,
        "note": (f"Outside results season. Next cluster begins {s} — a 30-day "
                 f"hold entered within {max(0, (s - today).days)} days would "
                 "still run into it."),
    }
