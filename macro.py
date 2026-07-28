"""Indian macro data — regime context, not alpha.

Deliberate framing
------------------
This module does NOT add trading signals. It refines the regime gate, which
decides whether to trade at all.

That distinction matters. Adding macro variables as candidate signals would
mean more things tested, and more things tested means more chances for noise to
clear a significance bar — the exact multiple-testing problem that the t > 3
threshold exists to guard against. Twelve technical signals were already tested
and ten failed. The thirteenth through twentieth would not fare better.

Refining an existing gate is different. The regime gate currently uses price
alone: Nifty against its 200 DMA, plus breadth. Volatility and rate conditions
are documented drivers of when momentum fails, and adding them makes the gate
better at its one job without introducing a new selection problem.

Point-in-time discipline
------------------------
Macro releases carry a publication lag that is easy to get wrong and fatal when
you do. Indian CPI for a given month is released around the 12th of the
*following* month. Using April's CPI to inform an April decision means the model
knows something nobody knew for six weeks.

Every series here carries an explicit `release_lag_days`, and `as_of()` refuses
to return a value that had not been published by the requested date. Market data
(VIX, index levels) has zero lag; official statistics do not.

Sources — all free
------------------
  India VIX            NSE, via yfinance (^INDIAVIX)
  Nifty / sector index NSE, via yfinance
  10-year G-Sec yield  proxied from an index or manually maintained
  Repo rate            RBI, changes infrequently; maintained as a small table
  CPI inflation        MoSPI, released ~12th of the following month
  USD/INR              yfinance
  Crude, gold          yfinance

Where an official API is unavailable or unstable, a small maintained table is
used rather than a fragile scraper. A scraper that breaks silently is worse than
a table that is obviously stale.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_PATH = Path("macro_cache.csv")

# yfinance tickers for market-observable series. These have no publication lag —
# the value is known the moment it prints.
MARKET_SERIES = {
    "india_vix": "^INDIAVIX",
    "nifty": "^NSEI",
    "nifty_bank": "^NSEBANK",
    "usdinr": "INR=X",
    "crude": "CL=F",
    "gold": "GC=F",
    "dxy": "DX-Y.NYB",
}

# Publication lag in days for official statistics. Using a figure before its
# release date is lookahead, and a particularly seductive kind because the data
# feels historical.
RELEASE_LAGS = {
    "cpi_inflation": 12,        # MoSPI, ~12th of the following month
    "wpi_inflation": 14,
    "iip": 42,                  # Index of Industrial Production, ~6 weeks
    "gdp_growth": 60,
    "repo_rate": 0,             # announced and effective immediately
    "gsec_10y": 0,              # market observable
}

# RBI repo rate history. Maintained by hand because it changes a handful of
# times a year and no free stable API exists. A wrong-but-obvious table beats a
# scraper that fails quietly.
#
# VERIFY BEFORE RELYING ON THIS — check rbi.org.in for changes after the last
# entry. An outdated final value will silently persist forever.
REPO_RATE_HISTORY = [
    ("2020-05-22", 4.00),
    ("2022-05-04", 4.40),
    ("2022-06-08", 4.90),
    ("2022-08-05", 5.40),
    ("2022-09-30", 5.90),
    ("2022-12-07", 6.25),
    ("2023-02-08", 6.50),
    ("2025-02-07", 6.25),
    ("2025-04-09", 6.00),
    ("2025-06-06", 5.50),
]


@dataclass
class MacroSnapshot:
    as_of: dt.date
    values: dict
    stale: list[str]
    unavailable: list[str]

    @property
    def coverage(self) -> float:
        total = len(self.values) + len(self.unavailable)
        return len(self.values) / total if total else 0.0


def fetch_market_series(period: str = "2y") -> dict[str, pd.DataFrame]:
    """Market-observable macro series. No publication lag."""
    import yfinance as yf

    out = {}
    for name, ticker in MARKET_SERIES.items():
        try:
            df = yf.download(ticker, period=period, interval="1d",
                             auto_adjust=True, progress=False)
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).title() for c in df.columns]
            if "Close" in df.columns:
                out[name] = df[["Close"]].dropna()
        except Exception:                                      # noqa: BLE001
            continue
    return out


def repo_rate_as_of(date: dt.date | str) -> float | None:
    """Repo rate in effect on a given date. Announced rates apply immediately."""
    d = pd.Timestamp(date).date()
    applicable = [(pd.Timestamp(eff).date(), rate)
                  for eff, rate in REPO_RATE_HISTORY
                  if pd.Timestamp(eff).date() <= d]
    return applicable[-1][1] if applicable else None


def as_of(series: pd.Series, date: dt.date | str, series_name: str) -> float | None:
    """Value that was PUBLISHED by `date`, honouring the release lag.

    This is the function that prevents the most common macro lookahead error.
    A CPI print for April exists in the data with an April timestamp, but was
    not published until mid-May. Asking for CPI "as of 1 May" must return
    March's figure, not April's.
    """
    if series is None or series.empty:
        return None

    d = pd.Timestamp(date)
    lag = RELEASE_LAGS.get(series_name, 0)
    cutoff = d - pd.Timedelta(days=lag)

    available = series[series.index <= cutoff]
    return float(available.iloc[-1]) if not available.empty else None


def snapshot(market: dict[str, pd.DataFrame] | None = None,
             date: dt.date | None = None) -> MacroSnapshot:
    """Macro state as of a date, using only what was published by then."""
    d = date or dt.date.today()
    market = market if market is not None else fetch_market_series()

    values, stale, unavailable = {}, [], []

    for name, df in market.items():
        if df is None or df.empty:
            unavailable.append(name)
            continue
        s = df["Close"]
        v = as_of(s, d, name)
        if v is None:
            unavailable.append(name)
            continue
        values[name] = round(v, 4)

        age = (pd.Timestamp(d) - s.index[-1]).days
        if age > 7:
            stale.append(f"{name} ({age}d old)")

    repo = repo_rate_as_of(d)
    if repo is not None:
        values["repo_rate"] = repo
        last_change = pd.Timestamp(REPO_RATE_HISTORY[-1][0]).date()
        if (d - last_change).days > 400:
            stale.append(f"repo_rate (table last updated {last_change} — "
                         "verify against rbi.org.in)")
    else:
        unavailable.append("repo_rate")

    for name in ("india_vix", "nifty"):
        if name in market and not market[name].empty:
            s = market[name]["Close"]
            if len(s) > 252:
                values[f"{name}_pctile_1y"] = round(
                    float((s.tail(252) <= s.iloc[-1]).mean()) * 100, 1)

    return MacroSnapshot(d, values, stale, unavailable)


# --------------------------------------------------------------------------
# Regime enrichment — the actual purpose
# --------------------------------------------------------------------------
@dataclass
class MacroRegime:
    volatility_state: str        # calm | normal | elevated | extreme
    rate_state: str              # easing | neutral | tightening | unknown
    risk_appetite: str           # risk_on | neutral | risk_off | unknown
    momentum_favourable: bool
    confidence: str              # high | medium | low
    reasons: list[str]
    detail: dict


def classify_macro(market: dict[str, pd.DataFrame],
                   date: dt.date | None = None) -> MacroRegime:
    """Macro conditions relevant to whether momentum is likely to work.

    Two documented relationships drive this:

      **Volatility.** Momentum crashes cluster in high-volatility states
      (Barroso & Santa-Clara; Daniel & Moskowitz). India VIX in its upper decile
      is a warning independent of where the index sits.

      **Rate direction.** Tightening cycles compress valuations and tend to
      produce choppier, less trending markets — conditions in which momentum
      underperforms.

    This does NOT predict returns. It conditions how much to trade.
    """
    d = date or dt.date.today()
    reasons, detail = [], {}

    # --- Volatility state ---
    vol_state = "unknown"
    if "india_vix" in market and not market["india_vix"].empty:
        vix = market["india_vix"]["Close"]
        cur = float(vix.iloc[-1])
        detail["india_vix"] = round(cur, 2)
        if len(vix) > 252:
            pct = float((vix.tail(252) <= cur).mean())
            detail["vix_percentile_1y"] = round(pct * 100, 1)
            if pct >= 0.90:
                vol_state = "extreme"
                reasons.append(f"India VIX at {cur:.1f} sits in the top decile of "
                               "its one-year range. Momentum crashes cluster here.")
            elif pct >= 0.75:
                vol_state = "elevated"
                reasons.append(f"India VIX at {cur:.1f} is in the upper quartile.")
            elif pct <= 0.25:
                vol_state = "calm"
                reasons.append(f"India VIX at {cur:.1f} is in the lower quartile — "
                               "historically favourable for trend persistence.")
            else:
                vol_state = "normal"
        else:
            vol_state = "normal"
            reasons.append("Insufficient VIX history for a percentile reading.")

    # --- Rate direction ---
    rate_state = "unknown"
    repo_now = repo_rate_as_of(d)
    repo_year_ago = repo_rate_as_of(d - dt.timedelta(days=365))
    if repo_now is not None and repo_year_ago is not None:
        detail["repo_rate"] = repo_now
        detail["repo_change_1y"] = round(repo_now - repo_year_ago, 2)
        if repo_now > repo_year_ago + 0.24:
            rate_state = "tightening"
            reasons.append(f"Repo rate up {repo_now - repo_year_ago:.2f}pp over a "
                           "year. Tightening cycles tend to produce choppier markets.")
        elif repo_now < repo_year_ago - 0.24:
            rate_state = "easing"
            reasons.append(f"Repo rate down {repo_year_ago - repo_now:.2f}pp over a "
                           "year — generally supportive of trending conditions.")
        else:
            rate_state = "neutral"

    # --- Risk appetite, from currency and gold ---
    risk = "unknown"
    signals = []
    if "usdinr" in market and len(market["usdinr"]) > 60:
        s = market["usdinr"]["Close"]
        chg = float(s.iloc[-1] / s.iloc[-60] - 1) * 100
        detail["usdinr_60d_pct"] = round(chg, 2)
        # A rapidly weakening rupee usually accompanies foreign outflows
        signals.append(-1 if chg > 2.0 else (1 if chg < -1.0 else 0))
    if "gold" in market and len(market["gold"]) > 60:
        s = market["gold"]["Close"]
        chg = float(s.iloc[-1] / s.iloc[-60] - 1) * 100
        detail["gold_60d_pct"] = round(chg, 2)
        signals.append(-1 if chg > 8.0 else 0)      # sharp gold rallies = flight to safety

    if signals:
        score = sum(signals)
        risk = "risk_off" if score <= -1 else ("risk_on" if score >= 1 else "neutral")
        if risk == "risk_off":
            reasons.append("Currency and metals point to defensive positioning.")

    # --- Overall judgement ---
    favourable = True
    if vol_state in ("extreme",):
        favourable = False
        reasons.append("VERDICT: volatility state alone argues for reduced exposure.")
    elif vol_state == "elevated" and rate_state == "tightening":
        favourable = False
        reasons.append("VERDICT: elevated volatility plus tightening rates — "
                       "historically poor for momentum.")

    known = sum(1 for s in (vol_state, rate_state, risk) if s != "unknown")
    confidence = "high" if known == 3 else ("medium" if known == 2 else "low")

    return MacroRegime(vol_state, rate_state, risk, favourable,
                       confidence, reasons, detail)


def combine_with_price_regime(price_state: str, macro: MacroRegime) -> dict:
    """Combine the existing price-based gate with macro conditions.

    Deliberately conservative: macro can only *tighten* the gate, never loosen
    it. If the price regime says risk-off, favourable macro does not override
    that. Asymmetry is the point — the cost of trading in a bad regime far
    exceeds the cost of missing a good one.
    """
    order = ["risk_off", "neutral", "risk_on"]
    idx = order.index(price_state) if price_state in order else 1

    downgraded, why = False, []
    if not macro.momentum_favourable and idx > 0:
        idx -= 1
        downgraded = True
        why.append("Macro conditions unfavourable for momentum.")
    if macro.volatility_state == "extreme" and idx > 0:
        idx = 0
        downgraded = True
        why.append("Volatility in the top decile overrides the price regime.")

    return {
        "price_regime": price_state,
        "macro_favourable": macro.momentum_favourable,
        "volatility_state": macro.volatility_state,
        "rate_state": macro.rate_state,
        "combined_regime": order[idx],
        "downgraded": downgraded,
        "reasons": why + macro.reasons,
        "confidence": macro.confidence,
        "note": ("Macro can only tighten the gate, never loosen it. Trading in a "
                 "bad regime costs far more than missing a good one."),
    }


def cache(snap: MacroSnapshot, path: Path = CACHE_PATH) -> int:
    """Append a dated snapshot, building a point-in-time macro archive."""
    row = {"snapshot_date": snap.as_of.isoformat(), **snap.values}
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not existing.empty and "snapshot_date" in existing.columns:
        if (existing["snapshot_date"].astype(str) == row["snapshot_date"]).any():
            return len(existing)
    combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    combined.to_csv(path, index=False)
    return len(combined)
