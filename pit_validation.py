"""Point-in-time validation — measuring the edge without hindsight.

The gap this closes
-------------------
Item #1 in the compliance audit, outstanding since it was written. Every
published figure — IC 0.0311, gross spread 0.46% — was measured on **today's**
constituent list backfilled through history. Companies that were delisted,
merged or wound up over the period are absent entirely.

That biases results upward, and the magnitude is unknown. A stock that was in
the index in 2015 and collapsed in 2018 never appears, so the ranking is scored
against a population of survivors.

What this does differently
--------------------------
Rebuilds the universe **at each observation date** from NSE bhavcopy, which
records every security that actually traded that day. A stock is in the
universe on 15 March 2018 if and only if it traded on 15 March 2018 and met the
liquidity floor then.

The comparison is the point. Running standard validation and PIT validation on
the same period and same signal isolates the survivorship contribution — and
that number has never been measured for this system.

What it still cannot fix
------------------------
Bhavcopy records securities that traded. A company that was wound up leaves no
final trade, and a merger simply stops appearing. So this removes **index
membership** bias, which is the larger component, and reduces but does not
eliminate **delisting** bias.

It is a substantial improvement, not a complete solution, and the difference
matters when reporting the result.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import indicators as ind
import momentum as momo
from backtest import MIN_HISTORY
from factor_analysis import _newey_west_se, _spearman


@dataclass
class PITResult:
    mean_ic: float
    ic_t: float
    windows: int
    pct_positive: float
    gross_spread_pct: float
    spread_t: float
    mean_universe_size: float
    universe_churn_pct: float
    per_window: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    standard: PITResult | None
    point_in_time: PITResult | None
    ic_difference: float | None
    spread_difference: float | None
    survivorship_inflation_pct: float | None
    verdict: str
    message: str


def _universe_at(bhav_frames: dict, date: pd.Timestamp,
                 *, min_turnover_cr: float, min_trades: int,
                 eq_only: bool) -> set[str]:
    """Which securities were investable on this date, judged only on that day."""
    import bhavcopy as bc

    key = date.date() if hasattr(date, "date") else date
    day = bhav_frames.get(key)
    if day is None or day.empty:
        return set()
    try:
        inv = bc.pit_universe(day, min_turnover_cr=min_turnover_cr,
                              min_trades=min_trades, eq_only=eq_only)
        return set(inv["symbol"].astype(str))
    except Exception:                                          # noqa: BLE001
        return set()


def validate_pit(
    frames: dict[str, pd.DataFrame],
    bhav_frames: dict,
    *,
    horizon: int = 30,
    min_turnover_cr: float = 10.0,
    min_trades: int = 500,
    eq_only: bool = True,
    min_names: int = 30,
    progress=None,
) -> PITResult:
    """IC measured against a universe rebuilt at every observation date.

    Args:
        frames: {ticker: OHLCV} price history, superset of any universe.
        bhav_frames: {date: bhavcopy DataFrame} from bhavcopy.load_range().
    """
    # Checked first. During development a synthetic cache with placeholder
    # symbols was left in place; without this it would have produced a
    # confident, entirely meaningless result.
    if bhav_frames:
        sample = bhav_frames[sorted(bhav_frames)[0]]
        if "symbol" in sample.columns:
            syms = set(sample["symbol"].astype(str).head(50))
            if any(s.startswith(("SURV", "DEAD", "STK", "TEST", "S0", "T0"))
                   for s in syms):
                return PITResult(0, 0, 0, 0, 0, 0, 0, 0, notes=[
                    "SYNTHETIC TEST DATA detected in the bhavcopy cache "
                    f"(symbols like {sorted(syms)[:3]}). Refusing to validate "
                    "against placeholder data. Clear the cache and download "
                    "real bhavcopies first."])

    enriched = {}
    for t, df in frames.items():
        if df is None or len(df) < MIN_HISTORY + horizon + 5:
            continue
        try:
            enriched[t] = ind.enrich(df)
        except Exception:                                      # noqa: BLE001
            continue
    if not enriched:
        return PITResult(0, 0, 0, 0, 0, 0, 0, 0,
                         notes=["No usable price data."])

    dates = sorted(bhav_frames)
    if len(dates) < horizon + 60:
        return PITResult(0, 0, 0, 0, 0, 0, 0, 0,
                         notes=[f"Only {len(dates)} bhavcopy days cached. "
                                f"Need at least {horizon + 60}. Download more "
                                "history first."])

    cal = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    step = horizon + 3
    rows = []
    prev_members: set[str] = set()
    churns = []

    start_idx = 0
    for k in range(start_idx, len(cal) - horizon - 1, step):
        date = cal[k]
        if progress:
            progress(k, len(cal), date.date().isoformat())

        members = _universe_at(bhav_frames, date,
                               min_turnover_cr=min_turnover_cr,
                               min_trades=min_trades, eq_only=eq_only)
        if len(members) < min_names:
            continue

        if prev_members:
            churn = len(members ^ prev_members) / max(len(prev_members), 1)
            churns.append(churn * 100)
        prev_members = members

        sc, fwd = [], []
        for sym in members:
            tkr = f"{sym}.NS"
            e = enriched.get(tkr)
            if e is None:
                continue
            try:
                i = e.index.get_loc(date)
            except KeyError:
                continue
            if not isinstance(i, int) or i < MIN_HISTORY or i + horizon >= len(e):
                continue
            m = momo.raw_momentum(e, i)
            if not np.isfinite(m):
                continue
            p0 = float(e["Close"].iloc[i])
            p1 = float(e["Close"].iloc[i + horizon])
            if p0 <= 0 or not np.isfinite(p1):
                continue
            sc.append(m)
            fwd.append((p1 / p0 - 1) * 100)

        if len(sc) < min_names:
            continue

        s, f = np.array(sc), np.array(fwd)
        ic = _spearman(s, f)
        if not np.isfinite(ic):
            continue
        order = np.argsort(s)
        q = max(1, len(s) // 5)
        rows.append({
            "date": date.date().isoformat(),
            "n": len(s),
            "ic": ic,
            "spread": f[order[-q:]].mean() - f[order[:q]].mean(),
            "universe": len(members),
        })

    if len(rows) < 8:
        return PITResult(0, 0, 0, 0, 0, 0, 0, 0,
                         notes=[f"Only {len(rows)} valid windows. Need at least 8."])

    df = pd.DataFrame(rows)
    ic_arr = df["ic"].to_numpy()
    sp_arr = df["spread"].to_numpy()
    ic_se = _newey_west_se(ic_arr)
    sp_se = _newey_west_se(sp_arr)

    notes = [
        "Universe rebuilt at every observation date from NSE bhavcopy — a "
        "security is included only if it actually traded that day and met the "
        "liquidity floor then.",
        "Index-membership bias removed. Delisting bias reduced but not "
        "eliminated: a wound-up company leaves no final trade.",
    ]
    if churns:
        notes.append(f"Universe churn averaged {np.mean(churns):.1f}% between "
                     "observations — that turnover is exactly what a fixed "
                     "present-day list hides.")

    return PITResult(
        mean_ic=round(float(ic_arr.mean()), 4),
        ic_t=round(float(ic_arr.mean() / ic_se), 2) if ic_se else np.nan,
        windows=len(df),
        pct_positive=round(float((ic_arr > 0).mean()) * 100, 1),
        gross_spread_pct=round(float(sp_arr.mean()), 3),
        spread_t=round(float(sp_arr.mean() / sp_se), 2) if sp_se else np.nan,
        mean_universe_size=round(float(df["universe"].mean()), 1),
        universe_churn_pct=round(float(np.mean(churns)), 1) if churns else 0.0,
        per_window=df, notes=notes,
    )


def validate_standard(
    frames: dict[str, pd.DataFrame],
    *,
    horizon: int = 30,
    min_names: int = 30,
) -> PITResult:
    """The same measurement on a fixed present-day universe, for comparison."""
    enriched = {}
    for t, df in frames.items():
        if df is None or len(df) < MIN_HISTORY + horizon + 5:
            continue
        try:
            enriched[t] = ind.enrich(df)
        except Exception:                                      # noqa: BLE001
            continue
    if not enriched:
        return PITResult(0, 0, 0, 0, 0, 0, 0, 0, notes=["No usable data."])

    cal = pd.DatetimeIndex(max((e.index for e in enriched.values()), key=len))
    step = horizon + 3
    rows = []

    for k in range(MIN_HISTORY, len(cal) - horizon - 1, step):
        date = cal[k]
        sc, fwd = [], []
        for t, e in enriched.items():
            try:
                i = e.index.get_loc(date)
            except KeyError:
                continue
            if not isinstance(i, int) or i < MIN_HISTORY or i + horizon >= len(e):
                continue
            m = momo.raw_momentum(e, i)
            if not np.isfinite(m):
                continue
            p0 = float(e["Close"].iloc[i])
            p1 = float(e["Close"].iloc[i + horizon])
            if p0 <= 0 or not np.isfinite(p1):
                continue
            sc.append(m)
            fwd.append((p1 / p0 - 1) * 100)
        if len(sc) < min_names:
            continue
        s, f = np.array(sc), np.array(fwd)
        ic = _spearman(s, f)
        if not np.isfinite(ic):
            continue
        order = np.argsort(s)
        q = max(1, len(s) // 5)
        rows.append({"date": date.date().isoformat(), "n": len(s), "ic": ic,
                     "spread": f[order[-q:]].mean() - f[order[:q]].mean(),
                     "universe": len(enriched)})

    if len(rows) < 8:
        return PITResult(0, 0, 0, 0, 0, 0, 0, 0,
                         notes=[f"Only {len(rows)} windows."])

    df = pd.DataFrame(rows)
    ic_arr, sp_arr = df["ic"].to_numpy(), df["spread"].to_numpy()
    ic_se, sp_se = _newey_west_se(ic_arr), _newey_west_se(sp_arr)

    return PITResult(
        mean_ic=round(float(ic_arr.mean()), 4),
        ic_t=round(float(ic_arr.mean() / ic_se), 2) if ic_se else np.nan,
        windows=len(df),
        pct_positive=round(float((ic_arr > 0).mean()) * 100, 1),
        gross_spread_pct=round(float(sp_arr.mean()), 3),
        spread_t=round(float(sp_arr.mean() / sp_se), 2) if sp_se else np.nan,
        mean_universe_size=float(len(enriched)),
        universe_churn_pct=0.0,
        per_window=df,
        notes=["Fixed present-day universe — carries survivorship bias."],
    )


def compare(standard: PITResult, pit: PITResult) -> ComparisonResult:
    """How much of the measured edge was survivorship?"""
    if not standard or not pit or standard.windows < 8 or pit.windows < 8:
        return ComparisonResult(standard, pit, None, None, None, "error",
                                "Not enough windows in one or both runs.")

    ic_diff = round(pit.mean_ic - standard.mean_ic, 4)
    sp_diff = round(pit.gross_spread_pct - standard.gross_spread_pct, 3)
    inflation = (round((1 - pit.mean_ic / standard.mean_ic) * 100, 1)
                 if standard.mean_ic > 0 else None)

    if pit.mean_ic <= 0:
        verdict, msg = "fail", (
            f"**Point-in-time IC is {pit.mean_ic:+.4f}.** The edge does not "
            f"survive an honest universe. The standard measurement of "
            f"{standard.mean_ic:+.4f} was survivorship, not signal.")
    elif inflation is not None and inflation > 50:
        verdict, msg = "warn", (
            f"**Survivorship inflated the result by {inflation:.0f}%.** "
            f"IC falls from {standard.mean_ic:+.4f} to {pit.mean_ic:+.4f} once "
            f"the universe is rebuilt honestly. Gross spread falls from "
            f"{standard.gross_spread_pct:.2f}% to {pit.gross_spread_pct:.2f}%. "
            "The edge is real but materially smaller than reported.")
    elif inflation is not None and inflation > 20:
        verdict, msg = "ok", (
            f"Survivorship accounted for {inflation:.0f}% of the measured edge. "
            f"IC {standard.mean_ic:+.4f} → {pit.mean_ic:+.4f}. Meaningful but "
            "not fatal — use the point-in-time figure as the benchmark.")
    elif inflation is not None and inflation < 0:
        verdict, msg = "good", (
            f"Point-in-time IC ({pit.mean_ic:+.4f}) is *higher* than the "
            f"standard measurement ({standard.mean_ic:+.4f}). Unusual but not "
            "impossible: a wider, churning universe gives cross-sectional "
            "momentum more dispersion to rank. Treat the PIT figure as the "
            "honest one either way.")
    elif inflation is None:
        # Standard IC was zero or negative, so a percentage change is undefined.
        verdict, msg = "good", (
            f"The standard measurement produced IC {standard.mean_ic:+.4f} — no "
            f"usable signal — while point-in-time gives {pit.mean_ic:+.4f}. "
            "A percentage comparison is undefined here. The likely reason is "
            "that a survivors-only universe strips out the failing companies "
            "that give cross-sectional momentum something real to rank against.")
    else:
        verdict, msg = "good", (
            f"Survivorship contributed only {inflation:.0f}% of the measured "
            f"edge. IC {standard.mean_ic:+.4f} → {pit.mean_ic:+.4f}. The "
            "result substantially holds.")

    msg += (f"\n\nPIT universe averaged {pit.mean_universe_size:.0f} securities "
            f"with {pit.universe_churn_pct:.1f}% churn between observations, "
            f"against a fixed {standard.mean_universe_size:.0f}.")

    return ComparisonResult(standard, pit, ic_diff, sp_diff, inflation,
                            verdict, msg)


def summary_table(cmp_: ComparisonResult) -> pd.DataFrame:
    """Side-by-side comparison for reports."""
    if not cmp_.standard or not cmp_.point_in_time:
        return pd.DataFrame()
    s, p = cmp_.standard, cmp_.point_in_time
    return pd.DataFrame([
        {"metric": "Mean IC", "standard": s.mean_ic, "point_in_time": p.mean_ic,
         "difference": cmp_.ic_difference},
        {"metric": "IC t-stat", "standard": s.ic_t, "point_in_time": p.ic_t,
         "difference": round(p.ic_t - s.ic_t, 2)},
        {"metric": "Gross spread %", "standard": s.gross_spread_pct,
         "point_in_time": p.gross_spread_pct, "difference": cmp_.spread_difference},
        {"metric": "Positive windows %", "standard": s.pct_positive,
         "point_in_time": p.pct_positive,
         "difference": round(p.pct_positive - s.pct_positive, 1)},
        {"metric": "Windows", "standard": s.windows, "point_in_time": p.windows,
         "difference": p.windows - s.windows},
        {"metric": "Mean universe", "standard": s.mean_universe_size,
         "point_in_time": p.mean_universe_size,
         "difference": round(p.mean_universe_size - s.mean_universe_size, 1)},
    ])
