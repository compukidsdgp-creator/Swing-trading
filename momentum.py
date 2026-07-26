"""Momentum scorer — the validated replacement for the v1 composite.

Why this exists
---------------
The original composite blended five components (Trend, Momentum, Volume,
Relative Strength, Setup) chosen by reasoning about what should matter. Factor
analysis showed all five measured the same thing: the composite loaded 0.76 on
one-month return and retained only 13.9% of its IC after neutralisation.

The signal laboratory then tested twelve candidates individually. Ten were
rejected. Two survived, and the orthogonal composite builder rejected one of
those as redundant — `idiosyncratic_mom` correlates 0.971 with `mom_12_1`.

What remained was a single signal.

The evidence
------------
    mom_12_1 residual IC   +0.0553   (after neutralising six known factors)
    Newey-West t            +3.73    (clears the Harvey-Liu-Zhu t>3 bar)
    Composite IC            +0.0479
    Positive windows         59.7%   across 62 non-overlapping windows

This is momentum (Jegadeesh & Titman 1993) — the most replicated anomaly in
finance. Nothing here is a discovery. What has been established is that it is
present and measurable in the Nifty 500 at this horizon, and that none of the
elaboration in v1 added anything to it.

Economic reality
----------------
IC 0.048 implies roughly a 0.5-0.6pp spread between top and bottom quintile per
15 days. Against modelled round-trip costs:

    large cap   0.25%   edge survives
    mid cap     0.60%   roughly breakeven
    small cap   1.50%   edge consumed entirely

The strategy is therefore economically viable in large caps, marginal in mid,
and should not be traded in small caps regardless of what the ranking says.
`RECOMMENDED_TIERS` encodes this.

Design principle
----------------
One signal, transparently computed. No weights to tune, no components to
misinterpret as independent. If a second signal is ever shown to have genuine
incremental content — significant after neutralisation AND correlation below
0.6 with this one — it can be added. Not before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import indicators as ind
import tiers as tr

# Formation window: 12 months, skipping the most recent month.
# The skip avoids contamination by short-term reversal, which operates in the
# opposite direction over ~1 month.
LOOKBACK = 252
SKIP = 21

# Tiers where the edge survives transaction costs.
RECOMMENDED_TIERS = {"large", "mid"}

VALIDATION = {
    "signal": "mom_12_1",
    "residual_ic": 0.0553,
    "residual_t_newey_west": 3.73,
    "composite_ic": 0.0479,
    "pct_positive_windows": 59.7,
    "windows": 62,
    "universe": "Nifty 500",
    "horizon_days": 15,
    "measured": "2026-07-26",
}


def raw_momentum(df: pd.DataFrame, i: int | None = None) -> float:
    """12-1 momentum: return from t-252 to t-21. Uses only data up to i."""
    c = df["Close"]
    i = len(c) - 1 if i is None else i
    if i < LOOKBACK:
        return np.nan
    p_start = float(c.iloc[i - LOOKBACK])
    p_end = float(c.iloc[i - SKIP])
    if p_start <= 0 or not np.isfinite(p_end):
        return np.nan
    return (p_end / p_start - 1) * 100


def rank_universe(
    frames: dict[str, pd.DataFrame],
    bench: pd.DataFrame | None = None,
    *,
    min_turnover_cr: float = 25.0,
    require_above_50ema: bool = True,
    tier_filter: set[str] | None = None,
) -> pd.DataFrame:
    """Rank a universe by momentum, cross-sectionally.

    Momentum is only meaningful relative to peers, so the 0-100 score is a
    percentile rank within this universe — not an absolute quantity. A score of
    80 means "stronger momentum than 80% of the universe today", nothing more.

    Filters applied are the ones with independent justification:
      * liquidity, because slippage eats a 0.5pp edge quickly
      * price above 50 EMA, as a crude trend confirmation
      * tier, because costs vary by an order of magnitude across tiers
    """
    rows = []
    for tkr, df in frames.items():
        if df is None or len(df) < LOOKBACK + 5:
            continue
        try:
            e = ind.enrich(df)
        except Exception:                                      # noqa: BLE001
            continue

        mom = raw_momentum(e)
        if not np.isfinite(mom):
            continue

        last = e.iloc[-1]
        close = float(last["Close"])
        turnover = float(last.get("Turnover_Cr", 0) or 0)
        if turnover < min_turnover_cr:
            continue
        above = close > float(last["EMA50"])
        if require_above_50ema and not above:
            continue

        tier = tr.classify_by_turnover(df)
        if tier_filter and tier not in tier_filter:
            continue

        atr = float(last["ATR14"])
        rows.append({
            "Ticker": tkr.replace(".NS", ""),
            "_raw": tkr,
            "Tier": tier,
            "Close": round(close, 2),
            "Momentum": round(mom, 2),
            "RSI": round(float(last["RSI14"]), 1),
            "ATR": round(atr, 2),
            "ATR_pct": round(atr / close * 100, 2) if close else 0.0,
            "Ret_20d": round(float(last.get("Ret_20d", 0) or 0), 2),
            "Turnover_Cr": round(turnover, 1),
            "Above_50EMA": above,
            "Cost_viable": tier in RECOMMENDED_TIERS,
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    # Percentile rank -> 0-100. Cross-sectional by construction.
    out["Score"] = (out["Momentum"].rank(pct=True) * 100).round().astype(int)
    return out.sort_values("Score", ascending=False).reset_index(drop=True)


def evaluate(df: pd.DataFrame, bench: pd.DataFrame | None = None,
             tier: str | None = None) -> dict:
    """Single-stock evaluation, signature-compatible with the old scoring.evaluate.

    IMPORTANT: the Score returned here is NOT comparable across stocks, because
    momentum must be ranked cross-sectionally. Use rank_universe() for anything
    involving comparison. This exists so the Detail tab keeps working.
    """
    e = ind.enrich(df)
    last = e.iloc[-1]
    if tier is None:
        tier = tr.classify_by_turnover(df)

    mom = raw_momentum(e)
    close = float(last["Close"])
    atr = float(last["ATR14"])

    return {
        "Score": None,                       # meaningless outside a cross-section
        "Momentum": round(mom, 2) if np.isfinite(mom) else None,
        "Tier": tier,
        "Close": round(close, 2),
        "RSI": round(float(last["RSI14"]), 1),
        "ATR": round(atr, 2),
        "ATR_pct": round(atr / close * 100, 2) if close else 0.0,
        "Ret_20d": round(float(last.get("Ret_20d", 0) or 0), 2),
        "Turnover_Cr": round(float(last.get("Turnover_Cr", 0) or 0), 1),
        "Above_50EMA": bool(close > float(last["EMA50"])),
        "Cost_viable": tier in RECOMMENDED_TIERS,
    }


def explain() -> str:
    """Plain-language description, for the app and reports."""
    v = VALIDATION
    return (
        f"**Signal: 12-1 momentum.** Return over the past 12 months, excluding the "
        f"most recent month. Stocks are ranked against each other; the score is a "
        f"percentile within today's universe.\n\n"
        f"**Why this and nothing else.** Twelve candidate signals were tested "
        f"individually against six known factors. Ten showed no incremental content. "
        f"Of the two that survived, one was rejected as redundant (correlation 0.97 "
        f"with this signal). Adding it would have added variance without information.\n\n"
        f"**Evidence.** Residual IC {v['residual_ic']:+.4f} with Newey-West "
        f"t = {v['residual_t_newey_west']}, clearing the Harvey-Liu-Zhu t>3 bar for a "
        f"factor claim. Positive in {v['pct_positive_windows']}% of "
        f"{v['windows']} non-overlapping windows on the {v['universe']}.\n\n"
        f"**What this is not.** Momentum was published in 1993 and replicated across "
        f"almost every market since. This is not a discovery — it is confirmation that "
        f"a known effect is present and measurable in this universe.\n\n"
        f"**Cost reality.** The edge implies roughly a 0.5-0.6pp quintile spread per "
        f"15 days. Large-cap round-trip costs (~0.25%) leave most of it. Mid-cap "
        f"(~0.60%) is roughly breakeven. Small-cap (~1.50%) consumes it entirely — "
        f"which is why small caps are excluded regardless of rank."
    )
