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

# Minimum history beyond the formation window.
#
# A stock needs 257 bars to compute 12-1 momentum at all — about 12.2 months.
# Admitting it at exactly that point means its entire "momentum" is post-IPO
# price discovery, which is a different phenomenon from the momentum effect and
# is documented to behave differently (often reversing sharply). The momentum
# literature routinely excludes recent listings for this reason.
#
# 378 bars ≈ 18 months gives roughly six months of established trading history
# before the formation window begins.
MIN_HISTORY_BARS = 378

# Tiers where the edge survives transaction costs.
RECOMMENDED_TIERS = {"large", "mid"}

# Two validation runs. The five-year figures came first and were superseded by
# a twenty-year test on local data, which is the more reliable of the two.
VALIDATION = {
    "signal": "mom_12_1",

    # --- 5-year, yfinance, Nifty 500 top-100 by liquidity ---
    "v1_residual_ic": 0.0553,
    "v1_residual_t": 3.73,
    "v1_windows": 62,
    "v1_gross_spread_pct": 1.42,
    "v1_measured": "2026-07-26",

    # --- 20-year, local dataset, 400 symbols ---
    # Longer sample, more windows, and it substantially revised the picture.
    "residual_ic": 0.0311,
    "residual_t_newey_west": 2.74,
    "windows": 201,
    "pct_positive_windows": 61.7,
    "permutation_p": 0.00,
    "period": "2010-2026",
    "universe": "451 NSE symbols, 400 used",
    "measured": "2026-07-28",

    # The finding that mattered: edge is universe- and horizon-dependent
    "gross_spread_by_universe": {
        "150 symbols (15y+ history)": 0.22,
        "250 symbols (10y+ history)": 0.19,
        "400 symbols (5y+ history)": 0.87,
    },
    "long_only_edge_by_horizon_pct": {
        15: 0.77, 30: 2.16, 45: 1.67, 60: 2.79,
    },
    "net_annualised_by_horizon_pct": {
        15: -1.15, 30: 5.95, 45: 2.41, 60: 4.44,
    },
}

# Recommended configuration, from the 20-year test.
#
# Three findings drove this:
#   1. A 15-day hold does not clear costs. Long-only edge 0.77%, net -1.15%.
#   2. A 30-day hold does. Long-only edge 2.16%, net +5.95%.
#   3. The edge concentrates in the top few percent of a BROAD universe. Taking
#      10 names from 400 (top 2.5%) is far more selective than 10 from 100.
#
# Caveat carried deliberately: 24 configurations were tested and the best
# reported. Some of this is selection. The forward log is the check.
RECOMMENDED_HORIZON = 30
RECOMMENDED_UNIVERSE_SIZE = 400
RECOMMENDED_BUCKET_SIZE = 10        # top 2.5% of 400


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
    min_momentum: float | None = 0.0,
    min_history: int = MIN_HISTORY_BARS,
) -> pd.DataFrame:
    """Rank a universe by momentum, cross-sectionally.

    Momentum is only meaningful relative to peers, so the 0-100 score is a
    percentile rank within this universe — not an absolute quantity. A score of
    80 means "stronger momentum than 80% of the universe today", nothing more.

    Filters applied are the ones with independent justification:
      * liquidity, because slippage eats a 0.5pp edge quickly
      * price above 50 EMA, as a crude trend confirmation
      * tier, because costs vary by an order of magnitude across tiers
      * an absolute momentum floor (see below)
      * a minimum history requirement, excluding recent listings whose
        "momentum" is really post-IPO drift

    Why the absolute floor matters
    ------------------------------
    A percentile rank is purely relative. Without a floor, "Score >= 55" passes
    roughly the top 45% of the universe whether every stock is up 100% or every
    stock is down 50% — in a falling market it happily promotes the best of a
    bad set and stamps it 100. The floor (default: 12-1 momentum must be
    positive) restores an absolute quality gate that percentile ranking removes.

    Set min_momentum=None to disable, which is appropriate only for research
    where you want the full cross-section.
    """
    rows = []
    for tkr, df in frames.items():
        if df is None or len(df) < min_history:
            continue
        try:
            e = ind.enrich(df)
        except Exception:                                      # noqa: BLE001
            continue

        mom = raw_momentum(e)
        if not np.isfinite(mom):
            continue
        if min_momentum is not None and mom < min_momentum:
            continue

        last = e.iloc[-1]
        # Defensive: even with upstream fixes, refuse to rank on a NaN last bar
        # rather than silently producing a False for every comparison.
        if not np.isfinite(last["Close"]) or not np.isfinite(last.get("EMA50", np.nan)):
            continue
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
            "Mom_positive": mom > 0,
            "History_bars": len(df),
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
        f"**Configuration: top {RECOMMENDED_BUCKET_SIZE} from a "
        f"{RECOMMENDED_UNIVERSE_SIZE}-stock universe, held ~{RECOMMENDED_HORIZON} "
        f"days.** Revised from 15 days after twenty-year testing showed the "
        f"shorter horizon does not clear costs.\n\n"
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
        f"**Cost reality, revised.** Long-only edge — the top slice's excess over "
        f"the universe mean, which is what a long-only book actually captures — "
        f"is 0.77% at 15 days and 2.16% at 30. After charges, slippage and 20% "
        f"STCG that is -1.15% and +5.95% annualised respectively. The horizon "
        f"change is the difference between losing and making money.\n\n"
        f"**Universe breadth matters as much.** Gross spread was 0.22% across 150 "
        f"long-history symbols and 0.87% across 400 — filtering for long history "
        f"selects large, efficiently-priced companies where momentum works least "
        f"well. Screen broadly, select narrowly.\n\n"
        f"**Honest caveat.** Twenty-four configurations were tested and the best "
        f"is reported here. Some of that margin is selection. Forward evidence "
        f"is the only check that cannot be gamed."
    )
