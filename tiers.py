"""Market-cap tier classification and tier-specific calibration.

A smallcap and a largecap behave differently enough over 15-20 days that one
set of weights cannot serve both. The parameters below are set from reasoning
about *why* each tier behaves as it does — not fitted to backtest returns.
That distinction is what keeps this from being curve-fitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Market cap boundaries in INR crore (SEBI-style, approximate)
LARGE_MIN_CR = 50_000
MID_MIN_CR = 15_000

TIERS = ("large", "mid", "small")

# Rationale for each weighting is in the comments — read before changing.
TIER_WEIGHTS = {
    # Largecaps are mostly index beta. A largecap not beating the index isn't a
    # trade, it's a slower index fund. Hence RS dominates.
    "large": {"Trend": 0.20, "Momentum": 0.15, "Volume_S": 0.10,
              "RelStrength": 0.35, "Setup": 0.20},
    # Midcaps sit between the two — sector rotation matters most here.
    "mid":   {"Trend": 0.25, "Momentum": 0.20, "Volume_S": 0.15,
              "RelStrength": 0.20, "Setup": 0.20},
    # Smallcap moves are stock-specific; RS vs Nifty is mostly noise. Momentum
    # persists far longer, and volume is the main confirmation that a move is real.
    "small": {"Trend": 0.30, "Momentum": 0.28, "Volume_S": 0.22,
              "RelStrength": 0.10, "Setup": 0.10},
}

TIER_PARAMS = {
    "large": {
        "rsi_band": (45, 65),      # mean-reverts quickly above 65
        "rsi_peak": (52, 62),
        "atr_mult": 2.0,           # tighter stops work — less gap risk
        "min_turnover_cr": 50.0,
        "max_position_pct": 25.0,  # can size up; you can always exit
        "max_pct_of_adv": 5.0,
        "est_cost_pct": 0.36,      # 0.26% statutory + 0.10% slippage round trip
        "slippage_pct": 0.10,
    },
    "mid": {
        "rsi_band": (50, 72),
        "rsi_peak": (55, 68),
        "atr_mult": 2.5,
        "min_turnover_cr": 25.0,
        "max_position_pct": 15.0,
        "max_pct_of_adv": 3.0,
        "est_cost_pct": 0.56,      # 0.26% statutory + 0.30% slippage round trip
        "slippage_pct": 0.30,
    },
    "small": {
        "rsi_band": (55, 80),      # smallcap trends stay hot for weeks
        "rsi_peak": (60, 75),
        "atr_mult": 3.0,           # wide stops mandatory — gaps are routine
        "min_turnover_cr": 10.0,
        "max_position_pct": 8.0,   # exit risk, not entry risk, caps this
        "max_pct_of_adv": 2.0,
        "est_cost_pct": 0.96,      # 0.26% statutory + 0.70% slippage round trip
        "slippage_pct": 0.70,
    },
}


def classify_by_mcap(market_cap_inr: float | None) -> str:
    """Tier from market cap in raw INR (as yfinance reports it)."""
    if market_cap_inr is None or not np.isfinite(market_cap_inr) or market_cap_inr <= 0:
        return "mid"
    cr = market_cap_inr / 1e7          # INR -> crore
    if cr >= LARGE_MIN_CR:
        return "large"
    if cr >= MID_MIN_CR:
        return "mid"
    return "small"


def classify_by_turnover(df: pd.DataFrame) -> str:
    """Fallback tier estimate from traded value when market cap is unavailable.

    Crude, but turnover correlates strongly enough with cap tier to be a usable
    proxy — and it never fails the way yfinance's .info does.
    """
    if df is None or len(df) < 20:
        return "mid"
    turn_cr = float((df["Close"] * df["Volume"]).tail(20).mean() / 1e7)
    if turn_cr >= 150:
        return "large"
    if turn_cr >= 30:
        return "mid"
    return "small"


def params(tier: str) -> dict:
    return TIER_PARAMS.get(tier, TIER_PARAMS["mid"])


def weights(tier: str) -> dict:
    return TIER_WEIGHTS.get(tier, TIER_WEIGHTS["mid"])


def position_limits(tier: str, capital: float, price: float,
                    adv_shares: float) -> dict:
    """Hard caps on position size. These bind *before* the ATR risk calc.

    The smallcap failure mode isn't being wrong — it's being right and unable
    to exit. Capping position as a share of average daily volume is the single
    most effective guard against that.
    """
    p = params(tier)
    max_by_capital = capital * p["max_position_pct"] / 100
    max_by_liquidity = adv_shares * p["max_pct_of_adv"] / 100 * price if adv_shares else max_by_capital
    binding = min(max_by_capital, max_by_liquidity)
    return {
        "max_value": binding,
        "max_qty": int(binding // price) if price > 0 else 0,
        "capped_by": "liquidity" if max_by_liquidity < max_by_capital else "capital",
        "max_by_capital": max_by_capital,
        "max_by_liquidity": max_by_liquidity,
    }


def net_expected_r(gross_r: float, tier: str, stop_pct: float) -> float:
    """Expected R after transaction costs INCLUDING slippage.

    Note: est_cost_pct now decomposes into statutory charges (0.26%, identical
    across tiers because STT and stamp duty are ad valorem) plus tier-specific
    slippage. Slippage is the part that actually varies — spreads and market
    impact are what separate a large cap from a small one, not the taxes.


    A 2R smallcap setup with a 6% stop gives up ~0.25R to costs. The same setup
    in a largecap gives up ~0.06R. Ranking without this systematically
    over-values smallcaps.
    """
    cost = params(tier)["est_cost_pct"]
    if stop_pct <= 0:
        return gross_r
    return gross_r - (cost / stop_pct)
