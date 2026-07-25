"""Composite swing score.

Design intent: every component is bounded 0-100, weights are explicit, and the
sub-scores are returned alongside the total so a user can see *why* something
ranked highly and disagree with the weighting.

Calibrated for a 15-20 trading day hold. Nothing here is predictive — it
measures whether a setup matches a trend-continuation profile that historically
precedes such moves. It says nothing about whether this particular instance
will work.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import indicators as ind
import tiers as tr

WEIGHTS = {
    "Trend": 0.25,
    "Momentum": 0.20,
    "Volume_S": 0.15,
    "RelStrength": 0.20,
    "Setup": 0.20,
}


def _clip(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    if x is None or not np.isfinite(x):
        return 0.0
    return float(np.clip(x, lo, hi))


def _trend_score(last: pd.Series) -> float:
    """Price above rising EMAs, in proper alignment, with real trend strength."""
    score = 0.0
    close = last["Close"]

    if close > last["EMA20"]:
        score += 20
    if close > last["EMA50"]:
        score += 25
    if close > last["EMA200"]:
        score += 15
    if last["EMA20"] > last["EMA50"]:
        score += 20

    # ADX: reward genuine trend, but a runaway ADX often marks exhaustion
    adx_v = last.get("ADX14", 0)
    if 20 <= adx_v <= 40:
        score += 20
    elif adx_v > 40:
        score += 10
    elif adx_v >= 15:
        score += 10

    return _clip(score)


def _momentum_score(last: pd.Series, tier: str = "mid") -> float:
    """RSI in the constructive band plus a confirming MACD posture.

    The band is tier-specific: smallcap trends stay hot far longer, so
    penalising a smallcap at RSI 70 systematically excludes the moves you are
    trying to catch. Largecaps mean-revert quickly and get a tighter band.
    """
    score = 0.0
    r = last["RSI14"]
    lo, hi = tr.params(tier)["rsi_peak"]

    if lo <= r <= hi:
        score += 45
    elif hi < r <= hi + 8:
        score += 30
    elif lo - 6 <= r < lo:
        score += 30
    elif r > hi + 8:
        score += 10          # stretched even by this tier's standards
    else:
        score += 5

    if last["MACD_Hist"] > 0:
        score += 30
    if last["MACD"] > last["MACD_Signal"]:
        score += 25

    return _clip(score)


def _volume_score(df: pd.DataFrame, last: pd.Series) -> float:
    """Participation confirming the move. Thin breakouts fail."""
    ratio = last.get("Vol_Ratio", np.nan)
    if not np.isfinite(ratio):
        return 0.0

    if ratio >= 2.0:
        score = 100.0
    elif ratio >= 1.5:
        score = 85.0
    elif ratio >= 1.2:
        score = 70.0
    elif ratio >= 0.9:
        score = 50.0
    else:
        score = 25.0

    # Up-days should carry the heavier volume
    recent = df.tail(10)
    up_vol = recent.loc[recent["Close"] >= recent["Open"], "Volume"].sum()
    dn_vol = recent.loc[recent["Close"] < recent["Open"], "Volume"].sum()
    if dn_vol > 0 and up_vol / dn_vol > 1.3:
        score += 10

    return _clip(score)


def _relative_strength(df: pd.DataFrame, bench: pd.DataFrame | None) -> float:
    """Outperformance vs the index over 20 and 60 days."""
    if bench is None or len(bench) < 61 or len(df) < 61:
        return 50.0

    def _ret(frame: pd.DataFrame, n: int) -> float:
        return float(frame["Close"].iloc[-1] / frame["Close"].iloc[-n - 1] - 1) * 100

    try:
        d20 = _ret(df, 20) - _ret(bench, 20)
        d60 = _ret(df, 60) - _ret(bench, 60)
    except (IndexError, ZeroDivisionError):
        return 50.0

    # +10pp outperformance over 20d maps to roughly the top of the range
    s20 = 50 + d20 * 3.5
    s60 = 50 + d60 * 1.8
    return _clip(0.65 * s20 + 0.35 * s60)


def _setup_score(df: pd.DataFrame, last: pd.Series) -> float:
    """Reward tight consolidation near highs — the classic swing launchpad."""
    score = 0.0

    # Volatility squeeze: current band width vs its own 6-month range
    bw = df["BB_Width"].tail(126).dropna()
    if len(bw) > 30:
        pct = float((bw.iloc[-1] <= bw).mean()) * 100  # low percentile = tight
        score += min(40.0, pct * 0.4)

    # Proximity to 52-week high without being extended
    high_52 = float(df["High"].tail(252).max())
    dist = (last["Close"] / high_52 - 1) * 100 if high_52 > 0 else -100
    if -3 <= dist <= 0:
        score += 35          # coiled right under resistance
    elif -8 < dist < -3:
        score += 28
    elif -15 <= dist <= -8:
        score += 18
    elif dist > 0:
        score += 25          # fresh breakout

    # Controlled pullback depth from recent swing high
    recent_high = float(df["High"].tail(20).max())
    pullback = (last["Close"] / recent_high - 1) * 100 if recent_high > 0 else 0
    if -5 <= pullback <= 0:
        score += 25
    elif -10 < pullback < -5:
        score += 15

    return _clip(score)


def evaluate(df: pd.DataFrame, bench: pd.DataFrame | None = None,
             tier: str | None = None) -> dict:
    """Run the full evaluation on one OHLCV frame.

    If `tier` is None it is inferred from traded value. Weights and the RSI
    band both vary by tier — see tiers.py for the reasoning.
    """
    enriched = ind.enrich(df)
    last = enriched.iloc[-1]
    if tier is None:
        tier = tr.classify_by_turnover(df)

    parts = {
        "Trend": _trend_score(last),
        "Momentum": _momentum_score(last, tier),
        "Volume_S": _volume_score(enriched, last),
        "RelStrength": _relative_strength(enriched, bench),
        "Setup": _setup_score(enriched, last),
    }
    weights = tr.weights(tier)
    total = sum(parts[k] * w for k, w in weights.items())

    close = float(last["Close"])
    atr_v = float(last["ATR14"])

    return {
        **{k: round(v) for k, v in parts.items()},
        "Score": round(total),
        "Tier": tier,
        "Close": round(close, 2),
        "RSI": round(float(last["RSI14"]), 1),
        "ATR": round(atr_v, 2),
        "ATR_pct": round(atr_v / close * 100, 2) if close else 0.0,
        "Ret_20d": round(float(last.get("Ret_20d", 0) or 0), 2),
        "Turnover_Cr": round(float(last.get("Turnover_Cr", 0) or 0), 1),
        "Above_50EMA": bool(close > float(last["EMA50"])),
        "ADX": round(float(last.get("ADX14", 0)), 1),
    }
