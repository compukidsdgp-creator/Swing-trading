"""Market regime detection.

The single highest-value component in this system. Smallcaps fall 2-3x the
index in drawdowns and no chart pattern survives that. Knowing when *not* to
trade beats any amount of weight tuning.

Regimes are deliberately simple — three states from two signals. Complex regime
models overfit as badly as complex entry models, and the whole point of this
gate is that it should be hard to break.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

RISK_ON = "risk_on"
NEUTRAL = "neutral"
RISK_OFF = "risk_off"

# Which tiers are permitted to trade in each regime
TIER_PERMISSION = {
    RISK_ON: {"large", "mid", "small"},
    NEUTRAL: {"large", "mid"},
    RISK_OFF: {"large"},
}

DESCRIPTION = {
    RISK_ON: "Index above 200 DMA with a rising 50 DMA. All tiers permitted.",
    NEUTRAL: "Index above 200 DMA but 50 DMA flat or falling. Large and mid only.",
    RISK_OFF: "Index below 200 DMA. Large cap only — or sit out.",
}


@dataclass
class Regime:
    state: str
    above_200dma: bool
    dma50_rising: bool
    pct_from_200dma: float
    breadth: float | None = None

    @property
    def allowed_tiers(self) -> set[str]:
        return TIER_PERMISSION[self.state]

    @property
    def description(self) -> str:
        return DESCRIPTION[self.state]

    def permits(self, tier: str) -> bool:
        return tier in self.allowed_tiers


def classify(bench: pd.DataFrame, breadth: float | None = None) -> Regime:
    """Classify market regime from the benchmark index.

    Args:
        bench: OHLCV frame for the index (Nifty 50).
        breadth: optional — share of universe above its own 50 DMA (0-1).
                 Used only as a tiebreaker, never as the primary signal.
    """
    if bench is None or len(bench) < 200:
        # Not enough history to judge — assume the cautious case
        return Regime(NEUTRAL, False, False, 0.0, breadth)

    close = bench["Close"]
    dma200 = close.rolling(200).mean()
    dma50 = close.rolling(50).mean()

    last = float(close.iloc[-1])
    d200 = float(dma200.iloc[-1])
    above = last > d200
    rising = bool(float(dma50.diff(10).iloc[-1]) > 0)
    pct = (last / d200 - 1) * 100 if d200 else 0.0

    if above and rising:
        state = RISK_ON
    elif above:
        state = NEUTRAL
    else:
        state = RISK_OFF

    # Breadth override: index can be held up by a handful of heavyweights while
    # the average stock is already broken. If fewer than 35% of names are above
    # their own 50 DMA, downgrade regardless of what the index says.
    if breadth is not None and breadth < 0.35 and state == RISK_ON:
        state = NEUTRAL
    if breadth is not None and breadth < 0.20:
        state = RISK_OFF

    return Regime(state, above, rising, round(pct, 2), breadth)


def classify_at(bench: pd.DataFrame, i: int) -> str:
    """Regime state as of bar `i` — used by the backtest to avoid lookahead."""
    if i < 200:
        return NEUTRAL
    window = bench.iloc[: i + 1]
    close = window["Close"]
    d200 = close.rolling(200).mean().iloc[-1]
    d50_slope = close.rolling(50).mean().diff(10).iloc[-1]
    if pd.isna(d200) or pd.isna(d50_slope):
        return NEUTRAL
    above = float(close.iloc[-1]) > float(d200)
    if above and float(d50_slope) > 0:
        return RISK_ON
    if above:
        return NEUTRAL
    return RISK_OFF


def compute_breadth(frames: dict[str, pd.DataFrame]) -> float | None:
    """Share of the universe trading above its own 50 DMA."""
    vals = []
    for df in frames.values():
        if df is None or len(df) < 50:
            continue
        c = df["Close"]
        ma = c.rolling(50).mean().iloc[-1]
        if pd.notna(ma):
            vals.append(float(c.iloc[-1]) > float(ma))
    if not vals:
        return None
    return sum(vals) / len(vals)
