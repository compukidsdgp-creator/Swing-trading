"""Market breadth — how many stocks are participating, not just the index.

Why this matters
----------------
The regime gate already uses one breadth measure: the share of the universe
above its own 50 DMA. That is a reasonable single number and it does useful
work, but it cannot distinguish situations that behave very differently.

A recent reading made the point. The Nifty sat 3.2% below its 200 DMA — clearly
risk-off — while breadth was 58%, meaning most stocks were above their own 50
DMA. Those two facts together say something specific: the index is being dragged
by a handful of heavyweights while the broader market is holding up. A single
breadth number cannot express that.

What this adds
--------------
  **Advance/decline ratio** — how many rose against how many fell today.
  **Above-200-DMA share** — long-term participation, slower and more reliable
  than the 50 DMA measure.
  **New highs vs new lows** — the classic deterioration warning. Indices often
  make new highs on narrowing leadership while new lows quietly expand.
  **Breadth divergence** — index direction against breadth direction. The
  configuration that precedes most significant declines.
  **Breadth thrust** — a rare, well-documented bullish signal (Zweig): breadth
  moving from below 40% to above 61.5% within ten sessions.

Deliberately a gate refinement, not a signal
--------------------------------------------
None of this ranks stocks or generates picks. It informs *whether* to trade,
which means no new candidate signals and no multiple-testing problem. Twelve
signals were tested and ten failed; this module deliberately adds none.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Zweig breadth thrust: 10-day EMA of advancers/(advancers+decliners) moving
# from below 0.40 to above 0.615 within ten sessions. Rare, and historically
# followed by strong returns.
THRUST_LOW = 0.40
THRUST_HIGH = 0.615
THRUST_WINDOW = 10

# Divergence thresholds. Deliberately wide — breadth is noisy day to day, and
# a sensitive divergence detector fires constantly and gets ignored.
DIVERGENCE_LOOKBACK = 20
DIVERGENCE_MIN_GAP = 0.15


@dataclass
class BreadthReading:
    pct_above_50dma: float | None = None
    pct_above_200dma: float | None = None
    advance_decline_ratio: float | None = None
    new_highs: int = 0
    new_lows: int = 0
    high_low_ratio: float | None = None
    net_new_highs_pct: float | None = None
    universe_size: int = 0
    thrust_active: bool = False
    divergence: str = "none"          # none | bearish | bullish
    state: str = "unknown"            # strong | healthy | narrowing | weak
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pct_above_50dma": self.pct_above_50dma,
            "pct_above_200dma": self.pct_above_200dma,
            "advance_decline_ratio": self.advance_decline_ratio,
            "new_highs": self.new_highs,
            "new_lows": self.new_lows,
            "net_new_highs_pct": self.net_new_highs_pct,
            "universe_size": self.universe_size,
            "thrust_active": self.thrust_active,
            "divergence": self.divergence,
            "state": self.state,
        }


def _pct_above_ma(frames: dict[str, pd.DataFrame], window: int) -> float | None:
    vals = []
    for df in frames.values():
        if df is None or len(df) < window:
            continue
        c = df["Close"]
        ma = c.rolling(window).mean().iloc[-1]
        if pd.notna(ma) and np.isfinite(ma):
            vals.append(float(c.iloc[-1]) > float(ma))
    return (sum(vals) / len(vals)) if vals else None


def _advance_decline(frames: dict[str, pd.DataFrame]) -> tuple[int, int]:
    adv = dec = 0
    for df in frames.values():
        if df is None or len(df) < 2:
            continue
        c = df["Close"]
        if not (np.isfinite(c.iloc[-1]) and np.isfinite(c.iloc[-2])):
            continue
        if float(c.iloc[-1]) > float(c.iloc[-2]):
            adv += 1
        elif float(c.iloc[-1]) < float(c.iloc[-2]):
            dec += 1
    return adv, dec


def _new_highs_lows(frames: dict[str, pd.DataFrame],
                    lookback: int = 252) -> tuple[int, int]:
    """Stocks at a 52-week high or low today.

    The deterioration this catches is specific: an index making new highs while
    new lows expand means leadership is narrowing to a handful of names.
    """
    highs = lows = 0
    for df in frames.values():
        if df is None or len(df) < 60:
            continue
        w = df.tail(lookback)
        if "High" in w.columns and "Low" in w.columns:
            hi, lo = float(w["High"].max()), float(w["Low"].min())
            last_h = float(w["High"].iloc[-1])
            last_l = float(w["Low"].iloc[-1])
        else:
            c = w["Close"]
            hi, lo = float(c.max()), float(c.min())
            last_h = last_l = float(c.iloc[-1])
        if not (np.isfinite(hi) and np.isfinite(lo)):
            continue
        if last_h >= hi * 0.999:
            highs += 1
        elif last_l <= lo * 1.001:
            lows += 1
    return highs, lows


def _breadth_series(frames: dict[str, pd.DataFrame], *, window: int = 50,
                    days: int = 60) -> pd.Series:
    """Historical breadth, for divergence and thrust detection."""
    common = None
    for df in frames.values():
        if df is None or len(df) < window + days:
            continue
        idx = df.index[-days:]
        common = idx if common is None else common.intersection(idx)
    if common is None or len(common) < 10:
        return pd.Series(dtype=float)

    rows = []
    for d in common:
        above = total = 0
        for df in frames.values():
            if df is None or d not in df.index:
                continue
            sub = df.loc[:d, "Close"]
            if len(sub) < window:
                continue
            ma = sub.rolling(window).mean().iloc[-1]
            if pd.notna(ma):
                total += 1
                above += float(sub.iloc[-1]) > float(ma)
        if total:
            rows.append((d, above / total))
    return pd.Series(dict(rows)).sort_index() if rows else pd.Series(dtype=float)


def _detect_thrust(frames: dict[str, pd.DataFrame]) -> bool:
    """Zweig breadth thrust — rare, and historically a strong bullish signal."""
    series = _breadth_series(frames, days=THRUST_WINDOW * 3)
    if len(series) < THRUST_WINDOW + 2:
        return False
    ema = series.ewm(span=THRUST_WINDOW, adjust=False).mean()
    recent = ema.tail(THRUST_WINDOW + 1)
    return bool(recent.min() < THRUST_LOW and recent.iloc[-1] > THRUST_HIGH)


def _detect_divergence(frames: dict[str, pd.DataFrame],
                       bench: pd.DataFrame | None) -> str:
    """Index direction against breadth direction.

    Bearish divergence — index up, participation falling — is the configuration
    that precedes most significant declines. It is not a timing signal; it is a
    reason to size down.
    """
    if bench is None or len(bench) < DIVERGENCE_LOOKBACK + 5:
        return "none"
    series = _breadth_series(frames, days=DIVERGENCE_LOOKBACK + 10)
    if len(series) < DIVERGENCE_LOOKBACK:
        return "none"

    b = bench["Close"]
    idx_chg = float(b.iloc[-1] / b.iloc[-DIVERGENCE_LOOKBACK] - 1)
    br_now = float(series.iloc[-1])
    br_then = float(series.iloc[-min(DIVERGENCE_LOOKBACK, len(series))])
    br_chg = br_now - br_then

    if idx_chg > 0.01 and br_chg < -DIVERGENCE_MIN_GAP:
        return "bearish"
    if idx_chg < -0.01 and br_chg > DIVERGENCE_MIN_GAP:
        return "bullish"
    return "none"


def analyse(frames: dict[str, pd.DataFrame],
            bench: pd.DataFrame | None = None) -> BreadthReading:
    """Full breadth reading for the current session."""
    r = BreadthReading()
    if not frames:
        r.notes.append("No price data.")
        return r

    r.universe_size = len(frames)
    r.pct_above_50dma = _pct_above_ma(frames, 50)
    r.pct_above_200dma = _pct_above_ma(frames, 200)

    adv, dec = _advance_decline(frames)
    if adv + dec:
        r.advance_decline_ratio = round(adv / max(dec, 1), 2)

    r.new_highs, r.new_lows = _new_highs_lows(frames)
    total_hl = r.new_highs + r.new_lows
    if total_hl:
        r.high_low_ratio = round(r.new_highs / max(r.new_lows, 1), 2)
    if r.universe_size:
        r.net_new_highs_pct = round(
            (r.new_highs - r.new_lows) / r.universe_size * 100, 1)

    r.thrust_active = _detect_thrust(frames)
    r.divergence = _detect_divergence(frames, bench)

    # --- State, from the slower and more reliable measures ---
    p50 = r.pct_above_50dma
    p200 = r.pct_above_200dma

    if p50 is None:
        r.state = "unknown"
    elif p50 >= 0.60 and (p200 is None or p200 >= 0.50):
        r.state = "strong"
    elif p50 >= 0.45:
        r.state = "healthy"
    elif p50 >= 0.30:
        r.state = "narrowing"
    else:
        r.state = "weak"

    # --- Warnings ---
    if r.divergence == "bearish":
        r.warnings.append(
            "Bearish divergence: the index has risen while participation fell. "
            "This is the configuration that precedes most significant declines "
            "— a reason to size down rather than a timing signal.")
    if r.net_new_highs_pct is not None and r.net_new_highs_pct < -5:
        r.warnings.append(
            f"New lows exceed new highs by {abs(r.net_new_highs_pct):.1f}% of "
            "the universe. Leadership is narrowing.")
    if p50 is not None and p200 is not None and p50 > 0.55 and p200 < 0.35:
        r.warnings.append(
            "Short-term participation is healthy but long-term is weak — a "
            "bounce within a downtrend rather than a recovery.")
    if r.state == "weak":
        r.warnings.append(
            f"Only {p50:.0%} of the universe is above its own 50 DMA. Momentum "
            "requires participation; it has little to work with here.")

    # --- Notes ---
    if r.thrust_active:
        r.notes.append(
            "Breadth thrust detected (Zweig): participation moved from below "
            "40% to above 61.5% within ten sessions. Rare, and historically "
            "followed by strong returns.")
    if p50 is not None and p200 is not None:
        gap = p50 - p200
        if gap > 0.25:
            r.notes.append(
                f"Short-term participation ({p50:.0%}) far exceeds long-term "
                f"({p200:.0%}) — an early recovery, or a bear-market rally.")
        elif gap < -0.15:
            r.notes.append(
                f"Long-term participation ({p200:.0%}) exceeds short-term "
                f"({p50:.0%}) — a pullback inside an intact uptrend.")

    return r


def regime_adjustment(reading: BreadthReading, price_regime: str) -> dict:
    """How breadth should modify the price-based regime.

    Deliberately asymmetric: breadth can only tighten the gate, never loosen
    it. The single exception is a confirmed breadth thrust, which is rare and
    well documented enough to justify one step up — and even then only from
    risk_off to neutral, never straight to risk_on.
    """
    order = ["risk_off", "neutral", "risk_on"]
    idx = order.index(price_regime) if price_regime in order else 1
    reasons, changed = [], False

    if reading.state == "weak" and idx > 0:
        idx = 0
        changed = True
        reasons.append("Breadth is weak — fewer than 30% of stocks above their "
                       "own 50 DMA.")
    elif reading.state == "narrowing" and idx > 1:
        idx = 1
        changed = True
        reasons.append("Breadth is narrowing — participation below 45%.")

    if reading.divergence == "bearish" and idx > 0:
        idx -= 1
        changed = True
        reasons.append("Bearish breadth divergence.")

    if reading.thrust_active and idx == 0:
        idx = 1
        changed = True
        reasons.append("Breadth thrust — the one documented case where breadth "
                       "justifies loosening rather than tightening.")

    return {
        "price_regime": price_regime,
        "breadth_state": reading.state,
        "adjusted_regime": order[idx],
        "changed": changed,
        "reasons": reasons,
        "warnings": reading.warnings,
        "note": ("Breadth may only tighten the gate. The sole exception is a "
                 "confirmed thrust, and even then only one step."),
    }


def summary_line(reading: BreadthReading) -> str:
    """One line for logs and Telegram."""
    parts = []
    if reading.pct_above_50dma is not None:
        parts.append(f"{reading.pct_above_50dma:.0%} above 50DMA")
    if reading.pct_above_200dma is not None:
        parts.append(f"{reading.pct_above_200dma:.0%} above 200DMA")
    if reading.advance_decline_ratio is not None:
        parts.append(f"A/D {reading.advance_decline_ratio:.2f}")
    if reading.new_highs or reading.new_lows:
        parts.append(f"{reading.new_highs}H/{reading.new_lows}L")
    return f"breadth {reading.state}: " + " · ".join(parts)
