"""Portfolio construction — risk parity, trailing stops, rebalancing.

What was already in place
-------------------------
Position sizing is already ATR-based and tier-aware, with hard caps on both
capital share (25/15/8% by tier) and participation in average daily volume.
Total exposure already scales with the regime gate and momentum's own realised
volatility. Sectors are already capped. Correlation is already measured.

Three things were genuinely missing, and this module adds them.

1. Risk parity
--------------
Correlation was *measured* but never *used* for weighting. Equal-rupee
allocation across ten names sounds diversified and often is not: a basket
measured earlier showed ten positions behaving as 1.8 independent bets, with
real risk 2.4x what per-position sizing assumed.

Risk parity sets weights so each position contributes equally to portfolio
variance rather than equally to capital. A volatile, highly-correlated name
gets less; an uncorrelated quiet one gets more.

Implemented without scipy — the iterative solver converges reliably for the
long-only, fully-invested case and is transparent enough to audit by eye.

2. Trailing stops
-----------------
Stops were static: entry minus a multiple of ATR, fixed for the holding period.
For a momentum strategy that is the wrong shape. Momentum's return distribution
is positively skewed in the tail — a minority of positions produce most of the
gain — and a fixed stop caps exactly the outcome you are relying on.

A trailing stop ratchets upward with price and never retreats. It gives up some
profit on reversals in exchange for not truncating the winners.

3. Rebalancing
--------------
A position that doubles becomes twice the risk it was sized for. Without a
trigger, portfolio concentration drifts upward silently, and the largest
position is by definition the one that has run furthest — the one most exposed
to mean reversion.

Deliberately threshold-based rather than calendar-based: rebalancing on a
schedule generates turnover for its own sake, and turnover is the binding
constraint here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Rebalancing thresholds. Wide enough that ordinary fluctuation does not
# trigger trading — every rebalance costs a full round trip.
MAX_POSITION_WEIGHT = 0.15        # trim above 15% of portfolio equity
DRIFT_TOLERANCE = 0.05            # ignore drift below 5 percentage points
MIN_TRADE_VALUE = 5_000           # do not bother below this


# --------------------------------------------------------------------------
# Risk parity
# --------------------------------------------------------------------------
@dataclass
class Weights:
    weights: pd.Series
    method: str
    risk_contributions: pd.Series | None = None
    diversification_ratio: float | None = None
    effective_n: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({"weight_pct": (self.weights * 100).round(2)})
        if self.risk_contributions is not None:
            df["risk_contrib_pct"] = (self.risk_contributions * 100).round(2)
        return df.sort_values("weight_pct", ascending=False)


def _covariance(returns: pd.DataFrame, *, shrinkage: float = 0.15) -> np.ndarray:
    """Shrunk covariance matrix.

    A raw sample covariance from ~60 observations across 10 assets is badly
    estimated — the smallest eigenvalues are noise, and an optimiser will load
    the portfolio onto exactly those. Shrinking toward a diagonal target is the
    standard remedy (Ledoit-Wolf in spirit, simplified here).
    """
    cov = returns.cov().to_numpy()
    if shrinkage <= 0:
        return cov
    target = np.diag(np.diag(cov))
    return (1 - shrinkage) * cov + shrinkage * target


def risk_parity(returns: pd.DataFrame, *, max_weight: float = 0.25,
                min_weight: float = 0.02, iterations: int = 500,
                shrinkage: float = 0.15) -> Weights:
    """Equal risk contribution weights.

    Each position contributes the same share of total portfolio variance.
    Solved iteratively: repeatedly nudge weights toward the inverse of their
    current marginal risk contribution. Converges for long-only, fully invested
    portfolios and is simple enough to verify by inspection.
    """
    if returns is None or returns.shape[1] < 2:
        n = max(returns.shape[1], 1) if returns is not None else 1
        cols = list(returns.columns) if returns is not None else ["single"]
        return Weights(pd.Series([1.0 / n] * n, index=cols), "equal (too few assets)")

    r = returns.dropna(axis=1, how="all").dropna()
    if r.shape[1] < 2 or len(r) < 20:
        cols = list(r.columns) if r.shape[1] else list(returns.columns)
        n = max(len(cols), 1)
        return Weights(pd.Series([1.0 / n] * n, index=cols), "equal (insufficient history)",
                       notes=[f"Only {len(r)} observations across {r.shape[1]} assets — "
                              "covariance would be noise. Falling back to equal weight."])

    cov = _covariance(r, shrinkage=shrinkage)
    n = cov.shape[0]
    w = np.full(n, 1.0 / n)

    for _ in range(iterations):
        port_var = float(w @ cov @ w)
        if port_var <= 0:
            break
        marginal = cov @ w                     # d(variance)/d(weight)
        contrib = w * marginal                 # risk contribution per asset
        target = port_var / n                  # what each should contribute
        adjust = np.divide(target, contrib, out=np.ones_like(contrib),
                           where=contrib > 1e-12)
        w = w * np.power(adjust, 0.10)         # damped step, avoids oscillation
        w = np.clip(w, min_weight, max_weight)
        s = w.sum()
        if s <= 0:
            break
        w = w / s

    weights = pd.Series(w, index=r.columns)

    port_var = float(w @ cov @ w)
    contrib = pd.Series((w * (cov @ w)) / port_var if port_var > 0 else w,
                        index=r.columns)

    # Diversification ratio: weighted average volatility over portfolio
    # volatility. Higher is better; 1.0 means no diversification benefit.
    vols = np.sqrt(np.diag(cov))
    div_ratio = float((w @ vols) / np.sqrt(port_var)) if port_var > 0 else 1.0

    corr = r.corr().to_numpy()
    off = corr[~np.eye(len(corr), dtype=bool)]
    rho = float(np.nanmean(off))
    eff_n = len(w) / (1 + (len(w) - 1) * rho) if (1 + (len(w) - 1) * rho) > 0 else len(w)

    notes = []
    spread = float(contrib.max() - contrib.min())
    if spread > 0.05:
        notes.append(
            f"Risk contributions still span {spread:.1%} — the weight caps "
            "prevented full equalisation. That is usually the right trade: "
            "unconstrained risk parity can concentrate heavily in whichever "
            "asset happens to look quietest.")
    if eff_n < len(w) * 0.5:
        notes.append(
            f"{len(w)} positions behave as {eff_n:.1f} independent bets "
            f"(mean correlation {rho:.2f}). Risk parity improves the weighting "
            "but cannot manufacture diversification that is not there.")

    return Weights(weights, "risk_parity", contrib, round(div_ratio, 3),
                   round(eff_n, 2), notes)


def inverse_volatility(returns: pd.DataFrame, *, max_weight: float = 0.25,
                       min_weight: float = 0.02) -> Weights:
    """Weight by inverse volatility — risk parity's simpler cousin.

    Ignores correlation entirely, which makes it robust when the covariance
    estimate is unreliable. With few observations this is often the better
    choice, because a bad covariance matrix does more damage than ignoring
    correlation altogether.
    """
    if returns is None or returns.empty:
        return Weights(pd.Series(dtype=float), "none")

    r = returns.dropna(axis=1, how="all").dropna()
    vol = r.std(ddof=1)
    vol = vol.replace(0, np.nan).dropna()
    if vol.empty:
        n = max(r.shape[1], 1)
        return Weights(pd.Series([1.0 / n] * n, index=r.columns), "equal (zero vol)")

    w = (1.0 / vol)
    w = w / w.sum()
    w = w.clip(min_weight, max_weight)
    w = w / w.sum()
    return Weights(w, "inverse_volatility",
                   notes=["Ignores correlation by design — more robust than "
                          "full risk parity when observations are few."])


def compare_schemes(returns: pd.DataFrame) -> pd.DataFrame:
    """Equal weight vs inverse volatility vs risk parity, side by side."""
    if returns is None or returns.shape[1] < 2:
        return pd.DataFrame()
    r = returns.dropna(axis=1, how="all").dropna()
    if r.shape[1] < 2:
        return pd.DataFrame()

    n = r.shape[1]
    schemes = {
        "equal": pd.Series([1.0 / n] * n, index=r.columns),
        "inv_vol": inverse_volatility(r).weights,
        "risk_parity": risk_parity(r).weights,
    }

    cov = _covariance(r)
    rows = []
    for name, w in schemes.items():
        wv = w.reindex(r.columns).fillna(0).to_numpy()
        var = float(wv @ cov @ wv)
        vols = np.sqrt(np.diag(cov))
        contrib = (wv * (cov @ wv)) / var if var > 0 else wv
        rows.append({
            "scheme": name,
            "portfolio_vol_ann_pct": round(float(np.sqrt(var * 252)) * 100, 2),
            "max_weight_pct": round(float(w.max()) * 100, 1),
            "min_weight_pct": round(float(w.min()) * 100, 1),
            "max_risk_contrib_pct": round(float(contrib.max()) * 100, 1),
            "risk_concentration": round(float(contrib.max() / contrib.mean()), 2),
            "diversification_ratio": round(float((wv @ vols) / np.sqrt(var)), 3)
                                      if var > 0 else None,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Trailing stops
# --------------------------------------------------------------------------
@dataclass
class StopState:
    ticker: str
    entry: float
    initial_stop: float
    current_stop: float
    highest_close: float
    stop_type: str
    distance_pct: float
    locked_in_pct: float
    triggered: bool = False
    note: str = ""


def trailing_stop(entry: float, closes: pd.Series, atr: pd.Series,
                  *, atr_mult: float = 2.5, activate_at_r: float = 1.0,
                  ticker: str = "") -> StopState:
    """ATR trailing stop that ratchets up and never retreats.

    Behaviour: the stop starts fixed at entry minus atr_mult x ATR. Once the
    position is up by `activate_at_r` risk units, the stop begins trailing the
    highest close by the same ATR multiple.

    The delayed activation matters. Trailing from the first bar stops out
    ordinary noise before a position has established anything. Waiting until
    1R of profit exists means the trail only ever protects gains that are
    actually there.
    """
    if closes is None or closes.empty or atr is None or atr.empty:
        return StopState(ticker, entry, entry, entry, entry, "none", 0, 0,
                         note="No price data.")

    initial_risk = float(atr.iloc[0]) * atr_mult
    initial_stop = entry - initial_risk
    stop = initial_stop
    high_close = entry
    activated = False

    for i in range(len(closes)):
        c = float(closes.iloc[i])
        a = float(atr.iloc[min(i, len(atr) - 1)])

        if c > high_close:
            high_close = c

        # Activate the trail only once the position has earned it
        if not activated and initial_risk > 0:
            if (high_close - entry) / initial_risk >= activate_at_r:
                activated = True

        if activated:
            candidate = high_close - a * atr_mult
            stop = max(stop, candidate)        # ratchet: never lower the stop

        if c <= stop:
            return StopState(
                ticker, entry, initial_stop, stop, high_close,
                "trailing" if activated else "initial",
                round((stop / entry - 1) * 100, 2),
                round((stop / entry - 1) * 100, 2), True,
                f"Stopped out at {stop:.2f} on bar {i + 1}.")

    return StopState(
        ticker, entry, initial_stop, stop, high_close,
        "trailing" if activated else "initial",
        round((stop / float(closes.iloc[-1]) - 1) * 100, 2),
        round((stop / entry - 1) * 100, 2), False,
        ("Trailing — profit locked in." if activated
         else f"Initial stop still in force; needs {activate_at_r}R to activate."))


def compare_stop_types(entry: float, closes: pd.Series, atr: pd.Series,
                       *, atr_mult: float = 2.5) -> pd.DataFrame:
    """Fixed versus trailing, on the same price path."""
    rows = []
    initial_risk = float(atr.iloc[0]) * atr_mult
    fixed_stop = entry - initial_risk

    hit = closes[closes <= fixed_stop]
    fixed_exit = float(hit.iloc[0]) if len(hit) else float(closes.iloc[-1])
    rows.append({
        "stop_type": "fixed",
        "exit_price": round(fixed_exit, 2),
        "return_pct": round((fixed_exit / entry - 1) * 100, 2),
        "stopped": bool(len(hit)),
    })

    t = trailing_stop(entry, closes, atr, atr_mult=atr_mult)
    trail_exit = t.current_stop if t.triggered else float(closes.iloc[-1])
    rows.append({
        "stop_type": "trailing",
        "exit_price": round(trail_exit, 2),
        "return_pct": round((trail_exit / entry - 1) * 100, 2),
        "stopped": t.triggered,
    })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Rebalancing
# --------------------------------------------------------------------------
@dataclass
class RebalanceAction:
    ticker: str
    action: str                 # trim | add | hold
    current_weight: float
    target_weight: float
    drift_pp: float
    value_change: float
    reason: str


def check_rebalance(positions: pd.DataFrame, *, target_weights: pd.Series | None = None,
                    max_weight: float = MAX_POSITION_WEIGHT,
                    drift_tolerance: float = DRIFT_TOLERANCE,
                    min_trade: float = MIN_TRADE_VALUE) -> list[RebalanceAction]:
    """Which positions need trimming or topping up.

    Args:
        positions: needs `ticker` and `value` columns.
        target_weights: from risk_parity() or similar. Equal weight if absent.

    Threshold-based rather than calendar-based: rebalancing on a schedule
    creates turnover for its own sake, and every rebalance costs a full round
    trip against a thin edge.
    """
    if positions is None or positions.empty or "value" not in positions.columns:
        return []

    total = float(positions["value"].sum())
    if total <= 0:
        return []

    actions: list[RebalanceAction] = []
    n = len(positions)

    for _, row in positions.iterrows():
        tkr = str(row["ticker"])
        val = float(row["value"])
        cur_w = val / total

        if target_weights is not None and tkr in target_weights.index:
            tgt_w = float(target_weights[tkr])
        else:
            tgt_w = 1.0 / n

        # Hard cap always applies, whatever the target says
        capped = min(tgt_w, max_weight)
        drift = cur_w - capped
        change = -drift * total

        if cur_w > max_weight:
            actions.append(RebalanceAction(
                tkr, "trim", round(cur_w, 4), round(max_weight, 4),
                round(drift * 100, 2), round((max_weight - cur_w) * total, 0),
                f"Position is {cur_w:.1%} of equity, above the {max_weight:.0%} "
                "cap. A winner that grows unchecked becomes concentrated "
                "single-stock risk."))
        elif abs(drift) > drift_tolerance and abs(change) >= min_trade:
            actions.append(RebalanceAction(
                tkr, "trim" if drift > 0 else "add",
                round(cur_w, 4), round(capped, 4),
                round(drift * 100, 2), round(change, 0),
                f"Drifted {drift*100:+.1f}pp from target — beyond the "
                f"{drift_tolerance:.0%} tolerance."))
        else:
            actions.append(RebalanceAction(
                tkr, "hold", round(cur_w, 4), round(capped, 4),
                round(drift * 100, 2), 0.0,
                "Within tolerance. Every rebalance costs a round trip; do not "
                "trade for tidiness."))

    return actions


def rebalance_summary(actions: list[RebalanceAction]) -> dict:
    """Aggregate view, including the cost of acting."""
    if not actions:
        return {"actions": 0, "note": "No positions to assess."}

    trims = [a for a in actions if a.action == "trim"]
    adds = [a for a in actions if a.action == "add"]
    turnover = sum(abs(a.value_change) for a in actions if a.action != "hold")

    return {
        "positions": len(actions),
        "trim": len(trims),
        "add": len(adds),
        "hold": len(actions) - len(trims) - len(adds),
        "turnover_value": round(turnover, 0),
        "est_cost": round(turnover * 0.0036, 0),      # ~0.36% large-cap round trip
        "largest_overweight_pp": (round(max((a.drift_pp for a in trims), default=0), 2)),
        "note": ("No action needed." if not (trims or adds) else
                 f"{len(trims)} to trim, {len(adds)} to add. Estimated cost "
                 f"₹{turnover * 0.0036:,.0f} — weigh that against the "
                 "concentration being corrected."),
    }
