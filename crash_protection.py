"""Momentum crash protection — volatility scaling.

The problem this solves
-----------------------
Momentum does not fail gently. It produces steady returns for years, then loses
30-60% in weeks. The mechanism is documented: after a major market decline, past
losers acquire very high beta and past winners very low beta, so a momentum book
carries a large negative conditional beta. When the market rebounds sharply, the
strategy is positioned exactly wrong.

The regime gate in this system watches the *index* — whether the Nifty is above
its 200 DMA. That is useful but it is not the same thing. Momentum crashes are
predicted by **momentum's own realised volatility**, not the market's, and the
two diverge precisely when it matters.

The fix
-------
Barroso and Santa-Clara (2015), "Momentum Has Its Moments": scale exposure by
the inverse of the strategy's own recent realised volatility, targeting a
constant risk level. Their finding — risk management of this kind virtually
eliminated the crashes and roughly doubled the Sharpe ratio (0.97 against 0.53
unmanaged).

Later work refined it. Daniel and Moskowitz (2016) scale on forecasts of both
mean and variance. Bongaerts, Kang and van Dijk (2020) adjust only in extreme
volatility states, leaving exposure unscaled otherwise — which also reduces
turnover materially. Comparative work has generally found the simpler constant
scaling competitive with, and often superior to, the dynamic version.

This module implements constant scaling with an optional conditional mode.

On the target
-------------
Barroso and Santa-Clara used 12% annualised, and were criticised for offering no
justification — the implied risk preference is arbitrary. The default here is
also 12% for comparability, but it is exposed as a parameter because the right
value depends on your tolerance, not on a paper.

Important limitation
--------------------
This scales a *portfolio*, using the realised volatility of the momentum
strategy's own returns. That series is only observable once you have been
running the strategy. Before then, the module falls back to a proxy built from
the cross-sectional dispersion of your current holdings, which is correlated
with but not identical to true strategy volatility. Treat early readings as
approximate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET_VOL_ANNUAL = 0.12          # Barroso & Santa-Clara's choice; arbitrary
VOL_LOOKBACK_DAYS = 126           # ~6 months of daily returns
MIN_OBS = 40
MAX_LEVERAGE = 1.5                # never lever beyond this, whatever the maths says
MIN_EXPOSURE = 0.10               # never go fully flat on a scaling signal alone

# Conditional mode thresholds — scale only in extreme states, leave the middle
# alone. Reduces turnover substantially relative to always-on scaling.
HIGH_VOL_PERCENTILE = 0.80
LOW_VOL_PERCENTILE = 0.20


@dataclass
class ScalingResult:
    realised_vol: float
    target_vol: float
    raw_scale: float
    applied_scale: float
    exposure_pct: float
    mode: str
    state: str
    capped: bool = False
    n_observations: int = 0
    note: str = ""

    @property
    def is_defensive(self) -> bool:
        return self.applied_scale < 0.75


def realised_volatility(returns: pd.Series, *, lookback: int = VOL_LOOKBACK_DAYS,
                        annualise: bool = True) -> float:
    """Realised volatility of a return series, annualised by default."""
    if returns is None or len(returns) < MIN_OBS:
        return float("nan")
    r = pd.to_numeric(returns, errors="coerce").dropna().tail(lookback)
    if len(r) < MIN_OBS:
        return float("nan")
    sd = float(r.std(ddof=1))
    return sd * np.sqrt(252) if annualise else sd


def strategy_returns_from_log(log: pd.DataFrame) -> pd.Series:
    """Approximate daily strategy returns from the forward log.

    Each evaluated pick contributes its return spread over its holding period.
    This is coarse — it assumes equal weighting and ignores overlap — but it is
    the only strategy-level series available before live trading begins.
    """
    if log is None or log.empty:
        return pd.Series(dtype=float)
    ev = log[log["status"] == "evaluated"].copy()
    if ev.empty:
        return pd.Series(dtype=float)

    ev["fwd_return_pct"] = pd.to_numeric(ev["fwd_return_pct"], errors="coerce")
    ev = ev.dropna(subset=["fwd_return_pct"])
    if ev.empty:
        return pd.Series(dtype=float)

    # Mean return per snapshot, converted to a per-day figure
    grp = ev.groupby("snapshot_date").agg(
        ret=("fwd_return_pct", "mean"),
        days=("holding_days_actual", "mean"),
    )
    grp["days"] = pd.to_numeric(grp["days"], errors="coerce").fillna(15).clip(lower=1)
    daily = (grp["ret"] / 100.0) / grp["days"]
    daily.index = pd.to_datetime(grp.index, errors="coerce")
    return daily.dropna().sort_index()


def proxy_returns_from_holdings(frames: dict[str, pd.DataFrame],
                                lookback: int = VOL_LOOKBACK_DAYS) -> pd.Series:
    """Fallback: equal-weight daily returns of the current basket.

    Used before a real strategy return series exists. It captures the basket's
    volatility, which correlates with — but understates — the volatility of a
    long-short momentum book, since it omits the short leg entirely.
    """
    series = []
    for t, df in (frames or {}).items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        r = df["Close"].tail(lookback + 1).pct_change().dropna()
        if len(r) >= MIN_OBS:
            series.append(r)
    if not series:
        return pd.Series(dtype=float)
    return pd.concat(series, axis=1).mean(axis=1).dropna()


def compute_scale(
    returns: pd.Series,
    *,
    target_vol: float = TARGET_VOL_ANNUAL,
    lookback: int = VOL_LOOKBACK_DAYS,
    mode: str = "constant",
    history: pd.Series | None = None,
) -> ScalingResult:
    """Exposure multiplier from realised volatility.

    mode:
      'constant'    — always scale to the target (Barroso & Santa-Clara).
      'conditional' — scale only in extreme volatility states, leave the middle
                      unscaled. Lower turnover, per Bongaerts et al. (2020).
                      Needs 250+ observations of history; below that the
                      percentile is unstable and it falls back to constant.

    Default is 'constant'. Comparative studies have generally found the simpler
    constant scaling competitive with or superior to dynamic variants, and it
    has no history requirement.
    """
    n = 0 if returns is None else len(returns.dropna())
    rv = realised_volatility(returns, lookback=lookback)

    if not np.isfinite(rv) or rv <= 1e-9:
        return ScalingResult(
            realised_vol=float("nan"), target_vol=target_vol, raw_scale=1.0,
            applied_scale=1.0, exposure_pct=100.0, mode=mode, state="unknown",
            n_observations=n,
            note=(f"Insufficient return history ({n} observations, need {MIN_OBS}). "
                  "No scaling applied — this is the correct default, not a signal "
                  "to size up."),
        )

    raw = target_vol / rv
    state = "normal"

    if mode == "conditional" and history is not None and len(history.dropna()) >= 250:
        # Rolling realised vol, to locate the current reading within its own
        # history. Requires a long history: with fewer than ~250 observations
        # the percentile is unstable and will misclassify normal volatility as
        # extreme, which defeats the purpose of leaving the middle alone.
        h = pd.to_numeric(history, errors="coerce").dropna()
        roll = (h.rolling(lookback).std(ddof=1) * np.sqrt(252)).dropna()
        if len(roll) >= 60:
            pct = float((roll <= rv).mean())
            if pct >= HIGH_VOL_PERCENTILE:
                state = "high_vol"
            elif pct <= LOW_VOL_PERCENTILE:
                state = "low_vol"
            else:
                state = "normal"
                raw = 1.0            # leave the middle alone
        else:
            state = "insufficient_history"
    elif mode == "conditional":
        state = "insufficient_history"

    capped = False
    applied = raw
    if applied > MAX_LEVERAGE:
        applied, capped = MAX_LEVERAGE, True
    if applied < MIN_EXPOSURE:
        applied, capped = MIN_EXPOSURE, True

    if applied < 0.75:
        note = (f"Realised volatility {rv:.1%} is well above the {target_vol:.0%} "
                f"target — exposure cut to {applied:.0%}. Elevated momentum "
                "volatility is the documented precursor to momentum crashes.")
    elif applied > 1.15:
        note = (f"Realised volatility {rv:.1%} is below the {target_vol:.0%} target. "
                f"Scaling to {applied:.0%} increases exposure — only do this with "
                "capital you can afford to have leveraged into a reversal.")
    else:
        note = f"Realised volatility {rv:.1%} is close to target; exposure roughly unchanged."

    return ScalingResult(
        realised_vol=round(rv, 4), target_vol=target_vol,
        raw_scale=round(raw, 3), applied_scale=round(applied, 3),
        exposure_pct=round(applied * 100, 1), mode=mode, state=state,
        capped=capped, n_observations=n, note=note,
    )


def apply_to_positions(picks: pd.DataFrame, scale: ScalingResult,
                       *, qty_col: str = "Qty") -> pd.DataFrame:
    """Scale a position-size column by the volatility multiplier."""
    if picks is None or picks.empty:
        return picks
    out = picks.copy()
    if qty_col in out.columns:
        out[f"{qty_col}_scaled"] = (pd.to_numeric(out[qty_col], errors="coerce")
                                    * scale.applied_scale).round().astype("Int64")
    out["vol_scale"] = scale.applied_scale
    return out


def crash_risk_indicator(bench: pd.DataFrame, *, lookback: int = 126) -> dict:
    """Market-side conditions historically associated with momentum crashes.

    Two documented ingredients: a preceding severe drawdown, and elevated
    volatility. The dangerous configuration is a rebound out of a deep decline —
    past losers have very high beta at that point and rally hardest, which is
    precisely when a long-winners book suffers.
    """
    if bench is None or len(bench) < lookback + 20:
        return {"available": False}

    c = bench["Close"]
    r = c.pct_change().dropna()
    vol = float(r.tail(lookback).std(ddof=1) * np.sqrt(252))
    long_vol = float(r.std(ddof=1) * np.sqrt(252))
    peak = float(c.tail(252).max())
    dd = float(c.iloc[-1] / peak - 1) if peak > 0 else 0.0
    rebound = float(c.iloc[-1] / c.tail(60).min() - 1) if len(c) > 60 else 0.0

    elevated = vol > long_vol * 1.3
    deep_dd = dd < -0.15
    sharp_rebound = rebound > 0.10 and deep_dd

    risk = "elevated" if (elevated and deep_dd) else "normal"
    if sharp_rebound:
        risk = "high"

    return {
        "available": True,
        "realised_vol": round(vol, 4),
        "long_run_vol": round(long_vol, 4),
        "vol_ratio": round(vol / long_vol, 2) if long_vol else None,
        "drawdown_from_52w_high": round(dd * 100, 1),
        "rebound_from_60d_low": round(rebound * 100, 1),
        "risk_state": risk,
        "note": {
            "high": ("Sharp rebound from a deep drawdown. This is the documented "
                     "momentum-crash configuration: past losers carry very high "
                     "beta here and rally hardest, while a winners book is "
                     "positioned against it. Reduce exposure."),
            "elevated": ("Elevated volatility alongside a significant drawdown — "
                         "conditions under which momentum has historically been "
                         "fragile."),
            "normal": "No crash-specific warning from market conditions.",
        }[risk],
    }
