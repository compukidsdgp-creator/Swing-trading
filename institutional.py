"""Institutional validation — out-of-sample, Monte Carlo, standard risk metrics.

Three gaps this closes
----------------------
Audited against the standard quantitative workflow, three genuine
methodological gaps stood out. All three are about *confidence in the result*
rather than the result itself, which is why they matter.

**1. No out-of-sample test.** The signal was selected from twelve candidates
and validated on the whole five-year sample. Every parameter — horizon, floor,
tier bands — was chosen with the full period visible. The standard discipline
is to fit on one period and test on another that was never looked at. Without
it there is no way to distinguish a real effect from a well-fitted one.

**2. No Monte Carlo.** A single equity curve is one realisation of a random
process. Scrambling the trade sequence produces a distribution of outcomes,
which answers the question that actually matters before committing capital:
how bad can the drawdown get if the same trades arrive in a different order?
The observed maximum drawdown is almost always optimistic.

**3. No standard risk metrics.** Sharpe, Sortino, Calmar, VaR and CVaR are the
common language of performance. Reporting expectancy in R units alone makes the
strategy incomparable to anything else.

A note on out-of-sample honesty
-------------------------------
An out-of-sample test is only out-of-sample once. Looking at the held-out
period, adjusting, and re-testing converts it into in-sample data and destroys
its value. This module records how many times each split has been evaluated,
because the discipline is easy to lose by accident.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

OOS_LEDGER = Path("oos_evaluations.json")

TRADING_DAYS = 252
RISK_FREE_ANNUAL = 0.065          # ~6.5%, roughly the Indian 10-year


# --------------------------------------------------------------------------
# 1. Out-of-sample splitting
# --------------------------------------------------------------------------
@dataclass
class SplitResult:
    in_sample: dict
    out_of_sample: dict
    degradation_pct: float | None
    verdict: str
    message: str
    evaluations_of_this_split: int = 0


def split_dates(frames: dict[str, pd.DataFrame], *, oos_fraction: float = 0.3
                ) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Chronological split point. Never random — that would leak the future."""
    all_dates = sorted(set().union(*(set(df.index) for df in frames.values()
                                     if df is not None and not df.empty)))
    cal = pd.DatetimeIndex(all_dates)
    cut = cal[int(len(cal) * (1 - oos_fraction))]
    return cal[0], cut, cal[-1]


def _record_evaluation(split_id: str) -> int:
    """Count how many times a split has been looked at.

    An out-of-sample test stops being out-of-sample the moment it informs a
    change. Counting makes that visible rather than letting it happen quietly.
    """
    ledger = {}
    if OOS_LEDGER.exists():
        try:
            ledger = json.loads(OOS_LEDGER.read_text())
        except json.JSONDecodeError:
            ledger = {}
    entry = ledger.get(split_id, {"count": 0, "first": None, "last": None})
    entry["count"] += 1
    entry["first"] = entry["first"] or dt.date.today().isoformat()
    entry["last"] = dt.date.today().isoformat()
    ledger[split_id] = entry
    OOS_LEDGER.write_text(json.dumps(ledger, indent=2))
    return entry["count"]


def out_of_sample_test(
    frames: dict[str, pd.DataFrame],
    bench: pd.DataFrame | None,
    *,
    horizon: int = 15,
    model: str = "momentum",
    oos_fraction: float = 0.3,
    record: bool = True,
) -> SplitResult:
    """Fit-period versus held-out-period comparison.

    Momentum has no fitted parameters, so 'in-sample' here means the period
    over which the signal was selected and the horizon chosen. That selection
    is itself a form of fitting, and this measures how much of the result
    survives outside it.
    """
    import validate as val

    start, cut, end = split_dates(frames, oos_fraction=oos_fraction)
    split_id = f"{start.date()}|{cut.date()}|{end.date()}|h{horizon}|{model}"
    n_evals = _record_evaluation(split_id) if record else 0

    is_frames = {t: df[df.index <= cut] for t, df in frames.items()}
    oos_frames = {t: df[df.index > cut] for t, df in frames.items()}
    is_bench = bench[bench.index <= cut] if bench is not None else None
    oos_bench = bench[bench.index > cut] if bench is not None else None

    # The out-of-sample slice must still contain enough history for the signal
    # to be computable at all — 12-1 momentum needs 252 bars before the first
    # observation, so the frames are extended backwards for computation while
    # only the post-cut window is measured.
    oos_ext = {t: df[df.index > cut - pd.Timedelta(days=560)]
               for t, df in frames.items()}
    oos_ext_bench = (bench[bench.index > cut - pd.Timedelta(days=560)]
                     if bench is not None else None)

    is_res = val.run_ic(is_frames, is_bench, horizon=horizon,
                        n_permutations=200, model=model)
    oos_res = val.run_ic(oos_ext, oos_ext_bench, horizon=horizon,
                         n_permutations=200, model=model)

    is_s, oos_s = is_res.summary, oos_res.summary
    if "error" in is_s or "error" in oos_s:
        return SplitResult(is_s, oos_s, None, "error",
                           f"Could not complete: IS={is_s.get('error')} "
                           f"OOS={oos_s.get('error')}", n_evals)

    is_ic = is_s.get("mean_ic", 0) or 0
    oos_ic = oos_s.get("mean_ic", 0) or 0
    deg = (1 - oos_ic / is_ic) * 100 if is_ic else None

    if oos_ic <= 0:
        verdict, msg = "fail", (
            f"In-sample IC {is_ic:+.4f}, out-of-sample {oos_ic:+.4f}. The signal "
            "does not survive outside the period it was selected on. This is the "
            "signature of a fitted result, and it is the reason out-of-sample "
            "testing exists.")
    elif deg is not None and deg > 70:
        verdict, msg = "fail", (
            f"Out-of-sample IC {oos_ic:+.4f} retains only {100-deg:.0f}% of the "
            f"in-sample {is_ic:+.4f}. Decay this severe suggests the result was "
            "largely selection.")
    elif deg is not None and deg > 40:
        verdict, msg = "warn", (
            f"Out-of-sample IC {oos_ic:+.4f}, {deg:.0f}% below in-sample "
            f"{is_ic:+.4f}. Meaningful decay. Some edge survives but it is "
            "thinner than the headline figure suggests.")
    else:
        verdict, msg = "pass", (
            f"Out-of-sample IC {oos_ic:+.4f} against in-sample {is_ic:+.4f} "
            f"({100-(deg or 0):.0f}% retained). The signal holds outside its "
            "selection period.")

    if n_evals > 1:
        msg += (f"\n\nNOTE: this split has now been evaluated {n_evals} times. "
                "It is no longer genuinely out-of-sample — each look leaks "
                "information into subsequent decisions.")

    return SplitResult(is_s, oos_s, round(deg, 1) if deg is not None else None,
                       verdict, msg, n_evals)


# --------------------------------------------------------------------------
# 2. Monte Carlo
# --------------------------------------------------------------------------
@dataclass
class MonteCarloResult:
    n_simulations: int
    observed_max_drawdown: float
    drawdown_percentiles: dict
    final_equity_percentiles: dict
    prob_loss: float
    prob_ruin: float
    worst_case: float
    verdict: str
    message: str
    curves: np.ndarray | None = field(default=None, repr=False)


def monte_carlo(
    returns_r: pd.Series | np.ndarray,
    *,
    n_sims: int = 5000,
    risk_per_trade_pct: float = 1.0,
    start_capital: float = 500_000,
    ruin_threshold: float = 0.50,
    seed: int = 0,
) -> MonteCarloResult:
    """Scramble the trade sequence to get a distribution of outcomes.

    The observed equity curve is one draw. Its maximum drawdown is whatever
    order the trades happened to arrive in — and that order was luck. Shuffling
    thousands of times gives the distribution, and the 95th percentile drawdown
    is a far better planning figure than the one that actually occurred.

    The trades themselves are held fixed; only their order changes. This
    isolates sequence risk from selection risk.
    """
    r = np.asarray(pd.Series(returns_r).dropna(), dtype=float)
    n = len(r)
    if n < 10:
        return MonteCarloResult(0, 0.0, {}, {}, 0.0, 0.0, 0.0, "insufficient",
                                f"Only {n} trades. At least 10 are needed, and "
                                "several hundred before the distribution means "
                                "much.")

    rng = np.random.default_rng(seed)
    risk = risk_per_trade_pct / 100.0

    max_dds = np.empty(n_sims)
    finals = np.empty(n_sims)
    keep = min(200, n_sims)
    curves = np.empty((keep, n + 1))

    for i in range(n_sims):
        seq = rng.permutation(r)
        eq = start_capital * np.cumprod(1 + risk * seq)
        eq = np.concatenate([[start_capital], eq])
        peak = np.maximum.accumulate(eq)
        dd = (eq / peak - 1).min()
        max_dds[i] = dd
        finals[i] = eq[-1]
        if i < keep:
            curves[i] = eq

    # The drawdown of the observed ordering, for comparison
    eq_obs = start_capital * np.cumprod(1 + risk * r)
    eq_obs = np.concatenate([[start_capital], eq_obs])
    obs_dd = float((eq_obs / np.maximum.accumulate(eq_obs) - 1).min())

    dd_pct = {f"p{p}": round(float(np.percentile(max_dds, 100 - p)) * 100, 2)
              for p in (50, 75, 90, 95, 99)}
    eq_pct = {f"p{p}": round(float(np.percentile(finals, p)), 0)
              for p in (5, 25, 50, 75, 95)}

    prob_loss = float((finals < start_capital).mean())
    prob_ruin = float((max_dds < -ruin_threshold).mean())
    worst = float(max_dds.min())

    if prob_ruin > 0.05:
        verdict, msg = "bad", (
            f"{prob_ruin:.1%} of simulations breached a {ruin_threshold:.0%} "
            f"drawdown. At {risk_per_trade_pct}% risk per trade this sizing is "
            "too aggressive for the return distribution — reduce it.")
    elif dd_pct["p95"] < -30:
        verdict, msg = "warn", (
            f"95th-percentile drawdown is {dd_pct['p95']:.1f}%. The observed "
            f"{obs_dd*100:.1f}% was a favourable ordering. Plan for the former, "
            "not the latter.")
    elif prob_loss > 0.35:
        verdict, msg = "warn", (
            f"{prob_loss:.1%} of sequences ended below starting capital. The "
            "edge is real but thin enough that ordering luck dominates over "
            f"{n} trades.")
    else:
        verdict, msg = "ok", (
            f"95th-percentile drawdown {dd_pct['p95']:.1f}%, "
            f"{prob_loss:.1%} of sequences ended down. Sizing looks proportionate "
            "to the distribution.")

    msg += (f"\n\nObserved drawdown was {obs_dd*100:.1f}% — the median simulated "
            f"outcome is {dd_pct['p50']:.1f}%, so the realised path was "
            f"{'luckier' if obs_dd > dd_pct['p50']/100 else 'harsher'} than typical.")

    return MonteCarloResult(n_sims, round(obs_dd * 100, 2), dd_pct, eq_pct,
                            round(prob_loss, 4), round(prob_ruin, 4),
                            round(worst * 100, 2), verdict, msg, curves)


# --------------------------------------------------------------------------
# 3. Standard risk metrics
# --------------------------------------------------------------------------
def risk_metrics(returns: pd.Series | np.ndarray, *,
                 periods_per_year: float = TRADING_DAYS / 15,
                 risk_free: float = RISK_FREE_ANNUAL,
                 capital: float = 500_000,
                 risk_per_trade_pct: float = 1.0) -> dict:
    """Sharpe, Sortino, Calmar, VaR, CVaR — the common language of performance.

    Args:
        returns: per-trade returns in R units.
        periods_per_year: trading cycles per year (252/horizon for a swing book).
    """
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if len(r) < 5:
        return {"error": f"only {len(r)} observations"}

    # Convert R units to fractional capital returns
    pct = r * (risk_per_trade_pct / 100.0)

    mean_p = float(pct.mean())
    sd_p = float(pct.std(ddof=1))
    ann_ret = mean_p * periods_per_year
    ann_vol = sd_p * np.sqrt(periods_per_year)

    downside = pct[pct < 0]
    dd_dev = (float(np.sqrt((downside ** 2).mean())) * np.sqrt(periods_per_year)
              if len(downside) else 0.0)

    eq = capital * np.cumprod(1 + pct)
    eq = np.concatenate([[capital], eq])
    peak = np.maximum.accumulate(eq)
    max_dd = float((eq / peak - 1).min())

    wins, losses = pct[pct > 0], pct[pct <= 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) else 0.0

    return {
        "n_observations": len(r),
        "mean_r": round(float(r.mean()), 4),
        "annualised_return_pct": round(ann_ret * 100, 2),
        "annualised_vol_pct": round(ann_vol * 100, 2),
        "sharpe": round((ann_ret - risk_free) / ann_vol, 3) if ann_vol > 0 else None,
        "sortino": round((ann_ret - risk_free) / dd_dev, 3) if dd_dev > 0 else None,
        "calmar": round(ann_ret / abs(max_dd), 3) if max_dd < 0 else None,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "var_95_pct": round(float(np.percentile(pct, 5)) * 100, 3),
        "cvar_95_pct": round(float(pct[pct <= np.percentile(pct, 5)].mean()) * 100, 3),
        "win_rate_pct": round(float((r > 0).mean()) * 100, 1),
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else None,
        "expectancy_r": round(float(r.mean()), 4),
        "skew": round(float(pd.Series(r).skew()), 3),
        "kurtosis": round(float(pd.Series(r).kurtosis()), 3),
    }


def interpret_metrics(m: dict) -> list[str]:
    """Plain-language reading of the standard metrics."""
    if "error" in m:
        return [m["error"]]

    notes = []
    s = m.get("sharpe")
    if s is not None:
        if s < 0:
            notes.append(f"Sharpe {s:.2f} — the strategy underperforms the "
                         "risk-free rate. Cash is better.")
        elif s < 0.5:
            notes.append(f"Sharpe {s:.2f} — weak. Below roughly 0.5 the return "
                         "does not justify the volatility for most investors.")
        elif s < 1.0:
            notes.append(f"Sharpe {s:.2f} — respectable for a long-only equity "
                         "strategy.")
        else:
            notes.append(f"Sharpe {s:.2f} — strong. Verify it is not an artefact "
                         "of a short sample or a single favourable regime.")

    # Only meaningful when both are positive. A ratio of two negative numbers
    # can exceed 1.5 while Sortino is the *worse* figure, which produced a
    # nonsensical "well above Sharpe" reading during testing.
    sor = m.get("sortino")
    if sor is not None and s is not None and sor > 0 and s > 0:
        if sor / s > 1.5:
            notes.append(f"Sortino {sor:.2f} well above Sharpe {s:.2f} — "
                         "volatility is mostly upside, which is the good kind.")
    elif sor is not None and sor < 0:
        notes.append(f"Sortino {sor:.2f} is negative — downside volatility "
                     "exceeds excess return. Not a viable risk profile.")

    if m.get("skew") is not None and m["skew"] < -0.5:
        notes.append(f"Negative skew ({m['skew']:.2f}) — occasional large losses "
                     "against frequent small gains. Characteristic of momentum, "
                     "and the reason crash protection matters.")

    if m.get("kurtosis") is not None and m["kurtosis"] > 3:
        notes.append(f"Excess kurtosis ({m['kurtosis']:.2f}) — fat tails. Extreme "
                     "outcomes are more likely than a normal distribution implies, "
                     "so VaR understates the real risk.")

    if m.get("cvar_95_pct") is not None:
        notes.append(f"CVaR(95%) {m['cvar_95_pct']:.2f}% — the average loss in the "
                     "worst 5% of trades. This is the number to size against, not VaR.")

    pf = m.get("profit_factor")
    if pf is not None:
        if pf < 1.0:
            notes.append(f"Profit factor {pf:.2f} — gross losses exceed gross gains.")
        elif pf < 1.3:
            notes.append(f"Profit factor {pf:.2f} — thin. Small changes in costs "
                         "or slippage could flip it.")

    return notes
