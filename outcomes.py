"""Conditional outcome distributions — what actually happened, not what will.

Why this is not a prediction tab
--------------------------------
A tab that says "68% probability this stock rises" would be false precision.
With an IC of 0.031 the signal explains roughly 0.1% of the variance in forward
returns. The confidence interval on any such estimate spans the coin flip, and
presenting a point probability invites you to read certainty that is not there.

What this shows instead is the **conditional distribution**: for stocks that
historically scored in this decile, here is the full range of what happened
next — with the interval, the overlap against other deciles, and how much of
the result is simply market drift rather than the signal.

The most useful thing it does is make the weakness visible. Top-decile and
bottom-decile outcome distributions overlap almost entirely. Seeing that is
more informative than any single number, and harder to misread.

Three honest framings used throughout
-------------------------------------
1. **Base rate, not forecast.** "62% of similar past cases rose" is a
   historical frequency. It says nothing about this instance.

2. **Interval, never a point.** Every estimate carries its confidence interval,
   and the interval is usually wide enough to include "no edge".

3. **Excess over drift.** In a rising market most stocks rise. The number that
   matters is how much *better* than the universe average, not the raw hit
   rate — otherwise you are measuring the market and calling it skill.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import indicators as ind
import momentum as momo
from backtest import MIN_HISTORY


@dataclass
class ConditionalOutcome:
    decile: int
    n_observations: int
    mean_return: float
    median_return: float
    std_return: float
    hit_rate: float
    hit_rate_ci: tuple
    mean_ci: tuple
    percentiles: dict
    excess_over_universe: float
    excess_ci: tuple
    overlap_with_bottom: float | None = None


@dataclass
class DistributionResult:
    by_decile: pd.DataFrame
    universe_mean: float
    universe_hit_rate: float
    n_windows: int
    horizon: int
    raw_outcomes: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list[str] = field(default_factory=list)


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple:
    """Wilson interval for a proportion.

    Preferred over the normal approximation because it behaves sensibly near 0
    and 1 and at small n — exactly the conditions here.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _mean_ci(values: np.ndarray, z: float = 1.96) -> tuple:
    if len(values) < 2:
        return (np.nan, np.nan)
    se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    m = float(np.mean(values))
    return (m - z * se, m + z * se)


def _overlap(a: np.ndarray, b: np.ndarray, bins: int = 50) -> float:
    """Overlapping coefficient between two distributions, 0 to 1.

    1.0 means the distributions are identical — the signal separates nothing.
    Momentum deciles typically land above 0.85, which is the honest headline.
    """
    if len(a) < 10 or len(b) < 10:
        return np.nan
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    width = edges[1] - edges[0]
    return float(np.sum(np.minimum(ha, hb)) * width)


def build_distributions(
    frames: dict[str, pd.DataFrame],
    bench: pd.DataFrame | None = None,
    *,
    horizon: int = 30,
    n_deciles: int = 10,
    min_names: int = 30,
) -> DistributionResult:
    """Historical forward-return distribution conditional on momentum decile.

    Uses only data available at each observation point — the same no-lookahead
    discipline as validation.
    """
    enriched = {}
    for t, df in frames.items():
        if df is None or len(df) < MIN_HISTORY + horizon + 5:
            continue
        try:
            enriched[t] = ind.enrich(df)
        except Exception:                                      # noqa: BLE001
            continue
    if not enriched:
        return DistributionResult(pd.DataFrame(), 0, 0, 0, horizon,
                                  notes=["No usable data."])

    bench_e = ind.enrich(bench) if bench is not None and len(bench) > 300 else None
    cal = (pd.DatetimeIndex(bench_e.index) if bench_e is not None
           else pd.DatetimeIndex(max((e.index for e in enriched.values()), key=len)))

    step = horizon + 3          # non-overlapping, and avoids weekday clustering
    rows = []
    start = max(MIN_HISTORY, 300)

    for k in range(start, len(cal) - horizon - 1, step):
        date = cal[k]
        obs = []
        for t, e in enriched.items():
            try:
                i = e.index.get_loc(date)
            except KeyError:
                continue
            if not isinstance(i, int) or i < start or i + horizon >= len(e):
                continue
            m = momo.raw_momentum(e, i)
            if not np.isfinite(m):
                continue
            p0 = float(e["Close"].iloc[i])
            p1 = float(e["Close"].iloc[i + horizon])
            if p0 <= 0 or not np.isfinite(p1):
                continue
            obs.append({"ticker": t, "momentum": m,
                        "fwd_return": (p1 / p0 - 1) * 100})

        if len(obs) < min_names:
            continue

        d = pd.DataFrame(obs)
        d["decile"] = pd.qcut(d["momentum"].rank(method="first"),
                              n_deciles, labels=False, duplicates="drop") + 1
        d["date"] = date
        d["universe_mean"] = d["fwd_return"].mean()
        d["excess"] = d["fwd_return"] - d["universe_mean"]
        rows.append(d)

    if not rows:
        return DistributionResult(pd.DataFrame(), 0, 0, 0, horizon,
                                  notes=["Too few valid windows."])

    allobs = pd.concat(rows, ignore_index=True)
    n_windows = allobs["date"].nunique()

    uni_mean = float(allobs["fwd_return"].mean())
    uni_hit = float((allobs["fwd_return"] > 0).mean())

    bottom = allobs.loc[allobs["decile"] == 1, "fwd_return"].to_numpy()

    out = []
    for dec in sorted(allobs["decile"].dropna().unique()):
        sub = allobs[allobs["decile"] == dec]
        r = sub["fwd_return"].to_numpy()
        ex = sub["excess"].to_numpy()
        n = len(r)
        hits = int((r > 0).sum())

        out.append({
            "decile": int(dec),
            "n": n,
            "mean_return_pct": round(float(r.mean()), 3),
            "median_return_pct": round(float(np.median(r)), 3),
            "std_pct": round(float(r.std(ddof=1)), 3) if n > 1 else np.nan,
            "hit_rate_pct": round(hits / n * 100, 1),
            "hit_ci_low": round(_wilson_ci(hits, n)[0] * 100, 1),
            "hit_ci_high": round(_wilson_ci(hits, n)[1] * 100, 1),
            "excess_pct": round(float(ex.mean()), 3),
            "excess_ci_low": round(_mean_ci(ex)[0], 3),
            "excess_ci_high": round(_mean_ci(ex)[1], 3),
            "p10": round(float(np.percentile(r, 10)), 2),
            "p25": round(float(np.percentile(r, 25)), 2),
            "p75": round(float(np.percentile(r, 75)), 2),
            "p90": round(float(np.percentile(r, 90)), 2),
            "overlap_with_d1": (round(_overlap(r, bottom), 3)
                                if dec != 1 else 1.0),
        })

    df_out = pd.DataFrame(out)

    notes = [
        f"{n_windows} non-overlapping windows, {len(allobs):,} stock-observations, "
        f"{horizon}-day horizon.",
        "Hit rate is a historical frequency, not a forecast for any instance.",
    ]

    top = df_out[df_out["decile"] == df_out["decile"].max()]
    if not top.empty:
        t = top.iloc[0]
        if t["excess_ci_low"] <= 0 <= t["excess_ci_high"]:
            notes.append(
                f"Top decile excess is {t['excess_pct']:+.2f}% with a 95% interval "
                f"of [{t['excess_ci_low']:+.2f}%, {t['excess_ci_high']:+.2f}%] — "
                "the interval includes zero, so no edge is demonstrated at this "
                "sample size.")
        if pd.notna(t["overlap_with_d1"]) and t["overlap_with_d1"] > 0.8:
            notes.append(
                f"Top and bottom decile outcome distributions overlap "
                f"{t['overlap_with_d1']:.0%}. Knowing the decile shifts the odds "
                "slightly; it does not separate winners from losers.")

    return DistributionResult(df_out, round(uni_mean, 3), round(uni_hit * 100, 1),
                              n_windows, horizon, allobs, notes)


def outcome_for_score(result: DistributionResult, percentile_score: float) -> dict:
    """Historical outcomes for stocks that scored at this percentile.

    Deliberately returns a range and a caveat, never a bare probability.
    """
    if result.by_decile.empty:
        return {"error": "No distribution available."}

    dec = int(np.clip(np.ceil(percentile_score / 10), 1, 10))
    row = result.by_decile[result.by_decile["decile"] == dec]
    if row.empty:
        return {"error": f"No data for decile {dec}."}
    r = row.iloc[0]

    return {
        "decile": dec,
        "n_similar_cases": int(r["n"]),
        "rose_pct_of_time": r["hit_rate_pct"],
        "hit_rate_range": f"{r['hit_ci_low']}% – {r['hit_ci_high']}%",
        "median_outcome_pct": r["median_return_pct"],
        "typical_range_pct": f"{r['p25']}% to {r['p75']}%",
        "worst_decile_10pct": r["p10"],
        "best_decile_10pct": r["p90"],
        "excess_over_universe_pct": r["excess_pct"],
        "excess_range": f"{r['excess_ci_low']}% to {r['excess_ci_high']}%",
        "distribution_overlap_with_worst": r["overlap_with_d1"],
        "caveat": (
            "This is what happened to similar past cases. It is not a forecast. "
            f"Outcomes ranged from {r['p10']}% to {r['p90']}% across the middle "
            "80% of cases, and the interval on the excess return "
            f"({r['excess_ci_low']}% to {r['excess_ci_high']}%) is the honest "
            "measure of how little is known."
        ),
    }


def calibration_check(result: DistributionResult) -> pd.DataFrame:
    """Is the ranking monotonic in outcome, or only at the extremes?

    A well-behaved signal produces steadily rising outcomes across deciles. A
    signal that only separates the extremes — common with momentum — is still
    usable but should not be treated as a fine-grained ranking.
    """
    if result.by_decile.empty:
        return pd.DataFrame()

    d = result.by_decile.sort_values("decile")
    ex = d["excess_pct"].to_numpy()
    steps = np.diff(ex)

    return pd.DataFrame([{
        "deciles": len(d),
        "monotonic": bool((steps >= -0.05).all()),
        "rising_steps": int((steps > 0).sum()),
        "falling_steps": int((steps < 0).sum()),
        "d10_minus_d1_pct": round(float(ex[-1] - ex[0]), 3),
        "largest_single_step_pct": round(float(np.abs(steps).max()), 3),
        "interpretation": (
            "Monotonic — outcomes rise steadily with rank."
            if (steps >= -0.05).all() else
            "Not monotonic — the ranking separates extremes but is unreliable "
            "in between. Treat decile 7 vs 8 as indistinguishable."
        ),
    }])
