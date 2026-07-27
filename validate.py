"""Predictive validation — does the score actually rank forward returns?

This is a different question from the backtest. The backtest asks "would these
rules have made money?". This asks the cleaner, more fundamental question:
"does a higher score correspond to a higher forward return?"

Method
------
At each rebalance date (every N trading days over the test period):
  1. Score every stock in the universe using ONLY data up to that date.
  2. Measure each stock's actual forward return over the next N days.
  3. Compute the Spearman rank correlation between score and forward return.
     That correlation is the Information Coefficient (IC) for that window.

Aggregate the ICs and you get an honest read on predictive power.

Interpreting IC (equity cross-sectional factors)
------------------------------------------------
  IC < 0.02        no meaningful signal
  IC 0.02 - 0.04   weak but potentially tradeable after costs
  IC 0.04 - 0.06   good for a retail-accessible factor
  IC 0.06 - 0.10   strong
  IC > 0.15        suspicious — check for lookahead bias before celebrating

The t-statistic matters as much as the mean. IC of 0.05 across 20 windows is
noise; IC of 0.03 across 100 windows with t > 2 is a finding.

Permutation test
----------------
Any test produces *some* number. To know whether yours means anything, the
same pipeline is re-run with scores randomly shuffled. That gives the null
distribution. If your real IC is not clearly outside it, you have found
nothing — regardless of how good the headline number looks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import indicators as ind
import regime as rg
import tiers as tr
from backtest import get_scorer, MIN_HISTORY


@dataclass
class ICResult:
    windows: pd.DataFrame                  # per-window IC and stats
    buckets: pd.DataFrame                  # quintile forward returns
    summary: dict = field(default_factory=dict)
    null_ic: np.ndarray | None = None      # permutation distribution


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without a scipy dependency."""
    if len(a) < 5:
        return np.nan
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else np.nan


def run_ic(
    frames: dict[str, pd.DataFrame],
    bench: pd.DataFrame | None,
    *,
    horizon: int = 15,
    step: int | None = None,
    min_names: int = 15,
    n_permutations: int = 200,
    seed: int = 0,
    model: str = "momentum",
) -> ICResult:
    """Walk forward, computing an IC at each rebalance date.

    Args:
        horizon: forward return window in trading days (your 15-20 day swing).
        step: days between rebalance dates. Defaults to `horizon + 3`, NOT
              `horizon`.

              Why the +3: a step of exactly `horizon` business days lands on
              the same weekday every single time. With horizon=15, every
              rebalance was a Friday — 100% of windows. Any day-of-week effect
              would be fully baked into the result. Adding 3 rotates the
              weekday evenly across all five, and incidentally adds a small
              embargo gap between windows, which further reduces correlation
              between observations.
        min_names: skip a window if fewer than this many stocks are scoreable.
        n_permutations: size of the null distribution.
    """
    # +3 breaks the weekday lock. See the docstring above.
    step = step or (horizon + 3)
    scorer = get_scorer(model)
    bench_e = ind.enrich(bench) if bench is not None and len(bench) > 200 else None

    # Pre-enrich once; indicators are causal so this introduces no lookahead.
    enriched: dict[str, pd.DataFrame] = {}
    tier_of: dict[str, str] = {}
    for t, df in frames.items():
        if df is None or len(df) < MIN_HISTORY + horizon + 2:
            continue
        try:
            enriched[t] = ind.enrich(df)
            tier_of[t] = tr.classify_by_turnover(df)
        except Exception:
            continue

    if not enriched:
        return ICResult(pd.DataFrame(), pd.DataFrame(), {"error": "no usable data"})

    # Trading calendar.
    #
    # NOTE: an earlier version used the INTERSECTION of every ticker's dates.
    # That silently collapsed the calendar whenever the universe contained a
    # recent listing — one stock that IPO'd last year reduced a 5-year test to
    # a single window. The per-ticker loop below already skips names missing a
    # given date, so the intersection was never necessary.
    #
    # Prefer the benchmark's calendar (it spans the full period); otherwise
    # fall back to the ticker with the longest history.
    if bench_e is not None and len(bench_e) >= MIN_HISTORY + horizon + 2:
        cal = pd.DatetimeIndex(bench_e.index)
    else:
        cal = pd.DatetimeIndex(max((e.index for e in enriched.values()), key=len))

    if len(cal) < MIN_HISTORY + horizon + 2:
        return ICResult(
            pd.DataFrame(), pd.DataFrame(),
            {"error": f"insufficient history: {len(cal)} bars, need "
                      f"{MIN_HISTORY + horizon + 2}"},
        )

    rows, bucket_rows = [], []
    rng = np.random.default_rng(seed)
    # perm_matrix[w] = list of shuffled ICs for window w. Averaging ACROSS
    # windows per permutation gives the null distribution of the MEAN IC,
    # which is the statistic we actually report. Comparing a mean against a
    # distribution of single-window ICs would understate significance badly.
    perm_matrix: list[list[float]] = []

    for k in range(MIN_HISTORY, len(cal) - horizon - 1, step):
        date = cal[k]
        scores, fwd, tiers_w = [], [], []

        for t, e in enriched.items():
            try:
                i = e.index.get_loc(date)
            except KeyError:
                continue
            if not isinstance(i, int) or i < MIN_HISTORY or i + horizon >= len(e):
                continue

            try:
                sc = scorer(e, i, bench_e, tier_of[t])
            except Exception:
                continue
            if not np.isfinite(sc):
                continue

            p0 = float(e["Close"].iloc[i])
            p1 = float(e["Close"].iloc[i + horizon])
            if p0 <= 0 or not np.isfinite(p1):
                continue

            scores.append(sc)
            fwd.append((p1 / p0 - 1) * 100)
            tiers_w.append(tier_of[t])

        if len(scores) < min_names:
            continue

        s_arr = np.array(scores)
        f_arr = np.array(fwd)
        ic = _spearman(s_arr, f_arr)
        if not np.isfinite(ic):
            continue

        state = rg.classify_at(bench_e, k) if bench_e is not None else "n/a"

        # Quintile spread: top 20% by score vs bottom 20%
        order = np.argsort(s_arr)
        q = max(1, len(s_arr) // 5)
        bot = f_arr[order[:q]].mean()
        top = f_arr[order[-q:]].mean()

        rows.append({
            "date": date,
            "n_stocks": len(scores),
            "ic": round(ic, 4),
            "regime": state,
            "mean_fwd_ret": round(f_arr.mean(), 2),
            "top_quintile_ret": round(top, 2),
            "bot_quintile_ret": round(bot, 2),
            "spread": round(top - bot, 2),
            "mean_score": round(s_arr.mean(), 1),
        })

        # Bucket detail for the aggregate table
        qs = pd.qcut(pd.Series(s_arr).rank(method="first"), 5,
                     labels=["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"])
        for lab in qs.cat.categories:
            m = (qs == lab).to_numpy()
            if m.sum():
                bucket_rows.append({"quintile": lab, "fwd_ret": f_arr[m].mean(),
                                    "date": date})

        # Permutation: same forward returns, shuffled scores
        perm_matrix.append([
            _spearman(rng.permutation(s_arr), f_arr) for _ in range(n_permutations)
        ])

    if not rows:
        return ICResult(pd.DataFrame(), pd.DataFrame(), {"error": "no valid windows"})

    win = pd.DataFrame(rows)
    ics = win["ic"].to_numpy()
    n = len(ics)
    mean_ic = float(ics.mean())
    std_ic = float(ics.std(ddof=1)) if n > 1 else np.nan
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic and std_ic > 0 else np.nan

    # Null distribution of the MEAN IC across all windows
    if perm_matrix:
        pm = np.array(perm_matrix, dtype=float)          # (windows, permutations)
        null_arr = np.nanmean(pm, axis=0)                # one mean IC per permutation
        null_arr = null_arr[np.isfinite(null_arr)]
    else:
        null_arr = np.array([])
    pval = float((np.abs(null_arr) >= abs(mean_ic)).mean()) if len(null_arr) else np.nan

    buckets = (pd.DataFrame(bucket_rows)
               .groupby("quintile", observed=True)["fwd_ret"]
               .agg(["mean", "std", "count"])
               .round(3)
               .reset_index())

    monotonic = bool(buckets["mean"].is_monotonic_increasing)

    summary = {
        "windows": n,
        "horizon_days": horizon,
        "mean_ic": round(mean_ic, 4),
        "std_ic": round(std_ic, 4) if np.isfinite(std_ic) else None,
        "t_stat": round(t_stat, 2) if np.isfinite(t_stat) else None,
        "ic_ir": round(mean_ic / std_ic, 3) if std_ic and std_ic > 0 else None,
        "pct_positive_ic": round(float((ics > 0).mean()) * 100, 1),
        "mean_quintile_spread_pct": round(float(win["spread"].mean()), 2),
        "quintiles_monotonic": monotonic,
        "permutation_p_value": round(pval, 4) if np.isfinite(pval) else None,
        "null_mean_ic_std": round(float(null_arr.std()), 4) if len(null_arr) else None,
        "z_vs_null": (round(float((mean_ic - null_arr.mean()) / null_arr.std()), 2)
                      if len(null_arr) and null_arr.std() > 0 else None),
    }

    return ICResult(win, buckets, summary, null_arr)


def verdict(summary: dict) -> tuple[str, str]:
    """Plain-language read on the result. Returns (level, message)."""
    if "error" in summary:
        return "error", summary["error"]

    ic = summary.get("mean_ic") or 0
    t = summary.get("t_stat") or 0
    p = summary.get("permutation_p_value")
    n = summary.get("windows", 0)
    mono = summary.get("quintiles_monotonic")

    if n < 20:
        return "warn", (
            f"Only {n} windows. Too few to conclude anything — extend the test period "
            "or widen the universe."
        )
    if ic <= 0:
        return "bad", (
            f"Mean IC is {ic:+.4f}. The score does not rank forward returns. "
            "Do not trade this model."
        )
    if p is not None and p > 0.10:
        return "bad", (
            f"IC of {ic:+.4f} is not distinguishable from random (permutation p = {p:.3f}). "
            "Shuffled scores produce results this good roughly "
            f"{p*100:.0f}% of the time."
        )
    if ic < 0.02 or abs(t) < 2:
        return "warn", (
            f"IC {ic:+.4f}, t = {t:.2f}. Weak and not statistically convincing. "
            "Possibly real but too small to survive costs."
        )
    if ic > 0.15:
        return "warn", (
            f"IC of {ic:+.4f} is implausibly high for a retail factor. "
            "Check for lookahead bias or data errors before believing it."
        )
    mono_txt = "monotonic across quintiles" if mono else "NOT monotonic across quintiles"
    return "good", (
        f"IC {ic:+.4f}, t = {t:.2f}, permutation p = {p}. Quintiles are {mono_txt}. "
        "This is a real but modest signal — consistent with a tradeable edge, "
        "not a reason to size up."
    )
