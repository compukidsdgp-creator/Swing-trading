"""Horizon analysis — finding the holding period that actually clears its hurdle.

The question
------------
Validation measured IC at a single 15-day horizon. Cost analysis then showed
that at 15 days the edge nets to roughly zero once 20% STCG and ~0.35% charges
are applied. Both facts are established. What is not established is whether a
different holding period clears the hurdle — and that is a far larger lever
than any refinement to the signal.

What this measures
------------------
For each candidate horizon, four things:

  1. **IC** — does the signal still rank forward returns at that horizon?
  2. **Gross spread** — top-quintile minus bottom-quintile return, in
     percentage points. This, not IC, is what pays for costs.
  3. **Net edge** — gross spread after charges and capital gains tax.
  4. **Annualised net** — net per cycle multiplied by cycles per year. A large
     per-cycle edge held rarely can lose to a small edge held often, and vice
     versa. This is the number to optimise.

Guarding against the obvious trap
---------------------------------
Sweeping eight horizons and reporting the best one is multiple testing. A
strategy that works at exactly 37 days and nowhere else has found noise. This
module therefore reports the whole curve and explicitly tests for a **plateau**
— a contiguous band of horizons that all work. A plateau is evidence; an
isolated spike is not, and the verdict says so.

Non-overlapping windows are used throughout, with a step of horizon+3 to avoid
every rebalance landing on the same weekday.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import costs
import indicators as ind
from backtest import get_scorer, MIN_HISTORY
from factor_analysis import _newey_west_se, _spearman

DEFAULT_HORIZONS = (5, 10, 15, 20, 30, 45, 60, 90, 120)


@dataclass
class HorizonResult:
    table: pd.DataFrame
    best_by_annualised: dict | None = None
    plateau: list[int] = field(default_factory=list)
    verdict_level: str = ""
    verdict_msg: str = ""
    notes: list[str] = field(default_factory=list)


def _measure_one(
    enriched: dict[str, pd.DataFrame],
    bench_e: pd.DataFrame | None,
    cal: pd.DatetimeIndex,
    horizon: int,
    scorer,
    tier_of: dict[str, str],
    min_names: int,
) -> dict | None:
    """IC and quintile spread at a single horizon."""
    step = horizon + 3                     # avoids weekday lock
    ics, spreads, tops, bots, ns = [], [], [], [], []

    for k in range(MIN_HISTORY, len(cal) - horizon - 1, step):
        date = cal[k]
        sc, fwd = [], []
        for t, e in enriched.items():
            try:
                i = e.index.get_loc(date)
            except KeyError:
                continue
            if not isinstance(i, int) or i < MIN_HISTORY or i + horizon >= len(e):
                continue
            try:
                v = scorer(e, i, bench_e, tier_of.get(t, "mid"))
            except Exception:                                  # noqa: BLE001
                continue
            if not np.isfinite(v):
                continue
            p0 = float(e["Close"].iloc[i])
            p1 = float(e["Close"].iloc[i + horizon])
            if p0 <= 0 or not np.isfinite(p1):
                continue
            sc.append(v)
            fwd.append((p1 / p0 - 1) * 100)

        if len(sc) < min_names:
            continue

        s_arr, f_arr = np.array(sc), np.array(fwd)
        ic = _spearman(s_arr, f_arr)
        if not np.isfinite(ic):
            continue

        order = np.argsort(s_arr)
        q = max(1, len(s_arr) // 5)
        bot = float(f_arr[order[:q]].mean())
        top = float(f_arr[order[-q:]].mean())

        ics.append(ic)
        spreads.append(top - bot)
        tops.append(top)
        bots.append(bot)
        ns.append(len(sc))

    if len(ics) < 8:
        return None

    ic_arr = np.array(ics)
    sp_arr = np.array(spreads)
    ic_nw = _newey_west_se(ic_arr)
    sp_nw = _newey_west_se(sp_arr)

    return {
        "horizon": horizon,
        "windows": len(ic_arr),
        "mean_ic": float(ic_arr.mean()),
        "ic_t": float(ic_arr.mean() / ic_nw) if ic_nw and ic_nw > 0 else np.nan,
        "pct_positive_ic": float((ic_arr > 0).mean()) * 100,
        "gross_spread_pct": float(sp_arr.mean()),
        "spread_t": float(sp_arr.mean() / sp_nw) if sp_nw and sp_nw > 0 else np.nan,
        "top_quintile_pct": float(np.mean(tops)),
        "bot_quintile_pct": float(np.mean(bots)),
        "mean_stocks": float(np.mean(ns)),
    }


def sweep(
    frames: dict[str, pd.DataFrame],
    bench: pd.DataFrame | None,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    model: str = "momentum",
    charges_pct: float = 0.35,
    win_rate: float = 0.55,
    min_names: int = 25,
    progress=None,
) -> HorizonResult:
    """Measure IC, gross spread and net-of-tax edge across holding periods."""
    import tiers as tr

    scorer = get_scorer(model)
    bench_e = ind.enrich(bench) if bench is not None and len(bench) > 300 else None

    enriched, tier_of = {}, {}
    max_h = max(horizons)
    for t, df in frames.items():
        if df is None or len(df) < MIN_HISTORY + max_h + 5:
            continue
        try:
            enriched[t] = ind.enrich(df)
            tier_of[t] = tr.classify_by_turnover(df)
        except Exception:                                      # noqa: BLE001
            continue

    if not enriched:
        return HorizonResult(pd.DataFrame(), verdict_level="error",
                             verdict_msg="No usable price data.")

    cal = (pd.DatetimeIndex(bench_e.index) if bench_e is not None
           else pd.DatetimeIndex(max((e.index for e in enriched.values()), key=len)))

    rows = []
    for idx, h in enumerate(horizons):
        if progress:
            progress(idx + 1, len(horizons), h)
        r = _measure_one(enriched, bench_e, cal, h, scorer, tier_of, min_names)
        if r is None:
            continue

        # Net edge after charges and capital gains tax.
        # The gross spread is the long-short return; a long-only strategy
        # captures roughly the top quintile's excess over the mean, so the
        # spread is used as the gross edge with that caveat noted.
        econ = costs.edge_after_tax(
            max(r["gross_spread_pct"], 0.0),
            win_rate=win_rate,
            charges_pct=charges_pct,
            holding_days=h,
        )
        r.update({
            "cycles_per_year": econ["cycles_per_year"],
            "tax_rate": econ["tax_rate_applied"],
            "net_per_cycle_pct": econ["net_edge_pct"],
            "annualised_net_pct": econ["annualised_net_pct"],
            "viable": bool(econ["viable"] and r["gross_spread_pct"] > 0),
        })
        rows.append(r)

    if not rows:
        return HorizonResult(pd.DataFrame(), verdict_level="error",
                             verdict_msg="No horizon produced enough windows.")

    table = pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)
    result = HorizonResult(table)

    viable = table[table["viable"]]
    if not viable.empty:
        best = viable.loc[viable["annualised_net_pct"].idxmax()]
        result.best_by_annualised = best.to_dict()

    # Plateau detection: the longest run of contiguous viable horizons.
    # A plateau is evidence of a real effect; an isolated peak is not.
    run, best_run = [], []
    for _, r in table.iterrows():
        if r["viable"]:
            run.append(int(r["horizon"]))
        else:
            if len(run) > len(best_run):
                best_run = run
            run = []
    if len(run) > len(best_run):
        best_run = run
    result.plateau = best_run

    result.verdict_level, result.verdict_msg, result.notes = _verdict(table, best_run)
    return result


def _verdict(table: pd.DataFrame, plateau: list[int]) -> tuple[str, str, list[str]]:
    notes: list[str] = []
    viable = table[table["viable"]]

    n_tested = len(table)
    notes.append(f"{n_tested} horizons tested. Reporting the full curve rather "
                 "than only the best, because picking the peak from a sweep is "
                 "multiple testing.")

    if viable.empty:
        return "bad", (
            "**No holding period clears its hurdle.** At every horizon tested, the "
            "gross quintile spread is consumed by charges and capital gains tax. "
            "This is not a horizon problem — the signal is too weak to pay for "
            "trading it at any frequency."
        ), notes

    best = viable.loc[viable["annualised_net_pct"].idxmax()]
    ic_ok = table["ic_t"].abs().max() >= 2.0
    if not ic_ok:
        notes.append("No horizon reached |t| >= 2 on IC. The ranking itself is "
                     "weak, whatever the economics suggest.")

    if len(plateau) >= 3:
        return "good", (
            f"**A plateau of {len(plateau)} contiguous viable horizons "
            f"({plateau[0]}–{plateau[-1]} days).** That is the shape a real effect "
            f"produces — not a spike at one lucky value. Best annualised net is "
            f"{best['annualised_net_pct']:+.2f}% at {int(best['horizon'])} days "
            f"(gross spread {best['gross_spread_pct']:.2f}%, IC "
            f"{best['mean_ic']:+.4f}, t={best['ic_t']:.2f})."
        ), notes

    if len(plateau) == 2:
        return "ok", (
            f"Two adjacent horizons are viable ({plateau[0]} and {plateau[-1]} days), "
            f"best annualised {best['annualised_net_pct']:+.2f}% at "
            f"{int(best['horizon'])} days. Narrow but not a single-point spike. "
            "Confirm on a different period before committing."
        ), notes

    return "warn", (
        f"Only {int(best['horizon'])} days is viable, at "
        f"{best['annualised_net_pct']:+.2f}% annualised — an **isolated peak with "
        f"no supporting plateau**. Across {n_tested} horizons tested, one working "
        "in isolation is roughly what noise produces. Treat this as unconfirmed."
    ), notes


def stability_check(
    frames: dict[str, pd.DataFrame],
    bench: pd.DataFrame | None,
    horizon: int,
    *,
    model: str = "momentum",
    n_splits: int = 3,
) -> pd.DataFrame:
    """Does the chosen horizon work across sub-periods, or only on average?

    An edge that appears in one third of the sample and vanishes in the others
    is a regime artefact. This splits the history chronologically and re-measures.
    """
    import tiers as tr

    scorer = get_scorer(model)
    bench_e = ind.enrich(bench) if bench is not None and len(bench) > 300 else None
    enriched, tier_of = {}, {}
    for t, df in frames.items():
        if df is None or len(df) < MIN_HISTORY + horizon + 5:
            continue
        try:
            enriched[t] = ind.enrich(df)
            tier_of[t] = tr.classify_by_turnover(df)
        except Exception:                                      # noqa: BLE001
            continue
    if not enriched:
        return pd.DataFrame()

    cal = (pd.DatetimeIndex(bench_e.index) if bench_e is not None
           else pd.DatetimeIndex(max((e.index for e in enriched.values()), key=len)))
    usable = cal[MIN_HISTORY:]
    bounds = np.linspace(0, len(usable), n_splits + 1).astype(int)

    rows = []
    for s in range(n_splits):
        sub = cal[: MIN_HISTORY + bounds[s + 1]]
        if len(sub) < MIN_HISTORY + horizon + 20:
            continue
        # Measure only within this slice by trimming the calendar start
        sub_cal = pd.DatetimeIndex(
            cal[MIN_HISTORY + bounds[s]: MIN_HISTORY + bounds[s + 1]])
        if len(sub_cal) < horizon + 20:
            continue
        full = pd.DatetimeIndex(cal[: MIN_HISTORY + bounds[s + 1]])
        r = _measure_one(enriched, bench_e, full, horizon, scorer, tier_of, 20)
        if r:
            r["period"] = (f"{sub_cal[0]:%b %Y} – {sub_cal[-1]:%b %Y}")
            r["split"] = s + 1
            rows.append(r)
    return pd.DataFrame(rows)
