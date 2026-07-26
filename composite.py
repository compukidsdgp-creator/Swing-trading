"""Orthogonal composite construction — the fix for the v1 scoring failure.

Why v1 failed
-------------
The original composite blended five components — Trend, Momentum, Volume,
Relative Strength, Setup — with weights chosen by reasoning about what *should*
matter. Nobody checked whether the five were independent. They were not: all
five measured recent price movement, the composite loaded 0.76 on one-month
return, and 13.9% of its IC survived factor neutralisation.

The lesson is not "pick better weights". It is that **weights are meaningless
until the components are shown to be independent.**

The rule this module enforces
-----------------------------
A signal earns a place in the composite only if it clears **both** tests:

  1. **Incremental content** — residual IC significantly different from zero
     after the standard factor set is neutralised (Newey-West t >= threshold).
  2. **Independence** — correlation with every already-selected component below
     a ceiling (default 0.6). A signal that duplicates an existing one adds
     variance without adding information.

Selection is greedy: take the strongest surviving signal, then repeat among
those still independent of everything chosen so far.

Weighting
---------
Components are weighted by residual IC divided by residual volatility — a
Sharpe-like allocation that gives more to signals which are both predictive and
consistent. Weights are NOT fitted to returns. Fitting weights on the same data
used to select signals is how backtests become fiction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import indicators as ind
from factor_analysis import _newey_west_se, _spearman, _winsorise, _zscore
from signals import SIGNALS


@dataclass
class CompositeSpec:
    components: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    correlations: pd.DataFrame | None = None
    rejected: list[tuple[str, str]] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.components

    def describe(self) -> str:
        if self.is_empty:
            return "No components qualified — no composite can honestly be built."
        parts = [f"{c} ({self.weights[c]:.0%})" for c in self.components]
        return " + ".join(parts)


def correlation_matrix(
    frames: dict[str, pd.DataFrame],
    bench: pd.DataFrame | None,
    *,
    step: int = 15,
    min_names: int = 25,
    max_windows: int = 60,
) -> pd.DataFrame:
    """Average cross-sectional correlation between every pair of signals.

    This is the diagnostic that would have caught the v1 failure before it was
    ever deployed. Run it on any candidate set before blending.
    """
    bench_e = ind.enrich(bench) if bench is not None and len(bench) > 300 else None
    enriched = {}
    for t, df in frames.items():
        if df is None or len(df) < 320:
            continue
        try:
            enriched[t] = ind.enrich(df)
        except Exception:                                      # noqa: BLE001
            continue
    if not enriched:
        return pd.DataFrame()

    cal = (pd.DatetimeIndex(bench_e.index) if bench_e is not None
           else pd.DatetimeIndex(max((e.index for e in enriched.values()), key=len)))

    names = list(SIGNALS)
    acc, count = np.zeros((len(names), len(names))), 0

    for k in range(300, len(cal) - 20, step):
        if count >= max_windows:
            break
        date = cal[k]
        recs = []
        for t, e in enriched.items():
            try:
                i = e.index.get_loc(date)
            except KeyError:
                continue
            if not isinstance(i, int) or i < 300:
                continue
            row = {}
            for n, fn in SIGNALS.items():
                try:
                    row[n] = fn(e, i, bench_e)
                except Exception:                              # noqa: BLE001
                    row[n] = np.nan
            recs.append(row)

        if len(recs) < min_names:
            continue
        d = pd.DataFrame(recs)[names]
        if d.isna().mean().max() > 0.4:
            continue
        d = d.apply(lambda c: pd.Series(
            _zscore(_winsorise(np.where(np.isnan(c.to_numpy(dtype=float)),
                                        np.nanmedian(c.to_numpy(dtype=float)),
                                        c.to_numpy(dtype=float))))))
        cm = d.corr(method="spearman").to_numpy()
        if np.isfinite(cm).all():
            acc += cm
            count += 1

    if count == 0:
        return pd.DataFrame()
    return pd.DataFrame(acc / count, index=names, columns=names).round(3)


def build(
    signal_table: pd.DataFrame,
    corr: pd.DataFrame,
    *,
    min_t: float = 2.0,
    max_correlation: float = 0.6,
    max_components: int = 4,
) -> CompositeSpec:
    """Greedily select independent, statistically significant signals.

    Args:
        signal_table: output of signals.run() — needs signal, residual_ic,
                      residual_t_nw columns.
        corr: pairwise correlation matrix from correlation_matrix().
        min_t: Newey-West t threshold for admission. 2.0 is conventional;
               3.0 is the Harvey-Liu-Zhu bar for a new factor claim.
        max_correlation: independence ceiling between selected components.
    """
    spec = CompositeSpec()
    if signal_table is None or signal_table.empty:
        spec.diagnostics["error"] = "no signal results supplied"
        return spec

    t = signal_table.copy()
    t = t[t["residual_t_nw"].notna() & t["residual_ic"].notna()]

    # Gate 1 — statistical significance of incremental content
    qualified = t[(t["residual_t_nw"].abs() >= min_t) & (t["residual_ic"] > 0)]
    for _, r in t.iterrows():
        if r["signal"] not in set(qualified["signal"]):
            reason = (f"residual t {r['residual_t_nw']:+.2f} below {min_t}"
                      if abs(r["residual_t_nw"]) < min_t
                      else f"residual IC {r['residual_ic']:+.4f} not positive")
            spec.rejected.append((r["signal"], reason))

    if qualified.empty:
        spec.diagnostics["error"] = (
            f"No signal reached |t| >= {min_t} with positive residual IC. "
            "A composite cannot honestly be built from components that show no "
            "incremental content."
        )
        return spec

    qualified = qualified.sort_values("residual_ic", ascending=False)

    # Gate 2 — independence from already-selected components
    chosen: list[str] = []
    for _, r in qualified.iterrows():
        name = r["signal"]
        if len(chosen) >= max_components:
            spec.rejected.append((name, f"composite already at {max_components} components"))
            continue
        if corr is not None and not corr.empty and name in corr.index:
            clash = None
            for c in chosen:
                if c in corr.columns:
                    rho = abs(float(corr.loc[name, c]))
                    if rho > max_correlation:
                        clash = (c, rho)
                        break
            if clash:
                spec.rejected.append(
                    (name, f"correlation {clash[1]:.2f} with {clash[0]} exceeds "
                           f"{max_correlation} — redundant")
                )
                continue
        chosen.append(name)

    if not chosen:
        spec.diagnostics["error"] = "all qualifying signals were mutually redundant"
        return spec

    # Weight by residual IC / residual volatility (Sharpe-like, not fitted)
    sub = qualified[qualified["signal"].isin(chosen)]
    raw_w = {}
    for _, r in sub.iterrows():
        ic = float(r["residual_ic"])
        tt = abs(float(r["residual_t_nw"])) or 1.0
        vol_proxy = ic / tt if tt > 0 else ic       # implied SE
        raw_w[r["signal"]] = ic / vol_proxy if vol_proxy > 1e-9 else 0.0

    total = sum(raw_w.values())
    weights = ({k: v / total for k, v in raw_w.items()}
               if total > 0 else {k: 1 / len(chosen) for k in chosen})

    spec.components = chosen
    spec.weights = {k: round(v, 4) for k, v in weights.items()}
    spec.correlations = (corr.loc[chosen, chosen]
                         if corr is not None and not corr.empty
                         and all(c in corr.index for c in chosen) else None)
    spec.diagnostics = {
        "n_qualified": len(qualified),
        "n_selected": len(chosen),
        "n_rejected": len(spec.rejected),
        "min_t_used": min_t,
        "max_correlation_used": max_correlation,
        "max_pairwise_corr": (float(np.abs(spec.correlations.to_numpy()[
            ~np.eye(len(chosen), dtype=bool)]).max())
            if spec.correlations is not None and len(chosen) > 1 else 0.0),
    }
    return spec


def score_at(spec: CompositeSpec, e: pd.DataFrame, i: int,
             bench_e: pd.DataFrame | None) -> float:
    """Evaluate the composite for one stock at one bar.

    Note this returns a RAW value, not a 0-100 score. Composites must be
    z-scored cross-sectionally at each date — a raw value is meaningless in
    isolation, since ranking is the entire point.
    """
    if spec.is_empty:
        return np.nan
    total = 0.0
    for name in spec.components:
        fn = SIGNALS.get(name)
        if fn is None:
            continue
        try:
            v = fn(e, i, bench_e)
        except Exception:                                      # noqa: BLE001
            return np.nan
        if not np.isfinite(v):
            return np.nan
        total += spec.weights[name] * v
    return float(total)


def validate(
    spec: CompositeSpec,
    frames: dict[str, pd.DataFrame],
    bench: pd.DataFrame | None,
    *,
    horizon: int = 15,
    step: int | None = None,
    min_names: int = 25,
) -> dict:
    """Measure the assembled composite's own IC.

    Important: this is measured on the same data used to select components, so
    it is optimistic. Treat it as an upper bound and confirm forward.
    """
    if spec.is_empty:
        return {"error": "empty composite"}

    step = step or horizon
    bench_e = ind.enrich(bench) if bench is not None and len(bench) > 300 else None
    enriched = {}
    for t, df in frames.items():
        if df is None or len(df) < 320 + horizon:
            continue
        try:
            enriched[t] = ind.enrich(df)
        except Exception:                                      # noqa: BLE001
            continue
    if not enriched:
        return {"error": "no usable data"}

    cal = (pd.DatetimeIndex(bench_e.index) if bench_e is not None
           else pd.DatetimeIndex(max((e.index for e in enriched.values()), key=len)))

    ics = []
    for k in range(300, len(cal) - horizon - 1, step):
        date = cal[k]
        vals, fwd = [], []
        for t, e in enriched.items():
            try:
                i = e.index.get_loc(date)
            except KeyError:
                continue
            if not isinstance(i, int) or i < 300 or i + horizon >= len(e):
                continue
            v = score_at(spec, e, i, bench_e)
            if not np.isfinite(v):
                continue
            p0 = float(e["Close"].iloc[i])
            p1 = float(e["Close"].iloc[i + horizon])
            if p0 <= 0 or not np.isfinite(p1):
                continue
            vals.append(v)
            fwd.append((p1 / p0 - 1) * 100)

        if len(vals) < min_names:
            continue
        ic = _spearman(np.array(vals), np.array(fwd))
        if np.isfinite(ic):
            ics.append(ic)

    if len(ics) < 10:
        return {"error": f"only {len(ics)} valid windows"}

    arr = np.array(ics)
    mean = float(arr.mean())
    nw = _newey_west_se(arr)
    return {
        "windows": len(arr),
        "composite_ic": round(mean, 4),
        "t_newey_west": round(mean / nw, 2) if nw and nw > 0 else None,
        "pct_positive": round(float((arr > 0).mean()) * 100, 1),
        "components": spec.components,
        "weights": spec.weights,
        "caveat": ("Measured on the selection data — optimistic by construction. "
                   "Confirm on a held-out period or forward."),
    }
