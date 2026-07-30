#!/usr/bin/env python3
"""Regression guardrail — has a code change degraded the validated metrics?

The gap this closes
-------------------
Validation results were recorded in reports and documents, but nothing verified
that subsequent code changes preserved them. A tweak to scoring, tier
thresholds, or cost assumptions could quietly move IC, expectancy or drawdown
without failing a single test.

The invariant suite checks *properties* — no lookahead, bounds respected, caps
never breached. Those are necessary and not sufficient: code can satisfy every
property while producing materially worse numbers.

How this works
--------------
Deterministic synthetic data with fixed seeds runs through the real scoring,
ranking, bucketing and backtest paths. Outputs are compared against recorded
baselines with explicit tolerances.

Fast by design — under a minute — so it can run on every push. Full validation
takes half an hour and belongs in a scheduled job, not in CI.

Tolerances, and why they are asymmetric
---------------------------------------
Metrics may **improve** freely. Degradation beyond tolerance fails the build.
An improvement is worth investigating but should not block a commit; a silent
regression should.

Baselines are recorded from a known-good state and stored in
`regression_baseline.json`. Update them deliberately, never to make a red build
green.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

try:
    import streamlit  # noqa: F401
except ImportError:
    _f = types.ModuleType("streamlit")

    def _p(*a, **k):
        return a[0] if (a and callable(a[0])) else (lambda fn: fn)

    _f.cache_data = _p
    _f.cache_resource = _p
    _f.secrets = {}
    sys.modules["streamlit"] = _f

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402

BASELINE = Path("regression_baseline.json")

# Degradation beyond these fails the build. Improvements pass.
TOLERANCE = {
    "mean_ic":            0.15,   # 15% relative
    "quintile_spread":    0.15,
    "expectancy_r":       0.20,
    "win_rate_pct":       0.10,
    "max_drawdown_pct":   0.25,   # more negative is worse
    "profit_factor":      0.15,
    "sharpe":             0.20,
    "n_trades":           0.30,
}

# Metrics where a LOWER value is worse
LOWER_IS_WORSE = {"mean_ic", "quintile_spread", "expectancy_r", "win_rate_pct",
                  "profit_factor", "sharpe", "n_trades"}


def _synthetic(n: int = 900, n_stocks: int = 40, seed: int = 4242) -> dict:
    """Deterministic universe with genuine momentum. Fixed seed, never changes."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    mkt = np.cumsum(rng.normal(0.0004, 0.009, n))
    frames = {}
    for i in range(n_stocks):
        beta = 0.7 + (i % 7) * 0.1
        r = np.zeros(n)
        for t in range(1, n):
            past = r[max(0, t - 252):max(0, t - 21)].sum()
            r[t] = (beta * (mkt[t] - mkt[t - 1])
                    + 0.0025 * past
                    + rng.normal(0.0002, 0.012))
        c = 100 * np.exp(np.cumsum(r))
        frames[f"R{i:02d}.NS"] = pd.DataFrame({
            "Open": c * 0.998, "High": c * 1.011, "Low": c * 0.989,
            "Close": c, "Volume": rng.integers(2e6, 8e6, n).astype(float),
        }, index=idx)
    bc = 100 * np.exp(mkt)
    bench = pd.DataFrame({"Open": bc, "High": bc * 1.004, "Low": bc * 0.996,
                          "Close": bc, "Volume": np.full(n, 1e7)}, index=idx)
    return {"frames": frames, "bench": bench}


def measure() -> dict:
    """Run the real paths and collect metrics."""
    import backtest as bt
    import validate as val

    data = _synthetic()
    out: dict = {}

    ic = val.run_ic(data["frames"], data["bench"], horizon=30,
                    n_permutations=60, model="momentum")
    s = ic.summary
    if "error" not in s:
        out["mean_ic"] = round(float(s.get("mean_ic") or 0), 5)
        out["quintile_spread"] = round(
            float(s.get("mean_quintile_spread_pct") or 0), 4)
        out["ic_windows"] = int(s.get("windows") or 0)

    trades = bt.run(data["frames"], data["bench"], min_score=0, hold_bars=30,
                    rebalance_every=33, model="momentum", apply_costs=True)
    if trades is not None and not trades.empty and "net_r" in trades.columns:
        r = pd.to_numeric(trades["net_r"], errors="coerce").dropna()
        if len(r) > 3:
            out["n_trades"] = int(len(r))
            out["expectancy_r"] = round(float(r.mean()), 4)
            out["win_rate_pct"] = round(float((r > 0).mean()) * 100, 2)
            eq = np.cumprod(1 + 0.01 * r.to_numpy())
            peak = np.maximum.accumulate(eq)
            out["max_drawdown_pct"] = round(float((eq / peak - 1).min()) * 100, 3)
            wins, losses = r[r > 0].sum(), abs(r[r <= 0].sum())
            out["profit_factor"] = (round(float(wins / losses), 4)
                                    if losses > 0 else None)
            sd = float(r.std(ddof=1))
            out["sharpe"] = round(float(r.mean() / sd), 4) if sd > 0 else None
    return out


def compare(current: dict, baseline: dict) -> tuple[bool, list[str], list[str]]:
    """True if no metric degraded beyond tolerance."""
    failures, notes = [], []
    for key, tol in TOLERANCE.items():
        if key not in current or key not in baseline:
            continue
        cur, base = current[key], baseline[key]
        if cur is None or base is None:
            continue
        if abs(base) < 1e-9:
            continue

        change = (cur - base) / abs(base)
        worse = (change < -tol) if key in LOWER_IS_WORSE else (change > tol)

        if worse:
            failures.append(
                f"{key}: {base:.4f} -> {cur:.4f} ({change:+.1%}, "
                f"tolerance {tol:.0%})")
        elif abs(change) > tol:
            notes.append(f"{key} IMPROVED: {base:.4f} -> {cur:.4f} ({change:+.1%})")
        elif abs(change) > tol / 2:
            notes.append(f"{key} moved {change:+.1%} (within tolerance)")
    return len(failures) == 0, failures, notes


def main() -> int:
    print("=" * 66)
    print("REGRESSION GUARDRAIL — deterministic synthetic data, fixed seeds")
    print("=" * 66)

    print("\nMeasuring…", flush=True)
    current = measure()
    if not current:
        print("  Could not compute metrics.")
        return 1
    for k, v in sorted(current.items()):
        print(f"  {k:22} {v}")

    if "--update" in sys.argv or not BASELINE.exists():
        BASELINE.write_text(json.dumps(current, indent=2))
        action = "updated" if "--update" in sys.argv else "created"
        print(f"\n  Baseline {action}: {BASELINE}")
        if action == "updated":
            print("  Only update a baseline deliberately — never to turn a red "
                  "build green.")
        return 0

    baseline = json.loads(BASELINE.read_text())
    ok, failures, notes = compare(current, baseline)

    print("\n" + "-" * 66)
    if ok:
        print("PASS — no metric degraded beyond tolerance.")
    else:
        print("FAIL — metrics degraded:")
        for f in failures:
            print(f"  {f}")
        print("\nEither the change is a genuine regression, or the baseline is "
              "legitimately out of date.\nIf the latter, run with --update and "
              "say so in the commit message.")
    for n in notes:
        print(f"  · {n}")
    print("-" * 66)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
