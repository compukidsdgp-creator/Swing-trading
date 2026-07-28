#!/usr/bin/env python3
"""Execute the full statistical validation suite.

These tests were built and never run. That is the single largest contributor to
a weak statistical-validation score: the tooling exists, the results do not.

Runs, in order:
  1. Out-of-sample split — does the signal survive outside its selection period?
  2. Monte Carlo — what does the drawdown distribution look like across
     orderings, rather than the one ordering that happened?
  3. Standard risk metrics — Sharpe, Sortino, Calmar, VaR, CVaR.

Results are written to reports/ and appended to the audit trail.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import types
from pathlib import Path

# Headless Streamlit stub
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

import backtest as bt                   # noqa: E402
import config                           # noqa: E402
import datasource as dsrc               # noqa: E402
import governance as gov                # noqa: E402
import institutional as inst            # noqa: E402
import universe as uni                  # noqa: E402

UNIVERSE = "Nifty 500"
HORIZON = 15
PERIOD = "5y"
N_STOCKS = 100
MODEL = "momentum"


def main() -> int:
    print("=" * 70)
    print(f"SwingScope validation suite — {dt.datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 70)
    results: dict = {"run_at": dt.datetime.now().isoformat(timespec="seconds")}

    print(f"\nFetching {PERIOD} for up to {N_STOCKS} tickers from {UNIVERSE}…")
    res = uni.fetch_index_constituents(UNIVERSE)
    fetch = dsrc.fetch(tuple(res.tickers[: N_STOCKS * 3]), period=PERIOD, min_bars=300)
    print(f"  {fetch.summary()}")
    if not fetch.frames:
        print("  ABORT: no data.")
        return 1

    keep, how = uni.trim_universe(tuple(fetch.frames), N_STOCKS,
                                  method="liquidity", frames=fetch.frames)
    frames = {t: fetch.frames[t] for t in keep if t in fetch.frames}
    bench = dsrc.fetch((config.BENCHMARK,), period=PERIOD).frames.get(config.BENCHMARK)
    print(f"  universe: {len(frames)} — {how}")

    # ---- 1. Out-of-sample ----
    print("\n" + "-" * 70)
    print("1. OUT-OF-SAMPLE SPLIT")
    print("-" * 70)
    print("  Chronological, never random — a random split leaks the future.")
    oos = inst.out_of_sample_test(frames, bench, horizon=HORIZON, model=MODEL,
                                  oos_fraction=0.3)
    print(f"\n  in-sample  IC {oos.in_sample.get('mean_ic')}  "
          f"t={oos.in_sample.get('t_stat')}  windows={oos.in_sample.get('windows')}")
    print(f"  out-sample IC {oos.out_of_sample.get('mean_ic')}  "
          f"t={oos.out_of_sample.get('t_stat')}  windows={oos.out_of_sample.get('windows')}")
    if oos.degradation_pct is not None:
        print(f"  degradation: {oos.degradation_pct:.1f}%")
    print(f"\n  VERDICT [{oos.verdict.upper()}]")
    print("  " + oos.message.replace("\n", "\n  "))
    results["out_of_sample"] = {
        "in_sample_ic": oos.in_sample.get("mean_ic"),
        "oos_ic": oos.out_of_sample.get("mean_ic"),
        "degradation_pct": oos.degradation_pct,
        "verdict": oos.verdict,
        "evaluations_of_split": oos.evaluations_of_this_split,
    }

    # ---- 2. Backtest, then Monte Carlo ----
    print("\n" + "-" * 70)
    print("2. MONTE CARLO — sequence risk")
    print("-" * 70)
    print("  Generating trades to scramble…")
    trades = bt.run(frames, bench, min_score=0, hold_bars=HORIZON,
                    rebalance_every=HORIZON + 3, model=MODEL, apply_costs=True)
    print(f"  {len(trades)} trades generated")

    if len(trades) < 10:
        print("  Too few trades for Monte Carlo.")
        results["monte_carlo"] = {"error": f"only {len(trades)} trades"}
    else:
        mc = inst.monte_carlo(trades["net_r"], n_sims=5000, risk_per_trade_pct=1.0)
        print(f"\n  observed max drawdown : {mc.observed_max_drawdown:.2f}%")
        print("  simulated distribution:")
        for k, v in mc.drawdown_percentiles.items():
            print(f"    {k:>4}: {v:>7.2f}%")
        print(f"  worst of {mc.n_simulations}: {mc.worst_case:.2f}%")
        print(f"  P(ending down): {mc.prob_loss:.1%}   P(>50% drawdown): {mc.prob_ruin:.1%}")
        print(f"\n  VERDICT [{mc.verdict.upper()}]")
        print("  " + mc.message.replace("\n", "\n  "))
        results["monte_carlo"] = {
            "observed_dd": mc.observed_max_drawdown,
            "p95_dd": mc.drawdown_percentiles.get("p95"),
            "prob_loss": mc.prob_loss,
            "prob_ruin": mc.prob_ruin,
            "verdict": mc.verdict,
        }

        # ---- 3. Risk metrics ----
        print("\n" + "-" * 70)
        print("3. RISK METRICS")
        print("-" * 70)
        m = inst.risk_metrics(trades["net_r"], periods_per_year=252 / HORIZON,
                              risk_per_trade_pct=1.0)
        for k, v in m.items():
            print(f"  {k:26} {v}")
        print("\n  interpretation:")
        for note in inst.interpret_metrics(m):
            print(f"    - {note}")
        results["risk_metrics"] = m

    # ---- Persist ----
    Path("reports").mkdir(exist_ok=True)
    out = Path("reports/validation_suite.json")
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  written: {out}")

    try:
        gov.audit("validation_suite", results)
        print("  appended to audit trail")
    except Exception:                                          # noqa: BLE001
        pass

    print("\n" + "=" * 70)
    print("A held-out period is only held out once. Every subsequent look leaks")
    print("information into your decisions — the ledger records how many times")
    print("this split has been evaluated.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
