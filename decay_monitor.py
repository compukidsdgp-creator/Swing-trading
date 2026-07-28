#!/usr/bin/env python3
"""Alpha decay monitor — is the edge eroding?

Why this matters
----------------
Sub-period analysis showed IC declining monotonically across the sample:

    Aug 2022 – Dec 2023    IC 0.1223
    Dec 2023 – Mar 2025    IC 0.0825
    Mar 2025 – Jul 2026    IC 0.0762

A 38% fall from first period to last. That has three possible explanations
with very different implications:

  **Arbitrage.** Published anomalies weaken as capital chases them. If this is
  the cause, erosion continues and the strategy eventually stops paying.

  **Regime.** 2022-23 was a strong trending recovery — ideal momentum
  conditions. Choppier markets since. On this reading the edge returns when
  trends do.

  **Noise.** The sub-periods had 18, 36 and 54 windows respectively. The
  earliest has the widest error bars, so part of the apparent decline may be
  nothing at all.

Three points cannot distinguish these. A time series can.

What this does
--------------
Re-measures IC on a fixed protocol, appends to a history file, and reports the
trend. Run quarterly — monthly is too frequent to be informative and invites
reacting to noise.

Deliberately does not act
-------------------------
It reports and alerts. It does not adjust parameters, change the model, or
suppress trading. Automatic reaction to a declining metric is how systems get
optimised into oblivion — every response fits the most recent noise. The
decision to stop or adapt belongs to a human looking at the whole picture.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Stub Streamlit for headless operation
try:
    import streamlit  # noqa: F401
except ImportError:
    import types
    _f = types.ModuleType("streamlit")

    def _p(*a, **k):
        return a[0] if (a and callable(a[0])) else (lambda fn: fn)

    _f.cache_data = _p
    _f.cache_resource = _p
    _f.secrets = {}
    sys.modules["streamlit"] = _f

import config          # noqa: E402
import universe as uni  # noqa: E402
import validate as val  # noqa: E402

HISTORY_PATH = Path("decay_history.csv")

# Thresholds for alerting. Set from the economics, not from the data:
# a gross spread below roughly 0.6% does not clear charges plus 20% STCG.
IC_FLOOR = 0.030            # below this, viability is doubtful
IC_CRITICAL = 0.015         # below this, stop
DECLINE_WARN_PCT = 40.0     # fall from the historical peak


def _fetch(tickers: tuple[str, ...], period: str) -> dict[str, pd.DataFrame]:
    if not tickers:
        return {}
    raw = yf.download(list(tickers), period=period, interval="1d",
                      auto_adjust=True, progress=False, group_by="ticker",
                      threads=True)
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    for t in tickers:
        try:
            df = raw[t].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
        except (KeyError, IndexError):
            continue
        df = df.dropna(how="all")
        if df.empty or len(df) < 300:
            continue
        df.columns = [str(c).title() for c in df.columns]
        if {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
            out[t] = df
    return out


def load_history() -> pd.DataFrame:
    if HISTORY_PATH.exists():
        return pd.read_csv(HISTORY_PATH)
    return pd.DataFrame(columns=[
        "measured_on", "universe", "horizon", "period", "n_stocks",
        "windows", "mean_ic", "ic_t", "pct_positive", "permutation_p", "model",
    ])


def measure(universe: str, horizon: int, period: str, n_stocks: int,
            model: str) -> dict | None:
    """One measurement on a fixed protocol. Protocol must not change between runs."""
    res = uni.fetch_index_constituents(universe)
    tickers = tuple(res.tickers[: n_stocks * 3])
    frames = _fetch(tickers, period)
    if not frames:
        return None

    keep, _ = uni.trim_universe(tuple(frames), n_stocks,
                                method="liquidity", frames=frames)
    frames = {t: frames[t] for t in keep if t in frames}
    bench = _fetch((config.BENCHMARK,), period).get(config.BENCHMARK)

    ic = val.run_ic(frames, bench, horizon=horizon, n_permutations=300, model=model)
    s = ic.summary
    if "error" in s:
        return None

    return {
        "measured_on": dt.date.today().isoformat(),
        "universe": universe,
        "horizon": horizon,
        "period": period,
        "n_stocks": len(frames),
        "windows": s.get("windows"),
        "mean_ic": s.get("mean_ic"),
        "ic_t": s.get("t_stat"),
        "pct_positive": s.get("pct_positive_ic"),
        "permutation_p": s.get("permutation_p_value"),
        "model": model,
    }


def analyse(history: pd.DataFrame) -> dict:
    """Trend across measurements. Requires at least two."""
    if history.empty:
        return {"status": "empty", "message": "No measurements recorded yet."}

    h = history.copy()
    h["mean_ic"] = pd.to_numeric(h["mean_ic"], errors="coerce")
    h = h.dropna(subset=["mean_ic"]).sort_values("measured_on")
    if h.empty:
        return {"status": "empty", "message": "No usable measurements."}

    latest = float(h["mean_ic"].iloc[-1])
    peak = float(h["mean_ic"].max())
    first = float(h["mean_ic"].iloc[0])
    n = len(h)

    out = {
        "n_measurements": n,
        "latest_ic": round(latest, 4),
        "peak_ic": round(peak, 4),
        "first_ic": round(first, 4),
        "decline_from_peak_pct": round((1 - latest / peak) * 100, 1) if peak > 0 else None,
        "change_since_first_pct": round((latest / first - 1) * 100, 1) if first > 0 else None,
    }

    # Linear trend, once there is enough to fit one
    if n >= 3:
        x = np.arange(n, dtype=float)
        slope = float(np.polyfit(x, h["mean_ic"].to_numpy(), 1)[0])
        out["trend_per_measurement"] = round(slope, 5)
        out["trend_direction"] = ("declining" if slope < -0.002
                                  else "rising" if slope > 0.002 else "flat")

    # Verdict, ordered by severity
    if latest < IC_CRITICAL:
        out.update(status="critical", message=(
            f"IC has fallen to {latest:.4f}, below the {IC_CRITICAL} floor. At this "
            "level the gross spread cannot cover charges and 20% short-term capital "
            "gains tax. Stop trading the model and investigate before resuming."))
    elif latest < IC_FLOOR:
        out.update(status="warning", message=(
            f"IC is {latest:.4f}, below the {IC_FLOOR} viability threshold. The edge "
            "may no longer clear its costs. Reduce size and watch the next reading "
            "closely."))
    elif out.get("decline_from_peak_pct", 0) and out["decline_from_peak_pct"] > DECLINE_WARN_PCT:
        out.update(status="warning", message=(
            f"IC {latest:.4f} is {out['decline_from_peak_pct']:.0f}% below its peak of "
            f"{peak:.4f}. Consistent with either arbitrage of the anomaly or an "
            "unfavourable regime — the distinction matters and needs more readings."))
    elif n < 3:
        out.update(status="baseline", message=(
            f"{n} measurement(s) recorded. At least three are needed before a trend "
            "means anything."))
    else:
        out.update(status="ok", message=(
            f"IC {latest:.4f}, {out.get('trend_direction', 'stable')} across "
            f"{n} measurements. No decay signal."))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Alpha decay monitor")
    p.add_argument("--universe", default="Nifty 500")
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--period", default="5y")
    p.add_argument("--stocks", type=int, default=100)
    p.add_argument("--model", default="momentum")
    p.add_argument("--notify", action="store_true")
    p.add_argument("--analyse-only", action="store_true",
                   help="report on existing history without measuring")
    args = p.parse_args()

    print("=" * 66)
    print(f"Alpha decay monitor — {dt.datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 66)

    history = load_history()

    if not args.analyse_only:
        print(f"\nMeasuring: {args.universe}, {args.horizon}d horizon, "
              f"{args.period}, {args.stocks} stocks, model={args.model}")
        print("(protocol must stay fixed between runs or the series is meaningless)")
        row = measure(args.universe, args.horizon, args.period,
                      args.stocks, args.model)
        if row is None:
            print("\nMeasurement failed — no data, or too few valid windows.")
            return 1
        print(f"\n  IC {row['mean_ic']:+.4f}  t={row['ic_t']}  "
              f"windows={row['windows']}  p={row['permutation_p']}")
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
        history.to_csv(HISTORY_PATH, index=False)
        print(f"  appended to {HISTORY_PATH} ({len(history)} measurements)")

    result = analyse(history)
    print("\n" + "-" * 66)
    print(f"STATUS: {result['status'].upper()}")
    print(result["message"])
    if result.get("n_measurements", 0) >= 2:
        print(f"\n  latest {result['latest_ic']:+.4f} | peak {result['peak_ic']:+.4f} "
              f"| first {result['first_ic']:+.4f}")
        if result.get("decline_from_peak_pct") is not None:
            print(f"  decline from peak: {result['decline_from_peak_pct']:.1f}%")
        if "trend_direction" in result:
            print(f"  trend: {result['trend_direction']} "
                  f"({result['trend_per_measurement']:+.5f} per measurement)")

    if not history.empty:
        print("\n  history:")
        cols = [c for c in ("measured_on", "mean_ic", "ic_t", "windows",
                            "permutation_p") if c in history.columns]
        print(history[cols].to_string(index=False))

    if args.notify and result["status"] in ("warning", "critical"):
        try:
            import notify
            icon = "🔴" if result["status"] == "critical" else "🟠"
            notify.dispatch(
                subject=f"SwingScope decay alert — {result['status']}",
                html_body=f"<h2>Decay monitor: {result['status']}</h2>"
                          f"<p>{result['message']}</p>",
                text_body=f"{icon} *SwingScope decay alert*\n\n{result['message']}",
                channels="auto",
            )
        except Exception as exc:                               # noqa: BLE001
            print(f"  notification failed: {type(exc).__name__}: {exc}")

    Path("reports").mkdir(exist_ok=True)
    Path("reports/decay_latest.json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
