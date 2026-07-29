#!/usr/bin/env python3
"""Point-in-time validation — closes the largest outstanding audit gap.

Runs the same measurement twice: once on today's constituent list (the standard
approach, which carries survivorship bias) and once on a universe rebuilt at
every observation date from NSE bhavcopy.

The difference is the survivorship contribution — a number that has never been
measured for this system, and which affects every figure currently reported.

Requires a populated bhavcopy cache. Download history first via the app's NSE
bhavcopy section or `bhavcopy.fetch_range()`.
"""

from __future__ import annotations

import argparse
import datetime as dt
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

import pandas as pd                     # noqa: E402

import bhavcopy as bc                   # noqa: E402
import config                           # noqa: E402
import datasource as dsrc               # noqa: E402
import governance as gov                # noqa: E402
import pit_validation as pv             # noqa: E402
import universe as uni                  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Point-in-time validation")
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--period", default="5y")
    p.add_argument("--stocks", type=int, default=300)
    p.add_argument("--min-turnover", type=float, default=10.0)
    p.add_argument("--local-data", default=None,
                   help="directory of per-symbol CSVs, e.g. the 20-year dataset")
    p.add_argument("--prices-from-bhavcopy", action="store_true",
                   help="reconstruct price history from bhavcopy itself rather "
                        "than yfinance. This is the version that actually "
                        "removes survivorship bias: universe AND prices come "
                        "from the same source, so delisted companies appear in "
                        "both. Strongly recommended.")
    args = p.parse_args()

    print("=" * 70)
    print(f"Point-in-time validation — {dt.datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 70)

    stats = bc.cache_stats()
    print(f"\nBhavcopy cache: {stats['days']} days "
          f"({stats.get('earliest')} to {stats.get('latest')})")
    if stats["days"] < args.horizon + 60:
        print(f"\nNot enough cached history. Need at least "
              f"{args.horizon + 60} days; have {stats['days']}.")
        print("Download more via the app's NSE bhavcopy section first.")
        return 1

    # --- Price data ---
    if args.prices_from_bhavcopy:
        print("\nReconstructing price history from bhavcopy…")
        print("  (universe and prices from the same source — delisted "
              "companies present in both)")
        bhav_all = bc.load_cached()
        frames, rep = bc.build_price_history(bhav_all, min_days=400)
        if "error" in rep:
            print(f"  {rep['error']}")
            return 1
        print(f"  {rep['symbols']} symbols · {rep['total_bars']:,} bars · "
              f"{rep['date_range'][0]} to {rep['date_range'][1]}")
        print(f"  {rep['split_adjusted']} split adjustment(s) applied")
        if rep["split_adjusted"] > rep["symbols"] * 0.15:
            print("  ! Unusually many adjustments — inspect before trusting "
                  "the result.")

        gone = bc.delisted_symbols(bhav_all)
        if not gone.empty:
            print(f"\n  {len(gone)} symbols stopped trading during the period.")
            print("  These are exactly what a present-day constituent list hides.")
    elif args.local_data:
        import local_data as ld
        print(f"\nLoading local dataset from {args.local_data}…")
        res = ld.load(args.local_data, use_adjusted=True, min_bars=500,
                      max_symbols=args.stocks)
        frames = res.frames
        print(f"  {res.summary()}")
    else:
        print(f"\nFetching {args.period} for up to {args.stocks} tickers…")
        u = uni.fetch_index_constituents("Nifty 500")
        fetched = dsrc.fetch(tuple(u.tickers[: args.stocks * 2]),
                             period=args.period, min_bars=400)
        frames = fetched.frames
        print(f"  {fetched.summary()}")

    if not frames:
        print("  No price data.")
        return 1

    bhav = bc.load_cached()
    print(f"  {len(bhav)} bhavcopy days loaded from cache")

    if not args.prices_from_bhavcopy:
        print("\n  NOTE: prices come from yfinance, which only covers surviving "
              "companies.\n  Bhavcopy fixes index-membership drift but a "
              "delisted stock still has no\n  price series, so it is skipped. "
              "Use --prices-from-bhavcopy for the\n  version that genuinely "
              "removes survivorship bias.")

    # --- Standard ---
    print("\n" + "-" * 70)
    print("1. STANDARD — fixed present-day universe")
    print("-" * 70)
    std = pv.validate_standard(frames, horizon=args.horizon)
    if std.windows < 8:
        print("  " + "; ".join(std.notes))
        return 1
    print(f"  IC {std.mean_ic:+.4f}  t={std.ic_t}  "
          f"spread {std.gross_spread_pct:+.3f}%")
    print(f"  {std.windows} windows · universe {std.mean_universe_size:.0f} · "
          f"{std.pct_positive:.1f}% positive")

    # --- Point in time ---
    print("\n" + "-" * 70)
    print("2. POINT-IN-TIME — universe rebuilt at every observation date")
    print("-" * 70)

    def _prog(i, n, d):
        if i % 200 == 0:
            print(f"    {d}…", flush=True)

    pit = pv.validate_pit(frames, bhav, horizon=args.horizon,
                          min_turnover_cr=args.min_turnover, progress=_prog)
    if pit.windows < 8:
        print("  " + "; ".join(pit.notes))
        return 1
    print(f"  IC {pit.mean_ic:+.4f}  t={pit.ic_t}  "
          f"spread {pit.gross_spread_pct:+.3f}%")
    print(f"  {pit.windows} windows · universe {pit.mean_universe_size:.0f} · "
          f"churn {pit.universe_churn_pct:.1f}% · {pit.pct_positive:.1f}% positive")
    for n in pit.notes:
        print(f"  · {n}")

    # --- Comparison ---
    print("\n" + "=" * 70)
    print("SURVIVORSHIP CONTRIBUTION")
    print("=" * 70)
    cmp_ = pv.compare(std, pit)
    print()
    print(pv.summary_table(cmp_).to_string(index=False))
    print(f"\n[{cmp_.verdict.upper()}]")
    print(cmp_.message)

    out = {
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "horizon": args.horizon,
        "standard": {"ic": std.mean_ic, "t": std.ic_t,
                     "spread": std.gross_spread_pct, "windows": std.windows},
        "point_in_time": {"ic": pit.mean_ic, "t": pit.ic_t,
                          "spread": pit.gross_spread_pct, "windows": pit.windows,
                          "churn_pct": pit.universe_churn_pct},
        "survivorship_inflation_pct": cmp_.survivorship_inflation_pct,
        "verdict": cmp_.verdict,
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/pit_validation.json").write_text(json.dumps(out, indent=2))
    print("\n  written: reports/pit_validation.json")

    try:
        gov.audit("pit_validation", out)
        print("  appended to audit trail")
    except Exception:                                          # noqa: BLE001
        pass

    print("\n" + "=" * 70)
    print("Use the POINT-IN-TIME figure as your benchmark from here.")
    print("It is the honest one. Update BACKLOG.md accordingly.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
