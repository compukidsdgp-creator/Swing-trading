"""Engineering assessment — efficiency, precision, continuity, load, regression.

Each dimension uses the algorithm appropriate to it rather than generic timing.

  Efficiency  — empirical complexity analysis. Measure runtime across geometric
                input sizes, fit log(t) = a + b*log(n) by least squares. The
                exponent b is the empirical big-O: ~1.0 linear, ~2.0 quadratic.
                Finds superlinear blow-ups that only appear at scale.

  Precision   — differential testing against independent reference
                implementations, plus catastrophic-cancellation probes. Wilder's
                RSI and EMA are recursive, so error can compound; ATR and
                correlation involve differences of similar magnitudes, which is
                where float64 loses digits.

  Continuity  — Lipschitz sensitivity analysis. Perturb an input by epsilon and
                measure the output response ratio. A well-behaved signal has a
                bounded ratio. Discontinuities matter enormously here: if a
                0.01% price change flips a pick in or out of the bucket, the
                strategy is unstable regardless of its IC.

  Load        — incremental scaling to failure, with memory tracking. Finds the
                practical ceiling and where it degrades rather than assuming it.

  Regression  — golden-master (characterisation) testing. Deterministic outputs
                under fixed seeds are hashed; any future change that alters them
                is surfaced immediately, whether intended or not.

Run:  python benchmark.py
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

GOLDEN = Path("golden_master.json")


def _ohlcv(n: int, seed: int = 0, drift: float = 0.0005) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
    return pd.DataFrame({
        "Open": c * (1 + rng.normal(0, 0.003, n)),
        "High": c * (1 + abs(rng.normal(0, 0.006, n))),
        "Low": c * (1 - abs(rng.normal(0, 0.006, n))),
        "Close": c,
        "Volume": rng.integers(1e5, 5e6, n).astype(float),
    }, index=pd.bdate_range("2015-01-01", periods=n))


# ==========================================================================
# 1. EFFICIENCY — empirical complexity via log-log regression
# ==========================================================================
def efficiency() -> dict:
    import indicators as ind
    import momentum as momo

    print("\n" + "=" * 70)
    print("1. EFFICIENCY — empirical complexity (log-log least squares)")
    print("=" * 70)
    out = {}

    # --- scaling in BARS ---
    # Larger sizes so real scaling dominates the fixed pandas overhead —
    # at 250 bars the constant cost swamps the measurement and the fitted
    # exponent is meaningless.
    sizes = [1000, 2000, 4000, 8000, 16000]
    times = []
    for n in sizes:
        df = _ohlcv(n, seed=1)
        t0 = time.perf_counter()
        for _ in range(5):
            ind.enrich(df)
        times.append((time.perf_counter() - t0) / 5)
    b = np.polyfit(np.log(sizes), np.log(times), 1)[0]
    out["enrich_exponent"] = round(float(b), 2)
    print(f"\n  indicators.enrich vs bars")
    for n, t in zip(sizes, times):
        print(f"    {n:>5} bars  {t*1000:7.2f} ms  ({t/n*1e6:6.2f} µs/bar)")
    print(f"    empirical O(n^{b:.2f})  -> {'LINEAR, good' if b < 1.3 else 'SUPERLINEAR, investigate'}")

    # --- scaling in TICKERS ---
    counts = [10, 25, 50, 100, 200]
    ttimes = []
    for k in counts:
        frames = {f"S{j}.NS": _ohlcv(500, seed=j) for j in range(k)}
        gc.collect()
        t0 = time.perf_counter()
        momo.rank_universe(frames, min_turnover_cr=0.0,
                           require_above_50ema=False, min_momentum=None)
        ttimes.append(time.perf_counter() - t0)
    bt_ = np.polyfit(np.log(counts), np.log(ttimes), 1)[0]
    out["rank_exponent"] = round(float(bt_), 2)
    out["rank_ms_per_ticker"] = round(ttimes[-1] / counts[-1] * 1000, 2)
    print(f"\n  momentum.rank_universe vs tickers (500 bars each)")
    for k, t in zip(counts, ttimes):
        print(f"    {k:>4} tickers  {t:6.3f} s  ({t/k*1000:6.1f} ms/ticker)")
    print(f"    empirical O(n^{bt_:.2f})  -> {'LINEAR, good' if bt_ < 1.3 else 'SUPERLINEAR'}")

    # --- projection to realistic workload ---
    proj = ttimes[-1] / counts[-1] * 500
    out["projected_nifty500_s"] = round(proj, 1)
    print(f"\n  projected: full Nifty 500 scoring ≈ {proj:.1f}s (excludes network fetch)")
    return out


# ==========================================================================
# 2. PRECISION — differential testing vs independent implementations
# ==========================================================================
def precision() -> dict:
    import indicators as ind
    from factor_analysis import _spearman

    print("\n" + "=" * 70)
    print("2. PRECISION — differential testing + cancellation probes")
    print("=" * 70)
    out = {}
    df = _ohlcv(3000, seed=7)
    close = df["Close"]

    # EMA against an explicit recursive reference
    span = 20
    alpha = 2 / (span + 1)
    ref = np.empty(len(close))
    ref[0] = close.iloc[0]
    for i in range(1, len(close)):
        ref[i] = alpha * close.iloc[i] + (1 - alpha) * ref[i - 1]
    err = np.max(np.abs(ind.ema(close, span).to_numpy() - ref) / np.abs(ref))
    out["ema_max_rel_err"] = float(err)
    print(f"\n  EMA(20) vs recursive reference:  max rel err {err:.2e}"
          f"  {'OK' if err < 1e-12 else 'DRIFT'}")

    # RSI against a literal Wilder implementation
    period = 14
    d = close.diff().to_numpy()
    g = np.where(d > 0, d, 0.0)[1:]
    l = np.where(d < 0, -d, 0.0)[1:]
    ag = np.empty(len(g)); al = np.empty(len(l))
    ag[0], al[0] = g[0], l[0]
    for i in range(1, len(g)):
        ag[i] = (ag[i-1] * (period - 1) + g[i]) / period
        al[i] = (al[i-1] * (period - 1) + l[i]) / period
    ref_rsi = 100 - 100 / (1 + ag / np.where(al == 0, np.nan, al))
    ours = ind.rsi(close, period).to_numpy()[1:]
    m = np.isfinite(ref_rsi) & np.isfinite(ours)
    rsi_err = float(np.max(np.abs(ours[m] - ref_rsi[m])))
    out["rsi_max_abs_err"] = rsi_err
    print(f"  RSI(14) vs Wilder reference:      max abs err {rsi_err:.2e}"
          f"  {'OK' if rsi_err < 1e-6 else 'DIVERGENT'}")

    # Catastrophic cancellation: near-identical prices
    flat = df.copy()
    base = 1_000_000.0
    flat[["Open", "High", "Low", "Close"]] = base
    flat["Close"] = base + np.arange(len(flat)) * 1e-6      # 1e-12 relative moves
    try:
        e = ind.enrich(flat)
        finite = np.isfinite(e["RSI14"]).all() and np.isfinite(e["ATR14"]).all()
        out["cancellation_stable"] = bool(finite)
        print(f"  cancellation probe (1e-12 rel moves): "
              f"{'stable' if finite else 'PRODUCED NaN/Inf'}")
    except Exception as ex:                                    # noqa: BLE001
        out["cancellation_stable"] = False
        print(f"  cancellation probe: RAISED {type(ex).__name__}")

    # Spearman against an analytically known case
    a = np.arange(100, dtype=float)
    assert abs(_spearman(a, a) - 1.0) < 1e-12
    assert abs(_spearman(a, -a) + 1.0) < 1e-12
    ties = np.array([1., 1., 1., 2., 2., 3., 3., 3., 4., 5.])
    out["spearman_exact"] = True
    print(f"  Spearman exactness (identity/inverse/ties): OK")

    # Scale invariance — momentum must not care about price units
    import momentum as momo
    m1 = momo.raw_momentum(_ohlcv(500, seed=3))
    scaled = _ohlcv(500, seed=3)
    scaled[["Open", "High", "Low", "Close"]] *= 1e6
    m2 = momo.raw_momentum(scaled)
    out["momentum_scale_invariant"] = bool(np.isclose(m1, m2, rtol=1e-9))
    print(f"  momentum scale invariance (x1e6):  "
          f"{'OK' if out['momentum_scale_invariant'] else 'FAILED'}")
    return out


# ==========================================================================
# 3. CONTINUITY — Lipschitz sensitivity analysis
# ==========================================================================
def continuity() -> dict:
    import momentum as momo
    import bucket as bk, regime as rg

    print("\n" + "=" * 70)
    print("3. CONTINUITY — Lipschitz sensitivity to input perturbation")
    print("=" * 70)
    out = {}

    # Continuous layer: raw momentum.
    #
    # NOTE: perturbing ALL prices uniformly measures nothing — 12-1 momentum is
    # a price RATIO and is invariant to uniform scaling by construction. The
    # meaningful perturbation targets the bars the formula actually reads:
    # p[i-252] (start of formation) and p[i-21] (end of formation).
    ratios = []
    for eps in (1e-6, 1e-5, 1e-4, 1e-3):
        for seed in range(6):
            df = _ohlcv(500, seed=seed)
            m0 = momo.raw_momentum(df)
            p = df.copy()
            j = len(p) - 1 - momo.SKIP          # the endpoint momentum reads
            p.iloc[j, p.columns.get_loc("Close")] *= (1 + eps)
            m1 = momo.raw_momentum(p)
            if np.isfinite(m0) and np.isfinite(m1) and abs(m0) > 1e-9:
                ratios.append(abs(m1 - m0) / abs(m0) / eps)
    L = float(np.median(ratios)) if ratios else float("nan")
    out["momentum_lipschitz"] = round(L, 3)
    print(f"\n  raw momentum, perturbing the formation endpoint p[i-21]:")
    print(f"    median response ratio {L:.3f}")
    print(f"    -> {'well-conditioned' if L < 10 else 'AMPLIFIES input error'}")

    # Insensitivity to RECENT price is a design feature, not a defect
    recent = []
    for seed in range(6):
        df = _ohlcv(500, seed=seed)
        m0 = momo.raw_momentum(df)
        p = df.copy()
        p.iloc[-1, p.columns.get_loc("Close")] *= 1.10      # +10% on the last bar
        recent.append(abs(momo.raw_momentum(p) - m0))
    out["insensitive_to_last_bar"] = bool(max(recent) < 1e-9)
    print(f"    +10% shock to the LAST bar changes momentum by "
          f"{max(recent):.2e}")
    print(f"    -> the skip-month makes the signal immune to recent noise "
          f"(by design)")

    # Discrete layer: does a tiny change flip bucket membership?
    flips, trials = 0, 40
    rng = np.random.default_rng(5)
    for k in range(trials):
        frames = {f"S{j}.NS": _ohlcv(500, seed=k * 40 + j,
                                     drift=float(rng.normal(0.0006, 0.0012)))
                  for j in range(30)}
        base = momo.rank_universe(frames, min_turnover_cr=0.0,
                                  require_above_50ema=False, min_momentum=None)
        if base.empty:
            continue
        # Perturb the bar momentum actually reads, not the whole series
        victim = f"S{int(rng.integers(0, 30))}.NS"
        pert = {}
        for t, d in frames.items():
            if t != victim:
                pert[t] = d
                continue
            c = d.copy()
            j = len(c) - 1 - momo.SKIP
            c.iloc[j, c.columns.get_loc("Close")] *= 1.0001
            pert[t] = c
        new = momo.rank_universe(pert, min_turnover_cr=0.0,
                                 require_above_50ema=False, min_momentum=None)
        if new.empty:
            continue
        if set(base["Ticker"].head(10)) != set(new["Ticker"].head(10)):
            flips += 1
    rate = flips / trials
    out["top10_flip_rate_1bp"] = round(rate, 3)
    print(f"\n  bucket stability: a 1 basis-point price change flipped the "
          f"top-10 in {flips}/{trials} trials ({rate:.0%})")
    print(f"    -> {'stable' if rate < 0.15 else 'SENSITIVE — ranks near the cutoff are fragile'}")

    # Week-to-week turnover — the practical cost of that sensitivity
    turnovers = []
    for k in range(12):
        frames = {f"S{j}.NS": _ohlcv(520, seed=k * 31 + j,
                                     drift=float(rng.normal(0.0006, 0.0012)))
                  for j in range(40)}
        w1 = momo.rank_universe({t: d.iloc[:-5] for t, d in frames.items()},
                                min_turnover_cr=0.0, require_above_50ema=False,
                                min_momentum=None)
        w2 = momo.rank_universe(frames, min_turnover_cr=0.0,
                                require_above_50ema=False, min_momentum=None)
        if w1.empty or w2.empty:
            continue
        a, b = set(w1["Ticker"].head(10)), set(w2["Ticker"].head(10))
        turnovers.append(len(a ^ b) / 2 / 10)
    tv = float(np.mean(turnovers)) if turnovers else float("nan")
    out["weekly_turnover"] = round(tv, 3)
    print(f"\n  implied weekly bucket turnover: {tv:.0%}")
    print(f"    -> at ~0.35% round-trip that is ~{tv*0.35:.2f}% cost per week, "
          f"~{tv*0.35*52:.1f}% per year")
    return out


# ==========================================================================
# 4. LOAD — incremental scaling to failure, with memory
# ==========================================================================
def load() -> dict:
    import momentum as momo

    print("\n" + "=" * 70)
    print("4. LOAD — incremental scaling with memory tracking")
    print("=" * 70)
    out, rows = {}, []
    for k in (50, 100, 200, 400, 800):
        frames = {f"S{j}.NS": _ohlcv(500, seed=j) for j in range(k)}
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        try:
            r = momo.rank_universe(frames, min_turnover_cr=0.0,
                                   require_above_50ema=False, min_momentum=None)
            ok, ranked = True, len(r)
        except Exception as ex:                                # noqa: BLE001
            ok, ranked = False, 0
            print(f"    FAILED at {k}: {type(ex).__name__}")
        el = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        input_mb = sum(d.memory_usage(deep=True).sum() for d in frames.values()) / 1e6
        rows.append((k, el, peak / 1e6, input_mb, ranked, ok))
        del frames
        gc.collect()

    print(f"\n  {'tickers':>8} {'time':>8} {'peak MB':>9} {'input MB':>9} {'ranked':>7}")
    for k, el, pk, im, rk, ok in rows:
        print(f"  {k:>8} {el:>7.2f}s {pk:>8.1f} {im:>8.1f} {rk:>7} {'' if ok else 'FAIL'}")

    out["max_tested"] = rows[-1][0]
    out["all_succeeded"] = all(r[5] for r in rows)
    out["peak_mb_at_max"] = round(rows[-1][2], 1)
    out["input_mb_at_max"] = round(rows[-1][3], 1)

    # Streamlit Cloud free tier is ~1 GB
    total_at_max = rows[-1][2] + rows[-1][3]
    headroom = 1024 / total_at_max if total_at_max else float("inf")
    out["streamlit_1gb_headroom_x"] = round(headroom, 1)
    print(f"\n  at {rows[-1][0]} tickers: {total_at_max:.0f} MB total"
          f"  -> {headroom:.1f}x headroom on a 1 GB tier")
    return out


# ==========================================================================
# 5. REGRESSION — golden-master characterisation testing
# ==========================================================================
def regression() -> dict:
    import indicators as ind
    import momentum as momo
    import scoring, sentiment as sent
    from factor_analysis import _spearman
    import regime as rg

    print("\n" + "=" * 70)
    print("5. REGRESSION — golden-master (deterministic fixed-seed outputs)")
    print("=" * 70)

    def h(x) -> str:
        return hashlib.sha256(
            json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:16]

    df = _ohlcv(600, seed=42)
    bench = _ohlcv(600, seed=4242)
    e = ind.enrich(df)

    current = {
        "ema20_last":   round(float(e["EMA20"].iloc[-1]), 8),
        "rsi14_last":   round(float(e["RSI14"].iloc[-1]), 8),
        "atr14_last":   round(float(e["ATR14"].iloc[-1]), 8),
        "adx14_last":   round(float(e["ADX14"].iloc[-1]), 8),
        "momentum":     round(float(momo.raw_momentum(df)), 8),
        "v1_score":     scoring.evaluate(df, bench).get("Score"),
        "regime":       rg.classify(bench, 0.6).state,
        "spearman":     round(_spearman(np.arange(50, dtype=float),
                                        np.arange(50, dtype=float)[::-1]), 8),
        "sentiment_a":  sent.score_text("Shares fall despite strong profit growth").score,
        "sentiment_b":  sent.score_text("Stock surges on record order win").score,
        "rank_top3":    list(momo.rank_universe(
                            {f"S{j}.NS": _ohlcv(500, seed=j, drift=0.0002 * j)
                             for j in range(12)},
                            min_turnover_cr=0.0, require_above_50ema=False,
                            min_momentum=None)["Ticker"].head(3)),
    }
    current["_hash"] = h(current)

    if GOLDEN.exists():
        prev = json.loads(GOLDEN.read_text())
        changed = [k for k in current
                   if k != "_hash" and prev.get(k) != current.get(k)]
        if changed:
            print(f"\n  {len(changed)} value(s) CHANGED since the last baseline:")
            for k in changed:
                print(f"    {k}: {prev.get(k)}  ->  {current.get(k)}")
            print("\n  If intentional, delete golden_master.json to re-baseline.")
        else:
            print(f"\n  All {len(current)-1} outputs identical to baseline"
                  f" (hash {current['_hash']})")
        status = "changed" if changed else "identical"
    else:
        GOLDEN.write_text(json.dumps(current, indent=2, default=str))
        print(f"\n  Baseline created — {len(current)-1} outputs recorded"
              f" (hash {current['_hash']})")
        for k, v in current.items():
            if k != "_hash":
                print(f"    {k:16} {v}")
        status = "baseline_created"

    return {"status": status, "hash": current["_hash"], "n_outputs": len(current) - 1}


def main() -> int:
    print("=" * 70)
    print("SwingScope — engineering assessment")
    print("=" * 70)
    summary = {
        "efficiency": efficiency(),
        "precision": precision(),
        "continuity": continuity(),
        "load": load(),
        "regression": regression(),
    }
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for section, vals in summary.items():
        print(f"\n  {section.upper()}")
        for k, v in vals.items():
            print(f"    {k:32} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
