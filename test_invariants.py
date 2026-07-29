"""Invariant test suite — property-based, adversarial, randomised.

Rationale
---------
Re-reading code finds fewer bugs with each pass. Running it against thousands
of randomised and adversarial inputs does not. This suite asserts *properties*
that must hold for every possible input, then hammers them.

The most valuable check here is the lookahead detector: it computes a value at
bar i on the full series, then recomputes it on a series truncated at i, and
demands they match exactly. Any future leakage shows up immediately, which is
the failure mode most likely to silently inflate results.

Run:  python test_invariants.py [iterations]
"""

from __future__ import annotations

import datetime as dt
import sys
import traceback

import numpy as np
import pandas as pd

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    try:
        fn()
        RESULTS.append((name, True, ""))
    except AssertionError as e:
        RESULTS.append((name, False, str(e) or "assertion failed"))
    except Exception as e:                                     # noqa: BLE001
        RESULTS.append((name, False, f"{type(e).__name__}: {e}\n"
                                     f"{traceback.format_exc(limit=3)}"))


# --------------------------------------------------------------------------
# Data generators, including adversarial cases
# --------------------------------------------------------------------------
def gen_ohlcv(n=500, seed=0, *, drift=0.0005, vol=0.015, pathology=None):
    """Generate OHLCV, optionally with a specific pathology injected."""
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    idx = pd.bdate_range("2022-01-03", periods=n)
    df = pd.DataFrame({
        "Open": c * (1 + rng.normal(0, 0.003, n)),
        "High": c * (1 + abs(rng.normal(0, 0.006, n))),
        "Low": c * (1 - abs(rng.normal(0, 0.006, n))),
        "Close": c,
        "Volume": rng.integers(1e5, 5e6, n).astype(float),
    }, index=idx)

    if pathology == "flat":
        df[["Open", "High", "Low", "Close"]] = 100.0
    elif pathology == "zero_volume":
        df["Volume"] = 0.0
    elif pathology == "gap":
        df.iloc[n // 2:, df.columns.get_loc("Close")] *= 3.0
    elif pathology == "single_price":
        df["Close"] = df["Close"].iloc[0]
    elif pathology == "tiny_price":
        df[["Open", "High", "Low", "Close"]] *= 0.001
    elif pathology == "huge_price":
        df[["Open", "High", "Low", "Close"]] *= 1e6
    elif pathology == "nan_tail":
        df.iloc[-5:, :] = np.nan
    elif pathology == "monotonic_up":
        base = 100 * (1.01 ** np.arange(n))
        df["Close"] = base
        df["Open"] = base * 0.999
        df["High"] = base * 1.002
        df["Low"] = base * 0.998
    return df


# --------------------------------------------------------------------------
# INVARIANT 1 — no lookahead anywhere
# --------------------------------------------------------------------------
def test_no_lookahead(iters: int) -> None:
    """A value computed at bar i must not change when future bars are added.

    This is the single most important property in the system. Any violation
    silently inflates every historical result.
    """
    import indicators as ind
    import momentum as momo
    from backtest import _score_at, _momentum_at
    import factor_analysis as fa

    rng = np.random.default_rng(99)
    for k in range(iters):
        n = int(rng.integers(400, 800))
        df = gen_ohlcv(n, seed=k, drift=float(rng.normal(0.0005, 0.001)))
        bench = gen_ohlcv(n, seed=k + 10_000)
        i = int(rng.integers(320, n - 30))

        full_e = ind.enrich(df)
        full_b = ind.enrich(bench)
        trunc_e = ind.enrich(df.iloc[: i + 1])
        trunc_b = ind.enrich(bench.iloc[: i + 1])

        # momentum
        a = momo.raw_momentum(full_e, i)
        b = momo.raw_momentum(trunc_e, i)
        assert np.isclose(a, b, equal_nan=True), \
            f"momentum lookahead at i={i}: full={a} trunc={b}"

        # v1 composite
        a = _score_at(full_e, i, full_b, "mid")
        b = _score_at(trunc_e, i, trunc_b, "mid")
        assert np.isclose(a, b, atol=1e-6, equal_nan=True), \
            f"_score_at lookahead at i={i}: full={a} trunc={b}"

        # control factors
        fa_full = fa._factors_at(full_e, i, full_b)
        fa_trunc = fa._factors_at(trunc_e, i, trunc_b)
        if fa_full and fa_trunc:
            for key in fa_full:
                assert np.isclose(fa_full[key], fa_trunc[key], atol=1e-6), \
                    f"factor '{key}' lookahead at i={i}"


def test_no_lookahead_signals(iters: int) -> None:
    """Every candidate signal must be causal."""
    import indicators as ind
    from signals import SIGNALS

    rng = np.random.default_rng(7)
    for k in range(iters):
        n = int(rng.integers(450, 700))
        df = gen_ohlcv(n, seed=k + 500)
        bench = gen_ohlcv(n, seed=k + 900)
        i = int(rng.integers(340, n - 20))
        fe, fb = ind.enrich(df), ind.enrich(bench)
        te, tb = ind.enrich(df.iloc[: i + 1]), ind.enrich(bench.iloc[: i + 1])

        for name, fn in SIGNALS.items():
            a, b = fn(fe, i, fb), fn(te, i, tb)
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            assert np.isclose(a, b, rtol=1e-5, atol=1e-8), \
                f"signal '{name}' lookahead at i={i}: full={a} trunc={b}"


def test_no_lookahead_regime(iters: int) -> None:
    """Regime state at bar i must not depend on later bars."""
    import indicators as ind
    import regime as rg

    rng = np.random.default_rng(31)
    for k in range(iters):
        n = int(rng.integers(400, 900))
        bench = gen_ohlcv(n, seed=k + 2000)
        i = int(rng.integers(250, n - 5))
        full = rg.classify_at(ind.enrich(bench), i)
        trunc = rg.classify_at(ind.enrich(bench.iloc[: i + 1]), i)
        assert full == trunc, f"regime lookahead at i={i}: {full} vs {trunc}"


# --------------------------------------------------------------------------
# INVARIANT 2 — bounded outputs
# --------------------------------------------------------------------------
def test_score_bounds(iters: int) -> None:
    import scoring, momentum as momo

    rng = np.random.default_rng(3)
    paths = [None, "flat", "zero_volume", "gap", "single_price",
             "tiny_price", "huge_price", "monotonic_up"]
    for k in range(iters):
        p = paths[k % len(paths)]
        df = gen_ohlcv(int(rng.integers(300, 600)), seed=k, pathology=p)
        try:
            m = scoring.evaluate(df, None)
        except Exception:                                      # noqa: BLE001
            continue
        if m.get("Score") is not None:
            assert 0 <= m["Score"] <= 100, f"v1 Score {m['Score']} out of range ({p})"
        for comp in ("Trend", "Momentum", "Volume_S", "RelStrength", "Setup"):
            if comp in m and m[comp] is not None:
                assert 0 <= m[comp] <= 100, f"{comp}={m[comp]} out of range ({p})"


def test_rank_percentile_bounds(iters: int) -> None:
    import momentum as momo

    rng = np.random.default_rng(11)
    for k in range(iters):
        size = int(rng.integers(5, 60))
        frames = {f"S{j}.NS": gen_ohlcv(450, seed=k * 100 + j,
                                        drift=float(rng.normal(0.0005, 0.0015)))
                  for j in range(size)}
        r = momo.rank_universe(frames, min_turnover_cr=0.0,
                               require_above_50ema=False, min_momentum=None)
        if r.empty:
            continue
        assert r["Score"].between(0, 100).all(), "percentile out of 0-100"
        assert r["Momentum"].is_monotonic_decreasing, \
            "score ordering must track momentum ordering"
        assert r["Ticker"].is_unique, "duplicate tickers in ranking"


def test_ic_bounds(iters: int) -> None:
    from factor_analysis import _spearman

    rng = np.random.default_rng(5)
    for _ in range(iters):
        n = int(rng.integers(5, 200))
        a = rng.normal(size=n)
        b = rng.normal(size=n)
        r = _spearman(a, b)
        if np.isfinite(r):
            assert -1.0001 <= r <= 1.0001, f"correlation {r} outside [-1,1]"
        # perfect correlation must read as 1
        assert np.isclose(_spearman(a, a), 1.0, atol=1e-9), "self-correlation != 1"


# --------------------------------------------------------------------------
# INVARIANT 3 — risk controls cannot be bypassed
# --------------------------------------------------------------------------
def test_regime_gate_never_leaks(iters: int) -> None:
    import bucket as bk, regime as rg

    rng = np.random.default_rng(17)
    for k in range(iters):
        n = int(rng.integers(5, 80))
        ranked = pd.DataFrame({
            "Ticker": [f"T{j}" for j in range(n)],
            "Tier": rng.choice(["large", "mid", "small"], n),
            "Score": rng.integers(0, 101, n),
            "Close": rng.uniform(10, 5000, n),
            "RSI": rng.uniform(20, 85, n),
            "ATR_pct": rng.uniform(0.5, 9.0, n),
        }).sort_values("Score", ascending=False)
        state = ["risk_on", "neutral", "risk_off"][k % 3]
        reg = rg.Regime(state, state != "risk_off", state == "risk_on",
                        float(rng.normal(0, 5)), float(rng.uniform(0, 1)))
        b = bk.build(ranked, reg, size=int(rng.integers(1, 20)),
                     max_per_sector=99, min_score=float(rng.integers(0, 80)))
        if b.is_empty:
            continue
        assert set(b.picks["Tier"]) <= reg.allowed_tiers, \
            f"regime {state} leaked tiers {set(b.picks['Tier'])}"
        cap = bk.REGIME_SIZE_CAP.get(state)
        if cap:
            assert b.actual_size <= cap, f"{state} cap {cap} breached: {b.actual_size}"
        assert b.picks["Rank"].is_monotonic_increasing, "ranks out of order"


def test_sector_cap_never_breached(iters: int) -> None:
    import bucket as bk, regime as rg

    rng = np.random.default_rng(23)
    for k in range(iters):
        n = int(rng.integers(10, 90))
        tickers = [f"T{j}" for j in range(n)]
        sectors = {t: f"SEC{rng.integers(0, 5)}" for t in tickers}
        ranked = pd.DataFrame({
            "Ticker": tickers,
            "Tier": ["large"] * n,
            "Score": rng.integers(50, 101, n),
            "Close": rng.uniform(10, 5000, n),
            "RSI": rng.uniform(30, 80, n),
            "ATR_pct": rng.uniform(1, 5, n),
        }).sort_values("Score", ascending=False)
        cap = int(rng.integers(1, 4))
        reg = rg.Regime("risk_on", True, True, 3.0, 0.6)
        b = bk.build(ranked, reg, size=20, max_per_sector=cap,
                     min_score=0, sector_lookup=sectors)
        if b.is_empty:
            continue
        counts = b.picks["Ticker"].map(sectors).value_counts()
        assert (counts <= cap).all(), f"sector cap {cap} breached: {dict(counts)}"


def test_position_limits(iters: int) -> None:
    import tiers as tr

    rng = np.random.default_rng(29)
    for _ in range(iters):
        tier = ["large", "mid", "small"][int(rng.integers(0, 3))]
        cap = float(rng.uniform(10_000, 5_000_000))
        px = float(rng.uniform(1, 50_000))
        adv = float(rng.uniform(0, 5_000_000))
        lim = tr.position_limits(tier, cap, px, adv)
        p = tr.params(tier)
        assert lim["max_value"] <= cap * p["max_position_pct"] / 100 + 1e-6, \
            "capital cap exceeded"
        if adv > 0:
            assert lim["max_value"] <= adv * p["max_pct_of_adv"] / 100 * px + 1e-6, \
                "liquidity cap exceeded"
        assert lim["max_qty"] >= 0, "negative quantity"
        assert lim["max_qty"] * px <= lim["max_value"] + px, "qty exceeds value cap"


# --------------------------------------------------------------------------
# INVARIANT 4 — backtest accounting
# --------------------------------------------------------------------------
def test_backtest_accounting(iters: int) -> None:
    import backtest as bt

    rng = np.random.default_rng(41)
    for k in range(iters):
        frames = {f"S{j}.NS": gen_ohlcv(700, seed=k * 50 + j,
                                        drift=float(rng.normal(0.0005, 0.0015)))
                  for j in range(int(rng.integers(3, 10)))}
        bench = gen_ohlcv(700, seed=k + 7777)
        model = ["momentum", "composite_v1"][k % 2]
        trades = bt.run(frames, bench, min_score=float(rng.integers(40, 80)),
                        hold_bars=int(rng.integers(5, 30)), model=model)
        if trades.empty:
            continue
        assert (trades["exit_date"] >= trades["entry_date"]).all(), "exit before entry"
        assert (trades["bars_held"] >= 0).all(), "negative holding period"
        assert (trades["entry"] > 0).all() and (trades["exit"] > 0).all(), \
            "non-positive prices"
        assert (trades["net_r"] <= trades["gross_r"] + 1e-9).all(), \
            "net R exceeds gross R — costs applied with wrong sign"
        stops = trades[trades["exit_reason"] == "stop"]
        if not stops.empty:
            assert (stops["exit"] <= stops["entry"] + 1e-6).all(), \
                "stop exit above entry price"
            assert (stops["gross_r"] <= 0.001).all(), "stopped trade with positive R"


# --------------------------------------------------------------------------
# INVARIANT 5 — forward log integrity
# --------------------------------------------------------------------------
def test_forward_log_integrity(iters: int) -> None:
    import forward_log as flog

    rng = np.random.default_rng(53)
    for k in range(iters):
        n = int(rng.integers(1, 15))
        picks = pd.DataFrame({
            "Ticker": [f"T{j}" for j in range(n)],
            "Tier": rng.choice(["large", "mid", "small"], n),
            "Score": rng.integers(0, 101, n),
            "Close": rng.uniform(1, 10_000, n),
        })
        days_ago = int(rng.integers(1, 120))
        d = dt.date.today() - dt.timedelta(days=days_ago)
        log = flog.record_snapshot(flog.empty_log(), picks, regime_state="risk_on",
                                   horizon=int(rng.integers(5, 30)),
                                   top_n=n, snapshot_date=d)
        assert len(log) == n, "snapshot row count mismatch"
        assert (log["status"] == "open").all(), "new picks not marked open"

        # duplicate guard
        again = flog.record_snapshot(log, picks, regime_state="risk_on",
                                     horizon=15, top_n=n, snapshot_date=d)
        assert len(again) == len(log), "duplicate snapshot on same date"

        idx = pd.bdate_range(d - dt.timedelta(days=5), dt.date.today())
        if len(idx) < 3:
            continue
        hist = {t: pd.DataFrame(
            {"Close": 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, len(idx))))},
            index=idx) for t in picks["Ticker"]}
        lookup = {t: float(h["Close"].iloc[-1]) for t, h in hist.items()}
        ev, filled = flog.evaluate_open(log, lookup, price_history=hist)

        done = ev[ev["status"] == "evaluated"]
        if not done.empty:
            hd = pd.to_numeric(done["holding_days_actual"], errors="coerce").dropna()
            assert (hd >= 0).all(), "negative holding period recorded"
            assert done["eval_method"].isin(["at_target", "latest_price"]).all(), \
                "unknown eval_method"
            assert (pd.to_numeric(done["price_at_eval"]) > 0).all(), \
                "non-positive evaluation price"

        # round-trip must preserve schema
        rt = flog.from_csv(flog.to_csv(ev))
        assert list(rt.columns) == flog.COLUMNS, "CSV round-trip changed schema"
        assert len(rt) == len(ev), "CSV round-trip lost rows"


# --------------------------------------------------------------------------
# INVARIANT 6 — sentiment and data quality
# --------------------------------------------------------------------------
def test_sentiment_bounds(iters: int) -> None:
    import sentiment as sent

    words = ["surge", "plunge", "profit", "loss", "beat", "miss", "not", "but",
             "despite", "record", "fraud", "dividend", "shares", "the", "and",
             "very", "slightly", "fails", "wins", "downgrade", "upgrade"]
    rng = np.random.default_rng(61)
    for _ in range(iters):
        text = " ".join(rng.choice(words, int(rng.integers(1, 25))))
        r = sent.score_text(text)
        assert -5.0001 <= r.score <= 5.0001, f"sentiment {r.score} out of range"
        assert r.label in ("pos", "neg", "neu"), f"bad label {r.label}"
    # empty and pathological inputs
    for t in ("", "   ", "!!!", "123 456", "a" * 500):
        r = sent.score_text(t)
        assert r.label in ("pos", "neg", "neu")


def test_data_quality_detects(iters: int) -> None:
    import data_quality as dq

    for k in range(min(iters, 40)):
        clean = gen_ohlcv(400, seed=k)
        assert not dq.audit_frame(clean), f"clean frame flagged: {dq.audit_frame(clean)}"
        bad = clean.copy()
        bad.iloc[200, bad.columns.get_loc("Close")] *= 5.0
        assert dq.audit_frame(bad), "500% jump not detected"
        neg = clean.copy()
        neg.iloc[50, neg.columns.get_loc("Close")] = -1
        assert any("non-positive" in i for i in dq.audit_frame(neg)), \
            "negative price not detected"


# --------------------------------------------------------------------------
# INVARIANT 7 — universe trimming is unbiased
# --------------------------------------------------------------------------
def test_no_dataframe_truthiness(iters: int) -> None:
    """Guard against `df_a or df_b`, which raises on ambiguous truthiness.

    This exact bug was introduced twice — first in forward_log.py, then again
    in daily_tracker.py. A static check is cheaper than finding it a third time
    at runtime.
    """
    import pathlib
    import re

    offenders = []
    pattern = re.compile(r"\.get\([^)]*\)\s+or\s+\w+\.get\(")
    # Float equality: std()/mean()/sum() compared with == is almost always a
    # bug. The standard deviation of identical values is ~1e-14, not 0.0.
    float_eq = re.compile(r"float\([^)]*\.(std|mean|sum|var)\([^)]*\)\)\s*==")
    for p in sorted(pathlib.Path(".").glob("*.py")):
        if p.name.startswith("test_"):
            continue
        for n, line in enumerate(p.read_text().splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            if pattern.search(line):
                offenders.append(f"{p.name}:{n}: [dict-or] {line.strip()[:60]}")
            if float_eq.search(line):
                offenders.append(f"{p.name}:{n}: [float-eq] {line.strip()[:60]}")
    assert not offenders, (
        "Fragile patterns found:\n"
        "  [dict-or]  a.get(x) or b.get(y) — raises on DataFrames, and discards "
        "legitimate 0.0 values\n"
        "  [float-eq] float(...std()) == — floating point is never exactly equal\n\n"
        + "\n  ".join(offenders)
    )


def test_daily_tracker_integrity(iters: int) -> None:
    """Daily tracker must be idempotent and never corrupt the weekly log."""
    import datetime as _dt
    import daily_tracker as dtm

    rng = np.random.default_rng(83)
    for k in range(min(iters, 25)):
        n = int(rng.integers(1, 8))
        picks = pd.DataFrame({
            "Rank": range(1, n + 1),
            "Ticker": [f"T{j}" for j in range(n)],
            "Tier": ["large"] * n,
            "Score": rng.integers(50, 101, n),
            "Momentum": rng.uniform(-20, 150, n).round(1),
            "Close": rng.uniform(10, 5000, n).round(2),
        })
        obs = dtm._empty(dtm.OBS_COLUMNS)
        d = _dt.date(2026, 6, 1) + _dt.timedelta(days=k)

        obs, added = dtm.record_bucket(obs, picks, regime_state="risk_on", obs_date=d)
        assert added == n, "wrong observation count"
        obs, again = dtm.record_bucket(obs, picks, regime_state="risk_on", obs_date=d)
        assert again == 0, "duplicate observation recorded for the same date"

        obs2, empty_added = dtm.record_bucket(
            dtm._empty(dtm.OBS_COLUMNS), pd.DataFrame(),
            regime_state="risk_off", obs_date=d)
        assert empty_added == 1, "empty bucket must still be recorded"

        assert list(obs.columns) == dtm.OBS_COLUMNS, "observation schema drifted"


def test_partial_bar_immunity(iters: int) -> None:
    """A NaN in the final bar must not silently empty the screener.

    yfinance returns the current session with Close still NaN during market
    hours. `dropna(how="all")` leaves that row, and every last-row read then
    yields NaN — `NaN > NaN` is False, so Above_50EMA becomes False for EVERY
    stock and the turnover filter rejects everything. The screener returns
    nothing and gives no indication why.

    This is the single most damaging silent-failure mode found in the system.
    """
    import momentum as momo

    rng = np.random.default_rng(97)
    for k in range(min(iters, 20)):
        # Must exceed momentum.MIN_HISTORY_BARS (378) even after the poisoned
        # bar is dropped, or an empty result is correct rather than a bug.
        n = int(rng.integers(450, 700))
        frames = {}
        for j in range(12):
            df = gen_ohlcv(n, seed=k * 20 + j, drift=0.001)
            # Poison the last bar the way yfinance does
            df.iloc[-1, df.columns.get_loc("Close")] = np.nan
            df.iloc[-1, df.columns.get_loc("Volume")] = np.nan
            frames[f"S{j}.NS"] = df

        # Unsanitised input must be refused, never silently mis-ranked
        r = momo.rank_universe(frames, min_turnover_cr=0.0,
                               require_above_50ema=False, min_momentum=None)
        if not r.empty:
            assert not r["Turnover_Cr"].isna().any(), \
                "NaN turnover leaked into the ranking"
            assert r["Above_50EMA"].sum() > 0, \
                "every Above_50EMA is False — the NaN comparison bug is back"

        # Sanitised input must work normally
        clean = {t: d.dropna(subset=["Close"]) for t, d in frames.items()}
        rc = momo.rank_universe(clean, min_turnover_cr=0.0,
                                require_above_50ema=False, min_momentum=None)
        assert not rc.empty, "sanitised frames should still rank"
        assert not rc["Turnover_Cr"].isna().any(), "NaN turnover after cleaning"
        assert rc["Above_50EMA"].sum() > 0, \
            "no stock above its 50 EMA — implausible, indicates the NaN bug"


def test_universe_trim_unbiased(iters: int) -> None:
    import universe as uni
    import string

    rng = np.random.default_rng(67)
    for k in range(min(iters, 50)):
        tickers = tuple(sorted(
            f"{c}{i:03d}.NS" for c in string.ascii_uppercase for i in range(8)))
        keep = int(rng.integers(20, 150))
        trimmed, how = uni.trim_universe(tickers, keep, method="random",
                                         seed=int(rng.integers(0, 10_000)))
        assert len(trimmed) == keep, "wrong trim size"
        assert len(set(trimmed)) == keep, "duplicates after trim"
        letters = {t[0] for t in trimmed}
        assert len(letters) >= 15, \
            f"alphabetical bias: only {len(letters)} distinct first letters"


def main(iters: int = 100) -> int:
    print("=" * 68)
    print(f"SwingScope invariant suite — {iters} iterations per property")
    print("=" * 68)

    suites = [
        ("no lookahead: indicators & composite", test_no_lookahead),
        ("no lookahead: all 12 signals", test_no_lookahead_signals),
        ("no lookahead: regime", test_no_lookahead_regime),
        ("bounds: v1 score components", test_score_bounds),
        ("bounds: percentile ranking", test_rank_percentile_bounds),
        ("bounds: correlation", test_ic_bounds),
        ("risk: regime gate never leaks", test_regime_gate_never_leaks),
        ("risk: sector cap never breached", test_sector_cap_never_breached),
        ("risk: position limits", test_position_limits),
        ("backtest: accounting consistency", test_backtest_accounting),
        ("forward log: integrity", test_forward_log_integrity),
        ("sentiment: bounded", test_sentiment_bounds),
        ("data quality: detects defects", test_data_quality_detects),
        ("universe: trim unbiased", test_universe_trim_unbiased),
        ("partial bar immunity", test_partial_bar_immunity),
        ("static: no DataFrame truthiness", test_no_dataframe_truthiness),
        ("daily tracker: integrity", test_daily_tracker_integrity),
    ]
    for name, fn in suites:
        print(f"  running {name} …", flush=True)
        check(name, lambda f=fn: f(iters))

    print("\n" + "=" * 68)
    failed = [(n, m) for n, ok, m in RESULTS if not ok]
    for name, ok, msg in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"        {msg.splitlines()[0]}")
    print("=" * 68)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} properties held")
    if failed:
        print("\nFAILURES:")
        for n, m in failed:
            print(f"\n--- {n} ---\n{m}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 100))
