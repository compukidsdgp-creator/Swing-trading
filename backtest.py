"""Walk-forward backtest.

The point of this module is to answer one question honestly: does the score
predict anything, or does it just describe charts that already went up?

Lookahead discipline
--------------------
Indicators (EMA, RSI, MACD, ATR, ADX, Bollinger) are causal — the value at bar
i uses only bars <= i. So enriching the full series once and *reading* index i
introduces no lookahead. The parts that would leak are the rolling-window
lookups in the setup score (52-week high, recent swing high), so those are
explicitly sliced to [:i+1] here.

Entry is at the *next* bar's open after a signal, never the signal bar's close.
Exits check the low against the stop before checking the holding period, so a
bar that both stops out and would have hit target is recorded as a loss — the
pessimistic assumption, which is the right one.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

import indicators as ind
import regime as rg
import tiers as tr

MIN_HISTORY = 260          # need 252 for the 52w window plus buffer


@dataclass
class Trade:
    ticker: str
    tier: str
    regime: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry: float
    exit: float
    stop: float
    score: int
    bars_held: int
    exit_reason: str
    gross_r: float
    net_r: float
    ret_pct: float


def _score_at(e: pd.DataFrame, i: int, be: pd.DataFrame | None, tier: str) -> float:
    """Composite score at bar i using only data up to i. Mirrors scoring.evaluate."""
    row = e.iloc[i]
    close = float(row["Close"])
    if not np.isfinite(close) or close <= 0:
        return 0.0

    # --- Trend ---
    t = 0.0
    if close > row["EMA20"]:
        t += 20
    if close > row["EMA50"]:
        t += 25
    if close > row["EMA200"]:
        t += 15
    if row["EMA20"] > row["EMA50"]:
        t += 20
    adx = float(row.get("ADX14", 0) or 0)
    if 20 <= adx <= 40:
        t += 20
    elif adx > 40:
        t += 10
    elif adx >= 15:
        t += 10
    t = min(t, 100.0)

    # --- Momentum (tier-aware RSI band) ---
    m = 0.0
    r = float(row["RSI14"])
    lo, hi = tr.params(tier)["rsi_peak"]
    if lo <= r <= hi:
        m += 45
    elif hi < r <= hi + 8:
        m += 30
    elif lo - 6 <= r < lo:
        m += 30
    elif r > hi + 8:
        m += 10
    else:
        m += 5
    if row["MACD_Hist"] > 0:
        m += 30
    if row["MACD"] > row["MACD_Signal"]:
        m += 25
    m = min(m, 100.0)

    # --- Volume ---
    ratio = float(row.get("Vol_Ratio", np.nan))
    if not np.isfinite(ratio):
        v = 0.0
    elif ratio >= 2.0:
        v = 100.0
    elif ratio >= 1.5:
        v = 85.0
    elif ratio >= 1.2:
        v = 70.0
    elif ratio >= 0.9:
        v = 50.0
    else:
        v = 25.0
    win = e.iloc[max(0, i - 9): i + 1]
    up_v = win.loc[win["Close"] >= win["Open"], "Volume"].sum()
    dn_v = win.loc[win["Close"] < win["Open"], "Volume"].sum()
    if dn_v > 0 and up_v / dn_v > 1.3:
        v = min(v + 10, 100.0)

    # --- Relative strength ---
    if be is None or i < 60 or len(be) <= i:
        rs = 50.0
    else:
        try:
            s20 = (close / float(e["Close"].iloc[i - 20]) - 1) * 100
            b20 = (float(be["Close"].iloc[i]) / float(be["Close"].iloc[i - 20]) - 1) * 100
            s60 = (close / float(e["Close"].iloc[i - 60]) - 1) * 100
            b60 = (float(be["Close"].iloc[i]) / float(be["Close"].iloc[i - 60]) - 1) * 100
            rs = np.clip(0.65 * (50 + (s20 - b20) * 3.5) + 0.35 * (50 + (s60 - b60) * 1.8), 0, 100)
        except (IndexError, ZeroDivisionError, ValueError):
            rs = 50.0

    # --- Setup (explicitly sliced to avoid lookahead) ---
    s = 0.0
    bw_hist = e["BB_Width"].iloc[max(0, i - 125): i + 1].dropna()
    if len(bw_hist) > 30:
        pct = float((bw_hist.iloc[-1] <= bw_hist).mean()) * 100
        s += min(40.0, pct * 0.4)
    hi_252 = float(e["High"].iloc[max(0, i - 251): i + 1].max())
    dist = (close / hi_252 - 1) * 100 if hi_252 > 0 else -100
    if -3 <= dist <= 0:
        s += 35
    elif -8 < dist < -3:
        s += 28
    elif -15 <= dist <= -8:
        s += 18
    elif dist > 0:
        s += 25
    rh = float(e["High"].iloc[max(0, i - 19): i + 1].max())
    pb = (close / rh - 1) * 100 if rh > 0 else 0
    if -5 <= pb <= 0:
        s += 25
    elif -10 < pb < -5:
        s += 15
    s = min(s, 100.0)

    w = tr.weights(tier)
    return (t * w["Trend"] + m * w["Momentum"] + v * w["Volume_S"]
            + rs * w["RelStrength"] + s * w["Setup"])


def run(
    frames: dict[str, pd.DataFrame],
    bench: pd.DataFrame | None,
    *,
    min_score: float = 60.0,
    hold_bars: int = 18,
    rebalance_every: int = 5,
    use_regime_filter: bool = True,
    apply_costs: bool = True,
) -> pd.DataFrame:
    """Walk every stock forward, recording trades the rules would have taken.

    rebalance_every=5 means signals are only checked weekly, matching the
    recommended workflow — and cutting compute by 5x.
    """
    bench_e = ind.enrich(bench) if bench is not None and len(bench) > 200 else None
    trades: list[Trade] = []

    for ticker, df in frames.items():
        if df is None or len(df) < MIN_HISTORY + hold_bars + 2:
            continue

        tier = tr.classify_by_turnover(df)
        p = tr.params(tier)

        try:
            e = ind.enrich(df)
        except Exception:
            continue

        n = len(e)
        i = MIN_HISTORY
        while i < n - hold_bars - 2:
            state = rg.classify_at(bench_e, i) if bench_e is not None else rg.RISK_ON
            if use_regime_filter and tier not in rg.TIER_PERMISSION[state]:
                i += rebalance_every
                continue

            try:
                score = _score_at(e, i, bench_e, tier)
            except Exception:
                i += rebalance_every
                continue

            rsi_lo, rsi_hi = p["rsi_band"]
            rsi_v = float(e["RSI14"].iloc[i])
            if score < min_score or not (rsi_lo <= rsi_v <= rsi_hi):
                i += rebalance_every
                continue
            if float(e["Close"].iloc[i]) <= float(e["EMA50"].iloc[i]):
                i += rebalance_every
                continue

            # --- Enter at next bar's open ---
            entry_i = i + 1
            entry = float(e["Open"].iloc[entry_i])
            atr_v = float(e["ATR14"].iloc[i])
            if not np.isfinite(entry) or entry <= 0 or not np.isfinite(atr_v) or atr_v <= 0:
                i += rebalance_every
                continue

            stop_dist = atr_v * p["atr_mult"]
            stop = entry - stop_dist

            exit_i, exit_px, reason = None, None, ""
            for j in range(entry_i, min(entry_i + hold_bars, n)):
                if float(e["Low"].iloc[j]) <= stop:
                    exit_i, exit_px, reason = j, stop, "stop"
                    break
            if exit_i is None:
                exit_i = min(entry_i + hold_bars - 1, n - 1)
                exit_px = float(e["Close"].iloc[exit_i])
                reason = "time"

            gross_r = (exit_px - entry) / stop_dist
            stop_pct = stop_dist / entry * 100
            net_r = tr.net_expected_r(gross_r, tier, stop_pct) if apply_costs else gross_r

            trades.append(Trade(
                ticker=ticker.replace(".NS", ""),
                tier=tier,
                regime=state,
                entry_date=e.index[entry_i],
                exit_date=e.index[exit_i],
                entry=round(entry, 2),
                exit=round(float(exit_px), 2),
                stop=round(stop, 2),
                score=int(round(score)),
                bars_held=exit_i - entry_i,
                exit_reason=reason,
                gross_r=round(gross_r, 3),
                net_r=round(net_r, 3),
                ret_pct=round((exit_px / entry - 1) * 100, 2),
            ))

            # No overlapping positions in the same name
            i = exit_i + rebalance_every

    return pd.DataFrame([asdict(t) for t in trades])


def summarize(trades: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Aggregate stats. `by` can be 'tier', 'regime', or 'score_bucket'."""
    if trades.empty:
        return pd.DataFrame()

    work = trades.copy()
    if by == "score_bucket":
        work["score_bucket"] = pd.cut(
            work["score"], [0, 60, 65, 70, 75, 80, 100],
            labels=["<60", "60-65", "65-70", "70-75", "75-80", "80+"],
        )

    def _agg(g: pd.DataFrame) -> pd.Series:
        wins = g[g["net_r"] > 0]
        losses = g[g["net_r"] <= 0]
        avg_w = wins["net_r"].mean() if len(wins) else 0.0
        avg_l = losses["net_r"].mean() if len(losses) else 0.0
        wr = len(wins) / len(g) if len(g) else 0.0
        return pd.Series({
            "trades": len(g),
            "win_rate": round(wr * 100, 1),
            "avg_win_R": round(avg_w, 2),
            "avg_loss_R": round(avg_l, 2),
            "expectancy_R": round(g["net_r"].mean(), 3),
            "total_R": round(g["net_r"].sum(), 1),
            "median_ret_pct": round(g["ret_pct"].median(), 2),
            "stopped_pct": round((g["exit_reason"] == "stop").mean() * 100, 1),
            "avg_bars": round(g["bars_held"].mean(), 1),
        })

    if by is None:
        return _agg(work).to_frame("all").T

    return work.groupby(by, observed=True).apply(_agg, include_groups=False).reset_index()


def equity_curve(trades: pd.DataFrame, risk_per_trade_pct: float = 1.0,
                 start_capital: float = 500_000) -> pd.DataFrame:
    """Sequential equity curve assuming fixed fractional risk per trade."""
    if trades.empty:
        return pd.DataFrame()
    t = trades.sort_values("exit_date").copy()
    equity, curve = start_capital, []
    for _, row in t.iterrows():
        equity += equity * (risk_per_trade_pct / 100) * row["net_r"]
        curve.append({"date": row["exit_date"], "equity": equity, "ticker": row["ticker"]})
    out = pd.DataFrame(curve)
    out["peak"] = out["equity"].cummax()
    out["drawdown_pct"] = (out["equity"] / out["peak"] - 1) * 100
    return out
