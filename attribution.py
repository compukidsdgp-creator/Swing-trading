"""Trade attribution — why a trade worked, not just whether it did.

The gap this closes
-------------------
Nothing in the retail category does this, and without it eight weeks of forward
evidence produces a single number: an IC. That tells you whether the picks beat
a ranking, not whether the *signal* earned anything.

A stock can rise 8% over a hold because the market rose 6%, its sector rose
another 3%, and the stock itself lagged by 1%. On a raw return that reads as a
win. Attributed properly it is a loss — the signal picked a laggard and the tide
carried it.

The decomposition
-----------------
Each trade's return splits three ways:

    total = market + sector + idiosyncratic

  **Market** — beta times the benchmark's return over the same window. What you
  would have earned holding the index with the same exposure.

  **Sector** — the sector's excess over the market, times exposure. What you
  earned by being in the right industry rather than the right stock.

  **Idiosyncratic** — what is left. This is the only component the signal can
  claim credit for.

Why this matters more than it sounds
------------------------------------
Momentum is documented to load on sector rotation. A momentum screen in a market
where one sector is running will select from that sector repeatedly, and the
resulting returns look like stock selection while being sector exposure.

If most of your forward return turns out to be market and sector, the honest
conclusion is that you have built an expensive index tracker — and you would
never learn that from an IC alone.

Beta estimation
---------------
Beta is estimated from data available **before** entry, never over the holding
period itself. Using in-window beta would fit the very return being explained
and guarantee a small residual, which is the standard way this analysis gets
quietly rigged.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BETA_LOOKBACK = 250          # trading days before entry
MIN_BETA_OBS = 60
DEFAULT_BETA = 1.0


@dataclass
class TradeAttribution:
    ticker: str
    entry_date: str
    exit_date: str
    total_return_pct: float
    market_component_pct: float
    sector_component_pct: float
    idiosyncratic_pct: float
    beta: float
    market_return_pct: float
    sector_return_pct: float
    days_held: int
    verdict: str

    @property
    def signal_share(self) -> float:
        """Fraction of the return attributable to stock selection."""
        if abs(self.total_return_pct) < 1e-9:
            return 0.0
        return self.idiosyncratic_pct / self.total_return_pct


@dataclass
class AttributionSummary:
    trades: pd.DataFrame
    n: int
    mean_total: float
    mean_market: float
    mean_sector: float
    mean_idio: float
    idio_share_pct: float
    idio_t_stat: float
    verdict: str
    message: str
    notes: list[str] = field(default_factory=list)


def estimate_beta(stock: pd.Series, bench: pd.Series,
                  *, lookback: int = BETA_LOOKBACK) -> float:
    """Beta from returns BEFORE the window being explained.

    Estimating beta over the holding period would fit the return being
    decomposed and force a small residual — the standard way this analysis gets
    quietly rigged.
    """
    s = stock.pct_change().dropna().tail(lookback)
    b = bench.pct_change().dropna().tail(lookback)
    joined = pd.concat([s, b], axis=1, join="inner").dropna()
    if len(joined) < MIN_BETA_OBS:
        return DEFAULT_BETA
    sv, bv = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
    var = float(np.var(bv, ddof=1))
    if var <= 0:
        return DEFAULT_BETA
    beta = float(np.cov(sv, bv, ddof=1)[0, 1] / var)
    # Clamp: estimates outside this range are almost always noise at this
    # sample size, and an extreme beta distorts the whole decomposition.
    return float(np.clip(beta, 0.2, 2.5))


def attribute_trade(
    ticker: str,
    entry_date: str | dt.date,
    exit_date: str | dt.date,
    stock_prices: pd.DataFrame,
    bench_prices: pd.DataFrame,
    sector_prices: pd.DataFrame | None = None,
) -> TradeAttribution | None:
    """Decompose one trade into market, sector and idiosyncratic components."""
    d0, d1 = pd.Timestamp(entry_date), pd.Timestamp(exit_date)
    if d1 <= d0:
        return None

    def _window(df):
        if df is None or df.empty or "Close" not in df.columns:
            return None
        w = df[(df.index >= d0) & (df.index <= d1)]["Close"]
        return w if len(w) >= 2 else None

    sw, bw = _window(stock_prices), _window(bench_prices)
    if sw is None or bw is None:
        return None

    total = (float(sw.iloc[-1]) / float(sw.iloc[0]) - 1) * 100
    mkt = (float(bw.iloc[-1]) / float(bw.iloc[0]) - 1) * 100

    # Beta from pre-entry data only
    pre_s = stock_prices[stock_prices.index < d0]["Close"]
    pre_b = bench_prices[bench_prices.index < d0]["Close"]
    beta = estimate_beta(pre_s, pre_b)

    market_component = beta * mkt

    sector_component, sector_ret = 0.0, 0.0
    cw = _window(sector_prices) if sector_prices is not None else None
    if cw is not None:
        sector_ret = (float(cw.iloc[-1]) / float(cw.iloc[0]) - 1) * 100
        # Sector's excess over the market — being in the right industry
        sector_component = sector_ret - mkt

    idio = total - market_component - sector_component

    if abs(total) < 0.5:
        verdict = "flat"
    elif idio > 0 and idio > abs(market_component) + abs(sector_component):
        verdict = "signal driven"
    elif abs(market_component) > abs(idio) and abs(market_component) > abs(sector_component):
        verdict = "market driven"
    elif abs(sector_component) > abs(idio):
        verdict = "sector driven"
    else:
        verdict = "mixed"

    return TradeAttribution(
        ticker=ticker.replace(".NS", ""),
        entry_date=d0.date().isoformat(), exit_date=d1.date().isoformat(),
        total_return_pct=round(total, 2),
        market_component_pct=round(market_component, 2),
        sector_component_pct=round(sector_component, 2),
        idiosyncratic_pct=round(idio, 2),
        beta=round(beta, 2),
        market_return_pct=round(mkt, 2),
        sector_return_pct=round(sector_ret, 2),
        days_held=int((d1 - d0).days),
        verdict=verdict,
    )


def attribute_log(
    log: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    bench: pd.DataFrame,
    sector_indices: dict[str, pd.DataFrame] | None = None,
    sector_map: dict[str, str] | None = None,
) -> AttributionSummary:
    """Attribute every evaluated pick in the forward log.

    Args:
        log: forward_log.csv with status == 'evaluated'.
        prices: {ticker: OHLCV}.
        bench: benchmark OHLCV.
        sector_indices: {sector_name: OHLCV}, optional.
        sector_map: {ticker: sector_name}, optional.
    """
    if log is None or log.empty:
        return AttributionSummary(pd.DataFrame(), 0, 0, 0, 0, 0, 0, 0,
                                  "empty", "No forward log entries.")

    ev = log[log["status"] == "evaluated"].copy() if "status" in log.columns else log.copy()
    if ev.empty:
        return AttributionSummary(pd.DataFrame(), 0, 0, 0, 0, 0, 0, 0,
                                  "empty", "No evaluated picks yet.")

    rows = []
    for _, r in ev.iterrows():
        tkr = str(r.get("ticker", ""))
        entry = r.get("snapshot_date")
        exit_ = r.get("evaluated_on")
        if exit_ is None or (isinstance(exit_, float) and pd.isna(exit_)):
            exit_ = r.get("target_date")
        if not entry or not exit_:
            continue

        # `a or b` raises on DataFrames. This is the fourth occurrence of the
        # same bug in this codebase; the CI check exists precisely for it.
        sp = prices.get(tkr)
        if sp is None:
            sp = prices.get(f"{tkr}.NS")
        if sp is None:
            continue

        sec_px = None
        if sector_indices and sector_map:
            sec = sector_map.get(tkr)
            if sec is None:
                sec = sector_map.get(f"{tkr}.NS")
            if sec:
                sec_px = sector_indices.get(sec)

        a = attribute_trade(tkr, entry, exit_, sp, bench, sec_px)
        if a:
            rows.append({
                "ticker": a.ticker, "entry": a.entry_date, "exit": a.exit_date,
                "days": a.days_held, "total_pct": a.total_return_pct,
                "market_pct": a.market_component_pct,
                "sector_pct": a.sector_component_pct,
                "idio_pct": a.idiosyncratic_pct,
                "beta": a.beta, "verdict": a.verdict,
            })

    if not rows:
        return AttributionSummary(pd.DataFrame(), 0, 0, 0, 0, 0, 0, 0,
                                  "empty", "No trades could be attributed — "
                                           "check that price history covers "
                                           "the holding windows.")

    df = pd.DataFrame(rows)
    n = len(df)
    mean_total = float(df["total_pct"].mean())
    mean_mkt = float(df["market_pct"].mean())
    mean_sec = float(df["sector_pct"].mean())
    mean_idio = float(df["idio_pct"].mean())

    idio_share = (mean_idio / mean_total * 100) if abs(mean_total) > 1e-9 else 0.0
    sd = float(df["idio_pct"].std(ddof=1)) if n > 1 else 0.0
    t_stat = (mean_idio / (sd / np.sqrt(n))) if sd > 0 else 0.0

    notes = [
        f"{n} trades attributed. Beta estimated from {BETA_LOOKBACK} days "
        "before each entry — never over the holding period, which would fit "
        "the return being explained.",
    ]
    if n < 20:
        notes.append(f"Only {n} trades. Below roughly 20 the split is mostly "
                     "noise; treat this as directional.")
    if not sector_indices:
        notes.append("No sector indices supplied — sector effects are absorbed "
                     "into the idiosyncratic component, which overstates it.")

    if mean_idio <= 0:
        verdict, msg = "bad", (
            f"**The signal contributed nothing.** Mean total return "
            f"{mean_total:+.2f}%, of which market {mean_mkt:+.2f}% and sector "
            f"{mean_sec:+.2f}%. The idiosyncratic component is "
            f"{mean_idio:+.2f}% — stock selection subtracted value. Whatever "
            "was earned came from being in the market, not from the picks.")
    elif idio_share < 25:
        verdict, msg = "warn", (
            f"**Only {idio_share:.0f}% of the return came from stock "
            f"selection.** Market contributed {mean_mkt:+.2f}% and sector "
            f"{mean_sec:+.2f}% against {mean_idio:+.2f}% idiosyncratic. This is "
            "close to an expensive index tracker — you are paying 0.36% round "
            "trip plus 20% tax for exposure available at a fraction of that.")
    elif abs(t_stat) < 2.0:
        verdict, msg = "unproven", (
            f"Idiosyncratic return averages {mean_idio:+.2f}% "
            f"({idio_share:.0f}% of total), but t = {t_stat:.2f} across {n} "
            "trades. The signal component is not statistically distinguishable "
            "from zero at this sample size.")
    else:
        verdict, msg = "good", (
            f"**Stock selection contributed {idio_share:.0f}% of the return** "
            f"({mean_idio:+.2f}% of {mean_total:+.2f}% total), t = {t_stat:.2f}. "
            f"Market {mean_mkt:+.2f}%, sector {mean_sec:+.2f}%. The signal is "
            "earning its keep rather than riding the tide.")

    counts = df["verdict"].value_counts().to_dict()
    msg += "\n\nPer-trade: " + ", ".join(f"{k} {v}" for k, v in counts.items())

    return AttributionSummary(
        df, n, round(mean_total, 3), round(mean_mkt, 3), round(mean_sec, 3),
        round(mean_idio, 3), round(idio_share, 1), round(t_stat, 2),
        verdict, msg, notes)


def attribution_chart_data(summary: AttributionSummary) -> pd.DataFrame:
    """Component averages, for plotting."""
    if summary.n == 0:
        return pd.DataFrame()
    return pd.DataFrame([
        {"component": "Market", "mean_pct": summary.mean_market},
        {"component": "Sector", "mean_pct": summary.mean_sector},
        {"component": "Signal (idiosyncratic)", "mean_pct": summary.mean_idio},
    ])


def hit_rate_by_component(summary: AttributionSummary) -> pd.DataFrame:
    """How often each component was positive.

    A signal that helps more often than it hurts shows a high idiosyncratic
    hit rate even when the average is modest.
    """
    if summary.trades.empty:
        return pd.DataFrame()
    df = summary.trades
    rows = []
    for col, label in (("market_pct", "Market"), ("sector_pct", "Sector"),
                       ("idio_pct", "Signal")):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append({
            "component": label,
            "positive_pct": round(float((s > 0).mean()) * 100, 1),
            "mean_pct": round(float(s.mean()), 2),
            "median_pct": round(float(s.median()), 2),
            "best_pct": round(float(s.max()), 2),
            "worst_pct": round(float(s.min()), 2),
        })
    return pd.DataFrame(rows)
