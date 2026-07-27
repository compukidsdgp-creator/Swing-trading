"""Signal laboratory — testing candidate signals for incremental content.

The composite score failed because its five components all measured the same
underlying thing: recent price movement. It scored 0.76 correlation with
one-month return and 13.9% IC retention after factor neutralisation.

Rather than hand-craft another composite and hope, this module tests a library
of candidate signals **individually**, reporting for each:

  * **Raw IC** — does it predict forward returns at all?
  * **Residual IC** — does it predict anything after the standard factor set
    is controlled for?
  * **Newey-West t** — is the residual statistically meaningful once
    autocorrelation across overlapping windows is corrected for?

A signal with high raw IC and near-zero residual IC is a repackaged factor.
A signal with modest raw IC but *positive residual IC* is genuinely incremental,
and is worth far more than a high-scoring redundant one.

Design principle
----------------
Each signal is a single, transparent calculation with a documented rationale
drawn from published research. No weighted blends. If several turn out to have
independent content, they can be combined afterwards — but only *after* each has
earned its place, never before.

Honest expectation
------------------
Most of these will fail. That is the normal outcome of signal research, and the
point of running the test rather than assuming.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import indicators as ind
from factor_analysis import (
    FACTOR_NAMES, HLZ_THRESHOLD, _factors_at, _newey_west_se,
    _ols, _spearman, _winsorise, _zscore,
)


# --------------------------------------------------------------------------
# Candidate signals
#
# Each takes (enriched_frame, index, benchmark_frame) and returns a float,
# using ONLY data up to `i`. Return np.nan when not computable.
# --------------------------------------------------------------------------
def sig_mom_12_1(e, i, b):
    """Classic momentum (Jegadeesh & Titman 1993). Included as the benchmark
    to beat — by construction it should show ~zero residual content."""
    c = e["Close"]
    p252, p21 = float(c.iloc[i - 252]), float(c.iloc[i - 21])
    return (p21 / p252 - 1) if p252 > 0 else np.nan


def sig_mom_6_1(e, i, b):
    """Six-month momentum. Shorter formation window than the classic."""
    c = e["Close"]
    p126, p21 = float(c.iloc[i - 126]), float(c.iloc[i - 21])
    return (p21 / p126 - 1) if p126 > 0 else np.nan


def sig_idio_momentum(e, i, b):
    """Idiosyncratic momentum (Blitz, Huij & Martens 2011).

    Momentum of the residual after regressing out market beta. Documented as
    distinct from — and stronger than — raw momentum, because it strips out the
    part explained by market exposure.
    """
    if b is None or len(b) <= i:
        return np.nan
    r = e["Close"].iloc[i - 251: i - 20].pct_change().dropna()
    m = b["Close"].iloc[i - 251: i - 20].pct_change().dropna()
    n = min(len(r), len(m))
    if n < 100:
        return np.nan
    rv, mv = r.to_numpy()[-n:], m.to_numpy()[-n:]
    var = np.var(mv)
    if var < 1e-12:
        return np.nan
    beta = np.cov(rv, mv)[0, 1] / var
    return float(np.sum(rv - beta * mv))


def sig_mom_consistency(e, i, b):
    """Momentum consistency (path, not just endpoint).

    Two stocks can post the same 6-month return, one via steady grind and one
    via a single gap. Grinders are documented to continue more reliably.
    Measured as the share of positive days minus 0.5.
    """
    r = e["Close"].iloc[i - 126: i + 1].pct_change().dropna()
    return float((r > 0).mean() - 0.5) if len(r) > 60 else np.nan


def sig_vol_adjusted_mom(e, i, b):
    """Momentum scaled by realised volatility — a Sharpe-like formulation.

    Penalises returns achieved through sheer volatility rather than trend.
    """
    c = e["Close"]
    p252, p21 = float(c.iloc[i - 252]), float(c.iloc[i - 21])
    if p252 <= 0:
        return np.nan
    mom = p21 / p252 - 1
    vol = float(c.iloc[i - 251: i + 1].pct_change().std() * np.sqrt(252))
    return mom / vol if vol > 1e-6 else np.nan


def sig_52w_high(e, i, b):
    """Proximity to the 52-week high (George & Hwang 2004).

    Documented as predicting returns *independently* of momentum — an anchoring
    effect, where traders under-react to news near the high.
    """
    c = e["Close"]
    hi = float(e["High"].iloc[i - 251: i + 1].max())
    return float(c.iloc[i]) / hi if hi > 0 else np.nan


def sig_acceleration(e, i, b):
    """Change in momentum — is the trend speeding up or fading?

    Recent 3-month return minus the prior 3-month return.
    """
    c = e["Close"]
    p0, p63, p126 = (float(c.iloc[i]), float(c.iloc[i - 63]), float(c.iloc[i - 126]))
    if p63 <= 0 or p126 <= 0:
        return np.nan
    return (p0 / p63 - 1) - (p63 / p126 - 1)


def sig_accumulation(e, i, b):
    """Volume-weighted accumulation over 60 days.

    Volume on up-days minus volume on down-days, normalised. Aims to detect
    institutional accumulation rather than price alone.
    """
    w = e.iloc[i - 59: i + 1]
    up = w.loc[w["Close"] >= w["Open"], "Volume"].sum()
    dn = w.loc[w["Close"] < w["Open"], "Volume"].sum()
    tot = up + dn
    return float((up - dn) / tot) if tot > 0 else np.nan


def sig_low_volatility(e, i, b):
    """Negative realised volatility — the low-volatility anomaly.

    Low-vol stocks have historically outperformed on a risk-adjusted basis.
    Sign flipped so higher is better, keeping every signal directionally
    comparable.
    """
    v = float(e["Close"].iloc[i - 59: i + 1].pct_change().std() * np.sqrt(252))
    return -v if np.isfinite(v) else np.nan


def sig_reversal_1m(e, i, b):
    """Short-term reversal, sign-flipped (recent losers tend to bounce).

    Included because the failed composite loaded 0.76 on this — worth seeing
    what it does in isolation.
    """
    c = e["Close"]
    p21 = float(c.iloc[i - 21])
    return -(float(c.iloc[i]) / p21 - 1) if p21 > 0 else np.nan


def sig_illiquidity(e, i, b):
    """Amihud (2002) illiquidity — |return| per unit of traded value.

    Illiquid stocks carry a documented premium. Also a useful sanity check:
    if this is your best signal, you are being paid for liquidity risk, not skill.
    """
    w = e.iloc[i - 59: i + 1]
    r = w["Close"].pct_change().abs()
    tv = (w["Close"] * w["Volume"]) / 1e7
    ratio = (r / tv.replace(0, np.nan)).dropna()
    return float(np.log1p(ratio.mean() * 1e4)) if len(ratio) > 30 else np.nan


def sig_range_compression(e, i, b):
    """Volatility compression — current 20-day range vs its 6-month norm.

    Coiled springs. Sign flipped so tighter compression scores higher.
    """
    tr = (e["High"] - e["Low"]) / e["Close"].replace(0, np.nan)
    recent = float(tr.iloc[i - 19: i + 1].mean())
    base = float(tr.iloc[i - 125: i + 1].mean())
    return -(recent / base) if base > 1e-9 else np.nan


SIGNALS = {
    "mom_12_1 (benchmark)": sig_mom_12_1,
    "mom_6_1": sig_mom_6_1,
    "idiosyncratic_mom": sig_idio_momentum,
    "mom_consistency": sig_mom_consistency,
    "vol_adjusted_mom": sig_vol_adjusted_mom,
    "52w_high_proximity": sig_52w_high,
    "acceleration": sig_acceleration,
    "accumulation": sig_accumulation,
    "low_volatility": sig_low_volatility,
    "reversal_1m": sig_reversal_1m,
    "illiquidity": sig_illiquidity,
    "range_compression": sig_range_compression,
}


@dataclass
class SignalResult:
    table: pd.DataFrame
    per_window: pd.DataFrame
    n_windows: int
    notes: list[str]


def run(
    frames: dict[str, pd.DataFrame],
    bench: pd.DataFrame | None,
    *,
    horizon: int = 15,
    step: int | None = None,
    min_names: int = 25,
) -> SignalResult:
    """Test every candidate signal for raw and residual predictive content."""
    # +3 avoids every window landing on the same weekday (see validate.py)
    step = step or (horizon + 3)
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
        return SignalResult(pd.DataFrame(), pd.DataFrame(), 0, ["no usable data"])

    cal = (pd.DatetimeIndex(bench_e.index) if bench_e is not None
           else pd.DatetimeIndex(max((e.index for e in enriched.values()), key=len)))

    start = 300
    names = list(SIGNALS)
    raw_ic = {n: [] for n in names}
    res_ic = {n: [] for n in names}
    win_rows = []

    for k in range(start, len(cal) - horizon - 1, step):
        date = cal[k]
        recs = []
        for t, e in enriched.items():
            try:
                i = e.index.get_loc(date)
            except KeyError:
                continue
            if not isinstance(i, int) or i < start or i + horizon >= len(e):
                continue

            f = _factors_at(e, i, bench_e)
            if f is None:
                continue

            p0 = float(e["Close"].iloc[i])
            p1 = float(e["Close"].iloc[i + horizon])
            if p0 <= 0 or not np.isfinite(p1):
                continue

            row = {"fwd": (p1 / p0 - 1) * 100, **f}
            for n, fn in SIGNALS.items():
                try:
                    row[n] = fn(e, i, bench_e)
                except Exception:                              # noqa: BLE001
                    row[n] = np.nan
            recs.append(row)

        if len(recs) < min_names:
            continue

        d = pd.DataFrame(recs)
        y = _winsorise(d["fwd"].to_numpy())

        wr = {"date": date, "n": len(d)}
        for n in names:
            col = d[n].to_numpy(dtype=float)
            if np.isnan(col).mean() > 0.3:
                continue
            col = np.where(np.isnan(col), np.nanmedian(col), col)
            s = _zscore(_winsorise(col))

            r = _spearman(s, y)
            if np.isfinite(r):
                raw_ic[n].append(r)
                wr[f"raw_{n}"] = r

            # Neutralise against the standard factor set. If this signal IS one
            # of the controls, drop that control to avoid regressing it on itself.
            ctrl = [c for c in FACTOR_NAMES if c.split(" ")[0] not in n]
            if not ctrl:
                continue
            F = np.column_stack([
                _zscore(_winsorise(np.where(np.isnan(d[c].to_numpy(dtype=float)),
                                            np.nanmedian(d[c].to_numpy(dtype=float)),
                                            d[c].to_numpy(dtype=float))))
                for c in ctrl
            ])
            _, resid = _ols(s, F)
            rr = _spearman(resid, y)
            if np.isfinite(rr):
                res_ic[n].append(rr)
                wr[f"res_{n}"] = rr

        win_rows.append(wr)

    if not win_rows:
        return SignalResult(pd.DataFrame(), pd.DataFrame(), 0, ["no valid windows"])

    rows = []
    for n in names:
        rv, sv = np.array(raw_ic[n]), np.array(res_ic[n])
        if len(rv) < 10:
            continue
        r_mean = float(rv.mean())
        s_mean = float(sv.mean()) if len(sv) else np.nan
        s_nw = _newey_west_se(sv) if len(sv) > 3 else np.nan
        s_t = s_mean / s_nw if s_nw and np.isfinite(s_nw) and s_nw > 0 else np.nan
        rows.append({
            "signal": n,
            "raw_ic": round(r_mean, 4),
            "residual_ic": round(s_mean, 4) if np.isfinite(s_mean) else None,
            "retention_pct": (round(s_mean / r_mean * 100, 1)
                              if abs(r_mean) > 1e-9 and np.isfinite(s_mean) else None),
            "residual_t_nw": round(s_t, 2) if np.isfinite(s_t) else None,
            "pct_positive": round(float((rv > 0).mean()) * 100, 1),
            "windows": len(rv),
            "verdict": _verdict(s_mean, s_t),
        })

    table = (pd.DataFrame(rows)
             .sort_values("residual_ic", ascending=False, na_position="last")
             .reset_index(drop=True))

    notes = []
    live = table[table["residual_t_nw"].notna()]
    strong = live[live["residual_t_nw"].abs() >= 2.0]
    if strong.empty:
        notes.append(
            "No signal shows statistically significant incremental content. That is the "
            "normal outcome of signal research, and it is a real finding: at this data "
            "quality, price-based signals do not clear the bar once known factors are "
            "controlled for."
        )
    else:
        for _, r in strong.iterrows():
            notes.append(
                f"**{r['signal']}** retains residual IC {r['residual_ic']:+.4f} "
                f"(t = {r['residual_t_nw']:+.2f}) — worth pursuing."
            )
    return SignalResult(table, pd.DataFrame(win_rows), len(win_rows), notes)


def _verdict(res_ic: float, t: float) -> str:
    if not np.isfinite(res_ic) or not np.isfinite(t):
        return "insufficient data"
    if abs(t) >= HLZ_THRESHOLD:
        return "strong — clears t>3"
    if abs(t) >= 2.0:
        return "significant — clears t>2"
    if abs(t) >= 1.5:
        return "marginal"
    if res_ic > 0:
        return "no incremental content"
    return "negative"
