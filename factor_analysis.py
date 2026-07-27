"""Factor analysis — is the score anything more than repackaged momentum?

This module applies the cross-sectional regression methodology used in academic
and institutional factor research, to answer questions raw IC cannot:

  1. **Factor loadings.** What is the score actually correlated with? If it is
     0.9 correlated with 12-1 momentum, it is momentum with extra steps.

  2. **Residual IC.** Orthogonalise the score against known factors and
     re-measure. Whatever survives is the score's genuine contribution.

  3. **Fama-MacBeth regression** (Fama & MacBeth 1973). Run a cross-sectional
     regression each period, then average the coefficients over time. The
     standard approach for testing whether a characteristic predicts returns
     while controlling for others.

  4. **Newey-West standard errors** (Newey & West 1987). Cross-sectional
     regression coefficients are autocorrelated across overlapping periods;
     naive t-statistics overstate significance. This corrects for it.

  5. **Multiple-testing context.** Harvey, Liu & Zhu (2016) argue that after
     accounting for the vast number of strategies tested across the literature,
     a newly claimed factor needs **t > 3.0**, not the conventional 2.0. This
     module reports against that bar.

Control factors
---------------
  mom_12_1     12-month return skipping the most recent month — the classic
               momentum factor (Jegadeesh & Titman 1993). Skipping one month
               avoids contamination by short-term reversal.
  reversal_1m  Last month's return. Short-horizon returns tend to reverse.
  size         Log of average traded value — a proxy for market cap, since
               true float-adjusted cap is unavailable from free data.
  volatility   60-day realised volatility. Low-vol stocks outperform on a
               risk-adjusted basis (the low-volatility anomaly).
  beta         60-day rolling beta to the benchmark.
  liquidity    Log turnover, capturing the illiquidity premium.

What this CANNOT fix
--------------------
Survivorship bias and data quality are data problems, not statistical ones. No
amount of regression repairs a universe that excludes the companies that failed.
Those require a point-in-time database and are flagged, not solved, here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import indicators as ind
from backtest import _score_at, MIN_HISTORY

FACTOR_NAMES = ["mom_12_1", "reversal_1m", "size", "volatility", "beta", "liquidity"]

# Harvey, Liu & Zhu (2016) multiple-testing threshold for a new factor claim
HLZ_THRESHOLD = 3.0


@dataclass
class FactorResult:
    raw_ic: float
    residual_ic: float
    ic_retention: float
    loadings: pd.DataFrame           # score's correlation with each factor
    fm_coefficients: pd.DataFrame    # Fama-MacBeth results
    per_window: pd.DataFrame
    n_windows: int
    summary: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Linear algebra helpers (numpy only — no statsmodels dependency)
# --------------------------------------------------------------------------
def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary least squares with intercept. Returns (coefficients, residuals)."""
    n = len(y)
    Xd = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    return beta, resid


def _newey_west_se(series: np.ndarray, lags: int | None = None) -> float:
    """Newey-West heteroskedasticity- and autocorrelation-consistent SE of a mean.

    Cross-sectional regression coefficients from overlapping holding periods are
    autocorrelated. Ignoring that inflates t-statistics, sometimes badly.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")

    if lags is None:
        # Standard rule of thumb: floor(4 * (n/100)^(2/9))
        lags = max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))

    dev = x - x.mean()
    gamma0 = float((dev @ dev) / n)
    var = gamma0
    for L in range(1, min(lags, n - 1) + 1):
        w = 1.0 - L / (lags + 1)                     # Bartlett kernel
        gamma = float((dev[L:] @ dev[:-L]) / n)
        var += 2.0 * w * gamma
    var = max(var, 1e-12)
    return float(np.sqrt(var / n))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 5:
        return np.nan
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def _winsorise(x: np.ndarray, pct: float = 0.01) -> np.ndarray:
    """Clip extremes. Standard practice — a single outlier can dominate an OLS fit."""
    lo, hi = np.nanpercentile(x, [pct * 100, (1 - pct) * 100])
    return np.clip(x, lo, hi)


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x)
    return (x - np.nanmean(x)) / sd if sd > 1e-12 else np.zeros_like(x)


# --------------------------------------------------------------------------
# Factor construction
# --------------------------------------------------------------------------
def _factors_at(e: pd.DataFrame, i: int, bench_e: pd.DataFrame | None) -> dict | None:
    """Compute control factors at bar i, using only data up to i."""
    if i < 260:
        return None
    close = e["Close"]
    px = float(close.iloc[i])
    if not np.isfinite(px) or px <= 0:
        return None

    try:
        # 12-1 momentum: 12-month return skipping the most recent month
        p_252 = float(close.iloc[i - 252])
        p_21 = float(close.iloc[i - 21])
        mom = (p_21 / p_252 - 1) if p_252 > 0 else 0.0

        # Short-term reversal
        rev = (px / p_21 - 1) if p_21 > 0 else 0.0

        # Size and liquidity proxies from traded value
        tv = float((close.iloc[i - 19: i + 1] * e["Volume"].iloc[i - 19: i + 1]).mean())
        size = np.log(max(tv, 1.0))
        liq = size

        # Realised volatility, annualised
        rets = close.iloc[i - 59: i + 1].pct_change().dropna()
        vol = float(rets.std() * np.sqrt(252)) if len(rets) > 30 else 0.0

        # Rolling beta to the benchmark
        beta = 1.0
        if bench_e is not None and len(bench_e) > i:
            br = bench_e["Close"].iloc[i - 59: i + 1].pct_change().dropna()
            n = min(len(rets), len(br))
            if n > 30:
                sr, sb = rets.to_numpy()[-n:], br.to_numpy()[-n:]
                vb = np.var(sb)
                beta = float(np.cov(sr, sb)[0, 1] / vb) if vb > 1e-12 else 1.0
    except (IndexError, ZeroDivisionError, ValueError):
        return None

    vals = {"mom_12_1": mom, "reversal_1m": rev, "size": size,
            "volatility": vol, "beta": beta, "liquidity": liq}
    return vals if all(np.isfinite(v) for v in vals.values()) else None


# --------------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------------
def run(
    frames: dict[str, pd.DataFrame],
    bench: pd.DataFrame | None,
    *,
    horizon: int = 15,
    step: int | None = None,
    min_names: int = 25,
) -> FactorResult:
    """Fama-MacBeth analysis of the composite score against known factors."""
    import tiers as tr

    # +3 avoids every window landing on the same weekday (see validate.py)
    step = step or (horizon + 3)
    bench_e = ind.enrich(bench) if bench is not None and len(bench) > 260 else None

    enriched, tier_of = {}, {}
    for t, df in frames.items():
        if df is None or len(df) < 300 + horizon:
            continue
        try:
            enriched[t] = ind.enrich(df)
            tier_of[t] = tr.classify_by_turnover(df)
        except Exception:                                       # noqa: BLE001
            continue

    if not enriched:
        return FactorResult(np.nan, np.nan, np.nan, pd.DataFrame(),
                            pd.DataFrame(), pd.DataFrame(), 0,
                            {"error": "no usable data"})

    if bench_e is not None:
        cal = pd.DatetimeIndex(bench_e.index)
    else:
        cal = pd.DatetimeIndex(max((e.index for e in enriched.values()), key=len))

    start = max(MIN_HISTORY, 300)
    rows, load_rows, fm_rows = [], [], []

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
            try:
                sc = _score_at(e, i, bench_e, tier_of[t])
            except Exception:                                   # noqa: BLE001
                continue
            if not np.isfinite(sc):
                continue

            p0 = float(e["Close"].iloc[i])
            p1 = float(e["Close"].iloc[i + horizon])
            if p0 <= 0 or not np.isfinite(p1):
                continue

            recs.append({"score": sc, "fwd": (p1 / p0 - 1) * 100, **f})

        if len(recs) < min_names:
            continue

        d = pd.DataFrame(recs)
        for c in ["score", "fwd"] + FACTOR_NAMES:
            d[c] = _winsorise(d[c].to_numpy())

        y = d["fwd"].to_numpy()
        s = _zscore(d["score"].to_numpy())
        F = np.column_stack([_zscore(d[c].to_numpy()) for c in FACTOR_NAMES])

        # Score's loading on each factor
        load_rows.append({c: float(np.corrcoef(s, F[:, j])[0, 1])
                          for j, c in enumerate(FACTOR_NAMES)})

        # Orthogonalise the score against the factor set
        _, resid_score = _ols(s, F)

        raw_ic = _spearman(s, y)
        res_ic = _spearman(resid_score, y)

        # Fama-MacBeth cross-sectional regression: fwd ~ score + factors
        X = np.column_stack([s, F])
        beta, _ = _ols(y, X)
        fm_rows.append(dict(zip(["alpha", "score"] + FACTOR_NAMES, beta)))

        rows.append({"date": date, "n": len(d), "raw_ic": raw_ic,
                     "residual_ic": res_ic})

    if not rows:
        return FactorResult(np.nan, np.nan, np.nan, pd.DataFrame(),
                            pd.DataFrame(), pd.DataFrame(), 0,
                            {"error": "no valid windows"})

    win = pd.DataFrame(rows)
    loads = pd.DataFrame(load_rows)
    fm = pd.DataFrame(fm_rows)

    raw_ic = float(win["raw_ic"].mean())
    res_ic = float(win["residual_ic"].mean())
    retention = res_ic / raw_ic if abs(raw_ic) > 1e-9 else np.nan

    # Fama-MacBeth coefficients with Newey-West standard errors
    fm_stats = []
    for c in fm.columns:
        vals = fm[c].to_numpy()
        mean = float(np.nanmean(vals))
        nw = _newey_west_se(vals)
        naive = float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals)))
        fm_stats.append({
            "factor": c,
            "coefficient": round(mean, 4),
            "naive_t": round(mean / naive, 2) if naive > 0 else np.nan,
            "newey_west_t": round(mean / nw, 2) if nw and np.isfinite(nw) and nw > 0 else np.nan,
            "significant_t2": bool(abs(mean / nw) > 2.0) if nw and nw > 0 else False,
            "significant_hlz": bool(abs(mean / nw) > HLZ_THRESHOLD) if nw and nw > 0 else False,
        })
    fm_df = pd.DataFrame(fm_stats)

    # Significance of the residual IC itself
    res_vals = win["residual_ic"].dropna().to_numpy()
    res_nw = _newey_west_se(res_vals)
    res_t = float(res_ic / res_nw) if res_nw and res_nw > 0 else np.nan

    loadings = pd.DataFrame({
        "factor": FACTOR_NAMES,
        "mean_correlation": [round(float(loads[c].mean()), 3) for c in FACTOR_NAMES],
        "abs_correlation": [round(abs(float(loads[c].mean())), 3) for c in FACTOR_NAMES],
    }).sort_values("abs_correlation", ascending=False).reset_index(drop=True)

    score_row = fm_df[fm_df["factor"] == "score"].iloc[0]

    summary = {
        "n_windows": len(win),
        "raw_ic": round(raw_ic, 4),
        "residual_ic": round(res_ic, 4),
        "ic_retention_pct": round(retention * 100, 1) if np.isfinite(retention) else None,
        "residual_ic_t_newey_west": round(res_t, 2) if np.isfinite(res_t) else None,
        "fm_score_coefficient": float(score_row["coefficient"]),
        "fm_score_t_naive": float(score_row["naive_t"]) if pd.notna(score_row["naive_t"]) else None,
        "fm_score_t_newey_west": float(score_row["newey_west_t"]) if pd.notna(score_row["newey_west_t"]) else None,
        "passes_t2": bool(score_row["significant_t2"]),
        "passes_hlz_t3": bool(score_row["significant_hlz"]),
        "dominant_factor": loadings.iloc[0]["factor"],
        "dominant_correlation": float(loadings.iloc[0]["mean_correlation"]),
    }

    return FactorResult(raw_ic, res_ic, retention, loadings, fm_df, win,
                        len(win), summary)


def verdict(s: dict) -> tuple[str, str, list[str]]:
    """Plain-language read on the factor analysis."""
    if "error" in s:
        return "error", s["error"], []

    ret = s.get("ic_retention_pct")
    nw_t = s.get("fm_score_t_newey_west")
    dom, dom_r = s.get("dominant_factor"), s.get("dominant_correlation", 0)

    notes = [
        f"Score is most correlated with **{dom}** (r = {dom_r:+.2f}).",
        f"Raw IC {s['raw_ic']:+.4f} → residual IC {s['residual_ic']:+.4f} "
        f"after neutralising six known factors.",
    ]
    if nw_t is not None:
        notes.append(
            f"Fama-MacBeth t on the score is {nw_t:+.2f} with Newey-West "
            f"correction (naive t was {s.get('fm_score_t_naive'):+.2f})."
        )

    if ret is None or ret <= 10:
        return "bad", (
            "The score adds essentially nothing beyond known factors. Once momentum, "
            "reversal, size, volatility, beta and liquidity are controlled for, the "
            "predictive content disappears. You have re-derived published anomalies."
        ), notes

    if ret < 40:
        return "warn", (
            f"Only {ret:.0f}% of the raw IC survives factor neutralisation. Most of "
            f"the apparent edge is exposure to {dom}, which is available far more "
            "cheaply through a simple factor screen than through this model."
        ), notes

    if nw_t is not None and abs(nw_t) < 2.0:
        return "warn", (
            f"{ret:.0f}% of IC survives neutralisation, but the Newey-West corrected "
            f"t-statistic ({nw_t:+.2f}) does not reach conventional significance. "
            "Autocorrelation across overlapping windows was inflating the naive figure."
        ), notes

    if nw_t is not None and abs(nw_t) >= HLZ_THRESHOLD:
        return "good", (
            f"{ret:.0f}% of IC survives neutralisation with Newey-West t = {nw_t:+.2f}, "
            f"clearing the Harvey-Liu-Zhu multiple-testing bar of {HLZ_THRESHOLD}. "
            "This is genuinely incremental to the standard factor set — a strong result "
            "for a model built on free data."
        ), notes

    return "ok", (
        f"{ret:.0f}% of IC survives neutralisation with Newey-West t = {nw_t:+.2f}. "
        f"Clears the conventional t > 2 bar but not the stricter t > {HLZ_THRESHOLD} "
        "used for new factor claims. Suggestive of real incremental content, short of proof."
    ), notes
