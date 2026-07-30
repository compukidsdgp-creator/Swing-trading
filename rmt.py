"""Random matrix theory — separating signal from noise in a covariance matrix.

The problem this solves
----------------------
Risk parity and every other covariance-based weighting scheme needs an estimate
of how assets co-move. With 10 positions there are 55 distinct covariance terms
to estimate, and roughly 60 observations to estimate them from. That ratio is
the difficulty: most of what the sample matrix reports is estimation error, not
structure.

Worse, the error is not benign. An optimiser searches for the direction of
lowest apparent variance, and the smallest eigenvalues of a noisy sample matrix
are precisely the ones most corrupted. So a mean-variance optimiser reliably
concentrates the portfolio into whichever combination happens to look quietest
by accident.

Shrinkage toward a diagonal target (already used in portfolio.py) helps but is
crude: it pulls everything toward the target uniformly, including the parts that
were well estimated.

The Marchenko-Pastur approach
-----------------------------
Random matrix theory gives the eigenvalue distribution of a covariance matrix
computed from pure noise. Any eigenvalue falling inside that predicted range is
statistically indistinguishable from noise, and can be replaced by their average
while preserving the trace.

Eigenvalues above the upper bound carry genuine structure — typically a large
first eigenvalue representing the market factor, and a handful of sector modes.

The practical consequence: with 10 assets and 60 observations, typically only
one or two eigenvalues survive. That is not a failure of the method. It is the
method telling you that a 60-observation sample supports one or two facts about
co-movement, and that any weighting scheme claiming more precision is inventing
it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class RMTResult:
    n_assets: int
    n_observations: int
    q_ratio: float                  # observations / assets
    lambda_plus: float              # Marchenko-Pastur upper bound
    lambda_minus: float
    eigenvalues: list
    n_signal: int                   # eigenvalues above the noise band
    n_noise: int
    variance_explained_by_signal: float
    filtered_correlation: np.ndarray | None = field(default=None, repr=False)
    filtered_covariance: np.ndarray | None = field(default=None, repr=False)
    condition_before: float = 0.0
    condition_after: float = 0.0
    verdict: str = ""
    message: str = ""
    notes: list[str] = field(default_factory=list)


def marchenko_pastur_bounds(n_assets: int, n_obs: int,
                            sigma_sq: float = 1.0) -> tuple[float, float]:
    """Eigenvalue range a pure-noise correlation matrix would produce.

    q = n_obs / n_assets. As q approaches 1 the band widens dramatically — with
    as many observations as assets, essentially nothing is distinguishable from
    noise.
    """
    if n_assets <= 0 or n_obs <= 0:
        return (0.0, 0.0)
    q = n_obs / n_assets
    if q <= 1.0:
        # Below this the sample matrix is singular and the bound is unbounded
        # above in the usual formulation.
        return (0.0, sigma_sq * (1 + math.sqrt(1 / max(q, 1e-9))) ** 2)
    root = math.sqrt(1.0 / q)
    return (sigma_sq * (1 - root) ** 2, sigma_sq * (1 + root) ** 2)


def filter_correlation(returns: pd.DataFrame,
                       *, keep_market_mode: bool = True) -> RMTResult:
    """Denoise a correlation matrix by clipping noise eigenvalues.

    Eigenvalues inside the Marchenko-Pastur band are replaced by their mean,
    which preserves the trace (total variance) while removing the spurious
    structure that an optimiser would otherwise exploit.

    Args:
        keep_market_mode: retain the largest eigenvalue regardless of the bound.
            It is almost always the market factor and almost always genuine;
            discarding it would remove the one thing the sample estimates well.
    """
    r = returns.dropna(axis=1, how="all").dropna()
    n_obs, n_assets = r.shape
    if n_assets < 2 or n_obs < n_assets + 2:
        return RMTResult(n_assets, n_obs, 0, 0, 0, [], 0, 0, 0,
                         verdict="insufficient",
                         message=(f"{n_obs} observations for {n_assets} assets. "
                                  "Need more observations than assets, and "
                                  "ideally several times more."))

    corr = r.corr().to_numpy()
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    vals, vecs = np.linalg.eigh(corr)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]

    lo, hi = marchenko_pastur_bounds(n_assets, n_obs)

    # Which eigenvalues carry structure
    is_signal = vals > hi
    if keep_market_mode and not is_signal[0]:
        is_signal[0] = True
    n_signal = int(is_signal.sum())
    n_noise = n_assets - n_signal

    filtered_vals = vals.copy()
    if n_noise > 0:
        noise_mean = float(vals[~is_signal].mean())
        filtered_vals[~is_signal] = noise_mean

    filtered = vecs @ np.diag(filtered_vals) @ vecs.T
    # Restore unit diagonal — clipping perturbs it slightly
    d = np.sqrt(np.clip(np.diag(filtered), 1e-12, None))
    filtered = filtered / np.outer(d, d)
    np.fill_diagonal(filtered, 1.0)

    vol = r.std(ddof=1).to_numpy()
    filtered_cov = filtered * np.outer(vol, vol)

    cond_before = float(np.linalg.cond(corr))
    cond_after = float(np.linalg.cond(filtered))
    var_signal = float(vals[is_signal].sum() / vals.sum())

    notes = [
        f"q = {n_obs / n_assets:.1f} observations per asset. The "
        f"Marchenko-Pastur noise band is [{lo:.3f}, {hi:.3f}].",
        f"{n_signal} of {n_assets} eigenvalues carry structure; {n_noise} are "
        "indistinguishable from noise and have been clipped to their mean.",
    ]
    if cond_before > 0 and cond_after > 0:
        notes.append(
            f"Condition number improved from {cond_before:,.0f} to "
            f"{cond_after:,.0f}. A high condition number is what makes an "
            "optimiser unstable — small changes in input produce large changes "
            "in weights.")

    if n_obs < n_assets * 3:
        notes.append(
            f"Only {n_obs / n_assets:.1f} observations per asset. Even after "
            "filtering, treat the result as indicative. Ten observations per "
            "asset is a reasonable minimum for weighting decisions.")

    if n_signal <= 1:
        verdict, msg = "minimal", (
            f"**Only the market mode survives.** {n_signal} of {n_assets} "
            f"eigenvalues exceed the noise bound, explaining "
            f"{var_signal:.0%} of variance. With {n_obs} observations the "
            "sample supports one fact about co-movement: these assets move "
            "together. Any weighting scheme claiming to know more than that is "
            "fitting noise, and equal-weight is a defensible choice here.")
    elif n_signal <= 3:
        verdict, msg = "ok", (
            f"{n_signal} eigenvalues carry structure ({var_signal:.0%} of "
            f"variance) — typically the market factor plus one or two sector "
            f"modes. {n_noise} noise eigenvalues clipped. Filtered weights will "
            "be materially more stable than raw ones.")
    else:
        verdict, msg = "good", (
            f"{n_signal} of {n_assets} eigenvalues carry structure, explaining "
            f"{var_signal:.0%} of variance. The sample is informative enough to "
            "support a differentiated weighting scheme.")

    return RMTResult(
        n_assets=n_assets, n_observations=n_obs, q_ratio=round(n_obs / n_assets, 2),
        lambda_plus=round(hi, 4), lambda_minus=round(lo, 4),
        eigenvalues=[round(float(v), 4) for v in vals],
        n_signal=n_signal, n_noise=n_noise,
        variance_explained_by_signal=round(var_signal, 4),
        filtered_correlation=filtered, filtered_covariance=filtered_cov,
        condition_before=round(cond_before, 1),
        condition_after=round(cond_after, 1),
        verdict=verdict, message=msg, notes=notes)


def effective_bets_filtered(result: RMTResult) -> dict:
    """Independent bets implied by the filtered matrix.

    Comparing this against the raw figure shows how much apparent
    diversification was estimation error.
    """
    if result.filtered_correlation is None:
        return {"error": "No filtered matrix."}

    c = result.filtered_correlation
    n = c.shape[0]
    off = c[~np.eye(n, dtype=bool)]
    rho = float(np.nanmean(off))
    denom = 1 + (n - 1) * rho
    eff = n / denom if denom > 0 else float(n)

    return {
        "positions": n,
        "mean_correlation_filtered": round(rho, 4),
        "effective_bets_filtered": round(eff, 2),
        "signal_eigenvalues": result.n_signal,
        "note": (
            f"After denoising, {n} positions behave as {eff:.1f} independent "
            f"bets (mean correlation {rho:.2f}). Only {result.n_signal} "
            "eigenvalue(s) carried real structure, so any weighting scheme "
            "should be correspondingly humble."),
    }


def compare_weighting(returns: pd.DataFrame) -> pd.DataFrame:
    """Portfolio volatility under raw, shrunk and RMT-filtered covariance.

    The useful comparison is not which produces the lowest apparent volatility —
    a noisy matrix always wins that, because it has found a spurious hedge. It
    is which produces stable weights.
    """
    r = returns.dropna(axis=1, how="all").dropna()
    if r.shape[1] < 2 or len(r) < r.shape[1] + 2:
        return pd.DataFrame()

    n = r.shape[1]
    w_eq = np.full(n, 1.0 / n)

    raw_cov = r.cov().to_numpy()
    shrunk = 0.85 * raw_cov + 0.15 * np.diag(np.diag(raw_cov))
    rmt = filter_correlation(r)
    filt_cov = rmt.filtered_covariance if rmt.filtered_covariance is not None else raw_cov

    rows = []
    for name, cov in (("raw", raw_cov), ("shrunk", shrunk), ("rmt_filtered", filt_cov)):
        try:
            inv = np.linalg.pinv(cov)
            w_mv = inv @ np.ones(n)
            s = w_mv.sum()
            w_mv = w_mv / s if abs(s) > 1e-12 else w_eq
        except Exception:                                      # noqa: BLE001
            w_mv = w_eq
        rows.append({
            "covariance": name,
            "condition_number": round(float(np.linalg.cond(cov)), 1),
            "min_weight_pct": round(float(w_mv.min()) * 100, 1),
            "max_weight_pct": round(float(w_mv.max()) * 100, 1),
            "weight_spread_pct": round(float(w_mv.max() - w_mv.min()) * 100, 1),
            "has_negative_weights": bool((w_mv < -0.001).any()),
            "portfolio_vol_ann_pct": round(
                float(np.sqrt(max(w_mv @ cov @ w_mv, 0) * 252)) * 100, 2),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Lempel-Ziv complexity — a cheap structure detector
# --------------------------------------------------------------------------
def lempel_ziv_complexity(series: pd.Series | np.ndarray,
                          *, n_bins: int = 3) -> dict:
    """How structured is recent price action?

    Prices are converted to a symbolic sequence (down / flat / up) and the
    number of distinct substrings counted. Fewer distinct patterns means more
    repetition, which means more structure.

    Included because it is cheap, non-parametric, and used as a gate rather
    than a signal. It says something about whether conditions are orderly, not
    which stock to buy.

    Normalised against a shuffled control, so the figure is comparable across
    series lengths.
    """
    x = np.asarray(pd.Series(series).dropna(), dtype=float)
    if len(x) < 40:
        return {"error": f"Only {len(x)} observations; need 40+."}

    rets = np.diff(np.log(np.clip(x, 1e-12, None)))
    if len(rets) < 30:
        return {"error": "Too few returns."}

    # Symbolise by tercile, so the alphabet is balanced by construction
    edges = np.percentile(rets, np.linspace(0, 100, n_bins + 1)[1:-1])
    symbols = np.digitize(rets, edges)
    seq = "".join(str(s) for s in symbols)

    def _lz(s: str) -> int:
        i, k, l, c = 0, 1, 1, 1
        n = len(s)
        k_max = 1
        while True:
            if i + k > n or l + k > n:
                break
            if s[i:i + k] == s[l:l + k]:
                k += 1
                if l + k > n:
                    c += 1
                    break
            else:
                if k > k_max:
                    k_max = k
                i += 1
                if i == l:
                    c += 1
                    l += k_max
                    if l + 1 > n:
                        break
                    i, k, k_max = 0, 1, 1
                else:
                    k = 1
        return c

    observed = _lz(seq)

    rng = np.random.default_rng(0)
    controls = [_lz("".join(str(s) for s in rng.permutation(symbols)))
                for _ in range(20)]
    baseline = float(np.mean(controls))
    normalised = observed / baseline if baseline > 0 else 1.0

    if normalised < 0.90:
        state = "structured"
        note = ("Price action is more repetitive than a shuffled control — "
                "recognisable patterns are recurring. Historically associated "
                "with compressing volatility ahead of a directional move.")
    elif normalised > 1.05:
        state = "disordered"
        note = ("More complex than a shuffled control. Unusual, and generally "
                "means conditions are noisy rather than trending.")
    else:
        state = "typical"
        note = "Complexity close to random. No structural signal."

    return {
        "lz_complexity": observed,
        "shuffled_baseline": round(baseline, 1),
        "normalised": round(normalised, 3),
        "state": state,
        "sequence_length": len(seq),
        "note": note,
        "caveat": ("A gate, not a signal. It says whether conditions look "
                   "orderly; it does not rank stocks and has not been tested "
                   "for predictive content."),
    }
