"""Advanced statistical validation — correcting for the things already known wrong.

Why these four and not the other eighty
---------------------------------------
A long list of statistical methods exists for swing trading. Most of it is
inapplicable here: ARIMA and GARCH forecast a price level rather than rank a
cross-section; cointegration and copulas belong to pairs trading; tree
ensembles, topological data analysis and chaos measures need orders of magnitude
more observations than 30 non-overlapping windows provide.

The four implemented here each address a weakness this project already has on
record.

**1. Deflated Sharpe Ratio** (Bailey & López de Prado). Twenty-four horizon and
bucket configurations were tested and the best reported. That is textbook
selection bias, and nothing in the system currently corrects for it. DSR adjusts
a Sharpe ratio for the number of trials, the length of the track record, and
non-normality — and frequently reveals that an apparently strong result is what
noise produces when you look twenty-four times.

**2. Combinatorial Purged Cross-Validation** (López de Prado). A 30-day holding
period means an observation made on day 1 overlaps observations made on days
2 through 29. Standard cross-validation leaks across that boundary. Purging
removes overlapping samples and embargoing adds a gap, which is the only honest
way to split a series with multi-day labels.

**3. Hurst exponent and Variance Ratio.** The 30-day horizon was selected as the
best of a sweep, which is weak grounds. These test the question directly: does
this series exhibit trend persistence at all, and over what horizon? A Hurst
exponent below 0.5 would mean momentum is the wrong family of strategy
regardless of what the sweep found.

**4. Block bootstrap** (Politis & Romano). Return series are serially
correlated, so an ordinary bootstrap that resamples individual observations
destroys the dependence structure and produces confidence intervals that are too
narrow. Resampling blocks preserves it.

None of these adds a signal. All four make existing conclusions more honest —
and two of them may well weaken conclusions rather than strengthen them, which
is the point.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EULER_MASCHERONI = 0.5772156649015329


# --------------------------------------------------------------------------
# 1. Deflated and Probabilistic Sharpe Ratio
# --------------------------------------------------------------------------
@dataclass
class SharpeAssessment:
    observed_sharpe: float
    n_observations: int
    n_trials: int
    skew: float
    kurtosis: float
    psr: float                  # P(true Sharpe > benchmark)
    expected_max_sharpe: float  # what noise alone would produce
    deflated_sharpe: float      # P(true Sharpe > 0) after selection correction
    verdict: str
    message: str
    notes: list[str] = field(default_factory=list)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF, Acklam's rational approximation."""
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def probabilistic_sharpe(sharpe: float, n: int, skew: float, kurt: float,
                         benchmark: float = 0.0) -> float:
    """P(true Sharpe > benchmark), adjusted for skew, kurtosis and track length.

    A Sharpe of 1.0 over 20 observations is far weaker evidence than the same
    figure over 200, and negative skew weakens it further. PSR expresses that
    as a probability rather than leaving it implicit.
    """
    if n < 3:
        return float("nan")
    # kurt here is EXCESS kurtosis; the formula needs the raw fourth moment
    k = kurt + 3.0
    denom = 1.0 - skew * sharpe + (k - 1.0) / 4.0 * sharpe ** 2
    if denom <= 0:
        return float("nan")
    z = (sharpe - benchmark) * math.sqrt(n - 1) / math.sqrt(denom)
    return _norm_cdf(z)


def expected_max_sharpe(n_trials: int, sharpe_variance: float = 1.0) -> float:
    """Sharpe ratio the best of N random trials would show by chance.

    This is the quantity that makes "we tested 24 configurations and the best
    gave 0.9" so much weaker than "we tested one and it gave 0.9". With enough
    trials, an impressive maximum is guaranteed.
    """
    if n_trials < 2:
        return 0.0
    sd = math.sqrt(sharpe_variance)
    a = (1 - EULER_MASCHERONI) * _norm_ppf(1 - 1.0 / n_trials)
    b = EULER_MASCHERONI * _norm_ppf(1 - 1.0 / (n_trials * math.e))
    return sd * (a + b)


def deflated_sharpe(returns: pd.Series | np.ndarray, *, n_trials: int,
                    sharpe_variance: float | None = None) -> SharpeAssessment:
    """Sharpe ratio corrected for selection bias, length and non-normality.

    Args:
        returns: per-period returns.
        n_trials: how many configurations were tried before this one was chosen.
                  Be honest here — understating it inflates the result, which
                  defeats the purpose.
    """
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    n = len(r)
    if n < 5:
        return SharpeAssessment(0, n, n_trials, 0, 0, float("nan"), 0,
                                float("nan"), "insufficient",
                                f"Only {n} observations.")

    mean, sd = float(r.mean()), float(r.std(ddof=1))
    sharpe = mean / sd if sd > 0 else 0.0
    skew = float(pd.Series(r).skew())
    kurt = float(pd.Series(r).kurtosis())     # excess

    psr = probabilistic_sharpe(sharpe, n, skew, kurt, 0.0)

    # Variance across trials. Without the full set of trial Sharpes, assume the
    # observed variance — a conservative default.
    sv = sharpe_variance if sharpe_variance is not None else 1.0 / max(n - 1, 1)
    exp_max = expected_max_sharpe(n_trials, sv)
    dsr = probabilistic_sharpe(sharpe, n, skew, kurt, exp_max)

    notes = [
        f"{n_trials} configurations assumed tested. Noise alone would produce a "
        f"best Sharpe of {exp_max:.3f} across that many trials.",
    ]
    if n < 30:
        notes.append(f"Only {n} observations — PSR and DSR both widen sharply "
                     "below about 30, and this is well inside that range.")
    if skew < -0.5:
        notes.append(f"Negative skew ({skew:.2f}) reduces PSR: occasional large "
                     "losses against frequent small gains.")
    if kurt > 3:
        notes.append(f"Excess kurtosis ({kurt:.2f}) — fat tails weaken the "
                     "Sharpe ratio's reliability as a summary.")

    if np.isnan(dsr):
        verdict, msg = "error", "Could not compute DSR — check the return series."
    elif dsr < 0.50:
        verdict, msg = "fail", (
            f"**Deflated Sharpe {dsr:.1%}.** After correcting for {n_trials} "
            f"trials, the observed Sharpe of {sharpe:.3f} is no better than what "
            f"selecting the best of {n_trials} random attempts would produce "
            f"({exp_max:.3f}). This result is not distinguishable from "
            "overfitting.")
    elif dsr < 0.90:
        verdict, msg = "warn", (
            f"Deflated Sharpe {dsr:.1%}. The result survives correction for "
            f"{n_trials} trials but not comfortably. The conventional bar is "
            "95%.")
    else:
        verdict, msg = "pass", (
            f"**Deflated Sharpe {dsr:.1%}.** The observed Sharpe of "
            f"{sharpe:.3f} exceeds what {n_trials} random trials would produce "
            f"({exp_max:.3f}) with {dsr:.1%} confidence. Selection bias does not "
            "explain it.")

    return SharpeAssessment(round(sharpe, 4), n, n_trials, round(skew, 3),
                            round(kurt, 3), round(psr, 4), round(exp_max, 4),
                            round(dsr, 4), verdict, msg, notes)


# --------------------------------------------------------------------------
# 2. Combinatorial Purged Cross-Validation
# --------------------------------------------------------------------------
@dataclass
class CPCVResult:
    n_splits: int
    n_paths: int
    mean_score: float
    std_score: float
    scores: list[float]
    purged_samples: int
    embargo_samples: int
    verdict: str
    message: str


def purged_kfold_indices(n_samples: int, n_splits: int, *, horizon: int,
                         embargo_pct: float = 0.01) -> list:
    # NOTE: `horizon` here IS in samples, because this operates on raw
    # observations rather than the already-spaced window series used by
    # combinatorial_purged_cv.
    """Train/test index pairs with overlapping samples purged.

    With a 30-day holding period, an observation made on day 1 shares 29 days
    with one made on day 2. Standard k-fold puts those in different folds and
    the model effectively sees its test data.

    Purging drops training samples whose label window overlaps the test set.
    Embargoing adds a further gap after the test set, because information leaks
    forwards as well as backwards.
    """
    indices = np.arange(n_samples)
    fold_size = n_samples // n_splits
    embargo = int(n_samples * embargo_pct)
    out = []

    for k in range(n_splits):
        start = k * fold_size
        stop = start + fold_size if k < n_splits - 1 else n_samples
        test = indices[start:stop]

        # Purge: any training sample whose label window touches the test range
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[max(0, start - horizon):min(n_samples, stop + horizon)] = False
        # Embargo: extend the exclusion after the test block
        if embargo:
            train_mask[stop:min(n_samples, stop + horizon + embargo)] = False
        train = indices[train_mask]
        if len(train) > horizon and len(test) > 2:
            out.append((train, test))
    return out


def combinatorial_purged_cv(
    scores_by_window: pd.Series,
    *,
    purge_windows: int = 1,
    n_splits: int = 6,
    n_test_groups: int = 2,
    embargo_pct: float = 0.02,
) -> CPCVResult:
    """Evaluate stability across purged combinatorial splits.

    Rather than one train/test division, this forms every combination of
    `n_test_groups` held-out blocks from `n_splits` groups, giving many
    backtest paths instead of a single one. The spread across paths is the
    useful output: a strategy whose score varies wildly between paths has not
    demonstrated much.

    Args:
        scores_by_window: per-window metric, e.g. IC per rebalance.
        purge_windows: how many neighbouring WINDOWS to drop around each test
            block. Note the unit: the input series is already non-overlapping
            (windows are spaced horizon+3 days apart), so purging is measured in
            windows, not days. Passing the day-count here purges everything and
            leaves no valid paths — which is exactly what happened in testing.
    """
    s = pd.Series(scores_by_window).dropna()
    n = len(s)
    if n < n_splits * 3:
        return CPCVResult(0, 0, 0, 0, [], 0, 0, "insufficient",
                          f"Only {n} windows for {n_splits} splits. Need at "
                          f"least {n_splits * 3}.")

    fold_size = n // n_splits
    groups = [np.arange(k * fold_size,
                        (k + 1) * fold_size if k < n_splits - 1 else n)
              for k in range(n_splits)]

    embargo = max(1, int(n * embargo_pct))
    scores, purged_total = [], 0

    for combo in itertools.combinations(range(n_splits), n_test_groups):
        test_idx = np.concatenate([groups[c] for c in combo])
        mask = np.ones(n, dtype=bool)
        # Purge and embargo around every held-out block
        for c in combo:
            lo = max(0, groups[c][0] - purge_windows)
            hi = min(n, groups[c][-1] + purge_windows + embargo + 1)
            mask[lo:hi] = False
        train_idx = np.arange(n)[mask]
        purged_total += (n - len(train_idx) - len(test_idx))
        if len(train_idx) < 5 or len(test_idx) < 3:
            continue
        scores.append(float(s.iloc[test_idx].mean()))

    if len(scores) < 3:
        return CPCVResult(n_splits, len(scores), 0, 0, scores, purged_total,
                          embargo, "insufficient",
                          "Too few valid paths after purging.")

    arr = np.array(scores)
    mean, sd = float(arr.mean()), float(arr.std(ddof=1))
    pos = float((arr > 0).mean())

    if mean <= 0:
        verdict, msg = "fail", (
            f"Mean score across {len(scores)} purged paths is {mean:+.4f}. The "
            "result does not survive leakage-free splitting.")
    elif pos < 0.7:
        verdict, msg = "warn", (
            f"Only {pos:.0%} of {len(scores)} purged paths were positive "
            f"(mean {mean:+.4f}, sd {sd:.4f}). The result is path-dependent — it "
            "holds on some splits and not others.")
    elif sd > abs(mean):
        verdict, msg = "warn", (
            f"Mean {mean:+.4f} across paths but standard deviation {sd:.4f} "
            "exceeds it. Wide dispersion means the single-path figure was "
            "partly luck of the split.")
    else:
        verdict, msg = "pass", (
            f"Mean {mean:+.4f} across {len(scores)} purged paths, sd {sd:.4f}, "
            f"{pos:.0%} positive. The result holds under leakage-free splitting.")

    return CPCVResult(n_splits, len(scores), round(mean, 5), round(sd, 5),
                      [round(x, 5) for x in scores], purged_total, embargo,
                      verdict, msg)


# --------------------------------------------------------------------------
# 3. Hurst exponent and Variance Ratio
# --------------------------------------------------------------------------
@dataclass
class PersistenceResult:
    hurst: float
    hurst_interpretation: str
    variance_ratios: dict
    vr_verdict: str
    optimal_horizon_hint: int | None
    verdict: str
    message: str


def hurst_exponent(series: pd.Series | np.ndarray,
                   *, min_lag: int = 8, max_lag: int = 100,
                   calibrate: bool = True, n_shuffles: int = 20,
                   seed: int = 0) -> float:
    """Hurst exponent, self-calibrated against a shuffled control.

    Classical rescaled-range analysis is biased upward on finite samples: a
    genuine random walk returns roughly 0.6 rather than 0.5, and the bias grows
    as the series shortens. During testing a random series scored 0.645 and a
    mean-reverting one 0.532 — both wrong in the same direction.

    The correction: shuffling the series destroys any temporal structure, so its
    true Hurst is 0.5 by construction. Estimating H on the shuffle measures the
    estimator's bias directly, and subtracting it leaves a calibrated figure.
    Both estimates suffer the same bias, so it cancels.
    """
    return _hurst_calibrated(series, min_lag=min_lag, max_lag=max_lag,
                             calibrate=calibrate, n_shuffles=n_shuffles,
                             seed=seed)


def _hurst_raw(series: pd.Series | np.ndarray,
               *, min_lag: int = 8, max_lag: int = 100) -> float:
    """Hurst exponent by rescaled-range analysis.

    H > 0.5  trend persistence — momentum is the appropriate family
    H = 0.5  random walk — no exploitable structure
    H < 0.5  mean reversion — momentum is the wrong family entirely

    This asks a question the horizon sweep cannot: whether trend-following is
    the right approach at all, independent of which specific horizon looked
    best across twenty-four attempts.
    """
    x = np.asarray(pd.Series(series).dropna(), dtype=float)
    if len(x) < max_lag * 2:
        max_lag = max(min_lag + 4, len(x) // 3)
    if len(x) < min_lag * 3:
        return float("nan")

    lags = np.unique(np.logspace(np.log10(min_lag), np.log10(max_lag),
                                 12).astype(int))
    rs = []
    for lag in lags:
        n_chunks = len(x) // lag
        if n_chunks < 2:
            continue
        vals = []
        for i in range(n_chunks):
            chunk = x[i * lag:(i + 1) * lag]
            dev = np.cumsum(chunk - chunk.mean())
            r = dev.max() - dev.min()
            s = chunk.std(ddof=1)
            if s > 0 and np.isfinite(r):
                vals.append(r / s)
        if vals:
            rs.append((lag, np.mean(vals)))

    if len(rs) < 4:
        return float("nan")
    lg = np.log([a for a, _ in rs])
    lr = np.log([b for _, b in rs])
    return float(np.polyfit(lg, lr, 1)[0])


def _hurst_calibrated(series, *, min_lag: int = 8, max_lag: int = 100,
                      calibrate: bool = True, n_shuffles: int = 20,
                      seed: int = 0) -> float:
    """Raw R/S estimate less the bias measured on shuffled controls."""
    x = np.asarray(pd.Series(series).dropna(), dtype=float)
    raw = _hurst_raw(x, min_lag=min_lag, max_lag=max_lag)
    if not calibrate or not np.isfinite(raw) or len(x) < min_lag * 3:
        return raw

    rng = np.random.default_rng(seed)
    controls = []
    for _ in range(n_shuffles):
        h = _hurst_raw(rng.permutation(x), min_lag=min_lag, max_lag=max_lag)
        if np.isfinite(h):
            controls.append(h)
    if len(controls) < 5:
        return raw

    bias = float(np.mean(controls)) - 0.5
    return raw - bias


def variance_ratio(returns: pd.Series | np.ndarray, q: int) -> float:
    """Lo-MacKinlay variance ratio at lag q.

    VR > 1 — returns are positively autocorrelated at this horizon (trending)
    VR = 1 — random walk
    VR < 1 — mean reverting

    Computing this across horizons tests the 30-day choice directly rather than
    accepting it as the best of a sweep.
    """
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    n = len(r)
    if n < q * 3:
        return float("nan")
    var1 = np.var(r, ddof=1)
    if var1 <= 0:
        return float("nan")
    agg = np.array([r[i:i + q].sum() for i in range(0, n - q + 1)])
    varq = np.var(agg, ddof=1)
    return float(varq / (q * var1))


def assess_persistence(prices: pd.Series,
                       horizons: tuple[int, ...] = (5, 10, 20, 30, 45, 60)
                       ) -> PersistenceResult:
    """Does this series trend, and over what horizon?"""
    p = pd.Series(prices).dropna()
    if len(p) < 200:
        return PersistenceResult(float("nan"), "insufficient", {}, "insufficient",
                                 None, "insufficient",
                                 f"Only {len(p)} observations; need 200+.")

    rets = np.log(p).diff().dropna()

    # Hurst must be computed on RETURNS, not log prices. Rescaled-range
    # analysis on an already-integrated series returns H close to 1 whatever
    # the underlying dynamics — a random walk and a mean-reverting series both
    # scored 0.99 during testing, which is how the bug was found.
    h = hurst_exponent(rets)
    # Thresholds apply to the CALIBRATED estimate, where a random walk sits at
    # 0.5 by construction rather than at the ~0.6 raw R/S produces.
    if not np.isfinite(h):
        hint = "could not compute"
    elif h > 0.56:
        hint = "persistence — trend following is appropriate"
    elif h > 0.52:
        hint = "mild persistence"
    elif h >= 0.48:
        hint = "random walk — little exploitable structure at this scale"
    elif h > 0.44:
        hint = "mild mean reversion"
    else:
        hint = "mean reversion — momentum is the wrong family"

    vrs = {q: round(variance_ratio(rets, q), 3) for q in horizons}
    valid = {k: v for k, v in vrs.items() if np.isfinite(v)}

    best = None
    if valid:
        best = max(valid, key=lambda k: valid[k])
        trending = [k for k, v in valid.items() if v > 1.05]
        if not trending:
            vr_verdict = "no trending horizon"
        elif best in (30, 45, 60):
            vr_verdict = f"trending strongest at {best} days"
        else:
            vr_verdict = f"trending strongest at {best} days"
    else:
        vr_verdict = "insufficient"

    parts = [f"Hurst {h:.3f} — {hint}." if np.isfinite(h) else "Hurst unavailable."]
    if valid:
        parts.append("Variance ratios: " +
                     ", ".join(f"{k}d {v:.2f}" for k, v in valid.items()) + ".")
        if best and best not in (30, 45):
            parts.append(f"The strongest trending horizon is {best} days, not "
                         "the 30 currently configured. Worth noting, though a "
                         "single series is weak evidence — this should be run "
                         "across the universe.")

    if np.isfinite(h) and h < 0.46:
        verdict = "bad"
        parts.append("A Hurst exponent this low argues against momentum as a "
                     "family, not merely against the chosen horizon.")
    elif np.isfinite(h) and h > 0.54:
        verdict = "good"
    else:
        verdict = "neutral"
        parts.append("Close to a random walk. Consistent with a small edge that "
                     "needs many trades to express itself.")

    return PersistenceResult(round(h, 4) if np.isfinite(h) else float("nan"),
                             hint, vrs, vr_verdict, best, verdict,
                             " ".join(parts))


# --------------------------------------------------------------------------
# 4. Block bootstrap
# --------------------------------------------------------------------------
def block_bootstrap_ci(series: pd.Series | np.ndarray, *,
                       statistic=np.mean, n_boot: int = 2000,
                       block_size: int | None = None,
                       confidence: float = 0.95,
                       seed: int = 0) -> dict:
    """Confidence interval that preserves serial correlation.

    An ordinary bootstrap resamples individual observations and destroys the
    dependence structure, producing intervals that are too narrow — sometimes
    dramatically so for financial series. Resampling contiguous blocks keeps it.

    Block size defaults to n^(1/3), the standard rule of thumb.
    """
    x = np.asarray(pd.Series(series).dropna(), dtype=float)
    n = len(x)
    if n < 12:
        return {"error": f"Only {n} observations."}

    bs = block_size or max(2, int(round(n ** (1 / 3))))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)

    for i in range(n_boot):
        out, filled = [], 0
        while filled < n:
            start = rng.integers(0, n)
            blk = x[start:start + bs]
            if len(blk) < bs:            # wrap around, keeps blocks intact
                blk = np.concatenate([blk, x[:bs - len(blk)]])
            out.append(blk)
            filled += len(blk)
        stats[i] = statistic(np.concatenate(out)[:n])

    alpha = 1 - confidence
    lo = float(np.percentile(stats, alpha / 2 * 100))
    hi = float(np.percentile(stats, (1 - alpha / 2) * 100))
    point = float(statistic(x))

    # Naive bootstrap for comparison, to show how much the dependence matters
    naive = np.array([statistic(rng.choice(x, n, replace=True))
                      for _ in range(min(n_boot, 500))])
    naive_lo = float(np.percentile(naive, alpha / 2 * 100))
    naive_hi = float(np.percentile(naive, (1 - alpha / 2) * 100))

    widening = ((hi - lo) / (naive_hi - naive_lo)
                if (naive_hi - naive_lo) > 0 else 1.0)

    return {
        "point_estimate": round(point, 5),
        "ci_low": round(lo, 5),
        "ci_high": round(hi, 5),
        "block_size": bs,
        "n_bootstrap": n_boot,
        "includes_zero": bool(lo <= 0 <= hi),
        "naive_ci": [round(naive_lo, 5), round(naive_hi, 5)],
        "widening_vs_naive": round(widening, 2),
        "note": (
            f"Block bootstrap interval is {widening:.2f}x the width of a naive "
            "one. Ignoring serial correlation would have understated the "
            "uncertainty by that factor."
            if widening > 1.1 else
            "Block and naive intervals are similar — serial correlation is "
            "weak in this series."),
    }
