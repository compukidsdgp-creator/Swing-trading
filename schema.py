"""Schema contracts — validating data shape before it reaches the model.

The gap this closes
-------------------
Health checks already exist and catch staleness, sparse coverage and implausible
prices. What was missing is a check on *structure*: that a frame has the columns
expected, in the types expected, with an index that behaves like a trading
calendar.

Those are different failures. A frame can be perfectly fresh and still have a
duplicated index, a string in a price column, or High below Low — and each
produces a confident, wrong answer rather than an error.

Two such defects reached production in this project:

  * yfinance returned a partial bar with a NaN close. Every last-row comparison
    became False, and the screener silently returned nothing.
  * Bhavcopy frames grouped by symbol alone carried duplicate dates, inflating
    the calendar 7.8x and multiplying window counts.

Both are structural, and both would have been caught here.

Why plain pandas rather than pandera
------------------------------------
Pandera and Pydantic are the conventional choices. Neither is used, for one
reason: they are additional dependencies in a project where a hand-written
version pin already broke the deployment once. The checks below are twenty lines
of pandas and have no install step.

If the validation logic ever grows beyond what is here, pandera becomes the
right call. It has not yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Required columns for an OHLCV frame
OHLCV_REQUIRED = ("Open", "High", "Low", "Close", "Volume")

# Plausible bounds for NSE equities, in rupees. Deliberately wide — these catch
# structural nonsense, not unusual prices.
PRICE_MIN = 0.5
PRICE_MAX = 500_000.0


@dataclass
class SchemaViolation:
    ticker: str
    rule: str
    detail: str
    severity: str          # critical | warning


@dataclass
class ValidationReport:
    checked: int = 0
    passed: int = 0
    violations: list[SchemaViolation] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)

    @property
    def critical(self) -> list[SchemaViolation]:
        return [v for v in self.violations if v.severity == "critical"]

    @property
    def ok(self) -> bool:
        return len(self.critical) == 0

    def summary(self) -> str:
        if not self.violations:
            return f"All {self.checked} frames satisfy the schema."
        crit = len(self.critical)
        warn = len(self.violations) - crit
        parts = [f"{self.checked} frames checked, {self.passed} clean."]
        if crit:
            parts.append(f"{crit} CRITICAL violation(s) — {len(self.rejected)} "
                         "frame(s) rejected.")
        if warn:
            parts.append(f"{warn} warning(s).")
        return " ".join(parts)

    def detail_table(self) -> pd.DataFrame:
        if not self.violations:
            return pd.DataFrame()
        return pd.DataFrame([{
            "ticker": v.ticker, "rule": v.rule,
            "severity": v.severity, "detail": v.detail,
        } for v in self.violations])


def validate_ohlcv(df: pd.DataFrame, ticker: str = "") -> list[SchemaViolation]:
    """Structural checks on a single OHLCV frame.

    Critical violations mean the frame cannot be used. Warnings mean it is
    usable but odd, and worth knowing about.
    """
    v: list[SchemaViolation] = []

    def crit(rule, detail):
        v.append(SchemaViolation(ticker, rule, detail, "critical"))

    def warn(rule, detail):
        v.append(SchemaViolation(ticker, rule, detail, "warning"))

    if df is None:
        crit("exists", "Frame is None.")
        return v
    if df.empty:
        crit("non_empty", "Frame has no rows.")
        return v

    # --- Columns ---
    missing = [c for c in OHLCV_REQUIRED if c not in df.columns]
    if missing:
        crit("required_columns", f"Missing {missing}.")
        return v

    # --- Types ---
    for c in OHLCV_REQUIRED:
        if not pd.api.types.is_numeric_dtype(df[c]):
            crit("numeric_types", f"Column '{c}' is {df[c].dtype}, not numeric.")
    if v:
        return v

    # --- Index ---
    if not isinstance(df.index, pd.DatetimeIndex):
        crit("datetime_index", f"Index is {type(df.index).__name__}, "
                               "not DatetimeIndex.")
        return v
    if not df.index.is_monotonic_increasing:
        crit("index_sorted", "Index is not sorted ascending.")
    if not df.index.is_unique:
        n_dupes = int(df.index.duplicated().sum())
        # This exact defect inflated a calendar 7.8x in production.
        crit("index_unique", f"{n_dupes} duplicate date(s) in the index.")

    # --- Prices present ---
    if df["Close"].isna().all():
        crit("close_populated", "Every close is NaN.")
        return v
    n_nan_close = int(df["Close"].isna().sum())
    if n_nan_close:
        # The partial-bar defect. Critical when it is the LAST row, because
        # every last-row comparison then silently evaluates False.
        last_nan = bool(pd.isna(df["Close"].iloc[-1]))
        if last_nan:
            crit("close_last_row", "Final close is NaN — a partial bar. Every "
                                   "last-row comparison would evaluate False.")
        else:
            warn("close_gaps", f"{n_nan_close} NaN close(s) mid-series.")

    # --- Price sanity ---
    c = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if (c <= 0).any():
        crit("positive_prices", f"{int((c <= 0).sum())} non-positive close(s).")
    if len(c):
        if c.max() > PRICE_MAX:
            warn("price_range", f"Max close {c.max():,.0f} exceeds "
                                f"{PRICE_MAX:,.0f}.")
        if c.min() < PRICE_MIN:
            warn("price_range", f"Min close {c.min():.2f} below {PRICE_MIN}.")

    # --- OHLC coherence ---
    h = pd.to_numeric(df["High"], errors="coerce")
    l = pd.to_numeric(df["Low"], errors="coerce")
    o = pd.to_numeric(df["Open"], errors="coerce")
    cl = pd.to_numeric(df["Close"], errors="coerce")
    valid = h.notna() & l.notna()
    if valid.any():
        n_bad = int((h[valid] < l[valid]).sum())
        if n_bad:
            crit("high_ge_low", f"{n_bad} bar(s) where High < Low.")
    both = valid & cl.notna() & o.notna()
    if both.any():
        outside = int(((cl[both] > h[both]) | (cl[both] < l[both])
                       | (o[both] > h[both]) | (o[both] < l[both])).sum())
        if outside:
            frac = outside / int(both.sum())
            if frac > 0.01:
                crit("ohlc_coherent",
                     f"{outside} bar(s) with Open/Close outside High-Low "
                     f"({frac:.1%}).")
            else:
                warn("ohlc_coherent", f"{outside} bar(s) with Open/Close "
                                      "outside High-Low.")

    # --- Volume ---
    vol = pd.to_numeric(df["Volume"], errors="coerce")
    if (vol < 0).any():
        crit("volume_non_negative", f"{int((vol < 0).sum()) } negative volume(s).")
    if vol.notna().any() and (vol.fillna(0) == 0).mean() > 0.5:
        warn("volume_present", f"{(vol.fillna(0) == 0).mean():.0%} of bars have "
                               "zero volume.")

    return v


def validate_frames(frames: dict[str, pd.DataFrame], *,
                    drop_invalid: bool = True) -> tuple[dict, ValidationReport]:
    """Validate a universe. Returns (clean_frames, report).

    Frames with critical violations are dropped by default. Passing bad data
    through to be caught later produces a confident wrong answer rather than an
    error, which is the failure mode this exists to prevent.
    """
    report = ValidationReport()
    clean = {}

    for t, df in (frames or {}).items():
        report.checked += 1
        violations = validate_ohlcv(df, t)
        report.violations.extend(violations)
        if any(v.severity == "critical" for v in violations):
            report.rejected.append(t)
            if not drop_invalid:
                clean[t] = df
        else:
            report.passed += 1
            clean[t] = df

    return clean, report


def validate_bhavcopy(day: pd.DataFrame) -> list[SchemaViolation]:
    """Structural checks on a single day's bhavcopy."""
    v: list[SchemaViolation] = []
    if day is None or day.empty:
        v.append(SchemaViolation("", "non_empty", "Empty bhavcopy.", "critical"))
        return v

    required = ("symbol", "close")
    missing = [c for c in required if c not in day.columns]
    if missing:
        v.append(SchemaViolation("", "required_columns", f"Missing {missing}.",
                                 "critical"))
        return v

    if day["symbol"].duplicated().any():
        n = int(day["symbol"].duplicated().sum())
        # Multiple series per symbol. Grouping by symbol alone then produces
        # duplicate dates, which is how a calendar got inflated 7.8x.
        v.append(SchemaViolation(
            "", "symbol_unique",
            f"{n} duplicate symbol(s) — likely multiple series (EQ, BE). "
            "Deduplicate on (symbol, series) before building price history.",
            "warning"))

    c = pd.to_numeric(day["close"], errors="coerce")
    if c.isna().all():
        v.append(SchemaViolation("", "close_numeric",
                                 "No parseable close prices.", "critical"))
    elif (c.dropna() <= 0).any():
        v.append(SchemaViolation(
            "", "positive_prices",
            f"{int((c.dropna() <= 0).sum())} non-positive close(s).", "warning"))

    if len(day) < 100:
        v.append(SchemaViolation(
            "", "row_count",
            f"Only {len(day)} rows — a full NSE session has 2,000+. Likely a "
            "partial download.", "warning"))

    return v


def validate_returns_matrix(returns: pd.DataFrame) -> list[SchemaViolation]:
    """Checks on a returns matrix before it reaches a covariance estimator.

    A returns matrix with too few observations relative to assets produces a
    covariance estimate that is mostly noise — which an optimiser will then
    exploit. Random matrix theory quantifies this; the check below flags it.
    """
    v: list[SchemaViolation] = []
    if returns is None or returns.empty:
        v.append(SchemaViolation("", "non_empty", "Empty returns matrix.",
                                 "critical"))
        return v

    n_obs, n_assets = returns.shape
    if n_assets < 2:
        v.append(SchemaViolation("", "min_assets",
                                 f"Only {n_assets} asset(s).", "critical"))
    if n_obs <= n_assets:
        v.append(SchemaViolation(
            "", "observations_vs_assets",
            f"{n_obs} observations for {n_assets} assets — the sample covariance "
            "is singular and any inverse is meaningless.", "critical"))
    elif n_obs < n_assets * 10:
        # Threshold set from measurement, not convention. RMT filtering on a
        # 60x10 matrix (6 observations per asset) found exactly ONE eigenvalue
        # above the Marchenko-Pastur noise bound. Ten per asset is the point at
        # which a second mode becomes reliably detectable.
        v.append(SchemaViolation(
            "", "observations_vs_assets",
            f"Only {n_obs / n_assets:.1f} observations per asset. At this ratio "
            "typically just one eigenvalue exceeds the random-matrix noise "
            "bound, so any weighting scheme beyond equal-weight is fitting "
            "noise.", "warning"))

    if returns.isna().any().any():
        v.append(SchemaViolation(
            "", "no_missing",
            f"{int(returns.isna().sum().sum())} missing value(s).", "warning"))

    extreme = (returns.abs() > 1.0).sum().sum()
    if extreme:
        v.append(SchemaViolation(
            "", "plausible_returns",
            f"{int(extreme)} daily return(s) beyond ±100% — likely unadjusted "
            "corporate actions.", "warning"))

    return v
