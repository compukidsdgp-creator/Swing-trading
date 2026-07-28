"""Local historical dataset loader — 20 years of NSE daily bars.

What this dataset is
--------------------
451 CSV files, one per symbol, 1.9 million rows, 2 January 2006 to 27 July 2026.
Schema is identical across every file. 331 symbols carry 15+ years of history,
which is what makes 2008 testable for the first time.

Columns: Date, Open, High, Low, Close, Adj Close, Volume.

**Close is split-adjusted but not dividend-adjusted. Adj Close is both.**
Verified by checking known bonus issues — Reliance 1:1 in 2017, TCS 1:1 in 2018,
Infosys several — none appear as large drops in Close. The Close/Adj Close ratio
(1.17 for Reliance, 1.59 for Infosys over twenty years) is cumulative dividends.

For momentum, `Adj Close` is the correct series: total return is what an investor
actually earns, and dividend drops would otherwise register as price weakness.

The limitation, stated plainly
------------------------------
Every file runs to the present. Not one ends early. That means this is **today's
constituent list backfilled twenty years** — companies that were delisted,
merged or wound up over that period are absent entirely.

So this extends history without fixing survivorship bias. Results computed on it
remain optimistic for the same reason five-year results were, just over a longer
window. What it genuinely unlocks is crisis coverage: 2008 and 2020 are now
testable, and momentum crash behaviour has never been examined against a real
crash.

Treat conclusions accordingly: strong evidence about *regime behaviour*, weaker
evidence about *absolute returns*.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DIR = Path("data")

# Notable regimes in Indian equities, for sub-period analysis. Momentum is
# documented to behave very differently across these.
REGIMES = {
    "pre_gfc_bull":    ("2006-01-01", "2008-01-08"),
    "gfc_crash":       ("2008-01-09", "2009-03-09"),
    "gfc_recovery":    ("2009-03-10", "2010-11-05"),
    "sideways":        ("2010-11-06", "2013-08-28"),
    "modi_rally":      ("2013-08-29", "2015-03-04"),
    "consolidation":   ("2015-03-05", "2017-01-01"),
    "bull_2017":       ("2017-01-02", "2018-01-29"),
    "correction":      ("2018-01-30", "2020-01-14"),
    "covid_crash":     ("2020-01-15", "2020-03-23"),
    "covid_recovery":  ("2020-03-24", "2021-10-18"),
    "rate_hikes":      ("2021-10-19", "2023-03-28"),
    "recent_bull":     ("2023-03-29", "2026-07-27"),
}


def _trim_padding(df: pd.DataFrame, *, min_run: int = 3) -> tuple[pd.DataFrame, int]:
    """Drop leading bars that are padding rather than trading.

    Padding is identifiable by two signatures together: zero volume, and a
    price that does not move. Either alone can occur legitimately — a genuinely
    illiquid session, or a stock at a circuit limit — but the combination,
    sustained, means no trading took place.

    Returns (trimmed_frame, bars_removed).
    """
    if df.empty or len(df) < min_run * 2:
        return df, 0

    vol = df["Volume"].to_numpy()
    close = df["Close"].to_numpy()

    # Walk forward while volume is zero AND price is static
    i = 0
    n = len(df)
    while i < n - min_run:
        if vol[i] > 0:
            break
        # price static relative to its own level, not an absolute epsilon
        window = close[i:i + min_run]
        level = float(np.mean(window))
        if level > 0 and float(np.std(window)) / level > 1e-9:
            break
        i += 1

    if i == 0:
        return df, 0

    # Extend through any remaining contiguous zero-volume bars
    while i < n and vol[i] == 0:
        i += 1

    return df.iloc[i:], i


@dataclass
class LoadResult:
    frames: dict[str, pd.DataFrame]
    n_files: int
    n_loaded: int
    date_range: tuple
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        s, e = self.date_range
        return (f"{self.n_loaded}/{self.n_files} symbols, "
                f"{s} to {e}, {sum(len(d) for d in self.frames.values()):,} bars")


def load(
    directory: str | Path = DEFAULT_DIR,
    *,
    use_adjusted: bool = True,
    min_bars: int = 300,
    max_symbols: int | None = None,
    start: str | None = None,
    end: str | None = None,
    symbols: list[str] | None = None,
) -> LoadResult:
    """Load the dataset into the frame format the rest of the system expects.

    Args:
        use_adjusted: use Adj Close (total return) rather than Close. Strongly
                      recommended — dividend drops would otherwise look like
                      price weakness to a momentum signal.
        min_bars: skip symbols with less history than this.
        max_symbols: cap for faster iteration during development.
        start / end: restrict the date window, for sub-period analysis.
        symbols: explicit list, otherwise everything in the directory.
    """
    d = Path(directory)
    if not d.exists():
        return LoadResult({}, 0, 0, (None, None),
                          warnings=[f"Directory not found: {d}"])

    files = sorted(glob.glob(str(d / "*.csv")))
    if symbols:
        wanted = {s.upper().replace(".NS", "") for s in symbols}
        files = [f for f in files
                 if Path(f).stem.replace("_NS_1d", "").upper() in wanted]

    frames, skipped, warnings = {}, [], []
    all_start, all_end = None, None

    for f in files:
        sym = Path(f).stem.replace("_NS_1d", "").upper()
        try:
            df = pd.read_csv(f, parse_dates=["Date"])
        except Exception as exc:                               # noqa: BLE001
            skipped.append(f"{sym}: {type(exc).__name__}")
            continue

        needed = {"Date", "Open", "High", "Low", "Close", "Volume"}
        if not needed.issubset(df.columns):
            skipped.append(f"{sym}: missing columns")
            continue

        df = df.set_index("Date").sort_index()

        # Rescale OHLC by the Close/Adj Close factor so the whole bar is
        # consistent. Using Adj Close alongside raw High/Low would corrupt ATR
        # and every range-based calculation.
        if use_adjusted and "Adj Close" in df.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                factor = df["Adj Close"] / df["Close"].replace(0, np.nan)
            factor = factor.replace([np.inf, -np.inf], np.nan).ffill().bfill()
            for c in ("Open", "High", "Low"):
                df[c] = df[c] * factor
            df["Close"] = df["Adj Close"]

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

        # --- Remove pre-listing padding ---
        #
        # yfinance pads periods before a stock actually traded with a repeated
        # placeholder price and zero volume. When real data begins, the jump
        # registers as an enormous return: HINDZINC shows +5,575% and
        # ABBOTINDIA +188%, both pure artefacts.
        #
        # A momentum screen would rank those stocks first. Trimming the padded
        # prefix is essential, not cosmetic.
        df, trimmed = _trim_padding(df)
        if trimmed:
            warnings.append(f"{sym}: trimmed {trimmed} padded leading bars "
                            "(zero volume, frozen price)")

        if len(df) < min_bars:
            skipped.append(f"{sym}: only {len(df)} bars")
            continue

        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        if len(df) < min_bars:
            skipped.append(f"{sym}: {len(df)} bars after date filter")
            continue

        frames[f"{sym}.NS"] = df
        all_start = df.index[0] if all_start is None else min(all_start, df.index[0])
        all_end = df.index[-1] if all_end is None else max(all_end, df.index[-1])

        if max_symbols and len(frames) >= max_symbols:
            break

    if use_adjusted:
        warnings.append(
            "Using Adj Close (total return). OHLC rescaled by the adjustment "
            "factor so ranges stay internally consistent."
        )
    warnings.append(
        "SURVIVORSHIP: every file runs to the present, so this is today's "
        "constituent list backfilled. Delisted and merged companies are absent. "
        "Absolute returns are optimistic; regime behaviour is the more reliable "
        "conclusion."
    )

    return LoadResult(
        frames, len(files), len(frames),
        (all_start.date().isoformat() if all_start is not None else None,
         all_end.date().isoformat() if all_end is not None else None),
        skipped, warnings,
    )


def load_regime(directory: str | Path, regime: str, **kwargs) -> LoadResult:
    """Load a single named market regime."""
    if regime not in REGIMES:
        raise ValueError(f"Unknown regime '{regime}'. Options: {list(REGIMES)}")
    s, e = REGIMES[regime]
    return load(directory, start=s, end=e, **kwargs)


def coverage_by_year(directory: str | Path = DEFAULT_DIR) -> pd.DataFrame:
    """How many symbols have data in each year — the sample is not constant."""
    files = sorted(glob.glob(str(Path(directory) / "*.csv")))
    counts: dict[int, int] = {}
    for f in files:
        try:
            df = pd.read_csv(f, usecols=["Date"], parse_dates=["Date"])
        except Exception:                                      # noqa: BLE001
            continue
        for y in df["Date"].dt.year.unique():
            counts[int(y)] = counts.get(int(y), 0) + 1
    out = pd.DataFrame(sorted(counts.items()), columns=["year", "symbols"])
    out["pct_of_max"] = (out["symbols"] / out["symbols"].max() * 100).round(1)
    return out


def audit(result: LoadResult) -> pd.DataFrame:
    """Data-quality pass over the loaded frames."""
    import data_quality as dq

    rows = []
    for t, df in result.frames.items():
        issues = dq.audit_frame(df)
        rows.append({
            "symbol": t.replace(".NS", ""),
            "bars": len(df),
            "start": df.index[0].date().isoformat(),
            "end": df.index[-1].date().isoformat(),
            "issues": len(issues),
            "detail": "; ".join(issues[:2]) if issues else "",
        })
    return pd.DataFrame(rows).sort_values("issues", ascending=False)
