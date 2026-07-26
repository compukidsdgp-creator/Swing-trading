"""Data foundation — quality checks and survivorship-bias mitigation.

Two separate problems, with very different prospects.

Problem 1: data quality (LARGELY FIXABLE)
-----------------------------------------
yfinance is free and adequate for research but carries known defects: bad
adjusted closes around corporate actions, stale repeated prices, zero-volume
gaps, and occasional impossible returns. None of these are detectable by the
scoring code, which will happily rank a stock on a 400% single-day move that
was actually a mis-applied split.

`audit()` finds them. `clean()` removes the worst offenders. Neither is
sophisticated, but running no checks at all is how a single bad tick ends up
dominating an entire backtest.

Problem 2: survivorship bias (PARTIALLY FIXABLE)
------------------------------------------------
Using today's Nifty 500 to test the past five years means every company that
failed and was removed is invisible. Returns are biased upward, because the
sample is conditioned on survival.

The proper fix is a point-in-time database (CRSP, Compustat PIT, Refinitiv)
that reconstructs index membership by date. Those cost institutional money and
have no free Indian equivalent.

The partial fix implemented here: build the universe from **point-in-time
liquidity** rather than current index membership. At each rebalance date,
include every stock that had sufficient traded value *as of that date*, using
only data available then. Inclusion no longer depends on being in an index
today.

Honest limitation
-----------------
This removes *index-membership* survivorship bias — the reason a stock that was
in the Nifty 500 in 2021 and got demoted in 2023 is currently excluded. It does
NOT remove *delisting* bias, because a fully delisted company has no yfinance
history to retrieve at all. That residual bias is real, unavoidable at retail
data access, and should be assumed to inflate results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# A daily move beyond this is almost certainly a data error rather than a real
# price change. NSE circuit limits cap most stocks at 20% per day; 50% allows
# genuine outliers (results-day gaps, ex-bonus adjustments) while catching the
# split-adjustment errors that produce 500%+ jumps.
MAX_DAILY_MOVE = 0.50
MAX_STALE_RUN = 10          # consecutive identical closes
MAX_ZERO_VOLUME_PCT = 0.20


@dataclass
class AuditResult:
    clean: list[str] = field(default_factory=list)
    flagged: dict[str, list[str]] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    @property
    def summary(self) -> pd.DataFrame:
        rows = [{"ticker": t.replace(".NS", ""), "issues": "; ".join(v),
                 "n_issues": len(v)}
                for t, v in sorted(self.flagged.items())]
        return pd.DataFrame(rows)


def audit_frame(df: pd.DataFrame) -> list[str]:
    """Return a list of data-quality issues found in one OHLCV frame."""
    issues: list[str] = []
    if df is None or df.empty:
        return ["empty frame"]

    close = df["Close"]

    if len(df) < 260:
        issues.append(f"short history ({len(df)} bars)")

    ret = close.pct_change().abs()
    extreme = int((ret > MAX_DAILY_MOVE).sum())
    if extreme:
        worst = float(ret.max()) * 100
        issues.append(f"{extreme} move(s) over {MAX_DAILY_MOVE:.0%} (max {worst:.0f}%)")

    # Stale prices — identical closes for many consecutive sessions
    same = (close.diff() == 0)
    if same.any():
        run, best = 0, 0
        for v in same.to_numpy():
            run = run + 1 if v else 0
            best = max(best, run)
        if best >= MAX_STALE_RUN:
            issues.append(f"{best} consecutive identical closes")

    if "Volume" in df.columns:
        zpct = float((df["Volume"] <= 0).mean())
        if zpct > MAX_ZERO_VOLUME_PCT:
            issues.append(f"{zpct:.0%} of days have zero volume")

    if (close <= 0).any():
        issues.append("non-positive prices present")

    # OHLC internal consistency
    bad_hl = int((df["High"] < df["Low"]).sum())
    if bad_hl:
        issues.append(f"{bad_hl} bars with High < Low")
    bad_range = int(((df["Close"] > df["High"]) | (df["Close"] < df["Low"])).sum())
    if bad_range:
        issues.append(f"{bad_range} bars with Close outside High-Low")

    gaps = df.index.to_series().diff().dt.days
    big_gaps = int((gaps > 10).sum())
    if big_gaps:
        issues.append(f"{big_gaps} gap(s) over 10 days")

    return issues


def audit(frames: dict[str, pd.DataFrame]) -> AuditResult:
    """Audit an entire universe."""
    res = AuditResult()
    for t, df in frames.items():
        issues = audit_frame(df)
        if issues:
            res.flagged[t] = issues
        else:
            res.clean.append(t)

    total = len(frames)
    res.stats = {
        "total": total,
        "clean": len(res.clean),
        "flagged": len(res.flagged),
        "clean_pct": round(len(res.clean) / total * 100, 1) if total else 0.0,
    }
    return res


def clean(frames: dict[str, pd.DataFrame], *,
          drop_flagged: bool = False) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Repair or remove problematic data.

    Default behaviour repairs rather than drops: extreme single-day moves are
    neutralised by forward-filling the price, which is less destructive than
    discarding an otherwise sound series. Set drop_flagged=True to exclude any
    ticker with issues entirely.
    """
    out, removed = {}, []
    for t, df in frames.items():
        issues = audit_frame(df)
        if not issues:
            out[t] = df
            continue
        if drop_flagged:
            removed.append(t)
            continue

        fatal = any(k in " ".join(issues) for k in
                    ("empty frame", "non-positive prices", "High < Low"))
        if fatal:
            removed.append(t)
            continue

        d = df.copy()
        ret = d["Close"].pct_change().abs()
        bad = ret > MAX_DAILY_MOVE
        if bad.any():
            # Forward-fill through the suspect bar rather than trusting it
            d.loc[bad, ["Open", "High", "Low", "Close"]] = np.nan
            d[["Open", "High", "Low", "Close"]] = (
                d[["Open", "High", "Low", "Close"]].ffill()
            )
            d = d.dropna(subset=["Close"])
        if len(d) >= 260:
            out[t] = d
        else:
            removed.append(t)

    return out, removed


# --------------------------------------------------------------------------
# Point-in-time universe
# --------------------------------------------------------------------------
def pit_universe_at(
    frames: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    *,
    min_turnover_cr: float = 5.0,
    min_history_bars: int = 300,
    top_n: int | None = None,
) -> list[str]:
    """Which stocks would have qualified on `date`, judged only on data to date.

    This is the survivorship-bias mitigation. Membership is decided by traded
    value as of that date, not by whether the company is in an index today.
    """
    eligible = []
    for t, df in frames.items():
        if df is None or df.empty:
            continue
        try:
            i = df.index.get_loc(date)
        except KeyError:
            continue
        if not isinstance(i, int) or i < min_history_bars:
            continue

        w = df.iloc[max(0, i - 19): i + 1]
        turn_cr = float((w["Close"] * w["Volume"]).mean() / 1e7)
        if np.isfinite(turn_cr) and turn_cr >= min_turnover_cr:
            eligible.append((t, turn_cr))

    eligible.sort(key=lambda x: x[1], reverse=True)
    if top_n:
        eligible = eligible[:top_n]
    return [t for t, _ in eligible]


def pit_coverage_report(
    frames: dict[str, pd.DataFrame],
    *,
    step: int = 63,
    min_turnover_cr: float = 5.0,
) -> pd.DataFrame:
    """How universe membership would have changed over time.

    High churn is a good sign — it means membership is genuinely being decided
    at each date rather than inherited from today's index.
    """
    if not frames:
        return pd.DataFrame()
    cal = pd.DatetimeIndex(max((df.index for df in frames.values()), key=len))
    rows, prev = [], set()
    for k in range(300, len(cal), step):
        date = cal[k]
        members = set(pit_universe_at(frames, date, min_turnover_cr=min_turnover_cr))
        if not members:
            continue
        rows.append({
            "date": date,
            "n_eligible": len(members),
            "entered": len(members - prev) if prev else 0,
            "exited": len(prev - members) if prev else 0,
            "churn_pct": (round(len(members ^ prev) / max(len(prev), 1) * 100, 1)
                          if prev else 0.0),
        })
        prev = members
    return pd.DataFrame(rows)


def bias_note(frames: dict[str, pd.DataFrame]) -> str:
    """Plain statement of the residual bias that remains after mitigation."""
    n = len(frames)
    return (
        f"Universe built from {n} tickers with price history available today. "
        "Point-in-time liquidity filtering removes *index-membership* survivorship "
        "bias — a stock demoted from the Nifty 500 in 2023 is still included for the "
        "period it qualified. It does NOT remove *delisting* bias: companies that were "
        "wound up or delisted have no retrievable history and are structurally absent. "
        "Assume results remain modestly inflated. Eliminating this requires a "
        "point-in-time database, which has no free Indian equivalent."
    )
