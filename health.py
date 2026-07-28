"""Health checks — catching silent failure.

The failure mode this addresses
-------------------------------
An automated pipeline that breaks quietly is worse than one that breaks loudly.
If Monday's run fails — a rate limit, an API change, a dependency shift — the
symptom is simply that no Telegram message arrives. Which is indistinguishable
from a quiet weekend.

Three weeks of that and the forward log has holes in it. The eight-week evidence
being accumulated is then compromised, and it would be discovered at the worst
possible moment: when it is finally read.

Worse still is data that is stale rather than absent. If yfinance serves a cached
response from last week, the pipeline produces confident, well-formatted picks
from old prices and reports success. Nothing in the system currently notices.

What this checks
----------------
  **Freshness**   — is the newest bar actually recent, allowing for weekends
                    and NSE holidays?
  **Completeness** — did enough tickers return usable data?
  **Continuity**  — is the forward log still growing, or has it silently stalled?
  **Sanity**      — do prices and volumes look like real market data?

Design principle
----------------
Checks that matter abort the run. A pipeline that stops with a clear error is
recoverable; one that produces plausible output from bad data is not, because
the output goes into a log that is later read as evidence.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# NSE trades Mon-Fri. Allowing 5 calendar days covers a weekend plus a
# two-day holiday cluster (Diwali, for instance) without false alarms.
MAX_STALENESS_DAYS = 5
MIN_TICKER_COVERAGE = 0.60      # fraction of requested tickers that must return data
MAX_LOG_GAP_DAYS = 12           # weekly cadence plus slack before a stall is flagged


@dataclass
class HealthResult:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        if self.passed and not self.warnings:
            return f"All {len(self.checks)} health checks passed."
        parts = []
        if self.failures:
            parts.append("FAILURES:\n" + "\n".join(f"  - {f}" for f in self.failures))
        if self.warnings:
            parts.append("Warnings:\n" + "\n".join(f"  - {w}" for w in self.warnings))
        return "\n".join(parts)


def check_freshness(frames: dict[str, pd.DataFrame],
                    *, max_days: int = MAX_STALENESS_DAYS) -> tuple[bool, str, dict]:
    """Is the data actually current?

    Stale data is the dangerous case: the pipeline runs, produces picks, reports
    success, and the picks are based on last week's prices. Nothing downstream
    would notice.
    """
    if not frames:
        return False, "No price data at all.", {}

    latest = []
    for t, df in frames.items():
        if df is None or df.empty:
            continue
        try:
            latest.append(pd.Timestamp(df.index[-1]).normalize())
        except Exception:                                      # noqa: BLE001
            continue

    if not latest:
        return False, "No usable timestamps in any frame.", {}

    newest = max(latest)
    today = pd.Timestamp(dt.date.today())
    age = (today - newest).days

    # Weekends are expected, so measure business days rather than calendar days
    bdays = int(np.busday_count(newest.date(), today.date()))

    detail = {
        "newest_bar": newest.date().isoformat(),
        "calendar_days_old": age,
        "business_days_old": bdays,
        "frames_checked": len(latest),
    }

    if age > max_days:
        return False, (
            f"Data is stale: newest bar is {newest.date()} "
            f"({age} calendar days, {bdays} business days old). "
            f"Threshold is {max_days} calendar days. Refusing to produce picks "
            "from old prices."
        ), detail
    if bdays > 2:
        return True, (
            f"Data is {bdays} business days old (newest {newest.date()}). "
            "Within tolerance but worth noting — check for an NSE holiday."
        ), detail
    return True, f"Data current: newest bar {newest.date()}.", detail


def check_coverage(frames: dict[str, pd.DataFrame], requested: int,
                   *, min_fraction: float = MIN_TICKER_COVERAGE) -> tuple[bool, str, dict]:
    """Did enough tickers return usable data?

    Partial failure is easy to miss: 30 of 150 tickers returning data still
    produces a bucket, just a badly informed one.
    """
    got = len(frames)
    frac = got / requested if requested else 0.0
    detail = {"requested": requested, "returned": got, "fraction": round(frac, 3)}

    if frac < min_fraction:
        return False, (
            f"Only {got} of {requested} tickers returned data ({frac:.0%}). "
            f"Minimum is {min_fraction:.0%}. Likely a rate limit or an API change — "
            "a bucket built on this would be poorly informed."
        ), detail
    if frac < 0.85:
        return True, f"Coverage {frac:.0%} ({got}/{requested}) — lower than usual.", detail
    return True, f"Coverage {frac:.0%} ({got}/{requested}).", detail


def check_price_sanity(frames: dict[str, pd.DataFrame]) -> tuple[bool, str, dict]:
    """Do the numbers look like real market data?"""
    issues, checked = [], 0
    for t, df in frames.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        checked += 1
        c = df["Close"]
        if (c <= 0).any():
            issues.append(f"{t}: non-positive prices")
        if c.isna().all():
            issues.append(f"{t}: all prices NaN")
        else:
            # NEVER compare floats with ==. The standard deviation of twenty
            # identical values is ~1e-14, not 0.0, so an equality test silently
            # never fires. Compare against the price level instead.
            tail = c.tail(20)
            level = float(tail.mean())
            if level > 0 and float(tail.std()) / level < 1e-9:
                issues.append(f"{t}: price frozen for 20 sessions")
        if "Volume" in df.columns and (df["Volume"].tail(20) <= 0).all():
            issues.append(f"{t}: zero volume for 20 sessions")

    detail = {"checked": checked, "issues": len(issues), "examples": issues[:5]}
    if len(issues) > checked * 0.10:
        return False, (
            f"{len(issues)} of {checked} tickers show implausible data "
            f"({len(issues)/max(checked,1):.0%}). Examples: {'; '.join(issues[:3])}"
        ), detail
    if issues:
        return True, f"{len(issues)} minor data issues in {checked} tickers.", detail
    return True, f"All {checked} tickers look sane.", detail


def check_log_continuity(log_path: Path = Path("forward_log.csv"),
                         *, max_gap_days: int = MAX_LOG_GAP_DAYS) -> tuple[bool, str, dict]:
    """Is the forward log still growing?

    A stalled log is the quietest failure of all — everything appears to work
    while the evidence being accumulated silently stops accumulating.
    """
    if not log_path.exists():
        return True, "No forward log yet (expected before the first weekly run).", {}

    try:
        log = pd.read_csv(log_path)
    except Exception as exc:                                   # noqa: BLE001
        return False, f"Forward log unreadable: {type(exc).__name__}: {exc}", {}

    if log.empty or "snapshot_date" not in log.columns:
        return True, "Forward log is empty.", {"rows": 0}

    dates = pd.to_datetime(log["snapshot_date"], errors="coerce").dropna()
    if dates.empty:
        return False, "Forward log has no valid snapshot dates.", {"rows": len(log)}

    last = dates.max().date()
    gap = (dt.date.today() - last).days
    detail = {
        "rows": len(log),
        "snapshots": int(dates.dt.date.nunique()),
        "last_snapshot": last.isoformat(),
        "days_since": gap,
    }

    if gap > max_gap_days:
        return False, (
            f"Forward log has not grown in {gap} days (last snapshot {last}). "
            f"Expected weekly. The evidence being accumulated has stalled — "
            "check whether the weekly workflow is still running."
        ), detail
    return True, (f"Forward log healthy: {detail['snapshots']} snapshots, "
                  f"last {last} ({gap} days ago)."), detail


def run_all(frames: dict[str, pd.DataFrame] | None = None,
            requested: int = 0,
            *, check_log: bool = True) -> HealthResult:
    """Run every applicable check. Returns passed=False if any critical one fails."""
    res = HealthResult(passed=True)

    if frames is not None:
        for name, fn in (
            ("freshness", lambda: check_freshness(frames)),
            ("coverage", lambda: check_coverage(frames, requested or len(frames))),
            ("price_sanity", lambda: check_price_sanity(frames)),
        ):
            ok, msg, detail = fn()
            res.checks[name] = ok
            res.details[name] = detail
            if not ok:
                res.passed = False
                res.failures.append(f"[{name}] {msg}")
            elif "worth noting" in msg or "lower than usual" in msg or "minor" in msg:
                res.warnings.append(f"[{name}] {msg}")

    if check_log:
        ok, msg, detail = check_log_continuity()
        res.checks["log_continuity"] = ok
        res.details["log_continuity"] = detail
        if not ok:
            # A stalled log is serious but should not abort today's run —
            # today's run is how it starts growing again.
            res.warnings.append(f"[log_continuity] {msg}")

    return res


def heartbeat() -> dict:
    """State of the system, for a periodic 'still alive' report."""
    out: dict = {"checked_at": dt.datetime.now().isoformat(timespec="seconds")}

    for label, path in (("forward_log", Path("forward_log.csv")),
                        ("daily_observations", Path("daily_observations.csv")),
                        ("decay_history", Path("decay_history.csv"))):
        if not path.exists():
            out[label] = {"exists": False}
            continue
        try:
            df = pd.read_csv(path)
            info = {"exists": True, "rows": len(df)}
            for col in ("snapshot_date", "obs_date", "measured_on"):
                if col in df.columns and not df.empty:
                    d = pd.to_datetime(df[col], errors="coerce").dropna()
                    if not d.empty:
                        info["last_entry"] = d.max().date().isoformat()
                        info["days_since"] = (dt.date.today() - d.max().date()).days
                    break
            out[label] = info
        except Exception as exc:                               # noqa: BLE001
            out[label] = {"exists": True, "error": f"{type(exc).__name__}"}

    return out
