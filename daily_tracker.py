"""Daily observation tracker — a diary, not evidence.

Two logs, deliberately separate
-------------------------------
**forward_log.csv** — weekly, non-overlapping snapshots. This is the statistical
evidence. Its windows must not overlap, because overlapping observations are
correlated and inflate significance without adding information.

**daily_observations.csv** — every trading day. This is a diary: what the bucket
said, what the stocks then did, day by day. Useful for watching behaviour,
building intuition, and spotting operational problems early.

Why they must stay apart
------------------------
If daily picks fed the forward log, consecutive snapshots would share most of
their names and nearly all of their holding window. Forty such observations look
like forty independent trials and are closer to eight. The t-statistic roughly
doubles for no reason. That is how a mediocre strategy comes to look significant,
and it is worth going to some trouble to avoid.

So: read the daily log for texture. Read the weekly log for truth.

What gets recorded
------------------
Each run appends the day's bucket — ticker, rank, score, momentum, regime, the
reference close it was picked on. Separately, every stock ever picked has its
daily OHLC tracked forward, so you accumulate a full price history of the
strategy's choices over the observation period.

Timing
------
Intended to run **before** market open. The bucket is therefore computed from
the previous session's close, which is the correct basis for deciding what to
trade today. Each run also backfills OHLC for all previously tracked names up to
the last completed session.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OBS_PATH = Path("daily_observations.csv")
PRICE_PATH = Path("daily_prices.csv")
EXCEL_PATH = Path("reports/daily_tracker.xlsx")

OBS_COLUMNS = [
    "obs_date", "ticker", "rank", "tier", "score", "momentum_pct",
    "regime", "ref_close",
    # Stop and targets, carried through from the bucket. Previously the stop
    # existed only in the app's Detail tab and never reached any output, so a
    # recorded pick had no exit level attached to it.
    "stop_loss", "stop_pct", "target_1r", "target_2r", "risk_per_share",
    "sector", "cost_viable", "notes",
]

PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]


@dataclass
class TrackerResult:
    observations: pd.DataFrame
    prices: pd.DataFrame
    new_observations: int
    new_price_rows: int
    tracked_tickers: int
    excel_path: Path | None = None


def _empty(cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=cols)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = pd.read_csv(OBS_PATH) if OBS_PATH.exists() else _empty(OBS_COLUMNS)
    px = pd.read_csv(PRICE_PATH) if PRICE_PATH.exists() else _empty(PRICE_COLUMNS)
    for c in OBS_COLUMNS:
        if c not in obs.columns:
            obs[c] = np.nan
    for c in PRICE_COLUMNS:
        if c not in px.columns:
            px[c] = np.nan
    return obs[OBS_COLUMNS], px[PRICE_COLUMNS]


def save(obs: pd.DataFrame, px: pd.DataFrame) -> None:
    obs.to_csv(OBS_PATH, index=False)
    px.to_csv(PRICE_PATH, index=False)


def record_bucket(
    obs: pd.DataFrame,
    picks: pd.DataFrame,
    *,
    regime_state: str,
    obs_date: dt.date | None = None,
    notes: str = "",
) -> tuple[pd.DataFrame, int]:
    """Append today's bucket. Idempotent — re-running on the same date is a no-op."""
    d = str(obs_date or dt.date.today())
    if not obs.empty and (obs["obs_date"].astype(str) == d).any():
        return obs, 0
    if picks is None or picks.empty:
        # An empty bucket is itself an observation worth keeping
        row = pd.DataFrame([{
            "obs_date": d, "ticker": None, "rank": None, "tier": None,
            "score": None, "momentum_pct": None, "regime": regime_state,
            "ref_close": None, "stop_loss": None, "stop_pct": None,
            "target_1r": None, "target_2r": None, "risk_per_share": None,
            "sector": None, "cost_viable": None,
            "notes": notes or "no picks qualified",
        }])
        return pd.concat([obs, row], ignore_index=True), 1

    rows = []
    for _, r in picks.iterrows():
        close = r.get("Close")
        stop = r.get("Stop")
        risk = (round(float(close) - float(stop), 2)
                if close is not None and stop is not None
                and pd.notna(close) and pd.notna(stop) else None)
        rows.append({
            "obs_date": d,
            "ticker": r.get("Ticker"),
            "rank": r.get("Rank"),
            "tier": r.get("Tier"),
            "score": r.get("Score"),
            "momentum_pct": r.get("Momentum"),
            "regime": regime_state,
            "ref_close": close,
            "stop_loss": stop,
            "stop_pct": r.get("Stop_pct"),
            "target_1r": r.get("Target_1R"),
            "target_2r": r.get("Target_2R"),
            "risk_per_share": risk,
            "sector": r.get("Sector"),
            "cost_viable": r.get("Cost_viable"),
            "notes": notes,
        })
    return pd.concat([obs, pd.DataFrame(rows)], ignore_index=True), len(rows)


def update_prices(px: pd.DataFrame, frames: dict[str, pd.DataFrame],
                  tickers: set[str], *, lookback: int = 10) -> tuple[pd.DataFrame, int]:
    """Backfill daily OHLC for every tracked ticker. Duplicates are dropped."""
    if not tickers:
        return px, 0

    rows = []
    for t in tickers:
        # NOTE: `a or b` raises on DataFrames — pandas truthiness is ambiguous.
        # This exact bug appeared in forward_log.py first; check membership
        # explicitly instead.
        df = frames.get(t)
        if df is None:
            df = frames.get(f"{t}.NS")
        if df is None or df.empty:
            continue
        tail = df.tail(lookback)
        for idx, r in tail.iterrows():
            rows.append({
                "date": pd.Timestamp(idx).date().isoformat(),
                "ticker": t.replace(".NS", ""),
                "open": round(float(r.get("Open", np.nan)), 2),
                "high": round(float(r.get("High", np.nan)), 2),
                "low": round(float(r.get("Low", np.nan)), 2),
                "close": round(float(r.get("Close", np.nan)), 2),
                "volume": float(r.get("Volume", np.nan)),
            })
    if not rows:
        return px, 0

    new = pd.DataFrame(rows)
    before = len(px)
    combined = pd.concat([px, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    return combined, len(combined) - before


def performance(obs: pd.DataFrame, px: pd.DataFrame,
                horizons: tuple[int, ...] = (1, 5, 10, 15, 20)) -> pd.DataFrame:
    """Forward returns from each observation date, at several horizons.

    NOT statistical evidence. Consecutive daily observations share most of their
    names and nearly all of their window, so these returns are heavily
    correlated. Read them for texture — whether picks tend to move at all, how
    quickly, how much they swing — not for significance.
    """
    if obs.empty or px.empty:
        return pd.DataFrame()

    o = obs.dropna(subset=["ticker"]).copy()
    if o.empty:
        return pd.DataFrame()

    p = px.copy()
    p["date"] = pd.to_datetime(p["date"], errors="coerce")
    p = p.dropna(subset=["date"]).sort_values(["ticker", "date"])

    rows = []
    for _, r in o.iterrows():
        tkr = str(r["ticker"])
        try:
            d0 = pd.Timestamp(str(r["obs_date"]))
        except Exception:                                      # noqa: BLE001
            continue
        series = p[p["ticker"] == tkr]
        if series.empty:
            continue
        fwd = series[series["date"] >= d0]
        if fwd.empty:
            continue

        entry = float(fwd.iloc[0]["close"])
        if entry <= 0:
            continue

        row = {
            "obs_date": r["obs_date"], "ticker": tkr, "rank": r["rank"],
            "score": r["score"], "regime": r["regime"], "entry_close": entry,
        }
        for h in horizons:
            row[f"ret_{h}d_pct"] = (
                round((float(fwd.iloc[h]["close"]) / entry - 1) * 100, 2)
                if len(fwd) > h else np.nan
            )
        row["max_gain_pct"] = round((float(fwd["high"].max()) / entry - 1) * 100, 2)
        row["max_drawdown_pct"] = round((float(fwd["low"].min()) / entry - 1) * 100, 2)
        row["sessions_tracked"] = len(fwd)

        # Did the stop actually get hit, and did the targets get reached?
        # Recording the stop is only useful if you can later see whether it
        # would have fired.
        stop = r.get("stop_loss")
        if stop is not None and pd.notna(stop) and float(stop) > 0:
            low = float(fwd["low"].min())
            row["stop_loss"] = float(stop)
            row["stop_hit"] = bool(low <= float(stop))
            if row["stop_hit"]:
                below = fwd[fwd["low"] <= float(stop)]
                row["sessions_to_stop"] = (int(fwd.index.get_loc(below.index[0])) + 1
                                           if len(below) else None)
        for tgt, col in (("target_1r", "hit_1r"), ("target_2r", "hit_2r")):
            v = r.get(tgt)
            if v is not None and pd.notna(v) and float(v) > 0:
                row[col] = bool(float(fwd["high"].max()) >= float(v))
        rows.append(row)

    return pd.DataFrame(rows)


def summarise(perf: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the diary. Descriptive only."""
    if perf is None or perf.empty:
        return pd.DataFrame()

    ret_cols = [c for c in perf.columns if c.startswith("ret_")]
    rows = []
    for c in ret_cols:
        s = pd.to_numeric(perf[c], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append({
            "horizon": c.replace("ret_", "").replace("_pct", ""),
            "observations": len(s),
            "mean_pct": round(float(s.mean()), 2),
            "median_pct": round(float(s.median()), 2),
            "hit_rate_pct": round(float((s > 0).mean()) * 100, 1),
            "best_pct": round(float(s.max()), 2),
            "worst_pct": round(float(s.min()), 2),
            "std_pct": round(float(s.std(ddof=1)), 2) if len(s) > 1 else None,
        })
    return pd.DataFrame(rows)


def write_excel(obs: pd.DataFrame, px: pd.DataFrame, perf: pd.DataFrame,
                summary: pd.DataFrame, path: Path = EXCEL_PATH) -> Path | None:
    """Write the cumulative workbook. Returns the path, or None if unavailable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("  Excel export SKIPPED — openpyxl is not installed.")
        print("     Fix:  pip install openpyxl")
        print("     CSVs were still written; only the workbook is missing.")
        return None

    try:
        with pd.ExcelWriter(path, engine="openpyxl") as xl:
            # Read-me first, so the caveat cannot be missed
            pd.DataFrame({
                "SwingScope daily tracker": [
                    "",
                    "This workbook is a DIARY, not statistical evidence.",
                    "",
                    "Daily observations overlap heavily — consecutive days share most",
                    "of their picks and nearly all of their holding window. Returns here",
                    "are correlated, so hit rates and averages CANNOT be treated as",
                    "significance. Forty daily observations behave closer to eight",
                    "independent ones.",
                    "",
                    "For statistical evidence use forward_log.csv, which takes weekly",
                    "non-overlapping snapshots and is what the month-end report reads.",
                    "",
                    "Use this workbook to watch behaviour: do picks move at all, how",
                    "fast, how far do they draw down before working, does the regime",
                    "gate fire when it should.",
                    "",
                    f"Generated {dt.datetime.now():%d %b %Y %H:%M}",
                    "",
                    "Research output only. Not investment advice.",
                ]
            }).to_excel(xl, sheet_name="Read me", index=False)

            if not summary.empty:
                summary.to_excel(xl, sheet_name="Summary", index=False)
            if not obs.empty:
                obs.to_excel(xl, sheet_name="Daily picks", index=False)
                # A focused sheet for the levels you actually act on
                lvl = [c for c in ("obs_date", "ticker", "rank", "ref_close",
                                   "stop_loss", "stop_pct", "risk_per_share",
                                   "target_1r", "target_2r") if c in obs.columns]
                if lvl:
                    obs[lvl].dropna(subset=["ticker"]).to_excel(
                        xl, sheet_name="Levels", index=False)
            if not perf.empty:
                perf.sort_values("obs_date", ascending=False).to_excel(
                    xl, sheet_name="Performance", index=False)
            if not px.empty:
                px.to_excel(xl, sheet_name="Price history", index=False)

            # Per-day pick counts, useful for spotting gaps in the record
            if not obs.empty:
                daily = (obs.groupby("obs_date")
                         .agg(picks=("ticker", lambda s: s.notna().sum()),
                              regime=("regime", "first"))
                         .reset_index())
                daily.to_excel(xl, sheet_name="Coverage", index=False)
        return path
    except Exception as exc:                                   # noqa: BLE001
        print(f"  Excel export failed ({type(exc).__name__}: {exc}) — CSVs still written")
        return None


def run(picks: pd.DataFrame, frames: dict[str, pd.DataFrame],
        *, regime_state: str, obs_date: dt.date | None = None,
        notes: str = "") -> TrackerResult:
    """One full tracker cycle: record, backfill prices, compute, export."""
    obs, px = load()
    obs, n_new = record_bucket(obs, picks, regime_state=regime_state,
                               obs_date=obs_date, notes=notes)

    tracked = set(obs["ticker"].dropna().astype(str))
    px, n_px = update_prices(px, frames, tracked)

    save(obs, px)
    perf = performance(obs, px)
    summary = summarise(perf)
    xl = write_excel(obs, px, perf, summary)

    return TrackerResult(obs, px, n_new, n_px, len(tracked), xl)
