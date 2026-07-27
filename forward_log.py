"""Forward paper-trading log.

The one form of evidence that cannot be overfitted.

A backtest can be tuned until it looks good — change a threshold, re-run, keep
the version that flatters you. A forward log cannot: the picks are committed
before the outcome exists. Thirty days of honest forward records is worth more
than any amount of historical curve-fitting.

Workflow
--------
1. Each week, snapshot the screener's top N picks with their scores.
2. Do nothing for 15-20 trading days.
3. Re-run scoring; the log fills in actual returns automatically.
4. After ~8 weeks, compare forward IC against backtest IC. If forward is
   dramatically worse, the backtest was overfitted — which is exactly what
   you want to discover on paper rather than with capital.

Persistence
-----------
Streamlit Cloud has an ephemeral filesystem, so the log lives in session state
and is exported/imported as CSV. Download after every session. For permanent
storage, wire `load`/`save` to Google Sheets via gspread or to Supabase — the
interface is deliberately just two functions.
"""

from __future__ import annotations

import datetime as dt
import io

import numpy as np
import pandas as pd

COLUMNS = [
    "snapshot_date", "ticker", "tier", "score", "rank", "regime",
    "price_at_pick", "horizon_days", "target_eval_date",
    "price_at_eval", "fwd_return_pct", "bench_return_pct",
    "excess_return_pct", "status", "notes",
    # Added to make evaluation-timing drift visible rather than silent.
    "holding_days_actual", "eval_method",
]


def empty_log() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def record_snapshot(
    log: pd.DataFrame,
    picks: pd.DataFrame,
    *,
    regime_state: str,
    horizon: int = 15,
    top_n: int = 10,
    snapshot_date: dt.date | None = None,
) -> pd.DataFrame:
    """Commit this week's top picks to the log, before outcomes are known.

    Args:
        picks: the screener's filtered result frame. Needs Ticker, Score,
               Close and (optionally) Tier columns.
    """
    if picks is None or picks.empty:
        return log

    d = snapshot_date or dt.date.today()
    if not log.empty and (log["snapshot_date"] == str(d)).any():
        # Already snapshotted today — don't double-record
        return log

    rows = []
    for rank, (_, r) in enumerate(picks.head(top_n).iterrows(), start=1):
        rows.append({
            "snapshot_date": str(d),
            "ticker": r.get("Ticker", ""),
            "tier": r.get("Tier", "unknown"),
            "score": int(r.get("Score", 0)),
            "rank": rank,
            "regime": regime_state,
            "price_at_pick": round(float(r.get("Close", 0)), 2),
            "horizon_days": horizon,
            "target_eval_date": str(d + dt.timedelta(days=int(horizon * 1.45))),
            "price_at_eval": np.nan,
            "fwd_return_pct": np.nan,
            "bench_return_pct": np.nan,
            "excess_return_pct": np.nan,
            "status": "open",
            "notes": "",
        })

    return pd.concat([log, pd.DataFrame(rows)], ignore_index=True)


def evaluate_open(
    log: pd.DataFrame,
    price_lookup: dict[str, float],
    bench_lookup: dict[str, float] | None = None,
    price_history: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Fill in outcomes for entries whose evaluation date has passed.

    Timing correctness
    ------------------
    An earlier version always used the LATEST price. If the pipeline ran a week
    after a pick matured, a 15-day horizon was recorded using a 22-day return.
    Over months that drift compounds, and the forward log ends up measuring a
    different holding period than the one validated.

    This version prefers the close ON the target date, taken from
    `price_history`. It falls back to the latest price when history is
    unavailable, and records which method was used plus the actual holding
    period — so any remaining drift is visible in the data rather than silent.

    Args:
        price_lookup: {ticker (no .NS): latest price} — the fallback.
        bench_lookup: {'now': price, 'then_<date>': price} for excess return.
        price_history: {ticker: OHLCV frame} — enables exact-date evaluation.
                       Strongly preferred.
    """
    if log.empty:
        return log, 0

    out = log.copy()

    # Dtype care: from_csv backfills absent columns with NaN, giving them
    # float64. Assigning a string into a float64 column raises in pandas 2+.
    # Force the text column to object and the numeric one to float up front.
    if "holding_days_actual" not in out.columns:
        out["holding_days_actual"] = np.nan
    out["holding_days_actual"] = pd.to_numeric(
        out["holding_days_actual"], errors="coerce").astype("float64")

    if "eval_method" not in out.columns:
        out["eval_method"] = pd.Series([None] * len(out), dtype=object, index=out.index)
    else:
        out["eval_method"] = out["eval_method"].astype(object)

    today = dt.date.today()
    filled = 0

    for i, row in out.iterrows():
        if row["status"] != "open":
            continue
        try:
            target = dt.date.fromisoformat(str(row["target_eval_date"]))
            picked = dt.date.fromisoformat(str(row["snapshot_date"]))
        except (ValueError, TypeError):
            continue
        if today < target:
            continue

        ticker = str(row["ticker"])
        p0 = float(row["price_at_pick"])
        if p0 <= 0:
            continue

        px, method, eval_date = None, None, None

        # Preferred: the close on (or immediately after) the target date
        if price_history:
            # NOTE: never use `a or b` with DataFrames — pandas raises on
            # ambiguous truthiness. Check membership explicitly.
            hist = price_history.get(ticker)
            if hist is None:
                hist = price_history.get(f"{ticker}.NS")
            if hist is not None and not hist.empty and "Close" in hist.columns:
                try:
                    idx = pd.DatetimeIndex(hist.index).normalize()
                    tgt = pd.Timestamp(target)
                    on_or_after = idx[idx >= tgt]
                    if len(on_or_after):
                        d = on_or_after[0]
                        # Guard against a stale frame that stops before target
                        if (d - tgt).days <= 7:
                            px = float(hist.loc[idx == d, "Close"].iloc[0])
                            method = "at_target"
                            eval_date = d.date()
                except Exception:                              # noqa: BLE001
                    px = None

        # Fallback: latest price, clearly labelled
        if px is None:
            cand = price_lookup.get(ticker)
            if cand is not None and np.isfinite(cand) and cand > 0:
                px = float(cand)
                method = "latest_price"
                eval_date = today

        if px is None:
            continue

        ret = (px / p0 - 1) * 100
        out.at[i, "price_at_eval"] = round(px, 2)
        out.at[i, "fwd_return_pct"] = round(ret, 2)
        out.at[i, "status"] = "evaluated"
        out.at[i, "eval_method"] = method
        out.at[i, "holding_days_actual"] = (eval_date - picked).days if eval_date else np.nan

        if bench_lookup:
            b_now = bench_lookup.get("now")
            b_then = bench_lookup.get(f"then_{row['snapshot_date']}")
            if b_now and b_then and b_then > 0:
                b_ret = (b_now / b_then - 1) * 100
                out.at[i, "bench_return_pct"] = round(b_ret, 2)
                out.at[i, "excess_return_pct"] = round(ret - b_ret, 2)

        filled += 1

    return out, filled


def analyse(log: pd.DataFrame) -> dict:
    """Summarise forward performance. This is the number that matters."""
    if log.empty:
        return {"error": "log is empty"}

    ev = log[log["status"] == "evaluated"].copy()
    if ev.empty:
        n_open = int((log["status"] == "open").sum())
        return {"error": f"no evaluated picks yet ({n_open} still open)"}

    ev["fwd_return_pct"] = pd.to_numeric(ev["fwd_return_pct"], errors="coerce")
    ev = ev.dropna(subset=["fwd_return_pct"])
    if ev.empty:
        return {"error": "no usable outcomes"}

    # Forward IC: rank correlation of score vs realised return, per snapshot
    ics = []
    for _, grp in ev.groupby("snapshot_date"):
        if len(grp) < 5:
            continue
        rs = grp["score"].rank()
        rr = grp["fwd_return_pct"].rank()
        rs, rr = rs - rs.mean(), rr - rr.mean()
        den = np.sqrt((rs**2).sum() * (rr**2).sum())
        if den > 0:
            ics.append(float((rs * rr).sum() / den))

    mean_ic = float(np.mean(ics)) if ics else np.nan
    t_stat = (mean_ic / (np.std(ics, ddof=1) / np.sqrt(len(ics)))
              if len(ics) > 1 and np.std(ics, ddof=1) > 0 else np.nan)

    exc = pd.to_numeric(ev["excess_return_pct"], errors="coerce").dropna()

    return {
        "evaluated_picks": len(ev),
        "snapshots": ev["snapshot_date"].nunique(),
        "mean_return_pct": round(float(ev["fwd_return_pct"].mean()), 2),
        "median_return_pct": round(float(ev["fwd_return_pct"].median()), 2),
        "hit_rate_pct": round(float((ev["fwd_return_pct"] > 0).mean()) * 100, 1),
        "best_pct": round(float(ev["fwd_return_pct"].max()), 2),
        "worst_pct": round(float(ev["fwd_return_pct"].min()), 2),
        "mean_excess_pct": round(float(exc.mean()), 2) if len(exc) else None,
        "beat_bench_pct": round(float((exc > 0).mean()) * 100, 1) if len(exc) else None,
        "forward_ic": round(mean_ic, 4) if np.isfinite(mean_ic) else None,
        "mean_holding_days": (round(float(pd.to_numeric(
            ev["holding_days_actual"], errors="coerce").dropna().mean()), 1)
            if "holding_days_actual" in ev.columns
            and pd.to_numeric(ev["holding_days_actual"], errors="coerce").notna().any()
            else None),
        "pct_evaluated_at_target": (round(float(
            (ev["eval_method"] == "at_target").mean()) * 100, 1)
            if "eval_method" in ev.columns else None),
        "forward_ic_t": round(float(t_stat), 2) if np.isfinite(t_stat) else None,
        "ic_windows": len(ics),
        "open_picks": int((log["status"] == "open").sum()),
    }


def by_dimension(log: pd.DataFrame, dim: str) -> pd.DataFrame:
    """Break forward results down by tier, regime, or score bucket."""
    ev = log[log["status"] == "evaluated"].copy()
    if ev.empty:
        return pd.DataFrame()
    ev["fwd_return_pct"] = pd.to_numeric(ev["fwd_return_pct"], errors="coerce")
    ev = ev.dropna(subset=["fwd_return_pct"])
    if ev.empty:
        return pd.DataFrame()

    if dim == "score_bucket":
        ev["score_bucket"] = pd.cut(ev["score"], [0, 65, 70, 75, 80, 100],
                                    labels=["<65", "65-70", "70-75", "75-80", "80+"])
        dim = "score_bucket"

    g = ev.groupby(dim, observed=True)["fwd_return_pct"]
    return pd.DataFrame({
        "picks": g.count(),
        "mean_ret_pct": g.mean().round(2),
        "median_ret_pct": g.median().round(2),
        "hit_rate_pct": (ev.groupby(dim, observed=True)["fwd_return_pct"]
                         .apply(lambda s: (s > 0).mean() * 100).round(1)),
    }).reset_index()


def compare_to_backtest(forward_ic: float | None, backtest_ic: float | None) -> str:
    """The moment of truth — does live performance match the historical claim?"""
    if forward_ic is None or backtest_ic is None:
        return "Need both a forward log and a validation run to compare."
    if forward_ic <= 0 and backtest_ic > 0:
        return (
            f"Backtest IC {backtest_ic:+.4f} but forward IC {forward_ic:+.4f}. "
            "The historical result did not survive contact with live data — the "
            "classic signature of overfitting. Do not trade this."
        )
    ratio = forward_ic / backtest_ic if backtest_ic else 0
    if ratio < 0.4:
        return (
            f"Forward IC is {ratio:.0%} of backtest IC. Substantial decay. Some edge "
            "may exist but it is far weaker live than history suggested — normal, "
            "and the reason forward testing exists."
        )
    if ratio < 0.8:
        return (
            f"Forward IC is {ratio:.0%} of backtest IC. Moderate decay, which is "
            "typical and healthy. The signal appears real."
        )
    return (
        f"Forward IC is {ratio:.0%} of backtest IC. Holding up well. Note that few "
        "snapshots still means wide error bars — keep logging."
    )


def to_csv(log: pd.DataFrame) -> bytes:
    return log.to_csv(index=False).encode()


def from_csv(data) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(data) if isinstance(data, bytes) else data)
    except Exception:
        return empty_log()
    for c in COLUMNS:
        if c not in df.columns:
            # Text columns must be object dtype, or later string assignment fails
            df[c] = (pd.Series([None] * len(df), dtype=object, index=df.index)
                     if c in ("eval_method", "status", "notes") else np.nan)
    return df[COLUMNS]
