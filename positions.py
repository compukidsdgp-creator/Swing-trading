"""Position monitoring — the missing half.

The gap this closes
-------------------
Every competitor has this and SwingScope did not. The system generated picks,
computed stops and targets, recorded them, and then lost interest. No open
position view, no running P&L against the levels already calculated, no alert
when a holding reached its stop.

A screener without a portfolio is half a system. This is the other half.

What it does
------------
  * Tracks open positions with entry, stop, targets and current price
  * Computes P&L in rupees, percent and R-multiples
  * Detects stop hits, target hits and horizon expiry
  * Trails the stop upward once a position has earned it
  * Produces alerts worth sending, and stays quiet otherwise

Design principle
----------------
It reports. It never places or cancels an order.

The alert says "ADANIENSOL has touched its stop at ₹1,576" — not "sell". That
distinction matters while there is no forward evidence: the judgement between a
computed level and an actual instruction is doing real work, and automating it
would remove the thing currently protecting the account.

Alert discipline
----------------
An alert channel that fires constantly gets muted, and a muted channel is worse
than none. So:

  * Each event alerts once. State is persisted.
  * Routine daily movement is not an event.
  * Approaching a level is worth one warning; crossing it is worth one alert.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

POSITIONS_PATH = Path("open_positions.csv")
ALERTS_PATH = Path("alerts_sent.csv")

POSITION_COLUMNS = [
    "position_id", "ticker", "tier", "entry_date", "entry_price", "qty",
    "initial_stop", "current_stop", "target_1r", "target_2r",
    "horizon_days", "target_exit_date", "status", "exit_date", "exit_price",
    "exit_reason", "highest_close", "notes",
]

# Warn when price comes within this fraction of the remaining distance to a
# level. Approaching a stop is worth knowing before it triggers.
PROXIMITY_WARN = 0.20

# Trail the stop only once the position is up by this many risk units. Trailing
# from entry stops you out on ordinary noise before anything is established.
TRAIL_ACTIVATE_R = 1.0


@dataclass
class PositionState:
    ticker: str
    entry_price: float
    current_price: float
    qty: int
    stop: float
    days_held: int
    days_remaining: int
    pnl_rupees: float
    pnl_pct: float
    pnl_r: float
    distance_to_stop_pct: float
    status: str                    # open | stopped | target | expired
    events: list[str] = field(default_factory=list)


@dataclass
class MonitorResult:
    positions: pd.DataFrame
    alerts: list[dict]
    summary: dict
    closed: list[str] = field(default_factory=list)

    @property
    def has_alerts(self) -> bool:
        return len(self.alerts) > 0


def empty_positions() -> pd.DataFrame:
    return pd.DataFrame(columns=POSITION_COLUMNS)


def load_positions(path: Path = POSITIONS_PATH) -> pd.DataFrame:
    if not path.exists():
        return empty_positions()
    try:
        df = pd.read_csv(path)
    except Exception:                                          # noqa: BLE001
        return empty_positions()
    for c in POSITION_COLUMNS:
        if c not in df.columns:
            df[c] = (pd.Series([None] * len(df), dtype=object)
                     if c in ("status", "exit_reason", "notes") else np.nan)
    return df[POSITION_COLUMNS]


def save_positions(df: pd.DataFrame, path: Path = POSITIONS_PATH) -> None:
    df.to_csv(path, index=False)


def open_position(
    positions: pd.DataFrame,
    *,
    ticker: str,
    entry_price: float,
    qty: int,
    stop: float,
    target_1r: float | None = None,
    target_2r: float | None = None,
    tier: str = "mid",
    horizon_days: int = 30,
    entry_date: dt.date | None = None,
    notes: str = "",
) -> tuple[pd.DataFrame, str]:
    """Record a new position. Returns (positions, position_id)."""
    d = entry_date or dt.date.today()
    pid = f"{ticker}_{d:%Y%m%d}"

    if not positions.empty:
        live = positions[(positions["ticker"] == ticker)
                         & (positions["status"] == "open")]
        if not live.empty:
            return positions, ""      # already holding it

    risk = entry_price - stop
    row = {
        "position_id": pid, "ticker": ticker, "tier": tier,
        "entry_date": d.isoformat(), "entry_price": round(entry_price, 2),
        "qty": int(qty), "initial_stop": round(stop, 2),
        "current_stop": round(stop, 2),
        "target_1r": round(target_1r if target_1r else entry_price + risk, 2),
        "target_2r": round(target_2r if target_2r else entry_price + 2 * risk, 2),
        "horizon_days": horizon_days,
        "target_exit_date": (d + dt.timedelta(days=int(horizon_days * 1.45))).isoformat(),
        "status": "open", "exit_date": None, "exit_price": np.nan,
        "exit_reason": None, "highest_close": round(entry_price, 2),
        "notes": notes,
    }
    return pd.concat([positions, pd.DataFrame([row])], ignore_index=True), pid


def open_from_bucket(positions: pd.DataFrame, picks: pd.DataFrame,
                     capital: float, *, risk_pct: float = 1.0,
                     horizon_days: int = 30) -> tuple[pd.DataFrame, list[str]]:
    """Open positions from a bucket, sized by risk.

    Quantity is derived from the stop distance, not chosen. A wider stop means
    fewer shares, so every position risks the same rupee amount.
    """
    if picks is None or picks.empty:
        return positions, []
    if not {"Ticker", "Close", "Stop"}.issubset(picks.columns):
        return positions, []

    opened = []
    risk_budget = capital * risk_pct / 100.0
    for _, r in picks.iterrows():
        entry = float(r["Close"])
        stop = float(r["Stop"])
        per_share = entry - stop
        if per_share <= 0:
            continue
        qty = int(risk_budget / per_share)
        if qty < 1:
            continue
        positions, pid = open_position(
            positions, ticker=str(r["Ticker"]), entry_price=entry, qty=qty,
            stop=stop, target_1r=r.get("Target_1R"), target_2r=r.get("Target_2R"),
            tier=str(r.get("Tier", "mid")), horizon_days=horizon_days,
        )
        if pid:
            opened.append(pid)
    return positions, opened


def _load_alerts(path: Path = ALERTS_PATH) -> set:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
        return set(zip(df["position_id"].astype(str), df["event"].astype(str)))
    except Exception:                                          # noqa: BLE001
        return set()


def _record_alert(pid: str, event: str, path: Path = ALERTS_PATH) -> None:
    row = pd.DataFrame([{"position_id": pid, "event": event,
                         "sent_at": dt.datetime.now().isoformat(timespec="seconds")}])
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([existing, row], ignore_index=True).to_csv(path, index=False)


def monitor(
    positions: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    *,
    trail_stops: bool = True,
    atr_mult: float | None = None,
    as_of: dt.date | None = None,
) -> MonitorResult:
    """Update every open position and surface anything worth acting on.

    Args:
        prices: {ticker: OHLCV frame} for open positions.
        trail_stops: ratchet the stop up once the position is 1R ahead.
        atr_mult: override the trailing distance; otherwise tier-derived.
    """
    import tiers as tr

    today = as_of or dt.date.today()
    if positions.empty:
        return MonitorResult(positions, [], {"open": 0, "note": "No positions."})

    df = positions.copy()
    already = _load_alerts()
    alerts, closed, rows = [], [], []

    for i, p in df.iterrows():
        if p["status"] != "open":
            continue

        tkr = str(p["ticker"])
        # Never `a or b` with DataFrames — pandas truthiness is ambiguous.
        # This exact bug has appeared three times in this codebase.
        frame = prices.get(tkr)
        if frame is None:
            frame = prices.get(f"{tkr}.NS")
        if frame is None or frame.empty or "Close" not in frame.columns:
            continue

        entry = float(p["entry_price"])
        stop = float(p["current_stop"])
        qty = int(p["qty"])
        price = float(frame["Close"].iloc[-1])
        high = float(frame["High"].iloc[-1]) if "High" in frame.columns else price
        low = float(frame["Low"].iloc[-1]) if "Low" in frame.columns else price

        entry_date = dt.date.fromisoformat(str(p["entry_date"]))
        days_held = (today - entry_date).days
        horizon = int(p["horizon_days"])
        target_exit = dt.date.fromisoformat(str(p["target_exit_date"]))
        days_left = (target_exit - today).days

        risk = entry - float(p["initial_stop"])
        pnl_rs = (price - entry) * qty
        pnl_pct = (price / entry - 1) * 100
        pnl_r = (price - entry) / risk if risk > 0 else 0.0

        highest = max(float(p["highest_close"]), price)
        df.at[i, "highest_close"] = round(highest, 2)

        events = []
        pid = str(p["position_id"])

        # --- Trailing stop ---
        #
        # Only once the position is 1R ahead. Trailing from entry stops you out
        # on ordinary noise before anything has been established.
        if trail_stops and risk > 0 and (highest - entry) / risk >= TRAIL_ACTIVATE_R:
            mult = atr_mult if atr_mult else tr.params(str(p["tier"]))["atr_mult"]
            if "ATR14" in frame.columns:
                atr = float(frame["ATR14"].iloc[-1])
            else:
                rng = (frame["High"] - frame["Low"]).tail(14).mean()
                atr = float(rng) if np.isfinite(rng) else price * 0.02
            candidate = highest - atr * mult
            if candidate > stop:
                df.at[i, "current_stop"] = round(candidate, 2)
                events.append(f"stop trailed up to ₹{candidate:,.0f}")
                stop = candidate

        # --- Exit conditions, most severe first ---
        if low <= stop:
            df.at[i, "status"] = "stopped"
            df.at[i, "exit_date"] = today.isoformat()
            df.at[i, "exit_price"] = round(stop, 2)
            df.at[i, "exit_reason"] = "stop hit"
            closed.append(pid)
            if (pid, "stop") not in already:
                alerts.append({
                    "position_id": pid, "ticker": tkr, "event": "stop",
                    "severity": "high",
                    "message": (f"🔴 {tkr} touched its stop at ₹{stop:,.0f}. "
                                f"Held {days_held} days. "
                                f"P&L {(stop-entry)*qty:+,.0f} "
                                f"({(stop/entry-1)*100:+.1f}%, "
                                f"{(stop-entry)/risk if risk>0 else 0:+.2f}R)."),
                })
                _record_alert(pid, "stop")

        elif days_left <= 0:
            df.at[i, "status"] = "expired"
            df.at[i, "exit_date"] = today.isoformat()
            df.at[i, "exit_price"] = round(price, 2)
            df.at[i, "exit_reason"] = "horizon reached"
            closed.append(pid)
            if (pid, "expiry") not in already:
                alerts.append({
                    "position_id": pid, "ticker": tkr, "event": "expiry",
                    "severity": "medium",
                    "message": (f"⏱ {tkr} has reached its {horizon}-day horizon. "
                                f"P&L {pnl_rs:+,.0f} ({pnl_pct:+.1f}%, {pnl_r:+.2f}R). "
                                "The holding period the signal was validated for "
                                "has elapsed."),
                })
                _record_alert(pid, "expiry")

        else:
            # --- Targets, informational rather than instructions ---
            t2 = float(p["target_2r"]) if pd.notna(p["target_2r"]) else None
            t1 = float(p["target_1r"]) if pd.notna(p["target_1r"]) else None

            # Decide WHICH target was reached first, THEN check whether it has
            # already been alerted. An elif chain gated on "already sent" falls
            # through: once 2R is marked sent, the 1R branch fires on the next
            # run and the alert repeats.
            hit_2r = bool(t2 and high >= t2)
            hit_1r = bool(t1 and high >= t1)

            if hit_2r:
                if (pid, "target_2r") not in already:
                    alerts.append({
                        "position_id": pid, "ticker": tkr, "event": "target_2r",
                        "severity": "low",
                        "message": (f"🎯 {tkr} reached 2R at ₹{t2:,.0f} "
                                    f"({pnl_pct:+.1f}%). Stop now ₹{stop:,.0f}."),
                    })
                    _record_alert(pid, "target_2r")
                    # Passing 2R implies 1R; mark it so it never fires later.
                    if (pid, "target_1r") not in already:
                        _record_alert(pid, "target_1r")
            elif hit_1r:
                if (pid, "target_1r") not in already:
                    alerts.append({
                        "position_id": pid, "ticker": tkr, "event": "target_1r",
                        "severity": "low",
                        "message": (f"🎯 {tkr} reached 1R at ₹{t1:,.0f} "
                                    f"({pnl_pct:+.1f}%). Trailing stop is now "
                                    "active."),
                    })
                    _record_alert(pid, "target_1r")

            # --- Proximity warning: one per position, before it triggers ---
            to_stop = (price - stop) / (entry - float(p["initial_stop"])) \
                if entry > float(p["initial_stop"]) else 1.0
            if 0 < to_stop <= PROXIMITY_WARN and (pid, "near_stop") not in already:
                alerts.append({
                    "position_id": pid, "ticker": tkr, "event": "near_stop",
                    "severity": "medium",
                    "message": (f"⚠️ {tkr} is within {to_stop:.0%} of its stop "
                                f"(₹{price:,.0f} vs ₹{stop:,.0f})."),
                })
                _record_alert(pid, "near_stop")

        rows.append({
            "ticker": tkr, "entry": entry, "price": round(price, 2),
            "stop": round(stop, 2), "qty": qty,
            "pnl_rs": round(pnl_rs, 0), "pnl_pct": round(pnl_pct, 2),
            "pnl_r": round(pnl_r, 2), "days_held": days_held,
            "days_left": max(0, days_left),
            "status": df.at[i, "status"],
        })

    live = pd.DataFrame(rows)
    open_now = live[live["status"] == "open"] if not live.empty else pd.DataFrame()

    summary = {
        "open": len(open_now),
        "closed_today": len(closed),
        "total_pnl_rs": round(float(open_now["pnl_rs"].sum()), 0) if len(open_now) else 0.0,
        "total_pnl_r": round(float(open_now["pnl_r"].sum()), 2) if len(open_now) else 0.0,
        "winners": int((open_now["pnl_r"] > 0).sum()) if len(open_now) else 0,
        "losers": int((open_now["pnl_r"] <= 0).sum()) if len(open_now) else 0,
        "alerts": len(alerts),
        "detail": live,
    }
    return MonitorResult(df, alerts, summary, closed)


def alert_message(result: MonitorResult) -> str:
    """Format alerts for Telegram. Returns empty string when nothing is due."""
    if not result.alerts:
        return ""

    order = {"high": 0, "medium": 1, "low": 2}
    alerts = sorted(result.alerts, key=lambda a: order.get(a["severity"], 3))

    lines = ["*SwingScope — position alerts*", ""]
    lines += [a["message"] for a in alerts]

    s = result.summary
    if s.get("open"):
        lines += ["", f"_{s['open']} open · {s['total_pnl_r']:+.2f}R total · "
                      f"{s['winners']}W / {s['losers']}L_"]
    lines += ["", "Informational. No orders were placed."]
    return "\n".join(lines)


def closed_performance(positions: pd.DataFrame) -> dict:
    """Expectancy on closed positions — the number that decides everything."""
    if positions.empty:
        return {"error": "No positions."}

    done = positions[positions["status"].isin(["stopped", "expired", "closed"])].copy()
    if done.empty:
        return {"error": "No closed positions yet."}

    for c in ("entry_price", "exit_price", "initial_stop", "qty"):
        done[c] = pd.to_numeric(done[c], errors="coerce")
    done = done.dropna(subset=["entry_price", "exit_price", "initial_stop"])
    if done.empty:
        return {"error": "No usable closed positions."}

    risk = done["entry_price"] - done["initial_stop"]
    done = done[risk > 0]
    if done.empty:
        return {"error": "No positions with a valid stop."}

    r_mult = (done["exit_price"] - done["entry_price"]) / (
        done["entry_price"] - done["initial_stop"])
    pnl = (done["exit_price"] - done["entry_price"]) * done["qty"]

    wins = r_mult[r_mult > 0]
    losses = r_mult[r_mult <= 0]

    return {
        "closed": len(done),
        "win_rate_pct": round(float((r_mult > 0).mean()) * 100, 1),
        "expectancy_r": round(float(r_mult.mean()), 3),
        "avg_win_r": round(float(wins.mean()), 2) if len(wins) else None,
        "avg_loss_r": round(float(losses.mean()), 2) if len(losses) else None,
        "total_pnl_rs": round(float(pnl.sum()), 0),
        "best_r": round(float(r_mult.max()), 2),
        "worst_r": round(float(r_mult.min()), 2),
        "stopped_out_pct": round(float(
            (done["exit_reason"] == "stop hit").mean()) * 100, 1),
        "verdict": (
            "Positive expectancy — the strategy makes money over time."
            if r_mult.mean() > 0 else
            "NEGATIVE EXPECTANCY. This loses money over time regardless of win "
            "rate. No position sizing fixes it — the rules need changing."
        ),
        "caveat": ("Expectancy needs 30+ closed trades to mean much. Below that "
                   "it is mostly ordering luck."
                   if len(done) < 30 else None),
    }
