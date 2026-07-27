"""Paytm Money broker integration — READ-ONLY.

Purpose
-------
Your entire strategy rests on a number that was estimated, not measured: the
round-trip transaction cost per tier (0.25% / 0.60% / 1.50% in tiers.py).
Against an edge of roughly 0.5pp per 15 days, that estimate is the difference
between a viable strategy and a slow bleed.

This module replaces the estimate with your actual fills.

Deliberately read-only
----------------------
No order placement. No modification. No cancellation. Not because the API
lacks them, but because:

  * You have zero forward evidence yet. Automating execution before knowing
    whether the signal works is the wrong order of operations.
  * The judgement step between "here is a ranked list" and "I am buying this"
    is currently the main thing protecting you.
  * A credentials leak on a read-only integration is embarrassing. On a
    trading-enabled one it is expensive.

If you later want execution, that is a separate deliberate decision, not
something that should arrive as a side effect of a cost-analysis tool.

Credential handling
-------------------
Credentials come from environment variables or Streamlit secrets, never from
code. Note that Streamlit Community Cloud is a shared public environment —
running this locally is the safer default, and the module warns when it detects
a cloud deployment.

Setup
-----
1. Sign in at https://developer.paytmmoney.com with your Paytm Money account
   (KYC-ready equity account required).
2. Create an app, note the API key and secret.
3. Install the SDK — note it is NOT on PyPI:

       pip install git+https://github.com/paytmmoney/pyPMClient.git

4. Add to .streamlit/secrets.toml (already in .gitignore):

       [paytm]
       api_key    = "your_key"
       api_secret = "your_secret"

5. Authenticate once per session — access tokens are short-lived by design.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Read-only endpoints this module will touch. Anything not on this list is
# out of scope by design.
ALLOWED_OPERATIONS = {
    "profile", "holdings", "positions", "funds",
    "order_book", "trade_details", "get_ltp",
}


@dataclass
class BrokerStatus:
    connected: bool = False
    message: str = ""
    profile: dict = field(default_factory=dict)


def credentials() -> tuple[str, str, str]:
    """Read credentials from Streamlit secrets, falling back to env vars."""
    key = secret = token = ""
    try:
        import streamlit as st
        sec = st.secrets.get("paytm", {}) if hasattr(st, "secrets") else {}
        key = sec.get("api_key", "")
        secret = sec.get("api_secret", "")
        token = sec.get("access_token", "")
    except Exception:                                          # noqa: BLE001
        pass
    key = key or os.environ.get("PAYTM_API_KEY", "")
    secret = secret or os.environ.get("PAYTM_API_SECRET", "")
    token = token or os.environ.get("PAYTM_ACCESS_TOKEN", "")
    return key, secret, token


def is_cloud_deployment() -> bool:
    """Detect Streamlit Community Cloud, where secrets sit in a shared env."""
    return any(os.environ.get(v) for v in
               ("STREAMLIT_SHARING_MODE", "STREAMLIT_SERVER_HEADLESS_CLOUD"))


def connect() -> tuple[object | None, BrokerStatus]:
    """Create an authenticated read-only client."""
    key, secret, token = credentials()
    if not key or not secret:
        return None, BrokerStatus(
            False,
            "No credentials found. Add a [paytm] section to .streamlit/secrets.toml "
            "or set PAYTM_API_KEY and PAYTM_API_SECRET.",
        )

    # The SDK is NOT on PyPI — it must be installed from GitHub. Its module
    # layout has varied across versions, so try the known import paths.
    PMClient = None
    for module_path, attr in (
        ("pmClient.pmClient", "PMClient"),
        ("pyPMClient", "PMClient"),
        ("pmClient", "PMClient"),
        ("pmclient", "PMClient"),
    ):
        try:
            mod = __import__(module_path, fromlist=[attr])
            PMClient = getattr(mod, attr)
            break
        except (ImportError, AttributeError):
            continue

    if PMClient is None:
        return None, BrokerStatus(
            False,
            "Paytm SDK not found. It is not published on PyPI — install from GitHub:\n\n"
            "    pip install git+https://github.com/paytmmoney/pyPMClient.git\n\n"
            "If Git is unavailable, download the repo as a zip from\n"
            "github.com/paytmmoney/pyPMClient, extract it, then run\n"
            "'pip install -r requirements.txt' followed by 'pip install .' "
            "inside the extracted folder.",
        )

    try:
        client = (PMClient(api_key=key, api_secret=secret, access_token=token)
                  if token else PMClient(api_key=key, api_secret=secret))
        profile = client.get_user_details() or {}
        return client, BrokerStatus(True, "Connected (read-only).", profile)
    except Exception as exc:                                   # noqa: BLE001
        return None, BrokerStatus(
            False, f"Authentication failed — {type(exc).__name__}: {exc}"
        )


# --------------------------------------------------------------------------
# Data retrieval
# --------------------------------------------------------------------------
def _rows(payload) -> list[dict]:
    """Normalise the various shapes the API returns into a list of dicts."""
    if payload is None:
        return []
    if isinstance(payload, dict):
        for k in ("data", "results", "holdings", "positions", "orders", "trades"):
            if k in payload and isinstance(payload[k], list):
                return payload[k]
        return [payload]
    return list(payload) if isinstance(payload, list) else []


def fetch_holdings(client) -> pd.DataFrame:
    """Current holdings with cost basis."""
    try:
        rows = _rows(client.holdings())
    except Exception:                                          # noqa: BLE001
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    rename = {
        "security_id": "security_id", "display_name": "Ticker",
        "quantity": "Qty", "cost_price": "Avg_cost",
        "last_traded_price": "LTP", "isin": "ISIN",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    for c in ("Qty", "Avg_cost", "LTP"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if {"Qty", "Avg_cost", "LTP"}.issubset(df.columns):
        df["Value"] = df["Qty"] * df["LTP"]
        df["PnL"] = (df["LTP"] - df["Avg_cost"]) * df["Qty"]
        df["PnL_pct"] = (df["LTP"] / df["Avg_cost"] - 1) * 100
    return df


def fetch_trades(client) -> pd.DataFrame:
    """Executed trades — the raw material for cost measurement."""
    for method in ("trade_details", "get_trade_details", "trades"):
        fn = getattr(client, method, None)
        if fn is None:
            continue
        try:
            rows = _rows(fn())
            if rows:
                return pd.DataFrame(rows)
        except Exception:                                      # noqa: BLE001
            continue
    return pd.DataFrame()


def fetch_orders(client) -> pd.DataFrame:
    """Order book — needed to compare intended price against actual fill."""
    for method in ("order_book", "get_order_book", "orders"):
        fn = getattr(client, method, None)
        if fn is None:
            continue
        try:
            rows = _rows(fn())
            if rows:
                return pd.DataFrame(rows)
        except Exception:                                      # noqa: BLE001
            continue
    return pd.DataFrame()


# --------------------------------------------------------------------------
# The point of the exercise: measured costs vs assumed costs
# --------------------------------------------------------------------------
def measure_slippage(orders: pd.DataFrame) -> pd.DataFrame:
    """Compare intended price against actual fill, per order.

    Slippage is the gap between the price you expected and the price you got.
    It is invisible in a backtest and directly erodes a thin edge.
    """
    if orders is None or orders.empty:
        return pd.DataFrame()

    df = orders.copy()
    price_col = next((c for c in ("price", "order_price", "limit_price")
                      if c in df.columns), None)
    fill_col = next((c for c in ("avg_traded_price", "average_price",
                                 "traded_price", "avg_price") if c in df.columns), None)
    if not price_col or not fill_col:
        return pd.DataFrame()

    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df[fill_col] = pd.to_numeric(df[fill_col], errors="coerce")
    df = df[(df[price_col] > 0) & (df[fill_col] > 0)]
    if df.empty:
        return pd.DataFrame()

    side = next((c for c in ("txn_type", "transaction_type", "side")
                 if c in df.columns), None)
    # A buy filled above the intended price is adverse; a sell below is adverse.
    if side:
        sign = np.where(df[side].astype(str).str.upper().str.startswith("B"), 1, -1)
    else:
        sign = 1
    df["slippage_pct"] = (df[fill_col] - df[price_col]) / df[price_col] * 100 * sign

    keep = [c for c in ("security_id", "display_name", side, price_col, fill_col,
                        "quantity", "slippage_pct") if c and c in df.columns]
    return df[keep].round(4)


def cost_report(trades: pd.DataFrame, orders: pd.DataFrame,
                assumed: dict[str, float] | None = None) -> dict:
    """Measured round-trip cost against the assumptions in tiers.py.

    This is the number the strategy actually depends on. An edge of ~0.5pp per
    15 days does not survive being wrong about it.
    """
    import tiers as tr
    assumed = assumed or {t: tr.params(t)["est_cost_pct"]
                          for t in ("large", "mid", "small")}

    out: dict = {"assumed": assumed, "n_trades": 0, "n_orders": 0}

    if trades is not None and not trades.empty:
        out["n_trades"] = len(trades)
        # Brokerage and statutory charges, where the API exposes them
        charge_cols = [c for c in trades.columns
                       if any(k in c.lower() for k in
                              ("brokerage", "charge", "tax", "stt", "gst", "stamp"))]
        if charge_cols:
            tot = 0.0
            for c in charge_cols:
                tot += pd.to_numeric(trades[c], errors="coerce").fillna(0).sum()
            out["total_charges"] = round(float(tot), 2)
            val_col = next((c for c in ("traded_value", "value", "amount")
                            if c in trades.columns), None)
            if val_col:
                turnover = float(pd.to_numeric(trades[val_col],
                                               errors="coerce").fillna(0).abs().sum())
                if turnover > 0:
                    one_way = tot / turnover * 100
                    out["measured_one_way_pct"] = round(one_way, 4)
                    out["measured_round_trip_pct"] = round(one_way * 2, 4)
                    out["turnover"] = round(turnover, 2)

    slip = measure_slippage(orders)
    if not slip.empty:
        out["n_orders"] = len(slip)
        out["mean_slippage_pct"] = round(float(slip["slippage_pct"].mean()), 4)
        out["median_slippage_pct"] = round(float(slip["slippage_pct"].median()), 4)
        out["worst_slippage_pct"] = round(float(slip["slippage_pct"].max()), 4)

    # Total realistic cost = charges + slippage, both legs
    rt = out.get("measured_round_trip_pct")
    sl = out.get("mean_slippage_pct")
    if rt is not None and sl is not None:
        out["all_in_round_trip_pct"] = round(rt + abs(sl) * 2, 4)

    return out


def verdict(report: dict) -> tuple[str, str]:
    """Does the measured cost leave the edge intact?"""
    n = report.get("n_trades", 0) + report.get("n_orders", 0)
    if n == 0:
        return "none", ("No trade history found. Costs cannot be measured until you "
                        "have executed trades through this account.")
    if n < 10:
        return "warn", (f"Only {n} records. Enough for a rough read, not a reliable "
                        "one — costs vary with order size, time of day and liquidity.")

    allin = report.get("all_in_round_trip_pct") or report.get("measured_round_trip_pct")
    if allin is None:
        return "warn", ("Trade records found but the API did not expose charge or fill "
                        "fields needed to compute cost. Field names vary; check the raw "
                        "data below.")

    assumed = report.get("assumed", {})
    large, mid, small = (assumed.get("large", 0.25), assumed.get("mid", 0.60),
                         assumed.get("small", 1.50))

    # Edge is roughly 0.5-0.6pp per 15-day cycle
    EDGE = 0.55
    if allin >= EDGE:
        return "bad", (
            f"Measured all-in round-trip cost is {allin:.2f}%, which meets or exceeds "
            f"the ~{EDGE:.2f}pp edge the momentum signal produces per cycle. On these "
            "costs the strategy does not make money after fees, whatever the ranking says."
        )
    if allin > large:
        return "warn", (
            f"Measured cost {allin:.2f}% is above the {large:.2f}% assumed for large "
            f"caps. The edge survives but is thinner than modelled — consider raising "
            "the minimum score to trade less often, or lengthening the holding period."
        )
    return "good", (
        f"Measured all-in round-trip cost is {allin:.2f}%, at or below the {large:.2f}% "
        f"assumed for large caps. The edge survives costs on this evidence."
    )


def reconcile_with_forward_log(holdings: pd.DataFrame,
                               log: pd.DataFrame) -> pd.DataFrame:
    """Which logged picks did you actually buy?

    The forward log records what the model suggested. This shows what you did
    with it — useful for spotting whether you systematically skip certain kinds
    of pick, which is its own form of strategy drift.
    """
    if holdings is None or holdings.empty or log is None or log.empty:
        return pd.DataFrame()
    if "Ticker" not in holdings.columns or "ticker" not in log.columns:
        return pd.DataFrame()

    held = set(holdings["Ticker"].astype(str).str.upper())
    ev = log.copy()
    ev["ticker"] = ev["ticker"].astype(str).str.upper()
    ev["held"] = ev["ticker"].isin(held)

    g = ev.groupby("snapshot_date").agg(
        picks=("ticker", "count"),
        acted_on=("held", "sum"),
    ).reset_index()
    g["action_rate_pct"] = (g["acted_on"] / g["picks"] * 100).round(1)
    return g
