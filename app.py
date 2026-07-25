"""
SwingScope — an NSE swing-trading research dashboard.

Horizon: 15-20 trading day holds.
Data: yfinance (EOD). News: Google News RSS.

This is a research tool, not a signal service. Every score is transparent and
every component is shown so you can disagree with it.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import config
import indicators as ind
import newsfeed
import scoring
import universe as uni
import regime as rg
import tiers as tr
import backtest as bt

st.set_page_config(
    page_title="SwingScope — NSE Swing Research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------
# Data layer
# --------------------------------------------------------------------------
@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_history(tickers: tuple[str, ...], period: str = "1y") -> dict[str, pd.DataFrame]:
    """Download OHLCV for a set of tickers. Cached for 30 minutes.

    yfinance returns MultiIndex columns for multi-ticker downloads, so we
    normalise everything to a plain per-ticker frame.
    """
    if not tickers:
        return {}

    raw = yf.download(
        list(tickers),
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out

    for tkr in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if tkr not in raw.columns.get_level_values(0):
                    continue
                df = raw[tkr].copy()
            else:
                df = raw.copy()
        except (KeyError, IndexError):
            continue

        df = df.dropna(how="all")
        if df.empty or len(df) < 60:
            continue

        df.columns = [str(c).title() for c in df.columns]
        needed = {"Open", "High", "Low", "Close", "Volume"}
        if not needed.issubset(set(df.columns)):
            continue

        out[tkr] = df

    return out


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_fundamentals(ticker: str) -> dict:
    """Light fundamentals pull. Wrapped because yfinance .info is flaky."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return {}
    keys = [
        "longName", "sector", "industry", "marketCap", "trailingPE",
        "forwardPE", "priceToBook", "returnOnEquity", "debtToEquity",
        "earningsTimestamp", "targetMeanPrice", "recommendationKey",
        "numberOfAnalystOpinions", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
    ]
    return {k: info.get(k) for k in keys if info.get(k) is not None}


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def sidebar() -> dict:
    st.sidebar.title("📈 SwingScope")
    st.sidebar.caption("15–20 day swing research for NSE")

    st.sidebar.subheader("Universe")
    mode = st.sidebar.radio(
        "Build universe from",
        ["Index constituents (live)", "Live market screen", "Custom list only"],
        help=(
            "Index constituents refresh from NSE every 12 hours and follow rebalances "
            "automatically. Live screens refresh every 15 minutes and follow what's "
            "actually moving today."
        ),
    )

    result = None
    if mode == "Index constituents (live)":
        idx = st.sidebar.selectbox("Index", list(uni.INDEX_FILES.keys()), index=0)
        result = uni.fetch_index_constituents(idx)
        label = idx
    elif mode == "Live market screen":
        scr = st.sidebar.selectbox("Screen", list(uni.LIVE_SCREENS.keys()), index=0)
        top_n = st.sidebar.slider("Take top N", 20, 150, 60, 10)
        result = uni.fetch_live_screen(scr, top_n=top_n)
        label = scr
    else:
        label = "Custom"

    if st.sidebar.button("🔄 Refresh universe now", use_container_width=True):
        uni.fetch_index_constituents.clear()
        uni.fetch_live_screen.clear()
        st.rerun()

    tickers: list[str] = list(result.tickers) if result else []

    # Provenance — always tell the user whether this is live or cached
    if result:
        if result.is_live:
            st.sidebar.success(
                f"🟢 Live · {len(tickers)} tickers\n\n"
                f"{result.source}\n\nFetched {result.fetched_at:%H:%M}",
            )
        else:
            st.sidebar.warning(
                f"🟠 Cached · {len(tickers)} tickers\n\n{result.note}",
            )

        # Show what changed since the last fetch
        prev_key = f"prev_universe::{label}"
        prev = st.session_state.get(prev_key)
        if prev and tuple(prev) != tuple(tickers):
            d = uni.diff_universes(tuple(prev), tuple(tickers))
            if d["added"] or d["removed"]:
                with st.sidebar.expander("↕ Changed since last fetch", expanded=False):
                    if d["added"]:
                        st.markdown("**In:** " + ", ".join(d["added"][:15]))
                    if d["removed"]:
                        st.markdown("**Out:** " + ", ".join(d["removed"][:15]))
        st.session_state[prev_key] = list(tickers)

    custom = st.sidebar.text_area(
        "Add / define tickers",
        placeholder="DIXON, TANLA, KAYNES",
        help="Comma separated, NSE symbols. Appended to the universe above.",
    )
    if custom.strip():
        extra = [t.strip().upper() for t in custom.replace("\n", ",").split(",") if t.strip()]
        extra = [t if t.endswith((".NS", ".BO")) else f"{t}.NS" for t in extra]
        tickers = sorted(set(tickers) | set(extra))

    if not tickers:
        st.sidebar.error("Universe is empty — pick an index or add custom tickers.")

    max_scan = st.sidebar.slider(
        "Cap universe size", 20, 300, min(120, max(20, len(tickers))), 10,
        help="Streamlit's free tier struggles past ~130 tickers on a cold start.",
    )
    if len(tickers) > max_scan:
        tickers = tickers[:max_scan]
        st.sidebar.caption(f"Trimmed to first {max_scan} tickers.")

    st.sidebar.divider()
    st.sidebar.subheader("Filters")

    min_turnover = st.sidebar.number_input(
        "Min avg daily turnover (₹ Cr)", min_value=0.0, value=25.0, step=5.0,
        help="Liquidity floor. Below this, slippage eats swing profits.",
    )
    rsi_lo, rsi_hi = st.sidebar.slider(
        "RSI(14) band", 0, 100, (45, 70),
        help="Momentum present but not exhausted.",
    )
    require_uptrend = st.sidebar.checkbox("Require price > 50 EMA", value=True)
    min_score = st.sidebar.slider("Min composite score", 0, 100, 55)

    st.sidebar.divider()
    st.sidebar.subheader("Risk")
    capital = st.sidebar.number_input("Trading capital (₹)", min_value=10_000, value=500_000, step=10_000)
    risk_pct = st.sidebar.slider("Risk per trade (%)", 0.25, 5.0, 1.0, 0.25)
    atr_mult = st.sidebar.slider("Stop = ATR ×", 1.0, 4.0, 2.0, 0.5)

    return dict(
        tickers=tuple(tickers),
        min_turnover=min_turnover,
        rsi_band=(rsi_lo, rsi_hi),
        require_uptrend=require_uptrend,
        min_score=min_score,
        capital=capital,
        risk_pct=risk_pct,
        atr_mult=atr_mult,
        universe_name=label,
        universe_result=result,
    )


# --------------------------------------------------------------------------
# Screener tab
# --------------------------------------------------------------------------
def render_screener(cfg: dict) -> pd.DataFrame:
    st.subheader("Swing Screener")
    st.caption(
        "Scores are a transparent weighted blend of trend, momentum, volume, "
        "relative strength and setup quality. Expand any row to see the components."
    )

    with st.spinner(f"Fetching {len(cfg['tickers'])} tickers…"):
        data = fetch_history(cfg["tickers"])
        bench = fetch_history((config.BENCHMARK,))

    if not data:
        st.error(
            "No data returned. yfinance may be rate-limiting, or the tickers are wrong. "
            "Wait a minute and re-run, or reduce the universe size."
        )
        return pd.DataFrame()

    bench_df = bench.get(config.BENCHMARK)

    # --- Regime gate: the highest-value component in the system ---
    breadth = rg.compute_breadth(data)
    reg = rg.classify(bench_df, breadth)
    st.session_state["regime"] = reg

    banner = {
        rg.RISK_ON: st.success,
        rg.NEUTRAL: st.warning,
        rg.RISK_OFF: st.error,
    }[reg.state]
    breadth_txt = f" · breadth {reg.breadth:.0%}" if reg.breadth is not None else ""
    banner(
        f"**Regime: {reg.state.replace('_', ' ').upper()}** — {reg.description} "
        f"(Nifty {reg.pct_from_200dma:+.1f}% vs 200 DMA{breadth_txt})"
    )
    if reg.state == rg.RISK_OFF:
        st.caption(
            "In risk-off the most valuable output of a screener is telling you not to trade. "
            "Small and mid caps are suppressed below."
        )

    respect_regime = st.checkbox(
        "Respect regime filter", value=True,
        help="Suppresses tiers the current regime doesn't permit. Turning this off is how accounts die in drawdowns.",
    )

    rows = []
    for tkr, df in data.items():
        try:
            metrics = scoring.evaluate(df, bench_df)
        except Exception:
            continue
        metrics["Ticker"] = tkr.replace(".NS", "")
        metrics["_raw"] = tkr
        rows.append(metrics)

    if not rows:
        st.warning("No tickers produced valid metrics.")
        return pd.DataFrame()

    res = pd.DataFrame(rows)

    # Apply filters
    lo, hi = cfg["rsi_band"]
    mask = (
        (res["Turnover_Cr"] >= cfg["min_turnover"])
        & (res["RSI"].between(lo, hi))
        & (res["Score"] >= cfg["min_score"])
    )
    if cfg["require_uptrend"]:
        mask &= res["Above_50EMA"]
    if respect_regime:
        mask &= res["Tier"].isin(reg.allowed_tiers)

    filtered = res[mask].sort_values("Score", ascending=False).reset_index(drop=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scanned", len(res))
    c2.metric("Passed filters", len(filtered))
    c3.metric("Median score", f"{res['Score'].median():.0f}")
    if bench_df is not None and len(bench_df) > 20:
        b_ret = (bench_df["Close"].iloc[-1] / bench_df["Close"].iloc[-21] - 1) * 100
        c4.metric("Nifty 20d", f"{b_ret:+.1f}%")

    if filtered.empty:
        st.info("Nothing passed. Loosen the filters in the sidebar.")
        return filtered

    show_cols = [
        "Ticker", "Tier", "Close", "Score", "Trend", "Momentum", "Volume_S",
        "RelStrength", "Setup", "RSI", "ATR_pct", "Ret_20d", "Turnover_Cr",
    ]
    display = filtered[show_cols].copy()

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Tier": st.column_config.TextColumn("Tier", width="small"),
            "Close": st.column_config.NumberColumn("Close ₹", format="%.1f"),
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
            "Trend": st.column_config.NumberColumn("Trend", format="%d"),
            "Momentum": st.column_config.NumberColumn("Mom", format="%d"),
            "Volume_S": st.column_config.NumberColumn("Vol", format="%d"),
            "RelStrength": st.column_config.NumberColumn("RS", format="%d"),
            "Setup": st.column_config.NumberColumn("Setup", format="%d"),
            "RSI": st.column_config.NumberColumn("RSI", format="%.0f"),
            "ATR_pct": st.column_config.NumberColumn("ATR %", format="%.1f"),
            "Ret_20d": st.column_config.NumberColumn("20d %", format="%+.1f"),
            "Turnover_Cr": st.column_config.NumberColumn("₹Cr/day", format="%.0f"),
        },
    )

    st.download_button(
        "Download results (CSV)",
        filtered.drop(columns=["_raw"]).to_csv(index=False).encode(),
        file_name=f"swingscope_{dt.date.today()}.csv",
        mime="text/csv",
    )
    return filtered


# --------------------------------------------------------------------------
# Detail tab
# --------------------------------------------------------------------------
def render_detail(cfg: dict, shortlist: pd.DataFrame) -> None:
    st.subheader("Stock Detail")

    options = list(shortlist["_raw"]) if not shortlist.empty else list(cfg["tickers"])[:40]
    if not options:
        st.info("Run the screener first.")
        return

    tkr = st.selectbox("Ticker", options, format_func=lambda t: t.replace(".NS", ""))
    data = fetch_history((tkr,), period="1y")
    df = data.get(tkr)
    if df is None:
        st.error("Could not load data for this ticker.")
        return

    df = ind.enrich(df)
    last = df.iloc[-1]

    fund = fetch_fundamentals(tkr)
    if fund.get("longName"):
        st.markdown(f"**{fund['longName']}** — {fund.get('sector', '—')} / {fund.get('industry', '—')}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Close", f"₹{last['Close']:,.1f}")
    c2.metric("RSI(14)", f"{last['RSI14']:.0f}")
    c3.metric("ATR(14)", f"{last['ATR14']:.1f}", f"{last['ATR14']/last['Close']*100:.1f}% of price")
    c4.metric("vs 50 EMA", f"{(last['Close']/last['EMA50']-1)*100:+.1f}%")
    dist_52w = (last["Close"] / df["High"].tail(252).max() - 1) * 100
    c5.metric("From 52w high", f"{dist_52w:+.1f}%")

    _price_chart(df, tkr)

    # Risk block — tier-aware
    tier = tr.classify_by_turnover(df)
    tp = tr.params(tier)
    st.markdown(f"#### Position sizing · tier: `{tier}`")
    st.caption(
        f"Tier defaults: ATR ×{tp['atr_mult']}, max {tp['max_position_pct']}% of capital, "
        f"max {tp['max_pct_of_adv']}% of ADV, est. round-trip cost {tp['est_cost_pct']}%."
    )
    use_tier_atr = st.checkbox("Use tier ATR multiple", value=True)
    atr_mult_eff = tp["atr_mult"] if use_tier_atr else cfg["atr_mult"]
    stop_dist = last["ATR14"] * atr_mult_eff
    stop_price = last["Close"] - stop_dist
    risk_amt = cfg["capital"] * cfg["risk_pct"] / 100
    qty = int(risk_amt // stop_dist) if stop_dist > 0 else 0
    exposure = qty * last["Close"]

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Stop", f"₹{stop_price:,.1f}", f"-{stop_dist/last['Close']*100:.1f}%")
    r2.metric("Qty", f"{qty:,}")
    r3.metric("Exposure", f"₹{exposure:,.0f}", f"{exposure/cfg['capital']*100:.0f}% of capital")
    r4.metric("Risk", f"₹{risk_amt:,.0f}")

    adv_shares = float(df["Volume"].tail(20).mean())
    lim = tr.position_limits(tier, cfg["capital"], float(last["Close"]), adv_shares)
    if qty > lim["max_qty"]:
        st.warning(
            f"ATR sizing suggests {qty:,} shares, but the {tier}-cap cap is "
            f"{lim['max_qty']:,} (bound by {lim['capped_by']}). Use the lower number — "
            "in small caps the risk isn't being wrong, it's being right and unable to exit."
        )
        qty = lim["max_qty"]
        exposure = qty * last["Close"]
    if exposure > cfg["capital"]:
        st.warning(
            f"This position needs ₹{exposure:,.0f} but you have ₹{cfg['capital']:,.0f}. "
            "Either the stop is too tight or risk-per-trade is too high for this stock's volatility."
        )

    stop_pct = stop_dist / last["Close"] * 100
    net2 = tr.net_expected_r(2.0, tier, stop_pct)
    st.caption(
        f"Cost drag: a 2.0R gross win nets **{net2:.2f}R** after an estimated "
        f"{tp['est_cost_pct']}% round-trip on a {stop_pct:.1f}% stop."
    )

    targets = pd.DataFrame({
        "Level": ["Entry", "Stop", "1R", "2R", "3R"],
        "Price": [
            last["Close"], stop_price,
            last["Close"] + stop_dist,
            last["Close"] + 2 * stop_dist,
            last["Close"] + 3 * stop_dist,
        ],
    })
    targets["Move %"] = (targets["Price"] / last["Close"] - 1) * 100
    st.dataframe(targets, hide_index=True, use_container_width=True,
                 column_config={
                     "Price": st.column_config.NumberColumn(format="₹%.1f"),
                     "Move %": st.column_config.NumberColumn(format="%+.1f%%"),
                 })

    if fund:
        with st.expander("Fundamentals & analyst view"):
            st.json(fund)


def _price_chart(df: pd.DataFrame, tkr: str) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        st.line_chart(df[["Close", "EMA20", "EMA50"]].tail(180))
        return

    view = df.tail(180)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03,
        subplot_titles=(f"{tkr.replace('.NS','')} — daily", "Volume", "RSI(14)"),
    )
    fig.add_trace(go.Candlestick(
        x=view.index, open=view["Open"], high=view["High"],
        low=view["Low"], close=view["Close"], name="Price",
    ), row=1, col=1)
    for col, colour in (("EMA20", "#f59e0b"), ("EMA50", "#3b82f6"), ("EMA200", "#94a3b8")):
        if col in view:
            fig.add_trace(go.Scatter(x=view.index, y=view[col], name=col,
                                     line=dict(width=1.2, color=colour)), row=1, col=1)

    colors = np.where(view["Close"] >= view["Open"], "#16a34a", "#dc2626")
    fig.add_trace(go.Bar(x=view.index, y=view["Volume"], name="Vol",
                         marker_color=colors, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["RSI14"], name="RSI",
                             line=dict(color="#8b5cf6", width=1.4)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#dc2626", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#16a34a", row=3, col=1)

    fig.update_layout(height=680, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------
# News tab
# --------------------------------------------------------------------------
def render_news(cfg: dict, shortlist: pd.DataFrame) -> None:
    st.subheader("News & Catalysts")
    st.caption(
        "Headlines from Google News RSS. Sentiment is a crude keyword score — "
        "treat it as a triage aid, not a signal."
    )

    if not shortlist.empty:
        default = list(shortlist["Ticker"].head(6))
    else:
        default = [t.replace(".NS", "") for t in cfg["tickers"][:6]]

    picks = st.multiselect("Stocks", [t.replace(".NS", "") for t in cfg["tickers"]], default=default)
    if not picks:
        st.info("Pick at least one stock.")
        return

    for name in picks:
        with st.expander(f"📰 {name}", expanded=len(picks) <= 3):
            items = newsfeed.fetch(name)
            if not items:
                st.caption("No recent headlines found.")
                continue
            for it in items[:8]:
                tone = newsfeed.score_headline(it["title"])
                badge = {"pos": "🟢", "neg": "🔴", "neu": "⚪"}[tone]
                st.markdown(f"{badge} [{it['title']}]({it['link']})")
                st.caption(f"{it['source']} · {it['published']}")


# --------------------------------------------------------------------------
# Journal tab
# --------------------------------------------------------------------------
def render_journal() -> None:
    st.subheader("Trade Journal")
    st.caption(
        "Kept in session state — it resets when the app restarts. "
        "Download regularly, or wire it to a Google Sheet / database for persistence."
    )

    if "journal" not in st.session_state:
        st.session_state.journal = pd.DataFrame(
            columns=["Date", "Ticker", "Side", "Entry", "Stop", "Target", "Qty", "Exit", "Notes"]
        )

    with st.form("add_trade", clear_on_submit=True):
        c = st.columns(5)
        d = c[0].date_input("Date", dt.date.today())
        t = c[1].text_input("Ticker")
        side = c[2].selectbox("Side", ["Long", "Short"])
        entry = c[3].number_input("Entry", min_value=0.0, step=0.5)
        stop = c[4].number_input("Stop", min_value=0.0, step=0.5)
        c2 = st.columns(4)
        target = c2[0].number_input("Target", min_value=0.0, step=0.5)
        qty = c2[1].number_input("Qty", min_value=0, step=1)
        exit_px = c2[2].number_input("Exit (0 if open)", min_value=0.0, step=0.5)
        notes = c2[3].text_input("Notes")
        if st.form_submit_button("Add trade") and t:
            row = pd.DataFrame([{
                "Date": d, "Ticker": t.upper(), "Side": side, "Entry": entry,
                "Stop": stop, "Target": target, "Qty": qty,
                "Exit": exit_px, "Notes": notes,
            }])
            st.session_state.journal = pd.concat([st.session_state.journal, row], ignore_index=True)

    j = st.session_state.journal
    if j.empty:
        st.info("No trades logged yet.")
        return

    closed = j[j["Exit"] > 0].copy()
    if not closed.empty:
        sign = np.where(closed["Side"] == "Long", 1, -1)
        closed["PnL"] = (closed["Exit"] - closed["Entry"]) * closed["Qty"] * sign
        risk = (closed["Entry"] - closed["Stop"]).abs() * closed["Qty"]
        closed["R"] = np.where(risk > 0, closed["PnL"] / risk, np.nan)

        wins = closed[closed["PnL"] > 0]
        m = st.columns(5)
        m[0].metric("Closed", len(closed))
        m[1].metric("Win rate", f"{len(wins)/len(closed)*100:.0f}%")
        m[2].metric("Net P&L", f"₹{closed['PnL'].sum():,.0f}")
        m[3].metric("Avg R", f"{closed['R'].mean():.2f}")
        exp = closed["R"].mean()
        m[4].metric("Expectancy", f"{exp:.2f}R", "positive" if exp > 0 else "negative")

        if exp <= 0:
            st.warning(
                "Negative expectancy: this strategy loses money over time regardless of win rate. "
                "Check whether your losers are running past their stops."
            )

    st.dataframe(j, use_container_width=True, hide_index=True)
    st.download_button("Download journal (CSV)", j.to_csv(index=False).encode(),
                       file_name="trade_journal.csv", mime="text/csv")


# --------------------------------------------------------------------------
# Backtest tab
# --------------------------------------------------------------------------
def render_backtest(cfg: dict) -> None:
    st.subheader("Walk-forward backtest")
    st.caption(
        "Answers the only question that matters: does the score predict forward returns, "
        "or does it just describe charts that already went up? Signals are evaluated on "
        "past bars only; entries fill at the *next* bar's open."
    )

    c = st.columns(4)
    min_score = c[0].slider("Signal threshold", 40, 90, 65, 5)
    hold = c[1].slider("Hold (bars)", 5, 40, 18)
    rebal = c[2].slider("Check every N bars", 1, 10, 5)
    n_stocks = c[3].slider("Stocks to test", 10, 100, 30, 10)

    c2 = st.columns(2)
    use_regime = c2[0].checkbox("Apply regime filter", value=True)
    apply_costs = c2[1].checkbox("Subtract transaction costs", value=True)

    if not st.button("Run backtest", type="primary"):
        st.info("Configure above, then run. Expect 30–90 seconds for 30 stocks over 2 years.")
        return

    tickers = tuple(cfg["tickers"][:n_stocks])
    with st.spinner(f"Fetching 2y history for {len(tickers)} tickers…"):
        frames = fetch_history(tickers, period="2y")
        bench = fetch_history((config.BENCHMARK,), period="2y").get(config.BENCHMARK)

    if not frames:
        st.error("No data returned — try again shortly, yfinance may be rate-limiting.")
        return

    with st.spinner("Walking forward…"):
        trades = bt.run(frames, bench, min_score=min_score, hold_bars=hold,
                        rebalance_every=rebal, use_regime_filter=use_regime,
                        apply_costs=apply_costs)

    if trades.empty:
        st.warning(
            "No trades triggered. Lower the threshold, widen the hold period, or "
            "check that the regime filter isn't suppressing the whole universe."
        )
        return

    overall = bt.summarize(trades)
    exp = float(overall["expectancy_R"].iloc[0])
    n = int(overall["trades"].iloc[0])

    m = st.columns(5)
    m[0].metric("Trades", n)
    m[1].metric("Win rate", f"{overall['win_rate'].iloc[0]:.0f}%")
    m[2].metric("Expectancy", f"{exp:+.3f}R")
    m[3].metric("Total", f"{overall['total_R'].iloc[0]:+.0f}R")
    m[4].metric("Stopped out", f"{overall['stopped_pct'].iloc[0]:.0f}%")

    if n < 100:
        st.warning(
            f"Only {n} trades. That is not enough to distinguish edge from luck — "
            "aim for 200+ before drawing any conclusion."
        )
    if exp <= 0:
        st.error(
            f"**Negative expectancy ({exp:+.3f}R).** This configuration loses money over time "
            "regardless of win rate. Do not trade it. Change the rules, not the position sizing."
        )
    else:
        st.success(
            f"Positive expectancy ({exp:+.3f}R) across {n} trades. Treat as a hypothesis, "
            "not proof — verify on a period you have not tuned against."
        )

    st.markdown("##### By tier")
    st.dataframe(bt.summarize(trades, "tier"), hide_index=True, use_container_width=True)

    st.markdown("##### By regime")
    st.dataframe(bt.summarize(trades, "regime"), hide_index=True, use_container_width=True)

    st.markdown("##### By score bucket")
    st.caption(
        "This is the real test. If expectancy does **not** rise with score, the score "
        "is not ranking anything and the whole model is decoration."
    )
    st.dataframe(bt.summarize(trades, "score_bucket"), hide_index=True, use_container_width=True)

    ec = bt.equity_curve(trades, cfg["risk_pct"], cfg["capital"])
    if not ec.empty:
        st.markdown("##### Equity curve")
        st.line_chart(ec.set_index("date")["equity"])
        e1, e2 = st.columns(2)
        e1.metric("Final equity", f"₹{ec['equity'].iloc[-1]:,.0f}",
                  f"{(ec['equity'].iloc[-1]/cfg['capital']-1)*100:+.1f}%")
        e2.metric("Max drawdown", f"{ec['drawdown_pct'].min():.1f}%")

    with st.expander("All trades"):
        st.dataframe(trades.sort_values("entry_date", ascending=False),
                     hide_index=True, use_container_width=True)
    st.download_button("Download trades (CSV)", trades.to_csv(index=False).encode(),
                       file_name="backtest_trades.csv", mime="text/csv")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    cfg = sidebar()

    st.title("SwingScope")
    res = cfg.get("universe_result")
    provenance = "live from NSE" if (res and res.is_live) else "cached snapshot"
    st.caption(
        f"Universe: **{cfg['universe_name']}** ({len(cfg['tickers'])} tickers, {provenance}) · "
        f"Data: yfinance EOD · Generated {dt.datetime.now():%d %b %Y %H:%M}"
    )

    st.info(
        "**EOD data.** Prices are end-of-day, not live. Confirm every level on your "
        "broker terminal before acting. This is a research tool, not advice — "
        "swing trading carries real risk of loss.",
        icon="⚠️",
    )

    tabs = st.tabs(["🔍 Screener", "📊 Detail", "🧪 Backtest", "📰 News", "📓 Journal", "❓ Method"])

    with tabs[0]:
        shortlist = render_screener(cfg)
    with tabs[1]:
        render_detail(cfg, shortlist)
    with tabs[2]:
        render_backtest(cfg)
    with tabs[3]:
        render_news(cfg, shortlist)
    with tabs[4]:
        render_journal()
    with tabs[5]:
        st.markdown(config.METHOD_DOC)


if __name__ == "__main__":
    main()
