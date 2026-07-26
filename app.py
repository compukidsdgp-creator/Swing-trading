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
import momentum as momo
import universe as uni
import regime as rg
import tiers as tr
import backtest as bt
import validate as val
import factor_analysis as fana
import signals as sig_lab
import composite as comp
import data_quality as dq
import forward_log as flog
import sentiment as sent
import report as rep
import bucket as bk

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
    st.sidebar.subheader("Scoring model")
    model = st.sidebar.radio(
        "Ranking signal",
        ["Momentum (validated)", "Composite v1 (failed validation)"],
        index=0,
        help="v1 blended five components that all measured the same thing — it retained "
             "only 13.9% of its IC after factor neutralisation. Momentum (12-1) cleared "
             "t = 3.73. Kept switchable so you can compare.",
    )
    if model.startswith("Composite"):
        st.sidebar.warning(
            "v1 failed validation: residual IC +0.004, t = 0.17. Shown for comparison only.",
            icon="⚠️",
        )
    else:
        st.sidebar.caption(
            "12-1 momentum · residual IC +0.055 · t = 3.73 · 59.7% positive windows"
        )

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
    if model.startswith("Momentum"):
        min_score = st.sidebar.slider(
            "Min percentile rank", 0, 100, 50,
            help="Percentile within today's universe. Note this is RELATIVE — the "
                 "absolute floor below is what actually gates quality.",
        )
        require_positive_mom = st.sidebar.checkbox(
            "Require positive 12-1 momentum", value=True,
            help="Without this, a percentile rank promotes the best of a falling "
                 "market and stamps it 100. Strongly recommended.",
        )
    else:
        min_score = st.sidebar.slider("Min composite score", 0, 100, 55)
        require_positive_mom = False

    st.sidebar.divider()
    _report_links_sidebar()

    st.sidebar.subheader("Risk")
    capital = st.sidebar.number_input("Trading capital (₹)", min_value=10_000, value=500_000, step=10_000)
    risk_pct = st.sidebar.slider("Risk per trade (%)", 0.25, 5.0, 1.0, 0.25)
    atr_mult = st.sidebar.slider("Stop = ATR ×", 1.0, 4.0, 2.0, 0.5)

    return dict(
        tickers=tuple(tickers),
        model="momentum" if model.startswith("Momentum") else "composite_v1",
        min_turnover=min_turnover,
        rsi_band=(rsi_lo, rsi_hi),
        require_uptrend=require_uptrend,
        min_score=min_score,
        capital=capital,
        risk_pct=risk_pct,
        atr_mult=atr_mult,
        universe_name=label,
        universe_result=result,
        require_positive_mom=require_positive_mom,
    )


def _report_links_sidebar() -> None:
    """Quick links to the automated reports, if a repo has been configured."""
    links = config.report_links()
    if not links:
        with st.sidebar.expander("📄 Automated reports"):
            st.caption(
                "Set `GITHUB_USER` in config.py — or add a `[reports]` section to "
                "Streamlit secrets — and direct links to your weekly and month-end "
                "reports will appear here."
            )
        return

    with st.sidebar.expander("📄 Automated reports", expanded=False):
        if "weekly_html" in links:
            st.link_button("📈 Latest weekly report", links["weekly_html"],
                           use_container_width=True)
        if "monthly_html" in links:
            st.link_button("📊 Latest month-end review", links["monthly_html"],
                           use_container_width=True)
        st.link_button("📁 All reports", links["folder"], use_container_width=True)
        st.link_button("⚙️ Workflow runs", links["actions"], use_container_width=True)
        if "weekly_html" not in links:
            st.caption(
                "Enable GitHub Pages on the repo for one-click viewing; until then "
                "these open the files on GitHub."
            )


@st.dialog("Today's bucket", width="large")
def _bucket_dialog(b, reg) -> None:
    """Popup showing the assembled shortlist."""
    tone = {"risk_on": st.success, "neutral": st.warning, "risk_off": st.error}
    tone.get(b.regime_state, st.info)(
        f"**{b.regime_state.replace('_', ' ').upper()}** — {reg.description}"
    )

    if b.is_empty:
        st.warning("No picks qualified.")
        for n in b.notes:
            st.caption(f"· {n}")
        st.info("Taking no positions is a valid outcome, and in a hostile regime "
                "it is usually the correct one.")
        return

    m = st.columns(3)
    m[0].metric("Picks", b.actual_size)
    m[1].metric("Target", b.target_size)
    m[2].metric("Mix", " / ".join(f"{k[0].upper()}{v}"
                                  for k, v in sorted(b.tier_counts.items())))

    cols = [c for c in ["Rank", "Ticker", "Tier", "Sector", "Score", "Close",
                        "RSI", "ATR_pct"] if c in b.picks.columns]
    st.dataframe(
        b.picks[cols], hide_index=True, use_container_width=True,
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0,
                                                     max_value=100, format="%d"),
            "Close": st.column_config.NumberColumn("₹", format="%.1f"),
            "ATR_pct": st.column_config.NumberColumn("ATR %", format="%.1f"),
        },
    )

    if b.notes:
        with st.expander("How this bucket was built"):
            for n in b.notes:
                st.markdown(f"- {n}")

    st.download_button("⬇ Bucket (CSV)", b.picks.to_csv(index=False).encode(),
                       file_name=f"bucket_{dt.date.today()}.csv", mime="text/csv",
                       use_container_width=True)
    with st.expander("Plain text (for WhatsApp / Telegram)"):
        st.code(bk.to_text(b, regime_desc=reg.description), language=None)

    st.caption(
        "Analytical view, not advice. Scores measure how closely a chart matches a "
        "trend-continuation pattern — not how lucrative a stock is. Check each chart "
        "and the News tab for earnings inside your holding window before acting."
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

    if cfg.get("model") == "momentum":
        res = momo.rank_universe(
            data, bench_df,
            min_turnover_cr=0.0,        # sidebar filter applies below
            require_above_50ema=False,  # ditto
            min_momentum=0.0 if cfg.get("require_positive_mom", True) else None,
        )
        if res.empty:
            st.warning("No tickers produced valid momentum values.")
            return pd.DataFrame()
        with st.expander("What this signal is, and why"):
            st.markdown(momo.explain())
    else:
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
        st.warning(
            "**Composite v1 is shown for comparison only.** It failed factor "
            "neutralisation — residual IC +0.004, t = 0.17. Its ranking is not "
            "supported by evidence.", icon="⚠️",
        )

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

    if cfg.get("model") == "momentum":
        show_cols = ["Ticker", "Tier", "Close", "Score", "Momentum", "RSI",
                     "ATR_pct", "Ret_20d", "Turnover_Cr", "Cost_viable"]
    else:
        show_cols = ["Ticker", "Tier", "Close", "Score", "Trend", "Momentum",
                     "Volume_S", "RelStrength", "Setup", "RSI", "ATR_pct",
                     "Ret_20d", "Turnover_Cr"]
    show_cols = [c for c in show_cols if c in filtered.columns]
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

    if cfg.get("model") == "momentum" and "Cost_viable" in filtered.columns:
        n_bad = int((~filtered["Cost_viable"]).sum())
        if n_bad:
            st.caption(
                f"⚠️ {n_bad} of {len(filtered)} are small caps. At ~1.5% round-trip cost "
                "against a ~0.5pp edge, momentum is not economically viable there "
                "regardless of rank."
            )

    dl, bcol = st.columns([1, 1])
    dl.download_button(
        "Download results (CSV)",
        filtered.drop(columns=["_raw"]).to_csv(index=False).encode(),
        file_name=f"swingscope_{dt.date.today()}.csv",
        mime="text/csv", use_container_width=True,
    )

    with bcol.popover("⚙️ Bucket settings", use_container_width=True):
        b_size = st.slider("Bucket size", 3, 20, 10)
        b_sector = st.slider("Max per sector", 1, 5, 2)
        b_balance = st.checkbox("Balance across tiers", value=True,
                                help="Spread across large/mid/small within what the "
                                     "regime permits, rather than taking the top N.")

    if st.button("🎯 Build today's bucket", type="primary", use_container_width=True):
        b = bk.build(filtered, reg, size=b_size, max_per_sector=b_sector,
                     min_score=cfg["min_score"], balance_tiers=b_balance)
        st.session_state["bucket"] = b
        _bucket_dialog(b, reg)

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

    if not shortlist.empty:
        default = list(shortlist["Ticker"].head(8))
    else:
        default = [t.replace(".NS", "") for t in cfg["tickers"][:8]]

    picks = st.multiselect(
        "Stocks", [t.replace(".NS", "") for t in cfg["tickers"]], default=default
    )
    if not picks:
        st.info("Pick at least one stock.")
        return

    with st.spinner(f"Fetching headlines for {len(picks)} stocks…"):
        feeds = {name: newsfeed.fetch(name, limit=12) for name in picks}
        results = [sent.analyse_stock(name, items) for name, items in feeds.items()]

    report = sent.portfolio_report(results)

    # ---------------- Report panel ----------------
    st.markdown("### 📋 Sentiment snapshot")
    if "error" in report:
        st.warning(report["error"])
    else:
        m = st.columns(5)
        m[0].metric("Covered", f"{report['stocks_covered']}/{len(picks)}")
        m[1].metric("Headlines", report["total_headlines"])
        m[2].metric("Bullish", report["bullish"])
        m[3].metric("Bearish", report["bearish"])
        m[4].metric("Avg score", f"{report['avg_sentiment']:+.2f}")

        avg = report["avg_sentiment"]
        if avg >= 0.7:
            st.success(f"Overall tone across your watchlist is positive ({avg:+.2f}).")
        elif avg <= -0.7:
            st.error(
                f"Overall tone is negative ({avg:+.2f}). Be sceptical of long setups "
                "while the news flow is against you."
            )
        else:
            st.info(f"Overall tone is mixed or neutral ({avg:+.2f}).")

        c1, c2 = st.columns(2)
        mp, mn = report.get("most_positive"), report.get("most_negative")
        if mp:
            c1.markdown(
                f"**🟢 Most positive — {mp.ticker}** ({mp.label}, {mp.mean_score:+.2f})"
                + (f"\n\n_{mp.top_positive}_" if mp.top_positive else "")
            )
        if mn:
            c2.markdown(
                f"**🔴 Most negative — {mn.ticker}** ({mn.label}, {mn.mean_score:+.2f})"
                + (f"\n\n_{mn.top_negative}_" if mn.top_negative else "")
            )

        if report["top_catalysts"]:
            st.markdown("**Catalyst types detected:** " + " · ".join(
                f"`{k.replace('_',' ')}` ×{v}" for k, v in report["top_catalysts"]
            ))

        if report["flagged_count"]:
            with st.expander(f"⚠️ {report['flagged_count']} stocks with alerts", expanded=True):
                for r in report["flagged"]:
                    st.markdown(f"**{r.ticker}**")
                    for al in r.alerts:
                        st.markdown(f"- {al}")

        rows = [{
            "Stock": r.ticker, "Sentiment": r.label, "Score": r.mean_score,
            "Confidence": r.confidence, "Headlines": r.n_headlines,
            "Pos": r.pos, "Neg": r.neg, "Neutral": r.neu,
            "Catalysts": ", ".join(r.catalysts) if r.catalysts else "—",
        } for r in sorted(results, key=lambda x: x.mean_score, reverse=True)]
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, use_container_width=True,
            column_config={
                "Score": st.column_config.NumberColumn("Score", format="%+.2f"),
            },
        )

    st.caption(
        "Sentiment uses a weighted financial lexicon with negation and contrast-clause "
        "handling — it reads \"shares fall despite strong profit\" correctly. It is still "
        "a lexicon model, not comprehension. **Use it to decide what to read, never as a "
        "trading signal.** The main value here is catching scheduled events — especially "
        "earnings — inside your holding window."
    )

    # ---------------- Headlines ----------------
    st.markdown("### 📰 Headlines")
    order = {r.ticker: r for r in results}
    for name in picks:
        r = order.get(name)
        badge = {"Bullish": "🟢", "Leaning bullish": "🟢", "Bearish": "🔴",
                 "Leaning bearish": "🔴", "Mixed": "🟡", "Neutral": "⚪",
                 "No data": "⚫"}.get(r.label if r else "No data", "⚪")
        title = f"{badge} {name} — {r.label} ({r.mean_score:+.2f}, {r.n_headlines} items)" if r else name
        with st.expander(title, expanded=len(picks) <= 3):
            items = feeds.get(name, [])
            if not items:
                st.caption("No recent headlines found.")
                continue
            for it in items[:10]:
                sc = sent.score_text(it["title"])
                dot = {"pos": "🟢", "neg": "🔴", "neu": "⚪"}[sc.label]
                st.markdown(f"{dot} **{sc.score:+.1f}** · [{it['title']}]({it['link']})")
                meta = f"{it['source']} · {it['published']}"
                if sc.catalysts:
                    meta += " · " + ", ".join(f"`{c.replace('_',' ')}`" for c in sc.catalysts)
                st.caption(meta)


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
            st.session_state.journal = pd.concat([st.session_state.journal, row],
                                                 ignore_index=True)

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
                "Negative expectancy: this strategy loses money over time regardless of "
                "win rate. Check whether your losers are running past their stops."
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

    c2 = st.columns(3)
    use_regime = c2[0].checkbox("Apply regime filter", value=True)
    apply_costs = c2[1].checkbox("Subtract transaction costs", value=True)
    bt_model = c2[2].selectbox("Model", ["momentum", "composite_v1"], index=0,
                               key="bt_model",
                               help="composite_v1 failed factor neutralisation and is "
                                    "retained for comparison only.")

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
                        apply_costs=apply_costs, model=bt_model)

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
            f"**Negative expectancy ({exp:+.3f}R).** This configuration loses money over "
            "time regardless of win rate. Do not trade it. Change the rules, not the "
            "position sizing."
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
    st.dataframe(bt.summarize(trades, "score_bucket"), hide_index=True,
                 use_container_width=True)

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
# Factor analysis (inside Validation tab)
# --------------------------------------------------------------------------
def render_factor_analysis(cfg: dict) -> None:
    st.subheader("Factor neutralisation")
    st.caption(
        "IC tells you the score predicts returns. It does not tell you whether it predicts "
        "anything **new**. This runs Fama-MacBeth cross-sectional regressions against six "
        "well-known factors — 12-1 momentum, short-term reversal, size, volatility, beta and "
        "liquidity — and reports what survives. If a plain momentum screen gets you the same "
        "result, the model is an expensive wrapper around a published anomaly."
    )

    c = st.columns(3)
    horizon = c[0].slider("Horizon (days)", 5, 30, 15, key="fa_h")
    years = c[1].select_slider("Period", ["2y", "5y", "10y"], value="5y", key="fa_y")
    n_stocks = c[2].slider("Stocks", 30, 150, 100, 10, key="fa_n")

    with st.expander("What the numbers mean"):
        st.markdown(
            "**IC retention** — share of raw IC surviving after the six factors are "
            "controlled for. Below 40% means most of your edge is standard factor "
            "exposure.\n\n"
            "**Newey-West t** — corrects for autocorrelation across overlapping windows. "
            "The naive t-statistic is systematically too high; expect this to be lower.\n\n"
            "**Harvey-Liu-Zhu bar (t > 3.0)** — after accounting for how many strategies "
            "get tested across the literature, the conventional t > 2 produces too many "
            "false positives. Serious factor claims now use 3.0.\n\n"
            "**Factor loadings** — what the score is actually correlated with. A loading "
            "above 0.7 on any single factor is a warning sign."
        )

    if not st.button("Run factor analysis", key="fa_run"):
        st.info("Takes 2–4 minutes for 100 stocks over 5 years.")
        return

    tickers = tuple(cfg["tickers"][:n_stocks])
    with st.spinner(f"Fetching {years} history for {len(tickers)} tickers…"):
        frames = fetch_history(tickers, period=years)
        bench = fetch_history((config.BENCHMARK,), period=years).get(config.BENCHMARK)

    if not frames:
        st.error("No data returned — yfinance may be rate-limiting.")
        return

    with st.spinner("Running cross-sectional regressions…"):
        res = fana.run(frames, bench, horizon=horizon)

    s = res.summary
    if "error" in s:
        st.error(f"Could not run: {s['error']}")
        return

    m = st.columns(5)
    m[0].metric("Raw IC", f"{s['raw_ic']:+.4f}")
    m[1].metric("Residual IC", f"{s['residual_ic']:+.4f}")
    m[2].metric("Retention", f"{s['ic_retention_pct']}%"
                if s['ic_retention_pct'] is not None else "—")
    m[3].metric("Newey-West t", f"{s['fm_score_t_newey_west']}"
                if s['fm_score_t_newey_west'] is not None else "—")
    m[4].metric("Windows", s["n_windows"])

    level, message, notes = fana.verdict(s)
    {"good": st.success, "ok": st.info, "warn": st.warning,
     "bad": st.error, "error": st.error}[level](message)
    for n in notes:
        st.markdown(f"- {n}")

    b1, b2 = st.columns(2)
    b1.metric("Clears t > 2", "Yes" if s["passes_t2"] else "No")
    b2.metric("Clears t > 3 (HLZ)", "Yes" if s["passes_hlz_t3"] else "No",
              help="Harvey, Liu & Zhu (2016) multiple-testing threshold for "
                   "claiming a genuinely new factor.")

    st.markdown("##### What the score is correlated with")
    st.caption("A loading above 0.7 means the score is largely that factor in disguise.")
    st.dataframe(res.loadings, hide_index=True, use_container_width=True)
    st.bar_chart(res.loadings.set_index("factor")["mean_correlation"])

    st.markdown("##### Fama-MacBeth regression")
    st.caption(
        "Forward return regressed on the score plus all six controls, cross-sectionally "
        "each window, then averaged over time. The `score` row is the question: does it "
        "add anything once everything else is accounted for?"
    )
    st.dataframe(res.fm_coefficients, hide_index=True, use_container_width=True)

    st.markdown("##### Raw vs residual IC over time")
    st.line_chart(res.per_window.set_index("date")[["raw_ic", "residual_ic"]])

    st.download_button("Download per-window detail (CSV)",
                       res.per_window.to_csv(index=False).encode(),
                       file_name="factor_analysis_windows.csv", mime="text/csv")

    st.warning(
        "**This does not fix survivorship bias.** Factor neutralisation is a statistical "
        "correction; survivorship is a data problem. Your universe still excludes every "
        "company that failed and was delisted, which no regression can repair — it needs "
        "a point-in-time database.",
        icon="⚠️",
    )


# --------------------------------------------------------------------------
# Signal laboratory (inside Validation tab)
# --------------------------------------------------------------------------
def render_signal_lab(cfg: dict) -> None:
    st.subheader("Signal laboratory")
    st.caption(
        "The composite score failed because its five components all measured the same "
        "thing. Rather than hand-craft another blend, this tests **twelve candidate "
        "signals individually** and reports which — if any — predict returns *after* the "
        "standard factor set is controlled for. A signal with high raw IC and zero "
        "residual IC is a repackaged factor. One with modest raw IC but positive residual "
        "IC is worth far more."
    )

    c = st.columns(3)
    horizon = c[0].slider("Horizon (days)", 5, 30, 15, key="sl_h")
    years = c[1].select_slider("Period", ["2y", "5y", "10y"], value="5y", key="sl_y")
    n_stocks = c[2].slider("Stocks", 30, 150, 100, 10, key="sl_n")

    with st.expander("What each signal is, and why it's here"):
        st.markdown(
            "| Signal | Rationale |\n|---|---|\n"
            "| `mom_12_1` | Classic momentum (Jegadeesh & Titman 1993). The benchmark to beat. |\n"
            "| `mom_6_1` | Shorter formation window. |\n"
            "| `idiosyncratic_mom` | Momentum of the market-beta residual (Blitz et al. 2011). |\n"
            "| `mom_consistency` | Share of positive days — path, not just endpoint. |\n"
            "| `vol_adjusted_mom` | Momentum divided by volatility. |\n"
            "| `52w_high_proximity` | Anchoring effect (George & Hwang 2004). |\n"
            "| `acceleration` | Is the trend speeding up or fading? |\n"
            "| `accumulation` | Up-day volume minus down-day volume. |\n"
            "| `low_volatility` | The low-volatility anomaly. |\n"
            "| `reversal_1m` | Short-term reversal. |\n"
            "| `illiquidity` | Amihud (2002) illiquidity premium. |\n"
            "| `range_compression` | Volatility squeeze. |\n\n"
            "**Read the `residual_ic` column, not `raw_ic`.** Raw IC mostly reflects "
            "factor exposure you can buy cheaply elsewhere."
        )

    if not st.button("Run signal lab", key="sl_run"):
        st.info("Takes 3–6 minutes for 100 stocks over 5 years — it computes twelve "
                "signals plus six controls per stock per window.")
        return

    tickers = tuple(cfg["tickers"][:n_stocks])
    with st.spinner(f"Fetching {years} history for {len(tickers)} tickers…"):
        frames = fetch_history(tickers, period=years)
        bench = fetch_history((config.BENCHMARK,), period=years).get(config.BENCHMARK)

    if not frames:
        st.error("No data returned — yfinance may be rate-limiting.")
        return

    with st.spinner("Testing twelve signals…"):
        res = sig_lab.run(frames, bench, horizon=horizon)

    if res.table.empty:
        st.error("No valid windows — try a longer period or more stocks.")
        return

    st.caption(f"{res.n_windows} non-overlapping windows.")

    winners = res.table[res.table["residual_t_nw"].abs() >= 2.0] \
        if res.table["residual_t_nw"].notna().any() else res.table.iloc[0:0]

    if winners.empty:
        st.error(
            "**No signal shows significant incremental content.** Every candidate is "
            "either a repackaged factor or noise. This is the normal outcome of signal "
            "research — and a real finding: at retail data quality, price-based signals "
            "do not clear the bar once known factors are controlled for."
        )
    else:
        st.success(
            f"**{len(winners)} signal(s) retain significant content** after factor "
            "neutralisation. Treat as candidates for further testing, not conclusions."
        )

    for n in res.notes:
        st.markdown(f"- {n}")

    st.markdown("##### Results, ranked by residual IC")
    st.dataframe(
        res.table, hide_index=True, use_container_width=True,
        column_config={
            "raw_ic": st.column_config.NumberColumn("Raw IC", format="%.4f"),
            "residual_ic": st.column_config.NumberColumn("Residual IC", format="%.4f"),
            "retention_pct": st.column_config.NumberColumn("Retention %", format="%.1f"),
            "residual_t_nw": st.column_config.NumberColumn("Residual t (NW)", format="%.2f"),
            "pct_positive": st.column_config.NumberColumn("% pos windows", format="%.1f"),
        },
    )

    plot = res.table.set_index("signal")[["raw_ic", "residual_ic"]]
    st.markdown("##### Raw vs residual IC")
    st.caption("The gap between the two bars is the part that is standard factor exposure.")
    st.bar_chart(plot)

    st.download_button("Download results (CSV)", res.table.to_csv(index=False).encode(),
                       file_name="signal_lab.csv", mime="text/csv")

    st.info(
        "**If something looks promising, resist building on it immediately.** Twelve "
        "signals tested means twelve chances for one to clear t>2 by luck alone — roughly "
        "even odds. That is exactly the multiple-testing problem the t>3 bar exists for. "
        "Confirm on a different period or universe before trusting it.",
        icon="⚠️",
    )


# --------------------------------------------------------------------------
# Composite builder (inside Validation tab)
# --------------------------------------------------------------------------
def render_composite_builder(cfg: dict) -> None:
    st.subheader("Orthogonal composite builder")
    st.caption(
        "The v1 score failed because its five components all measured the same thing — "
        "nobody checked whether they were independent. This enforces two gates before a "
        "signal is admitted: **significant incremental content** (Newey-West t above the "
        "threshold, after factor neutralisation) and **independence** (correlation with "
        "every already-selected component below the ceiling). Weights are set by "
        "residual IC over residual volatility — never fitted to returns."
    )

    c = st.columns(4)
    horizon = c[0].slider("Horizon", 5, 30, 15, key="cb_h")
    years = c[1].select_slider("Period", ["2y", "5y", "10y"], value="5y", key="cb_y")
    min_t = c[2].select_slider("Min |t|", [1.5, 2.0, 2.5, 3.0], value=2.0, key="cb_t",
                               help="3.0 is the Harvey-Liu-Zhu bar for a new factor claim.")
    max_corr = c[3].slider("Max correlation", 0.3, 0.9, 0.6, 0.05, key="cb_c")

    if not st.button("Build composite", key="cb_run"):
        st.info("Runs the signal lab, computes the pairwise correlation matrix, then "
                "assembles a composite from whatever survives. 5–8 minutes.")
        return

    tickers = tuple(cfg["tickers"][:100])
    with st.spinner("Fetching data…"):
        frames = fetch_history(tickers, period=years)
        bench = fetch_history((config.BENCHMARK,), period=years).get(config.BENCHMARK)
    if not frames:
        st.error("No data returned.")
        return

    with st.spinner("Testing signals…"):
        lab = sig_lab.run(frames, bench, horizon=horizon)
    if lab.table.empty:
        st.error("Signal lab produced no results.")
        return

    with st.spinner("Computing correlation matrix…"):
        corr = comp.correlation_matrix(frames, bench, step=horizon)

    spec = comp.build(lab.table, corr, min_t=min_t, max_correlation=max_corr)

    if spec.is_empty:
        st.error(f"**No composite could be built.** {spec.diagnostics.get('error', '')}")
        if spec.rejected:
            with st.expander(f"Why each of {len(spec.rejected)} signals was rejected"):
                for n, r in spec.rejected:
                    st.markdown(f"- **{n}** — {r}")
        st.info(
            "This is a legitimate outcome, not a failure of the tool. A composite built "
            "from components with no demonstrated incremental content is exactly what v1 "
            "was.", icon="ℹ️",
        )
        return

    st.success(f"**Composite:** {spec.describe()}")

    m = st.columns(4)
    m[0].metric("Components", len(spec.components))
    m[1].metric("Qualified", spec.diagnostics.get("n_qualified", 0))
    m[2].metric("Rejected", spec.diagnostics.get("n_rejected", 0))
    m[3].metric("Max pairwise corr", f"{spec.diagnostics.get('max_pairwise_corr', 0):.2f}")

    st.markdown("##### Component weights")
    st.dataframe(pd.DataFrame([{"component": k, "weight": v}
                               for k, v in spec.weights.items()]),
                 hide_index=True, use_container_width=True)

    if spec.correlations is not None:
        st.markdown("##### Correlation between selected components")
        st.caption("All off-diagonal values should sit below your ceiling. "
                   "This is the check that v1 never had.")
        st.dataframe(spec.correlations, use_container_width=True)

    with st.expander(f"Rejected signals ({len(spec.rejected)})"):
        for n, r in spec.rejected:
            st.markdown(f"- **{n}** — {r}")

    with st.spinner("Validating the assembled composite…"):
        v = comp.validate(spec, frames, bench, horizon=horizon)

    st.markdown("##### Composite performance")
    if "error" in v:
        st.warning(v["error"])
    else:
        k = st.columns(3)
        k[0].metric("Composite IC", f"{v['composite_ic']:+.4f}")
        k[1].metric("Newey-West t", f"{v['t_newey_west']}")
        k[2].metric("Positive windows", f"{v['pct_positive']}%")
        st.warning(v["caveat"], icon="⚠️")

    st.markdown("##### Full correlation matrix")
    st.caption("The diagnostic that would have caught the v1 failure before deployment.")
    if not corr.empty:
        st.dataframe(corr, use_container_width=True)


# --------------------------------------------------------------------------
# Data quality (inside Validation tab)
# --------------------------------------------------------------------------
def render_data_quality(cfg: dict) -> None:
    st.subheader("Data quality & survivorship")
    st.caption(
        "yfinance carries known defects — bad split adjustments, stale prices, "
        "zero-volume gaps. The scoring code cannot see them and will happily rank a "
        "stock on a 400% move that was a mis-applied corporate action."
    )

    years = st.select_slider("Period to audit", ["1y", "2y", "5y"], value="2y", key="dq_y")

    if not st.button("Run audit", key="dq_run"):
        st.info("Checks every ticker in the current universe for data defects.")
        return

    tickers = tuple(cfg["tickers"][:150])
    with st.spinner(f"Auditing {len(tickers)} tickers…"):
        frames = fetch_history(tickers, period=years)
    if not frames:
        st.error("No data returned.")
        return

    res = dq.audit(frames)
    m = st.columns(3)
    m[0].metric("Tickers", res.stats["total"])
    m[1].metric("Clean", res.stats["clean"])
    m[2].metric("Flagged", res.stats["flagged"], f"{100 - res.stats['clean_pct']:.0f}%")

    if res.stats["clean_pct"] < 70:
        st.warning(
            f"Only {res.stats['clean_pct']}% of the universe is clean. Defects at this "
            "rate meaningfully distort backtests — a single bad tick can dominate a "
            "window's cross-section."
        )
    else:
        st.success(f"{res.stats['clean_pct']}% clean.")

    if res.flagged:
        st.markdown("##### Flagged tickers")
        st.dataframe(res.summary, hide_index=True, use_container_width=True)

        cleaned, removed = dq.clean(frames)
        st.caption(
            f"Repair would keep {len(cleaned)} tickers (extreme moves forward-filled) "
            f"and drop {len(removed)} as unsalvageable."
        )

    st.markdown("##### Point-in-time universe coverage")
    st.caption(
        "Membership decided by traded value **as of each date**, not by today's index. "
        "Churn is a good sign — it means the universe is genuinely being reconstructed "
        "rather than inherited."
    )
    with st.spinner("Building coverage report…"):
        cov = dq.pit_coverage_report(frames, step=63)
    if not cov.empty:
        st.dataframe(cov, hide_index=True, use_container_width=True)
        st.line_chart(cov.set_index("date")["n_eligible"])
        st.metric("Mean quarterly churn", f"{cov['churn_pct'].mean():.1f}%")

    st.error(dq.bias_note(frames), icon="⚠️")


# --------------------------------------------------------------------------
# Forward log tab — the only evidence that cannot be overfitted
# --------------------------------------------------------------------------
def render_forward_log(cfg: dict, shortlist: pd.DataFrame) -> None:
    st.subheader("Forward paper-trading log")
    st.caption(
        "A backtest can be tuned until it flatters you. A forward log cannot — the picks "
        "are committed before the outcome exists. Eight weeks of this is worth more than "
        "any amount of historical optimisation."
    )

    if "fwd_log" not in st.session_state:
        st.session_state.fwd_log = flog.empty_log()

    up = st.file_uploader("Load existing log (CSV)", type="csv")
    if up is not None and st.button("Load"):
        st.session_state.fwd_log = flog.from_csv(up.read())
        st.success(f"Loaded {len(st.session_state.fwd_log)} rows.")

    log = st.session_state.fwd_log

    st.markdown("##### 1. Record this week's picks")
    c = st.columns(3)
    top_n = c[0].slider("Picks to record", 5, 20, 10)
    horizon = c[1].slider("Horizon (trading days)", 10, 25, 15)
    snap_date = c[2].date_input("Snapshot date", dt.date.today())

    if shortlist is None or shortlist.empty:
        st.info("Run the Screener first — it supplies the picks to record.")
    else:
        st.caption(f"Screener currently has {len(shortlist)} names passing filters.")
        if st.button("📸 Snapshot top picks", type="primary"):
            reg = st.session_state.get("regime")
            st.session_state.fwd_log = flog.record_snapshot(
                log, shortlist, regime_state=(reg.state if reg else "unknown"),
                horizon=horizon, top_n=top_n, snapshot_date=snap_date,
            )
            log = st.session_state.fwd_log
            st.success(f"Recorded. Log now has {len(log)} entries. Download it before closing.")

    st.markdown("##### 2. Evaluate matured picks")
    if log.empty:
        st.info("Nothing logged yet.")
    else:
        n_open = int((log["status"] == "open").sum())
        st.caption(f"{n_open} picks still open, {int((log['status']=='evaluated').sum())} evaluated.")
        if n_open and st.button("Fetch prices and evaluate"):
            tickers = tuple(f"{t}.NS" for t in log.loc[log["status"] == "open", "ticker"].unique())
            with st.spinner(f"Fetching {len(tickers)} tickers…"):
                frames = fetch_history(tickers, period="6mo")
            lookup = {t.replace(".NS", ""): float(df["Close"].iloc[-1])
                      for t, df in frames.items() if df is not None and len(df)}
            st.session_state.fwd_log, filled = flog.evaluate_open(log, lookup)
            log = st.session_state.fwd_log
            st.success(f"Evaluated {filled} picks.") if filled else st.info(
                "None have reached their evaluation date yet."
            )

    st.markdown("##### 3. Results")
    res = flog.analyse(log)
    if "error" in res:
        st.info(res["error"])
    else:
        m = st.columns(5)
        m[0].metric("Evaluated", res["evaluated_picks"])
        m[1].metric("Mean return", f"{res['mean_return_pct']:+.2f}%")
        m[2].metric("Hit rate", f"{res['hit_rate_pct']}%")
        m[3].metric("Forward IC", f"{res['forward_ic']}" if res["forward_ic"] is not None else "—")
        m[4].metric("Snapshots", res["snapshots"])

        if res["snapshots"] < 6:
            st.warning(
                f"Only {res['snapshots']} snapshots. Wide error bars — keep logging weekly. "
                "Draw conclusions at 8+."
            )

        if res.get("mean_excess_pct") is not None:
            st.caption(
                f"Mean excess vs benchmark: {res['mean_excess_pct']:+.2f}% · "
                f"beat benchmark {res['beat_bench_pct']}% of the time"
            )

        st.markdown("**By score bucket** — the honest test. Does return rise with score?")
        st.dataframe(flog.by_dimension(log, "score_bucket"), hide_index=True,
                     use_container_width=True)
        st.markdown("**By tier**")
        st.dataframe(flog.by_dimension(log, "tier"), hide_index=True, use_container_width=True)
        st.markdown("**By regime**")
        st.dataframe(flog.by_dimension(log, "regime"), hide_index=True, use_container_width=True)

        st.markdown("##### 4. Forward vs backtest")
        bt_ic = st.number_input(
            "Backtest IC from the Validation tab", value=0.0, step=0.005, format="%.4f",
            help="Paste the mean IC the Validation tab reported. This comparison is the "
                 "single most informative number in the whole app.",
        )
        if bt_ic:
            st.info(flog.compare_to_backtest(res["forward_ic"], bt_ic))

    if not log.empty:
        st.markdown("##### 5. Export")
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Log (CSV)", flog.to_csv(log),
                           file_name=f"forward_log_{dt.date.today()}.csv",
                           mime="text/csv", use_container_width=True)

        reg = st.session_state.get("regime")
        html_text = rep.build_html(
            generated=dt.datetime.now(),
            universe_name=cfg.get("universe_name", "—"),
            universe_live=bool(cfg.get("universe_result") and cfg["universe_result"].is_live),
            n_tickers=len(cfg.get("tickers", ())),
            regime_state=(reg.state if reg else "neutral"),
            regime_desc=(reg.description if reg else ""),
            regime_pct=(reg.pct_from_200dma if reg else 0.0),
            breadth=(reg.breadth if reg else None),
            picks=(shortlist.head(10).reset_index(drop=True).assign(
                       Rank=range(1, min(10, len(shortlist)) + 1))
                   if shortlist is not None and not shortlist.empty else None),
            forward_summary=res if "error" not in res else None,
            bucket_table=flog.by_dimension(log, "score_bucket"),
            tier_table=flog.by_dimension(log, "tier"),
        )
        d2.download_button("⬇ Report (HTML)", html_text.encode(),
                           file_name=f"swingscope_report_{dt.date.today()}.html",
                           mime="text/html", use_container_width=True)
        st.caption(
            "The HTML report is self-contained — open it in any browser or forward it by "
            "email. Reports generated here are **not** saved anywhere: Streamlit's "
            "filesystem resets, so download it now if you want to keep it."
        )

        links = config.report_links()
        if links:
            st.markdown("##### 6. Automated reports")
            st.caption("Produced weekly by GitHub Actions and stored in your repo.")
            b = st.columns(4)
            if "weekly_html" in links:
                b[0].link_button("📈 Weekly", links["weekly_html"],
                                 use_container_width=True)
                b[1].link_button("📊 Month-end", links["monthly_html"],
                                 use_container_width=True)
            else:
                b[0].link_button("📈 Weekly PDF", links["weekly_pdf"],
                                 use_container_width=True)
                b[1].link_button("📊 Month-end PDF", links["monthly_pdf"],
                                 use_container_width=True)
            b[2].link_button("📁 Archive", links["folder"], use_container_width=True)
            b[3].link_button("⚙️ Runs", links["actions"], use_container_width=True)
            st.caption(
                f"Committed forward log: [forward_log.csv]({links['log_csv']}) — "
                "this is the persistent copy; the in-app log resets on restart."
            )
        else:
            st.info(
                "**Want one-click access to your automated reports?** Set `GITHUB_USER` "
                "in `config.py`, or add this to Streamlit secrets:\n\n"
                "```toml\n[reports]\ngithub_user = \"yourname\"\n"
                "github_repo = \"swingscope\"\npages_enabled = true\n```",
                icon="🔗",
            )
        st.caption(
            "**Download every session.** The log lives in session state and is lost on restart. "
            "For permanent storage, wire flog.to_csv/from_csv to Google Sheets."
        )
        with st.expander("Full log"):
            st.dataframe(log, hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------
# Validation tab — does the score rank forward returns?
# --------------------------------------------------------------------------
def render_validation(cfg: dict) -> None:
    st.subheader("Predictive validation (Information Coefficient)")
    st.caption(
        "Different question from the backtest. This asks: **does a higher score correspond "
        "to a higher forward return?** At each rebalance date every stock is scored using only "
        "data available then, and the rank correlation between score and the next N days' "
        "return is measured. That correlation is the Information Coefficient."
    )

    c = st.columns(4)
    horizon = c[0].slider("Forward horizon (days)", 5, 30, 15)
    years = c[1].select_slider("Test period", ["1y", "2y", "5y", "10y"], value="5y")
    n_stocks = c[2].slider("Stocks", 15, 100, 40, 5)
    n_perm = c[3].select_slider("Permutations", [100, 200, 400, 800], value=400)

    val_model = st.selectbox(
        "Model to validate", ["momentum", "composite_v1"], index=0, key="val_model",
        help="Validate what you actually trade. composite_v1 failed neutralisation "
             "(residual IC +0.004, t = 0.17).",
    )
    overlap = st.checkbox(
        "Use overlapping windows", value=False,
        help="Overlapping windows give more data points but correlated ones, which inflates "
             "the t-statistic. Leave off for an honest significance test.",
    )

    with st.expander("How to read the result"):
        st.markdown(
            "| Mean IC | Interpretation |\n|---|---|\n"
            "| < 0.02 | No meaningful signal |\n"
            "| 0.02 – 0.04 | Weak, maybe tradeable after costs |\n"
            "| 0.04 – 0.06 | Good for a retail-accessible factor |\n"
            "| 0.06 – 0.10 | Strong |\n"
            "| > 0.15 | Suspicious — check for lookahead bias |\n\n"
            "**The t-statistic matters as much as the mean.** IC of 0.05 over 20 windows is "
            "noise; IC of 0.03 over 100 windows with t > 2 is a finding.\n\n"
            "**The permutation p-value is the honest check.** The same pipeline is re-run with "
            "scores randomly shuffled. If your real result sits inside that null distribution, "
            "you have found nothing — however good the headline number looks."
        )

    if not st.button("Run validation", type="primary"):
        st.info("This takes 1–3 minutes for 40 stocks over 5 years.")
        return

    tickers = tuple(cfg["tickers"][:n_stocks])
    with st.spinner(f"Fetching {years} history for {len(tickers)} tickers…"):
        frames = fetch_history(tickers, period=years)
        bench = fetch_history((config.BENCHMARK,), period=years).get(config.BENCHMARK)

    if not frames:
        st.error("No data returned — yfinance may be rate-limiting. Wait and retry.")
        return

    with st.spinner("Walking forward and permuting…"):
        res = val.run_ic(frames, bench, horizon=horizon,
                         step=max(1, horizon // 3) if overlap else horizon,
                         n_permutations=n_perm, model=val_model)

    if "error" in res.summary:
        st.error(f"Could not run: {res.summary['error']}")
        return

    s_ = res.summary
    m = st.columns(5)
    m[0].metric("Mean IC", f"{s_['mean_ic']:+.4f}")
    m[1].metric("t-statistic", f"{s_['t_stat']}")
    m[2].metric("Windows", s_["windows"])
    m[3].metric("Positive IC", f"{s_['pct_positive_ic']}%")
    m[4].metric("Permutation p", f"{s_['permutation_p_value']}")

    level, message = val.verdict(s_)
    {"good": st.success, "warn": st.warning, "bad": st.error, "error": st.error}[level](message)

    st.markdown("##### Forward return by score quintile")
    st.caption(
        "The clearest read in the whole app. Q5 is the highest-scoring fifth. "
        "If mean return does not climb from Q1 to Q5, the score is not ranking anything."
    )
    st.dataframe(res.buckets, hide_index=True, use_container_width=True)
    st.bar_chart(res.buckets.set_index("quintile")["mean"])

    if s_["quintiles_monotonic"]:
        st.success("Quintiles are monotonic — return rises consistently with score.")
    else:
        st.warning(
            "Quintiles are NOT monotonic. The score may separate extremes without "
            "ranking cleanly in between."
        )

    st.markdown("##### IC over time")
    st.caption("Consistency matters more than any single window. Look for persistence, not spikes.")
    st.line_chart(res.windows.set_index("date")["ic"])

    st.markdown("##### Top-minus-bottom quintile spread, by window")
    st.line_chart(res.windows.set_index("date")["spread"])

    if res.null_ic is not None and len(res.null_ic):
        st.markdown("##### Permutation null distribution")
        st.caption(
            f"Mean IC from {len(res.null_ic)} runs with shuffled scores. Your actual result "
            f"({s_['mean_ic']:+.4f}) sits {s_.get('z_vs_null')} standard deviations from this null."
        )
        st.bar_chart(pd.Series(res.null_ic).round(3).value_counts().sort_index())

    with st.expander("Per-window detail"):
        st.dataframe(res.windows, hide_index=True, use_container_width=True)
    st.download_button("Download windows (CSV)", res.windows.to_csv(index=False).encode(),
                       file_name="ic_windows.csv", mime="text/csv")

    st.info(
        "**Two biases inflate these numbers.** The universe is today's index members, so "
        "companies that failed and were removed are invisible (survivorship bias). And Indian "
        "equities trended up through most of the last five years, which flatters any long-only "
        "model. Treat a good result as necessary evidence, not sufficient.",
        icon="⚠️",
    )


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

    tabs = st.tabs(["🔍 Screener", "📊 Detail", "🧪 Backtest", "🔬 Validation", "📋 Forward log", "📰 News", "📓 Journal", "❓ Method"])

    with tabs[0]:
        shortlist = render_screener(cfg)
    with tabs[1]:
        render_detail(cfg, shortlist)
    with tabs[2]:
        render_backtest(cfg)
    with tabs[3]:
        render_validation(cfg)
        st.divider()
        render_factor_analysis(cfg)
        st.divider()
        render_signal_lab(cfg)
        st.divider()
        render_composite_builder(cfg)
        st.divider()
        render_data_quality(cfg)
    with tabs[4]:
        render_forward_log(cfg, shortlist)
    with tabs[5]:
        render_news(cfg, shortlist)
    with tabs[6]:
        render_journal()
    with tabs[7]:
        st.markdown(config.METHOD_DOC)


if __name__ == "__main__":
    main()
