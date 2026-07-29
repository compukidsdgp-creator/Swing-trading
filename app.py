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
import portfolio as pfl


# --------------------------------------------------------------------------
# Optional modules.
#
# These power research and broker features that are not required for the core
# app. Importing them at module scope means one missing file takes down the
# whole deployment — which is exactly what happened when broker.py was absent.
# Each is now optional; its section degrades to a clear message instead.
# --------------------------------------------------------------------------
def _optional(name):
    try:
        return __import__(name)
    except Exception:
        return None


brk = _optional("broker")
oc = _optional("outcomes")
bhav = _optional("bhavcopy")
costs = _optional("costs")
hz = _optional("horizon")

_MISSING = [n for n, m in (("broker", brk), ("bhavcopy", bhav),
                           ("costs", costs), ("horizon", hz),
                           ("outcomes", oc)) if m is None]


def _unavailable(module, purpose):
    st.warning(
        f"**`{module}.py` is not present in this deployment.** {purpose}\n\n"
        "Upload the file to your repository to enable this section. The rest of "
        "the app is unaffected.",
        icon="📦",
    )

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
        df.columns = [str(c).title() for c in df.columns]
        needed = {"Open", "High", "Low", "Close", "Volume"}
        if not needed.issubset(set(df.columns)):
            continue

        # Drop partial bars — yfinance returns the current session with Close
        # still NaN during market hours, which makes every last-row read NaN
        # and silently empties the screener.
        df = df.dropna(subset=["Close"])
        if df.empty or len(df) < 60:
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
        "Cap universe size", 20, 500, min(400, max(20, len(tickers))), 10,
        help="Screen broadly, select narrowly. Gross spread measured 0.22% "
             "across 150 long-history symbols against 0.87% across 400 — "
             "filtering for long history selects large, efficiently-priced "
             "companies where momentum works least well. Note the free "
             "Streamlit tier slows noticeably past ~150 on a cold start.",
    )
    trim_method = st.sidebar.selectbox(
        "How to trim", ["liquidity", "random"], index=0,
        help="NSE returns constituents alphabetically. Taking the first N would "
             "keep only A-G — a pure artefact. Liquidity keeps the most tradeable "
             "names; random gives an unbiased sample.",
    )
    if len(tickers) > max_scan:
        if trim_method == "random":
            trimmed, how = uni.trim_universe(tuple(tickers), max_scan, method="random")
        else:
            # Liquidity trim needs price data, so defer to the screener; here we
            # note the intent and pass the full list through.
            trimmed, how = tuple(tickers), "liquidity trim applied after fetch"
        tickers = list(trimmed)
        st.sidebar.caption(how)

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
        exclude_small = st.sidebar.checkbox(
            "Exclude small caps", value=True,
            help="0.96% round-trip cost against a ~2% long-only edge. The "
                 "automated pipeline excludes them; matching that keeps the "
                 "app and the Monday bucket consistent.",
        )
        apply_rsi_band = st.sidebar.checkbox(
            "Apply RSI band (not validated)", value=False,
            help="Leftover from the v1 composite. The 20-year validation used "
                 "momentum alone — no RSI filter. Switching this on screens on "
                 "something untested and will make the app disagree with the "
                 "Monday bucket.",
        )
    else:
        min_score = st.sidebar.slider("Min composite score", 0, 100, 55)
        require_positive_mom = False
        exclude_small = False
        apply_rsi_band = True

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
        exclude_small=exclude_small,
        apply_rsi_band=apply_rsi_band,
        max_scan=max_scan,
        trim_method=trim_method,
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

    # Momentum needs 252 bars of formation window plus buffer; a 1y fetch
    # returns only ~250 NSE trading days, which silently rejects every ticker.
    period = "2y" if cfg.get("model") == "momentum" else "1y"
    fetch_list = tuple(cfg["tickers"][: max(cfg.get("max_scan", 120) * 3, 60)])
    with st.spinner(f"Fetching {len(fetch_list)} tickers ({period})…"):
        data_all = fetch_history(fetch_list, period=period)
        bench = fetch_history((config.BENCHMARK,), period=period)

    # Trim AFTER fetching, by liquidity — never positionally. NSE returns
    # constituents alphabetically, so taking the first N kept only A-G.
    if data_all and len(data_all) > cfg.get("max_scan", 120):
        keep, how = uni.trim_universe(
            tuple(data_all), cfg.get("max_scan", 120),
            method=cfg.get("trim_method", "liquidity"), frames=data_all,
        )
        data = {t: data_all[t] for t in keep if t in data_all}
        st.caption(f"Universe: {len(data)} of {len(data_all)} fetched — {how}")
    else:
        data = data_all

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
            lens = [len(d) for d in data.values() if d is not None]
            shortest, longest = (min(lens), max(lens)) if lens else (0, 0)
            st.error(
                "**No tickers produced valid momentum values.**\n\n"
                f"Fetched {len(data)} tickers with {shortest}–{longest} bars of history. "
                f"12-1 momentum needs at least {momo.LOOKBACK + 5}.\n\n"
                "If history looks sufficient, the absolute momentum floor is the likely "
                "cause — in a broadly falling market no stock has positive 12-month "
                "momentum, and zero picks is the correct answer. Untick "
                "**Require positive 12-1 momentum** in the sidebar to see the ranking "
                "anyway, understanding that it will promote the best of a bad set."
            )
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

    # Each filter evaluated separately, so it is visible which one is binding.
    # A chain of individually reasonable filters can multiply out to zero, and
    # "nothing passed" without attribution leaves you guessing.
    # Filter chain.
    #
    # IMPORTANT: the RSI band is a leftover from the v1 composite. The 20-year
    # validation ranked on momentum alone — no RSI filter was ever part of it.
    # Applying it here means screening on something untested, and it made the
    # app's picks diverge from the automated pipeline's.
    #
    # In momentum mode it is therefore off by default, and clearly labelled as
    # unvalidated when switched on.
    momentum_mode = cfg.get("model") == "momentum"

    checks = {
        f"turnover ≥ ₹{cfg['min_turnover']:.0f} Cr": res["Turnover_Cr"] >= cfg["min_turnover"],
        f"score ≥ {cfg['min_score']}": res["Score"] >= cfg["min_score"],
    }
    if not momentum_mode or cfg.get("apply_rsi_band", False):
        checks[f"RSI in {lo}–{hi} (not validated)" if momentum_mode
               else f"RSI in {lo}–{hi}"] = res["RSI"].between(lo, hi)
    if cfg["require_uptrend"]:
        checks["price > 50 EMA"] = res["Above_50EMA"]
    if respect_regime:
        checks[f"tier in {sorted(reg.allowed_tiers)}"] = res["Tier"].isin(reg.allowed_tiers)
    if momentum_mode and cfg.get("exclude_small", True):
        checks["cost-viable tier (large/mid)"] = res["Tier"].isin(momo.RECOMMENDED_TIERS)

    mask = pd.Series(True, index=res.index)
    for m in checks.values():
        mask &= m

    filtered = res[mask].sort_values("Score", ascending=False).reset_index(drop=True)

    # Attribution: how many survive each filter alone, and cumulatively
    if filtered.empty and not res.empty:
        rows, running = [], pd.Series(True, index=res.index)
        for label, m in checks.items():
            running = running & m
            rows.append({
                "filter": label,
                "passes_alone": int(m.sum()),
                "pct_alone": round(m.mean() * 100, 1),
                "cumulative": int(running.sum()),
            })
        attrib = pd.DataFrame(rows)

        st.error(
            f"**Nothing passed.** {len(res)} stocks scored, 0 survived the filter "
            "chain. The table below shows where they were lost."
        )
        st.dataframe(attrib, hide_index=True, use_container_width=True)

        # Identify the binding constraint: the step that removed the most
        binding = None
        prev = len(res)
        worst_drop = -1
        for _, r in attrib.iterrows():
            drop = prev - r["cumulative"]
            if drop > worst_drop:
                worst_drop, binding = drop, r["filter"]
            prev = r["cumulative"]

        strictest = attrib.loc[attrib["passes_alone"].idxmin()]
        st.warning(
            f"**Binding constraint: `{binding}`** — it removed {worst_drop} stocks.\n\n"
            f"The strictest filter overall is `{strictest['filter']}`, which only "
            f"{strictest['passes_alone']} of {len(res)} stocks "
            f"({strictest['pct_alone']}%) pass on its own."
        )

        if respect_regime and "tier" in str(binding):
            st.info(
                f"The regime is **{reg.state.replace('_', ' ')}**, which permits only "
                f"{', '.join(sorted(reg.allowed_tiers))} caps. In a hostile regime, "
                "zero picks is the intended output — not a fault to be filtered "
                "around.", icon="🛡️",
            )
        else:
            st.info(
                "Each filter is individually reasonable; they multiply out to zero "
                "together. Relax the binding one first rather than loosening "
                "everything at once.", icon="💡",
            )
        return pd.DataFrame()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Scanned", len(res))
    c2.metric("Passed filters", len(filtered))

    # Selectivity — the concentration that made the validated result work.
    # The 20-year test measured a 1.63% gross spread taking the top 5% of a
    # large ranked set against 0.56% for the top 20%. Filters that shrink the
    # universe before selection quietly convert the former into the latter, and
    # nothing on screen showed it.
    bucket_size = st.session_state.get("bucket_size", 10)
    if len(filtered):
        sel = bucket_size / len(filtered)
        c3.metric(
            "Selectivity", f"top {sel:.0%}",
            help=f"Taking {bucket_size} of {len(filtered)}. The validated "
                 "configuration was the top 5% — concentration is where the "
                 "edge lives.",
        )
    else:
        c3.metric("Selectivity", "—")

    c4.metric("Median score", f"{res['Score'].median():.0f}")
    if bench_df is not None and len(bench_df) > 20:
        b_ret = (bench_df["Close"].iloc[-1] / bench_df["Close"].iloc[-21] - 1) * 100
        c5.metric("Nifty 20d", f"{b_ret:+.1f}%")

    if momentum_mode:
        st.caption(
            "**This screen is not identical to the Monday bucket.** The automated "
            "pipeline additionally applies a macro overlay on the regime, a hard "
            "earnings-window exclusion, an exit-liquidity check, and volatility "
            "scaling. Expect the Monday picks to be a subset of what you see here."
        )

    if len(filtered):
        sel = bucket_size / len(filtered)
        if sel > 0.15:
            st.warning(
                f"**Selectivity is top {sel:.0%}, against the top 5% that was "
                f"validated.** Filters cut {len(res)} candidates to "
                f"{len(filtered)} before selection, so taking {bucket_size} is "
                "closer to the quintile configuration (0.56% gross spread) than "
                "the concentrated one (1.63%). Today's bucket is not the "
                "configuration the +5.95% figure came from.\n\n"
                "Raise the universe cap, or relax the filters so more names "
                "reach the ranking stage. In a weak regime few stocks have "
                "positive 12-month momentum, and this may simply be all that "
                "qualifies — in which case conditions do not support the "
                "strategy today.",
                icon="🎯",
            )
        elif sel <= 0.05:
            st.success(
                f"Selectivity top {sel:.0%} — at or better than the validated "
                "configuration.")

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
        st.session_state["bucket_size"] = b_size
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
# Outcomes tab — conditional distributions, not predictions
# --------------------------------------------------------------------------
def render_outcomes(cfg: dict) -> None:
    if oc is None:
        st.subheader("Outcomes")
        _unavailable("outcomes", "It shows historical outcome distributions "
                                 "conditional on momentum decile.")
        return

    st.subheader("Outcomes: what happened to similar cases")
    st.caption(
        "The closest thing to prediction this data honestly supports. Not "
        "'this stock will rise 8%' — rather, 'of stocks that historically "
        "scored in this decile, here is the full range of what happened next'."
    )

    st.warning(
        "**This is not a prediction tab, and the distinction is not pedantry.** "
        "With an IC of 0.031 the signal explains roughly 0.1% of the variance in "
        "forward returns. A stated probability would carry a confidence interval "
        "spanning the coin flip. What follows are historical frequencies with "
        "their intervals shown — read the interval, not the point estimate.",
        icon="⚠️",
    )

    c = st.columns(3)
    horizon = c[0].slider("Horizon (days)", 5, 90, 30, key="oc_h")
    years = c[1].select_slider("Period", ["2y", "5y", "10y"], value="5y", key="oc_y")
    n_stocks = c[2].slider("Stocks", 50, 300, 150, 25, key="oc_n")

    if not st.button("Build outcome distributions", key="oc_run"):
        st.info("Takes 3–6 minutes. Uses only data available at each historical "
                "observation point — the same no-lookahead discipline as validation.")
        return

    tickers = tuple(cfg["tickers"][:n_stocks])
    with st.spinner(f"Fetching {years} for {len(tickers)} tickers…"):
        frames = fetch_history(tickers, period=years)
        bench = fetch_history((config.BENCHMARK,), period=years).get(config.BENCHMARK)
    if not frames:
        st.error("No data returned.")
        return

    with st.spinner("Building distributions…"):
        res = oc.build_distributions(frames, bench, horizon=horizon)

    if res.by_decile.empty:
        st.error("; ".join(res.notes) or "Could not build distributions.")
        return

    m = st.columns(4)
    m[0].metric("Windows", res.n_windows)
    m[1].metric("Observations", f"{len(res.raw_outcomes):,}")
    m[2].metric("Universe mean", f"{res.universe_mean:+.2f}%")
    m[3].metric("Universe hit rate", f"{res.universe_hit_rate:.1f}%")

    for n in res.notes:
        st.caption(f"· {n}")

    st.markdown("##### Outcomes by momentum decile")
    st.caption(
        "`excess_pct` is the column that matters — raw returns mostly reflect "
        "market drift. Where the excess interval spans zero, no edge is "
        "demonstrated at this sample size."
    )
    show = res.by_decile[["decile", "n", "mean_return_pct", "hit_rate_pct",
                          "hit_ci_low", "hit_ci_high", "excess_pct",
                          "excess_ci_low", "excess_ci_high", "p10", "p90",
                          "overlap_with_d1"]]
    st.dataframe(
        show, hide_index=True, use_container_width=True,
        column_config={
            "excess_pct": st.column_config.NumberColumn("Excess %", format="%+.3f"),
            "hit_rate_pct": st.column_config.NumberColumn("Hit %", format="%.1f"),
            "overlap_with_d1": st.column_config.NumberColumn(
                "Overlap w/ D1", format="%.2f",
                help="1.00 means the outcome distributions are identical — the "
                     "signal separates nothing."),
        },
    )

    st.markdown("##### Excess return by decile")
    st.bar_chart(res.by_decile.set_index("decile")["excess_pct"])

    st.markdown("##### The overlap problem")
    top = res.by_decile.iloc[-1]
    ov = top["overlap_with_d1"]
    st.metric("Top vs bottom decile distribution overlap", f"{ov:.0%}")
    if ov > 0.8:
        st.error(
            f"**{ov:.0%} overlap.** The best and worst deciles produce almost "
            "the same distribution of outcomes. Momentum shifts the odds "
            "slightly across many trades; it does not identify winners. Any "
            "single pick is close to a coin flip, and no amount of conviction "
            "changes that.", icon="🎲",
        )

    st.markdown("##### Look up a score")
    score = st.slider("Percentile score", 0, 100, 90, key="oc_s")
    o = oc.outcome_for_score(res, score)
    if "error" in o:
        st.warning(o["error"])
    else:
        k = st.columns(3)
        k[0].metric("Rose", f"{o['rose_pct_of_time']}%",
                    help=f"Range: {o['hit_rate_range']}")
        k[1].metric("Median outcome", f"{o['median_outcome_pct']:+.2f}%")
        k[2].metric("Excess over universe", f"{o['excess_over_universe_pct']:+.2f}%",
                    help=f"Range: {o['excess_range']}")
        st.caption(f"Typical range (middle 50%): {o['typical_range_pct']} · "
                   f"middle 80%: {o['worst_decile_10pct']}% to "
                   f"{o['best_decile_10pct']}%")
        st.info(o["caveat"], icon="📊")

    st.markdown("##### Calibration")
    cal = oc.calibration_check(res)
    if not cal.empty:
        st.dataframe(cal, hide_index=True, use_container_width=True)
        st.caption(cal.iloc[0]["interpretation"])


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
            hist = {t.replace(".NS", ""): df for t, df in frames.items()
                    if df is not None and len(df)}
            st.session_state.fwd_log, filled = flog.evaluate_open(
                log, lookup, price_history=hist)
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
# Bhavcopy — free point-in-time data
# --------------------------------------------------------------------------
def render_bhavcopy() -> None:
    if bhav is None:
        st.subheader("Bhavcopy")
        _unavailable("bhavcopy", "It provides free point-in-time NSE data going back to 1994.")
        return

    st.subheader("NSE bhavcopy — free point-in-time data")
    st.caption(
        "I previously called survivorship bias, history depth and delivery data "
        "'money problems'. That was wrong. NSE publishes a complete daily snapshot of "
        "every traded security, archived back to 1994 — including companies later "
        "delisted. Building the universe from bhavcopies is genuinely point-in-time."
    )

    st.success(
        "**Three gaps this closes at zero cost.**\n\n"
        "**Survivorship** — every security that traded that day, not just today's "
        "index members.\n\n"
        "**History** — archives to 1994, versus five years from yfinance.\n\n"
        "**Delivery percentage** — the share of volume that resulted in actual "
        "delivery rather than intraday squaring. Not available in yfinance at all, "
        "and a genuine signal input.",
        icon="🔓",
    )

    stats = bhav.cache_stats()
    m = st.columns(4)
    m[0].metric("Days cached", stats["days"])
    m[1].metric("Cache size", f"{stats['size_mb']} MB")
    m[2].metric("Earliest", stats["earliest"] or "—")
    m[3].metric("Latest", stats["latest"] or "—")

    st.markdown("##### Download history")
    c = st.columns(3)
    start = c[0].date_input("From", dt.date.today() - dt.timedelta(days=90), key="bh_s")
    end = c[1].date_input("To", dt.date.today(), key="bh_e")
    max_days = c[2].slider("Max sessions", 20, 400, 60, 20, key="bh_n")

    st.caption(
        "Downloads prefer a community GitHub mirror rather than hammering NSE, which "
        "blacklists aggressive crawlers. Each date is fetched once and cached locally. "
        "Please keep batches modest — the mirror is someone's unpaid work."
    )

    if st.button("Download bhavcopies", key="bh_run"):
        bar = st.progress(0.0)
        status = st.empty()

        def _prog(i, total, date, source):
            bar.progress(i / total)
            status.caption(f"{i}/{total} · {date:%d %b %Y} · {source}")

        with st.spinner("Fetching…"):
            hist = bhav.fetch_range(str(start), str(end),
                                    max_days=max_days, progress=_prog)
        bar.empty(); status.empty()

        if hist.empty:
            st.error(
                "No data retrieved. The mirror may not carry these dates, or NSE may "
                "be blocking. Try a different range, or fewer days."
            )
            return

        st.session_state["bhav_hist"] = hist
        st.success(f"Retrieved {hist['date'].nunique()} sessions, "
                   f"{len(hist):,} rows, {hist['symbol'].nunique():,} unique symbols.")

    hist = st.session_state.get("bhav_hist")
    if hist is None or hist.empty:
        st.info("Download some history to see the analysis below.")
        return

    latest_date = hist["date"].max()
    latest = hist[hist["date"] == latest_date]

    st.markdown(f"##### Point-in-time universe — {latest_date:%d %b %Y}")
    c2 = st.columns(3)
    min_to = c2[0].number_input("Min turnover (₹ Cr)", 0.0, 500.0, 10.0, 5.0, key="bh_to")
    eq_only = c2[1].checkbox("EQ series only", value=True, key="bh_eq",
                             help="Excludes BE series, which carries ASM/GSM "
                                  "surveillance restrictions and is often untradeable.")
    min_tr = c2[2].number_input("Min trades", 0, 20000, 500, 100, key="bh_tr")

    u = bhav.pit_universe(latest, min_turnover_cr=min_to,
                          eq_only=eq_only, min_trades=min_tr)
    st.metric("Investable that day", f"{len(u):,} of {len(latest):,} traded")
    if not u.empty:
        cols = [c for c in ("symbol", "series", "close", "turnover_cr", "trades",
                            "deliv_pct") if c in u.columns]
        st.dataframe(u[cols].head(50), hide_index=True, use_container_width=True)

    # Churn — the evidence that survivorship is actually being addressed
    dates = sorted(hist["date"].unique())
    if len(dates) > 5:
        step = max(1, len(dates) // 10)
        frames = {d: hist[hist["date"] == d] for d in dates[::step]}
        churn = bhav.universe_churn(frames, min_turnover_cr=min_to,
                                    eq_only=eq_only, min_trades=min_tr)
        if not churn.empty:
            st.markdown("##### Universe churn over time")
            st.caption(
                "Entries and exits are the proof that membership is being decided "
                "at each date rather than inherited from today's index. A static "
                "list would show none."
            )
            st.dataframe(churn, hide_index=True, use_container_width=True)
            st.line_chart(churn.set_index("date")["n"])

    # Delivery signal
    if "deliv_pct" in hist.columns and hist["deliv_pct"].notna().any():
        st.markdown("##### Delivery-based accumulation")
        st.caption(
            "Delivery percentage on up-days minus down-days. Institutions "
            "accumulating take delivery; day-traders square off. Positive values "
            "suggest genuine accumulation. **This data does not exist in yfinance.**"
        )
        sig = bhav.delivery_signal(hist)
        if not sig.empty:
            top = sig.head(25)
            st.dataframe(top, hide_index=True, use_container_width=True)
            st.download_button("Download delivery signal (CSV)",
                               sig.to_csv(index=False).encode(),
                               file_name="delivery_signal.csv", mime="text/csv")
            st.info(
                "**Before using this as a signal, test it.** Add it to the Signal "
                "laboratory and check its residual IC after factor neutralisation. "
                "Novel data is not the same as predictive data — that is exactly the "
                "mistake the v1 composite made.", icon="🧪",
            )
    else:
        st.caption(
            "No delivery data in this range. DELIV_PER appears in the full "
            "`sec_bhavdata_full` files and in post-2024 formats; older archives "
            "may lack it."
        )


# --------------------------------------------------------------------------
# True cost: tax + correlation
# --------------------------------------------------------------------------
def render_true_cost(cfg: dict) -> None:
    if costs is None:
        st.subheader("Costs")
        _unavailable("costs", "It models capital gains tax and correlation-adjusted position sizing.")
        return

    st.subheader("True cost: tax and correlation")
    st.caption(
        "Two costs the model never charged. **Tax:** India levies 20% on "
        "short-term capital gains — every 15–20 day hold is short-term by "
        "definition. **Correlation:** ten momentum names in a rising market "
        "often behave as two or three independent bets, so per-position sizing "
        "understates real risk."
    )

    st.markdown("##### Does the edge survive?")
    c = st.columns(4)
    gross = c[0].number_input("Gross edge %/cycle", 0.0, 5.0, 0.55, 0.05, key="tc_g",
                              help="Top-minus-bottom quintile spread per holding period.")
    charges = c[1].number_input("Charges %", 0.0, 3.0, 0.35, 0.05, key="tc_c")
    win = c[2].slider("Win rate", 0.30, 0.80, 0.55, 0.05, key="tc_w")
    hold = c[3].number_input("Holding days", 1, 400, 15, 1, key="tc_h")

    r = costs.edge_after_tax(gross, win_rate=win, charges_pct=charges,
                             holding_days=hold)
    m = st.columns(5)
    m[0].metric("Gross", f"{r['gross_edge_pct']:.3f}%")
    m[1].metric("Charges", f"−{r['charges_pct']:.3f}%")
    m[2].metric("Tax", f"−{r['tax_pct']:.3f}%", help=f"{r['tax_rate_applied']:.1%} rate")
    m[3].metric("Net", f"{r['net_edge_pct']:+.3f}%")
    m[4].metric("Annualised", f"{r['annualised_net_pct']:+.2f}%")

    if not r["viable"]:
        st.error(
            f"**The edge does not survive.** {gross:.2f}% gross becomes "
            f"{r['net_edge_pct']:+.3f}% after {charges:.2f}% charges and "
            f"{r['tax_pct']:.3f}% tax. At this holding period the strategy loses "
            "money regardless of what the IC says. The fix is a bigger gross edge "
            "or a longer hold — not better ranking."
        )
    else:
        st.success(
            f"Net {r['net_edge_pct']:+.3f}% per cycle, annualising to "
            f"{r['annualised_net_pct']:+.2f}%. {r['retention_pct']:.0f}% of gross survives."
        )

    st.markdown("##### Holding period")
    st.caption(
        "Gross edge scaled by √time — a longer hold captures a larger move. Two "
        "effects favour patience: charges are paid fewer times a year, and past "
        "12 months the rate drops from 20% to 12.5%."
    )
    hz = costs.compare_horizons(gross, base_days=hold, charges_pct=charges)
    st.dataframe(hz, hide_index=True, use_container_width=True)
    st.bar_chart(hz.set_index("holding_days")["annualised_net_pct"])

    st.markdown("##### Portfolio weighting")
    st.caption(
        "Equal-rupee allocation is not equal risk. Risk parity sets weights so "
        "each position contributes the same share of portfolio variance — "
        "volatile and highly-correlated names get less."
    )
    if st.button("Compute weights", key="pf_w"):
        b_pf = st.session_state.get("bucket")
        if b_pf is None or b_pf.is_empty:
            st.warning("Build a bucket on the Screener tab first.")
        else:
            tk = tuple(f"{t}.NS" for t in b_pf.picks["Ticker"])
            with st.spinner("Fetching returns…"):
                fr = fetch_history(tk, period="1y")
            rets = costs.build_returns_matrix(fr, lookback=120) if costs else pd.DataFrame()
            if rets.empty or rets.shape[1] < 2:
                st.warning("Not enough overlapping history to weight.")
            else:
                cmp_df = pfl.compare_schemes(rets)
                st.markdown("**Weighting schemes compared**")
                st.caption(
                    "`risk_concentration` is the largest risk contribution over "
                    "the average. Equal weight typically lets one position "
                    "dominate portfolio variance."
                )
                st.dataframe(cmp_df, hide_index=True, use_container_width=True)

                rp = pfl.risk_parity(rets)
                st.markdown("**Risk parity weights**")
                st.dataframe(rp.to_frame(), use_container_width=True)
                k = st.columns(2)
                k[0].metric("Diversification ratio", rp.diversification_ratio)
                k[1].metric("Effective bets",
                            f"{rp.effective_n} of {rets.shape[1]}")
                for n_ in rp.notes:
                    st.caption(f"· {n_}")

                st.info(
                    "Risk parity improves how capital is split. It cannot create "
                    "diversification that is not there — if effective bets stay "
                    "far below position count, the bucket is concentrated "
                    "regardless of weighting.", icon="ℹ️",
                )

    st.markdown("##### Correlation: how many bets are you really making?")
    if not st.button("Analyse current bucket", key="tc_run"):
        st.info("Run the Screener first, then analyse the resulting basket.")
        return

    b = st.session_state.get("bucket")
    if b is None or b.is_empty:
        st.warning("No bucket found. Build one on the Screener tab first.")
        return

    tickers = tuple(f"{t}.NS" for t in b.picks["Ticker"])
    with st.spinner("Fetching returns…"):
        frames = fetch_history(tickers, period="1y")
    if not frames:
        st.error("Could not fetch price data.")
        return

    rets = costs.build_returns_matrix(frames, lookback=60)
    if rets.empty or rets.shape[1] < 2:
        st.warning("Not enough overlapping data to measure correlation.")
        return

    div = costs.effective_positions(rets)
    k = st.columns(4)
    k[0].metric("Positions", div.n_positions)
    k[1].metric("Mean correlation", f"{div.mean_correlation:.2f}")
    k[2].metric("Effective bets", f"{div.effective_n:.1f}")
    k[3].metric("Risk multiplier", f"{div.risk_multiplier:.2f}x")

    if div.is_concentrated:
        st.warning(
            f"**{div.n_positions} positions behave like {div.effective_n:.1f} "
            f"independent bets.** Portfolio risk is {div.risk_multiplier:.2f}x what "
            "per-position sizing assumes — a bad week hits harder than the "
            "arithmetic suggests."
        )
        adj = costs.adjusted_position_size(100, 1000, div)
        st.info(f"**Scale option:** {adj['note']}", icon="📉")
        alt = costs.adjusted_position_size(100, 1000, div, mode="reduce")
        st.info(f"**Concentrate option:** {alt['note']}", icon="🎯")
    else:
        st.success(f"Diversification adequate — {div.effective_n:.1f} effective "
                   f"positions from {div.n_positions}.")

    clusters = costs.cluster_positions(rets, threshold=0.7)
    multi = {i: v for i, v in clusters.items() if len(v) > 1}
    if multi:
        st.markdown("**Names that move together (correlation ≥ 0.70)**")
        st.caption("Each cluster is effectively one position, whatever its sector labels say.")
        for i, members in multi.items():
            st.markdown(f"- **Cluster {i+1}:** {', '.join(members)}")

    with st.expander("Correlation matrix"):
        st.dataframe(div.correlation_matrix, use_container_width=True)

    lvl, msg = costs.verdict(div, r)
    {"good": st.success, "warn": st.warning, "bad": st.error}[lvl](msg)

    st.caption(
        "Tax rules vary by individual circumstance — loss set-off, carry-forward, "
        "exemption limits and investor-versus-trader classification all matter. "
        "These are headline rates used as a modelling input, not tax advice. "
        "Confirm with a chartered accountant."
    )


# --------------------------------------------------------------------------
# Horizon sweep — the highest-leverage analysis
# --------------------------------------------------------------------------
def render_horizon(cfg: dict) -> None:
    if hz is None:
        st.subheader("Horizon")
        _unavailable("horizon", "It sweeps holding periods to find the economically viable band.")
        return

    st.subheader("Holding period sweep")
    st.caption(
        "The single largest lever available. At 15 days the edge nets to roughly "
        "zero once 20% STCG and ~0.35% charges are applied. This measures IC, "
        "**gross quintile spread in percentage points**, and net-of-tax economics "
        "across holding periods — then reports the whole curve, because picking "
        "the best from a sweep is multiple testing."
    )

    st.info(
        "**What to look for: a plateau, not a peak.** A real effect works across "
        "a contiguous band of horizons. A strategy viable at exactly one value and "
        "nowhere near it has found noise — and testing nine horizons gives noise "
        "nine chances to look good.",
        icon="🎯",
    )

    c = st.columns(4)
    years = c[0].select_slider("Period", ["2y", "5y", "10y"], value="5y", key="hz_y")
    n_stocks = c[1].slider("Stocks", 30, 150, 100, 10, key="hz_n")
    charges = c[2].number_input("Charges %", 0.0, 3.0, 0.35, 0.05, key="hz_c")
    win = c[3].slider("Win rate", 0.30, 0.80, 0.55, 0.05, key="hz_w")

    preset = st.radio("Horizons to test", ["Standard (9)", "Short focus", "Wide"],
                      horizontal=True, key="hz_p")
    horizons = {"Standard (9)": hz.DEFAULT_HORIZONS,
                "Short focus": (5, 8, 10, 12, 15, 18, 20, 25, 30),
                "Wide": (10, 20, 40, 60, 90, 120, 180, 250)}[preset]

    if not st.button("Run horizon sweep", type="primary", key="hz_run"):
        st.info(f"Tests {len(horizons)} horizons: {', '.join(map(str, horizons))} days. "
                "Takes 4–10 minutes.")
        return

    tickers = tuple(cfg["tickers"][:n_stocks])
    with st.spinner(f"Fetching {years} history for {len(tickers)} tickers…"):
        frames = fetch_history(tickers, period=years)
        bench = fetch_history((config.BENCHMARK,), period=years).get(config.BENCHMARK)
    if not frames:
        st.error("No data returned — yfinance may be rate-limiting.")
        return

    bar = st.progress(0.0)
    status = st.empty()

    def _p(i, n, h):
        bar.progress(i / n)
        status.caption(f"Measuring horizon {h} days… ({i}/{n})")

    res = hz.sweep(frames, bench, horizons=horizons,
                   model=cfg.get("model", "momentum"),
                   charges_pct=charges, win_rate=win, progress=_p)
    bar.empty(); status.empty()

    if res.table.empty:
        st.error(res.verdict_msg or "No horizon produced enough windows.")
        return

    {"good": st.success, "ok": st.info, "warn": st.warning,
     "bad": st.error, "error": st.error}[res.verdict_level](res.verdict_msg)
    for n in res.notes:
        st.caption(f"· {n}")

    if res.best_by_annualised:
        b = res.best_by_annualised
        m = st.columns(5)
        m[0].metric("Best horizon", f"{int(b['horizon'])}d")
        m[1].metric("Gross spread", f"{b['gross_spread_pct']:.2f}%")
        m[2].metric("Net/cycle", f"{b['net_per_cycle_pct']:+.3f}%")
        m[3].metric("Annualised", f"{b['annualised_net_pct']:+.2f}%")
        m[4].metric("IC (t)", f"{b['mean_ic']:+.4f} ({b['ic_t']:.1f})")

    if res.plateau:
        st.caption(f"**Viable plateau:** {res.plateau[0]}–{res.plateau[-1]} days "
                   f"({len(res.plateau)} contiguous horizons)")

    st.markdown("##### Full curve")
    show = res.table[["horizon", "windows", "mean_ic", "ic_t", "gross_spread_pct",
                      "spread_t", "cycles_per_year", "tax_rate",
                      "net_per_cycle_pct", "annualised_net_pct", "viable"]]
    st.dataframe(
        show, hide_index=True, use_container_width=True,
        column_config={
            "mean_ic": st.column_config.NumberColumn("IC", format="%.4f"),
            "ic_t": st.column_config.NumberColumn("IC t", format="%.2f"),
            "gross_spread_pct": st.column_config.NumberColumn("Gross %", format="%.2f"),
            "spread_t": st.column_config.NumberColumn("Spread t", format="%.2f"),
            "net_per_cycle_pct": st.column_config.NumberColumn("Net %", format="%+.3f"),
            "annualised_net_pct": st.column_config.NumberColumn("Annual %", format="%+.2f"),
            "tax_rate": st.column_config.NumberColumn("Tax", format="%.3f"),
        },
    )

    st.markdown("##### Annualised net return by horizon")
    st.caption("The curve that matters. Look for a broad hump, not a single spike.")
    st.bar_chart(res.table.set_index("horizon")["annualised_net_pct"])

    cc = st.columns(2)
    with cc[0]:
        st.markdown("**IC by horizon**")
        st.line_chart(res.table.set_index("horizon")["mean_ic"])
    with cc[1]:
        st.markdown("**Gross spread by horizon**")
        st.line_chart(res.table.set_index("horizon")["gross_spread_pct"])

    st.download_button("Download sweep (CSV)", res.table.to_csv(index=False).encode(),
                       file_name="horizon_sweep.csv", mime="text/csv")

    # Sub-period stability on the winner
    if res.best_by_annualised:
        best_h = int(res.best_by_annualised["horizon"])
        st.markdown(f"##### Sub-period stability at {best_h} days")
        st.caption(
            "An edge present in one third of the sample and absent elsewhere is a "
            "regime artefact, not a strategy. This splits the history chronologically."
        )
        with st.spinner("Splitting history…"):
            stab = hz.stability_check(frames, bench, best_h,
                                      model=cfg.get("model", "momentum"))
        if stab.empty:
            st.caption("Not enough history to split meaningfully.")
        else:
            st.dataframe(
                stab[["split", "period", "windows", "mean_ic", "ic_t",
                      "gross_spread_pct"]].round(4),
                hide_index=True, use_container_width=True)
            pos = (stab["mean_ic"] > 0).sum()
            if pos == len(stab):
                st.success(f"Positive IC in all {len(stab)} sub-periods — consistent.")
            elif pos == 0:
                st.error("Negative IC in every sub-period.")
            else:
                st.warning(
                    f"Positive in only {pos} of {len(stab)} sub-periods. The edge "
                    "is regime-dependent, which matters more than the average."
                )

    st.warning(
        "**Before acting on this:** the sweep is measured on the same data the "
        "signal was selected on, so it is optimistic. And a longer horizon changes "
        "what you are trading — a 60-day hold is position trading, not swing "
        "trading, with different drawdown behaviour and different psychology.",
        icon="⚠️",
    )


# --------------------------------------------------------------------------
# Broker tab — measured costs vs assumed costs
# --------------------------------------------------------------------------
def render_broker() -> None:
    if brk is None:
        st.subheader("Broker")
        _unavailable("broker", "It reads your actual fills to measure real transaction costs.")
        return

    st.subheader("Broker: measured transaction costs")
    st.caption(
        "The strategy rests on a number that was **estimated, not measured** — round-trip "
        "cost per tier. Against an edge of roughly 0.5pp per 15-day cycle, being wrong "
        "about it is the difference between profit and a slow bleed. This reads your "
        "actual fills and charges from Paytm Money and compares them."
    )

    st.error(
        "**Read-only by design.** This module cannot place, modify or cancel orders. "
        "That is deliberate: you have no forward evidence yet, and the judgement step "
        "between a ranked list and an actual purchase is currently doing real work.",
        icon="🔒",
    )

    if brk.is_cloud_deployment():
        st.warning(
            "**Running on Streamlit Cloud.** Broker credentials in a shared hosted "
            "environment are a poor idea even for read-only access. Run this locally "
            "(`streamlit run app.py`) with credentials in `.streamlit/secrets.toml`.",
            icon="⚠️",
        )

    with st.expander("Setup"):
        st.markdown(
            "1. Sign in at [developer.paytmmoney.com](https://developer.paytmmoney.com) "
            "with your Paytm Money account (KYC-ready equity account required).\n"
            "2. Create an app; note the API key and secret.\n"
            "3. `pip install pmclient`\n"
            "4. Add to `.streamlit/secrets.toml` — already excluded by `.gitignore`:\n"
            "```toml\n[paytm]\napi_key    = \"your_key\"\n"
            "api_secret = \"your_secret\"\n```\n"
            "Never commit these. If they leak, regenerate immediately."
        )

    if not st.button("Connect (read-only)", type="primary"):
        st.info("Nothing is fetched until you connect.")
        return

    client, status = brk.connect()
    if not status.connected:
        st.error(status.message)
        return
    st.success(status.message)

    with st.spinner("Fetching holdings, orders and trades…"):
        holdings = brk.fetch_holdings(client)
        orders = brk.fetch_orders(client)
        trades = brk.fetch_trades(client)

    m = st.columns(3)
    m[0].metric("Holdings", len(holdings))
    m[1].metric("Orders", len(orders))
    m[2].metric("Trades", len(trades))

    st.markdown("##### Cost analysis")
    report = brk.cost_report(trades, orders)
    level, message = brk.verdict(report)
    {"good": st.success, "warn": st.warning,
     "bad": st.error, "none": st.info}[level](message)

    if report.get("measured_round_trip_pct") is not None:
        c = st.columns(4)
        c[0].metric("Charges (round trip)", f"{report['measured_round_trip_pct']:.3f}%")
        if report.get("mean_slippage_pct") is not None:
            c[1].metric("Mean slippage", f"{report['mean_slippage_pct']:+.3f}%",
                        help="Positive is adverse — you paid more than intended.")
        if report.get("all_in_round_trip_pct") is not None:
            c[2].metric("All-in cost", f"{report['all_in_round_trip_pct']:.3f}%")
        c[3].metric("Assumed (large)", f"{report['assumed']['large']:.2f}%")

        allin = report.get("all_in_round_trip_pct")
        if allin:
            # Named cost_cmp, not comp — `comp` is the module alias for
            # composite, and shadowing it here would break any later use of
            # composite.* inside this function.
            cost_cmp = pd.DataFrame({
                "tier": ["large", "mid", "small"],
                "assumed_pct": [report["assumed"][t] for t in ("large", "mid", "small")],
                "measured_pct": [allin] * 3,
            })
            cost_cmp["gap_pct"] = (cost_cmp["measured_pct"]
                                   - cost_cmp["assumed_pct"]).round(3)
            st.markdown("**Assumed vs measured**")
            st.dataframe(cost_cmp, hide_index=True, use_container_width=True)
            st.caption(
                "If measured materially exceeds assumed, update `est_cost_pct` in "
                "`tiers.py` — the cost model feeds position sizing and the "
                "small-cap exclusion."
            )

    if not orders.empty:
        slip = brk.measure_slippage(orders)
        if not slip.empty:
            with st.expander(f"Per-order slippage ({len(slip)} orders)"):
                st.dataframe(slip, hide_index=True, use_container_width=True)

    if not holdings.empty:
        st.markdown("##### Holdings")
        cols = [c for c in ("Ticker", "Qty", "Avg_cost", "LTP", "Value", "PnL",
                            "PnL_pct") if c in holdings.columns]
        st.dataframe(holdings[cols] if cols else holdings,
                     hide_index=True, use_container_width=True)

        log = st.session_state.get("fwd_log")
        if log is not None and not log.empty:
            rec = brk.reconcile_with_forward_log(holdings, log)
            if not rec.empty:
                st.markdown("##### Which picks did you act on?")
                st.caption(
                    "The forward log records what the model suggested. This shows what "
                    "you did with it — a persistently low action rate is its own form "
                    "of strategy drift, and worth noticing."
                )
                st.dataframe(rec, hide_index=True, use_container_width=True)

    with st.expander("Raw data"):
        st.write("**Orders**"); st.dataframe(orders.head(20))
        st.write("**Trades**"); st.dataframe(trades.head(20))


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

    if _MISSING:
        st.caption(
            "Optional modules not found: "
            + ", ".join(f"`{m}.py`" for m in _MISSING)
            + " — those sections are disabled. Everything else works normally."
        )

    st.info(
        "**EOD data.** Prices are end-of-day, not live. Confirm every level on your "
        "broker terminal before acting. This is a research tool, not advice — "
        "swing trading carries real risk of loss.",
        icon="⚠️",
    )

    tabs = st.tabs(["🔍 Screener", "📊 Detail", "🧪 Backtest", "🔬 Validation", "🎲 Outcomes", "📋 Forward log", "💼 Broker", "📰 News", "📓 Journal", "❓ Method"])

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
        st.divider()
        render_bhavcopy()
        st.divider()
        render_horizon(cfg)
        st.divider()
        render_true_cost(cfg)
    with tabs[4]:
        render_outcomes(cfg)
    with tabs[5]:
        render_forward_log(cfg, shortlist)
    with tabs[6]:
        render_broker()
    with tabs[7]:
        render_news(cfg, shortlist)
    with tabs[8]:
        render_journal()
    with tabs[9]:
        st.markdown(config.METHOD_DOC)


if __name__ == "__main__":
    main()
