#!/usr/bin/env python3
"""End-to-end pipeline — the whole app, run in sequence, unattended.

Executes the stages in the order a human would work through the tabs:

    1. Universe    fetch live NSE constituents
    2. Data        download prices
    3. Regime      decide whether to trade at all, and which tiers
    4. Screener    score every candidate
    5. Bucket      assemble a balanced, regime-respecting shortlist
    6. News        sentiment check on the shortlist, flag scheduled events
    7. Log         commit picks to the forward log before outcomes exist
    8. Evaluate    score any previously-logged picks that have matured
    9. Report      HTML/PDF
   10. Notify      email / Telegram / WhatsApp

Cadence
-------
Default is weekly. A 15-20 day holding period means the score barely moves day
to day, so a daily run mostly generates noise and the temptation to trade more.
`--daily` exists because you may want it anyway; it is not the recommended mode.

Backtest and validation are deliberately NOT in this loop. They are calibration
exercises over 5 years of history — the answer does not meaningfully change in
24 hours, and reacting to daily wobbles in IC is overfitting to noise. Run them
monthly via `automate.py --monthly`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

# Stub Streamlit so the modules import cleanly headless.
try:
    import streamlit  # noqa: F401
except ImportError:
    import types
    _fake = types.ModuleType("streamlit")

    def _passthrough(*a, **k):
        return a[0] if (a and callable(a[0])) else (lambda fn: fn)

    _fake.cache_data = _passthrough
    _fake.cache_resource = _passthrough
    _fake.secrets = {}
    sys.modules["streamlit"] = _fake

import bucket as bk          # noqa: E402
import config                # noqa: E402
import forward_log as flog   # noqa: E402
import governance as gov    # noqa: E402
import macro as mac         # noqa: E402
import health               # noqa: E402
import newsfeed              # noqa: E402
import daily_tracker as dtrack  # noqa: E402
import notify                # noqa: E402
import regime as rg          # noqa: E402
import report as rep         # noqa: E402
import momentum as momo      # noqa: E402
import scoring               # noqa: E402
import sentiment as sent     # noqa: E402
import universe as uni       # noqa: E402

LOG_PATH = Path(os.environ.get("SWINGSCOPE_LOG", "forward_log.csv"))
REPORT_DIR = Path(os.environ.get("SWINGSCOPE_REPORTS", "reports"))


def _fetch(tickers: tuple[str, ...], period: str = "1y") -> dict[str, pd.DataFrame]:
    """Fetch OHLCV via the multi-source layer.

    Primary is yfinance; if it returns too little, the layer falls back to
    reconstructing from cached NSE bhavcopies. Thirteen modules previously
    depended on yfinance alone, which made an unsupported third-party scraper
    a single point of failure for the entire system.
    """
    res = dsrc.fetch(tickers, period=period, min_bars=60)
    if res.fallback_used:
        print(f"  DATA SOURCE FALLBACK: {res.summary()}")
        for w in res.warnings[:3]:
            print(f"    ! {w}")
    return res.frames


def _sectors(tickers: list[str]) -> dict[str, str]:
    """Best-effort sector lookup. Slow and flaky, so failures are tolerated."""
    out: dict[str, str] = {}
    for t in tickers:
        try:
            info = yf.Ticker(f"{t}.NS").info or {}
            s = info.get("sector")
            if s:
                out[t] = s
        except Exception:                                      # noqa: BLE001
            continue
    return out


def run(args) -> tuple[int, list[str], dict]:
    log_lines: list[str] = []
    ctx: dict = {}

    def say(msg: str = "") -> None:
        print(msg)
        log_lines.append(msg)

    say("=" * 66)
    say(f"SwingScope pipeline — {dt.datetime.now():%Y-%m-%d %H:%M}")
    say("=" * 66)

    # --- Model permission gate ---
    #
    # Guards the Knight Capital failure mode: dormant code reactivated by
    # accident. A retired or undocumented model cannot generate picks.
    model_id = ("momentum_12_1_v2" if args.model == "momentum" else "composite_v1")
    permitted, reason = gov.check_model_permitted(model_id, "research shortlist")
    if not permitted:
        say(f"\nBLOCKED: {reason}")
        return 1, log_lines, ctx
    ctx["model_id"] = model_id

    # --- 1. Universe ---
    say("\n[1/10] Universe")
    ures = uni.fetch_index_constituents(args.universe)
    all_tickers = ures.tickers
    say(f"  {args.universe}: {len(all_tickers)} constituents "
        f"({'live' if ures.is_live else 'CACHED FALLBACK'})")
    if not ures.is_live:
        say(f"  ! {ures.note}")
    ctx.update(universe_name=args.universe, universe_live=ures.is_live)

    # --- 2. Data ---
    say("\n[2/10] Price data")
    # 12-1 momentum needs 252 bars plus buffer; a 1y fetch returns only ~250
    # NSE trading days and would silently reject every ticker.
    period = "2y" if args.model == "momentum" else "1y"

    # Fetch the FULL constituent list, then trim by liquidity. The previous
    # code did tickers[:max_tickers], which — because NSE returns constituents
    # alphabetically — kept roughly A-G and discarded H-Z. Every pick came from
    # the early alphabet. Positional truncation of an ordered list is never a
    # valid selection.
    fetch_list = all_tickers[: max(args.max_tickers * 3, args.max_tickers)]
    data_all = _fetch(fetch_list, period=period)
    say(f"  fetch period: {period}")
    say(f"  fetched {len(data_all)} of {len(fetch_list)} candidates")

    tickers, how = uni.trim_universe(
        tuple(data_all), args.max_tickers,
        method=args.trim, frames=data_all,
    )
    data = {t: data_all[t] for t in tickers if t in data_all}
    say(f"  universe: {len(data)} tickers — {how}")
    ctx["n_tickers"] = len(data)

    bench = _fetch((config.BENCHMARK,), period=period).get(config.BENCHMARK)
    say(f"  {len(data)} of {len(tickers)} tickers returned usable history")
    if not data:
        say("  ABORT: no price data (yfinance rate limit?)")
        return 1, log_lines, ctx

    # --- Health gate ---
    #
    # Stale data is more dangerous than missing data: the pipeline would run,
    # produce confident picks from last week's prices, and report success. This
    # aborts instead.
    hc = health.run_all(data, requested=len(tickers))
    ctx["health"] = hc.checks
    for w in hc.warnings:
        say(f"  ! {w}")
    if not hc.passed:
        for f in hc.failures:
            say(f"  FAIL {f}")
        say("\n  ABORT: health checks failed. Producing picks from bad or stale "
            "data would corrupt the forward log, which is the one thing that "
            "cannot be reconstructed.")
        return 1, log_lines, ctx
    say(f"  health: {sum(hc.checks.values())}/{len(hc.checks)} checks passed")

    # --- Provenance ---
    #
    # yfinance silently revises history. Without a fingerprint there is no way
    # to tell later whether a decision looks wrong because the model erred or
    # because the data beneath it moved.
    prov = gov.stamp(data, params={
        "universe": args.universe, "size": args.size, "horizon": args.horizon,
        "model": args.model, "trim": args.trim,
    }, model_id=ctx["model_id"])
    ctx["provenance"] = prov
    say(f"  provenance: code {prov['code']['model_code_hash']} · "
        f"data {prov['data']['hash']} · git {prov['code']['git_commit']}")

    # --- 3. Regime ---
    say("\n[3/10] Regime")
    breadth = rg.compute_breadth(data)
    reg = rg.classify(bench, breadth)
    say(f"  {reg.state.upper()} — Nifty {reg.pct_from_200dma:+.1f}% vs 200DMA"
        + (f", breadth {reg.breadth:.0%}" if reg.breadth is not None else ""))
    say(f"  Tiers permitted: {', '.join(sorted(reg.allowed_tiers))}")
    ctx.update(regime_state=reg.state, regime_desc=reg.description,
               regime_pct=reg.pct_from_200dma, breadth=reg.breadth)

    # --- Macro overlay ---
    #
    # Refines the gate; adds no signals. Momentum crashes cluster in
    # high-volatility states, which the price-only gate cannot see. Macro can
    # only TIGHTEN the regime, never loosen it — the cost of trading in a bad
    # regime far exceeds the cost of missing a good one.
    if not args.skip_macro:
        try:
            mkt = mac.fetch_market_series(period="2y")
            mregime = mac.classify_macro(mkt)
            combined = mac.combine_with_price_regime(reg.state, mregime)
            ctx["macro"] = combined
            say(f"  macro: vol={mregime.volatility_state} "
                f"rate={mregime.rate_state} favourable={mregime.momentum_favourable} "
                f"(confidence {mregime.confidence})")
            if combined["downgraded"]:
                say(f"  REGIME DOWNGRADED by macro: {reg.state} -> "
                    f"{combined['combined_regime']}")
                for r in combined["reasons"][:2]:
                    say(f"    - {r}")
                reg = rg.Regime(combined["combined_regime"],
                                reg.above_200dma, reg.dma50_rising,
                                reg.pct_from_200dma, reg.breadth)
                ctx["regime_state"] = reg.state
        except Exception as exc:                               # noqa: BLE001
            say(f"  macro overlay unavailable ({type(exc).__name__}) — "
                "continuing on price regime alone")

    # --- 4. Screener ---
    say("\n[4/10] Screener")
    if args.model == "momentum":
        say("  model: 12-1 momentum (residual IC +0.055, Newey-West t = 3.73)")
        tier_filter = momo.RECOMMENDED_TIERS if args.exclude_small else None
        ranked = momo.rank_universe(
            data, bench,
            min_turnover_cr=0.0,
            require_above_50ema=True,
            tier_filter=tier_filter,
        )
        if ranked.empty:
            lens = [len(d) for d in data.values() if d is not None]
            say(f"  ABORT: no valid momentum values. Histories ranged "
                f"{min(lens) if lens else 0}-{max(lens) if lens else 0} bars; "
                f"need {momo.LOOKBACK + 5}. If history is sufficient, no stock has "
                "positive 12-month momentum — zero picks is the correct answer.")
            return 1, log_lines, ctx
        say(f"  ranked {len(ranked)} stocks by cross-sectional momentum")
        if args.exclude_small:
            say("  small caps excluded — ~1.5% round-trip cost exceeds the ~0.5pp edge")
    else:
        say("  model: composite v1  ** FAILED VALIDATION "
            "(residual IC +0.004, t = 0.17) **")
        rows = []
        for tkr, df in data.items():
            try:
                m = scoring.evaluate(df, bench)
            except Exception:                                  # noqa: BLE001
                continue
            m["Ticker"] = tkr.replace(".NS", "")
            rows.append(m)
        if not rows:
            say("  ABORT: no stocks produced valid scores")
            return 1, log_lines, ctx
        ranked = pd.DataFrame(rows).sort_values("Score", ascending=False)
        ranked = ranked[ranked["Above_50EMA"]]
        say(f"  scored {len(rows)}, {len(ranked)} above their 50 EMA")

    ctx["model"] = args.model

    # --- 5. Bucket ---
    say("\n[5/10] Bucket")
    sector_map = None
    if args.sector_cap and len(ranked) > 0:
        top_for_sector = ranked.head(min(40, len(ranked)))["Ticker"].tolist()
        sector_map = _sectors(top_for_sector)
        say(f"  sector data for {len(sector_map)} of {len(top_for_sector)} candidates")

    b = bk.build(ranked, reg, size=args.size, max_per_sector=args.sector_cap or 99,
                 min_score=args.min_score, balance_tiers=not args.no_balance,
                 sector_lookup=sector_map)
    say(f"  bucket: {b.actual_size} of {b.target_size} target"
        + (f" — mix {b.tier_counts}" if b.tier_counts else ""))
    for n in b.notes:
        say(f"  · {n}")
    # --- Exit-liquidity gate ---
    #
    # Position sizing assumes stops fill. For a non-F&O stock in a 5% band that
    # can be false: when price locks limit-down there is no bid, the stop never
    # executes, and the position is trapped until buyers return — often several
    # sessions and 25% lower. This is a catastrophic rather than gradual
    # failure, and no ATR calibration prevents it.
    if not b.is_empty and not args.skip_circuit:
        fno, fno_src, fno_live = cir.load_fno_list(allow_refresh=True)
        say(f"\n  exit-liquidity check (F&O list: {len(fno)} symbols, {fno_src})")
        if not fno_live:
            say("    ! F&O list is not live — verify against nseindia.com if it "
                "looks stale")
        kept, dropped = cir.filter_bucket(
            b.picks, fno=fno, require_fno=args.require_fno,
            atr_mult=tr.params("mid")["atr_mult"],
        )
        if not dropped.empty:
            say(f"    excluded {len(dropped)} of {len(b.picks)} on exit risk:")
            for _, r in dropped.iterrows():
                say(f"      {r['Ticker']}: {str(r['exclusion_reason'])[:80]}")
            b.picks = kept.reset_index(drop=True)
            if not b.picks.empty:
                b.picks["Rank"] = range(1, len(b.picks) + 1)
            b.actual_size = len(b.picks)
            b.notes.append(f"{len(dropped)} pick(s) excluded — stop not reachable "
                           "within the daily price band.")
        else:
            say("    all picks pass — stops are reachable")

    ctx["picks"] = b.picks if not b.is_empty else None
    ctx["bucket"] = b

    if b.is_empty:
        say("\n  No picks. Taking no positions is a valid, and often correct, outcome.")

    # --- 6. News ---
    say("\n[6/10] News sentiment")
    news_notes: list[str] = []
    if not b.is_empty and not args.skip_news:
        for name in b.picks["Ticker"].tolist():
            items = newsfeed.fetch(name, limit=8)
            s = sent.analyse_stock(name, items)
            flag = ""
            if s.catalysts.get("earnings"):
                flag = "  <-- EARNINGS in feed, check the date"
                news_notes.append(f"{name}: earnings-related news detected")
            elif s.catalysts.get("regulatory"):
                flag = "  <-- REGULATORY news"
                news_notes.append(f"{name}: regulatory news detected")
            elif s.label in ("Bearish", "Leaning bearish"):
                news_notes.append(f"{name}: negative news tone ({s.mean_score:+.2f})")
            say(f"  {name:14} {s.label:16} {s.mean_score:+.2f} "
                f"({s.n_headlines} items){flag}")
    else:
        say("  skipped")
    ctx["news_notes"] = news_notes

    # --- 5b. Momentum crash protection ---
    #
    # The regime gate watches the INDEX. Momentum crashes are predicted by
    # momentum's own realised volatility, and the two diverge precisely when it
    # matters. Barroso & Santa-Clara (2015) found volatility scaling roughly
    # doubled the Sharpe ratio and largely eliminated the crashes.
    #
    # This module was built and tested but never connected — a real gap.
    if not b.is_empty and not args.skip_crash_protection:
        say("\n[5b] Crash protection")
        try:
            # Prefer the strategy's own return series; fall back to a basket proxy
            log_for_vol = (flog.from_csv(LOG_PATH.read_bytes())
                           if LOG_PATH.exists() else flog.empty_log())
            strat_returns = cp.strategy_returns_from_log(log_for_vol)
            if len(strat_returns) < cp.MIN_OBS:
                strat_returns = cp.proxy_returns_from_holdings(data)
                basis = "basket proxy (insufficient strategy history)"
            else:
                basis = "strategy returns"

            scale = cp.compute_scale(strat_returns)
            ctx["vol_scale"] = scale
            say(f"  realised vol {scale.realised_vol:.1%} vs {scale.target_vol:.0%} "
                f"target -> exposure {scale.exposure_pct:.0f}%  [{basis}]")
            if scale.is_defensive:
                say(f"  DEFENSIVE: {scale.note[:100]}")
            if scale.capped:
                say("  (scaling was capped at the configured bound)")

            crash = cp.crash_risk_indicator(bench)
            if crash.get("available"):
                ctx["crash_risk"] = crash
                say(f"  market crash-risk state: {crash['risk_state']}")
                if crash["risk_state"] != "normal":
                    say(f"    {crash['note'][:110]}")
                    if crash["risk_state"] == "high":
                        # Documented momentum-crash configuration — treat as
                        # seriously as a regime downgrade.
                        say("    Reducing exposure further on crash-risk grounds.")
                        ctx["vol_scale"] = cp.ScalingResult(
                            scale.realised_vol, scale.target_vol, scale.raw_scale,
                            min(scale.applied_scale, 0.5),
                            min(scale.applied_scale, 0.5) * 100,
                            scale.mode, "crash_risk", True, scale.n_observations,
                            "Sharp rebound from deep drawdown — exposure halved.")
            b.notes.append(
                f"Volatility scaling: exposure {ctx['vol_scale'].exposure_pct:.0f}% "
                f"of nominal.")
        except Exception as exc:                               # noqa: BLE001
            say(f"  crash protection unavailable ({type(exc).__name__}) — "
                "proceeding at full exposure")

    # --- 6b. Daily observation diary ---
    #
    # Deliberately separate from the forward log. Daily snapshots overlap
    # heavily and would inflate the significance of the weekly evidence while
    # adding no information. This is a diary; forward_log.csv is the evidence.
    if args.daily_track:
        say("\n[6b] Daily tracker")
        try:
            tr = dtrack.run(
                b.picks if not b.is_empty else pd.DataFrame(),
                data, regime_state=reg.state,
                notes=f"model={args.model}",
            )
            say(f"  +{tr.new_observations} observations, +{tr.new_price_rows} price rows")
            say(f"  tracking {tr.tracked_tickers} tickers")
            if tr.excel_path:
                say(f"  workbook: {tr.excel_path}")
                ctx["tracker_excel"] = tr.excel_path
            else:
                say("  NO WORKBOOK produced — nothing will be attached to the "
                    "notification. Check for an openpyxl error above.")
            ctx["tracker_stats"] = {
                "new_observations": tr.new_observations,
                "tracked_tickers": tr.tracked_tickers,
                "total_rows": len(tr.observations),
                "days": int(tr.observations["obs_date"].nunique())
                        if not tr.observations.empty else 0,
            }
        except Exception as exc:                               # noqa: BLE001
            say(f"  tracker failed ({type(exc).__name__}: {exc}) — continuing")

    # --- 7. Log ---
    say("\n[7/10] Forward log")
    log = flog.from_csv(LOG_PATH.read_bytes()) if LOG_PATH.exists() else flog.empty_log()
    if not b.is_empty and not args.no_log:
        before = len(log)
        log = flog.record_snapshot(log, b.picks, regime_state=reg.state,
                                   horizon=args.horizon, top_n=args.size)
        LOG_PATH.write_bytes(flog.to_csv(log))
        say(f"  recorded {len(log) - before} picks (log now {len(log)} rows)")
    else:
        say("  nothing to record")

    # --- 8. Evaluate ---
    say("\n[8/10] Evaluate matured picks")
    open_t = log.loc[log["status"] == "open", "ticker"].unique() if not log.empty else []
    if len(open_t):
        pdata = _fetch(tuple(f"{t}.NS" for t in open_t), period="6mo")
        lookup = {t.replace(".NS", ""): float(d["Close"].iloc[-1])
                  for t, d in pdata.items() if d is not None and len(d)}
        # Pass full history so each pick is valued ON its target date rather
        # than at today's close — otherwise a late run records a longer
        # holding period than the horizon that was validated.
        hist = {t.replace(".NS", ""): d for t, d in pdata.items()
                if d is not None and len(d)}
        log, filled = flog.evaluate_open(log, lookup, price_history=hist)
        LOG_PATH.write_bytes(flog.to_csv(log))
        say(f"  evaluated {filled} (of {len(open_t)} open)")
    else:
        say("  none open")

    fs = flog.analyse(log)
    ctx["forward_summary"] = fs if "error" not in fs else None
    if "error" not in fs:
        say(f"  forward: {fs['evaluated_picks']} picks, mean "
            f"{fs['mean_return_pct']:+.2f}%, IC {fs.get('forward_ic')}")
        if args.backtest_ic:
            verdict = flog.compare_to_backtest(fs.get("forward_ic"), args.backtest_ic)
            say(f"  vs backtest IC {args.backtest_ic}: {verdict}")
            ctx["retention_note"] = verdict
    else:
        say(f"  {fs['error']}")
    ctx["bucket_table"] = flog.by_dimension(log, "score_bucket")
    ctx["tier_table"] = flog.by_dimension(log, "tier")

    # --- 9. Report ---
    say("\n[9/10] Report")
    stamp = dt.datetime.now()
    html = rep.build_html(
        generated=stamp, universe_name=args.universe,
        universe_live=ures.is_live, n_tickers=len(tickers),
        regime_state=reg.state, regime_desc=reg.description,
        regime_pct=reg.pct_from_200dma, breadth=reg.breadth,
        picks=ctx.get("picks"), forward_summary=ctx.get("forward_summary"),
        bucket_table=ctx.get("bucket_table"), tier_table=ctx.get("tier_table"),
        notes=([f"Ranking model: {args.model}"]
               + ([ctx["retention_note"]] if ctx.get("retention_note") else [])
               + news_notes + b.notes),
    )
    written = rep.save(html, REPORT_DIR, stamp, want_pdf=not args.no_pdf)
    for k, v in written.items():
        say(f"  wrote {k}: {v}")
    ctx["html"] = html
    ctx["written"] = written

    # --- 9b. Month-end review ---
    if args.monthly and "error" not in fs:
        import monthly as mon
        ev = log[log["status"] == "evaluated"]
        dates = pd.to_datetime(ev["snapshot_date"], errors="coerce").dropna()
        m_html = mon.build_monthly_html(
            generated=stamp,
            period_start=(dates.min().date() if len(dates)
                          else stamp.date() - dt.timedelta(days=30)),
            period_end=stamp.date(), log=log, forward_summary=fs,
            bucket_table=flog.by_dimension(log, "score_bucket"),
            tier_table=flog.by_dimension(log, "tier"),
            regime_table=flog.by_dimension(log, "regime"),
            backtest_ic=args.backtest_ic,
        )
        m_written = mon.save_monthly(m_html, REPORT_DIR, stamp, want_pdf=not args.no_pdf)
        for kind, path in m_written.items():
            say(f"  wrote monthly {kind}: {path}")
        written.update({f"monthly_{k}": v for k, v in m_written.items()})
        html = m_html          # email the month-end review instead

    # --- 10. Notify ---
    say("\n[10/10] Notify")
    if args.notify:
        text = bk.to_text(b, regime_desc=reg.description)
        if news_notes:
            text += "\n\nFlags:\n" + "\n".join(f"- {n}" for n in news_notes[:6])

        # Daily diary summary, if the tracker ran
        ts = ctx.get("tracker_stats")
        if ts:
            text += (f"\n\n_Diary: {ts['total_rows']} observations over "
                     f"{ts['days']} days, tracking {ts['tracked_tickers']} tickers._")

        send = dict(written)
        xl = ctx.get("tracker_excel")
        if xl and Path(xl).exists():
            send["xlsx"] = Path(xl)
            say(f"  attaching workbook: {Path(xl).name} "
                f"({Path(xl).stat().st_size/1024:.0f} KB)")
        elif args.daily_track:
            say("  no workbook to attach")

        res = notify.dispatch(
            subject=f"SwingScope {stamp:%d %b} — {b.actual_size} picks "
                    f"[{reg.state.replace('_', ' ')}]",
            html_body=html, text_body=text,
            attachments=send, channels=args.channels,
        )
        say(f"  {res if res else 'no channels configured'}")
    else:
        say("  skipped (pass --notify to enable)")

    # --- Append-only audit entry ---
    gov.audit("pipeline_run", {
        "model_id": ctx.get("model_id"),
        "universe": args.universe,
        "regime": reg.state,
        "picks": b.actual_size if not b.is_empty else 0,
        "tickers_scored": len(data),
        "provenance": ctx.get("provenance", {}),
        "health": ctx.get("health", {}),
    })

    say("\n" + "-" * 66)
    say("Analytical output only. Not investment advice. No orders were placed.")
    return 0, log_lines, ctx


def main() -> int:
    p = argparse.ArgumentParser(description="SwingScope end-to-end pipeline")
    p.add_argument("--universe", default="Nifty 500")
    p.add_argument("--size", type=int, default=10, help="target bucket size")
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--min-score", type=float, default=60.0)
    p.add_argument("--max-tickers", type=int, default=150)
    p.add_argument("--trim", choices=["liquidity", "random", "none"],
                   default="liquidity",
                   help="how to reduce the universe to --max-tickers. NEVER "
                        "positional: NSE returns constituents alphabetically, so "
                        "taking the first N keeps only A-G.")
    p.add_argument("--sector-cap", type=int, default=2,
                   help="max names per sector; 0 disables (and skips the slow lookup)")
    p.add_argument("--no-balance", action="store_true",
                   help="take the top N by score instead of balancing tiers")
    p.add_argument("--skip-news", action="store_true")
    p.add_argument("--skip-macro", action="store_true",
                   help="skip the macro overlay on the regime gate")
    p.add_argument("--skip-crash-protection", action="store_true",
                   help="skip volatility scaling. NOT recommended — momentum "
                        "crashes are the dominant tail risk for this strategy.")
    p.add_argument("--skip-circuit", action="store_true",
                   help="skip the exit-liquidity check (NOT recommended — it "
                        "prevents picks whose stop cannot fill inside the "
                        "daily price band)")
    p.add_argument("--require-fno", action="store_true",
                   help="only include F&O-eligible stocks, which have no "
                        "individual price band")
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--no-pdf", action="store_true")
    p.add_argument("--notify", action="store_true")
    p.add_argument("--channels", default="auto",
                   help="auto | email,telegram,whatsapp")
    p.add_argument("--daily", action="store_true",
                   help="acknowledge daily cadence (not recommended for 15-20d holds)")
    p.add_argument("--model", choices=["momentum", "composite_v1"], default="momentum",
                   help="ranking signal. momentum = 12-1 momentum, validated at "
                        "residual IC +0.055, t = 3.73. composite_v1 failed validation "
                        "(residual IC +0.004, t = 0.17) and is kept only for comparison.")
    p.add_argument("--exclude-small", action="store_true", default=True,
                   help="exclude small caps, where ~1.5%% round-trip costs exceed the "
                        "~0.5pp edge. On by default.")
    p.add_argument("--include-small", dest="exclude_small", action="store_false",
                   help="override the cost-viability exclusion")
    p.add_argument("--backtest-ic", type=float, default=None,
                   help="backtest IC from the Validation tab. Enables the "
                        "forward-vs-backtest retention comparison in the report — "
                        "the single most informative number once forward data exists.")
    p.add_argument("--monthly", action="store_true",
                   help="also produce the month-end review report")
    p.add_argument("--daily-track", action="store_true",
                   help="append to the daily observation diary and rebuild the "
                        "Excel workbook. Kept entirely separate from the weekly "
                        "forward log, whose windows must not overlap.")
    args = p.parse_args()

    if args.daily:
        print("NOTE: daily cadence on a 15-20 day system mostly produces noise "
              "and the temptation to overtrade. Weekly is recommended.\n")
    if args.model == "composite_v1":
        print("WARNING: composite_v1 failed factor neutralisation (residual IC +0.004, "
              "t = 0.17). Its ranking is not supported by evidence.\n")

    code, lines, _ = run(args)

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "pipeline_latest.txt").write_text("\n".join(lines))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
