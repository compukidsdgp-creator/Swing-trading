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
import newsfeed              # noqa: E402
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
    if not tickers:
        return {}
    raw = yf.download(list(tickers), period=period, interval="1d", auto_adjust=True,
                      progress=False, group_by="ticker", threads=True)
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    for t in tickers:
        try:
            df = raw[t].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
        except (KeyError, IndexError):
            continue
        df = df.dropna(how="all")
        if df.empty or len(df) < 60:
            continue
        df.columns = [str(c).title() for c in df.columns]
        if {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
            out[t] = df
    return out


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

    # --- 1. Universe ---
    say("\n[1/10] Universe")
    ures = uni.fetch_index_constituents(args.universe)
    tickers = tuple(ures.tickers[: args.max_tickers])
    say(f"  {args.universe}: {len(tickers)} tickers "
        f"({'live' if ures.is_live else 'CACHED FALLBACK'})")
    if not ures.is_live:
        say(f"  ! {ures.note}")
    ctx.update(universe_name=args.universe, universe_live=ures.is_live,
               n_tickers=len(tickers))

    # --- 2. Data ---
    say("\n[2/10] Price data")
    # 12-1 momentum needs 252 bars plus buffer; a 1y fetch returns only ~250
    # NSE trading days and would silently reject every ticker.
    period = "2y" if args.model == "momentum" else "1y"
    data = _fetch(tickers, period=period)
    bench = _fetch((config.BENCHMARK,), period=period).get(config.BENCHMARK)
    say(f"  fetch period: {period}")
    say(f"  {len(data)} of {len(tickers)} tickers returned usable history")
    if not data:
        say("  ABORT: no price data (yfinance rate limit?)")
        return 1, log_lines, ctx

    # --- 3. Regime ---
    say("\n[3/10] Regime")
    breadth = rg.compute_breadth(data)
    reg = rg.classify(bench, breadth)
    say(f"  {reg.state.upper()} — Nifty {reg.pct_from_200dma:+.1f}% vs 200DMA"
        + (f", breadth {reg.breadth:.0%}" if reg.breadth is not None else ""))
    say(f"  Tiers permitted: {', '.join(sorted(reg.allowed_tiers))}")
    ctx.update(regime_state=reg.state, regime_desc=reg.description,
               regime_pct=reg.pct_from_200dma, breadth=reg.breadth)

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
        log, filled = flog.evaluate_open(log, lookup)
        LOG_PATH.write_bytes(flog.to_csv(log))
        say(f"  evaluated {filled} (of {len(open_t)} open)")
    else:
        say("  none open")

    fs = flog.analyse(log)
    ctx["forward_summary"] = fs if "error" not in fs else None
    if "error" not in fs:
        say(f"  forward: {fs['evaluated_picks']} picks, mean "
            f"{fs['mean_return_pct']:+.2f}%, IC {fs.get('forward_ic')}")
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
        notes=([f"Ranking model: {args.model}"] + news_notes + b.notes),
    )
    written = rep.save(html, REPORT_DIR, stamp, want_pdf=not args.no_pdf)
    for k, v in written.items():
        say(f"  wrote {k}: {v}")
    ctx["html"] = html
    ctx["written"] = written

    # --- 10. Notify ---
    say("\n[10/10] Notify")
    if args.notify:
        text = bk.to_text(b, regime_desc=reg.description)
        if news_notes:
            text += "\n\nFlags:\n" + "\n".join(f"- {n}" for n in news_notes[:6])
        res = notify.dispatch(
            subject=f"SwingScope {stamp:%d %b} — {b.actual_size} picks "
                    f"[{reg.state.replace('_', ' ')}]",
            html_body=html, text_body=text,
            attachments=written, channels=args.channels,
        )
        say(f"  {res if res else 'no channels configured'}")
    else:
        say("  skipped (pass --notify to enable)")

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
    p.add_argument("--sector-cap", type=int, default=2,
                   help="max names per sector; 0 disables (and skips the slow lookup)")
    p.add_argument("--no-balance", action="store_true",
                   help="take the top N by score instead of balancing tiers")
    p.add_argument("--skip-news", action="store_true")
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
