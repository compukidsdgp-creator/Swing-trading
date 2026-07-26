#!/usr/bin/env python3
"""Headless weekly automation for the 30-day plan.

Runs without Streamlit so it can execute on a schedule (GitHub Actions, cron,
or any runner). Each invocation:

  1. Fetches the live universe from NSE
  2. Downloads prices, scores every stock, applies the regime gate
  3. Appends this week's top N picks to forward_log.csv
  4. Evaluates any previously-logged picks that have matured
  5. Writes a plain-text report

What it deliberately does NOT do
--------------------------------
It does not place orders and it never will. Automating the *recording* of picks
removes the human temptation to quietly drop the ones that went wrong — which is
exactly what makes a forward log trustworthy. Automating execution would remove
the judgement that keeps you solvent.

Usage
-----
    python automate.py --universe "Nifty 500" --top 10 --horizon 15
    python automate.py --evaluate-only
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

# Stub Streamlit's cache decorators so the modules import cleanly headless.
try:
    import streamlit  # noqa: F401
except ImportError:
    import types

    fake = types.ModuleType("streamlit")

    def _passthrough(*a, **k):
        def deco(fn):
            return fn
        return deco if not (a and callable(a[0])) else a[0]

    fake.cache_data = _passthrough
    fake.cache_resource = _passthrough
    sys.modules["streamlit"] = fake

import config          # noqa: E402
import regime as rg    # noqa: E402
import monthly as mon  # noqa: E402
import report as rep   # noqa: E402
import scoring         # noqa: E402
import forward_log as flog  # noqa: E402
import universe as uni      # noqa: E402

LOG_PATH = Path(os.environ.get("SWINGSCOPE_LOG", "forward_log.csv"))
REPORT_DIR = Path(os.environ.get("SWINGSCOPE_REPORTS", "reports"))


def fetch(tickers: tuple[str, ...], period: str = "1y") -> dict[str, pd.DataFrame]:
    if not tickers:
        return {}
    raw = yf.download(list(tickers), period=period, interval="1d",
                      auto_adjust=True, progress=False, group_by="ticker", threads=True)
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


def load_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        return flog.from_csv(LOG_PATH.read_bytes())
    return flog.empty_log()


def save_log(log: pd.DataFrame) -> None:
    LOG_PATH.write_bytes(flog.to_csv(log))


def run_snapshot(universe_name: str, top_n: int, horizon: int,
                 min_score: int, max_tickers: int) -> tuple[list[str], dict]:
    lines: list[str] = []
    ctx: dict = {"universe_name": universe_name, "universe_live": False,
                 "n_tickers": 0, "regime_state": "neutral", "regime_desc": "",
                 "regime_pct": 0.0, "breadth": None, "picks": None}
    res = uni.fetch_index_constituents(universe_name)
    tickers = tuple(res.tickers[:max_tickers])
    lines.append(f"Universe: {universe_name} — {len(tickers)} tickers "
                 f"({'live' if res.is_live else 'CACHED FALLBACK'})")
    ctx["universe_live"] = res.is_live
    ctx["n_tickers"] = len(tickers)
    if not res.is_live:
        lines.append(f"  ! {res.note}")

    data = fetch(tickers)
    bench = fetch((config.BENCHMARK,)).get(config.BENCHMARK)
    if not data:
        lines.append("ERROR: no price data returned (yfinance rate limit?).")
        return lines, ctx

    breadth = rg.compute_breadth(data)
    reg = rg.classify(bench, breadth)
    lines.append(f"Regime: {reg.state.upper()} — Nifty {reg.pct_from_200dma:+.1f}% vs 200DMA"
                 + (f", breadth {reg.breadth:.0%}" if reg.breadth is not None else ""))
    lines.append(f"Tiers permitted: {', '.join(sorted(reg.allowed_tiers))}")
    ctx.update(regime_state=reg.state, regime_desc=reg.description,
               regime_pct=reg.pct_from_200dma, breadth=reg.breadth)

    rows = []
    for tkr, df in data.items():
        try:
            m = scoring.evaluate(df, bench)
        except Exception:
            continue
        m["Ticker"] = tkr.replace(".NS", "")
        rows.append(m)

    if not rows:
        lines.append("ERROR: no stocks produced valid scores.")
        return lines, ctx

    scored = pd.DataFrame(rows)
    keep = scored[
        (scored["Score"] >= min_score)
        & (scored["Above_50EMA"])
        & (scored["Tier"].isin(reg.allowed_tiers))
    ].sort_values("Score", ascending=False)

    lines.append(f"Scored {len(scored)}, {len(keep)} passed filters (min score {min_score}).")

    if keep.empty:
        lines.append("No picks this week — nothing passed. This is a valid outcome.")
        return lines, ctx

    log = load_log()
    log = flog.record_snapshot(log, keep, regime_state=reg.state,
                               horizon=horizon, top_n=top_n)
    save_log(log)

    lines.append(f"\nRecorded top {min(top_n, len(keep))} picks:")
    for i, (_, r) in enumerate(keep.head(top_n).iterrows(), 1):
        lines.append(f"  {i:2}. {r['Ticker']:14} {r['Tier']:6} score {r['Score']:3}  "
                     f"₹{r['Close']:>9,.1f}  RSI {r['RSI']:>4.0f}  ATR {r['ATR_pct']:.1f}%")

    top = keep.head(top_n).reset_index(drop=True)
    top.insert(0, "Rank", range(1, len(top) + 1))
    ctx["picks"] = top
    return lines, ctx


def run_evaluate() -> tuple[list[str], dict]:
    lines: list[str] = []
    ctx: dict = {"forward_summary": None, "bucket_table": None, "tier_table": None}
    log = load_log()
    if log.empty:
        lines.append("Forward log is empty — nothing to evaluate.")
        return lines, ctx

    open_t = log.loc[log["status"] == "open", "ticker"].unique()
    if len(open_t) == 0:
        lines.append("No open picks.")
    else:
        data = fetch(tuple(f"{t}.NS" for t in open_t), period="6mo")
        lookup = {t.replace(".NS", ""): float(df["Close"].iloc[-1])
                  for t, df in data.items() if df is not None and len(df)}
        log, filled = flog.evaluate_open(log, lookup)
        save_log(log)
        lines.append(f"Evaluated {filled} matured picks ({len(open_t)} were open).")

    res = flog.analyse(log)
    ctx["forward_summary"] = res
    if "error" in res:
        lines.append(res["error"])
        return lines, ctx

    lines.append("\n--- Forward performance to date ---")
    for k in ("evaluated_picks", "snapshots", "mean_return_pct", "median_return_pct",
              "hit_rate_pct", "forward_ic", "forward_ic_t", "open_picks"):
        lines.append(f"  {k:20} {res.get(k)}")

    if res.get("snapshots", 0) < 6:
        lines.append("\n  NOTE: fewer than 6 snapshots — error bars are wide. Keep logging.")

    ctx["tier_table"] = flog.by_dimension(log, "tier")
    bucket = flog.by_dimension(log, "score_bucket")
    ctx["bucket_table"] = bucket
    if not bucket.empty:
        lines.append("\n--- Return by score bucket (does return rise with score?) ---")
        for _, r in bucket.iterrows():
            lines.append(f"  {str(r.iloc[0]):8} n={int(r['picks']):3}  "
                         f"mean {r['mean_ret_pct']:+7.2f}%  hit {r['hit_rate_pct']:5.1f}%")
    return lines, ctx


def main() -> int:
    p = argparse.ArgumentParser(description="SwingScope weekly automation")
    p.add_argument("--universe", default="Nifty 500")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--min-score", type=int, default=65)
    p.add_argument("--max-tickers", type=int, default=150)
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--snapshot-only", action="store_true")
    p.add_argument("--no-pdf", action="store_true", help="skip PDF generation")
    p.add_argument("--email", action="store_true",
                   help="email the report (needs SMTP_* env vars)")
    p.add_argument("--monthly", action="store_true",
                   help="also produce the month-end review report")
    p.add_argument("--backtest-ic", type=float, default=None,
                   help="backtest IC from the Validation tab, for comparison")
    args = p.parse_args()

    stamp = dt.datetime.now()
    out = [
        "=" * 68,
        f"SwingScope automated run — {stamp:%Y-%m-%d %H:%M}",
        "=" * 68,
        "",
    ]

    ctx: dict = {}
    if not args.evaluate_only:
        out += ["## SNAPSHOT", ""]
        lines, c = run_snapshot(args.universe, args.top, args.horizon,
                                args.min_score, args.max_tickers)
        out += lines
        out.append("")
        ctx.update(c)

    if not args.snapshot_only:
        out += ["## EVALUATE", ""]
        lines, c = run_evaluate()
        out += lines
        ctx.update(c)

    out += [
        "",
        "-" * 68,
        "Research output only. Not investment advice. No orders were placed.",
    ]

    text = "\n".join(out)
    print(text)

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / f"report_{stamp:%Y%m%d}.txt").write_text(text)
    (REPORT_DIR / "latest.txt").write_text(text)

    # --- Styled HTML + optional PDF ---
    html_text = rep.build_html(
        generated=stamp,
        universe_name=ctx.get("universe_name", args.universe),
        universe_live=ctx.get("universe_live", False),
        n_tickers=ctx.get("n_tickers", 0),
        regime_state=ctx.get("regime_state", "neutral"),
        regime_desc=ctx.get("regime_desc", ""),
        regime_pct=ctx.get("regime_pct", 0.0),
        breadth=ctx.get("breadth"),
        picks=ctx.get("picks"),
        forward_summary=ctx.get("forward_summary"),
        bucket_table=ctx.get("bucket_table"),
        tier_table=ctx.get("tier_table"),
    )
    written = rep.save(html_text, REPORT_DIR, stamp, want_pdf=not args.no_pdf)
    for kind, path in written.items():
        print(f"  wrote {kind}: {path}")

    # --- Month-end review ---
    if args.monthly:
        log = load_log()
        fs = flog.analyse(log)
        if "error" in fs:
            print(f"  Monthly report skipped: {fs['error']}")
        else:
            ev = log[log["status"] == "evaluated"]
            dates = pd.to_datetime(ev["snapshot_date"], errors="coerce").dropna()
            m_html = mon.build_monthly_html(
                generated=stamp,
                period_start=(dates.min().date() if len(dates)
                              else stamp.date() - dt.timedelta(days=30)),
                period_end=stamp.date(),
                log=log, forward_summary=fs,
                bucket_table=flog.by_dimension(log, "score_bucket"),
                tier_table=flog.by_dimension(log, "tier"),
                regime_table=flog.by_dimension(log, "regime"),
                backtest_ic=args.backtest_ic,
            )
            m_written = mon.save_monthly(m_html, REPORT_DIR, stamp,
                                         want_pdf=not args.no_pdf)
            for kind, path in m_written.items():
                print(f"  wrote monthly {kind}: {path}")
            written.update({f"monthly_{k}": v for k, v in m_written.items()})
            html_text = m_html      # email the month-end review instead

    # --- Optional email delivery ---
    if args.email:
        host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        port = int(os.environ.get("SMTP_PORT", "465"))
        user = os.environ.get("SMTP_USER", "")
        pwd = os.environ.get("SMTP_PASS", "")
        to = os.environ.get("REPORT_TO", user)
        if not (user and pwd and to):
            print("  Email skipped: set SMTP_USER, SMTP_PASS and REPORT_TO.")
        else:
            ok = rep.email_report(
                html_text, written, smtp_host=host, smtp_port=port,
                user=user, password=pwd, to_addr=to,
                subject=f"SwingScope report — {stamp:%d %b %Y} "
                        f"[{ctx.get('regime_state', 'n/a').replace('_', ' ')}]",
            )
            print("  Email sent." if ok else "  Email delivery failed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
