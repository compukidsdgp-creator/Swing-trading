"""Month-end review report.

The weekly report says "here is what happened this week". This one asks the
harder question: **after a month of data, what do we actually know?**

The central design principle here is calibrated honesty. At 30 days you will
have roughly 20-30 evaluated picks across 2-3 matured snapshots. That is not
enough to establish an edge, and a report that presented those numbers without
saying so would be actively harmful — it would manufacture confidence that the
data cannot support.

So every statistic here is paired with an explicit confidence level, a
confidence interval, and where relevant, the sample size that *would* be needed
to draw the conclusion the reader wants to draw.
"""

from __future__ import annotations

import datetime as dt
import html
import math
from pathlib import Path

import numpy as np
import pandas as pd

import attribution as attrib

from report import CSS, _cards, _table

# Typical cross-sectional standard deviation of per-window IC for equity
# factors. Used to estimate how many windows are needed for significance.
IC_SD_ASSUMPTION = 0.15


def assess_confidence(n_picks: int, n_windows: int,
                      mean_ret: float, sd_ret: float,
                      forward_ic: float | None) -> dict:
    """Translate sample size into an honest confidence statement."""
    out: dict = {"n_picks": n_picks, "n_windows": n_windows}

    # Confidence interval on mean return
    if n_picks >= 2 and sd_ret > 0:
        se = sd_ret / math.sqrt(n_picks)
        out["se"] = se
        out["ci_low"] = mean_ret - 1.96 * se
        out["ci_high"] = mean_ret + 1.96 * se
        out["ci_excludes_zero"] = (out["ci_low"] > 0) or (out["ci_high"] < 0)
        out["t_stat"] = mean_ret / se if se else 0.0
    else:
        out.update(se=None, ci_low=None, ci_high=None,
                   ci_excludes_zero=False, t_stat=None)

    # Windows needed to detect the observed IC at t=2
    if forward_ic and abs(forward_ic) > 0.005:
        need = (2 * IC_SD_ASSUMPTION / abs(forward_ic)) ** 2
        # Floor at 12 windows: below that, per-window IC variance is itself
        # unreliable, so the projection would be spuriously encouraging.
        need = max(12.0, min(need, 400.0))
        out["windows_needed"] = int(math.ceil(need))
        out["weeks_needed"] = int(math.ceil(need))
        out["months_needed"] = round(need / 4.33, 1)
        out["ic_implausible"] = abs(forward_ic) > 0.15
    else:
        out.update(windows_needed=None, weeks_needed=None,
                   months_needed=None, ic_implausible=False)

    # Overall confidence tier
    if n_picks < 20 or n_windows < 3:
        tier, label = "none", "Insufficient data"
        msg = ("Too few matured picks to say anything at all. This report documents "
               "that the process is running, nothing more.")
    elif n_picks < 60 or n_windows < 8:
        tier, label = "low", "Directional only"
        msg = ("Enough to catch gross problems — a badly broken model, an operational "
               "bug, or forward results that contradict the backtest outright. Not "
               "enough to confirm an edge, and not enough to rule one out.")
    elif n_picks < 150 or n_windows < 20:
        tier, label = "moderate", "Suggestive"
        msg = ("A trend is becoming visible. Still short of statistical proof, but "
               "consistent results here are meaningfully more informative than at "
               "one month.")
    else:
        tier, label = "reasonable", "Statistically meaningful"
        msg = ("Sample size is now large enough that consistent results carry real "
               "weight. Continue monitoring for regime dependence.")

    out.update(tier=tier, label=label, message=msg)
    return out


def month_verdict(fs: dict, conf: dict, backtest_ic: float | None) -> tuple[str, str, list[str]]:
    """Return (severity, headline, action_items)."""
    ic = fs.get("forward_ic")
    mean_ret = fs.get("mean_return_pct", 0) or 0
    n = conf["n_picks"]

    actions: list[str] = []

    # A confidence tier of "none" overrides everything below. Reporting a
    # direction on data that cannot support one is the exact failure this
    # report exists to prevent.
    if conf.get("tier") == "none" or n < 20:
        return ("neu", "Process verified, verdict pending", [
            "The automation is running and recording correctly.",
            "Keep the weekly cadence — do not change any parameters yet.",
            "Revisit at 8 weeks, when the first meaningful read becomes possible.",
            f"Currently {n} matured picks across {conf.get('n_windows', 0)} windows — "
            "below the threshold where any direction can be claimed.",
        ])

    if ic is not None and ic <= -0.02:
        actions += [
            "Do not commit capital on this model.",
            "Forward results contradict the historical backtest — the signature of "
            "overfitting, or of a market regime the model was never fitted to.",
            "Investigate before changing anything: check for data errors, look at "
            "whether losses concentrate in one tier or one regime.",
        ]
        return ("bad", "Forward performance is negative", actions)

    if backtest_ic and backtest_ic > 0 and ic is not None:
        ratio = ic / backtest_ic
        if ratio < 0.3:
            actions += [
                f"Forward IC is only {ratio:.0%} of the backtest figure — substantial decay.",
                "Some decay is normal and expected. This much suggests the backtest was "
                "at least partly fitted to historical noise.",
                "Continue paper trading. Do not size up.",
            ]
            return ("warn", "Substantial decay versus backtest", actions)

    if ic is not None and ic > 0 and mean_ret > 0:
        actions += [
            "Early signs are positive, but the sample is small — treat this as "
            "encouragement to continue, not as validation.",
            "Change nothing this month. Adding improvements now would confound the "
            "measurement.",
            "Continue to the 8-week mark before considering live capital, and start "
            "at quarter size when you do.",
        ]
        return ("ok", "Early results are directionally positive", actions)

    actions += [
        "Results are flat or mixed — the most common outcome at this stage.",
        "Keep collecting. Flat at one month is uninformative, not negative.",
    ]
    return ("neu", "Inconclusive — as expected at this sample size", actions)


def build_monthly_html(
    *,
    generated: dt.datetime,
    period_start: dt.date,
    period_end: dt.date,
    log: pd.DataFrame,
    forward_summary: dict,
    bucket_table: pd.DataFrame | None,
    tier_table: pd.DataFrame | None,
    regime_table: pd.DataFrame | None,
    backtest_ic: float | None = None,
) -> str:
    ev = log[log["status"] == "evaluated"].copy() if not log.empty else pd.DataFrame()
    if not ev.empty:
        ev["fwd_return_pct"] = pd.to_numeric(ev["fwd_return_pct"], errors="coerce")
        ev = ev.dropna(subset=["fwd_return_pct"])

    n_picks = len(ev)
    n_windows = int(ev["snapshot_date"].nunique()) if not ev.empty else 0
    mean_ret = float(ev["fwd_return_pct"].mean()) if n_picks else 0.0
    sd_ret = float(ev["fwd_return_pct"].std(ddof=1)) if n_picks > 1 else 0.0

    conf = assess_confidence(n_picks, n_windows, mean_ret, sd_ret,
                             forward_summary.get("forward_ic"))
    sev, headline, actions = month_verdict(forward_summary, conf, backtest_ic)

    sev_cls = {"bad": "off", "warn": "neu", "ok": "on", "neu": "neu"}[sev]

    extra_css = """
    .verdict { padding: 18px 20px; border-radius: 8px; margin: 0 0 22px; }
    .verdict h3 { margin: 0 0 8px; font-size: 18px; }
    .verdict ul { margin: 8px 0 0; padding-left: 20px; }
    .verdict li { margin-bottom: 5px; font-size: 13.5px; }
    .conf { display: inline-block; padding: 3px 10px; border-radius: 12px;
            font-size: 12px; font-weight: 700; margin-left: 8px; }
    .conf.none { background: #eceff3; color: #5b6472; }
    .conf.low { background: #fdf6e3; color: #7a5c10; }
    .conf.moderate { background: #e4edf9; color: #1c4680; }
    .conf.reasonable { background: #e8f5ea; color: #1a6b2c; }
    .ci { font-family: ui-monospace, monospace; font-size: 13px;
          background: #f8f9fb; padding: 10px 14px; border-radius: 6px;
          border-left: 3px solid #c9ced8; margin: 10px 0; }
    """

    p = [
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>SwingScope — Month-end review</title>"
        f"<style>{CSS}{extra_css}</style></head><body>",
        '<div class="wrap">',
        f'<div class="head"><h1>Month-End Review</h1>'
        f'<div class="meta">{period_start:%d %b} &ndash; {period_end:%d %b %Y} '
        f'&middot; generated {generated:%d %b %Y %H:%M}</div></div>',
        '<div class="body">',

        f'<div class="verdict banner {sev_cls}"><h3>{html.escape(headline)}'
        f'<span class="conf {conf["tier"]}">{html.escape(conf["label"])}</span></h3>'
        f'<div class="sub">{html.escape(conf["message"])}</div>'
        "<ul>" + "".join(f"<li>{html.escape(a)}</li>" for a in actions) + "</ul></div>",
    ]

    # --- What the month produced ---
    p.append("<h2>What the month produced</h2>")
    total_logged = len(log) if not log.empty else 0
    still_open = int((log["status"] == "open").sum()) if not log.empty else 0
    p.append(_cards([
        ("Snapshots taken", log["snapshot_date"].nunique() if total_logged else 0, ""),
        ("Picks recorded", total_logged, ""),
        ("Matured & scored", n_picks, ""),
        ("Still open", still_open, ""),
        ("Mean return", f"{mean_ret:+.2f}%", "pos" if mean_ret > 0 else "neg"),
        ("Hit rate", f"{forward_summary.get('hit_rate_pct', 0)}%", ""),
    ]))

    if still_open:
        p.append(f'<div class="note"><strong>{still_open} picks have not matured yet.</strong> '
                 "A 15-trading-day horizon is about three calendar weeks, so picks recorded "
                 "in the last fortnight are still open. This is why a one-month review sees "
                 "fewer outcomes than snapshots.</div>")

    # --- Statistical reality check ---
    p.append("<h2>How much can we conclude?</h2>")
    if conf.get("ci_low") is not None:
        p.append(
            f'<div class="ci">Mean return {mean_ret:+.2f}% '
            f'&plusmn; {1.96 * conf["se"]:.2f} (95% CI)<br>'
            f'&nbsp;&nbsp;range: {conf["ci_low"]:+.2f}% to {conf["ci_high"]:+.2f}%<br>'
            f'&nbsp;&nbsp;t-statistic: {conf["t_stat"]:+.2f}<br>'
            f'&nbsp;&nbsp;excludes zero: '
            f'<strong>{"yes" if conf["ci_excludes_zero"] else "no"}</strong></div>'
        )
        if not conf["ci_excludes_zero"]:
            p.append('<div class="note warn">The confidence interval spans zero. The true '
                     'mean return could plausibly be negative. This is the expected result '
                     'at one month and is not evidence against the model.</div>')

    if conf.get("weeks_needed"):
        p.append(
            f'<div class="note"><strong>Sample size needed.</strong> To establish the '
            f'observed forward IC at conventional significance (t &gt; 2), roughly '
            f'<strong>{conf["weeks_needed"]} weekly windows</strong> would be required '
            f'&mdash; about {conf["months_needed"]} months of weekly snapshots. '
            f'You currently have {n_windows}.</div>'
        )

    if backtest_ic:
        fic = forward_summary.get("forward_ic")
        if fic is not None:
            ratio = fic / backtest_ic if backtest_ic else 0
            p.append(
                f'<div class="ci">Backtest IC: {backtest_ic:+.4f}<br>'
                f'Forward IC:&nbsp; {fic:+.4f}<br>'
                f'Retention:&nbsp;&nbsp; <strong>{ratio:.0%}</strong> of the historical figure</div>'
            )

    # --- Breakdowns ---
    if bucket_table is not None and not bucket_table.empty:
        p.append("<h2>Return by score bucket</h2>")
        p.append('<div class="note">The core question: does return rise with score? '
                 'At this sample size the pattern is indicative at best &mdash; a single '
                 'outlier can reorder these rows entirely.</div>')
        p.append(_table(bucket_table))

    if tier_table is not None and not tier_table.empty:
        p.append("<h2>By market-cap tier</h2>")
        p.append(_table(tier_table))

    if regime_table is not None and not regime_table.empty:
        p.append("<h2>By market regime</h2>")
        p.append('<div class="note">Did the regime filter help? Compare returns in '
                 'risk-on against neutral and risk-off.</div>')
        p.append(_table(regime_table))

    # --- Best and worst ---
    if n_picks >= 5:
        p.append("<h2>Best and worst picks</h2>")
        cols = ["snapshot_date", "ticker", "tier", "score", "fwd_return_pct"]
        cols = [c for c in cols if c in ev.columns]
        best = ev.nlargest(5, "fwd_return_pct")[cols].round(2)
        worst = ev.nsmallest(5, "fwd_return_pct")[cols].round(2)
        p.append("<h3 style='font-size:14px;margin:14px 0 6px'>Top 5</h3>")
        p.append(_table(best))
        p.append("<h3 style='font-size:14px;margin:18px 0 6px'>Bottom 5</h3>")
        p.append(_table(worst))
        p.append('<div class="note">Check whether the worst picks share a cause &mdash; '
                 'one tier, one sector, one regime, or an earnings date that fell inside '
                 'the holding window.</div>')

    # --- Next month ---
    p.append("<h2>Next month</h2>")
    p.append(
        '<div class="note"><strong>Change one thing at a time, or nothing.</strong> '
        'The single most common way to waste a forward test is to alter several '
        'parameters at once, which makes it impossible to attribute any change in '
        'results. If the month was inconclusive &mdash; which it usually is &mdash; the '
        'correct action is to change nothing and keep collecting.</div>'
    )

    p += [
        "</div>",
        '<div class="foot"><strong>Research output only. Not investment advice.</strong> '
        'No orders were placed by this process. Statistics above are computed on a small '
        'sample and are presented with confidence intervals precisely because point '
        'estimates at this sample size are unreliable. Consider consulting a '
        'SEBI-registered adviser before acting on any of it.</div>',
        "</div></body></html>",
    ]
    return "".join(p)


def save_monthly(html_text: str, out_dir: Path, stamp: dt.datetime,
                 want_pdf: bool = True) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    h = out_dir / f"month_end_{stamp:%Y%m}.html"
    h.write_text(html_text, encoding="utf-8")
    (out_dir / "latest_monthly.html").write_text(html_text, encoding="utf-8")
    written["html"] = h
    if want_pdf:
        try:
            from weasyprint import HTML as WHTML
            pdf = out_dir / f"month_end_{stamp:%Y%m}.pdf"
            WHTML(string=html_text).write_pdf(str(pdf))
            (out_dir / "latest_monthly.pdf").write_bytes(pdf.read_bytes())
            written["pdf"] = pdf
        except Exception as exc:                       # noqa: BLE001
            print(f"  (monthly PDF skipped: {type(exc).__name__}: {exc})")
    return written


def attribution_section(log: pd.DataFrame,
                        prices: dict,
                        bench: pd.DataFrame) -> str:
    """Why the picks worked, not just whether they did.

    An IC says the ranking had content. It does not say whether the return came
    from stock selection or from being in the market while it rose. Momentum in
    particular loads on sector rotation, so returns that look like selection are
    frequently exposure.

    Returns markdown, or an empty string when nothing can be attributed.
    """
    s = attrib.attribute_log(log, prices, bench)
    if s.n == 0:
        return ""

    lines = [
        "", "## Return attribution", "",
        f"{s.n} trades decomposed into market, sector and stock-specific "
        "components. Beta is estimated from data before each entry, never over "
        "the holding period being explained.", "",
        "| Component | Mean contribution |",
        "|---|---|",
        f"| Market | {s.mean_market:+.2f}% |",
        f"| Sector | {s.mean_sector:+.2f}% |",
        f"| **Signal (idiosyncratic)** | **{s.mean_idio:+.2f}%** |",
        f"| Total | {s.mean_total:+.2f}% |",
        "",
        f"**Signal share: {s.idio_share_pct:.0f}%** (t = {s.idio_t_stat})",
        "", s.message, "",
    ]
    for note in s.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)
