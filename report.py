"""Report generation — styled HTML, optional PDF.

Produces a self-contained HTML file (no external assets, safe to email or open
offline) and, when WeasyPrint is available, a PDF alongside it.

PDF is optional by design: WeasyPrint needs system libraries (pango, cairo) that
are fiddly on some CI runners. HTML always works, renders identically in any
browser and email client, and is a fraction of the size.
"""

from __future__ import annotations

import datetime as dt
import html
from pathlib import Path

import pandas as pd

CSS = """
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: #f5f6f8; color: #1c1f26; margin: 0; padding: 24px 16px; line-height: 1.55;
}
.wrap { max-width: 900px; margin: 0 auto; background: #fff;
        border-radius: 10px; overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,.08), 0 8px 24px rgba(0,0,0,.06); }
.head { background: linear-gradient(135deg, #14213d 0%, #22406e 100%);
        color: #fff; padding: 26px 30px; }
.head h1 { margin: 0; font-size: 25px; letter-spacing: -.4px; }
.head .meta { margin-top: 5px; font-size: 13px; opacity: .8; }
.body { padding: 26px 30px; }
h2 { font-size: 16px; margin: 30px 0 12px; padding-bottom: 7px;
     border-bottom: 2px solid #eceff3; color: #14213d; }
h2:first-child { margin-top: 0; }

.banner { padding: 13px 16px; border-radius: 7px; margin: 0 0 20px;
          font-size: 14px; font-weight: 600; }
.banner.on   { background: #e8f5ea; color: #1a6b2c; border-left: 4px solid #2e9e4f; }
.banner.neu  { background: #fdf6e3; color: #7a5c10; border-left: 4px solid #d9a520; }
.banner.off  { background: #fdeaea; color: #8f2020; border-left: 4px solid #cc3333; }
.banner .sub { font-weight: 400; font-size: 13px; margin-top: 4px; opacity: .85; }

.cards { display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 20px; }
.card { flex: 1 1 130px; background: #f8f9fb; border: 1px solid #e6e9ef;
        border-radius: 7px; padding: 13px 14px; }
.card .k { font-size: 11px; text-transform: uppercase; letter-spacing: .5px;
           color: #6b7280; font-weight: 600; }
.card .v { font-size: 21px; font-weight: 700; margin-top: 3px; color: #14213d; }
.card .v.pos { color: #1a6b2c; }
.card .v.neg { color: #b02020; }

table { width: 100%; border-collapse: collapse; font-size: 13px; margin: 8px 0 4px; }
th { background: #14213d; color: #fff; text-align: left;
     padding: 8px 10px; font-weight: 600; font-size: 12px; }
td { padding: 7px 10px; border-bottom: 1px solid #eceff3; }
tr:last-child td { border-bottom: none; }
tbody tr:nth-child(even) { background: #fafbfc; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.tier { display: inline-block; padding: 1px 7px; border-radius: 10px;
        font-size: 11px; font-weight: 600; }
.tier.large { background: #e4edf9; color: #1c4680; }
.tier.mid   { background: #eae4f9; color: #4a2c80; }
.tier.small { background: #f9ecdf; color: #8a4a10; }

.note { background: #f8f9fb; border-left: 3px solid #c9ced8;
        padding: 11px 15px; margin: 16px 0; font-size: 13px; color: #4b5563; }
.warn { background: #fdf6e3; border-left: 3px solid #d9a520; color: #6b5312; }
.foot { padding: 16px 30px 24px; font-size: 11.5px; color: #8b93a1;
        border-top: 1px solid #eceff3; }
.empty { color: #8b93a1; font-style: italic; font-size: 13px; padding: 10px 0; }
"""


def _cards(items: list[tuple[str, str, str]]) -> str:
    out = ['<div class="cards">']
    for k, v, cls in items:
        out.append(f'<div class="card"><div class="k">{html.escape(k)}</div>'
                   f'<div class="v {cls}">{html.escape(str(v))}</div></div>')
    out.append("</div>")
    return "".join(out)


def _table(df: pd.DataFrame, num_cols: set[str] | None = None) -> str:
    if df is None or df.empty:
        return '<div class="empty">No data.</div>'
    num_cols = num_cols or set()
    head = "".join(
        f'<th class="{"num" if c in num_cols else ""}">{html.escape(str(c))}</th>'
        for c in df.columns
    )
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            txt = "" if pd.isna(v) else str(v)
            if c.lower() == "tier" and txt in ("large", "mid", "small"):
                txt = f'<span class="tier {txt}">{txt}</span>'
            else:
                txt = html.escape(txt)
            cells.append(f'<td class="{"num" if c in num_cols else ""}">{txt}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def build_html(
    *,
    generated: dt.datetime,
    universe_name: str,
    universe_live: bool,
    n_tickers: int,
    regime_state: str,
    regime_desc: str,
    regime_pct: float,
    breadth: float | None,
    picks: pd.DataFrame | None,
    forward_summary: dict | None,
    bucket_table: pd.DataFrame | None,
    tier_table: pd.DataFrame | None,
    notes: list[str] | None = None,
) -> str:
    """Assemble the full report."""
    cls = {"risk_on": "on", "neutral": "neu", "risk_off": "off"}.get(regime_state, "neu")
    breadth_txt = f" &middot; breadth {breadth:.0%}" if breadth is not None else ""

    parts = [
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>SwingScope — {generated:%d %b %Y}</title><style>{CSS}</style></head><body>",
        '<div class="wrap">',
        f'<div class="head"><h1>SwingScope Weekly Report</h1>'
        f'<div class="meta">{generated:%A, %d %B %Y · %H:%M} &middot; '
        f'{html.escape(universe_name)} ({n_tickers} tickers, '
        f'{"live" if universe_live else "cached fallback"})</div></div>',
        '<div class="body">',
        f'<div class="banner {cls}">Market regime: {regime_state.replace("_", " ").upper()}'
        f'<div class="sub">{html.escape(regime_desc)} &middot; '
        f'Nifty {regime_pct:+.1f}% vs 200 DMA{breadth_txt}</div></div>',
    ]

    if not universe_live:
        parts.append('<div class="note warn"><strong>Universe served from cached '
                     'fallback.</strong> NSE was unreachable, so constituents may be '
                     'out of date.</div>')

    # --- Picks ---
    parts.append("<h2>This week&rsquo;s picks</h2>")
    if picks is None or picks.empty:
        parts.append('<div class="note">Nothing passed the filters this week. '
                     'That is a valid outcome, not a failure &mdash; particularly in a '
                     'neutral or risk-off regime.</div>')
    else:
        cols = [c for c in ["Rank", "Ticker", "Tier", "Score", "Momentum", "Close",
                            "Stop", "Stop_pct", "Target_1R",
                            "RSI", "ATR_pct", "Turnover_Cr"] if c in picks.columns]
        view = picks[cols].copy()
        if "Momentum" in view.columns:
            view["Momentum"] = view["Momentum"].map(lambda v: f"{v:+.1f}%")
        view = view.rename(columns={"ATR_pct": "ATR %", "Turnover_Cr": "₹Cr/day",
                                    "Close": "Price ₹", "Momentum": "12-1 mom",
                                    "Stop": "Stop ₹", "Stop_pct": "Stop %",
                                    "Target_1R": "1R ₹"})
        parts.append(_table(view, num_cols={"Rank", "Score", "12-1 mom", "Price ₹",
                                            "Stop ₹", "Stop %", "1R ₹",
                                            "RSI", "ATR %", "₹Cr/day"}))
        parts.append('<div class="note">Recorded to the forward log before outcomes '
                     'exist. Confirm every level on your broker terminal &mdash; these '
                     'are end-of-day prices.<br><strong>Score is a percentile within '
                     'today&rsquo;s universe, not an absolute quality measure</strong> '
                     '&mdash; read it alongside the 12-1 momentum figure.</div>')

    # --- Forward performance ---
    parts.append("<h2>Forward log performance</h2>")
    if not forward_summary or "error" in (forward_summary or {}):
        msg = (forward_summary or {}).get("error", "No forward data yet.")
        parts.append(f'<div class="empty">{html.escape(str(msg))}</div>')
    else:
        fs = forward_summary
        ic = fs.get("forward_ic")
        mean_r = fs.get("mean_return_pct", 0) or 0
        parts.append(_cards([
            ("Evaluated", fs.get("evaluated_picks", 0), ""),
            ("Snapshots", fs.get("snapshots", 0), ""),
            ("Mean return", f"{mean_r:+.2f}%", "pos" if mean_r > 0 else "neg"),
            ("Hit rate", f"{fs.get('hit_rate_pct', 0)}%", ""),
            ("Forward IC", f"{ic:+.4f}" if ic is not None else "—",
             "pos" if (ic or 0) > 0 else "neg"),
            ("Still open", fs.get("open_picks", 0), ""),
        ]))
        if (fs.get("snapshots") or 0) < 6:
            parts.append('<div class="note warn"><strong>Fewer than 6 snapshots.</strong> '
                         'Error bars are wide &mdash; do not draw conclusions yet. '
                         'Keep logging weekly.</div>')
        elif ic is not None and ic <= 0:
            parts.append('<div class="note warn"><strong>Forward IC is not positive.</strong> '
                         'The score is not ranking forward returns on live data. '
                         'Do not commit capital on this basis.</div>')

    if bucket_table is not None and not bucket_table.empty:
        parts.append("<h2>Return by score bucket</h2>")
        parts.append('<div class="note">The honest test: does return rise as score '
                     'rises? If not, the score is not ranking anything.</div>')
        parts.append(_table(bucket_table))

    if tier_table is not None and not tier_table.empty:
        parts.append("<h2>By market-cap tier</h2>")
        parts.append(_table(tier_table))

    if notes:
        parts.append("<h2>Run notes</h2><ul>")
        parts += [f"<li>{html.escape(str(n))}</li>" for n in notes]
        parts.append("</ul>")

    parts += [
        "</div>",
        '<div class="foot"><strong>Research output only. Not investment advice.</strong> '
        'No orders were placed by this process. Prices are end-of-day and delayed. '
        'A high score reflects chart properties, not a prediction. Verify independently '
        'before acting, and consider consulting a SEBI-registered adviser.</div>',
        "</div></body></html>",
    ]
    return "".join(parts)


def save(html_text: str, out_dir: Path, stamp: dt.datetime,
         want_pdf: bool = True) -> dict[str, Path]:
    """Write HTML (always) and PDF (if WeasyPrint is importable)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    html_path = out_dir / f"report_{stamp:%Y%m%d}.html"
    html_path.write_text(html_text, encoding="utf-8")
    (out_dir / "latest.html").write_text(html_text, encoding="utf-8")
    written["html"] = html_path

    if want_pdf:
        try:
            from weasyprint import HTML as WHTML
            pdf_path = out_dir / f"report_{stamp:%Y%m%d}.pdf"
            WHTML(string=html_text).write_pdf(str(pdf_path))
            (out_dir / "latest.pdf").write_bytes(pdf_path.read_bytes())
            written["pdf"] = pdf_path
        except Exception as exc:                       # noqa: BLE001
            # PDF is a nice-to-have; never fail a run over it.
            print(f"  (PDF skipped: {type(exc).__name__}: {exc})")

    return written


def email_report(
    html_text: str,
    attachments: dict[str, Path],
    *,
    smtp_host: str,
    smtp_port: int,
    user: str,
    password: str,
    to_addr: str,
    subject: str,
) -> bool:
    """Send the report by email. Returns True on success.

    For Gmail, use an App Password (not your account password) with
    smtp.gmail.com:465.
    """
    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr
    msg.set_content("Your SwingScope weekly report is attached. "
                    "View the HTML version for full formatting.")
    msg.add_alternative(html_text, subtype="html")

    for name, path in attachments.items():
        if not path.exists():
            continue
        data = path.read_bytes()
        if name == "pdf":
            msg.add_attachment(data, maintype="application", subtype="pdf",
                               filename=path.name)
        else:
            msg.add_attachment(data, maintype="text", subtype="html",
                               filename=path.name)

    try:
        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as s:
                s.login(user, password)
                s.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as s:
                s.starttls()
                s.login(user, password)
                s.send_message(msg)
        return True
    except Exception as exc:                           # noqa: BLE001
        print(f"  Email failed: {type(exc).__name__}: {exc}")
        return False
