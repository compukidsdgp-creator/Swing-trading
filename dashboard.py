"""Dashboard — a self-contained HTML view of the day's picks and levels.

Design constraints, and why they matter
---------------------------------------
No JavaScript, no external stylesheets, no CDN. Everything — layout, colours,
charts — is inline. Three consequences follow, and each is the reason:

  * It renders inside Streamlit's sandboxed component frame, where external
    scripts are blocked.
  * It survives Telegram's document handling, which strips nothing but also
    fetches nothing.
  * It opens correctly in three years on a machine with no network, which is
    the property that matters for something intended as a record.

Charts are hand-drawn SVG paths rather than a plotting library. That is more
code, but a Chart.js dependency would fail the first constraint and add a CDN
request to a file meant to be permanent.

What it shows
-------------
KPI strip (regime, active picks, stops breached, hit rate), the position table
with entry, stop, targets and current level, and a small price-versus-stop
chart per holding so the distance to the stop is visible rather than inferred.
"""

from __future__ import annotations

import datetime as dt
import html as _html
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("reports")

# Palette matches the app rather than being invented separately.
NAVY = "#171a1f"
SUB = "#6b7280"
GREEN = "#1f9d63"
AMBER = "#c9860a"
RED = "#c0392b"
BORDER = "#e6e9ef"
LIGHT = "#f8f9fb"

_CSS = """
:root{--nav:#171a1f;--sub:#6b7280;--brd:#e6e9ef;--lt:#f8f9fb;
      --grn:#1f9d63;--amb:#c9860a;--red:#c0392b}
*{box-sizing:border-box}
body{margin:0;background:#fff;color:var(--nav);
     font:14px/1.55 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:26px 22px 48px}
.top{display:flex;justify-content:space-between;align-items:flex-start;
     border-bottom:2px solid var(--brd);padding-bottom:14px;margin-bottom:18px}
.brand{display:flex;align-items:baseline;gap:9px}
h1{font-size:25px;margin:0;letter-spacing:-.4px}
h2{font-size:15px;margin:26px 0 10px;padding-bottom:5px;
   border-bottom:1px solid var(--brd)}
h3{font-size:13px;margin:0 0 6px;color:var(--sub);font-weight:600}
.tag{font-size:11px;background:var(--lt);border:1px solid var(--brd);
     border-radius:11px;padding:2px 9px;color:var(--sub)}
.meta{font-size:11.5px;color:var(--sub);text-align:right;line-height:1.7}
.controls{margin:0 0 16px}
.btn{font:inherit;font-size:12.5px;background:var(--nav);color:#fff;border:0;
     border-radius:6px;padding:7px 14px;cursor:pointer}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));
      gap:10px;margin-bottom:8px}
.kpi{background:var(--lt);border:1px solid var(--brd);border-radius:8px;
     padding:11px 13px}
.kv{font-size:21px;font-weight:600;letter-spacing:-.4px}
.kl{font-size:11.5px;color:var(--sub);margin-top:1px}
.kx{font-size:10.5px;color:var(--sub);opacity:.8;margin-top:2px}
table{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:6px}
th{background:var(--nav);color:#fff;text-align:left;padding:7px 9px;
   font-size:11px;font-weight:600;white-space:nowrap}
td{padding:6px 9px;border-bottom:1px solid var(--brd);white-space:nowrap}
tbody tr:nth-child(even){background:#fafbfc}
.num{text-align:right;font-variant-numeric:tabular-nums}
.pos{color:var(--grn)}.neg{color:var(--red)}.warn{color:var(--amb)}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));
        gap:14px;margin-top:8px}
.card{border:1px solid var(--brd);border-radius:8px;padding:12px 13px}
.note{background:var(--lt);border-left:3px solid var(--brd);padding:9px 12px;
      font-size:12px;color:var(--sub);border-radius:5px;margin:14px 0}
.foot{margin-top:28px;padding-top:12px;border-top:1px solid var(--brd);
      font-size:11px;color:var(--sub)}
@media print{.controls{display:none}.card{break-inside:avoid}}
"""


def _esc(x) -> str:
    return _html.escape(str(x)) if x is not None else ""


def _fmt(v, dp: int = 2, dash: str = "—") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return dash
    try:
        return f"{float(v):,.{dp}f}"
    except (TypeError, ValueError):
        return dash


def _spark(prices: pd.Series, stop: float, entry: float,
           w: int = 300, h: int = 96) -> str:
    """Price line with entry and stop reference levels.

    Hand-drawn SVG. A plotting library would need a CDN, and this file has to
    open with no network years from now.
    """
    p = pd.Series(prices).dropna().astype(float)
    if len(p) < 2:
        return '<div style="font-size:11px;color:#6b7280">No price history.</div>'

    vals = p.to_numpy()
    lo = float(min(vals.min(), stop)) * 0.995
    hi = float(max(vals.max(), entry)) * 1.005
    rng = hi - lo if hi > lo else 1.0
    pad = 8

    def y(v):
        return pad + (h - 2 * pad) * (1 - (float(v) - lo) / rng)

    def x(i):
        return pad + (w - 2 * pad) * (i / max(len(vals) - 1, 1))

    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
    last = float(vals[-1])
    colour = GREEN if last >= entry else RED

    return (
        f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" '
        f'xmlns="http://www.w3.org/2000/svg" role="img">'
        f'<line x1="{pad}" y1="{y(entry):.1f}" x2="{w-pad}" y2="{y(entry):.1f}" '
        f'stroke="{SUB}" stroke-width="1" stroke-dasharray="3 3"/>'
        f'<line x1="{pad}" y1="{y(stop):.1f}" x2="{w-pad}" y2="{y(stop):.1f}" '
        f'stroke="{RED}" stroke-width="1.2" stroke-dasharray="5 3"/>'
        f'<polyline points="{pts}" fill="none" stroke="{colour}" '
        f'stroke-width="1.8" stroke-linejoin="round"/>'
        f'<circle cx="{x(len(vals)-1):.1f}" cy="{y(last):.1f}" r="2.8" '
        f'fill="{colour}"/>'
        f'<text x="{w-pad}" y="{y(stop)-3:.1f}" text-anchor="end" '
        f'font-size="9" fill="{RED}">stop {stop:,.0f}</text>'
        f'<text x="{pad}" y="{y(entry)-3:.1f}" font-size="9" fill="{SUB}">'
        f'entry {entry:,.0f}</text>'
        f'</svg>')


def _kpi(value: str, label: str, extra: str = "", colour: str = NAVY) -> str:
    return (f'<div class="kpi"><div class="kv" style="color:{colour}">'
            f'{_esc(value)}</div><div class="kl">{_esc(label)}</div>'
            f'<div class="kx">{_esc(extra)}</div></div>')


def build(
    picks: pd.DataFrame,
    prices: dict[str, pd.DataFrame] | None = None,
    *,
    regime: str = "unknown",
    regime_desc: str = "",
    obs_date: dt.date | None = None,
    summary: dict | None = None,
    notes: list[str] | None = None,
) -> str:
    """Render the dashboard. Returns a complete standalone HTML document."""
    d = obs_date or dt.date.today()
    prices = prices or {}
    picks = picks if picks is not None else pd.DataFrame()

    reg_colour = {"risk_on": GREEN, "neutral": AMBER,
                  "risk_off": RED}.get(regime, SUB)

    # --- KPIs ---
    n = len(picks)
    breached = 0
    if n and {"Close", "Stop"}.issubset(picks.columns):
        breached = int((pd.to_numeric(picks["Close"], errors="coerce")
                        <= pd.to_numeric(picks["Stop"], errors="coerce")).sum())

    kpis = [
        _kpi(regime.replace("_", " ").upper(), "Regime gate",
             "momentum model", reg_colour),
        _kpi(str(n), "Active picks", regime_desc[:34], NAVY),
        _kpi(str(breached), "Stops breached", "current picks",
             RED if breached else GREEN),
    ]
    s = summary or {}
    if s.get("mean_pct") is not None:
        v = float(s["mean_pct"])
        kpis.append(_kpi(f"{v:+.2f}%", f"{s.get('horizon','1')}d mean",
                         f"median {_fmt(s.get('median_pct'))}%",
                         GREEN if v > 0 else RED if v < 0 else NAVY))
    if s.get("hit_rate_pct") is not None:
        kpis.append(_kpi(f"{float(s['hit_rate_pct']):.0f}%", "Hit rate",
                         f"n={s.get('observations','—')}"))

    # --- Table ---
    cols = [("Rank", "rank", 0), ("Ticker", "tkr", None), ("Tier", "tier", None),
            ("Entry ₹", "Close", 2), ("Stop ₹", "Stop", 2),
            ("Stop %", "Stop_pct", 2), ("1R ₹", "Target_1R", 2),
            ("2R ₹", "Target_2R", 2), ("Score", "Score", 0),
            ("Mom %", "Momentum", 0)]
    head = "".join(f"<th>{_esc(h)}</th>" for h, _, _ in cols)
    body = []
    for _, r in picks.iterrows():
        cells = []
        for label, key, dp in cols:
            if key == "tkr":
                cells.append(f'<td><b>{_esc(r.get("Ticker",""))}</b></td>')
            elif key in ("rank", "tier"):
                k = "Rank" if key == "rank" else "Tier"
                cells.append(f'<td>{_esc(r.get(k, ""))}</td>')
            else:
                v = r.get(key)
                cls = "num"
                if key == "Stop_pct" and v is not None and np.isfinite(
                        pd.to_numeric(v, errors="coerce")):
                    cls += " neg"
                cells.append(f'<td class="{cls}">{_fmt(v, dp or 0)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")

    table = (f'<table><thead><tr>{head}</tr></thead>'
             f'<tbody>{"".join(body)}</tbody></table>'
             if body else '<div class="note">No picks today.</div>')

    # --- Charts ---
    cards = []
    for _, r in picks.iterrows():
        t = str(r.get("Ticker", ""))
        frame = prices.get(t)
        if frame is None:
            frame = prices.get(f"{t}.NS")
        if frame is None or "Close" not in getattr(frame, "columns", []):
            continue
        entry = float(pd.to_numeric(r.get("Close"), errors="coerce") or 0)
        stop = float(pd.to_numeric(r.get("Stop"), errors="coerce") or 0)
        if entry <= 0 or stop <= 0:
            continue
        series = frame["Close"].tail(60)
        last = float(series.iloc[-1])
        chg = (last / entry - 1) * 100
        cls = "pos" if chg >= 0 else "neg"
        cards.append(
            f'<div class="card"><h3>{_esc(t)} '
            f'<span class="{cls}">{chg:+.2f}%</span></h3>'
            f'{_spark(series, stop, entry)}</div>')

    charts = (f'<div class="charts">{"".join(cards)}</div>' if cards else "")

    note_html = "".join(f'<div class="note">{_esc(x)}</div>'
                        for x in (notes or []))

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SwingScope Dashboard — {d}</title><style>{_CSS}</style></head><body>
<div class="wrap">
<header class="top">
  <div>
    <div class="brand"><h1>SwingScope</h1>
      <span class="tag">Daily Tracker</span>
      <span class="tag">Momentum model</span></div>
    <div style="font-size:12.5px;color:var(--sub);margin-top:6px">
      Stop-loss &amp; target dashboard · observation date <b>{d}</b></div>
  </div>
  <div class="meta">Generated <b>{dt.datetime.now():%d %b %Y %H:%M}</b><br>
    Regime <b>{_esc(regime)}</b></div>
</header>
<div class="controls"><button class="btn" onclick="window.print()">
  &#10515; Save as PDF</button></div>
<div class="kpis">{"".join(kpis)}</div>
{note_html}
<h2>Today's picks &amp; levels</h2>
{table}
{"<h2>Price vs stop</h2>" + charts if charts else ""}
<div class="foot">
  Research output only. Not investment advice. Prices are end-of-day and
  delayed — confirm every level on your broker terminal before acting.
  Stops are ATR-based: entry less a tier-specific multiple of the 14-day range.
</div>
</div></body></html>"""


def save(html_text: str, *, obs_date: dt.date | None = None,
         out_dir: Path = OUT_DIR) -> Path:
    d = obs_date or dt.date.today()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"dashboard_{d:%Y%m%d}.html"
    p.write_text(html_text, encoding="utf-8")
    return p


def build_from_tracker(*, regime: str = "unknown",
                       prices: dict | None = None) -> tuple[str, Path] | None:
    """Build from the daily tracker CSV, for standalone or scheduled use."""
    import daily_tracker as dtm

    obs, px = dtm.load()
    if obs.empty:
        return None

    latest = obs["obs_date"].astype(str).max()
    today = obs[obs["obs_date"].astype(str) == latest].dropna(subset=["ticker"])
    if today.empty:
        return None

    picks = pd.DataFrame({
        "Rank": today.get("rank"),
        "Ticker": today["ticker"],
        "Tier": today.get("tier"),
        "Score": today.get("score"),
        "Momentum": today.get("momentum_pct"),
        "Close": today.get("ref_close"),
        "Stop": today.get("stop_loss"),
        "Stop_pct": today.get("stop_pct"),
        "Target_1R": today.get("target_1r"),
        "Target_2R": today.get("target_2r"),
    })

    # Reconstruct price history from the tracker's own price log when no live
    # frames are supplied — keeps this usable offline.
    if prices is None and not px.empty:
        prices = {}
        px2 = px.copy()
        px2["date"] = pd.to_datetime(px2["date"], errors="coerce")
        for t, g in px2.dropna(subset=["date"]).groupby("ticker"):
            g = g.sort_values("date").set_index("date")
            prices[str(t)] = g.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume"})

    perf = dtm.performance(obs, px)
    summ = dtm.summarise(perf)
    summary = summ.iloc[0].to_dict() if not summ.empty else {}

    d = pd.Timestamp(latest).date()
    html_text = build(picks, prices, regime=regime, obs_date=d, summary=summary)
    return html_text, save(html_text, obs_date=d)
