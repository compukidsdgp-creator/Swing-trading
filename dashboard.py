"""Dashboard — self-contained HTML matching the approved reference design.

Design lock
-----------
This follows SwingScope_Dashboard.html exactly: mono font for all numbers,
the ink/sub/faint/line colour tokens, the 760x320 chart geometry, the same
zone shading and reference-line styling. Nothing here should drift from that
without deliberately updating this docstring too.

No JavaScript, no CDN, no external stylesheet — inline SVG and inline CSS
only. Renders inside Streamlit's sandboxed component frame, survives
Telegram's document handling, and opens on a machine with no network.

Data source discipline
-----------------------
Every row shown comes from the daily tracker CSV — the actual picks that were
made — never from a live-scanned universe. A dashboard showing synthetic or
re-screened stocks would silently disagree with what was actually tracked.

Bucket rollover
----------------
Each day's picks are tracked from their own observation date with their own
entry price and stop. Once a day's bucket passes the tracking window, that
whole day's rows leave together — not one stock at a time — so the report
always reflects complete buckets.
"""

from __future__ import annotations

import datetime as dt
import html as _html
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("reports")

# --- Colour tokens, taken verbatim from the reference ---
INK = "#171a1f"
SUB = "#5b636e"
FAINT = "#9aa2ad"
LINE = "#e5e7eb"
BLUE = "#2f4bd8"
GREEN = "#1f9d63"
RED = "#d64545"
AMBER = "#c9860a"
GREY_DOT = "#8a92a0"
ZONE_RISK = "#fbeaea"
ZONE_TARGET = "#e8f6ee"

PRE_ENTRY_BARS = 12          # within the 10-15 day range requested
TRACK_DAYS = 30
CONTINUATION_WINDOW = 30

_CSS = """
:root{
  --mono:"SFMono-Regular",ui-monospace,"Menlo","Consolas","Liberation Mono",monospace;
  --sans:"Helvetica Neue","Arial","Segoe UI",system-ui,sans-serif;
  --ink:#171a1f; --sub:#5b636e; --faint:#9aa2ad; --line:#e5e7eb;
}
*{box-sizing:border-box}
html{-webkit-print-color-adjust:exact;print-color-adjust:exact}
body{margin:0;background:#eef0f3;color:var(--ink);font-family:var(--sans);
  font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:28px 22px 60px}

.top{display:flex;justify-content:space-between;align-items:flex-end;
  border-bottom:2px solid var(--ink);padding-bottom:14px;margin-bottom:22px;gap:16px}
.brand{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
.brand h1{font-size:26px;margin:0;letter-spacing:-.02em;font-weight:800}
.brand .tag{font-family:var(--mono);font-size:11px;color:var(--sub);
  border:1px solid var(--line);padding:2px 7px;border-radius:3px;text-transform:uppercase;letter-spacing:.08em}
.top .meta{font-family:var(--mono);font-size:11px;color:var(--sub);text-align:right;line-height:1.7}
.top .meta b{color:var(--ink)}

.btn{font-family:var(--mono);font-size:12px;background:var(--ink);color:#fff;border:0;
  padding:9px 16px;border-radius:5px;cursor:pointer;letter-spacing:.03em}
.btn:hover{opacity:.88}
.controls{display:flex;justify-content:flex-end;margin:-6px 0 18px}

.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:1px;
  background:var(--line);border:1px solid var(--line);border-radius:8px;overflow:hidden;margin-bottom:26px}
.kpi{background:#fff;padding:14px 15px}
.kv{font-family:var(--mono);font-size:19px;font-weight:700;letter-spacing:-.01em}
.kl{font-size:11px;color:var(--sub);margin-top:3px;text-transform:uppercase;letter-spacing:.05em}
.kx{font-family:var(--mono);font-size:10px;color:var(--faint);margin-top:2px}

.sh{display:flex;align-items:baseline;gap:10px;margin:30px 0 12px}
.sh h2{font-size:15px;margin:0;text-transform:uppercase;letter-spacing:.09em}
.sh .rule{flex:1;height:1px;background:var(--line)}
.sh .n{font-family:var(--mono);font-size:11px;color:var(--faint)}

.panel{background:#fff;border:1px solid var(--line);border-radius:8px;overflow:hidden}
table{width:100%;border-collapse:collapse;font-size:12.5px}
thead th{font-family:var(--mono);font-size:10px;text-transform:uppercase;letter-spacing:.06em;
  color:var(--sub);text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);background:#fafbfc;font-weight:600}
thead th.num{text-align:right}
tbody td{padding:10px 12px;border-bottom:1px solid #f1f2f4}
tbody tr:last-child td{border-bottom:0}
.num{font-family:var(--mono);text-align:right}
.strong{font-weight:700}
.tk{font-family:var(--mono);font-weight:700}
.mut{color:var(--sub)}
tr.tr-breach{background:#fbeaea}
tr.tr-near{background:#fff7ec}

.badge{font-family:var(--mono);font-size:10px;font-weight:700;padding:2px 8px;border-radius:20px;
  white-space:nowrap;letter-spacing:.02em;display:inline-block}
.b-hold{background:#eef2fb;color:#2f4bd8}
.b-target{background:#e8f6ee;color:#1f9d63}
.b-near{background:#fdf1dc;color:#c9860a}
.b-breach{background:#fbeaea;color:#d64545}
.b-expired{background:#f1eefb;color:#6b46c1}
.b-no-stop{background:#f1f2f4;color:var(--sub)}
.warn-icon{color:#c9860a;font-weight:700;margin-left:4px}

.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:16px 16px 12px;
  break-inside:avoid;page-break-inside:avoid}
.card-head{display:flex;justify-content:space-between;align-items:center}
.ch-l{display:flex;align-items:baseline;gap:9px}
.card h3{font-family:var(--mono);font-size:16px;margin:0;font-weight:700}
.sector{font-size:11px;color:var(--sub)}
.quote{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin:8px 0 4px}
.px{font-family:var(--mono);font-size:22px;font-weight:700}
.chg{font-family:var(--mono);font-size:12px;font-weight:600}
.chgp{color:var(--faint);font-weight:400}
.spark{font-family:var(--mono);font-size:10.5px;color:var(--faint);margin-left:auto}
.chart{width:100%;height:auto;display:block;margin:4px 0 6px}
.metrics{display:flex;flex-wrap:wrap;gap:6px 0;border-top:1px solid #f1f2f4;padding-top:10px}
.m{flex:1 1 33%;min-width:90px;display:flex;flex-direction:column;gap:1px}
.ml{font-size:10px;color:var(--faint);text-transform:uppercase;letter-spacing:.04em}
.mv{font-family:var(--mono);font-size:13px;font-weight:600}
.nolev{flex:1;font-size:11.5px;color:var(--sub);background:#fafbfc;border:1px dashed var(--line);
  border-radius:6px;padding:8px 10px;line-height:1.4}

.range{position:relative;height:5px;background:var(--line);border-radius:3px;margin:9px 0 3px}
.range .fill{position:absolute;top:0;bottom:0;background:#c3cad6;border-radius:3px}
.range .mark{position:absolute;top:-2px;width:2px;height:9px;background:var(--ink)}
.range-lbl{display:flex;justify-content:space-between;font-family:var(--mono);
  font-size:10px;color:var(--faint)}

.runs{margin-top:8px;padding-top:8px;border-top:1px solid #f1f2f4}
.runs-lbl{font-size:10px;color:var(--faint);text-transform:uppercase;letter-spacing:.04em;margin-bottom:3px}

.news{margin-top:8px;padding-top:8px;border-top:1px solid #f1f2f4}
.news ul{margin:0;padding:0}
.news li{font-size:11px;margin-bottom:4px;list-style:none;line-height:1.4;color:var(--sub)}
.nb{font-family:var(--mono);font-size:9px;padding:1px 5px;border-radius:8px;margin-right:5px;
    text-transform:uppercase}
.n-pos{background:#e8f6ee;color:#1f9d63}
.n-neg{background:#fbeaea;color:#d64545}
.n-neu{background:#f1f2f4;color:var(--sub)}

.eqwrap{background:#fff;border:1px solid var(--line);border-radius:8px;padding:14px 16px}

.legend{display:flex;flex-wrap:wrap;gap:16px;font-family:var(--mono);font-size:11px;
  color:var(--sub);margin:14px 2px 0}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:16px;height:0;border-top-width:2px;border-top-style:solid;display:inline-block}
.zone{width:14px;height:10px;border-radius:2px;display:inline-block}
.foot{margin-top:30px;border-top:1px solid var(--line);padding-top:16px;
  font-size:11.5px;color:var(--sub);line-height:1.6}
.foot b{color:var(--ink)}
.disclaim{background:#fffbeb;border:1px solid #f2e2b8;border-radius:8px;padding:12px 14px;
  font-size:12px;color:#7a5b12;margin-top:14px}

@media (max-width:820px){
  .kpis{grid-template-columns:repeat(3,1fr)}
  .grid{grid-template-columns:1fr}
  .top{flex-direction:column;align-items:flex-start}
  .top .meta{text-align:left}
}
@media print{
  body{background:#fff}
  .wrap{max-width:none;padding:0 6px}
  .controls{display:none}
  .grid{gap:12px}
  @page{margin:12mm}
}
"""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _esc(x) -> str:
    return _html.escape(str(x)) if x is not None else ""


def _fmt(v, dp: int = 2, dash: str = "—") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return dash
    try:
        return f"{float(v):,.{dp}f}"
    except (TypeError, ValueError):
        return dash


def _first_present(*values, default=""):
    """First value that is neither None nor NaN.

    Not `a or b` — that treats an empty string or zero as absent, and for a
    sector name or a price of zero those are legitimate values.
    """
    for v in values:
        if v is None:
            continue
        if isinstance(v, float) and not np.isfinite(v):
            continue
        s = str(v).strip()
        if s and s.lower() not in ("nan", "none"):
            return s
    return default


def _kpi(value: str, label: str, extra: str = "", colour: str = INK) -> str:
    return (f'<div class="kpi"><div class="kv" style="color:{colour}">'
            f'{_esc(value)}</div><div class="kl">{_esc(label)}</div>'
            f'<div class="kx">{_esc(extra)}</div></div>')


# ---------------------------------------------------------------------------
# The price chart — 760x320, matching the reference geometry exactly
# ---------------------------------------------------------------------------
def _price_chart(
    dates: pd.DatetimeIndex,
    closes: np.ndarray,
    *,
    entry: float,
    stop: float | None,
    target_1r: float | None,
    target_2r: float | None,
    stopped: bool = False,
    w: int = 760,
    h: int = 320,
) -> str:
    """Price line with entry/stop/target reference lines and zone shading.

    Geometry fixed to match the reference: plot area x in [58, 664], y in
    [20, 280]. Five gridlines with right-aligned price labels, a risk zone
    below the stop, a target zone above 1R, reference lines with right-side
    labels, and a date axis with four evenly spaced labels.
    """
    n = len(closes)
    if n < 2:
        return ('<div style="font-size:11px;color:#9aa2ad;padding:40px 0;'
                'text-align:center">No price history.</div>')

    px0, px1 = 58, 664
    py0, py1 = 20, 280
    plot_w, plot_h = px1 - px0, py1 - py0

    levels = [v for v in (entry, stop, target_1r, target_2r, closes.min(),
                          closes.max()) if v is not None and np.isfinite(v)]
    lo, hi = min(levels), max(levels)
    pad = (hi - lo) * 0.06 if hi > lo else max(hi * 0.02, 1)
    lo, hi = lo - pad, hi + pad
    rng = hi - lo if hi > lo else 1.0

    def y(v):
        return py0 + plot_h * (1 - (v - lo) / rng)

    def x(i):
        return px0 + plot_w * (i / max(n - 1, 1))

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart" '
            f'xmlns="http://www.w3.org/2000/svg" font-family="var(--mono)">']

    if stop is not None and np.isfinite(stop):
        sy = y(stop)
        parts.append(f'<rect x="{px0}" y="{sy:.1f}" width="{plot_w}" '
                     f'height="{py1-sy:.1f}" fill="{ZONE_RISK}"/>')
    if target_1r is not None and np.isfinite(target_1r):
        ty = y(target_1r)
        parts.append(f'<rect x="{px0}" y="{py0}" width="{plot_w}" '
                     f'height="{ty-py0:.1f}" fill="{ZONE_TARGET}"/>')

    for i in range(5):
        gy = py0 + (plot_h / 4) * i
        gv = hi - (rng / 4) * i
        parts.append(f'<line x1="{px0}" y1="{gy:.1f}" x2="{px1}" y2="{gy:.1f}" '
                     f'stroke="{LINE}" stroke-width="1"/>')
        parts.append(f'<text x="{px0-8}" y="{gy+3:.1f}" text-anchor="end" '
                     f'font-size="10" fill="{FAINT}">{gv:,.0f}</text>')

    top_pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(closes))
    bot_pts = " ".join(f"{x(i):.1f},{py1}" for i in range(n - 1, -1, -1))
    parts.append(f'<polygon points="{top_pts} {bot_pts}" fill="{BLUE}" '
                 f'opacity="0.06"/>')

    def ref_line(level, colour, label, dash, width):
        if level is None or not np.isfinite(level):
            return
        ly = y(level)
        parts.append(f'<line x1="{px0}" y1="{ly:.1f}" x2="{px1}" y2="{ly:.1f}" '
                     f'stroke="{colour}" stroke-width="{width}" '
                     f'stroke-dasharray="{dash}"/>')
        parts.append(f'<text x="{px1+6}" y="{ly-3:.1f}" font-size="9.5" '
                     f'fill="{colour}" font-weight="600">{label}</text>')
        parts.append(f'<text x="{px1+6}" y="{ly+8:.1f}" font-size="9.5" '
                     f'fill="{colour}">{level:,.2f}</text>')

    ref_line(target_2r, GREEN, "TARGET 2R", "5 4", 1.6)
    ref_line(target_1r, GREEN, "TARGET 1R", "5 4", 1.6)
    ref_line(entry, GREY_DOT, "ENTRY", "2 3", 1.2)
    ref_line(stop, RED, "STOP-LOSS", "5 4", 1.8)

    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(closes))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{BLUE}" '
                 f'stroke-width="2.2" stroke-linejoin="round" '
                 f'stroke-linecap="round"/>')
    for i, v in enumerate(closes):
        parts.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="2.1" '
                     f'fill="{BLUE}"/>')

    if stop is not None and np.isfinite(stop):
        worst_i = int(np.argmin(closes))
        wv = float(closes[worst_i])
        parts.append(f'<circle cx="{x(worst_i):.1f}" cy="{y(wv):.1f}" r="5.5" '
                     f'fill="none" stroke="{RED}" stroke-width="2"/>')
        tag = "STOPPED" if stopped else "LOW"
        parts.append(f'<text x="{x(worst_i):.1f}" y="{y(wv)+18:.1f}" '
                     f'text-anchor="middle" font-size="9" font-weight="700" '
                     f'fill="{RED}">{tag} {wv:,.0f}</text>')

    lv = float(closes[-1])
    parts.append(f'<circle cx="{x(n-1):.1f}" cy="{y(lv):.1f}" r="4" '
                 f'fill="{BLUE}" stroke="#fff" stroke-width="1.5"/>')

    idxs = sorted({0, n // 3, (2 * n) // 3, n - 1})
    for i in idxs:
        d = dates[i]
        parts.append(f'<text x="{x(i):.1f}" y="{py1+16}" text-anchor="middle" '
                     f'font-size="9" fill="{FAINT}">{d:%d %b}</text>')

    parts.append('</svg>')
    return "".join(parts)


# ---------------------------------------------------------------------------
# Momentum run histogram — a countable structural property, not a forecast
# ---------------------------------------------------------------------------
def momentum_runs(closes: pd.Series, *, window: int = CONTINUATION_WINDOW,
                  min_run: int = 10) -> dict:
    """Distribution of momentum run lengths for one stock, historically.

    Counts stretches of consecutive sessions where the trailing `window`-day
    return stays positive. This is a structural property of the series — some
    stocks sustain trends for months, others flicker in and out — and counting
    it does not carry the noise problem a conditional-probability version had:
    on synthetic series with known structure, "conditional continuation rate"
    could not distinguish a trending series from a mean-reverting one. Counting
    run lengths is a direct read of the series rather than an inference, so it
    does not inherit that problem.

    Says nothing about what happens next. It describes the stock's history,
    not this trade's odds.
    """
    c = pd.Series(closes).dropna().astype(float)
    if len(c) < window * 3:
        return {"runs": 0, "lengths": [], "note": "Insufficient history."}

    trailing = c / c.shift(window) - 1
    positive = (trailing > 0).to_numpy()

    lengths, cur = [], 0
    for v in positive:
        if v:
            cur += 1
        else:
            if cur >= min_run:
                lengths.append(cur)
            cur = 0
    if cur >= min_run:
        lengths.append(cur)

    years = len(c) / 252
    if not lengths:
        return {"runs": 0, "lengths": [], "years": round(years, 1),
                "note": f"No run of {min_run}+ sessions in {years:.1f} years."}

    arr = np.array(lengths)
    return {
        "runs": len(arr), "lengths": [int(x) for x in arr],
        "median": int(np.median(arr)), "longest": int(arr.max()),
        "runs_per_year": round(len(arr) / years, 1) if years else 0,
        "pct_time_in_run": round(float(arr.sum()) / len(c) * 100, 1),
        "years": round(years, 1),
        "over_30": int((arr >= 30).sum()),
    }


def _run_histogram(stats: dict, w: int = 300, h: int = 54) -> str:
    """Small bucketed bar chart of run lengths."""
    lengths = stats.get("lengths") or []
    if not lengths:
        return ""
    buckets = [(10, 20), (20, 30), (30, 45), (45, 60), (60, 90), (90, 10_000)]
    labels = ["10-20", "20-30", "30-45", "45-60", "60-90", "90+"]
    counts = [sum(1 for x in lengths if lo <= x < hi) for lo, hi in buckets]
    peak = max(counts) or 1
    bw = (w - 16) / len(counts)
    bars = []
    for i, (cnt, lab) in enumerate(zip(counts, labels)):
        bh = (h - 20) * (cnt / peak)
        bx, by = 8 + i * bw, h - 16 - bh
        colour = GREEN if i >= 2 else "#c3cad6"
        bars.append(f'<rect x="{bx+2:.1f}" y="{by:.1f}" width="{bw-4:.1f}" '
                    f'height="{max(bh,0.5):.1f}" fill="{colour}" rx="1.5"/>')
        if cnt:
            bars.append(f'<text x="{bx+bw/2:.1f}" y="{by-2:.1f}" '
                        f'text-anchor="middle" font-size="8" fill="{SUB}" '
                        f'font-family="var(--mono)">{cnt}</text>')
        bars.append(f'<text x="{bx+bw/2:.1f}" y="{h-4:.1f}" text-anchor="middle" '
                    f'font-size="7.5" fill="{FAINT}" font-family="var(--mono)">'
                    f'{lab}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" '
            f'xmlns="http://www.w3.org/2000/svg">{"".join(bars)}</svg>')


def _range_bar(last: float, lo: float, hi: float) -> str:
    if not all(np.isfinite(x) for x in (last, lo, hi)) or hi <= lo:
        return ""
    pct = float(np.clip((last - lo) / (hi - lo), 0, 1))
    return (f'<div class="range"><div class="fill" style="width:{pct*100:.0f}%">'
            f'</div><div class="mark" style="left:calc({pct*100:.0f}% - 1px)">'
            f'</div></div><div class="range-lbl"><span>{lo:,.0f}</span>'
            f'<span>{pct*100:.0f}% of 52w range</span><span>{hi:,.0f}</span></div>')


_NEWS_CACHE: dict = {}


def fetch_news(tickers, *, limit: int = 3, use_cache: bool = True) -> dict:
    """Top headlines per distinct ticker, sentiment-scored and cached.

    Cached per ticker per session, since the rolling window holds the same
    name on many rows and one RSS request per distinct ticker is enough.
    """
    try:
        import newsfeed
        import sentiment as sent
    except ImportError:
        return {}
    out = {}
    for t in {str(x).replace(".NS", "") for x in tickers if x}:
        if use_cache and t in _NEWS_CACHE:
            out[t] = _NEWS_CACHE[t]
            continue
        try:
            heads = newsfeed.fetch(t, limit=8)
        except Exception:                                      # noqa: BLE001
            heads = []
        scored = []
        for hd in heads:
            title = str(hd.get("title", ""))
            if not title:
                continue
            try:
                s = sent.score_text(title)
                lab, score = s.label, s.score
            except Exception:                                  # noqa: BLE001
                lab, score = "neu", 0.0
            scored.append({"title": title, "label": lab, "score": score})
        scored.sort(key=lambda z: -z["score"])
        top = scored[:limit]
        _NEWS_CACHE[t] = top
        out[t] = top
    return out


def _news_block(items: list) -> str:
    if not items:
        return ""
    rows = []
    for it in items[:3]:
        lab = str(it.get("label", "neu")).lower()[:3]
        cls = {"pos": "n-pos", "neg": "n-neg"}.get(lab, "n-neu")
        title = _esc(str(it.get("title", ""))[:90])
        rows.append(f'<li><span class="nb {cls}">{lab}</span>{title}</li>')
    return f'<div class="news"><ul>{"".join(rows)}</ul></div>'


def _equity_curve(r_values: list, w: int = 1030, h: int = 130) -> str:
    if not r_values or len(r_values) < 2:
        return ""
    cum = np.cumsum(np.asarray(r_values, dtype=float))
    lo, hi = float(min(cum.min(), 0)), float(max(cum.max(), 0))
    rng = hi - lo if hi > lo else 1.0
    pad = 12

    def y(v):
        return pad + (h - 2 * pad) * (1 - (v - lo) / rng)

    def x(i):
        return pad + (w - 2 * pad) * (i / max(len(cum) - 1, 1))

    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(cum))
    final = float(cum[-1])
    colour = GREEN if final >= 0 else RED
    zero = y(0.0)
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" '
            f'xmlns="http://www.w3.org/2000/svg" font-family="var(--mono)">'
            f'<line x1="{pad}" y1="{zero:.1f}" x2="{w-pad}" y2="{zero:.1f}" '
            f'stroke="{LINE}" stroke-width="1" stroke-dasharray="3 3"/>'
            f'<polyline points="{pts}" fill="none" stroke="{colour}" '
            f'stroke-width="2" stroke-linejoin="round"/>'
            f'<circle cx="{x(len(cum)-1):.1f}" cy="{y(final):.1f}" r="3.6" '
            f'fill="{colour}"/>'
            f'<text x="{w-pad}" y="{y(final)-9:.1f}" text-anchor="end" '
            f'font-size="12" font-weight="700" fill="{colour}">{final:+.2f}R'
            f'</text></svg>')


# ---------------------------------------------------------------------------
# Card and table builders
# ---------------------------------------------------------------------------
def _status_of(*, stop, low, high, target_1r, target_2r, held, track_days,
              latest_close) -> str:
    if stop is not None and np.isfinite(stop) and low <= stop:
        return "Stopped"
    if target_2r is not None and np.isfinite(target_2r) and high >= target_2r:
        return "Target hit"
    if target_1r is not None and np.isfinite(target_1r) and high >= target_1r:
        return "Target hit"
    if held >= track_days:
        return "Expired"
    if stop is None or not np.isfinite(stop):
        return "No active stop"
    if stop is not None and np.isfinite(stop) and latest_close and (latest_close / stop - 1) < 0.03:
        return "Near stop"
    return "Holding"


_BADGE = {"Holding": "b-hold", "Target hit": "b-target", "Near stop": "b-near",
         "Stopped": "b-breach", "Expired": "b-expired",
         "No active stop": "b-no-stop"}
_ROWCLS = {"Stopped": "tr-breach", "Near stop": "tr-near"}


def _table(rows: pd.DataFrame, *, capital: float, risk_pct: float) -> str:
    # Defensive: a caller (e.g. a Streamlit widget mid-edit, which can briefly
    # hold None) might pass a non-numeric capital or risk_pct. Rather than
    # crash the whole render over a Qty column, fall back to sane defaults —
    # this exact TypeError was reproduced from `capital=None` reaching here.
    capital = capital if isinstance(capital, (int, float)) and np.isfinite(capital) and capital > 0 else 500_000.0
    risk_pct = risk_pct if isinstance(risk_pct, (int, float)) and np.isfinite(risk_pct) and risk_pct > 0 else 1.0
    headers = ["Ticker", "Sector", "Entry", "Last", "Stop", "Qty", "1R", "2R",
               "Room→stop", "Max DD", "Held", "Status"]
    num_cols = {"Entry", "Last", "Stop", "Qty", "1R", "2R", "Room→stop", "Max DD"}
    head = "".join(f'<th{" class=" + chr(34) + "num" + chr(34) if h in num_cols else ""}>{_esc(h)}</th>'
                   for h in headers)

    body = []
    for _, r in rows.iterrows():
        entry = pd.to_numeric(r.get("Entry"), errors="coerce")
        last = pd.to_numeric(r.get("Last"), errors="coerce")
        stop = pd.to_numeric(r.get("Stop"), errors="coerce")
        status = str(r.get("Status") or "Holding")

        qty = pd.to_numeric(r.get("Qty"), errors="coerce")
        if not (qty is not None and np.isfinite(qty)):
            risk_ps = (entry - stop) if (np.isfinite(entry) and np.isfinite(stop)) else np.nan
            qty = (np.floor(capital * risk_pct / 100.0 / risk_ps)
                   if np.isfinite(risk_ps) and risk_ps > 0 else np.nan)
        qty_txt = f"{int(qty):,}" if np.isfinite(qty) else "—"

        room = ((last / stop - 1) * 100 if np.isfinite(last) and np.isfinite(stop)
                and stop else np.nan)
        maxdd = pd.to_numeric(r.get("Max_DD"), errors="coerce")
        held = r.get("Days_held")
        held_txt = (f"{int(held)}d" if held is not None
                    and np.isfinite(pd.to_numeric(held, errors="coerce"))
                    else "—")
        earn = (' <span class="warn-icon" title="Reports inside the holding '
               'window">&#9888;</span>'
               if str(r.get("Earnings_flag") or "").lower() == "avoid" else "")

        row_cls = _ROWCLS.get(status, "")
        badge_cls = _BADGE.get(status, "b-hold")

        body.append(
            f'<tr class="{row_cls}">'
            f'<td class="tk">{_esc(r.get("Ticker",""))}{earn}</td>'
            f'<td class="mut">{_esc(r.get("Sector") or "—")}</td>'
            f'<td class="num">{_fmt(entry)}</td>'
            f'<td class="num strong">{_fmt(last)}</td>'
            f'<td class="num" style="color:{RED}">{_fmt(stop)}</td>'
            f'<td class="num strong">{qty_txt}</td>'
            f'<td class="num" style="color:{GREEN}">{_fmt(r.get("Target_1R"))}</td>'
            f'<td class="num" style="color:{GREEN}">{_fmt(r.get("Target_2R"))}</td>'
            f'<td class="num">{("%+.1f%%" % room) if np.isfinite(room) else "—"}</td>'
            f'<td class="num" style="color:{RED}">'
            f'{("%.2f%%" % maxdd) if np.isfinite(maxdd) else "—"}</td>'
            f'<td class="num mut">{held_txt}</td>'
            f'<td><span class="badge {badge_cls}">{_esc(status)}</span></td>'
            f'</tr>')

    if not body:
        return '<div class="panel" style="padding:20px;color:var(--sub)">No positions.</div>'
    return (f'<div class="panel"><table><thead><tr>{head}</tr></thead>'
           f'<tbody>{"".join(body)}</tbody></table></div>')


def _card(row: dict, *, show_runs: bool = False, news: dict | None = None) -> str:
    """One stock card: header, quote line, chart, 52w bar, runs, news, metrics."""
    tkr = row["Ticker"]
    entry = row.get("Entry")
    last = row.get("Last")
    stop = row.get("Stop")
    t1, t2 = row.get("Target_1R"), row.get("Target_2R")
    status = row.get("Status", "Holding")
    dates = row.get("ChartDates")
    closes = row.get("ChartCloses")

    chg = ((last / entry - 1) * 100) if (entry and np.isfinite(entry)
                                         and last and np.isfinite(last)) else 0.0
    chg_colour = GREEN if chg >= 0 else RED

    earn = (' <span class="warn-icon">&#9888;</span>'
           if str(row.get("Earnings_flag") or "").lower() == "avoid" else "")

    if dates is not None and closes is not None and len(closes) >= 2:
        chart = _price_chart(dates, np.asarray(closes), entry=entry, stop=stop,
                             target_1r=t1, target_2r=t2,
                             stopped=(status == "Stopped"))
    else:
        chart = ('<div style="font-size:11px;color:#9aa2ad;padding:40px 0;'
                 'text-align:center">Awaiting price history.</div>')

    hi52, lo52 = row.get("High_52w"), row.get("Low_52w")
    range_html = (_range_bar(last, lo52, hi52)
                 if all(v is not None and np.isfinite(v) for v in (last, lo52, hi52))
                 else "")

    runs_html = ""
    if show_runs and row.get("RunStats"):
        rs = row["RunStats"]
        if rs.get("runs"):
            runs_html = (
                f'<div class="runs"><div class="runs-lbl">Momentum runs '
                f'({rs["years"]}y history) — {rs["runs"]} run(s) of 10+ days, '
                f'{rs["pct_time_in_run"]:.0f}% of time trending, longest '
                f'{rs["longest"]}d</div>{_run_histogram(rs)}</div>')

    news_html = _news_block((news or {}).get(tkr, [])) if news else ""

    room = ((last / stop - 1) * 100 if last and stop and np.isfinite(last)
            and np.isfinite(stop) else np.nan)
    maxdd = row.get("Max_DD")
    maxgain = row.get("Max_Gain")

    if stop is not None and np.isfinite(stop):
        metrics = (
            f'<div class="m"><span class="ml">Entry</span>'
            f'<span class="mv" style="color:{SUB}">{_fmt(entry)}</span></div>'
            f'<div class="m"><span class="ml">Stop-loss</span>'
            f'<span class="mv" style="color:{RED}">{_fmt(stop)}</span></div>'
            f'<div class="m"><span class="ml">Target 1R</span>'
            f'<span class="mv" style="color:{GREEN}">{_fmt(t1)}</span></div>'
            f'<div class="m"><span class="ml">Target 2R</span>'
            f'<span class="mv" style="color:{GREEN}">{_fmt(t2)}</span></div>'
            f'<div class="m"><span class="ml">Room to stop</span>'
            f'<span class="mv" style="color:{INK}">'
            f'{("%+.1f%%" % room) if np.isfinite(room) else "—"}</span></div>'
            f'<div class="m"><span class="ml">Max gain</span>'
            f'<span class="mv" style="color:{GREEN}">'
            f'{("%+.2f%%" % maxgain) if maxgain is not None and np.isfinite(maxgain) else "—"}'
            f'</span></div>'
            f'<div class="m"><span class="ml">Max drawdown</span>'
            f'<span class="mv" style="color:{RED}">'
            f'{("%.2f%%" % maxdd) if maxdd is not None and np.isfinite(maxdd) else "—"}'
            f'</span></div>')
    else:
        metrics = ('<div class="nolev">No stop recorded for this pick — '
                  'levels unavailable.</div>')

    qty = row.get("Qty")
    qty_html = (f'<div class="m"><span class="ml">Qty</span>'
               f'<span class="mv" style="color:{INK}">{int(qty):,}</span></div>'
               if qty is not None and np.isfinite(qty) else "")

    return (
        f'<section class="card"><div class="card-head"><div class="ch-l">'
        f'<h3>{_esc(tkr)}{earn}</h3>'
        f'<span class="sector">{_esc(row.get("Sector") or "")}</span></div>'
        f'<div class="ch-r"><span class="badge {_BADGE.get(status,"b-hold")}">'
        f'{_esc(status)}</span></div></div>'
        f'<div class="quote"><span class="px">{_fmt(last)}</span>'
        f'<span class="chg" style="color:{chg_colour}">{chg:+.2f}% '
        f'<span class="chgp">over window</span></span>'
        f'<span class="spark">score {_fmt(row.get("Score"),0)} · '
        f'momentum {_fmt(row.get("Momentum"),0)}%</span></div>'
        f'{chart}{range_html}{runs_html}{news_html}'
        f'<div class="metrics">{metrics}{qty_html}</div></section>')


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------
def _page(*, title_date, regime, kpis_html, table_html, cards_html,
          equity_html, notes, source_stamp) -> str:
    note_html = "".join(f'<div class="disclaim">{_esc(n)}</div>' for n in (notes or []))
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SwingScope Dashboard — {title_date}</title><style>{_CSS}</style></head>
<body><div class="wrap">
<header class="top">
  <div><div class="brand"><h1>SwingScope</h1>
    <span class="tag">Daily Tracker</span><span class="tag">Momentum model</span></div>
    <div style="font-size:12.5px;color:var(--sub);margin-top:6px">
      Stop-loss &amp; target dashboard · observation date <b>{title_date}</b></div>
  </div>
  <div class="meta">Source generated <b>{source_stamp}</b><br>
    Dashboard built <b>{dt.datetime.now():%d %b %Y}</b><br>
    Regime <b>{_esc(regime)}</b></div>
</header>
<div class="controls"><button class="btn" onclick="window.print()">&#8659; Download PDF</button></div>
<div class="kpis">{kpis_html}</div>
{note_html}
{equity_html}
<div class="sh"><h2>Positions</h2><div class="rule"></div></div>
{table_html}
<div class="sh"><h2>Price &amp; stop-loss charts</h2><div class="rule"></div></div>
<div class="grid">{cards_html}</div>
<div class="legend">
  <span><i class="sw" style="border-color:{BLUE}"></i>Close price</span>
  <span><i class="sw" style="border-color:{RED};border-top-style:dashed"></i>Stop-loss</span>
  <span><i class="sw" style="border-color:{GREEN};border-top-style:dashed"></i>Target 1R / 2R</span>
  <span><i class="sw" style="border-color:{GREY_DOT};border-top-style:dotted"></i>Entry</span>
  <span><i class="zone" style="background:{ZONE_RISK}"></i>Risk zone (below stop)</span>
  <span><i class="zone" style="background:{ZONE_TARGET}"></i>Target zone</span>
  <span>&#9675; ringed point = closest approach to stop</span>
</div>
<div class="disclaim"><b>Read this first.</b> This is a diary, not statistical
  evidence. Consecutive buckets share most picks and nearly all of their
  holding window, so returns are correlated and hit rates cannot be read as
  significance. Use it to watch behaviour, not to judge edge.</div>
<div class="foot"><b>How to read the charts.</b> Each line is the daily close
  over the tracked window, starting """ + str(PRE_ENTRY_BARS) + """ sessions before entry.
  The red dashed line is the stop-loss with the risk zone shaded below it; the
  green dashed lines are the 1R and 2R targets with the target zone shaded
  above 1R. Research output only. Not investment advice. Prices are
  end-of-day and delayed — confirm every level on your broker terminal before
  acting.</div>
</div></body></html>"""


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------
def build(picks: pd.DataFrame, prices: dict[str, pd.DataFrame] | None = None,
          *, regime: str = "unknown", regime_desc: str = "",
          obs_date: dt.date | None = None, summary: dict | None = None,
          notes: list[str] | None = None, capital: float = 500_000.0,
          risk_pct: float = 1.0) -> str:
    """Today's picks only — the daily dashboard."""
    d = obs_date or dt.date.today()
    prices = prices or {}
    picks = picks if picks is not None else pd.DataFrame()
    n = len(picks)

    breached = 0
    rows = []
    for _, r in picks.iterrows():
        tkr = str(r.get("Ticker", ""))
        frame = prices.get(tkr)
        if frame is None:
            frame = prices.get(f"{tkr}.NS")
        entry = pd.to_numeric(r.get("Close"), errors="coerce")
        stop = pd.to_numeric(r.get("Stop"), errors="coerce")

        chart_dates = chart_closes = None
        last = entry
        maxdd = maxgain = np.nan
        if frame is not None and "Close" in frame.columns:
            hist = frame["Close"].dropna()
            if len(hist):
                entry_i = hist.index.get_indexer([pd.Timestamp(d)], method="nearest")[0]
            else:
                entry_i = 0
            start_i = max(0, entry_i - PRE_ENTRY_BARS)
            window = hist.iloc[start_i:entry_i + 1]
            if len(window) >= 2:
                chart_dates, chart_closes = window.index, window.to_numpy()
                last = float(window.iloc[-1])
                if np.isfinite(entry) and entry:
                    maxdd = (float(window.min()) / entry - 1) * 100
                    maxgain = (float(window.max()) / entry - 1) * 100
        if np.isfinite(stop) and np.isfinite(last) and last <= stop:
            breached += 1

        status = "Holding" if np.isfinite(stop) else "No active stop"
        rows.append({
            "Ticker": tkr, "Sector": r.get("Sector", "—"),
            "Entry": entry, "Last": last, "Stop": stop,
            "Target_1R": r.get("Target_1R"), "Target_2R": r.get("Target_2R"),
            "Max_DD": maxdd, "Max_Gain": maxgain, "Days_held": 0,
            "Status": status, "ChartDates": chart_dates,
            "ChartCloses": chart_closes, "Score": r.get("Score"),
            "Momentum": r.get("Momentum"),
            "Earnings_flag": r.get("Earnings_flag"),
            "High_52w": r.get("High_52w"), "Low_52w": r.get("Low_52w"),
            "Qty": r.get("Qty"),
        })

    kpis = [
        _kpi(regime.replace("_", " ").upper(), "Regime gate", "momentum model",
             {"risk_on": GREEN, "neutral": AMBER,
              "risk_off": RED}.get(regime, SUB)),
        _kpi(str(n), "Active picks", regime_desc[:28], INK),
        _kpi(str(breached), "Stops breached", "current picks",
             RED if breached else GREEN),
    ]
    s = summary or {}
    if s.get("mean_pct") is not None:
        v = float(s["mean_pct"])
        kpis.append(_kpi(f"{v:+.2f}%", f"{s.get('horizon','1')}d mean",
                         f"median {_fmt(s.get('median_pct'))}%",
                         GREEN if v > 0 else RED if v < 0 else INK))
    if s.get("hit_rate_pct") is not None:
        kpis.append(_kpi(f"{float(s['hit_rate_pct']):.0f}%", "Hit rate",
                         f"n={s.get('observations','—')}"))

    df = pd.DataFrame(rows)
    table_html = _table(df, capital=capital, risk_pct=risk_pct) if not df.empty else \
        '<div class="panel" style="padding:20px;color:var(--sub)">No picks today.</div>'
    cards_html = "".join(_card(r) for r in rows)

    return _page(title_date=d, regime=regime, kpis_html="".join(kpis),
                table_html=table_html, cards_html=cards_html, equity_html="",
                notes=notes, source_stamp=f"{dt.datetime.now():%d %b %Y %H:%M}")


def save(html_text: str, *, obs_date: dt.date | None = None,
         out_dir: Path = OUT_DIR) -> Path:
    d = obs_date or dt.date.today()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"dashboard_{d:%Y%m%d}.html"
    p.write_text(html_text, encoding="utf-8")
    return p


def build_cumulative(
    *, track_days: int = TRACK_DAYS, regime: str = "unknown",
    sectors: dict | None = None, full_frames: dict | None = None,
    with_news: bool = False, capital: float = 500_000.0, risk_pct: float = 1.0,
    show_runs: bool = False,
) -> tuple[str, Path] | None:
    """Every pick from the daily tracker CSV in the last `track_days`.

    Source discipline: rows come ONLY from the tracker CSV — the picks that
    were actually made — never from a re-screened or synthetic universe.

    Bucket rollover: a day's rows leave together once that day's bucket ages
    past `track_days`, so the report is always whole buckets, never a stock
    picked one day removed while its neighbour from the same bucket remains.

    Each row is independently tracked: a stock picked on several different
    days appears once per day with the entry price and stop that applied
    then, not deduplicated into one line.
    """
    import daily_tracker as dtm

    obs, px = dtm.load()
    if obs.empty:
        return None
    obs = obs.dropna(subset=["ticker"]).copy()
    if obs.empty:
        return None
    obs["obs_date"] = pd.to_datetime(obs["obs_date"], errors="coerce")
    obs = obs.dropna(subset=["obs_date"])
    if obs.empty:
        return None

    today = obs["obs_date"].max()
    cutoff = today - pd.Timedelta(days=track_days)

    aged_out = obs[obs["obs_date"] < cutoff]
    dropped_rows = len(aged_out)
    dropped_buckets = int(aged_out["obs_date"].dt.date.nunique()) if len(aged_out) else 0

    windowed = obs[obs["obs_date"] >= cutoff].sort_values(
        ["obs_date", "rank"], na_position="last")
    if windowed.empty:
        return None

    px2 = px.copy()
    if not px2.empty:
        px2["date"] = pd.to_datetime(px2["date"], errors="coerce")
        px2 = px2.dropna(subset=["date"])

    rows, r_values = [], []
    for _, r in windowed.iterrows():
        tkr = str(r["ticker"])
        obs_d = r["obs_date"]
        entry = pd.to_numeric(r.get("ref_close"), errors="coerce")
        stop = pd.to_numeric(r.get("stop_loss"), errors="coerce")
        t1 = pd.to_numeric(r.get("target_1r"), errors="coerce")
        t2 = pd.to_numeric(r.get("target_2r"), errors="coerce")
        held = int((today - obs_d).days)

        chart_dates = chart_closes = None
        last, maxdd, maxgain, hi52, lo52 = entry, np.nan, np.nan, np.nan, np.nan
        low = high = entry

        if not px2.empty:
            tk_hist = px2[px2["ticker"] == tkr].sort_values("date")
            pre = tk_hist[tk_hist["date"] < obs_d].tail(PRE_ENTRY_BARS)
            post = tk_hist[tk_hist["date"] >= obs_d]
            window = pd.concat([pre, post]) if len(pre) else post
            if len(window) >= 2:
                chart_dates = pd.DatetimeIndex(window["date"])
                chart_closes = window["close"].to_numpy()
                last = float(post["close"].iloc[-1]) if len(post) else entry
                low = float(post["low"].min()) if "low" in post and len(post) else last
                high = float(post["high"].max()) if "high" in post and len(post) else last
                if np.isfinite(entry) and entry:
                    maxdd = (low / entry - 1) * 100
                    maxgain = (high / entry - 1) * 100

        ff = None
        if full_frames:
            ff = full_frames.get(tkr)
            if ff is None:
                ff = full_frames.get(f"{tkr}.NS")
        run_stats = None
        if ff is not None and "Close" in getattr(ff, "columns", []):
            up_to = ff[ff.index <= obs_d]
            if len(up_to) > 30:
                yr = up_to.tail(252)
                hi52 = float(yr["High"].max() if "High" in yr else yr["Close"].max())
                lo52 = float(yr["Low"].min() if "Low" in yr else yr["Close"].min())
            if show_runs and len(up_to) > CONTINUATION_WINDOW * 4:
                run_stats = momentum_runs(up_to["Close"])

        status = _status_of(stop=stop, low=low, high=high, target_1r=t1,
                            target_2r=t2, held=held, track_days=track_days,
                            latest_close=last)

        if status in ("Stopped", "Target hit", "Expired") and np.isfinite(entry) \
                and np.isfinite(stop) and entry > stop:
            r_values.append(float((last - entry) / (entry - stop)))

        risk_ps = (entry - stop) if (np.isfinite(entry) and np.isfinite(stop)) else np.nan
        qty = (np.floor(capital * risk_pct / 100.0 / risk_ps)
              if np.isfinite(risk_ps) and risk_ps > 0 else np.nan)

        rows.append({
            "Ticker": tkr,
            "Sector": _first_present((sectors or {}).get(tkr), r.get("sector"), default="—"),
            "Entry": entry, "Last": last, "Stop": stop,
            "Target_1R": t1, "Target_2R": t2, "Max_DD": maxdd, "Max_Gain": maxgain,
            "Days_held": held, "Status": status, "ChartDates": chart_dates,
            "ChartCloses": chart_closes, "Score": r.get("score"),
            "Momentum": r.get("momentum_pct"),
            "Earnings_flag": r.get("earnings_verdict"),
            "High_52w": hi52, "Low_52w": lo52, "Qty": qty,
            "RunStats": run_stats, "Entry_date": obs_d.date().isoformat(),
        })

    df_rows = sorted(rows, key=lambda z: (z["Entry_date"], z.get("Ticker")), reverse=True)
    df = pd.DataFrame(df_rows)

    breached = int((df["Status"] == "Stopped").sum())
    n = len(df)
    live = df[df["Status"] == "Holding"]
    summary = {}
    if not live.empty:
        ret = (pd.to_numeric(live["Last"], errors="coerce")
              / pd.to_numeric(live["Entry"], errors="coerce") - 1) * 100
        ret = ret.dropna()
        if len(ret):
            summary = {"horizon": str(track_days), "observations": len(ret),
                      "mean_pct": round(float(ret.mean()), 2),
                      "median_pct": round(float(ret.median()), 2),
                      "hit_rate_pct": round(float((ret > 0).mean()) * 100, 1)}

    eff_bets = None
    try:
        import costs as _costs
        live_t = set(live["Ticker"].astype(str))
        mats = {}
        for t in live_t:
            fr = None
            if full_frames:
                fr = full_frames.get(t)
                if fr is None:
                    fr = full_frames.get(f"{t}.NS")
            if fr is not None and len(fr) > 70:
                mats[t] = fr
        if len(mats) >= 3:
            rmat = _costs.build_returns_matrix(mats, lookback=60)
            if not rmat.empty and rmat.shape[1] >= 3:
                eff_bets = _costs.effective_positions(rmat).effective_n
    except Exception:                                          # noqa: BLE001
        pass

    kpis = [
        _kpi(regime.replace("_", " ").upper(), "Regime gate", "momentum model",
             {"risk_on": GREEN, "neutral": AMBER,
              "risk_off": RED}.get(regime, SUB)),
        _kpi(str(n), "Active picks", f"{track_days}d window", INK),
        _kpi(str(breached), "Stops breached", "current window",
             RED if breached else GREEN),
    ]
    if summary.get("mean_pct") is not None:
        v = summary["mean_pct"]
        kpis.append(_kpi(f"{v:+.2f}%", "Holding mean",
                         f"median {summary.get('median_pct')}%",
                         GREEN if v > 0 else RED if v < 0 else INK))
    if summary.get("hit_rate_pct") is not None:
        kpis.append(_kpi(f"{summary['hit_rate_pct']:.0f}%", "Hit rate",
                         f"n={summary.get('observations')}"))
    if eff_bets is not None:
        n_distinct = int(df["Ticker"].nunique())
        kpis.append(_kpi(f"{eff_bets:.1f}", "Effective bets",
                         f"of {n_distinct} name(s)",
                         RED if eff_bets < n_distinct * 0.35 else
                         AMBER if eff_bets < n_distinct * 0.6 else GREEN))

    news = fetch_news(df["Ticker"].unique()) if with_news else None

    table_html = _table(df, capital=capital, risk_pct=risk_pct)
    cards_html = "".join(_card(r, show_runs=show_runs, news=news) for r in df_rows)

    eq_svg = _equity_curve(r_values)
    equity_html = ""
    if eq_svg:
        equity_html = (
            f'<div class="sh"><h2>Cumulative R — closed picks</h2>'
            f'<div class="rule"></div><span class="n">{len(r_values)} closed</span></div>'
            f'<div class="eqwrap">{eq_svg}</div>')

    notes = [
        f"Rolling {track_days}-day window: {n} pick(s) across "
        f"{df['Entry_date'].nunique()} bucket(s) from the daily tracker. "
        f"{dropped_buckets} bucket(s) / {dropped_rows} row(s) aged out and were "
        "removed as whole buckets. Each row is independently tracked from its "
        "own observation date — a stock chosen on several days appears once "
        "per day with the entry price and stop that applied then.",
    ]

    html_text = _page(
        title_date=today.date(), regime=regime, kpis_html="".join(kpis),
        table_html=table_html, cards_html=cards_html, equity_html=equity_html,
        notes=notes, source_stamp=f"{today:%d %b %Y}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"dashboard_cumulative_{today.date():%Y%m%d}.html"
    p.write_text(html_text, encoding="utf-8")
    return html_text, p


def build_from_tracker(*, regime: str = "unknown", prices: dict | None = None,
                       capital: float = 500_000.0, risk_pct: float = 1.0
                       ) -> tuple[str, Path] | None:
    """Today's bucket only, built straight from the tracker CSV."""
    import daily_tracker as dtm

    obs, px = dtm.load()
    if obs.empty:
        return None
    latest = obs["obs_date"].astype(str).max()
    today_rows = obs[obs["obs_date"].astype(str) == latest].dropna(subset=["ticker"])
    if today_rows.empty:
        return None

    picks = pd.DataFrame({
        "Ticker": today_rows["ticker"], "Sector": today_rows.get("sector"),
        "Score": today_rows.get("score"), "Momentum": today_rows.get("momentum_pct"),
        "Close": today_rows.get("ref_close"), "Stop": today_rows.get("stop_loss"),
        "Target_1R": today_rows.get("target_1r"),
        "Target_2R": today_rows.get("target_2r"),
        "Earnings_flag": today_rows.get("earnings_verdict"),
    })

    if prices is None and not px.empty:
        prices = {}
        px2 = px.copy()
        px2["date"] = pd.to_datetime(px2["date"], errors="coerce")
        for t, g in px2.dropna(subset=["date"]).groupby("ticker"):
            g = g.sort_values("date").set_index("date")
            prices[str(t)] = g.rename(columns={"open": "Open", "high": "High",
                                               "low": "Low", "close": "Close",
                                               "volume": "Volume"})

    perf = dtm.performance(obs, px)
    summ = dtm.summarise(perf)
    summary = summ.iloc[0].to_dict() if not summ.empty else {}

    d = pd.Timestamp(latest).date()
    html_text = build(picks, prices, regime=regime, obs_date=d, summary=summary,
                      capital=capital, risk_pct=risk_pct)
    return html_text, save(html_text, obs_date=d)
