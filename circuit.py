"""Circuit breakers and exit liquidity — the stop that isn't there.

The failure this prevents
-------------------------
Position sizing throughout this system assumes a stop will fill near its price.
For a large, F&O-eligible stock that is broadly true. For a non-F&O smallcap in
a 5% price band it can be simply false.

When a stock locks limit-down there is no bid. The stop does not execute at
-6%; it does not execute at all. Price freezes, the exchange halts further
downward movement, and the position cannot be exited until buyers reappear —
often two or three sessions later, twenty-five percent lower.

This is not a gradual failure. It is the difference between a bad trade and a
position that eats a quarter of the account, and no amount of ATR calibration
helps because the arithmetic assumes an exit that does not exist.

Why F&O eligibility is the key filter
-------------------------------------
Securities in the F&O segment have **no individual price band** — the exchange
applies a dynamic 10% operating range that can be flexed intraday, so trading
continues. Non-F&O securities carry hard daily bands of 2%, 5%, 10% or 20%
depending on surveillance status, and once hit, that is the end of trading in
that direction for the session.

F&O eligibility also implies a liquidity floor: the exchange requires median
quarter-sigma order size and market-wide position limit thresholds before
admitting a stock. It is therefore a reasonable single proxy for both
exit-ability and liquidity.

Surveillance frameworks
-----------------------
ASM (Additional Surveillance Measure) and GSM (Graded Surveillance Measure)
impose 100% margin, trade-to-trade settlement, and progressively narrower
bands. A GSM Stage 4 security may trade once a week with a 5% band. These
should never appear in a swing bucket regardless of how they rank.

The BE series flag in bhavcopy already captures much of this, which is why the
point-in-time universe filters on series == EQ.

Data honesty
------------
The F&O list changes with each review cycle and NSE blocks aggressive scrapers,
so this module ships a bundled list with an explicit staleness warning and
attempts a live refresh where possible. A stale list that says so is far safer
than a scraper that fails silently and lets a 5%-band microcap into the bucket.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

FNO_CACHE = Path("fno_list.csv")

# NSE price bands for non-F&O securities. Once hit, trading in that direction
# stops for the session.
PRICE_BANDS = {
    "no_band": None,      # F&O eligible — dynamic operating range only
    "20pct": 0.20,
    "10pct": 0.10,
    "5pct": 0.05,
    "2pct": 0.02,         # typically GSM/ASM stage securities
}

# Bundled snapshot of F&O-eligible symbols. VERIFY against NSE before relying
# on it — the list is revised periodically and additions/exclusions matter.
# Source: nseindia.com > Products > Equity Derivatives > Underlying Info
FNO_SNAPSHOT_DATE = "2026-07-28"
FNO_SYMBOLS = {
    "AARTIIND", "ABB", "ABBOTINDIA", "ABCAPITAL", "ABFRL", "ACC", "ADANIENSOL",
    "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ALKEM", "AMBUJACEM", "ANGELONE",
    "APLAPOLLO", "APOLLOHOSP", "APOLLOTYRE", "ASHOKLEY", "ASIANPAINT",
    "ASTRAL", "ATGL", "AUBANK", "AUROPHARMA", "AXISBANK", "BAJAJ-AUTO",
    "BAJAJFINSV", "BAJFINANCE", "BALKRISIND", "BANDHANBNK", "BANKBARODA",
    "BANKINDIA", "BATAINDIA", "BEL", "BERGEPAINT", "BHARATFORG", "BHARTIARTL",
    "BHEL", "BIOCON", "BOSCHLTD", "BPCL", "BRITANNIA", "BSE", "BSOFT",
    "CAMS", "CANBK", "CANFINHOME", "CHAMBLFERT", "CHOLAFIN", "CIPLA",
    "COALINDIA", "COFORGE", "COLPAL", "CONCOR", "COROMANDEL", "CROMPTON",
    "CUB", "CUMMINSIND", "CYIENT", "DABUR", "DALBHARAT", "DEEPAKNTR",
    "DELHIVERY", "DIVISLAB", "DIXON", "DLF", "DMART", "DRREDDY", "EICHERMOT",
    "ESCORTS", "EXIDEIND", "FEDERALBNK", "GAIL", "GLENMARK", "GMRAIRPORT",
    "GNFC", "GODREJCP", "GODREJPROP", "GRANULES", "GRASIM", "GUJGASLTD",
    "HAL", "HAVELLS", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDCOPPER", "HINDPETRO", "HINDUNILVR",
    "ICICIBANK", "ICICIGI", "ICICIPRULI", "IDEA", "IDFCFIRSTB", "IEX",
    "IGL", "INDHOTEL", "INDIAMART", "INDIANB", "INDIGO", "INDUSINDBK",
    "INDUSTOWER", "INFY", "IOC", "IPCALAB", "IRB", "IRCTC", "IRFC",
    "ITC", "JINDALSTEL", "JIOFIN", "JKCEMENT", "JSL", "JSWENERGY",
    "JSWSTEEL", "JUBLFOOD", "KALYANKJIL", "KEI", "KOTAKBANK", "KPITTECH",
    "LALPATHLAB", "LAURUSLABS", "LICHSGFIN", "LICI", "LODHA", "LT",
    "LTF", "LTIM", "LTTS", "LUPIN", "M&M", "M&MFIN", "MANAPPURAM",
    "MARICO", "MARUTI", "MAXHEALTH", "MCX", "METROPOLIS", "MFSL",
    "MGL", "MOTHERSON", "MPHASIS", "MRF", "MUTHOOTFIN", "NATIONALUM",
    "NAUKRI", "NAVINFLUOR", "NCC", "NESTLEIND", "NHPC", "NMDC", "NTPC",
    "OBEROIRLTY", "OFSS", "OIL", "ONGC", "PAGEIND", "PAYTM", "PEL",
    "PERSISTENT", "PETRONET", "PFC", "PHOENIXLTD", "PIDILITIND", "PIIND",
    "PNB", "POLICYBZR", "POLYCAB", "POONAWALLA", "POWERGRID", "PRESTIGE",
    "RAMCOCEM", "RBLBANK", "RECLTD", "RELIANCE", "SAIL", "SBICARD",
    "SBILIFE", "SBIN", "SHREECEM", "SHRIRAMFIN", "SIEMENS", "SJVN",
    "SONACOMS", "SRF", "SUNPHARMA", "SUNTV", "SUPREMEIND", "SYNGENE",
    "TATACHEM", "TATACOMM", "TATACONSUM", "TATAELXSI", "TATAMOTORS",
    "TATAPOWER", "TATASTEEL", "TCS", "TECHM", "TIINDIA", "TITAN",
    "TORNTPHARM", "TRENT", "TVSMOTOR", "UBL", "ULTRACEMCO", "UNIONBANK",
    "UNITDSPR", "UPL", "VBL", "VEDL", "VOLTAS", "WIPRO", "YESBANK",
    "ZOMATO", "ZYDUSLIFE",
}


def _clean_series(value) -> str | None:
    """Coerce a series code safely.

    pandas converts None to NaN (a float) when a column is built from a list
    containing None, so a bare .upper() raises. Anything not a usable string
    becomes None.
    """
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    s = str(value).strip()
    return s.upper() if s and s.lower() not in ("nan", "none", "") else None


@dataclass
class ExitRisk:
    ticker: str
    fno_eligible: bool
    estimated_band: float | None      # None means no individual band
    series: str | None
    surveillance_flag: bool
    can_exit: bool
    risk_level: str                   # low | medium | high | severe
    reasons: list[str] = field(default_factory=list)


def load_fno_list(*, allow_refresh: bool = True) -> tuple[set[str], str, bool]:
    """F&O-eligible symbols. Returns (symbols, source, is_live)."""
    if FNO_CACHE.exists():
        try:
            df = pd.read_csv(FNO_CACHE)
            if "symbol" in df.columns and len(df) > 50:
                age_note = f"cache ({len(df)} symbols)"
                return set(df["symbol"].astype(str).str.upper()), age_note, False
        except Exception:                                      # noqa: BLE001
            pass

    if allow_refresh:
        try:
            s = requests.Session()
            s.headers.update({
                "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/122.0 Safari/537.36"),
                "Accept": "text/csv,*/*",
                "Referer": "https://www.nseindia.com/",
            })
            s.get("https://www.nseindia.com", timeout=8)
            r = s.get("https://nsearchives.nseindia.com/content/fo/"
                      "fo_mktlots.csv", timeout=12)
            if r.status_code == 200 and len(r.content) > 500:
                df = pd.read_csv(pd.io.common.StringIO(r.text))
                col = next((c for c in df.columns
                            if "symbol" in str(c).strip().lower()), None)
                if col:
                    syms = set(df[col].astype(str).str.strip().str.upper())
                    syms.discard("SYMBOL")
                    if len(syms) > 50:
                        pd.DataFrame({"symbol": sorted(syms)}).to_csv(
                            FNO_CACHE, index=False)
                        return syms, "NSE live", True
        except Exception:                                      # noqa: BLE001
            pass

    return set(FNO_SYMBOLS), f"bundled snapshot ({FNO_SNAPSHOT_DATE})", False


def estimate_band(ticker: str, fno: set[str], *, series: str | None = None,
                  price: float | None = None,
                  turnover_cr: float | None = None) -> float | None:
    """Estimate the applicable daily price band.

    Exact bands are set per-security by the exchange and revised regularly.
    This estimates conservatively from the signals available: F&O eligibility,
    series code, price level and liquidity. When uncertain it assumes the
    *tighter* band, because the cost of underestimating exit risk is far higher
    than the cost of skipping a tradeable stock.
    """
    sym = ticker.replace(".NS", "").upper()

    if sym in fno:
        return None                       # dynamic operating range, no hard band

    ser = _clean_series(series)
    if ser in ("BE", "BZ", "SM", "ST"):
        return 0.05                       # surveillance / SME segment

    if price is not None and price < 20:
        return 0.05                       # penny stocks typically tightly banded

    if turnover_cr is not None:
        if turnover_cr >= 100:
            return 0.20
        if turnover_cr >= 25:
            return 0.10
        return 0.05

    return 0.10                           # conservative default


def assess(ticker: str, *, fno: set[str] | None = None,
           series: str | None = None, price: float | None = None,
           turnover_cr: float | None = None,
           atr_pct: float | None = None,
           stop_distance_pct: float | None = None) -> ExitRisk:
    """Can this position actually be exited at its stop?"""
    if fno is None:
        fno, _, _ = load_fno_list(allow_refresh=False)

    sym = ticker.replace(".NS", "").upper()
    is_fno = sym in fno
    series = _clean_series(series)
    band = estimate_band(ticker, fno, series=series, price=price,
                         turnover_cr=turnover_cr)
    surveillance = series in ("BE", "BZ", "SM", "ST")

    reasons: list[str] = []
    can_exit = True
    level = "low"

    if is_fno:
        reasons.append("F&O eligible — no individual price band, so trading "
                       "continues even on a large move.")
    else:
        reasons.append(f"Not F&O eligible — subject to a hard daily band "
                       f"(estimated {band:.0%}).")
        level = "medium"

        # The critical comparison: can the stop even be reached before the
        # circuit halts trading?
        if stop_distance_pct is not None and band is not None:
            if stop_distance_pct / 100 >= band:
                can_exit = False
                level = "severe"
                reasons.append(
                    f"STOP UNREACHABLE: the stop sits {stop_distance_pct:.1f}% "
                    f"away but the band is {band:.0%}. Price freezes before the "
                    "stop is touched, and the position cannot be exited that "
                    "session."
                )
            elif stop_distance_pct / 100 >= band * 0.7:
                level = "high"
                reasons.append(
                    f"Stop at {stop_distance_pct:.1f}% is close to the "
                    f"{band:.0%} band. A gap could trigger the circuit before "
                    "the stop fills."
                )

        # A stock whose normal daily range approaches its band hits circuits often
        if atr_pct is not None and band is not None and atr_pct / 100 > band * 0.5:
            level = "high" if level != "severe" else level
            reasons.append(
                f"Daily ATR is {atr_pct:.1f}% against a {band:.0%} band — this "
                "security likely hits circuits regularly."
            )

    if surveillance:
        can_exit = False
        level = "severe"
        reasons.append(
            f"Series '{series}' indicates surveillance (ASM/GSM) or SME segment: "
            "100% margin, trade-to-trade settlement, narrow bands. Should not "
            "appear in a swing bucket."
        )

    return ExitRisk(sym, is_fno, band, series, surveillance,
                    can_exit, level, reasons)


def filter_bucket(picks: pd.DataFrame, *, fno: set[str] | None = None,
                  require_fno: bool = False,
                  atr_mult: float = 2.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a bucket into tradeable and excluded, on exit-risk grounds.

    Returns (kept, excluded). The excluded frame carries a reason column, so a
    rejection is never silent.
    """
    if picks is None or picks.empty:
        return picks, pd.DataFrame()

    if fno is None:
        fno, _, _ = load_fno_list(allow_refresh=False)

    kept, dropped = [], []
    for _, r in picks.iterrows():
        tkr = str(r.get("Ticker", ""))
        atr_pct = float(r["ATR_pct"]) if "ATR_pct" in r and pd.notna(r["ATR_pct"]) else None
        stop_pct = atr_pct * atr_mult if atr_pct is not None else None

        risk = assess(
            tkr, fno=fno,
            series=r.get("series"),
            price=float(r["Close"]) if "Close" in r and pd.notna(r["Close"]) else None,
            turnover_cr=(float(r["Turnover_Cr"])
                         if "Turnover_Cr" in r and pd.notna(r["Turnover_Cr"]) else None),
            atr_pct=atr_pct,
            stop_distance_pct=stop_pct,
        )

        row = r.to_dict()
        row.update({
            "fno_eligible": risk.fno_eligible,
            "price_band": risk.estimated_band,
            "exit_risk": risk.risk_level,
        })

        exclude = (not risk.can_exit) or (require_fno and not risk.fno_eligible)
        if exclude:
            row["exclusion_reason"] = risk.reasons[-1] if risk.reasons else "exit risk"
            dropped.append(row)
        else:
            kept.append(row)

    return (pd.DataFrame(kept).reset_index(drop=True),
            pd.DataFrame(dropped).reset_index(drop=True))


def max_safe_stop(ticker: str, *, fno: set[str] | None = None,
                  turnover_cr: float | None = None,
                  price: float | None = None) -> float | None:
    """Widest stop that can realistically fill, as a percentage.

    A stop must sit comfortably inside the band or it will never be reached.
    70% of the band leaves room for the circuit to be approached without being
    hit. Returns None for F&O stocks, where no hard band applies.
    """
    if fno is None:
        fno, _, _ = load_fno_list(allow_refresh=False)
    band = estimate_band(ticker, fno, price=price, turnover_cr=turnover_cr)
    return None if band is None else round(band * 0.70 * 100, 2)
