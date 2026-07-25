"""Bundled fallback universes.

Used only when NSE is unreachable. These are a point-in-time snapshot and WILL
go stale — the app labels any universe served from here as non-live so you know
you're looking at cached data.
"""

from __future__ import annotations

SNAPSHOT_DATE = "2026-07-25"

_NIFTY50 = ['RELIANCE', 'HDFCBANK', 'ICICIBANK', 'INFY', 'TCS', 'BHARTIARTL', 'SBIN', 'LT', 'ITC', 'HINDUNILVR', 'AXISBANK', 'KOTAKBANK', 'BAJFINANCE', 'MARUTI', 'M&M', 'SUNPHARMA', 'NTPC', 'TITAN', 'ULTRACEMCO', 'ASIANPAINT', 'TATAMOTORS', 'POWERGRID', 'ONGC', 'COALINDIA', 'TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'ADANIENT', 'ADANIPORTS', 'GRASIM', 'CIPLA', 'DRREDDY', 'APOLLOHOSP', 'BAJAJFINSV', 'BAJAJ-AUTO', 'HEROMOTOCO', 'EICHERMOT', 'NESTLEIND', 'BRITANNIA', 'TATACONSUM', 'HCLTECH', 'WIPRO', 'TECHM', 'INDUSINDBK', 'SHRIRAMFIN', 'SBILIFE', 'HDFCLIFE', 'TRENT', 'JIOFIN', 'ETERNAL']

_MIDCAP = ['PERSISTENT', 'COFORGE', 'LTIM', 'MPHASIS', 'OFSS', 'KPITTECH', 'TATAELXSI', 'POLYCAB', 'KEI', 'HAVELLS', 'CUMMINSIND', 'ABB', 'SIEMENS', 'THERMAX', 'BEL', 'HAL', 'BDL', 'MAZDOCK', 'SOLARINDS', 'ASTRAL', 'LUPIN', 'AUROPHARMA', 'TORNTPHARM', 'ZYDUSLIFE', 'ALKEM', 'LAURUSLABS', 'MAXHEALTH', 'FORTIS', 'METROPOLIS', 'LALPATHLAB', 'FEDERALBNK', 'IDFCFIRSTB', 'BANDHANBNK', 'AUBANK', 'CHOLAFIN', 'MUTHOOTFIN', 'LICHSGFIN', 'PFC', 'RECLTD', 'CANBK', 'DIXON', 'AMBER', 'VOLTAS', 'BLUESTARCO', 'CROMPTON', 'PIIND', 'SRF', 'DEEPAKNTR', 'AARTIIND', 'NAVINFLUOR']

_SMALLCAP = ['CAPLIPOINT', 'NATCOPHARM', 'JBCHEPHARM', 'GRANULES', 'AJANTPHARM', 'TANLA', 'ROUTE', 'HAPPSTMNDS', 'NEWGEN', 'INTELLECT', 'SONATSOFT', 'ZENSARTECH', 'BIRLASOFT', 'CYIENT', 'MASTEK', 'NUCLEUS', 'CDSL', 'MCX', 'BSE', 'ANGELONE', 'IIFL', 'MOTILALOFS', 'TRIVENI', 'KIRLOSENG', 'TDPOWERSYS', 'SHILCHAR', 'VOLTAMP', 'ELECON', 'GRINDWELL', 'CARBORUNIV', 'TIMKEN', 'SCHAEFFLER']

_MAP = {
    "Nifty 50": _NIFTY50,
    "Nifty Next 50": _MIDCAP[:50],
    "Nifty 100": _NIFTY50 + _MIDCAP[:50],
    "Nifty 200": _NIFTY50 + _MIDCAP + _SMALLCAP,
    "Nifty 500": _NIFTY50 + _MIDCAP + _SMALLCAP,
    "Nifty Midcap 150": _MIDCAP,
    "Nifty Midcap 100": _MIDCAP,
    "Nifty Smallcap 250": _SMALLCAP,
    "Nifty Smallcap 100": _SMALLCAP,
}

_DEFAULT = _NIFTY50 + _MIDCAP


def get(label: str) -> tuple[str, ...]:
    """Best-effort fallback list for a given universe label."""
    syms = _MAP.get(label)
    if syms is None:
        for key, val in _MAP.items():
            if key.lower() in label.lower():
                syms = val
                break
    if syms is None:
        syms = _DEFAULT
    seen, out = set(), []
    for s in syms:
        if s not in seen:
            seen.add(s)
            out.append(f"{s}.NS")
    return tuple(out)
