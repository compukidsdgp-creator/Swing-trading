"""Technical indicators, implemented in pandas so there's no TA-Lib dependency.

TA-Lib needs a C build step that Streamlit Cloud doesn't handle cleanly, so
everything here is plain pandas/numpy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — trend strength, direction-agnostic."""
    high, low = df["High"], df["Low"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = atr(df, period)
    tr_safe = tr.replace(0, np.nan)

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / tr_safe
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / tr_safe

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0)


def bollinger_width(close: pd.Series, period: int = 20, mult: float = 2.0) -> pd.Series:
    """Band width as % of price — low values mean consolidation (squeeze)."""
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return ((ma + mult * sd) - (ma - mult * sd)) / ma * 100


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the full indicator set to an OHLCV frame."""
    out = df.copy()
    close = out["Close"]

    out["EMA20"] = ema(close, 20)
    out["EMA50"] = ema(close, 50)
    out["EMA200"] = ema(close, 200)
    out["RSI14"] = rsi(close, 14)

    line, sig, hist = macd(close)
    out["MACD"], out["MACD_Signal"], out["MACD_Hist"] = line, sig, hist

    out["ATR14"] = atr(out, 14)
    out["ADX14"] = adx(out, 14)
    out["BB_Width"] = bollinger_width(close, 20)

    out["Vol20"] = out["Volume"].rolling(20).mean()
    out["Vol_Ratio"] = out["Volume"] / out["Vol20"].replace(0, np.nan)
    out["Turnover_Cr"] = (close * out["Volume"]).rolling(20).mean() / 1e7

    for n in (5, 10, 20, 60):
        out[f"Ret_{n}d"] = close.pct_change(n) * 100

    return out
