"""Headline retrieval via Google News RSS.

No API key needed. If you want something better, swap this module for
NewsAPI, Finnhub, or a paid Indian markets feed — the interface is just
fetch(name) -> list[dict].
"""

from __future__ import annotations

import urllib.parse
import xml.etree.ElementTree as ET

import requests
import streamlit as st

import sentiment

RSS = "https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"

POSITIVE = {
    "surge", "jump", "rally", "gain", "profit", "beat", "upgrade", "record",
    "high", "growth", "expansion", "order", "win", "wins", "bags", "acquires",
    "acquisition", "buyback", "dividend", "bonus", "outperform", "bullish",
    "raises", "hikes", "strong", "robust", "approval", "approved", "launch",
}
NEGATIVE = {
    "fall", "drop", "plunge", "slump", "loss", "miss", "downgrade", "low",
    "decline", "weak", "cut", "cuts", "probe", "fraud", "penalty", "fine",
    "resign", "resigns", "lawsuit", "raid", "default", "bearish", "warns",
    "warning", "delay", "delays", "recall", "slowdown", "concern",
}


@st.cache_data(ttl=60 * 20, show_spinner=False)
def fetch(company: str, limit: int = 10) -> list[dict]:
    """Pull recent headlines for a company. Cached 20 minutes."""
    query = urllib.parse.quote_plus(f"{company} stock NSE India")
    try:
        resp = requests.get(
            RSS.format(q=query),
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (SwingScope research tool)"},
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except (requests.RequestException, ET.ParseError):
        return []

    items = []
    for node in root.findall(".//item")[:limit]:
        title = node.findtext("title", default="").strip()
        if not title:
            continue
        items.append({
            "title": title,
            "link": node.findtext("link", default=""),
            "published": node.findtext("pubDate", default="")[:22],
            "source": (node.findtext("source") or "Google News").strip(),
        })
    return items


def score_headline(title: str) -> str:
    """Delegates to the sentiment module. Kept for backwards compatibility."""
    return sentiment.score_headline(title)
