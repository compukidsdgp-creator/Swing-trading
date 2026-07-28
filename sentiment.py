"""Financial-headline sentiment analysis.

The previous implementation was a flat keyword count, which fails on the two
things financial headlines do constantly:

  "Shares fall despite strong profit"      -> counts 1 pos, 1 neg, returns neutral
  "Company fails to beat estimates"        -> counts "beat" as positive

This module handles both, using a weighted finance lexicon (Loughran-McDonald
inspired — general-purpose sentiment tools mis-read financial language badly),
negation detection, and intensity modifiers.

It is still a lexicon model, not a language model. It will misread sarcasm,
complex clauses, and anything requiring real comprehension. Treat the output as
triage — what to read first — never as a trading input.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Weighted lexicon. Magnitude reflects how strongly the term moves a stock,
# not how emotionally charged the word is.
# --------------------------------------------------------------------------
POSITIVE = {
    # Strong
    "surge": 3.0, "soar": 3.0, "jump": 2.5, "rally": 2.5, "skyrocket": 3.0,
    "record": 2.5, "beat": 2.5, "beats": 2.5, "upgrade": 3.0, "upgraded": 3.0,
    "outperform": 2.5, "breakout": 2.5, "multibagger": 3.0,
    # Corporate action / fundamentals
    "bags": 2.5, "wins": 2.5, "win": 2.0, "secures": 2.0, "awarded": 2.0,
    "acquires": 1.5, "acquisition": 1.5, "merger": 1.2, "buyback": 2.5,
    "dividend": 1.5, "bonus": 2.0, "expansion": 1.5, "expands": 1.5,
    "approval": 2.0, "approved": 2.0, "clearance": 2.0, "launch": 1.5,
    "partnership": 1.5, "contract": 1.5, "order": 1.5, "deal": 1.2,
    # Performance
    "profit": 1.5, "growth": 1.5, "gain": 1.5, "gains": 1.5, "rise": 1.2,
    "rises": 1.2, "climb": 1.2, "advance": 1.2, "strong": 1.8, "robust": 1.8,
    "healthy": 1.2, "improve": 1.5, "improved": 1.5, "expand": 1.5,
    "raises": 1.5, "hikes": 1.5, "boost": 1.8, "boosts": 1.8,
    "bullish": 2.5, "optimistic": 1.8, "positive": 1.2, "high": 1.0,
    "rallies": 2.5, "rallied": 2.5, "jumps": 2.5, "soars": 3.0, "climbs": 1.2,
    "recovers": 1.5, "rebound": 1.8, "rebounds": 1.8, "outperforms": 2.5,
    "top": 1.2, "best": 1.5, "leads": 1.2, "surges": 3.0,
}

NEGATIVE = {
    # Strong
    "plunge": -3.0, "crash": -3.0, "slump": -2.5, "tumble": -2.5, "plummet": -3.0,
    "downgrade": -3.0, "downgraded": -3.0, "underperform": -2.5,
    "fraud": -3.5, "scam": -3.5, "probe": -2.5, "raid": -3.0, "default": -3.5,
    "insolvency": -3.5, "bankruptcy": -3.5, "delisting": -3.0,
    # Regulatory / governance
    "penalty": -2.5, "fine": -2.0, "fined": -2.0, "lawsuit": -2.0, "sued": -2.0,
    "notice": -1.5, "violation": -2.5, "resign": -2.0, "resigns": -2.0,
    "quits": -2.0, "exit": -1.2, "recall": -2.5, "ban": -2.5, "banned": -2.5,
    "suspend": -2.5, "suspended": -2.5, "sebi": -1.0, "investigation": -2.5,
    # Performance
    "loss": -2.0, "losses": -2.0, "miss": -2.0, "misses": -2.0, "decline": -1.5,
    "declines": -1.5, "fall": -1.5, "falls": -1.5, "drop": -1.5, "drops": -1.5,
    "weak": -1.8, "weakness": -1.8, "slowdown": -2.0, "sluggish": -1.8,
    "cut": -1.5, "cuts": -1.5, "reduce": -1.2, "lower": -1.2, "lowers": -1.2,
    "concern": -1.5, "concerns": -1.5, "warns": -2.0, "warning": -2.0,
    "risk": -1.2, "pressure": -1.5, "bearish": -2.5, "pessimistic": -1.8,
    "delay": -1.5, "delays": -1.5, "halt": -2.0, "shut": -2.0, "closure": -2.0,
    "negative": -1.2, "low": -1.0, "worst": -2.0, "drag": -1.5, "hit": -1.2,
    "plunges": -3.0, "tumbles": -2.5, "slumps": -2.5, "sinks": -2.5,
    "widens": -1.5, "erodes": -1.8, "downgrades": -3.0, "slips": -1.5,
}

# Words that flip the polarity of what follows
NEGATORS = {
    "no", "not", "never", "none", "fails", "fail", "failed", "failing",
    "without", "lacks", "lacking", "unable", "cannot", "denies", "denied",
    "rejects", "rejected", "misses", "miss", "unlikely", "doubt", "doubts",
}

# Contrast markers split a headline into clauses. English puts the operative
# point on different sides depending on the marker:
#   "Profit rises BUT margins under pressure"      -> second clause dominates
#   "Shares fall DESPITE strong profit growth"     -> first clause dominates
CONTRAST_SECOND_WINS = {"but", "however", "yet", "though", "still"}
CONTRAST_FIRST_WINS = {"despite", "although", "even as", "in spite of"}

# Words that scale the next sentiment term
INTENSIFIERS = {
    "very": 1.5, "highly": 1.5, "sharply": 1.6, "significantly": 1.5,
    "massive": 1.8, "huge": 1.7, "major": 1.4, "strongly": 1.5,
    "substantially": 1.5, "record": 1.6, "steep": 1.5, "dramatic": 1.6,
}
DIMINISHERS = {
    "slightly": 0.5, "marginally": 0.5, "modest": 0.6, "small": 0.6,
    "somewhat": 0.6, "partly": 0.6, "mildly": 0.5, "slight": 0.5,
}

# Event categories worth surfacing regardless of sentiment
CATALYSTS = {
    "earnings": ["result", "results", "q1", "q2", "q3", "q4", "quarter",
                 "earnings", "profit", "revenue", "ebitda", "margin"],
    "order_win": ["order", "orders", "contract", "bags", "wins", "awarded",
                  "tender", "lo", "letter of intent"],
    "regulatory": ["sebi", "rbi", "cci", "probe", "investigation", "notice",
                   "penalty", "compliance", "audit", "usfda", "fda"],
    "corporate_action": ["dividend", "bonus", "split", "buyback", "rights",
                         "demerger", "merger", "acquisition", "stake"],
    "management": ["ceo", "cfo", "md", "chairman", "resign", "appoint",
                   "board", "director"],
    "rating": ["upgrade", "downgrade", "target price", "rating", "initiate",
               "coverage", "brokerage"],
    "block_deal": ["block deal", "bulk deal", "stake sale", "promoter",
                   "pledge", "offloads"],
}

TOKEN_RE = re.compile(r"[a-z][a-z'&-]*")


@dataclass
class HeadlineScore:
    text: str
    score: float                       # roughly -5 .. +5
    label: str                         # pos / neg / neu
    matched: list[tuple[str, float]] = field(default_factory=list)
    catalysts: list[str] = field(default_factory=list)
    negated: bool = False


def _score_tokens(tokens: list[str]) -> tuple[float, list[tuple[str, float]], bool]:
    """Score one clause. Returns (raw_total, matched_terms, saw_negation)."""
    total = 0.0
    matched: list[tuple[str, float]] = []
    any_negation = False

    for idx, tok in enumerate(tokens):
        # Explicit membership check: `or` would treat a weight of 0.0 as absent
        # and fall through to the other lexicon.
        if tok in POSITIVE:
            base = POSITIVE[tok]
        elif tok in NEGATIVE:
            base = NEGATIVE[tok]
        else:
            continue

        weight = 1.0
        negate = False
        # Look back up to 5 tokens — "not able to secure the order" needs reach
        for back in range(1, 6):
            j = idx - back
            if j < 0:
                break
            prev = tokens[j]
            if prev in NEGATORS:
                negate = True
                any_negation = True
            if prev in INTENSIFIERS:
                weight *= INTENSIFIERS[prev]
            elif prev in DIMINISHERS:
                weight *= DIMINISHERS[prev]

        val = base * weight
        if negate:
            val = -val * 0.85          # negation is real but slightly weaker
        total += val
        matched.append((tok, round(val, 2)))

    return total, matched, any_negation


def _split_clauses(lower: str) -> tuple[list[str], str | None]:
    """Split on the first contrast marker found. Returns (clauses, marker)."""
    for marker in sorted(CONTRAST_SECOND_WINS | CONTRAST_FIRST_WINS,
                         key=len, reverse=True):
        pat = rf"\b{re.escape(marker)}\b"
        if re.search(pat, lower):
            parts = re.split(pat, lower, maxsplit=1)
            if len(parts) == 2 and all(p.strip() for p in parts):
                return [parts[0], parts[1]], marker
    return [lower], None


def score_text(text: str) -> HeadlineScore:
    """Score a single headline, respecting clause structure."""
    lower = text.lower()
    clauses, marker = _split_clauses(lower)

    if marker is None:
        total, matched, any_negation = _score_tokens(TOKEN_RE.findall(lower))
    else:
        a, m_a, n_a = _score_tokens(TOKEN_RE.findall(clauses[0]))
        b, m_b, n_b = _score_tokens(TOKEN_RE.findall(clauses[1]))
        if marker in CONTRAST_SECOND_WINS:
            dom, sub = b, a
        else:
            dom, sub = a, b

        # The dominant clause sets polarity. The subordinate clause can soften
        # the magnitude but must not flip the sign — "X despite Y" is a
        # statement about X, however positive Y sounds.
        if abs(dom) >= 1.0:
            # Cap softening relative to the dominant clause, so a keyword-heavy
            # subordinate ("strong profit growth") cannot cancel a clear main
            # clause ("shares fall").
            cap = 0.40 * abs(dom)
            softening = max(-cap, min(cap, 0.30 * sub))
            total = dom + softening
            if (total > 0) != (dom > 0):
                total = dom * 0.25          # subordinate clamps, never inverts
        else:
            total = 0.55 * dom + 0.45 * sub
        matched = m_a + m_b
        any_negation = n_a or n_b

    # Compress so a headline stuffed with keywords can't dominate
    if total:
        total = max(-5.0, min(5.0, total * (1.0 / (1 + 0.12 * (len(matched) - 1)))))

    found = [name for name, kws in CATALYSTS.items()
             if any(k in lower for k in kws)]

    if total >= 0.5:
        label = "pos"
    elif total <= -0.5:
        label = "neg"
    else:
        label = "neu"

    return HeadlineScore(text, round(total, 2), label, matched, found, any_negation)


def score_headline(title: str) -> str:
    """Backwards-compatible entry point. Returns 'pos' | 'neg' | 'neu'."""
    return score_text(title).label


@dataclass
class StockSentiment:
    ticker: str
    n_headlines: int
    mean_score: float
    net_score: float                   # sum, so volume of news matters
    pos: int
    neg: int
    neu: int
    label: str                         # Bullish / Bearish / Mixed / Neutral / No data
    confidence: str                    # High / Medium / Low
    catalysts: dict[str, int] = field(default_factory=dict)
    top_positive: str | None = None
    top_negative: str | None = None
    alerts: list[str] = field(default_factory=list)


def analyse_stock(ticker: str, headlines: list[dict]) -> StockSentiment:
    """Aggregate a set of headlines into one view for a stock."""
    if not headlines:
        return StockSentiment(ticker, 0, 0.0, 0.0, 0, 0, 0, "No data", "Low")

    scored = [score_text(h.get("title", "")) for h in headlines]
    scored = [s for s in scored if s.text.strip()]
    if not scored:
        return StockSentiment(ticker, 0, 0.0, 0.0, 0, 0, 0, "No data", "Low")

    vals = [s.score for s in scored]
    pos = sum(1 for s in scored if s.label == "pos")
    neg = sum(1 for s in scored if s.label == "neg")
    neu = len(scored) - pos - neg
    mean = sum(vals) / len(vals)

    # Label from mean plus agreement between headlines
    if mean >= 0.7 and pos > neg:
        label = "Bullish"
    elif mean <= -0.7 and neg > pos:
        label = "Bearish"
    elif pos and neg and abs(pos - neg) <= 1:
        label = "Mixed"
    elif abs(mean) < 0.4:
        label = "Neutral"
    else:
        label = "Leaning bullish" if mean > 0 else "Leaning bearish"

    n = len(scored)
    signed = sum(1 for s in scored if s.label != "neu")
    if n >= 6 and signed >= 4:
        conf = "High"
    elif n >= 3 and signed >= 2:
        conf = "Medium"
    else:
        conf = "Low"

    cat = Counter()
    for s in scored:
        cat.update(s.catalysts)

    pos_sorted = sorted(scored, key=lambda s: s.score, reverse=True)
    top_pos = pos_sorted[0].text if pos_sorted and pos_sorted[0].score > 0.5 else None
    top_neg = pos_sorted[-1].text if pos_sorted and pos_sorted[-1].score < -0.5 else None

    alerts = []
    if cat.get("earnings"):
        alerts.append("📊 Earnings-related news — check the reporting date before entering")
    if cat.get("regulatory"):
        alerts.append("⚖️ Regulatory news — read before taking a position")
    if cat.get("block_deal"):
        alerts.append("🔀 Block/bulk deal or pledge activity detected")
    if cat.get("rating"):
        alerts.append("📝 Analyst rating change in the feed")
    if neg >= 3 and neg > pos:
        alerts.append("🔴 Multiple negative headlines — treat any long setup sceptically")

    return StockSentiment(
        ticker=ticker, n_headlines=n, mean_score=round(mean, 2),
        net_score=round(sum(vals), 2), pos=pos, neg=neg, neu=neu,
        label=label, confidence=conf, catalysts=dict(cat),
        top_positive=top_pos, top_negative=top_neg, alerts=alerts,
    )


def portfolio_report(results: list[StockSentiment]) -> dict:
    """Summary across the whole watchlist, for the panel at the top of the tab."""
    usable = [r for r in results if r.n_headlines > 0]
    if not usable:
        return {"error": "No headlines retrieved for any selected stock."}

    bullish = [r for r in usable if "ullish" in r.label]
    bearish = [r for r in usable if "earish" in r.label]
    mixed = [r for r in usable if r.label == "Mixed"]

    all_cat = Counter()
    for r in usable:
        all_cat.update(r.catalysts)

    ranked = sorted(usable, key=lambda r: r.mean_score, reverse=True)
    flagged = [r for r in usable if r.alerts]

    return {
        "stocks_covered": len(usable),
        "no_news": len(results) - len(usable),
        "total_headlines": sum(r.n_headlines for r in usable),
        "bullish": len(bullish),
        "bearish": len(bearish),
        "mixed": len(mixed),
        "avg_sentiment": round(sum(r.mean_score for r in usable) / len(usable), 2),
        "most_positive": ranked[0] if ranked else None,
        "most_negative": ranked[-1] if ranked else None,
        "top_catalysts": all_cat.most_common(5),
        "flagged_count": len(flagged),
        "flagged": flagged,
    }
