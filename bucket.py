"""Bucket construction — assembling the final N-stock shortlist.

Turns the screener's ranked output into a balanced, risk-aware bucket.

Three constraints, applied in this order of precedence:

  1. **Regime** (hard). Tiers the regime forbids never appear, full stop. This
     overrides every preference below. A "balanced" bucket that smuggles small
     caps into a risk-off market has defeated the only real protection in the
     system.
  2. **Sector concentration** (hard). No more than N names per sector. Five
     financials is not five positions, it is one leveraged bet on rates.
  3. **Tier balance** (soft). Within what regime permits, spread across large,
     mid and small — but never pad with low-scoring names just to fill a quota.

The bucket is a *shortlist to examine*, not a portfolio to buy. Nothing here
measures how "lucrative" a stock is; the score measures how closely a chart
matches a trend-continuation pattern. Those are different claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

import regime as rg

# Preferred tier mix by regime. Values are proportions of the target size.
TIER_MIX = {
    rg.RISK_ON:  {"large": 0.40, "mid": 0.30, "small": 0.30},
    rg.NEUTRAL:  {"large": 0.60, "mid": 0.40, "small": 0.00},
    rg.RISK_OFF: {"large": 1.00, "mid": 0.00, "small": 0.00},
}

# In risk-off, cap the bucket regardless of what the caller asked for.
REGIME_SIZE_CAP = {rg.RISK_ON: None, rg.NEUTRAL: 8, rg.RISK_OFF: 3}


@dataclass
class Bucket:
    picks: pd.DataFrame
    regime_state: str
    target_size: int
    actual_size: int
    tier_counts: dict[str, int] = field(default_factory=dict)
    sector_counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.picks is None or self.picks.empty


def build(
    ranked: pd.DataFrame,
    reg: rg.Regime,
    *,
    size: int = 10,
    max_per_sector: int = 2,
    min_score: float = 60.0,
    balance_tiers: bool = True,
    sector_lookup: dict[str, str] | None = None,
) -> Bucket:
    """Assemble the final bucket from a ranked screener frame.

    Args:
        ranked: screener output, already sorted by Score descending.
        reg: current market regime.
        size: how many names you want. May be reduced by regime.
        max_per_sector: hard cap on same-sector names.
        balance_tiers: spread across tiers, or just take the top N.
        sector_lookup: {ticker: sector}. Sector caps are skipped without it.
    """
    notes: list[str] = []

    if ranked is None or ranked.empty:
        return Bucket(pd.DataFrame(), reg.state, size, 0,
                      notes=["Screener returned no candidates."])

    df = ranked.copy()
    if "Score" in df.columns:
        df = df[df["Score"] >= min_score].sort_values("Score", ascending=False)
    if df.empty:
        return Bucket(pd.DataFrame(), reg.state, size, 0,
                      notes=[f"Nothing scored at or above {min_score:.0f}."])

    # --- 1. Regime gate (hard) ---
    before = len(df)
    if "Tier" in df.columns:
        df = df[df["Tier"].isin(reg.allowed_tiers)]
        if len(df) < before:
            notes.append(
                f"Regime {reg.state.replace('_', ' ')}: excluded {before - len(df)} "
                f"candidates in disallowed tiers "
                f"(permitted: {', '.join(sorted(reg.allowed_tiers))})."
            )
    if df.empty:
        return Bucket(pd.DataFrame(), reg.state, size, 0,
                      notes=notes + ["No candidates survive the regime filter. "
                                     "Taking no positions is the correct outcome."])

    cap = REGIME_SIZE_CAP.get(reg.state)
    target = min(size, cap) if cap else size
    if cap and size > cap:
        notes.append(f"Bucket capped at {cap} by the {reg.state.replace('_', ' ')} regime "
                     f"(requested {size}).")

    # --- 2 & 3. Selection ---
    chosen: list[int] = []
    sector_count: dict[str, int] = {}

    def sector_of(row) -> str | None:
        if not sector_lookup:
            return None
        return sector_lookup.get(str(row.get("Ticker", "")))

    def can_take(row) -> bool:
        sec = sector_of(row)
        if sec is None:
            return True
        return sector_count.get(sec, 0) < max_per_sector

    def take(idx, row) -> None:
        chosen.append(idx)
        sec = sector_of(row)
        if sec:
            sector_count[sec] = sector_count.get(sec, 0) + 1

    if balance_tiers and "Tier" in df.columns:
        mix = TIER_MIX.get(reg.state, TIER_MIX[rg.NEUTRAL])
        quota = {t: int(round(target * w)) for t, w in mix.items()
                 if t in reg.allowed_tiers and w > 0}

        # Fill each tier's quota by score order
        for tier, want in quota.items():
            pool = df[df["Tier"] == tier]
            got = 0
            for idx, row in pool.iterrows():
                if got >= want:
                    break
                if can_take(row):
                    take(idx, row)
                    got += 1
            if got < want:
                notes.append(f"Only {got} of {want} {tier}-cap slots filled — "
                             "not enough qualifying candidates.")

        # Backfill any shortfall from the overall ranking rather than padding
        # a tier with weak names.
        if len(chosen) < target:
            for idx, row in df.iterrows():
                if len(chosen) >= target:
                    break
                if idx not in chosen and can_take(row):
                    take(idx, row)
    else:
        for idx, row in df.iterrows():
            if len(chosen) >= target:
                break
            if can_take(row):
                take(idx, row)

    picks = df.loc[chosen].sort_values("Score", ascending=False).reset_index(drop=True)
    picks.insert(0, "Rank", range(1, len(picks) + 1))

    # --- Stop loss and targets ---
    #
    # Computed here rather than in each consumer, so Telegram, the Excel diary
    # and the HTML report all quote identical levels. The stop was previously
    # only visible in the app's Detail tab and never reached any output.
    #
    # Stop = entry − (ATR × tier multiple). Wider for more volatile tiers,
    # which is what makes every position risk the same rupee amount.
    import tiers as _tr
    if not picks.empty and {"Close", "ATR_pct", "Tier"}.issubset(picks.columns):
        mults = picks["Tier"].map(lambda t: _tr.params(t)["atr_mult"])
        atr_abs = picks["Close"] * picks["ATR_pct"] / 100.0
        picks["Stop"] = (picks["Close"] - atr_abs * mults).round(2)
        picks["Stop_pct"] = ((picks["Stop"] / picks["Close"] - 1) * 100).round(2)
        risk = picks["Close"] - picks["Stop"]
        # R-multiples: the levels at which the trade has earned 1x and 2x its risk
        picks["Target_1R"] = (picks["Close"] + risk).round(2)
        picks["Target_2R"] = (picks["Close"] + 2 * risk).round(2)
    if sector_lookup is not None:
        picks["Sector"] = picks["Ticker"].map(sector_lookup).fillna("—")

    tier_counts = (picks["Tier"].value_counts().to_dict()
                   if "Tier" in picks.columns else {})

    if len(picks) < target:
        notes.append(f"Bucket has {len(picks)} of {target} requested. Padding with "
                     "lower-scoring names would defeat the purpose of ranking.")
    if sector_lookup and sector_count:
        maxed = [s for s, c in sector_count.items() if c >= max_per_sector]
        if maxed:
            notes.append(f"Sector cap reached: {', '.join(maxed)} "
                         f"(max {max_per_sector} each).")

    return Bucket(picks, reg.state, target, len(picks),
                  tier_counts, sector_count, notes)


def to_text(bucket: Bucket, *, regime_desc: str = "") -> str:
    """Compact plain-text rendering — used for WhatsApp and SMS."""
    if bucket.is_empty:
        return (f"SwingScope — no picks\n"
                f"Regime: {bucket.regime_state.replace('_', ' ').upper()}\n"
                + ("\n".join(f"- {n}" for n in bucket.notes) if bucket.notes else ""))

    lines = [
        "*SwingScope bucket*",
        f"Regime: {bucket.regime_state.replace('_', ' ').upper()}",
    ]
    if regime_desc:
        lines.append(f"_{regime_desc}_")
    lines.append("")

    has_mom = "Momentum" in bucket.picks.columns
    has_stop = "Stop" in bucket.picks.columns
    has_targets = "Target_1R" in bucket.picks.columns
    for _, r in bucket.picks.iterrows():
        tier = str(r.get("Tier", ""))[:1].upper()
        # Show the raw momentum figure alongside the percentile. A score of 100
        # means "best in today's universe", which in a weak market could still
        # be a negative return — the reader needs both numbers.
        mom = f" · mom {r['Momentum']:+.0f}%" if has_mom else ""
        lines.append(
            f"{int(r['Rank'])}. {r['Ticker']} [{tier}] "
            f"sc {int(r['Score'])}{mom} · ₹{r['Close']:,.0f} · "
            f"RSI {r['RSI']:.0f} · ATR {r['ATR_pct']:.1f}%"
        )
        # The stop is the number that decides position size, so it belongs on
        # its own line rather than buried in a run of statistics.
        if has_stop:
            lines.append(
                f"    stop ₹{r['Stop']:,.0f} ({r['Stop_pct']:+.1f}%)"
                + (f"  ·  1R ₹{r['Target_1R']:,.0f}"
                   f"  ·  2R ₹{r['Target_2R']:,.0f}" if has_targets else "")
            )

    if bucket.tier_counts:
        lines.append("")
        lines.append("Mix: " + ", ".join(f"{k} {v}" for k, v in
                                         sorted(bucket.tier_counts.items())))
    # Volatility scaling belongs in the message — a bucket at 40% exposure is a
    # materially different instruction from the same bucket at 100%.
    for n in bucket.notes:
        if "Volatility scaling" in n:
            lines.append(n)
    lines += ["",
              "Stops are ATR-based: entry minus a tier-specific multiple of the "
              "14-day range.",
              "Analytical view, not advice. EOD data — verify on your terminal."]
    return "\n".join(lines)
