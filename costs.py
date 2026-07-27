"""Tax and correlation — the two largest unmodelled costs.

Why these two
-------------
The system charges 0.25-0.60% in transaction costs and assumes ten positions
give ten units of diversification. Both assumptions are wrong, and in the same
direction: they flatter the strategy.

**Tax.** India taxes short-term capital gains on equity at 20% (raised from 15%
in the July 2024 Budget). Every 15-20 day hold is short-term by definition.
Long-term gains — holdings over 12 months — are taxed at 12.5% with an annual
exemption. A short-horizon strategy is therefore *structurally* tax-disadvantaged
against simply holding, and the gap is large enough to change conclusions.

**Correlation.** Ten momentum names in a rising market frequently move as one.
Two banks, two NBFCs and two housing financiers are one bet on interest rates
wearing six names. The sector cap is a crude proxy for this; measured
correlation is not. If effective diversification is 3 rather than 10, real
portfolio risk is roughly 1.8x what per-position sizing assumes.

Both are computable from data already in hand. Neither needs a new data source.

Caveat
------
Tax rules change and vary by individual circumstance — set-off of losses,
carry-forward, exemption limits, residency, and whether you are classified as
an investor or a trader all matter. The figures here implement the headline
rates as a modelling input. This is not tax advice; confirm with a chartered
accountant before relying on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Indian equity taxation (headline rates — verify against current law)
# --------------------------------------------------------------------------
STCG_RATE = 0.20            # <= 12 months, equity with STT paid
LTCG_RATE = 0.125           # > 12 months
LTCG_EXEMPTION = 125_000    # annual exemption on long-term gains
STCG_HOLDING_DAYS = 365

# Statutory charges on delivery trades, as fractions of turnover
STT_DELIVERY = 0.001        # 0.1% on both buy and sell
STAMP_DUTY_BUY = 0.000015   # 0.015% on buy only
EXCHANGE_TXN = 0.0000297    # NSE
SEBI_CHARGES = 0.000001
GST_ON_CHARGES = 0.18       # on brokerage + exchange charges


@dataclass
class CostBreakdown:
    brokerage: float = 0.0
    stt: float = 0.0
    stamp_duty: float = 0.0
    exchange: float = 0.0
    sebi: float = 0.0
    gst: float = 0.0
    total_charges: float = 0.0
    charges_pct: float = 0.0
    tax: float = 0.0
    tax_pct: float = 0.0
    all_in_pct: float = 0.0
    net_gain: float = 0.0
    gross_gain: float = 0.0

    def summary(self) -> dict:
        return {k: round(v, 4) for k, v in self.__dict__.items()}


def round_trip_cost(
    entry: float, exit_price: float, qty: int,
    *,
    brokerage_per_order: float = 20.0,
    brokerage_pct_cap: float = 0.0003,      # lower of ₹20 or 0.03%
    holding_days: int = 15,
    apply_tax: bool = True,
) -> CostBreakdown:
    """Full round-trip cost including statutory charges and capital gains tax."""
    b = CostBreakdown()
    buy_val = entry * qty
    sell_val = exit_price * qty
    turnover = buy_val + sell_val

    b.gross_gain = sell_val - buy_val

    # Brokerage: lower of flat fee or percentage, per leg
    per_leg = lambda v: min(brokerage_per_order, v * brokerage_pct_cap)  # noqa: E731
    b.brokerage = per_leg(buy_val) + per_leg(sell_val)

    b.stt = turnover * STT_DELIVERY
    b.stamp_duty = buy_val * STAMP_DUTY_BUY
    b.exchange = turnover * EXCHANGE_TXN
    b.sebi = turnover * SEBI_CHARGES
    b.gst = (b.brokerage + b.exchange + b.sebi) * GST_ON_CHARGES

    b.total_charges = (b.brokerage + b.stt + b.stamp_duty +
                       b.exchange + b.sebi + b.gst)
    b.charges_pct = b.total_charges / buy_val * 100 if buy_val else 0.0

    # Tax applies only to a net gain after charges. Losses generate no tax
    # here — in practice they offset other gains, which this does not model.
    if apply_tax:
        taxable = b.gross_gain - b.total_charges
        if taxable > 0:
            rate = STCG_RATE if holding_days <= STCG_HOLDING_DAYS else LTCG_RATE
            b.tax = taxable * rate
        b.tax_pct = b.tax / buy_val * 100 if buy_val else 0.0

    b.net_gain = b.gross_gain - b.total_charges - b.tax
    b.all_in_pct = b.charges_pct + b.tax_pct
    return b


def edge_after_tax(gross_edge_pct: float, *, win_rate: float = 0.55,
                   charges_pct: float = 0.35, holding_days: int = 15) -> dict:
    """What survives of a stated edge once charges and tax are applied.

    The asymmetry that matters: charges are paid on every trade, tax only on
    winners. A strategy with a small edge and a moderate win rate loses more
    than the headline tax rate suggests.
    """
    rate = STCG_RATE if holding_days <= STCG_HOLDING_DAYS else LTCG_RATE

    # Decompose the edge into an average win and average loss consistent with
    # the stated win rate, assuming a 1.5:1 payoff.
    payoff = 1.5
    avg_loss = gross_edge_pct / (win_rate * payoff - (1 - win_rate))
    avg_win = avg_loss * payoff

    gross_per_trade = win_rate * avg_win - (1 - win_rate) * avg_loss
    after_charges = gross_per_trade - charges_pct
    # Tax is levied on winning trades net of their own charges
    taxable_per_trade = win_rate * max(avg_win - charges_pct, 0.0)
    tax_per_trade = taxable_per_trade * rate
    net = after_charges - tax_per_trade

    return {
        "gross_edge_pct": round(gross_per_trade, 4),
        "charges_pct": round(charges_pct, 4),
        "tax_pct": round(tax_per_trade, 4),
        "net_edge_pct": round(net, 4),
        "retention_pct": round(net / gross_per_trade * 100, 1) if gross_per_trade else 0.0,
        "tax_rate_applied": rate,
        "viable": net > 0,
        "cycles_per_year": round(250 / max(holding_days, 1), 1),
        "annualised_net_pct": round(net * (250 / max(holding_days, 1)), 2),
    }


def compare_horizons(gross_edge_pct: float, base_days: int = 15,
                     charges_pct: float = 0.35) -> pd.DataFrame:
    """How net returns vary with holding period.

    Gross edge is scaled with sqrt(time), not held constant. Holding a
    diffusive price process twice as long captures roughly sqrt(2) times the
    move, not the same move — comparing a fixed per-cycle edge across horizons
    would be meaningless.

    Two effects then work in the same direction as you lengthen the hold:
    charges are paid fewer times per year, and beyond 12 months the tax rate
    drops from 20% to 12.5%.
    """
    rows = []
    for days in (5, 10, 15, 20, 30, 60, 120, 250, 300, 400):
        scaled_edge = gross_edge_pct * np.sqrt(days / base_days)
        r = edge_after_tax(scaled_edge, charges_pct=charges_pct, holding_days=days)
        rows.append({
            "holding_days": days,
            "gross_edge_pct": round(scaled_edge, 3),
            "cycles_per_year": r["cycles_per_year"],
            "tax_rate": f"{r['tax_rate_applied']:.1%}",
            "net_per_cycle_pct": r["net_edge_pct"],
            "annualised_net_pct": r["annualised_net_pct"],
            "viable": r["viable"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Correlation-adjusted position sizing
# --------------------------------------------------------------------------
@dataclass
class DiversificationResult:
    n_positions: int
    mean_correlation: float
    effective_n: float
    risk_multiplier: float
    correlation_matrix: pd.DataFrame | None = None
    clusters: dict[int, list[str]] = field(default_factory=dict)
    suggested_scale: float = 1.0

    @property
    def is_concentrated(self) -> bool:
        return self.effective_n < self.n_positions * 0.5


def effective_positions(returns: pd.DataFrame) -> DiversificationResult:
    """How many *independent* bets does a basket actually represent?

    With N positions of equal weight and average pairwise correlation rho, the
    effective number of independent positions is:

        N_eff = N / (1 + (N - 1) * rho)

    Ten positions at rho = 0.6 give N_eff = 1.5. Portfolio volatility scales
    with sqrt(N/N_eff), so per-position sizing that assumes independence
    understates real risk by that factor.
    """
    if returns is None or returns.shape[1] < 2:
        return DiversificationResult(returns.shape[1] if returns is not None else 0,
                                     0.0, float(returns.shape[1]) if returns is not None else 0.0,
                                     1.0)

    r = returns.dropna(axis=1, how="all").dropna()
    if r.shape[1] < 2 or len(r) < 10:
        return DiversificationResult(r.shape[1], 0.0, float(r.shape[1]), 1.0)

    corr = r.corr()
    n = corr.shape[0]
    off = corr.to_numpy()[~np.eye(n, dtype=bool)]
    rho = float(np.nanmean(off))

    denom = 1 + (n - 1) * rho
    n_eff = n / denom if denom > 0 else float(n)
    n_eff = float(np.clip(n_eff, 1.0, n))

    # Portfolio vol scales with sqrt(N / N_eff) relative to the independent case
    risk_mult = float(np.sqrt(n / n_eff)) if n_eff > 0 else 1.0

    return DiversificationResult(
        n_positions=n,
        mean_correlation=round(rho, 3),
        effective_n=round(n_eff, 2),
        risk_multiplier=round(risk_mult, 2),
        correlation_matrix=corr.round(3),
        suggested_scale=round(1.0 / risk_mult, 3),
    )


def cluster_positions(returns: pd.DataFrame, threshold: float = 0.7) -> dict[int, list[str]]:
    """Group names that move together, using single-linkage on correlation.

    No scipy dependency — a simple union-find over the correlation graph is
    sufficient and transparent. Names in the same cluster should be treated as
    one position for risk purposes, regardless of sector labels.
    """
    if returns is None or returns.shape[1] < 2:
        return {}
    r = returns.dropna(axis=1, how="all").dropna()
    if r.shape[1] < 2:
        return {}

    corr = r.corr()
    cols = list(corr.columns)
    parent = {c: c for c in cols}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if abs(float(corr.loc[a, b])) >= threshold:
                union(a, b)

    groups: dict[str, list[str]] = {}
    for c in cols:
        groups.setdefault(find(c), []).append(c)
    return {i: sorted(v) for i, v in enumerate(
        sorted(groups.values(), key=len, reverse=True))}


def adjusted_position_size(
    base_qty: int, price: float, div: DiversificationResult,
    *, mode: str = "scale",
) -> dict:
    """Scale ATR-derived sizing down to account for measured correlation.

    'scale'  — divide every position by the risk multiplier. Keeps the same
               number of names, reduces total exposure.
    'reduce' — keep position size but hold fewer names. Same total risk,
               less admin, and honest about how many bets you really have.
    """
    if mode == "reduce":
        return {
            "mode": "reduce",
            "suggested_positions": max(1, int(round(div.effective_n))),
            "qty_per_position": base_qty,
            "note": (f"{div.n_positions} names behave like "
                     f"{div.effective_n:.1f} independent bets. Holding "
                     f"{max(1, int(round(div.effective_n)))} names at full size "
                     "carries the same real risk with less complexity."),
        }
    scaled = max(1, int(base_qty * div.suggested_scale))
    return {
        "mode": "scale",
        "qty_per_position": scaled,
        "original_qty": base_qty,
        "scale_factor": div.suggested_scale,
        "exposure_reduction_pct": round((1 - div.suggested_scale) * 100, 1),
        "note": (f"Mean pairwise correlation {div.mean_correlation:.2f} means "
                 f"portfolio risk is {div.risk_multiplier:.2f}x what independent "
                 f"sizing assumes. Scaling each position by "
                 f"{div.suggested_scale:.2f} restores the intended risk."),
    }


def build_returns_matrix(frames: dict[str, pd.DataFrame], lookback: int = 60) -> pd.DataFrame:
    """Aligned daily returns for a basket, ready for correlation analysis."""
    series = {}
    for t, df in frames.items():
        if df is None or df.empty or "Close" not in df.columns or len(df) < lookback + 2:
            continue
        series[t.replace(".NS", "")] = df["Close"].tail(lookback + 1).pct_change().dropna()
    return pd.DataFrame(series).dropna() if series else pd.DataFrame()


def verdict(div: DiversificationResult, tax: dict) -> tuple[str, str]:
    """Combined read on whether the strategy clears its true hurdle."""
    if not tax.get("viable", False):
        return "bad", (
            f"After {tax['charges_pct']:.2f}% charges and "
            f"{tax['tax_pct']:.3f}% tax, the net edge is "
            f"{tax['net_edge_pct']:+.3f}% per cycle — negative. The strategy "
            "does not clear its true hurdle, regardless of what the raw IC says."
        )
    if div.is_concentrated:
        return "warn", (
            f"Net edge {tax['net_edge_pct']:+.3f}% per cycle "
            f"({tax['retention_pct']:.0f}% of gross survives). But "
            f"{div.n_positions} positions behave like {div.effective_n:.1f} "
            f"independent bets (mean correlation {div.mean_correlation:.2f}) — "
            f"real risk is {div.risk_multiplier:.2f}x what sizing assumes."
        )
    return "good", (
        f"Net edge {tax['net_edge_pct']:+.3f}% per cycle after charges and tax "
        f"({tax['retention_pct']:.0f}% retained), annualising to "
        f"{tax['annualised_net_pct']:+.1f}%. Diversification is adequate: "
        f"{div.effective_n:.1f} effective positions from {div.n_positions}."
    )
