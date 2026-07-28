"""Model governance — registry, provenance and audit trail.

Why this exists
---------------
Audited against SR 11-7 (Federal Reserve model risk management guidance), this
system was strong on two of the three validation pillars and had a genuine gap
on the third.

  **Conceptual soundness** — covered. Factor neutralisation, the signal
  laboratory, permutation testing and documented rationale for every parameter.

  **Ongoing monitoring** — covered. Decay monitor, health gate, forward log.

  **Outcomes analysis** — partially covered. The forward log exists but is empty.
  Only time fixes that.

  **Reproducibility** — NOT covered, and this was the real gap. SR 11-7 expects
  documentation "detailed enough that a knowledgeable third party could
  reproduce key work". Given a pick from three weeks ago, nothing here could
  reconstruct why. yfinance silently revises history, the code changes, and no
  record tied outputs to the inputs and version that produced them.

What this module adds
---------------------
  **Model registry** — a central inventory: what models exist, their status,
  validation evidence, known limitations. SR 11-7 expects this, and it also
  guards against the Knight Capital failure mode, where dormant code was
  accidentally reactivated and lost $440m in 45 minutes. `composite_v1` still
  exists in this codebase and remains selectable; the registry marks it
  RETIRED - FAILED VALIDATION so that cannot happen quietly.

  **Provenance stamping** — every output records the code version, parameters
  and a hash of the input data. A pick can then be traced to exactly what
  produced it.

  **Append-only decision log** — an immutable record. Entries are never edited
  or deleted, which is what makes a trail worth anything.

What cannot be closed
---------------------
SR 11-7 requires three lines of defence: developers, independent validation,
and internal audit. In a one-person project the same person occupies all three,
and no amount of code fixes that. It is a structural limitation and is recorded
here rather than papered over.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

REGISTRY_PATH = Path("model_registry.json")
AUDIT_LOG_PATH = Path("audit_trail.jsonl")      # append-only, one JSON per line


# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------
@dataclass
class ModelRecord:
    model_id: str
    name: str
    status: str                    # active | retired | candidate | failed
    risk_tier: str                 # high | medium | low
    owner: str
    purpose: str
    theoretical_basis: str
    validation_evidence: dict = field(default_factory=dict)
    known_limitations: list[str] = field(default_factory=list)
    last_validated: str = ""
    next_validation_due: str = ""
    approved_uses: list[str] = field(default_factory=list)
    prohibited_uses: list[str] = field(default_factory=list)


# The inventory. Every model that exists in the codebase must appear here,
# including retired ones — an undocumented model is exactly what examiners
# look for, and exactly what gets reactivated by accident.
MODELS = [
    ModelRecord(
        model_id="momentum_12_1_v2",
        name="12-1 Momentum",
        status="active",
        risk_tier="high",
        owner="sole developer (see limitations)",
        purpose="Cross-sectional ranking of NSE equities for 15-day holding periods.",
        theoretical_basis=(
            "Jegadeesh & Titman (1993). Twelve-month return excluding the most "
            "recent month; the skip avoids contamination by short-term reversal, "
            "which operates in the opposite direction over ~21 days."
        ),
        validation_evidence={
            "residual_ic": 0.0553,
            "newey_west_t": 3.73,
            "clears_harvey_liu_zhu_t3": True,
            "permutation_p": 0.02,
            "windows": 62,
            "horizon_plateau_days": "10-90",
            "sub_period_ic": [0.1223, 0.0825, 0.0762],
            "sub_periods_all_positive": True,
            "gross_quintile_spread_pct": 1.42,
            "measured_on": "2026-07-28",
            "universe": "Nifty 500",
        },
        known_limitations=[
            "Measured on the same data used to select it from twelve candidates. "
            "Optimistic by construction; expect decay live.",
            "Sub-period IC declining monotonically (0.122 -> 0.083 -> 0.076). "
            "Cause unresolved: arbitrage, regime, or noise.",
            "Delisting survivorship bias remains. Bhavcopy covers traded "
            "securities; wound-up companies leave no trace.",
            "No live forward evidence as of the validation date.",
            "Momentum is documented to crash in market reversals. Partially "
            "mitigated by the regime gate and volatility scaling.",
            "Effective diversification measured at ~1.8 independent bets from "
            "10 positions; real risk ~2.4x what per-position sizing assumes.",
        ],
        last_validated="2026-07-28",
        next_validation_due="2026-10-28",
        approved_uses=[
            "Generating a research shortlist for manual review",
            "Paper trading and forward-evidence accumulation",
        ],
        prohibited_uses=[
            "Automated order placement",
            "Any use without the regime gate active",
            "Small-cap trading — ~1.5% round-trip cost exceeds the ~1.4% gross spread",
            "Any use presented to third parties as investment advice",
        ],
    ),
    ModelRecord(
        model_id="composite_v1",
        name="Five-component composite (Trend/Momentum/Volume/RS/Setup)",
        status="retired",
        risk_tier="high",
        owner="sole developer",
        purpose="Superseded. Retained only for comparison.",
        theoretical_basis=(
            "Weighted blend of five technical components. Weights set by "
            "reasoning about what should matter, without testing whether the "
            "components were independent. They were not."
        ),
        validation_evidence={
            "raw_ic": 0.0506,
            "residual_ic": 0.0041,
            "ic_retention_pct": 13.9,
            "fama_macbeth_t": 0.17,
            "correlation_with_1m_reversal": 0.76,
            "verdict": "FAILED - no incremental content over known factors",
        },
        known_limitations=[
            "FAILED VALIDATION. All five components measured the same underlying "
            "quantity. Residual IC statistically indistinguishable from zero.",
        ],
        last_validated="2026-07-27",
        next_validation_due="n/a - retired",
        approved_uses=["Comparison against the active model only"],
        prohibited_uses=[
            "Any live use whatsoever",
            "Generating picks for trading or paper trading",
        ],
    ),
]


def registry() -> pd.DataFrame:
    return pd.DataFrame([asdict(m) for m in MODELS])


def active_model() -> ModelRecord | None:
    return next((m for m in MODELS if m.status == "active"), None)


def check_model_permitted(model_id: str, use: str) -> tuple[bool, str]:
    """Gate a model against its approved uses. Guards the Knight Capital failure."""
    rec = next((m for m in MODELS if m.model_id == model_id), None)
    if rec is None:
        return False, (f"Model '{model_id}' is not in the registry. Undocumented "
                       "models must not be used — add it to MODELS first.")
    if rec.status != "active":
        return False, (f"Model '{model_id}' has status '{rec.status}'. "
                       f"{rec.known_limitations[0] if rec.known_limitations else ''}")
    for p in rec.prohibited_uses:
        if use.lower() in p.lower() or p.lower() in use.lower():
            return False, f"Prohibited use for '{model_id}': {p}"
    return True, f"'{model_id}' permitted for '{use}'."


def validation_due(as_of: dt.date | None = None) -> list[dict]:
    """Which models are overdue for revalidation.

    SR 11-7 expects risk-based frequency; high-risk models annually at minimum,
    and any material change triggers out-of-cycle validation. Quarterly is used
    here given the observed IC decline.
    """
    today = as_of or dt.date.today()
    out = []
    for m in MODELS:
        if m.status != "active" or not m.next_validation_due:
            continue
        try:
            due = dt.date.fromisoformat(m.next_validation_due)
        except ValueError:
            continue
        out.append({
            "model_id": m.model_id,
            "due": m.next_validation_due,
            "days_remaining": (due - today).days,
            "overdue": today > due,
        })
    return out


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def code_version() -> dict:
    """Identify the code that produced an output."""
    info = {"python": platform.python_version(), "platform": platform.system()}
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=5).decode().strip()
        info["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL, timeout=5).decode().strip())
    except Exception:                                          # noqa: BLE001
        info["git_commit"] = "unavailable"
        info["git_dirty"] = None

    # Hash the modules that actually determine output
    critical = ["momentum.py", "scoring.py", "bucket.py", "regime.py", "tiers.py"]
    h = hashlib.sha256()
    for f in critical:
        p = Path(f)
        if p.exists():
            h.update(p.read_bytes())
    info["model_code_hash"] = h.hexdigest()[:16]
    return info


def data_fingerprint(frames: dict[str, pd.DataFrame]) -> dict:
    """Hash the input data, so a decision can be tied to the exact prices used.

    This matters because yfinance silently revises history — adjusted closes
    change after corporate actions. Without a fingerprint there is no way to
    tell whether a past decision looks wrong because the model erred or because
    the data beneath it moved.
    """
    if not frames:
        return {"n_tickers": 0, "hash": None}

    h = hashlib.sha256()
    latest_dates, n_bars = [], 0
    for t in sorted(frames):
        df = frames[t]
        if df is None or df.empty or "Close" not in df.columns:
            continue
        tail = df["Close"].tail(30)
        h.update(t.encode())
        h.update(np.ascontiguousarray(tail.to_numpy(dtype="float64")).tobytes())
        latest_dates.append(pd.Timestamp(df.index[-1]).date().isoformat())
        n_bars += len(df)

    return {
        "n_tickers": len(frames),
        "total_bars": n_bars,
        "latest_bar": max(latest_dates) if latest_dates else None,
        "hash": h.hexdigest()[:16],
    }


def stamp(frames: dict[str, pd.DataFrame] | None = None,
          params: dict | None = None, model_id: str = "momentum_12_1_v2") -> dict:
    """Full provenance record for one run."""
    return {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "model_id": model_id,
        "code": code_version(),
        "data": data_fingerprint(frames or {}),
        "params": params or {},
    }


# --------------------------------------------------------------------------
# Append-only audit trail
# --------------------------------------------------------------------------
def audit(event: str, detail: dict, *, path: Path = AUDIT_LOG_PATH) -> None:
    """Append one immutable entry. Never edits or deletes.

    JSONL rather than CSV: each line is independent, so a partial write cannot
    corrupt earlier entries, and appending never rewrites existing content.
    """
    entry = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "event": event,
        "detail": detail,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def read_audit(path: Path = AUDIT_LOG_PATH, limit: int | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # a corrupt line must not lose the rest of the trail
    df = pd.DataFrame(rows)
    return df.tail(limit) if limit else df


def verify_audit_integrity(path: Path = AUDIT_LOG_PATH) -> dict:
    """Basic tamper checks: parseability, chronological order, gaps."""
    if not path.exists():
        return {"exists": False}

    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    parsed, corrupt = [], 0
    for l in lines:
        try:
            parsed.append(json.loads(l))
        except json.JSONDecodeError:
            corrupt += 1

    ts = pd.to_datetime([p.get("ts") for p in parsed], errors="coerce")
    ts = pd.Series(ts).dropna()
    ordered = bool(ts.is_monotonic_increasing) if len(ts) > 1 else True

    return {
        "exists": True,
        "entries": len(parsed),
        "corrupt_lines": corrupt,
        "chronological": ordered,
        "first_entry": ts.min().isoformat() if len(ts) else None,
        "last_entry": ts.max().isoformat() if len(ts) else None,
        "note": ("Entries out of chronological order suggest the file was edited "
                 "rather than appended to." if not ordered else "No anomalies."),
    }


# --------------------------------------------------------------------------
# Compliance self-assessment
# --------------------------------------------------------------------------
def sr11_7_assessment() -> pd.DataFrame:
    """Honest self-assessment against SR 11-7 expectations.

    A self-assessment is not an audit — the same person built the system and
    is grading it, which is precisely the independence problem SR 11-7's
    three-lines structure exists to solve. Read it as a gap list, not assurance.
    """
    rows = [
        ("Conceptual soundness", "Met",
         "Factor neutralisation against six known factors, signal laboratory "
         "testing twelve candidates, permutation testing, documented rationale "
         "for every parameter."),
        ("Ongoing monitoring", "Met",
         "Quarterly decay monitor with alerting, health gate on every run, "
         "invariant test suite in CI, failure notification."),
        ("Outcomes analysis", "Partial",
         "Forward log built and running, but empty. Only elapsed time closes this."),
        ("Model inventory", "Met",
         "Registry in this module. All models catalogued including retired ones, "
         "with status, limitations, approved and prohibited uses."),
        ("Reproducibility", "Met",
         "Provenance stamping records code hash, git commit and a fingerprint of "
         "the input data for every run."),
        ("Audit trail", "Met",
         "Append-only JSONL log with integrity verification."),
        ("Documentation", "Met",
         "Full user manual, method documentation in-app, rationale in every "
         "module docstring, audit report."),
        ("Independent validation", "NOT MET",
         "Structurally impossible. One person is developer, validator and "
         "auditor. SR 11-7's three-lines-of-defence model cannot be satisfied "
         "by a solo project. This is a real limitation, not a formality."),
        ("Change management", "Partial",
         "Git history plus CI on every push. Not formally linked to model "
         "outputs, and no approval gate — the same person writes and merges."),
        ("Stress testing", "NOT MET",
         "No scenario analysis against historical crises. Momentum crash "
         "protection is implemented but its behaviour in 2008- or 2020-style "
         "conditions is untested."),
        ("Data lineage", "Partial",
         "Fingerprinting records what data was used. Vendor risk is real and "
         "unmitigated: yfinance is a single unofficial point of failure."),
        ("Kill switch", "Partial",
         "Health gate aborts on bad data; regime gate suppresses tiers; decay "
         "monitor alerts. No single deliberate stop control."),
    ]
    return pd.DataFrame(rows, columns=["requirement", "status", "evidence"])
