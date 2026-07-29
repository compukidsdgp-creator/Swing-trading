"""Resilience — surviving transient failures without hiding real ones.

The distinction this module rests on
------------------------------------
**Transient failures** are worth retrying: a rate limit, a timeout, a dropped
connection. The same call a minute later usually succeeds.

**Persistent failures** are not: an API that changed shape, a missing file, bad
credentials. Retrying those wastes time and, worse, turns a clear error into a
slow one.

Retrying indiscriminately is how a five-second failure becomes a five-minute
failure with the same outcome. The classifier below is deliberately
conservative — anything it does not recognise is treated as persistent and
raised immediately.

What this is not
----------------
This does not repair code. There is a fashion for AI systems that diagnose a
break and patch themselves, and for a system whose output influences money
decisions that is the wrong objective.

An auto-patcher optimises for "does it run now". Consider a real bug from this
codebase: yfinance returned a partial bar with a NaN close, which made every
last-row comparison false and silently emptied the screener. A patcher would
most likely have added a NaN filter at the screening stage — the symptom
disappears, the run succeeds, and corrupted data continues flowing into the
forward log. The correct fix was at the fetch stage, and it required
understanding why the comparison failed rather than that it failed.

So: retry transient failures, fall back where a genuine alternative exists,
degrade to a reduced but honest result, and otherwise **fail loudly**. Never
paper over.
"""

from __future__ import annotations

import functools
import random
import time
from dataclasses import dataclass, field

# Substrings that indicate a failure worth retrying. Deliberately narrow —
# anything unrecognised is treated as persistent and raised at once.
TRANSIENT_MARKERS = (
    "rate limit", "ratelimit", "too many requests", "429",
    "timeout", "timed out", "connection reset", "connection aborted",
    "connection refused", "temporarily unavailable", "503", "502", "504",
    "remote end closed", "read operation timed out", "max retries exceeded",
    "ssl", "eof occurred",
)

# Substrings that mean retrying is pointless. Checked first.
PERSISTENT_MARKERS = (
    "unauthorized", "forbidden", "401", "403", "404", "not found",
    "no such file", "invalid api key", "authentication",
    "unexpected keyword argument", "object has no attribute",
    "not defined", "cannot import",
)


@dataclass
class RetryStats:
    attempts: int = 0
    total_delay: float = 0.0
    succeeded: bool = False
    last_error: str = ""
    classified: str = ""
    log: list[str] = field(default_factory=list)


def classify(exc: Exception) -> str:
    """transient | persistent | unknown.

    Unknown is treated as persistent by the retry wrapper. Failing fast on an
    unrecognised error is better than a slow failure with the same outcome.
    """
    msg = f"{type(exc).__name__}: {exc}".lower()
    for m in PERSISTENT_MARKERS:
        if m in msg:
            return "persistent"
    for m in TRANSIENT_MARKERS:
        if m in msg:
            return "transient"
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "transient"
    if isinstance(exc, (TypeError, AttributeError, NameError,
                        ImportError, KeyError, ValueError)):
        return "persistent"
    return "unknown"


def retry(
    max_attempts: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    on_retry=None,
):
    """Retry transient failures with exponential backoff.

    Jitter is on by default. Without it, several retries scheduled together
    resume in lockstep and reproduce the burst that caused the rate limit.

    Persistent and unrecognised failures raise immediately.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:                       # noqa: BLE001
                    last = exc
                    kind = classify(exc)
                    if kind != "transient" or attempt == max_attempts:
                        raise
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    if jitter:
                        delay *= 0.5 + random.random()
                    if on_retry:
                        on_retry(attempt, delay, exc)
                    time.sleep(delay)
            if last:
                raise last
        return wrapper
    return decorator


def call_with_retry(fn, *args, max_attempts: int = 3, base_delay: float = 2.0,
                    max_delay: float = 30.0, say=None, **kwargs
                    ) -> tuple[object, RetryStats]:
    """Call a function with retry, returning (result, stats).

    Returns rather than raises so callers can degrade gracefully. `stats`
    records what happened, which matters: a run that succeeded on the third
    attempt is not the same as one that succeeded first time, and the
    difference is worth surfacing.
    """
    stats = RetryStats()
    for attempt in range(1, max_attempts + 1):
        stats.attempts = attempt
        try:
            result = fn(*args, **kwargs)
            stats.succeeded = True
            return result, stats
        except Exception as exc:                               # noqa: BLE001
            kind = classify(exc)
            stats.classified = kind
            stats.last_error = f"{type(exc).__name__}: {exc}"[:200]

            if kind != "transient":
                stats.log.append(
                    f"attempt {attempt}: {kind} failure, not retrying — "
                    f"{stats.last_error[:90]}")
                if say:
                    say(f"    {kind} failure, not retrying: "
                        f"{stats.last_error[:80]}")
                return None, stats

            if attempt == max_attempts:
                stats.log.append(
                    f"attempt {attempt}: transient failure, attempts exhausted")
                if say:
                    say(f"    transient failure after {attempt} attempts: "
                        f"{stats.last_error[:80]}")
                return None, stats

            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            delay *= 0.5 + random.random()
            stats.total_delay += delay
            stats.log.append(f"attempt {attempt}: transient, retrying in "
                             f"{delay:.1f}s")
            if say:
                say(f"    transient failure, retrying in {delay:.0f}s "
                    f"({attempt}/{max_attempts})")
            time.sleep(delay)

    return None, stats


@dataclass
class DegradationReport:
    """What the run gave up in order to complete.

    Degradation is acceptable; silent degradation is not. A bucket built on
    40% of the universe is a different object from one built on all of it, and
    the difference belongs in the output rather than in nobody's head.
    """
    degraded: bool = False
    items: list[str] = field(default_factory=list)

    def note(self, what: str) -> None:
        self.degraded = True
        self.items.append(what)

    def summary(self) -> str:
        if not self.degraded:
            return "No degradation — all components ran normally."
        return ("Completed with reduced capability:\n"
                + "\n".join(f"  - {i}" for i in self.items)
                + "\n\nResults are usable but less complete than usual. "
                  "Weigh that before acting on them.")


def health_gate(checks: dict, *, critical: set | None = None) -> tuple[bool, list[str]]:
    """Decide whether to proceed, given a set of pass/fail checks.

    Critical checks abort. Non-critical checks degrade and are reported.
    The default is to treat anything unlisted as non-critical, so adding a new
    check cannot accidentally start blocking runs.
    """
    critical = critical or set()
    failures = [k for k, v in checks.items() if not v]
    blocking = [f for f in failures if f in critical]
    return (len(blocking) == 0, failures)
