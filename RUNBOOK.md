# SwingScope — Operational Runbook

*What to do when something breaks, and how to verify it is working.*

This exists because an automated system that fails quietly is worse than one
that fails loudly. Most of what follows is about noticing.

---

## Daily / weekly checks

**Every Monday morning** — did the Telegram message arrive?

If not, that is the signal. No message looks exactly like a quiet market, which
is why the workflows carry an `if: failure()` alert. If neither arrived,
something failed *before* the alert step could run.

| Symptom | First check |
|---|---|
| No Telegram at all | Actions tab — did the run start? |
| Run shows red | Expand the failed step; error is in its output |
| Run green, no message | `[10/10] Notify` — look for `[telegram] skipped` |
| Message but no workbook | `[6b] Daily tracker` — check for an openpyxl error |

**Monthly** — confirm `forward_log.csv` has grown by roughly four snapshots. A
stalled log is the quietest failure in the system and the most damaging, because
the evidence it accumulates cannot be reconstructed later.

---

## Failure playbook

### 1. yfinance returns no data

**Symptom:** `ABORT: no price data` or coverage below 60%.

**Cause:** rate limiting, or Yahoo changed its endpoints. Both are routine —
yfinance is an unofficial scraper of an unsupported service.

**Action:**
1. Wait 15 minutes and re-run manually. Rate limits are usually transient.
2. If it persists, the data layer falls back to bhavcopy automatically. Check
   the log for `DATA SOURCE FALLBACK`.
3. If the bhavcopy cache is empty, the fallback cannot help. Populate it via the
   app's NSE bhavcopy section.
4. If yfinance is broken for days, `pip install --upgrade yfinance` locally,
   run `python test_invariants.py 50`, and commit the version bump only if it
   passes.

**Do not** disable the health gate to force a run through. It exists precisely
to stop bad data reaching the forward log.

### 2. NSE blocks the universe fetch

**Symptom:** `CACHED FALLBACK` in stage 1.

**Cause:** NSE rotates its bot defences periodically.

**Action:** Not urgent — the bundled snapshot keeps the system running. But the
constituent list drifts, so if it persists beyond a few weeks, update
`fallback_universe.py` manually from the NSE website.

### 3. Health gate aborts the run

**Symptom:** `ABORT: health checks failed`.

**This is the system working.** It refuses to produce picks from stale or sparse
data because bad output would corrupt the forward log.

**Action:** read which check failed.
- *Freshness* — data is more than five days old. Usually an NSE holiday cluster;
  wait a day.
- *Coverage* — fewer than 60% of tickers returned. Rate limit; retry later.
- *Price sanity* — implausible values. Check the flagged tickers.

### 4. Telegram stops delivering

**Symptom:** `[telegram] skipped` or `FAILED` in stage 10.

**Action:**
1. `skipped` means the secrets did not reach the runner. Verify
   `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID` in repo Settings → Secrets. Names
   are case-sensitive.
2. `Unauthorized` means the token was revoked or regenerated.
3. `chat not found` means the chat ID is wrong, or you blocked the bot.

### 5. CI fails on a commit

**Symptom:** red cross beside the commit.

**Action:** open the run, find the failed step.
- *Syntax* — a file did not parse. Line number is given.
- *Import* — a module is missing or has a circular import.
- *app.py wiring* — a function is called but not defined, usually from a
  careless edit. This exact failure has occurred before.
- *Invariant tests* — a property broke. Read which one; the message says what
  invariant was violated.
- *Secret scan* — **treat as urgent.** If a real credential was committed,
  revoke it immediately. It is already public; rotating is not optional.

**Never merge past a red CI run.** The suite exists because manual verification
has repeatedly missed things.

### 6. Decay monitor alerts

**Symptom:** Telegram alert saying `warning` or `critical`.

**Action:** do not immediately change parameters. Reacting to a single reading
fits noise.
1. Check how many measurements exist. Below three, the trend means nothing.
2. If IC is below 0.030, reduce position size.
3. If below 0.015, stop trading the model and investigate.
4. Record the reading. The series matters more than any point in it.

---

## Disaster recovery

**What is irreplaceable:** `forward_log.csv`. Everything else can be regenerated
— price data re-downloaded, reports rebuilt, universes re-fetched. Forward
evidence cannot, because it depends on having been recorded *before* the
outcome was known.

**Protection:** it is committed to git on every run, so every historical version
is recoverable via `git log -- forward_log.csv`.

**If the repository is lost:**
1. The Streamlit deployment holds no state — nothing to recover there.
2. Re-clone from any local copy, or from the zip.
3. Restore `forward_log.csv` from an Actions artifact (90-day retention) or from
   any Telegram-delivered workbook.
4. Re-add repository secrets — these are not in the repo by design.
5. Re-enable Actions write permissions.

**If credentials leak:**
1. **Telegram:** message @BotFather, `/revoke`, generate a new token.
2. **Broker:** revoke the app in the developer portal immediately. Do not wait.
3. Rotate before cleaning git history — the credential is already public, and
   removing it from history does not un-publish it.

---

## Recovery time expectations

| Scenario | Recovery |
|---|---|
| Missed weekly run | Manual trigger, ~5 min. One snapshot gap. |
| yfinance outage | Automatic fallback, no action |
| NSE block | Automatic fallback, no action |
| Repo lost | ~30 min from zip plus secrets |
| Forward log lost | **Not recoverable beyond the last commit** |

---

## Monthly maintenance

- Confirm the forward log grew by ~4 snapshots
- Read the month-end report
- Check the decay history for trend
- Review any CI failures from the month
- Verify the F&O list and repo-rate table are not badly stale (both are
  hand-maintained and will silently persist wrong values)

## Quarterly

- Run the decay monitor
- Re-run validation and compare against 0.076
- Upgrade dependencies one at a time, testing between each

---

## What to do when nothing is wrong

Nothing. The most common operational error in a system like this is
intervening because a week was quiet. The absence of trades in a risk-off
regime is the design working, and every parameter changed mid-experiment
restarts the clock on clean evidence.

---

*Research tool. Not investment advice.*
