"""Delivery channels — email, WhatsApp, Telegram.

Every sender is configured purely through environment variables and fails soft:
a delivery failure logs and returns False rather than raising, because a
notification problem should never lose the underlying data.

Channel comparison
------------------
| Channel  | Setup      | Cost              | Reliability | CI-friendly |
|----------|------------|-------------------|-------------|-------------|
| Email    | 5 min      | Free              | High        | Yes         |
| Telegram | 5 min      | Free              | High        | Yes         |
| WhatsApp | 20 min     | Free tier, then ~$0.005/msg | High | Yes  |

**Telegram is the best value** for this use case: free, no approval process,
supports long formatted messages and file attachments, and the bot setup takes
about five minutes. If you want messages on your phone, start there.

**WhatsApp via Twilio** works and is officially supported, but the sandbox
requires re-joining every 72 hours and production needs business verification.

**pywhatkit and similar libraries are deliberately not used.** They drive
web.whatsapp.com through a logged-in browser session, which cannot run headless
in CI, breaks whenever WhatsApp changes its UI, and risks your number being
flagged for automation.
"""

from __future__ import annotations

import os
from pathlib import Path


# --------------------------------------------------------------------------
# Email
# --------------------------------------------------------------------------
def send_email(
    subject: str,
    html_body: str,
    text_body: str = "",
    attachments: dict[str, Path] | None = None,
) -> bool:
    """Send via SMTP. Needs SMTP_USER, SMTP_PASS, REPORT_TO."""
    import smtplib
    from email.message import EmailMessage

    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "465"))
    user = os.environ.get("SMTP_USER", "")
    pwd = os.environ.get("SMTP_PASS", "")
    to = os.environ.get("REPORT_TO", user)

    if not (user and pwd and to):
        print("  [email] skipped — SMTP_USER / SMTP_PASS / REPORT_TO not set")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    msg.set_content(text_body or "See the HTML version of this message.")
    msg.add_alternative(html_body, subtype="html")

    for name, path in (attachments or {}).items():
        if not path or not Path(path).exists():
            continue
        p = Path(path)
        data = p.read_bytes()
        if p.suffix == ".pdf":
            msg.add_attachment(data, maintype="application", subtype="pdf",
                               filename=p.name)
        elif p.suffix in (".xlsx", ".xlsm"):
            msg.add_attachment(
                data, maintype="application",
                subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                filename=p.name)
        elif p.suffix == ".csv":
            msg.add_attachment(data, maintype="text", subtype="csv", filename=p.name)
        else:
            msg.add_attachment(data, maintype="text", subtype="html", filename=p.name)

    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=30) as s:
                s.login(user, pwd)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=30) as s:
                s.starttls()
                s.login(user, pwd)
                s.send_message(msg)
        print(f"  [email] sent to {to}")
        return True
    except Exception as exc:                                   # noqa: BLE001
        print(f"  [email] FAILED — {type(exc).__name__}: {exc}")
        return False


# --------------------------------------------------------------------------
# Telegram  (recommended)
# --------------------------------------------------------------------------
def send_telegram(text: str, document: Path | None = None) -> bool:
    """Send via Telegram bot. Needs TELEGRAM_TOKEN and TELEGRAM_CHAT_ID.

    Setup:
      1. Message @BotFather on Telegram, send /newbot, follow prompts.
      2. Copy the token it gives you -> TELEGRAM_TOKEN
      3. Message your new bot once (anything).
      4. Visit https://api.telegram.org/bot<TOKEN>/getUpdates and copy
         result[0].message.chat.id -> TELEGRAM_CHAT_ID
    """
    import requests

    token = os.environ.get("TELEGRAM_TOKEN", "")
    chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not (token and chat):
        print("  [telegram] skipped — TELEGRAM_TOKEN / TELEGRAM_CHAT_ID not set")
        return False

    # Telegram caps messages at 4096 characters
    body = text if len(text) <= 4000 else text[:3950] + "\n\n…(truncated)"

    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": body, "parse_mode": "Markdown",
                  "disable_web_page_preview": True},
            timeout=20,
        )
        r.raise_for_status()

        if document and Path(document).exists():
            with open(document, "rb") as fh:
                r2 = requests.post(
                    f"https://api.telegram.org/bot{token}/sendDocument",
                    data={"chat_id": chat},
                    files={"document": (Path(document).name, fh)},
                    timeout=60,
                )
                r2.raise_for_status()
        print("  [telegram] sent")
        return True
    except Exception as exc:                                   # noqa: BLE001
        print(f"  [telegram] FAILED — {type(exc).__name__}: {exc}")
        return False


# --------------------------------------------------------------------------
# WhatsApp via Twilio
# --------------------------------------------------------------------------
def send_whatsapp(text: str) -> bool:
    """Send via Twilio's WhatsApp API.

    Needs TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, WHATSAPP_TO.

    Setup:
      1. Sign up at twilio.com (free trial includes credit).
      2. Console -> Messaging -> Try it out -> Send a WhatsApp message.
      3. Join the sandbox by sending the given code from your phone.
      4. TWILIO_FROM  = "whatsapp:+14155238886"   (Twilio's sandbox number)
         WHATSAPP_TO  = "whatsapp:+919xxxxxxxxx"  (your number, E.164 format)

    Sandbox caveat: the session expires after 72 hours of inactivity and you
    must re-send the join code. For unattended use, move to a Twilio-approved
    production sender.
    """
    import requests

    sid = os.environ.get("TWILIO_SID", "")
    token = os.environ.get("TWILIO_TOKEN", "")
    frm = os.environ.get("TWILIO_FROM", "")
    to = os.environ.get("WHATSAPP_TO", "")

    if not all([sid, token, frm, to]):
        print("  [whatsapp] skipped — TWILIO_* / WHATSAPP_TO not set")
        return False

    body = text if len(text) <= 1500 else text[:1450] + "\n\n…(truncated)"

    try:
        r = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
            auth=(sid, token),
            data={"From": frm, "To": to, "Body": body},
            timeout=25,
        )
        r.raise_for_status()
        print(f"  [whatsapp] sent to {to}")
        return True
    except Exception as exc:                                   # noqa: BLE001
        detail = ""
        try:
            detail = f" — {r.json().get('message', '')}"       # noqa: F821
        except Exception:                                      # noqa: BLE001
            pass
        print(f"  [whatsapp] FAILED — {type(exc).__name__}: {exc}{detail}")
        return False


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------
def dispatch(
    *,
    subject: str,
    html_body: str,
    text_body: str,
    attachments: dict[str, Path] | None = None,
    channels: str = "auto",
) -> dict[str, bool]:
    """Send through the requested channels.

    channels: comma-separated ("email,telegram") or "auto" to use whichever
    are configured via environment variables.
    """
    if channels == "auto":
        wanted = []
        if os.environ.get("SMTP_USER"):
            wanted.append("email")
        if os.environ.get("TELEGRAM_TOKEN"):
            wanted.append("telegram")
        if os.environ.get("TWILIO_SID"):
            wanted.append("whatsapp")
    else:
        wanted = [c.strip().lower() for c in channels.split(",") if c.strip()]

    if not wanted:
        print("  [notify] no channels configured — nothing sent")
        return {}

    results: dict[str, bool] = {}
    if "email" in wanted:
        results["email"] = send_email(subject, html_body, text_body, attachments)
    if "telegram" in wanted:
        # Prefer the tracker workbook — it carries the most information.
        # Fall back to the PDF report, then the HTML.
        # NOTE: guard against `attachments` being None before calling .get().
        doc = None
        for key in ("xlsx", "pdf", "html"):
            cand = (attachments or {}).get(key)
            if cand is not None:
                doc = cand
                break
        results["telegram"] = send_telegram(text_body, doc)
    if "whatsapp" in wanted:
        results["whatsapp"] = send_whatsapp(text_body)
    return results
