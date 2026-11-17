"""Pure helpers (stdlib only): validation, missing-field detection, follow-up question."""
import re

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
EMPTY_VALUES = {"", "null", "none", "n/a", "na", "unknown", "not provided", "not stated"}

FIELD_LABELS = {
    "name": "your name",
    "email": "your email address",
    "phone": "your phone number",
    "requirements": "what you're looking for",
}


def clean(value) -> str | None:
    """Normalise junk like '', 'null', 'N/A' to None."""
    if value is None:
        return None
    s = str(value).strip()
    return None if s.lower() in EMPTY_VALUES else s


def digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def valid_email(email: str | None, source_text: str) -> str | None:
    """Keep only well-formed emails that really appear in what the user wrote."""
    email = clean(email)
    if not email or not EMAIL_RE.match(email):
        return None
    return email if email.lower() in source_text.lower() else None


def valid_phone(phone: str | None, source_text: str) -> str | None:
    phone = clean(phone)
    if not phone:
        return None
    d = digits(phone)
    if not (7 <= len(d) <= 15):
        return None
    return phone if d in digits(source_text) else None


def valid_name(name: str | None, source_text: str) -> str | None:
    """Every word of the name must appear in the user's text (anti-hallucination)."""
    name = clean(name)
    if not name:
        return None
    text = source_text.lower()
    tokens = re.findall(r"[A-Za-z'\-]+", name)
    return name if tokens and all(t.lower() in text for t in tokens) else None


def missing_fields(lead: dict) -> list[str]:
    return [f for f in ("name", "email", "phone", "requirements") if not clean(lead.get(f))]


def build_question(missing: list[str]) -> str | None:
    """Ask ONLY for what's missing, never for what we already have."""
    if not missing:
        return None
    labels = [FIELD_LABELS[f] for f in missing]
    if len(labels) == 1:
        joined = labels[0]
    else:
        joined = ", ".join(labels[:-1]) + f" and {labels[-1]}"
    return f"Could you please share {joined}?"
