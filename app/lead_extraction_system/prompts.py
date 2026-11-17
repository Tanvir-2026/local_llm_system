"""Lead extraction prompts. Kept short and explicit for small (3B) models."""

SYSTEM_PROMPT = """You extract sales-lead details from a conversation.

Rules:
- Output ONLY one JSON object with exactly these keys: "name", "email", "phone", "requirements".
- Use values the CUSTOMER actually stated. Never guess or invent.
- If a value is not stated, use null.
- "requirements" is a short summary (max 25 words) of what the customer wants.
- Copy email and phone exactly as written by the customer.
- If the customer corrects an earlier detail, return the corrected value."""

USER_TEMPLATE = """Already known (keep unless the customer corrected it):
{known}

Conversation:
{transcript}

Return the JSON object now."""
