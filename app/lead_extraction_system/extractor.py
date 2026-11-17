"""Calls the fallback-enabled LLM, validates JSON, merges with known lead state."""
import json
import re

from app.model_fallback_system.fallback_manager import FallbackManager
from app.lead_extraction_system import missing_info as mi
from app.lead_extraction_system.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from app.lead_extraction_system.schemas import (
    ExtractRequest,
    ExtractResponse,
    Lead,
)


def parse_json_object(text: str) -> dict:
    """Tolerant parser: strips ```json fences and finds the first {...} block."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError("JSON output is not an object")
    return obj


def _validator(text: str) -> None:
    obj = parse_json_object(text)
    if not any(k in obj for k in ("name", "email", "phone", "requirements")):
        raise ValueError("JSON has none of the expected keys")


class LeadExtractor:
    def __init__(self, manager: FallbackManager):
        self.manager = manager

    async def extract(self, req: ExtractRequest) -> ExtractResponse:
        known = (req.existing_lead or Lead()).model_dump()
        transcript = "\n".join(
            f"{'Customer' if m.role == 'user' else 'Agent'}: {m.content}"
            for m in req.conversation
        )
        user_text = "\n".join(m.content for m in req.conversation if m.role == "user")

        prompt = USER_TEMPLATE.format(
            known=json.dumps({k: v for k, v in known.items() if v}, indent=2) or "{}",
            transcript=transcript,
        )
        result = await self.manager.generate(
            prompt, system=SYSTEM_PROMPT, json_mode=True, validator=_validator
        )
        raw = parse_json_object(result.text)

        # Sanitise against what the customer actually wrote, then merge.
        # Include known values in the source so a re-validation never drops them.
        extracted = {
            "name": mi.valid_name(raw.get("name"), user_text),
            "email": mi.valid_email(raw.get("email"), user_text),
            "phone": mi.valid_phone(raw.get("phone"), user_text),
            "requirements": mi.clean(raw.get("requirements")),
        }
        merged = {
            f: extracted[f] or mi.clean(known.get(f)) for f in extracted
        }

        missing = mi.missing_fields(merged)
        return ExtractResponse(
            lead=Lead(**merged),
            already_collected=[f for f in merged if merged[f]],
            missing_fields=missing,
            is_complete=not missing,
            next_question=mi.build_question(missing),
            provider_used=result.provider,
            model_used=result.model,
            fallback_used=result.fallback_used,
            latency_ms=result.latency_ms,
        )
