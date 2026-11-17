"""Problem 3 tests for the pure helpers (no LLM needed)."""
from app.lead_extraction_system import missing_info as mi
from app.lead_extraction_system.extractor import parse_json_object

TEXT = "Hi, I'm Rahim Uddin. Reach me at rahim@example.com or +880 1711-123456."


def test_parse_json_with_fences():
    assert parse_json_object('```json\n{"name": "A"}\n```') == {"name": "A"}


def test_hallucinated_email_dropped():
    assert mi.valid_email("fake@nowhere.com", TEXT) is None
    assert mi.valid_email("rahim@example.com", TEXT) == "rahim@example.com"


def test_phone_and_name_validation():
    assert mi.valid_phone("+880 1711-123456", TEXT) == "+880 1711-123456"
    assert mi.valid_phone("+1 555 000 1111", TEXT) is None
    assert mi.valid_name("Rahim Uddin", TEXT) == "Rahim Uddin"
    assert mi.valid_name("John Smith", TEXT) is None


def test_missing_and_question_skip_collected():
    lead = {"name": "Rahim", "email": "r@x.com", "phone": None, "requirements": "N/A"}
    missing = mi.missing_fields(lead)
    assert missing == ["phone", "requirements"]
    q = mi.build_question(missing)
    assert "phone" in q and "email" not in q and "name" not in q
    assert mi.build_question([]) is None
