import json

import pytest

from goz_extract.enrichment import (
    build_enrichment_prompt,
    generate_enriched_descriptions,
    parse_enrichment_response,
)
from goz_extract.schema import GozCode

CODES = [
    GozCode(goz_nr="0090", bezeichnung="Intraorale Infiltrationsanästhesie"),
    GozCode(goz_nr="2080", bezeichnung="Kompositfüllung, zweiflächig"),
]

VALID_RESPONSE = json.dumps(
    {
        "0090": "Lokale Betäubung durch Einspritzen. Umgangssprachlich: 'Infiltration', 'Spritze'.",
        "2080": "Zweiflächige Kunststofffüllung. Umgangssprachlich: 'Füllung', 'Komposit-Füllung'.",
    }
)


def test_build_enrichment_prompt_mentions_all_codes():
    prompt = build_enrichment_prompt(CODES)
    assert "0090" in prompt
    assert "2080" in prompt
    assert "Intraorale Infiltrationsanästhesie" in prompt


def test_parse_enrichment_response_valid_json():
    result = parse_enrichment_response(VALID_RESPONSE)
    assert result["0090"] == "Lokale Betäubung durch Einspritzen. Umgangssprachlich: 'Infiltration', 'Spritze'."
    assert result["2080"].startswith("Zweiflächige")


def test_parse_enrichment_response_handles_prose_wrapper():
    wrapped = f"Hier ist die Liste:\n```json\n{VALID_RESPONSE}\n```\nViele Grüße"
    result = parse_enrichment_response(wrapped)
    assert len(result) == 2


def test_parse_enrichment_response_rejects_empty():
    with pytest.raises(ValueError):
        parse_enrichment_response("Ich kann das nicht generieren.")


def test_parse_enrichment_response_rejects_non_string_values():
    invalid = json.dumps({"0090": ["nicht", "ein", "string"]})
    with pytest.raises(ValueError):
        parse_enrichment_response(invalid)


class _FakeChatModel:
    def invoke(self, prompt: str):
        class _Msg:
            content = VALID_RESPONSE

        return _Msg()


def test_generate_enriched_descriptions_uses_injected_chat_model():
    result = generate_enriched_descriptions(_FakeChatModel(), CODES)
    assert set(result) == {"0090", "2080"}
