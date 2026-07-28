import json

from goz_extract.data_generation import (
    build_generation_prompt,
    generate_examples,
    parse_generation_response,
)
from goz_extract.schema import GozCode

CODES = [
    GozCode(goz_nr="0090", bezeichnung="Intraorale Infiltrationsanästhesie"),
    GozCode(goz_nr="2080", bezeichnung="Kompositfüllung, zweiflächig"),
]

VALID_RESPONSE = json.dumps(
    [
        {
            "text": "Zahn 36: Infiltrationsanästhesie, Karies excaviert, Kompositfüllung zweiflächig gelegt.",
            "expected_codes": ["0090", "2080"],
        },
        {
            "text": "Lokalanästhesie gesetzt, anschließend Füllung mit zwei Flächen in Komposit.",
            "expected_codes": ["0090", "2080"],
        },
    ]
)


def test_build_generation_prompt_mentions_all_codes_and_difficulty():
    prompt = build_generation_prompt(CODES, difficulty="medium", n_examples=2)
    assert "0090" in prompt
    assert "2080" in prompt
    assert "medium" in prompt or "mittel" in prompt
    assert "2" in prompt


def test_parse_generation_response_valid_json():
    examples = parse_generation_response(VALID_RESPONSE)
    assert len(examples) == 2
    assert examples[0].expected_codes == ["0090", "2080"]
    assert examples[0].source == "generated"


def test_parse_generation_response_handles_prose_wrapper():
    wrapped = f"Hier ist die Liste:\n```json\n{VALID_RESPONSE}\n```\nViele Grüße"
    examples = parse_generation_response(wrapped)
    assert len(examples) == 2


def test_parse_generation_response_rejects_empty():
    import pytest

    with pytest.raises(ValueError):
        parse_generation_response("Ich kann das nicht generieren.")


class _FakeChatModel:
    def invoke(self, prompt: str):
        class _Msg:
            content = VALID_RESPONSE

        return _Msg()


def test_generate_examples_uses_injected_chat_model_and_sets_difficulty():
    examples = generate_examples(_FakeChatModel(), CODES, difficulty="medium", n_examples=2)
    assert len(examples) == 2
    assert all(e.difficulty == "medium" for e in examples)
