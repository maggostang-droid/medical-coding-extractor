from goz_extract.prompting import (
    build_extraction_prompt,
    parse_code_list_response,
    restrict_to_candidates,
)
from goz_extract.schema import GozCode

NOTE = "Zahn 36: Infiltrationsanästhesie, Karies excaviert, Kompositfüllung zweiflächig gelegt."
CANDIDATES = [
    GozCode(goz_nr="0090", bezeichnung="Intraorale Infiltrationsanästhesie"),
    GozCode(goz_nr="2080", bezeichnung="Kompositfüllung, zweiflächig"),
]


def test_finetune_prompt_has_no_candidate_list():
    prompt = build_extraction_prompt(NOTE, candidates=None)
    assert NOTE in prompt
    assert "0090" not in prompt


def test_rag_prompt_includes_candidates():
    prompt = build_extraction_prompt(NOTE, candidates=CANDIDATES)
    assert NOTE in prompt
    assert "0090" in prompt
    assert "2080" in prompt
    assert "Intraorale Infiltrationsanästhesie" in prompt


def test_parse_code_list_response_plain_json_array():
    result = parse_code_list_response('["0090", "2080"]', valid_codes={"0090", "2080", "0010"})
    assert result == ["0090", "2080"]


def test_parse_code_list_response_filters_invalid_codes():
    result = parse_code_list_response(
        '["0090", "9999", "2080"]', valid_codes={"0090", "2080"}
    )
    assert result == ["0090", "2080"]


def test_parse_code_list_response_dedupes_preserving_order():
    result = parse_code_list_response('["2080", "0090", "2080"]', valid_codes={"0090", "2080"})
    assert result == ["2080", "0090"]


def test_parse_code_list_response_empty_json_array():
    result = parse_code_list_response("[]", valid_codes={"0090"})
    assert result == []


def test_parse_code_list_response_ignores_codes_mentioned_outside_the_json_array():
    # Der eigentliche Zweck des JSON-Formats: wenn das Modell im Fließtext um
    # das Array herum Kandidaten erwähnt/kommentiert, dürfen die NICHT als
    # Vorhersage zählen - nur was tatsächlich im Array steht.
    text = 'Die Kandidaten 0090 und 2080 wurden geprüft. Passend ist: ["2080"]'
    result = parse_code_list_response(text, valid_codes={"0090", "2080"})
    assert result == ["2080"]


def test_parse_code_list_response_falls_back_to_prose_when_no_json_array():
    # Sicherheitsnetz: liefert das Modell trotz Instruktion kein valides
    # JSON, wird auf die alte, tolerante Prosa-Regex zurückgefallen statt
    # eine leere Vorhersage zu erzwingen.
    text = "Die passenden Codes sind 0090 (Anästhesie) und 2080 (Füllung)."
    result = parse_code_list_response(text, valid_codes={"0090", "2080"})
    assert result == ["0090", "2080"]


def test_parse_code_list_response_empty_when_nothing_matches():
    result = parse_code_list_response("Keine passenden Codes gefunden.", valid_codes={"0090"})
    assert result == []


def test_restrict_to_candidates_drops_codes_not_offered():
    # Codes, die das Modell nennt, aber gar nicht als Kandidat im Prompt
    # standen, sind per Definition Halluzinationen und werden rausgefiltert.
    result = restrict_to_candidates(["0090", "9999"], CANDIDATES)
    assert result == ["0090"]


def test_restrict_to_candidates_keeps_order_and_dedup_from_input():
    result = restrict_to_candidates(["2080", "0090"], CANDIDATES)
    assert result == ["2080", "0090"]
