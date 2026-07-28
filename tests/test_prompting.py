from goz_extract.prompting import build_extraction_prompt, parse_code_list_response
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


def test_parse_code_list_response_plain_list():
    result = parse_code_list_response("0090, 2080", valid_codes={"0090", "2080", "0010"})
    assert result == ["0090", "2080"]


def test_parse_code_list_response_filters_invalid_codes():
    result = parse_code_list_response("0090, 9999, 2080", valid_codes={"0090", "2080"})
    assert result == ["0090", "2080"]


def test_parse_code_list_response_dedupes_preserving_order():
    result = parse_code_list_response("2080, 0090, 2080", valid_codes={"0090", "2080"})
    assert result == ["2080", "0090"]


def test_parse_code_list_response_extracts_from_prose():
    text = "Die passenden Codes sind 0090 (Anästhesie) und 2080 (Füllung)."
    result = parse_code_list_response(text, valid_codes={"0090", "2080"})
    assert result == ["0090", "2080"]


def test_parse_code_list_response_empty_when_nothing_matches():
    result = parse_code_list_response("Keine passenden Codes gefunden.", valid_codes={"0090"})
    assert result == []
