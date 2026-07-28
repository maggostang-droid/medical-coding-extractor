from goz_extract.curate import curate_codes

RAW_ENTRIES = [
    {"goz_nr": "0010", "bezeichnung": "Eingehende Untersuchung", "kategorie": "A__Allgemeine_zahnärztliche_Leistungen"},
    {"goz_nr": "2080", "bezeichnung": "Kompositfüllung zweiflächig", "kategorie": "C__Konservierende_Leistungen"},
    {"goz_nr": "9010", "bezeichnung": "Implantatinsertion", "kategorie": "K__Implantologische_Leistungen"},
    {"goz_nr": "0500", "bezeichnung": "Osteotomie", "kategorie": "D__Chirurgische_Leistungen"},
]


def test_curate_keeps_only_allowed_categories():
    result = curate_codes(RAW_ENTRIES, allowed_prefixes=("A__", "C__"))
    codes = {c.goz_nr for c in result}
    assert codes == {"0010", "2080"}


def test_curate_only_exposes_public_fields():
    result = curate_codes(RAW_ENTRIES, allowed_prefixes=("A__",))
    assert len(result) == 1
    assert result[0].goz_nr == "0010"
    assert result[0].bezeichnung == "Eingehende Untersuchung"
    assert not hasattr(result[0], "kategorie")


def test_curate_raises_on_missing_field():
    import pytest

    broken = [{"goz_nr": "0010", "kategorie": "A__x"}]
    with pytest.raises(KeyError):
        curate_codes(broken, allowed_prefixes=("A__",))
