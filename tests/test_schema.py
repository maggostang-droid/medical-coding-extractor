import pytest
from pydantic import ValidationError

from goz_extract.schema import GozCode, NoteExample, Prediction


def test_goz_code_requires_nr_and_bezeichnung():
    code = GozCode(goz_nr="0090", bezeichnung="Intraorale Infiltrationsanästhesie")
    assert code.goz_nr == "0090"
    assert code.bezeichnung.startswith("Intraorale")


def test_goz_code_rejects_missing_fields():
    with pytest.raises(ValidationError):
        GozCode(goz_nr="0090")


def test_note_example_defaults():
    note = NoteExample(
        text="Zahn 36: Infiltrationsanästhesie, Komposit-Füllung zweiflächig.",
        expected_codes=["0090", "2080"],
    )
    assert note.difficulty is None
    assert note.source == "generated"
    assert note.split is None


def test_prediction_holds_predicted_codes():
    pred = Prediction(text="...", predicted_codes=["0090", "2080"])
    assert pred.predicted_codes == ["0090", "2080"]
