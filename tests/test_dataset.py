import json

from goz_extract.dataset import (
    load_golden_synth_fixtures,
    restrict_to_valid_codes,
    split_dataset,
)
from goz_extract.schema import NoteExample


def test_load_golden_synth_fixtures(tmp_path):
    fixture = {
        "input": "Zahn 36: Anästhesie, Füllung zweiflächig.",
        "expected_codes": ["0090", "2080"],
    }
    (tmp_path / "002_synth_02_fuellung.json").write_text(json.dumps(fixture), encoding="utf-8")
    (tmp_path / "not_a_fixture.txt").write_text("ignore me", encoding="utf-8")

    examples = load_golden_synth_fixtures(tmp_path)

    assert len(examples) == 1
    assert examples[0].text == fixture["input"]
    assert examples[0].expected_codes == ["0090", "2080"]
    assert examples[0].source == "golden_synth"


def test_load_golden_synth_fixtures_ignores_real_fixtures(tmp_path):
    # Sicherheitsnetz: reale Praxisfall-Fixtures (real_*) dürfen NIEMALS
    # geladen werden, auch nicht versehentlich - siehe Design-Spec,
    # Abschnitt "Datenherkunft & IP-Abgrenzung".
    synth_fixture = {"input": "Synthetische Notiz.", "expected_codes": ["0090"]}
    real_fixture = {"input": "Echte Patientennotiz aus der Praxis.", "expected_codes": ["0090"]}
    (tmp_path / "005_synth_05_x.json").write_text(json.dumps(synth_fixture), encoding="utf-8")
    (tmp_path / "006_real_01_x.json").write_text(json.dumps(real_fixture), encoding="utf-8")

    examples = load_golden_synth_fixtures(tmp_path)

    assert len(examples) == 1
    assert examples[0].text == "Synthetische Notiz."


def test_load_golden_synth_fixtures_skips_codes_outside_valid_set(tmp_path):
    # Fixtures mit Codes außerhalb des kuratierten Label-Space (z.B. GOÄ-
    # Codes wie "Ä6") dürfen nicht geladen werden - sie sind von keinem der
    # beiden Ansätze beantwortbar und würden jede Eval-Metrik künstlich
    # nach unten ziehen (siehe finale Review, Punkt 1).
    in_space = {"input": "Notiz mit gültigen Codes.", "expected_codes": ["0090", "2080"]}
    out_of_space = {"input": "Notiz mit GOÄ-Code.", "expected_codes": ["0090", "Ä6"]}
    (tmp_path / "001_synth_in_space.json").write_text(json.dumps(in_space), encoding="utf-8")
    (tmp_path / "002_synth_out_of_space.json").write_text(json.dumps(out_of_space), encoding="utf-8")

    examples = load_golden_synth_fixtures(tmp_path, valid_codes={"0090", "2080"})

    assert len(examples) == 1
    assert examples[0].text == in_space["input"]


def test_restrict_to_valid_codes_projects_instead_of_dropping():
    examples = [
        NoteExample(text="Notiz A", expected_codes=["0090", "2080"]),
        NoteExample(text="Notiz B", expected_codes=["2080"]),
        NoteExample(text="Notiz C", expected_codes=["0090"]),
    ]

    result = restrict_to_valid_codes(examples, valid_codes={"0090"})

    assert len(result) == 2
    assert result[0].text == "Notiz A"
    assert result[0].expected_codes == ["0090"]
    assert result[1].text == "Notiz C"
    assert result[1].expected_codes == ["0090"]


def test_restrict_to_valid_codes_drops_examples_with_no_remaining_codes():
    examples = [NoteExample(text="Notiz A", expected_codes=["2080"])]

    result = restrict_to_valid_codes(examples, valid_codes={"0090"})

    assert result == []


def _make_examples(n: int, source: str = "generated") -> list[NoteExample]:
    return [
        NoteExample(text=f"Notiz {i}", expected_codes=["0090"], source=source)
        for i in range(n)
    ]


def test_split_dataset_is_deterministic_for_same_seed():
    examples = _make_examples(20)
    train_a, test_a = split_dataset(examples, test_fraction=0.2, seed=42)
    train_b, test_b = split_dataset(examples, test_fraction=0.2, seed=42)
    assert [e.text for e in train_a] == [e.text for e in train_b]
    assert [e.text for e in test_a] == [e.text for e in test_b]


def test_split_dataset_respects_fraction():
    examples = _make_examples(20)
    train, test = split_dataset(examples, test_fraction=0.2, seed=42)
    assert len(test) == 4
    assert len(train) == 16


def test_split_dataset_always_puts_golden_synth_in_test():
    examples = _make_examples(16, source="generated") + _make_examples(4, source="golden_synth")
    train, test = split_dataset(examples, test_fraction=0.2, seed=42)
    golden_texts = {e.text for e in examples if e.source == "golden_synth"}
    test_golden_texts = {e.text for e in test if e.source == "golden_synth"}
    assert test_golden_texts == golden_texts
    assert all(e.split == "test" for e in test)
    assert all(e.split == "train" for e in train)
