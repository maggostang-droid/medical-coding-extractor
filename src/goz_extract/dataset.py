"""Lädt die Golden-Synth-Fixtures, kombiniert sie mit den generierten
Notizen und teilt alles reproduzierbar in Train/Test. Die Golden-Synth-
Fixtures landen immer im Test-Set, weil sie die höchste Qualität haben
(siehe Design-Spec).

Das Glob-Pattern "*synth*.json" ist bewusst gewählt (nicht "*.json"): im
MAIKA-Fixture-Ordner liegen auch real_*.json-Dateien mit echten
Praxisfällen, die laut Design-Spec (Abschnitt "Datenherkunft &
IP-Abgrenzung") NIEMALS geladen werden dürfen - das Pattern verhindert das
schon auf Dateisystem-Ebene, statt sich auf nachträgliches Filtern zu
verlassen."""
import json
import random
from pathlib import Path

from goz_extract.schema import NoteExample


def load_golden_synth_fixtures(
    fixture_dir: Path, valid_codes: set[str] | None = None
) -> list[NoteExample]:
    """Lädt die Golden-Synth-Fixtures.

    Wenn `valid_codes` übergeben wird, werden Fixtures übersprungen, deren
    `expected_codes` nicht vollständig im kuratierten 55-Code-Label-Space
    liegen (z.B. GOÄ-Codes wie "Ä6"/"Ä5000" oder Codes außerhalb der
    Kategorien A__/C__). Sowohl RAG-Baseline als auch Finetune können
    ohnehin nur Codes aus diesem Space vorhersagen - unbeantwortbare
    Fixtures würden jede Eval-Metrik künstlich nach unten ziehen.
    """
    examples = []
    for path in sorted(fixture_dir.glob("*synth*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        expected_codes = raw["expected_codes"]
        if valid_codes is not None and not set(expected_codes).issubset(valid_codes):
            continue
        examples.append(
            NoteExample(
                text=raw["input"],
                expected_codes=expected_codes,
                source="golden_synth",
            )
        )
    return examples


def split_dataset(
    examples: list[NoteExample], test_fraction: float, seed: int
) -> tuple[list[NoteExample], list[NoteExample]]:
    golden = [e for e in examples if e.source == "golden_synth"]
    rest = [e for e in examples if e.source != "golden_synth"]

    rng = random.Random(seed)
    shuffled = rest.copy()
    rng.shuffle(shuffled)

    n_test_from_rest = max(0, round(len(examples) * test_fraction) - len(golden))
    test_rest, train_rest = shuffled[:n_test_from_rest], shuffled[n_test_from_rest:]

    for example in train_rest:
        example.split = "train"
    for example in test_rest + golden:
        example.split = "test"

    return train_rest, test_rest + golden
