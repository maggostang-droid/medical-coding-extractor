"""Integrationstest gegen die echten, ausgecheckten Datendateien: stellt
sicher, dass jede erwartete Code-Liste im Test-Split eine Teilmenge des
kuratierten 55-Code-Label-Space ist. Genau diese Lücke (Golden-Synth-
Fixtures mit Codes außerhalb des Label-Space, z.B. GOÄ-Codes) hat jede
Eval-Metrik künstlich gedrückt, ohne dass ein bestehender Test das
gemerkt hätte - siehe finale Review, Punkt 1."""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_all_test_split_codes_are_within_valid_code_space():
    codes = json.loads((DATA_DIR / "goz_codes.json").read_text(encoding="utf-8"))
    valid_codes = {c["goz_nr"] for c in codes}

    rows = [
        json.loads(line)
        for line in (DATA_DIR / "test.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows, "data/test.jsonl ist leer - Fixture nicht generiert?"

    offenders = [
        (row["text"], row["expected_codes"])
        for row in rows
        if not set(row["expected_codes"]).issubset(valid_codes)
    ]
    assert not offenders, (
        f"{len(offenders)} Testbeispiele enthalten Codes außerhalb des "
        f"kuratierten Label-Space: {offenders}"
    )
