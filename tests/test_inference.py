"""generate_codes braucht ein reales (~6GB) Sprachmodell und wird deshalb
standardmäßig übersprungen - siehe Task 9 im Implementierungsplan für die
manuelle Verifikation auf Colab. Der Test dokumentiert die erwartete
Schnittstelle und läuft, wenn RUN_MODEL_TESTS=1 gesetzt ist."""
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="Braucht ein reales, ~6GB großes Sprachmodell - siehe Task 9 im Implementierungsplan",
)


def test_generate_codes_returns_valid_codes_only():
    from goz_extract.inference import generate_codes, load_model

    model, tokenizer = load_model("meta-llama/Llama-3.2-3B-Instruct")
    result = generate_codes(
        model,
        tokenizer,
        note_text="Zahn 36: Infiltrationsanästhesie, Kompositfüllung zweiflächig.",
        valid_codes={"0090", "2080", "0010"},
    )
    assert all(code in {"0090", "2080", "0010"} for code in result)
