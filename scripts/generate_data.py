"""Erzeugt die synthetischen Trainingsnotizen über die echte Anthropic-API
und schreibt sie als JSONL. Braucht ANTHROPIC_API_KEY in .env (siehe
.env.example). Kostet echte API-Calls — nicht Teil der automatisierten Tests.

Aufruf:
    .venv/Scripts/python.exe scripts/generate_data.py \
        --codes data/goz_codes.json --out data/synthetic_notes.jsonl \
        --per-code 6
"""
import argparse
import json
import random
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from goz_extract.data_generation import generate_examples
from goz_extract.schema import GozCode

DIFFICULTIES = ["easy", "medium", "hard"]
CODES_PER_BATCH = 4
EXAMPLES_PER_BATCH = 5


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--per-code", type=int, default=6,
        help="Zielanzahl Batches pro Code (jeder Batch sampelt 4 Codes und erzeugt 5 Notizen)",
    )
    args = parser.parse_args()

    codes = [GozCode(**c) for c in json.loads(args.codes.read_text(encoding="utf-8"))]
    chat_model = init_chat_model(model="claude-sonnet-4-5", model_provider="anthropic", temperature=0.9)

    rng = random.Random(42)
    # Jeder Batch deckt CODES_PER_BATCH Codes ab -> um jeden Code im Schnitt
    # ~per_code mal in einem Batch zu haben, braucht es (len(codes)*per_code)/CODES_PER_BATCH
    # Batches. Bei 55 Codes und per_code=6: round(55*6/4) = 82 Batches x 5 Notizen
    # = ~410 Notizen (verifiziert gegen die reale 410-Zeilen-Ausgabe).
    total_batches = max(1, round(len(codes) * args.per_code / CODES_PER_BATCH))

    all_examples = []
    for _ in range(total_batches):
        batch_codes = rng.sample(codes, k=min(CODES_PER_BATCH, len(codes)))
        difficulty = rng.choice(DIFFICULTIES)
        try:
            examples = generate_examples(chat_model, batch_codes, difficulty, EXAMPLES_PER_BATCH)
        except (ValueError, KeyError) as e:
            # ValueError deckt u.a. json.JSONDecodeError und pydantic.ValidationError ab
            # (beides Subklassen). KeyError fängt strukturell unvollständige JSON-Items
            # ab (valides JSON, aber z.B. "text" oder "expected_codes" fehlt im Item) —
            # parse_generation_response greift dort per item["text"] direkt zu.
            print(f"Batch übersprungen (Parsing-Fehler): {e}")
            continue
        all_examples.extend(examples)
        print(f"+{len(examples)} Beispiele ({difficulty}), gesamt: {len(all_examples)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for example in all_examples:
            f.write(example.model_dump_json() + "\n")
    print(f"{len(all_examples)} Notizen geschrieben -> {args.out}")


if __name__ == "__main__":
    main()
