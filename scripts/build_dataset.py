"""Kombiniert die generierten synthetischen Notizen mit den Golden-Synth-
Fixtures, splittet und schreibt train/test als JSONL.

Aufruf:
    .venv/Scripts/python.exe scripts/build_dataset.py \
        --generated data/synthetic_notes.jsonl \
        --golden-synth-dir "C:/Users/Marco/Downloads/dentist-main/dentist-main/tests/fixtures/golden_single_v2" \
        --out-dir data/
"""
import argparse
import json
from pathlib import Path

from goz_extract.dataset import load_golden_synth_fixtures, split_dataset
from goz_extract.schema import NoteExample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", required=True, type=Path)
    parser.add_argument("--golden-synth-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generated = [
        NoteExample.model_validate_json(line)
        for line in args.generated.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # load_golden_synth_fixtures filtert real_-Fixtures bereits beim Einlesen
    # aus (Glob-Pattern "*synth*.json") - siehe dataset.py.
    golden_synth = load_golden_synth_fixtures(args.golden_synth_dir)

    train, test = split_dataset(generated + golden_synth, args.test_fraction, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_examples in [("train.jsonl", train), ("test.jsonl", test)]:
        out_path = args.out_dir / name
        with out_path.open("w", encoding="utf-8") as f:
            for example in split_examples:
                f.write(example.model_dump_json() + "\n")
        print(f"{len(split_examples)} Notizen -> {out_path}")


if __name__ == "__main__":
    main()
