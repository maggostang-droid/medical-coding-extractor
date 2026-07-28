"""Einmaliges Skript: liest die volle GOZ-Datenbank aus dem lokalen
MAIKA-Checkout und schreibt die kuratierte, öffentliche Teilmenge nach
data/goz_codes.json. Pfad zum MAIKA-Checkout wird per Argument übergeben,
damit kein persönlicher absoluter Pfad im Repo landet.

Aufruf (einmalig, lokal, nicht Teil der automatisierten Tests):
    .venv/Scripts/python.exe scripts/curate_codes.py \
        --source "C:/Users/Marco/Downloads/dentist-main/dentist-main/data/databases/goz_database_v4.json" \
        --out data/goz_codes.json
"""
import argparse
import json
from pathlib import Path

from goz_extract.curate import curate_codes

ALLOWED_PREFIXES = ("A__", "C__")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    raw_entries = json.loads(args.source.read_text(encoding="utf-8"))
    codes = curate_codes(raw_entries, allowed_prefixes=ALLOWED_PREFIXES)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps([c.model_dump() for c in codes], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"{len(codes)} Codes kuratiert -> {args.out}")


if __name__ == "__main__":
    main()
