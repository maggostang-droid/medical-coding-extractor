"""Generiert erweiterte Klartext-Beschreibungen für die kuratierten GOZ-Codes
über die echte Anthropic-API und ergänzt sie in-place in der Codeliste.
Braucht ANTHROPIC_API_KEY in .env. Kostet einen echten API-Call — nicht Teil
der automatisierten Tests.

Aufruf:
    .venv-data/Scripts/python.exe scripts/enrich_codes.py --codes data/goz_codes.json
"""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from goz_extract.enrichment import generate_enriched_descriptions
from goz_extract.schema import GozCode


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", required=True, type=Path)
    args = parser.parse_args()

    raw = json.loads(args.codes.read_text(encoding="utf-8"))
    codes = [GozCode(**c) for c in raw]

    chat_model = init_chat_model(model="claude-sonnet-4-5", model_provider="anthropic", temperature=0.3)
    descriptions = generate_enriched_descriptions(chat_model, codes)

    for entry in raw:
        nr = entry["goz_nr"]
        if nr in descriptions:
            entry["erweiterte_beschreibung"] = descriptions[nr]
        else:
            print(f"Warnung: keine Beschreibung für {nr} erhalten")

    args.codes.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{len(descriptions)} Beschreibungen ergänzt -> {args.codes}")


if __name__ == "__main__":
    main()
