"""Filtert die volle GOZ-Codeliste auf die öffentlichen Felder der
gewünschten Kategorien (siehe Global Constraints im Implementierungsplan:
nur A__ und C__)."""
from goz_extract.schema import GozCode


def curate_codes(raw_entries: list[dict], allowed_prefixes: tuple[str, ...]) -> list[GozCode]:
    result = []
    for entry in raw_entries:
        kategorie = entry["kategorie"]
        if not kategorie.startswith(allowed_prefixes):
            continue
        result.append(GozCode(goz_nr=entry["goz_nr"], bezeichnung=entry["bezeichnung"]))
    return result
