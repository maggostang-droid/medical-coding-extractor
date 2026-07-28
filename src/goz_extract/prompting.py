"""Gemeinsames Prompt-Format für RAG-Baseline und Finetune, damit beide
Ansätze identisch ausgewertet werden können. candidates=None -> Finetune-
Modus (Wissen steckt in den LoRA-Gewichten, kein Retrieval-Kontext).
candidates=[...] -> RAG-Modus (Retrieval-Kandidaten als Prompt-Kontext).

Antwortformat ist ein JSON-Array von Ziffern-Strings (z.B. `["2040", "0090"]`)
statt einer kommagetrennten Prosa-Liste. Lokale Modelle wie Llama-3.2-3B
haben (anders als z.B. OpenAIs `response_format`) keine echte
Grammar-Constrained-Decoding-Garantie fürs Einhalten eines Schemas - daher
parst `parse_code_list_response` das JSON-Array, fällt aber auf die alte
Prosa-Regex zurück, wenn das Modell trotz Instruktion kein sauberes JSON
liefert. Der eigentliche Robustheitsgewinn ggü. der alten Prosa-Regex: ein
Modell, das brav JSON produziert, kann nicht mehr "nebenbei" Kandidaten-
Ziffern in einer Begründung erwähnen und damit versehentlich als Vorhersage
gezählt werden, weil nur das Array ausgewertet wird, nicht der ganze Text."""
import json
import re

from goz_extract.schema import GozCode

_INSTRUCTION = (
    "Extrahiere alle zutreffenden GOZ-Ziffern aus der folgenden zahnärztlichen "
    "Behandlungsnotiz. Antworte AUSSCHLIESSLICH mit einem JSON-Array von "
    'Ziffern-Strings, z.B. ["1234", "5678"] - keine Erklärung, kein weiterer '
    "Text davor oder danach, keine Wiederholung der Notiz. Nenne nur Ziffern, "
    "die auf die Notiz zutreffen - nicht jede Ziffer, die dir zur Auswahl "
    "vorliegt. Erwähne keine anderen Ziffern aus der Kandidatenliste, auch "
    "nicht um zu begründen, warum sie nicht zutreffen."
)

_CODE_PATTERN = re.compile(r"\bÄ?\d{3,4}\b")
_JSON_ARRAY_PATTERN = re.compile(r"\[.*?\]", re.DOTALL)


def build_extraction_prompt(note_text: str, candidates: list[GozCode] | None = None) -> str:
    if candidates is None:
        return f"{_INSTRUCTION}\n\nNotiz:\n{note_text}"

    candidate_list = "\n".join(f"- {c.format_for_prompt()}" for c in candidates)
    return (
        f"{_INSTRUCTION}\n\n"
        f"Wähle ausschließlich aus den folgenden Kandidaten-Codes:\n{candidate_list}\n\n"
        f"Notiz:\n{note_text}"
    )


def parse_code_list_response(raw_text: str, valid_codes: set[str]) -> list[str]:
    codes = _parse_json_array(raw_text)
    if codes is None:
        # Modell hat kein valides JSON-Array geliefert (trotz Instruktion) -
        # Rückfall auf die alte, tolerante Prosa-Regex statt eine leere
        # Vorhersage zurückzugeben.
        codes = _CODE_PATTERN.findall(raw_text)

    result: list[str] = []
    for code in codes:
        if code in valid_codes and code not in result:
            result.append(code)
    return result


def restrict_to_candidates(codes: list[str], candidates: list[GozCode]) -> list[str]:
    """Validierungs-Layer fürs RAG-Modus (angelehnt an MAIKAs Post-
    Processing-Prinzip: roher LLM-Output ist nicht automatisch das
    Endergebnis): das Modell darf nur aus den angebotenen Kandidaten wählen
    - alles andere ist per Definition eine Fehlvorhersage (Halluzination
    oder ein Code, der gar nicht im Prompt stand) und wird rausgefiltert."""
    candidate_nrs = {c.goz_nr for c in candidates}
    return [code for code in codes if code in candidate_nrs]


def _parse_json_array(raw_text: str) -> list[str] | None:
    match = _JSON_ARRAY_PATTERN.search(raw_text)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        return None
    return parsed
