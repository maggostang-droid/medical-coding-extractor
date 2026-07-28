"""Gemeinsames Prompt-Format für RAG-Baseline und Finetune, damit beide
Ansätze identisch ausgewertet werden können. candidates=None -> Finetune-
Modus (Wissen steckt in den LoRA-Gewichten, kein Retrieval-Kontext).
candidates=[...] -> RAG-Modus (Retrieval-Kandidaten als Prompt-Kontext)."""
import re

from goz_extract.schema import GozCode

_INSTRUCTION = (
    "Extrahiere alle zutreffenden GOZ-Ziffern aus der folgenden zahnärztlichen "
    "Behandlungsnotiz. Antworte ausschließlich mit einer kommagetrennten Liste "
    "der Ziffern, ohne weiteren Text, ohne Erklärung, ohne Wiederholung der "
    "Notiz. Nenne nur Ziffern, die auf die Notiz zutreffen - nicht jede "
    "Ziffer, die dir zur Auswahl vorliegt."
)

_CODE_PATTERN = re.compile(r"\bÄ?\d{3,4}\b")


def build_extraction_prompt(note_text: str, candidates: list[GozCode] | None = None) -> str:
    if candidates is None:
        return f"{_INSTRUCTION}\n\nNotiz:\n{note_text}"

    candidate_list = "\n".join(f"- {c.goz_nr}: {c.bezeichnung}" for c in candidates)
    return (
        f"{_INSTRUCTION}\n\n"
        f"Wähle ausschließlich aus den folgenden Kandidaten-Codes:\n{candidate_list}\n\n"
        f"Notiz:\n{note_text}"
    )


def parse_code_list_response(raw_text: str, valid_codes: set[str]) -> list[str]:
    found = _CODE_PATTERN.findall(raw_text)
    result: list[str] = []
    for code in found:
        if code in valid_codes and code not in result:
            result.append(code)
    return result
