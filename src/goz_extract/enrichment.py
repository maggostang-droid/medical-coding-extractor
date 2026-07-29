"""Generiert verständlichere Klartext-Beschreibungen für kuratierte GOZ-Codes
über eine Chat-LLM-API - ergänzt die knappe amtliche Formulierung um eine
Erklärung plus gängige umgangssprachliche Begriffe/Abkürzungen, damit sowohl
die RAG-Kandidatenliste im Prompt als auch die Datengenerierung besser mit
dem jeweiligen Code arbeiten können.

Bewusst nur aus allgemeinem GOZ-Fachwissen generiert, kein Bezug zu
proprietären, urheberrechtlich geschützten Kommentarwerken oder fremder
Geschäftslogik - siehe Design-Spec, Abschnitt "Datenherkunft &
IP-Abgrenzung". Prompt-Bau und Antwort-Parsing
sind bewusst von der tatsächlichen API-Anbindung getrennt, damit beides
ohne Netzwerkzugriff testbar ist (gleiches Muster wie data_generation.py)."""
import json
import re
from typing import Protocol

from goz_extract.schema import GozCode


def build_enrichment_prompt(codes: list[GozCode]) -> str:
    code_list = "\n".join(f"- {c.goz_nr}: {c.bezeichnung}" for c in codes)
    return f"""Du bist Experte für die deutsche Gebührenordnung für Zahnärzte (GOZ).

Für jede der folgenden GOZ-Ziffern (mit amtlicher Bezeichnung):
{code_list}

Schreibe eine kurze, verständliche Klartext-Erklärung (1-2 Sätze), die
erklärt, wofür die Ziffer klinisch steht, plus gängige umgangssprachliche
Begriffe oder Abkürzungen, die Zahnärzte in Behandlungsnotizen dafür
benutzen könnten. Nutze ausschließlich dein allgemeines Fachwissen zur
GOZ, keine Erfindungen.

Antworte ausschließlich mit einem JSON-Objekt, Schlüssel = GOZ-Ziffer,
Wert = die Erklärung als String: {{"<goz_nr>": "<Erklärung>", ...}}
"""


def parse_enrichment_response(raw_text: str) -> dict[str, str]:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"Kein JSON-Objekt in der Antwort gefunden: {raw_text!r}")

    payload = json.loads(match.group(0))
    if not payload:
        raise ValueError("Generiertes Objekt ist leer")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in payload.items()):
        raise ValueError("Erwartet ein JSON-Objekt aus String-zu-String-Paaren")
    return payload


class _InvocableChatModel(Protocol):
    def invoke(self, prompt: str): ...


def generate_enriched_descriptions(
    chat_model: _InvocableChatModel, codes: list[GozCode]
) -> dict[str, str]:
    prompt = build_enrichment_prompt(codes)
    response = chat_model.invoke(prompt)
    return parse_enrichment_response(response.content)
