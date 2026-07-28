"""Erzeugt synthetische Behandlungsnotizen samt erwarteten GOZ-Codes über
eine Chat-LLM-API. Prompt-Bau und Antwort-Parsing sind bewusst von der
tatsächlichen API-Anbindung getrennt, damit beides ohne Netzwerkzugriff
testbar ist (siehe generate_examples: nimmt ein beliebiges Objekt mit
.invoke(prompt) -> Objekt mit .content entgegen, kompatibel zu
LangChains BaseChatModel)."""
import json
import re
from typing import Protocol

from goz_extract.schema import GozCode, NoteExample

_DIFFICULTY_LABELS = {
    "easy": "leicht (nah an der amtlichen Bezeichnung formuliert)",
    "medium": "mittel (umgangssprachliche Zahnarzt-Notiz, andere Wortwahl)",
    "hard": "schwer (Abkürzungen, implizite Formulierung, wenig Übereinstimmung mit der amtlichen Bezeichnung)",
}


def build_generation_prompt(codes: list[GozCode], difficulty: str, n_examples: int) -> str:
    code_list = "\n".join(f"- {c.format_for_prompt()}" for c in codes)
    label = _DIFFICULTY_LABELS.get(difficulty, difficulty)
    return f"""Du erstellst synthetische Trainingsdaten für ein GOZ-Code-Extraktionsmodell.

Verfügbare GOZ-Codes (nur diese dürfen als expected_codes vorkommen):
{code_list}

Erzeuge {n_examples} realistische, kurze zahnärztliche Behandlungsnotizen
(Schwierigkeitsgrad: {label}). Jede Notiz kombiniert 1-3 der obigen Codes
plausibel (z.B. Anästhesie + Füllung, nicht rein zufällig).

Verwende für jeden in einer Notiz erwähnten Code mindestens einen Begriff
aus dessen Bezeichnung oder den "Umgangssprachlich"-Begriffen oben (z.B.
"Kofferdam" statt "Spanngummi", "WK-Aufbereitung" statt "Wurzelkanal-
aufbereitung") - das sind reale Fachbegriffe/Abkürzungen, keine künstliche
Vereinfachung, und helfen einem Retrieval-System, die richtigen Codes zu
finden. Gilt für alle Schwierigkeitsgrade, auch "schwer" - dort trotzdem
mindestens einen Fachbegriff oder eine gängige Abkürzung pro Code nutzen,
den Rest der Notiz aber weiterhin implizit/abgekürzt formulieren.

Antworte ausschließlich mit einem JSON-Array, jedes Element:
{{"text": "<Notiz>", "expected_codes": ["<goz_nr>", ...]}}
"""


def parse_generation_response(raw_text: str) -> list[NoteExample]:
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"Keine JSON-Liste in der Antwort gefunden: {raw_text!r}")

    payload = json.loads(match.group(0))
    if not payload:
        raise ValueError("Generierte Liste ist leer")

    return [NoteExample(text=item["text"], expected_codes=item["expected_codes"]) for item in payload]


class _InvocableChatModel(Protocol):
    def invoke(self, prompt: str): ...


def generate_examples(
    chat_model: _InvocableChatModel, codes: list[GozCode], difficulty: str, n_examples: int
) -> list[NoteExample]:
    prompt = build_generation_prompt(codes, difficulty, n_examples)
    response = chat_model.invoke(prompt)
    examples = parse_generation_response(response.content)
    for example in examples:
        example.difficulty = difficulty
    return examples
