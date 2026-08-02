"""Checker-Node: der Graph-Schritt, der aus den gepoolten Kandidaten beider
Extraktoren die richtige Teilmenge auswählt.

Warum dieser Knoten und nicht der Aggregator aus `graph_merge`: Die Messung
über die 81 Testnotizen (`scripts/analyze_merge_headroom.py`) zeigt zwei weit
auseinanderliegende Obergrenzen.

* Wer pro Notiz perfekt zwischen den vier fertigen Mengen wählt (RAG, LoRA,
  Vereinigung, Schnittmenge), kommt auf Exact Match **0.42** — Mengen-Algebra
  ist damit ausgereizt.
* Die erwarteten Codes stecken aber in **65 von 81 Notizen (0.80)** vollständig
  in der Vereinigung beider Pfade.

Der Hebel liegt also im Auswählen, nicht im Verrechnen. Genau das macht dieser
Knoten: Er behandelt die beiden Extraktoren als *Retriever* und stellt die
eigentliche Auswahl als eigene Aufgabe.

Damit ist der Knoten strukturell dieselbe Aufgabe wie die RAG-Baseline — nur
mit einem anderen Kandidatenpool. Die RAG-Baseline wählt aus BM25-/Embedding-
Treffern (Precision des Pools entsprechend niedrig), der Checker aus dem, was
zwei Extraktoren tatsächlich vorgeschlagen haben. Deshalb benutzt er
absichtlich denselben `build_extraction_prompt`-Pfad: Der Unterschied soll im
Kandidatenpool liegen, nicht in der Prompt-Formulierung.

Das Modul bleibt frei von Torch/HTTP-Abhängigkeiten. Wie ausgewählt wird,
steckt in `select_fn` — auf Colab das Basismodell über
`goz_extract.inference.generate_codes`, in der Offline-Sonde ein API-Modell,
im Test eine Funktion ohne Modell. Nur so lässt sich der Knoten überhaupt
ohne GPU testen.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from goz_extract.graph_merge import MergeResult, verify
from goz_extract.prompting import build_extraction_prompt, restrict_to_candidates
from goz_extract.schema import GozCode

#: Signatur der Auswahlfunktion: bekommt Notiz und Kandidaten, liefert die
#: ausgewählten Ziffern. Absichtlich schmal gehalten, damit Colab-Modell,
#: API-Modell und Test-Dummy dieselbe Schnittstelle erfüllen.
SelectFn = Callable[[str, list[GozCode]], list[str]]


def pool_candidates(rag_codes: Sequence[str], lora_codes: Sequence[str]) -> list[str]:
    """Vereinigung beider Pfade — der Kandidatenpool des Checkers.

    Bewusst die Vereinigung und nicht etwa nur die Schnittmenge: Die
    Obergrenzen-Rechnung bezieht sich auf genau diesen Pool (0.80). Jede
    Vorfilterung würde die erreichbare Decke senken, bevor der Checker
    überhaupt zum Zug kommt.
    """
    return sorted(set(rag_codes) | set(lora_codes))


def build_checker_prompt(note_text: str, pooled: Sequence[str], code_by_nr: dict[str, GozCode]) -> str:
    """Prompt für die Auswahlaufgabe — reiner Code, ohne Modell, testbar."""
    candidates = [code_by_nr[nr] for nr in pooled if nr in code_by_nr]
    return build_extraction_prompt(note_text, candidates=candidates)


def checker_predict(
    note_text: str,
    rag_codes: Sequence[str],
    lora_codes: Sequence[str],
    code_by_nr: dict[str, GozCode],
    select_fn: SelectFn,
    mutually_exclusive: set[frozenset[str]] | None = None,
) -> MergeResult:
    """Ein Graph-Durchlauf mit Checker: Fan-in -> Auswahl -> Verifier.

    Unterschied zum Aggregator in `graph_merge`: Verworfene Kandidaten lösen
    hier **kein** `needs_review` aus. Das Aussortieren ist die Aufgabe des
    Knotens, nicht ein Fall für einen Menschen — genau daran scheiterte die
    Merge-Regel mit ihrer Prüfquote von 95 %.

    Eskaliert wird nur noch, wo der Knoten selbst zweifelhaft ist: wenn der
    Verifier etwas ablehnt, bei einem Ausschlusskonflikt, oder wenn aus einem
    nicht-leeren Kandidatenpool nichts ausgewählt wurde (dann hat entweder die
    Notiz keine abrechenbare Leistung oder der Checker hat versagt — das
    auseinanderzuhalten ist nicht seine Aufgabe).
    """
    pooled = pool_candidates(rag_codes, lora_codes)
    candidates = [code_by_nr[nr] for nr in pooled if nr in code_by_nr]

    selected = select_fn(note_text, candidates) if candidates else []
    # Auch wenn der Prompt die Auswahl vorgibt: roher Modelloutput ist nie
    # automatisch das Ergebnis (dasselbe Prinzip wie im RAG-Pfad).
    selected = restrict_to_candidates(list(dict.fromkeys(selected)), candidates)

    dropped = [nr for nr in pooled if nr not in selected]
    merged = MergeResult(
        accepted=sorted(selected),
        uncertain=[],
        sources={
            "gepoolt": pooled,
            "ausgewaehlt": sorted(selected),
            "verworfen": dropped,
            "nur_rag": sorted(set(rag_codes) - set(lora_codes)),
            "nur_lora": sorted(set(lora_codes) - set(rag_codes)),
        },
    )

    result = verify(merged, set(code_by_nr), mutually_exclusive=mutually_exclusive)

    if pooled and not result.accepted and not result.rejected and not result.conflicts:
        # Leere Auswahl aus nicht-leerem Pool: einziger Fall, in dem der
        # Checker selbst eine Prüfung auslöst.
        result.uncertain = list(pooled)
    return result
