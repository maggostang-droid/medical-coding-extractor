"""Fan-in-Logik des GOZ-Graphen: zwei Extraktor-Pfade (RAG, LoRA) werden
zusammengeführt und anschließend deterministisch verifiziert.

Bewusst OHNE LangGraph-Abhängigkeit: Offline-Eval (scripts/run_graph_eval.py)
und eine spätere LangGraph-Runtime sollen dieselbe Logik benutzen, statt sie
zu duplizieren. Der Graph verdrahtet nur, entschieden wird hier.

Warum überhaupt ein Graph: Die gemessenen Fehlerprofile sind komplementär —
RAG hat den höheren Recall (0.70) bei niedriger Precision (0.40), der
LoRA-Finetune umgekehrt (P 0.65 / R 0.58). Ein Merge kann deshalb mehr sein
als der bessere der beiden Einzelpfade.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

#: Paare von GOZ-Ziffern, die nicht gemeinsam abgerechnet werden dürfen.
#: BEWUSST LEER — Ausschlüsse sind eine fachliche Aussage und werden nicht
#: geraten. Struktur steht, Befüllung erfolgt gegen die GOZ-Quelle.
MUTUALLY_EXCLUSIVE: set[frozenset[str]] = set()

#: Wie mit Codes umgegangen wird, die nur der RAG-Pfad geliefert hat.
RAG_ONLY_UNCERTAIN = "uncertain"
RAG_ONLY_ACCEPT = "accept"


@dataclass
class MergeResult:
    """Ergebnis eines Graph-Durchlaufs für genau eine Notiz."""

    accepted: list[str] = field(default_factory=list)
    uncertain: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    sources: dict[str, list[str]] = field(default_factory=dict)
    conflicts: list[tuple[str, str]] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        """True, wenn ein Mensch draufschauen muss.

        Drei Auslöser: unsichere Codes, vom Verifier abgelehnte Codes,
        oder ein Ausschluss-Konflikt.
        """
        return bool(self.uncertain or self.rejected or self.conflicts)

    @property
    def predicted_codes(self) -> list[str]:
        """Was der Graph als Vorhersage ausgibt — nur akzeptierte Codes.

        Unsichere Codes zählen NICHT als Vorhersage. Das hebt die Precision
        mechanisch an und verschiebt Aufwand zum Menschen; die
        needs_review-Quote gehört deshalb immer neben die Metriken.
        """
        return sorted(self.accepted)


def load_valid_codes(codes_path: Path) -> set[str]:
    """Lädt die gültigen GOZ-Ziffern aus data/goz_codes.json."""
    codes = json.loads(Path(codes_path).read_text(encoding="utf-8"))
    return {entry["goz_nr"] for entry in codes}


def merge_candidates(
    rag_codes: list[str],
    lora_codes: list[str],
    rag_only_policy: str = RAG_ONLY_UNCERTAIN,
) -> MergeResult:
    """Führt die Kandidaten beider Pfade zusammen.

    Regeln (aus den gemessenen Fehlerprofilen abgeleitet):
    - von beiden geliefert  -> accepted (die Quellen sind sich einig)
    - nur LoRA              -> accepted (präziseste Einzelquelle, P=0.65)
    - nur RAG               -> uncertain (P=0.40 allein zu schwach)

    Mit ``rag_only_policy=RAG_ONLY_ACCEPT`` wird die dritte Regel gelockert —
    gedacht als Hebel, falls die Review-Quote zu hoch ausfällt.
    """
    if rag_only_policy not in (RAG_ONLY_UNCERTAIN, RAG_ONLY_ACCEPT):
        raise ValueError(f"Unbekannte rag_only_policy: {rag_only_policy}")

    rag_set, lora_set = set(rag_codes), set(lora_codes)
    both = rag_set & lora_set
    lora_only = lora_set - rag_set
    rag_only = rag_set - lora_set

    accepted = both | lora_only
    uncertain: set[str] = set()
    if rag_only_policy == RAG_ONLY_ACCEPT:
        accepted |= rag_only
    else:
        uncertain = rag_only

    return MergeResult(
        accepted=sorted(accepted),
        uncertain=sorted(uncertain),
        sources={
            "beide": sorted(both),
            "nur_lora": sorted(lora_only),
            "nur_rag": sorted(rag_only),
        },
    )


def verify(
    result: MergeResult,
    valid_codes: set[str],
    mutually_exclusive: set[frozenset[str]] | None = None,
) -> MergeResult:
    """Deterministischer Verifier — kein LLM, reiner Code.

    Zwei Prüfungen:
    1. Katalog-Check: unbekannte Ziffern fliegen raus (halluzinierte Codes).
    2. Ausschluss-Check: unvereinbare Paare werden nicht aufgelöst, sondern
       eskaliert — beide Codes wandern nach ``uncertain``, der Konflikt wird
       protokolliert. Bei Abrechnungsziffern ist Raten die teuerste Option.
    """
    exclusions = MUTUALLY_EXCLUSIVE if mutually_exclusive is None else mutually_exclusive

    accepted = [c for c in result.accepted if c in valid_codes]
    rejected = sorted(set(result.accepted) - set(accepted))
    uncertain = set(result.uncertain)

    conflicts: list[tuple[str, str]] = []
    for pair in exclusions:
        a, b = sorted(pair)
        if a in accepted and b in accepted:
            conflicts.append((a, b))
            uncertain.update({a, b})
    if conflicts:
        conflicted = {code for pair in conflicts for code in pair}
        accepted = [c for c in accepted if c not in conflicted]

    return MergeResult(
        accepted=sorted(accepted),
        uncertain=sorted(uncertain),
        rejected=rejected,
        sources=result.sources,
        conflicts=conflicts,
    )


def graph_predict(
    rag_codes: list[str],
    lora_codes: list[str],
    valid_codes: set[str],
    rag_only_policy: str = RAG_ONLY_UNCERTAIN,
    mutually_exclusive: set[frozenset[str]] | None = None,
) -> MergeResult:
    """Ein kompletter Graph-Durchlauf: Fan-in -> Verifier."""
    merged = merge_candidates(rag_codes, lora_codes, rag_only_policy=rag_only_policy)
    return verify(merged, valid_codes, mutually_exclusive=mutually_exclusive)
