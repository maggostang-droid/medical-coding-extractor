"""BM25- und Embedding-basiertes Retrieval über die kuratierte GOZ-Codeliste,
kombiniert per Reciprocal Rank Fusion (RRF) — die RAG-Baseline für den
Vergleich gegen das LoRA-Finetune. Bewusst mit injizierbarer encode_fn
gebaut, damit die Kern-Logik ohne ein echtes Embedding-Modell testbar ist."""
import re
from typing import Callable

import numpy as np
from rank_bm25 import BM25Okapi

from goz_extract.schema import GozCode


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def segment_note(note_text: str) -> list[str]:
    """Splittet eine Notiz grob in einzelne Behandlungsschritte (Trenner:
    Satzzeichen). Ein Produktivsystem würde das über einen dedizierten
    LLM-Call lösen; hier bewusst eine simple Regel statt eines
    dritten Modell-Calls. Zweck: Retrieval pro Schritt statt über die ganze
    Notiz auf einmal, weil ein kurz erwähnter, seltener Schritt (z.B. eine
    Anästhesie neben einer Füllung) sonst im BM25-/Embedding-Vektor der
    gesamten Notiz untergeht."""
    segments = re.split(r"[.,;:]", note_text)
    return [s.strip() for s in segments if s.strip()]


class BM25Index:
    def __init__(self, codes: list[GozCode]) -> None:
        self._codes = codes
        # format_for_prompt() statt nur bezeichnung: hängt die
        # umgangssprachlichen Begriffe aus erweiterte_beschreibung an, falls
        # vorhanden (z.B. "Kofferdam" für 2040) - genau die Begriffe, die in
        # echten Notizen auftauchen, nicht die amtliche Bezeichnung. Ohne
        # das kann BM25 nur exakte Amtsbegriffe matchen.
        self._corpus = [tokenize(c.format_for_prompt()) for c in codes]
        self._bm25 = BM25Okapi(self._corpus)

    def rank(self, query: str) -> list[str]:
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1]
        return [self._codes[i].goz_nr for i in order]


class EmbeddingIndex:
    """encode_fn und encode_query_fn getrennt, weil asymmetrische Encoder wie
    intfloat/multilingual-e5-base unterschiedliche Präfixe für Korpus-Texte
    ("passage: ") und Suchanfragen ("query: ") erwarten - beide mit derselben
    Funktion zu encodieren verletzt diese trainierte Konvention und verschlechtert
    die Retrieval-Qualität. encode_query_fn fällt auf encode_fn zurück für
    symmetrische Encoder (z.B. den Fake-Encoder in Tests)."""

    def __init__(
        self,
        codes: list[GozCode],
        encode_fn: Callable[[list[str]], np.ndarray],
        encode_query_fn: Callable[[list[str]], np.ndarray] | None = None,
    ) -> None:
        self._codes = codes
        self._encode_query_fn = encode_query_fn or encode_fn
        # format_for_prompt() statt nur bezeichnung - siehe Kommentar in BM25Index.
        embeddings = encode_fn([c.format_for_prompt() for c in codes])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._embeddings = embeddings / norms

    def rank(self, query: str) -> list[str]:
        query_vec = self._encode_query_fn([query])[0]
        norm = np.linalg.norm(query_vec) or 1.0
        query_vec = query_vec / norm
        scores = self._embeddings @ query_vec
        order = np.argsort(scores)[::-1]
        return [self._codes[i].goz_nr for i in order]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, goz_nr in enumerate(ranking):
            scores[goz_nr] = scores.get(goz_nr, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda goz_nr: scores[goz_nr], reverse=True)


def retrieve_candidates(
    note_text: str, bm25_index: BM25Index, embedding_index: EmbeddingIndex, top_n: int
) -> list[str]:
    segments = segment_note(note_text) or [note_text]
    rankings = []
    for segment in segments:
        rankings.append(bm25_index.rank(segment))
        rankings.append(embedding_index.rank(segment))
    fused = reciprocal_rank_fusion(rankings)
    return fused[:top_n]
