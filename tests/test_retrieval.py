import numpy as np

from goz_extract.retrieval import (
    BM25Index,
    EmbeddingIndex,
    reciprocal_rank_fusion,
    retrieve_candidates,
    tokenize,
)
from goz_extract.schema import GozCode

CODES = [
    GozCode(goz_nr="0090", bezeichnung="Intraorale Infiltrationsanästhesie"),
    GozCode(goz_nr="2080", bezeichnung="Kompositfüllung, zweiflächig, Adhäsivtechnik"),
    GozCode(goz_nr="0010", bezeichnung="Eingehende Untersuchung zur Feststellung von Erkrankungen"),
]


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Kompositfüllung, zweiflächig!") == ["kompositfüllung", "zweiflächig"]


def test_bm25_ranks_exact_term_match_first():
    index = BM25Index(CODES)
    ranking = index.rank("Kompositfüllung zweiflächig")
    assert ranking[0] == "2080"


def _fake_encode_fn(texts: list[str]) -> np.ndarray:
    # Deterministische Fake-Embeddings: Vektor = Zeichenhäufigkeit von 'a','u','n'
    def vec(t: str) -> list[float]:
        t = t.lower()
        return [t.count("a"), t.count("u"), t.count("n")]

    return np.array([vec(t) for t in texts], dtype=float)


def test_embedding_index_ranks_by_cosine_similarity():
    index = EmbeddingIndex(CODES, encode_fn=_fake_encode_fn)
    ranking = index.rank("Untersuchung")
    assert set(ranking) == {"0090", "2080", "0010"}
    assert len(ranking) == 3


def test_embedding_index_uses_separate_query_encoder_when_given():
    calls = []

    def encode_corpus(texts):
        calls.append(("corpus", list(texts)))
        return _fake_encode_fn(texts)

    def encode_query(texts):
        calls.append(("query", list(texts)))
        return _fake_encode_fn(texts)

    index = EmbeddingIndex(CODES, encode_fn=encode_corpus, encode_query_fn=encode_query)
    index.rank("Untersuchung")

    assert calls[0] == ("corpus", [c.bezeichnung for c in CODES])
    assert calls[1] == ("query", ["Untersuchung"])


def test_reciprocal_rank_fusion_prefers_items_ranked_high_in_both():
    ranking_a = ["0090", "2080", "0010"]
    ranking_b = ["2080", "0090", "0010"]
    fused = reciprocal_rank_fusion([ranking_a, ranking_b])
    assert fused[0] in {"0090", "2080"}
    assert fused[-1] == "0010"


def test_retrieve_candidates_combines_bm25_and_embeddings():
    bm25_index = BM25Index(CODES)
    embedding_index = EmbeddingIndex(CODES, encode_fn=_fake_encode_fn)
    candidates = retrieve_candidates(
        "Füllung zweiflächig nach Anästhesie", bm25_index, embedding_index, top_n=2
    )
    assert len(candidates) == 2
    assert "2080" in candidates
