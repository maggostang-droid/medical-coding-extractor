"""Streamlit-Demo: Behandlungsnotiz eingeben, Vorhersage von RAG-Baseline
und LoRA-Finetune nebeneinander vergleichen."""
import json
from pathlib import Path

import streamlit as st
from peft import PeftModel
from sentence_transformers import SentenceTransformer

from goz_extract.inference import generate_codes, load_model
from goz_extract.retrieval import BM25Index, EmbeddingIndex, retrieve_candidates
from goz_extract.schema import GozCode

st.set_page_config(page_title="GOZ-Extraktion: Finetune vs. RAG", layout="wide")

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"


@st.cache_resource
def load_resources():
    codes = [
        GozCode(**c)
        for c in json.loads(Path("data/goz_codes.json").read_text(encoding="utf-8"))
    ]
    code_by_nr = {c.goz_nr: c for c in codes}
    valid_codes = set(code_by_nr)

    embed_model = SentenceTransformer("intfloat/multilingual-e5-base")

    # intfloat/multilingual-e5-base ist asymmetrisch trainiert: Korpus-Texte
    # brauchen "passage: ", Suchanfragen "query: " - dieselbe Präfixierung für
    # beide Seiten zu verwenden verschlechtert die Embedding-Retrieval-Qualität.
    def encode_passages(texts):
        return embed_model.encode([f"passage: {t}" for t in texts])

    def encode_query(texts):
        return embed_model.encode([f"query: {t}" for t in texts])

    bm25_index = BM25Index(codes)
    embedding_index = EmbeddingIndex(codes, encode_fn=encode_passages, encode_query_fn=encode_query)

    # Lädt das Basismodell nur einmal (statt wie zuvor zwei komplette
    # Kopien mit je eigenem load_model()-Aufruf) und schaltet den LoRA-
    # Adapter für die Finetune-Variante per PEFT dazu - halbiert den
    # Speicherbedarf. torch_dtype="auto" statt hartkodiertem bfloat16,
    # weil diese Demo auf einer CPU-Maschine laufen soll.
    base_model, base_tokenizer = load_model(MODEL_ID, torch_dtype="auto")
    finetuned_model = PeftModel.from_pretrained(base_model, "adapters/goz-extract-llama32-3b")
    return code_by_nr, valid_codes, bm25_index, embedding_index, base_tokenizer, finetuned_model


(code_by_nr, valid_codes, bm25_index, embedding_index,
 base_tokenizer, finetuned_model) = load_resources()

st.title("GOZ-Code-Extraktion: LoRA-Finetuning vs. RAG-Baseline")
note_text = st.text_area(
    "Behandlungsnotiz",
    "Zahn 36: Infiltrationsanästhesie, Karies excaviert, Kompositfüllung zweiflächig gelegt.",
)

if st.button("Extrahieren"):
    candidate_nrs = retrieve_candidates(note_text, bm25_index, embedding_index, top_n=12)
    candidates = [code_by_nr[nr] for nr in candidate_nrs]

    col_rag, col_finetune = st.columns(2)

    with col_rag:
        st.subheader("RAG-Baseline")
        # disable_adapter() schaltet den LoRA-Adapter kurzzeitig ab, damit
        # die RAG-Baseline auf den unveränderten Basis-Gewichten läuft,
        # ohne eine zweite Modellkopie zu brauchen.
        with finetuned_model.disable_adapter():
            rag_codes = generate_codes(finetuned_model, base_tokenizer, note_text, valid_codes, candidates=candidates)
        st.write([f"{nr}: {code_by_nr[nr].bezeichnung}" for nr in rag_codes])
        with st.expander("Retrieval-Kandidaten (Prompt-Kontext)"):
            st.write([f"{nr}: {code_by_nr[nr].bezeichnung}" for nr in candidate_nrs])

    with col_finetune:
        st.subheader("LoRA-Finetune")
        finetune_codes = generate_codes(finetuned_model, base_tokenizer, note_text, valid_codes, candidates=None)
        st.write([f"{nr}: {code_by_nr[nr].bezeichnung}" for nr in finetune_codes])
        st.caption("Kein Retrieval zur Inferenzzeit — Wissen steckt in den LoRA-Gewichten.")
