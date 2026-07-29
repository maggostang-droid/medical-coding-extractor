"""Streamlit-Demo: Behandlungsnotiz eingeben, Vorhersage von RAG-Baseline
und LoRA-Finetune nebeneinander vergleichen."""
import json
from pathlib import Path

import streamlit as st
import torch
from dotenv import load_dotenv
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig

from goz_extract.inference import generate_codes, load_model
from goz_extract.retrieval import BM25Index, EmbeddingIndex, retrieve_candidates
from goz_extract.schema import GozCode

load_dotenv()  # HF_TOKEN fürs gated meta-llama-Modell, siehe .env

st.set_page_config(page_title="GOZ-Extraktion: Finetune vs. RAG", page_icon="🦷", layout="wide")

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

BEISPIEL_NOTIZEN = {
    "Füllung (einfach)": "Zahn 36: Infiltrationsanästhesie, Karies excaviert, Kompositfüllung zweiflächig gelegt.",
    "Wurzelkanal": "Zahn 46 distal: Exkavation, Konditionierung, Adhäsiv, Kompo geschichtet, finiert. Biss kontrolliert.",
    "Mehrere Schritte": "Regio 34-33: Interdent. Gingiva abgedr., IOS-Scan mit Farbaufn., digit. Biss reg.",
}


@st.cache_resource(show_spinner="Lade Modelle (einmalig, danach im Cache) …")
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
    # Speicherbedarf. 8-bit-Quantisierung (~3-4GB statt ~6GB in fp16),
    # weil diese Demo auf einer CPU-Maschine mit wenig freiem RAM laufen
    # soll (voller fp16-Load hat hier zu einem OOM-Absturz geführt).
    base_model, base_tokenizer = load_model(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    )
    finetuned_model = PeftModel.from_pretrained(base_model, "adapters/goz-extract-llama32-3b")
    return code_by_nr, valid_codes, bm25_index, embedding_index, base_tokenizer, finetuned_model


def code_card(nr: str, code_by_nr: dict[str, GozCode]) -> None:
    """Zeigt einen GOZ-Code als lesbare Karte statt als rohen JSON-Eintrag."""
    code = code_by_nr[nr]
    with st.container(border=True):
        st.markdown(f"**GOZ {nr}** — {code.bezeichnung}")
        if code.erweiterte_beschreibung:
            st.caption(code.erweiterte_beschreibung)


def code_list(title: str, nrs: list[str], code_by_nr: dict[str, GozCode]) -> None:
    st.markdown(f"**{title}**")
    if not nrs:
        st.info("Keine passenden GOZ-Codes gefunden.")
        return
    for nr in nrs:
        code_card(nr, code_by_nr)


(code_by_nr, valid_codes, bm25_index, embedding_index,
 base_tokenizer, finetuned_model) = load_resources()

st.title("🦷 GOZ-Code-Extraktion: LoRA-Finetuning vs. RAG")

st.markdown(
    "Diese Demo vergleicht zwei Ansätze, um aus einer zahnärztlichen "
    "Behandlungsnotiz automatisch die passenden **GOZ-Abrechnungscodes** "
    "(amtliche Gebührenordnung für Zahnärzte) zu extrahieren — einmal mit "
    "**Retrieval-Augmented Generation (RAG)**, einmal mit einem eigens "
    "**feingetunten Modell (LoRA)**. Beide laufen auf demselben "
    "Basismodell (Llama-3.2-3B-Instruct), damit der Vergleich fair ist."
)

with st.expander("ℹ️ Wie funktionieren die beiden Ansätze?"):
    st.markdown(
        """
**RAG-Baseline** — *Nachschlagen statt Auswendiglernen*
1. Aus den 10 möglichen GOZ-Codes werden per Textsuche (BM25 + Embeddings)
   die 5 am besten passenden **Kandidaten** herausgesucht.
2. Diese Kandidatenliste wird dem Modell zusammen mit der Notiz im Prompt
   gezeigt — es wählt daraus aus, statt die Codes selbst zu kennen.
3. Vorteil: das Basismodell braucht kein Zahnmedizin-Fachwissen. Nachteil:
   die Vorhersage ist nur so gut wie die Suche, die die Kandidaten liefert.

**LoRA-Finetune** — *Auswendiggelerntes Fachwissen*
1. Das Basismodell wurde mit einem kleinen zusätzlichen Gewichtssatz
   (LoRA-Adapter) auf 325 Beispielnotizen trainiert.
2. Zur Laufzeit gibt es **kein Nachschlagen** — das Modell entscheidet
   direkt aus dem, was es beim Training gelernt hat.
3. Vorteil: kann eigene Formulierungen/Fachjargon verinnerlichen. Nachteil:
   nur so gut wie die Trainingsdaten, die es gesehen hat.

**Ergebnis der Auswertung** über 81 unabhängige Testnotizen (siehe
`results/results.md`): RAG erreicht F1 0.48 (höherer Recall, findet öfter
*irgendeinen* richtigen Code), das Finetune F1 0.59 (höhere Precision,
trifft öfter die *exakte* Code-Kombination).
        """
    )

st.subheader("Behandlungsnotiz eingeben")
beispiel = st.selectbox(
    "Beispielnotiz laden (optional) — oder unten eigenen Text eingeben",
    ["— eigenen Text eingeben —"] + list(BEISPIEL_NOTIZEN.keys()),
)
vorbefuellt = BEISPIEL_NOTIZEN.get(beispiel, "Zahn 36: Infiltrationsanästhesie, Karies excaviert, Kompositfüllung zweiflächig gelegt.")
note_text = st.text_area("Notiztext", vorbefuellt, height=100)

extrahieren = st.button("🔍 GOZ-Codes extrahieren", type="primary")

if extrahieren:
    with st.spinner("Suche Retrieval-Kandidaten und generiere Vorhersagen …"):
        candidate_nrs = retrieve_candidates(note_text, bm25_index, embedding_index, top_n=5)
        candidates = [code_by_nr[nr] for nr in candidate_nrs]

        # disable_adapter() schaltet den LoRA-Adapter kurzzeitig ab, damit
        # die RAG-Baseline auf den unveränderten Basis-Gewichten läuft,
        # ohne eine zweite Modellkopie zu brauchen.
        with finetuned_model.disable_adapter():
            rag_codes = generate_codes(finetuned_model, base_tokenizer, note_text, valid_codes, candidates=candidates)

        finetune_codes = generate_codes(finetuned_model, base_tokenizer, note_text, valid_codes, candidates=None)

    st.divider()
    col_rag, col_finetune = st.columns(2)

    with col_rag:
        st.subheader("🔎 RAG-Baseline")
        st.caption("Wählt aus den unten gezeigten Retrieval-Kandidaten aus.")
        code_list("Vorhergesagte Codes", rag_codes, code_by_nr)
        with st.expander(f"Retrieval-Kandidaten, die dem Modell gezeigt wurden ({len(candidate_nrs)})"):
            for nr in candidate_nrs:
                code_card(nr, code_by_nr)

    with col_finetune:
        st.subheader("🧠 LoRA-Finetune")
        st.caption("Kein Retrieval zur Laufzeit — Wissen steckt in den LoRA-Gewichten.")
        code_list("Vorhergesagte Codes", finetune_codes, code_by_nr)
else:
    st.caption("👆 Notiz eingeben oder Beispiel wählen, dann auf den Button klicken.")
