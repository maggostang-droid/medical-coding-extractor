"""Streamlit-Demo für Streamlit Community Cloud: GOZ-Codes aus zahnärztlichen
Behandlungsnotizen — LoRA-Finetune gegen RAG-Baseline.

Unterschied zur lokalen Fassung (app.py im Repo-Root): **kein Modell.** Die 81
Testnotizen kommen als vorberechnete, echte Ausgaben des Auswertungslaufs,
aus dem auch die Metriken stammen. Die Seite ist damit sofort da und braucht
weder Torch noch Transformers (siehe requirements.txt) — entsprechend schnell
und zuverlässig baut sie.

Live-Inferenz ist hier bewusst nicht vorgesehen: sie bräuchte Llama-3.2-3B in
bfloat16 (~6,4 GB) plus das Embedding-Modell der RAG-Baseline, was keine
kostenlose Hosting-Stufe hergibt. Ursprünglich war dafür ein Hugging-Face-
Space geplant; HF hat das Streamlit-SDK inzwischen abgeschafft und verlangt
für Docker-/Gradio-Spaces auch auf `cpu-basic` ein PRO-Abo. Die Seite erklärt
das Fehlen offen (Expander am Ende) und verlinkt Adapter und Code, statt einen
Knopf anzubieten, der garantiert in einen Speicherfehler läuft.

Wer die Live-Variante doch hosten will, findet den Bauplan in DEPLOY.md; die
Inferenz selbst steht unverändert in `goz_extract.inference` und
`goz_extract.retrieval`, die lokale Fassung im Repo-Root nutzt sie.
"""
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

# Community Cloud klont das ganze Repo, installiert aber nur requirements.txt
# — das Projektpaket selbst liegt unter src/ (siehe pyproject: packages.find
# where=["src"]) und ist sonst nicht importierbar.
sys.path.insert(0, str(REPO / "src"))

# Bewusst hier statt oben: braucht den sys.path-Eintrag. graph_merge ist
# framework-frei (nur Stdlib) und laeuft daher auch auf der freien Stufe.
from goz_extract.graph_merge import graph_predict  # noqa: E402

CODES_PATH = REPO / "data" / "goz_codes.json"
EXAMPLES_PATH = ROOT / "data" / "demo_examples.json"

# Der trainierte Adapter liegt oeffentlich auf dem Hub — wer die Inferenz
# selbst laufen lassen will, braucht ihn (siehe Hinweis am Seitenende).
ADAPTER_URL = "https://huggingface.co/VoidFloat/goz-extract-llama32-3b"

st.set_page_config(page_title="GOZ-Extraktion: Finetune vs. RAG", page_icon="🦷", layout="wide")


# --- Daten ------------------------------------------------------------------
@st.cache_data
def load_codes():
    from goz_extract.schema import GozCode

    codes = [
        GozCode(**c) for c in json.loads(CODES_PATH.read_text(encoding="utf-8"))
    ]
    return {c.goz_nr: c for c in codes}


@st.cache_data
def load_examples():
    return json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))


# --- Darstellung ------------------------------------------------------------
def code_card(nr, code_by_nr, mark=None):
    code = code_by_nr.get(nr)
    with st.container(border=True):
        label = f"**GOZ {nr}**"
        if code:
            label += f" — {code.bezeichnung}"
        if mark == "hit":
            label = "✅ " + label
        elif mark == "miss":
            label = "❌ " + label
        st.markdown(label)
        if code and code.erweiterte_beschreibung:
            st.caption(code.erweiterte_beschreibung)


def code_list(title, nrs, code_by_nr, expected=None):
    st.markdown(f"**{title}**")
    if not nrs:
        st.info("Keine passenden GOZ-Codes gefunden.")
        return
    for nr in nrs:
        mark = None
        if expected is not None:
            mark = "hit" if nr in expected else "miss"
        code_card(nr, code_by_nr, mark)


code_by_nr = load_codes()
examples = load_examples()

st.title("🦷 GOZ-Code-Extraktion: LoRA-Finetuning vs. RAG")
st.markdown(
    "Zwei Wege, aus einer zahnärztlichen **Behandlungsnotiz** die passenden "
    "**GOZ-Abrechnungscodes** zu ziehen — einmal per **Retrieval-Augmented "
    "Generation**, einmal per **LoRA-Finetune**. Beide laufen auf demselben "
    "Basismodell (Llama-3.2-3B-Instruct), damit der Vergleich fair ist."
)

m1, m2, m3 = st.columns(3)
m1.metric("F1 — Finetune", "0,59", "RAG: 0,48")
m2.metric("Exact Match — Finetune", "0,38", "RAG: 0,07")
m3.metric("Recall — RAG", "0,70", "Finetune: 0,58")
st.caption(
    "Gemessen über 81 unabhängige Testnotizen. Das Finetune trifft öfter die "
    "exakte Codekombination, die RAG-Baseline findet öfter irgendeinen "
    "richtigen Code."
)

with st.expander("ℹ️ Wie funktionieren die beiden Ansätze?"):
    st.markdown(
        """
**RAG-Baseline** — *Nachschlagen statt Auswendiglernen*: BM25 + Embeddings
(`multilingual-e5-base`) holen aus den 10 möglichen GOZ-Codes die 5 besten
Kandidaten; das unveränderte Basismodell wählt daraus aus. Braucht kein
Fachwissen im Modell, ist aber nur so gut wie die Suche davor.

**LoRA-Finetune** — *Auswendiggelerntes Fachwissen*: Ein kleiner zusätzlicher
Gewichtssatz wurde auf 325 Beispielnotizen trainiert. Zur Laufzeit wird nichts
nachgeschlagen. Kann Fachjargon verinnerlichen, ist aber an die
Trainingsdaten gebunden.

Die Trainingsdaten sind vollständig synthetisch erzeugt — kein Abgleich mit
realen Praxisfällen.
        """
    )

st.caption(
    "Echte Ausgaben aus dem Auswertungslauf — dieselben, aus denen die "
    "Metriken oben berechnet wurden. Kein Modell im Speicher, daher sofort da."
)
stufe = st.select_slider("Schwierigkeit", ["easy", "medium", "hard", "alle"], value="alle")
pool = [e for e in examples if stufe == "alle" or e["difficulty"] == stufe]
idx = st.selectbox(
    f"Notiz auswählen ({len(pool)} verfügbar)",
    range(len(pool)),
    format_func=lambda i: f"[{pool[i]['difficulty']}] {pool[i]['text'][:90]}…",
)
beispiel = pool[idx]
st.text_area("Behandlungsnotiz", beispiel["text"], height=100, disabled=True)

st.divider()
st.markdown("**Erwartete Codes:** " + ", ".join(f"`{c}`" for c in beispiel["expected"]))
col_rag, col_ft, col_graph = st.columns(3)
with col_rag:
    st.subheader("🔎 RAG-Baseline")
    code_list("Vorhergesagt", beispiel["rag"], code_by_nr, expected=beispiel["expected"])
with col_ft:
    st.subheader("🧠 LoRA-Finetune")
    code_list("Vorhergesagt", beispiel["finetune"], code_by_nr, expected=beispiel["expected"])
with col_graph:
    st.subheader("🕸️ Graph")
    graph_result = graph_predict(beispiel["rag"], beispiel["finetune"], set(code_by_nr))
    code_list(
        "Übernommen", graph_result.predicted_codes, code_by_nr, expected=beispiel["expected"]
    )
    if graph_result.uncertain:
        st.warning(
            "⚠️ `needs_review` — nur von der RAG-Baseline geliefert, "
            "nicht übernommen: " + ", ".join(f"`{c}`" for c in graph_result.uncertain)
        )

with st.expander("🕸️ Was macht die Graph-Spalte — und warum ist sie ehrlich gescheitert?"):
    st.markdown(
        """
Beide Pfade laufen parallel, ein **Fan-in** führt die Kandidaten zusammen,
ein **deterministischer Verifier** (reiner Code, kein LLM) prüft gegen den
GOZ-Katalog. Merge-Regel: Codes, die beide liefern, werden übernommen; Codes
nur vom Finetune ebenfalls; Codes nur von der RAG-Baseline gelten als
unsicher und lösen `needs_review` aus.

**Das Ergebnis über alle 81 Notizen: exakt die Zahlen des Finetunes — plus
eine Prüfquote von 95 %.** Die Regel ist algebraisch identisch zur
Finetune-Vorhersage; was die RAG-Baseline exklusiv beisteuert, ist zu drei
Vierteln falsch (Precision 0,24). Diese Spalte zeigt pro Notiz, *warum*: Man
sieht, welche Codes als unsicher aussortiert wurden.

Der Befund samt Obergrenzen-Rechnung steht im
[README](https://github.com/maggostang-droid/medical-coding-extractor#dritter-weg-ein-graph--und-warum-er-hier-nicht-hält)
— inklusive des Hebels, der tatsächlich Luft hätte (ein Checker-Knoten, der
aus den gepoolten Kandidaten auswählt: Obergrenze 0,80 statt 0,38).
        """
    )

with st.expander("Warum kann ich hier keinen eigenen Text eingeben?"):
    st.markdown(
        f"""
Ehrliche Antwort: **Arbeitsspeicher.** Live-Inferenz hiesse, Llama-3.2-3B in
bfloat16 zu laden — rund 6,4 GB — plus das Embedding-Modell fuer die
RAG-Baseline. Das passt in keine der kostenlosen Hosting-Stufen, auf denen
dieses Portfolio laeuft.

Ein Ladebalken, der nach Minuten in einen Speicherfehler kippt, waere die
schlechtere Demo. Diese Seite zeigt deshalb **vorberechnete, echte Ausgaben**
desselben Auswertungslaufs, aus dem auch die Metriken oben stammen — keine
Attrappe und keine geschoenten Zahlen.

Selbst laufen lassen geht trotzdem, beides ist oeffentlich:

* **LoRA-Adapter:** [{ADAPTER_URL.split("//")[1]}]({ADAPTER_URL})
* **Code und Auswertung:** [medical-coding-extractor](https://github.com/maggostang-droid/medical-coding-extractor)
        """
    )

st.divider()
st.caption(
    "Portfolio-Projekt von Dr.-Ing. Marco Stang · "
    "[Code auf GitHub](https://github.com/maggostang-droid/medical-coding-extractor) · "
    "[MARCO.OS](https://maggostang-droid.github.io/marco-os/)"
)
