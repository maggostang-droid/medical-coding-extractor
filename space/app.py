"""Streamlit-Demo für Streamlit Community Cloud: GOZ-Codes aus zahnärztlichen
Behandlungsnotizen — RAG-Baseline, LoRA-Finetune und Graph im Vergleich.

Unterschied zur lokalen Fassung (app.py im Repo-Root): **kein Modell.** Die 81
Testnotizen kommen als vorberechnete, echte Ausgaben des Auswertungslaufs,
aus dem auch die Metriken stammen. Die Seite ist damit sofort da und braucht
weder Torch noch Transformers (siehe requirements.txt).

Aufbau (bewusst dreistufig, damit niemand erst Metriken deuten muss, bevor er
die Aufgabe verstanden hat):

1. **Was hier passiert** — ein handannotiertes Beispiel: Textstelle → Ziffer.
   Steht ohne Klick da und erklärt den Use Case in fünf Sekunden.
2. **Der Vergleich** — eine Zeile pro Ansatz, Ziffern als Chips mit
   Treffer/Fehler-Markierung und einem Urteil rechts. Vorausgewählt ist
   dieselbe Notiz wie oben, damit der Faden nicht reißt.
3. **Alles Weitere** — Metriken, Erklärungen, Graph-Befund: aufklappbar.

Live-Inferenz ist hier bewusst nicht vorgesehen: sie bräuchte Llama-3.2-3B in
bfloat16 (~6,4 GB) plus das Embedding-Modell der RAG-Baseline, was keine
kostenlose Hosting-Stufe hergibt. Die Seite erklärt das offen, statt einen
Knopf anzubieten, der garantiert in einen Speicherfehler läuft.
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
ADAPTER_URL = "https://huggingface.co/VoidFloat/goz-extract-llama32-3b"
REPO_URL = "https://github.com/maggostang-droid/medical-coding-extractor"

# Handannotiertes Einstiegsbeispiel. Die Zuordnung Textstelle -> Ziffer ist
# hier von Hand gesetzt (nur zur Erklaerung); alles darunter ist echter
# Modell-Output. Diese Notiz ist auch im Testset und wird unten vorausgewaehlt.
HERO_TEXT = (
    "Zahn 36 starke Schmerzen. Infiltration gesetzt, alte Krone musste runter. "
    "Nach Dezemention deutlich kariöser Defekt sichtbar, WK-Aufbereitung zwei "
    "Kanäle durchgeführt."
)
HERO_MARKUPS = [
    ("Infiltration gesetzt", "0090", "Infiltrationsanästhesie"),
    ("alte Krone musste runter", "2290", "Entfernung einer Krone"),
    ("WK-Aufbereitung zwei Kanäle", "2410", "Wurzelkanal-Aufbereitung"),
]

st.set_page_config(page_title="GOZ-Extraktion: Finetune vs. RAG vs. Graph", page_icon="🦷", layout="wide")

st.markdown(
    """
<style>
  .goz-notiz {
    border-left: 3px solid rgba(128,128,128,.45);
    padding: .7rem 1rem; border-radius: 0 8px 8px 0;
    background: rgba(128,128,128,.08); line-height: 1.75; font-size: 1.02rem;
  }
  .goz-notiz mark {
    background: rgba(42,120,214,.22); color: inherit;
    padding: .05rem .25rem; border-radius: 3px; font-weight: 600;
  }
  .goz-map { display: grid; gap: .35rem; margin: .8rem 0 .2rem; }
  .goz-map-row {
    display: grid; grid-template-columns: 1fr auto auto; gap: .9rem;
    align-items: baseline; padding: .45rem .8rem; border-radius: 7px;
    background: rgba(128,128,128,.07); font-size: .93rem;
  }
  .goz-map-row .von { opacity: .75; font-style: italic; }
  .goz-map-row .nach { font-weight: 700; font-variant-numeric: tabular-nums; }
  .goz-map-row .was { opacity: .6; font-size: .85rem; }
  .goz-chips { display: flex; flex-wrap: wrap; gap: .32rem; align-items: center; }
  .goz-chip {
    font-size: .84rem; font-weight: 650; font-variant-numeric: tabular-nums;
    padding: .2rem .5rem; border-radius: 6px;
    border: 1px solid rgba(128,128,128,.35); white-space: nowrap;
  }
  .goz-chip.hit { border-color: #0ca30c; }
  .goz-chip.extra { border-color: #d03b3b; opacity: .72; text-decoration: line-through; }
  .goz-chip.fehlt { border: 1px dashed #d03b3b; color: #d03b3b; }
  .goz-urteil {
    font-size: .82rem; font-weight: 650; padding: .22rem .6rem;
    border-radius: 20px; white-space: nowrap; display: inline-block;
  }
  .goz-urteil.ok { background: rgba(12,163,12,.16); color: #0ca30c; }
  .goz-urteil.teil { background: rgba(250,178,25,.20); color: #b07d00; }
  .goz-urteil.nein { background: rgba(208,59,59,.14); color: #d03b3b; }
  .goz-name { font-weight: 650; font-size: .95rem; line-height: 1.25; }
  .goz-name small { display: block; font-weight: 400; opacity: .6; font-size: .78rem; }
</style>
""",
    unsafe_allow_html=True,
)


# --- Daten ------------------------------------------------------------------
@st.cache_data
def load_codes():
    from goz_extract.schema import GozCode

    codes = [GozCode(**c) for c in json.loads(CODES_PATH.read_text(encoding="utf-8"))]
    return {c.goz_nr: c for c in codes}


@st.cache_data
def load_examples():
    return json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))


code_by_nr = load_codes()
examples = load_examples()


# --- Bausteine --------------------------------------------------------------
def chips_html(predicted, expected):
    """Ziffern als Chips: getroffen, zu viel, fehlend — Symbol statt nur Farbe."""
    parts = []
    for nr in sorted(predicted):
        if nr in expected:
            parts.append(f'<span class="goz-chip hit">✓ {nr}</span>')
        else:
            parts.append(f'<span class="goz-chip extra">✕ {nr}</span>')
    for nr in sorted(set(expected) - set(predicted)):
        parts.append(f'<span class="goz-chip fehlt">! {nr} fehlt</span>')
    return f'<div class="goz-chips">{"".join(parts) or "<em>nichts vorhergesagt</em>"}</div>'


def urteil_html(predicted, expected):
    zuviel = len(set(predicted) - set(expected))
    fehlt = len(set(expected) - set(predicted))
    if not zuviel and not fehlt:
        return '<span class="goz-urteil ok">✓ exakt</span>'
    teile = []
    if zuviel:
        teile.append(f"{zuviel} zu viel")
    if fehlt:
        teile.append(f"{fehlt} fehlt" if fehlt == 1 else f"{fehlt} fehlen")
    klasse = "teil" if zuviel + fehlt == 1 else "nein"
    zeichen = "▲" if klasse == "teil" else "✕"
    return f'<span class="goz-urteil {klasse}">{zeichen} {", ".join(teile)}</span>'


def ergebniszeile(name, unterzeile, predicted, expected):
    c1, c2, c3 = st.columns([1.15, 3.3, 1.05], vertical_alignment="center")
    c1.markdown(f'<div class="goz-name">{name}<small>{unterzeile}</small></div>', unsafe_allow_html=True)
    c2.markdown(chips_html(predicted, expected), unsafe_allow_html=True)
    c3.markdown(urteil_html(predicted, expected), unsafe_allow_html=True)


# --- Ebene 1: Was hier passiert ---------------------------------------------
st.title("🦷 Von der Behandlungsnotiz zur Abrechnungsziffer")
st.markdown(
    "Zahnärzte schreiben nach der Behandlung eine **Notiz in Fließtext**. Für die "
    "Abrechnung müssen daraus die passenden **GOZ-Ziffern** werden — heute Handarbeit. "
    "Genau das macht dieses Projekt automatisch, und vergleicht dabei drei Wege dorthin."
)

hero_markiert = HERO_TEXT
for phrase, _, _ in HERO_MARKUPS:
    hero_markiert = hero_markiert.replace(phrase, f"<mark>{phrase}</mark>")
st.markdown(f'<div class="goz-notiz">{hero_markiert}</div>', unsafe_allow_html=True)

zeilen = "".join(
    f'<div class="goz-map-row"><span class="von">„{phrase}“</span>'
    f'<span class="nach">{nr}</span><span class="was">{was}</span></div>'
    for phrase, nr, was in HERO_MARKUPS
)
st.markdown(f'<div class="goz-map">{zeilen}</div>', unsafe_allow_html=True)
st.caption(
    "Ein handverlesenes Beispiel zur Erklärung — die Zuordnung Textstelle → Ziffer ist "
    "hier von Hand markiert. Alles ab hier ist echter Modell-Output."
)

# --- Ebene 2: Der Vergleich -------------------------------------------------
st.divider()
st.subheader("Drei Wege, dasselbe zu tun")

hero_idx = next((i for i, e in enumerate(examples) if e["text"] == HERO_TEXT), 0)
idx = st.selectbox(
    f"Notiz auswählen — {len(examples)} aus dem Testset",
    range(len(examples)),
    index=hero_idx,
    format_func=lambda i: f"[{examples[i]['difficulty']}] {examples[i]['text'][:95]}…",
)
beispiel = examples[idx]
if idx != hero_idx:
    st.markdown(f'<div class="goz-notiz">{beispiel["text"]}</div>', unsafe_allow_html=True)
    st.write("")

expected = beispiel["expected"]
graph = graph_predict(beispiel["rag"], beispiel["finetune"], set(code_by_nr))

ergebniszeile("Soll", "so wäre es richtig", expected, expected)
ergebniszeile("RAG-Baseline", "schlägt nach", beispiel["rag"], expected)
ergebniszeile("LoRA-Finetune", "hat gelernt", beispiel["finetune"], expected)
ergebniszeile("Graph · Aggregator", "verrechnet beide", graph.predicted_codes, expected)
if graph.uncertain:
    st.caption(
        "Der Aggregator legt " + ", ".join(f"`{c}`" for c in graph.uncertain)
        + " als unsicher einem Menschen vor, statt sie vorherzusagen — bei 95 % der "
        "Notizen. Genau daran ist diese Fassung gescheitert."
    )
ergebniszeile("Graph · Checker <sup>1</sup>", "wählt aus beiden aus", beispiel["checker"], expected)
st.caption(
    "¹ Vorab-Sonde mit einem stärkeren Modell — zeigt, dass die Auswahlaufgabe lösbar "
    "ist, ist aber **kein fairer Vergleich** zu den Zeilen darüber. Details unten."
)

# --- Ebene 3: alles Weitere, aufklappbar ------------------------------------
st.divider()

with st.expander("Was bedeuten die Ziffern in diesem Beispiel?"):
    beteiligt = sorted(set(expected) | set(beispiel["rag"]) | set(beispiel["finetune"]))
    for nr in beteiligt:
        code = code_by_nr.get(nr)
        if not code:
            continue
        marker = "✓ richtig" if nr in expected else "— trifft hier nicht zu"
        st.markdown(f"**GOZ {nr}** · {code.bezeichnung}  \n*{marker}*")
        if code.erweiterte_beschreibung:
            st.caption(code.erweiterte_beschreibung)

with st.expander("Wie gut sind die drei Ansätze insgesamt? (81 Testnotizen)"):
    m1, m2, m3 = st.columns(3)
    m1.metric("F1 — Finetune", "0,59", "RAG: 0,48")
    m2.metric("Exact Match — Finetune", "0,38", "RAG: 0,07")
    m3.metric("Recall — RAG", "0,70", "Finetune: 0,58")
    st.caption(
        "Das Finetune trifft öfter die exakte Codekombination, die RAG-Baseline findet "
        "öfter irgendeinen richtigen Code — sie schlägt dafür aber deutlich zu breit vor "
        "(Ø 3,0 Ziffern bei 1,7 erwarteten)."
    )

with st.expander("Wie funktionieren RAG-Baseline und LoRA-Finetune?"):
    st.markdown(
        """
**RAG-Baseline** — *Nachschlagen statt Auswendiglernen*: BM25 + Embeddings
(`multilingual-e5-base`) holen aus den 10 möglichen GOZ-Codes die 5 besten
Kandidaten; das unveränderte Basismodell wählt daraus aus. Braucht kein
Fachwissen im Modell, ist aber nur so gut wie die Suche davor.

**LoRA-Finetune** — *Auswendiggelerntes Fachwissen*: Ein kleiner zusätzlicher
Gewichtssatz wurde auf 325 Beispielnotizen trainiert. Zur Laufzeit wird nichts
nachgeschlagen. Kann Fachjargon verinnerlichen, ist aber an die Trainingsdaten
gebunden.

Beide laufen auf demselben Basismodell (Llama-3.2-3B-Instruct), damit der
Vergleich fair ist. Die Trainingsdaten sind vollständig synthetisch erzeugt —
kein Abgleich mit realen Praxisfällen.
        """
    )

with st.expander("Und der Graph? Der erste Versuch ist gescheitert."):
    st.markdown(
        f"""
Die Idee lag nahe: Beide Pfade parallel laufen lassen, die Vorschläge
zusammenführen, ein deterministischer Verifier prüft gegen den GOZ-Katalog.
Übernommen wird, was beide liefern oder was vom präziseren Finetune kommt; was
nur die RAG-Baseline vorschlägt, gilt als unsicher.

**Das Ergebnis über alle 81 Notizen: exakt die Zahlen des Finetunes — bei einer
Prüfquote von 95 %.** Die Regel ist algebraisch identisch zur
Finetune-Vorhersage. Was die RAG-Baseline exklusiv beisteuert, ist zu drei
Vierteln falsch (Precision 0,24).

Die Diagnose zeigt aber, wo der Hebel wirklich sitzt: Die richtigen Ziffern
stecken in **65 von 81 Notizen** bereits in der Vereinigung beider Pfade — sie
werden nur nicht ausgewählt. Ein **Checker-Knoten**, der genau das tut, erreicht
in einer Vorab-Sonde Exact Match **0,74** statt 0,38 und drückt die Prüfquote von
95 % auf 1 %.

Damit verschiebt sich das Problem: Die Auswahl ist weitgehend gelöst, der
Engpass ist jetzt der Kandidatenpool. Messung, Diagnose-Skript und beide
Fassungen stehen im [Repo]({REPO_URL}#dritter-weg-ein-graph--und-warum-er-hier-nicht-hält).
        """
    )

with st.expander("Warum kann ich hier keinen eigenen Text eingeben?"):
    st.markdown(
        f"""
Ehrliche Antwort: **Arbeitsspeicher.** Live-Inferenz hieße, Llama-3.2-3B in
bfloat16 zu laden — rund 6,4 GB — plus das Embedding-Modell für die
RAG-Baseline. Das passt in keine der kostenlosen Hosting-Stufen, auf denen
dieses Portfolio läuft.

Ein Ladebalken, der nach Minuten in einen Speicherfehler kippt, wäre die
schlechtere Demo. Diese Seite zeigt deshalb **vorberechnete, echte Ausgaben**
desselben Auswertungslaufs, aus dem auch die Metriken stammen — keine Attrappe
und keine geschönten Zahlen.

Selbst laufen lassen geht trotzdem, beides ist öffentlich:

* **LoRA-Adapter:** [{ADAPTER_URL.split("//")[1]}]({ADAPTER_URL})
* **Code und Auswertung:** [medical-coding-extractor]({REPO_URL})
        """
    )

st.divider()
st.caption(
    "Portfolio-Projekt von Dr.-Ing. Marco Stang · "
    f"[Code auf GitHub]({REPO_URL}) · "
    "[MARCO.OS](https://maggostang-droid.github.io/marco-os/)"
)
