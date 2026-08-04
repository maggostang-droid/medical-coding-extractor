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
REPO_URL = "https://github.com/marco-stang/medical-coding-extractor"

# Handannotiertes Einstiegsbeispiel. Die Zuordnung Textstelle -> Ziffer ist
# hier von Hand gesetzt (nur zur Erklaerung); alles darunter ist echter
# Modell-Output. Diese Notiz ist auch im Testset und wird unten vorausgewaehlt.
HERO_TEXT = (
    "Zahn 36 starke Schmerzen. Infiltration gesetzt, alte Krone musste runter. "
    "Nach Dezemention deutlich kariöser Defekt sichtbar, WK-Aufbereitung zwei "
    "Kanäle durchgeführt."
)
HERO_MARKUPS = [
    ("Infiltration gesetzt", "0090", "Infiltrationsanästhesie", 60),
    ("alte Krone musste runter", "2290", "Entfernung einer Krone", 180),
    ("WK-Aufbereitung zwei Kanäle", "2410", "Wurzelkanal-Aufbereitung, je Kanal", 392),
]

# GOZ 2012: Betrag = Punktzahl x Punktwert x Steigerungsfaktor. Der Punktwert ist
# im Gesetz festgeschrieben; 2,3 ist der Regelhoechstsatz, den Praxen im
# Normalfall ansetzen. Gegenprobe: 0090 ergibt damit 7,76 EUR - genau der Wert,
# den die Gebuehrenverzeichnisse veroeffentlichen.
GOZ_PUNKTWERT = 0.0562421
GOZ_FAKTOR = 2.3


def euro(punkte: int) -> str:
    return f"{punkte * GOZ_PUNKTWERT * GOZ_FAKTOR:.2f}".replace(".", ",") + " €"

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
    display: grid; grid-template-columns: 1fr auto minmax(0, 15rem) 5.2rem; gap: .9rem;
    align-items: baseline; padding: .45rem .8rem; border-radius: 7px;
    background: rgba(128,128,128,.07); font-size: .93rem;
  }
  .goz-map-row .von { opacity: .75; font-style: italic; }
  .goz-map-row .nach { font-weight: 700; font-variant-numeric: tabular-nums; }
  .goz-map-row .was { opacity: .6; font-size: .85rem; }
  .goz-map-row .preis { font-variant-numeric: tabular-nums; text-align: right; opacity: .85; }
  .goz-map-row.summe {
    background: none; border-top: 1px solid rgba(128,128,128,.3);
    border-radius: 0; margin-top: .15rem; padding-top: .55rem;
  }
  .goz-map-row.summe .von { font-style: normal; opacity: 1; font-weight: 600; }
  .goz-map-row.summe .preis { font-weight: 700; opacity: 1; font-size: 1.05rem; }
  .goz-raster { display: grid; gap: .3rem; align-items: center; }
  .goz-zelle { display: flex; justify-content: flex-start; min-height: 1.6rem; }
  .goz-zelle.trenn {
    border-left: 1px solid rgba(128,128,128,.35);
    margin-left: -.15rem; padding-left: .45rem;
  }
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
def spalten_fuer(expected, *vorhersagen):
    """Feste Spaltenordnung für alle Zeilen: erst die erwarteten Ziffern, dann
    die überzähligen. So steht dieselbe Ziffer in jeder Zeile an derselben
    Stelle — nur dann ist der Vergleich auf einen Blick lesbar. Die Zweiteilung
    macht zusätzlich sichtbar, was links fehlt und was rechts zu viel ist."""
    alle = set().union(*(set(v) for v in vorhersagen)) if vorhersagen else set()
    return sorted(expected), sorted(alle - set(expected))


def raster_html(predicted, expected, soll_spalten, extra_spalten):
    """Eine Zeile im festen Raster — leere Zellen halten die Ausrichtung."""
    predicted, expected = set(predicted), set(expected)
    zellen = []
    for i, nr in enumerate(soll_spalten + extra_spalten):
        trenn = " trenn" if extra_spalten and i == len(soll_spalten) else ""
        if nr in predicted and nr in expected:
            inhalt = f'<span class="goz-chip hit">✓ {nr}</span>'
        elif nr in predicted:
            inhalt = f'<span class="goz-chip extra">✕ {nr}</span>'
        elif nr in expected:
            inhalt = f'<span class="goz-chip fehlt">! {nr}</span>'
        else:
            inhalt = ""
        zellen.append(f'<span class="goz-zelle{trenn}">{inhalt}</span>')
    spalten = len(soll_spalten) + len(extra_spalten)
    return (
        f'<div class="goz-raster" style="grid-template-columns: repeat({spalten}, 4.9rem)">'
        f'{"".join(zellen)}</div>'
    )


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


def ergebniszeile(name, unterzeile, predicted, expected, soll_spalten, extra_spalten):
    c1, c2, c3 = st.columns([1.15, 3.3, 1.05], vertical_alignment="center")
    c1.markdown(f'<div class="goz-name">{name}<small>{unterzeile}</small></div>', unsafe_allow_html=True)
    c2.markdown(raster_html(predicted, expected, soll_spalten, extra_spalten), unsafe_allow_html=True)
    c3.markdown(urteil_html(predicted, expected), unsafe_allow_html=True)


# --- Ebene 1: Was hier passiert ---------------------------------------------
st.title("🦷 Von der Behandlungsnotiz zur Abrechnungsziffer")
st.markdown(
    "Zahnärzte schreiben nach der Behandlung eine **Notiz in Fließtext**. Für die "
    "Abrechnung müssen daraus die passenden **GOZ-Ziffern** werden — heute Handarbeit. "
    "Genau das macht dieses Projekt automatisch, und vergleicht dabei drei Wege dorthin."
)

hero_markiert = HERO_TEXT
for phrase, _, _, _ in HERO_MARKUPS:
    hero_markiert = hero_markiert.replace(phrase, f"<mark>{phrase}</mark>")
st.markdown(f'<div class="goz-notiz">{hero_markiert}</div>', unsafe_allow_html=True)

zeilen = "".join(
    f'<div class="goz-map-row"><span class="von">„{phrase}“</span>'
    f'<span class="nach">{nr}</span><span class="was">{was}</span>'
    f'<span class="preis">{euro(pkt)}</span></div>'
    for phrase, nr, was, pkt in HERO_MARKUPS
)
summe = euro(sum(pkt for *_, pkt in HERO_MARKUPS))
zeilen += (
    f'<div class="goz-map-row summe"><span class="von">Diese eine Notiz ist wert</span>'
    f'<span class="nach"></span><span class="was"></span>'
    f'<span class="preis">{summe}</span></div>'
)
st.markdown(f'<div class="goz-map">{zeilen}</div>', unsafe_allow_html=True)
st.markdown(
    "**Übersieht jemand eine Ziffer, fehlt der Betrag auf der Rechnung** — bei tausenden "
    "Notizen im Jahr ist das der Grund, warum sich Automatisierung hier lohnt."
)
st.caption(
    "Ein handverlesenes Beispiel zur Erklärung — die Zuordnung Textstelle → Ziffer ist "
    "hier von Hand markiert. Beträge nach GOZ 2012 (Punktwert 5,62421 Cent) zum "
    "2,3-fachen Regelhöchstsatz. 2410 wird je Kanal berechnet, hier also zweimal — "
    "diese Mehrfachabrechnung bildet das Projekt bewusst noch nicht ab. "
    "Alles ab hier ist echter Modell-Output."
)

# --- Ebene 2: Der Vergleich -------------------------------------------------
st.divider()
st.subheader("Drei Wege, das Fachwissen ins Modell zu bekommen")
st.markdown(
    "Ein Sprachmodell kennt die GOZ nicht. Dieses Wissen kann an drei Stellen sitzen — "
    "und jede Stelle hat einen anderen Preis:"
)
st.markdown(
    "- **Außerhalb des Modells** *(RAG-Baseline)* — die Codeliste wird zur Laufzeit "
    "durchsucht. Neue oder geänderte Ziffern kosten kein Training, dafür ist das "
    "Ergebnis nur so gut wie die Suche davor.\n"
    "- **Im Modell** *(LoRA-Finetune)* — das Wissen steckt nach dem Training in den "
    "Gewichten. Versteht Fachjargon ohne Nachschlagen, muss für jede Änderung aber neu "
    "trainiert werden.\n"
    "- **In der Verdrahtung** *(Graph)* — beide Pfade laufen und werden zusammengeführt. "
    "Kostet kein zusätzliches Training, dafür mehr bewegliche Teile und doppelte Inferenz."
)
st.caption(
    "Alle drei laufen auf demselben Basismodell (Llama-3.2-3B-Instruct) — nur so misst "
    "der Vergleich die Methode und nicht die Modellgröße."
)

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

soll_spalten, extra_spalten = spalten_fuer(
    expected, beispiel["rag"], beispiel["finetune"], graph.predicted_codes
)

ergebniszeile("Soll", "so wäre es richtig", expected, expected, soll_spalten, extra_spalten)
ergebniszeile("RAG-Baseline", "sucht nach", beispiel["rag"], expected, soll_spalten, extra_spalten)
ergebniszeile("LoRA-Finetune", "hat gelernt", beispiel["finetune"], expected, soll_spalten, extra_spalten)
ergebniszeile("Graph", "führt zusammen", graph.predicted_codes, expected, soll_spalten, extra_spalten)
st.caption(
    "Jede Spalte ist eine GOZ-Ziffer, in allen Zeilen an derselben Stelle. "
    "Links vom Strich stehen die richtigen Ziffern, rechts die überzähligen — "
    "✓ getroffen · ✕ zu viel · ! nicht gefunden."
)
if graph.uncertain:
    st.caption(
        "Der Graph übernimmt nur, was **beide** Pfade liefern oder was allein vom "
        "Finetune kommt. " + ", ".join(f"`{c}`" for c in graph.uncertain)
        + " stammen allein von der RAG-Baseline und gelten deshalb als unsicher — "
        "unabhängig davon, ob sie stimmen. Bei 95 % der Notizen passiert genau das, "
        "und deshalb ist diese Fassung gescheitert (siehe unten)."
    )
st.caption(
    "Ein Beispiel ist ein Beispiel: Über alle 81 Testnotizen trifft das Finetune 38 % "
    "exakt, die RAG-Baseline 7 %, der Graph ebenfalls 38 %."
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

Der Grund ist strukturell: „von beiden geliefert" plus „nur vom Finetune" ist
genau die Menge, die das Finetune vorhergesagt hat. Diese Regel kann also nie
etwas *hinzufügen* — sie entscheidet nach Herkunft, nicht nach Inhalt.

Die Diagnose zeigt, wo der Hebel stattdessen sitzt: Die richtigen Ziffern
stecken in **65 von 81 Notizen** bereits in der Vereinigung beider Pfade — sie
werden nur nicht ausgewählt. Ein **Checker-Knoten**, der die gepoolten Kandidaten
gegen die Notiz prüft statt nach Herkunft zu urteilen, ist deshalb der nächste
Ausbauschritt.

Eine Vorab-Sonde bestätigt, dass die Auswahlaufgabe lösbar ist. Sie lief
allerdings auf einem **stärkeren Modell** als die drei Ansätze oben und ist
deshalb kein Vergleich zu ihnen — sie klärt nur die Machbarkeit, bevor eine
GPU-Sitzung investiert wird. Die faire Messung mit demselben Basismodell steht
noch aus; sie kommt hier als vierte Zeile dazu, sobald sie existiert.

Messung, Diagnose-Skript und beide Fassungen stehen im
[Repo]({REPO_URL}#dritter-weg-ein-graph--und-warum-er-hier-nicht-hält).
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
    "[MARCO.OS](https://marco-stang.github.io/)"
)
