# Live-Demo veröffentlichen — Streamlit Community Cloud

Diese Demo zeigt **vorberechnete, echte Ausgaben** des Auswertungslaufs und
lädt kein Modell. Sie braucht darum weder GPU noch viel RAM und läuft auf der
kostenlosen Stufe von Streamlit Community Cloud.

## 1. Adapter (erledigt)

Der trainierte LoRA-Adapter liegt öffentlich auf dem Hub:
<https://huggingface.co/VoidFloat/goz-extract-llama32-3b>

Die Demo *benutzt* ihn nicht — sie zeigt vorberechnete Ergebnisse. Er ist
oben, damit ein technischer Besucher die Inferenz selbst nachvollziehen kann.

## 2. Repo pushen

Community Cloud deployt aus GitHub, nicht per Upload. Der Ordner `space/`
muss also committed und auf `origin/master` sein.

## 3. Deployen

Auf <https://share.streamlit.io> → *Create app* → *Deploy a public app from
GitHub*:

| Feld | Wert |
|---|---|
| Repository | `maggostang-droid/medical-coding-extractor` |
| Branch | `master` |
| Main file path | `space/app.py` |

**Branch beachten:** Der Default-Branch des Repos ist `main` (ein alter
Einzelcommit). Die echte Historie liegt auf `master` — falsch gewählt, und
Community Cloud findet `space/app.py` nicht.

Der Build dauert ein bis zwei Minuten; `requirements.txt` enthält nur
`streamlit` und `pydantic`.

## 4. Ins Portfolio eintragen

In `marco-os/data/projects.js` beim Eintrag `goz-finetune-vs-rag`:

```js
    demoUrl: "<die URL von Schritt 3>",
    status: "live",
    coldStartNote: "…",
```

`coldStartNote` lohnt sich, weil Community Cloud Apps nach längerer
Inaktivität einfriert und der erste Aufruf danach den Kaltstart bezahlt —
dieselbe Mechanik wie beim Ask-Marco-Chat.

## Warum nicht Hugging Face Spaces

War der ursprüngliche Plan, ist aber nicht mehr möglich (geprüft am
2026-07-31 direkt gegen die API):

| SDK | Ergebnis |
|---|---|
| `streamlit` | existiert nicht mehr — API akzeptiert nur `gradio\|docker\|static` |
| `docker` | 402 Payment Required |
| `gradio` | 402 Payment Required |
| `static` | frei, aber nur HTML/JS — kein Python |

Wörtlich von HF: *„Static Spaces are free for everyone, but hosting Gradio and
Docker Spaces on free cpu-basic requires a PRO subscription."*

## Wenn Live-Inferenz doch sein soll

Sie braucht Llama-3.2-3B in bfloat16 (~6,4 GB) plus `multilingual-e5-base`
für die RAG-Baseline — mehr, als eine kostenlose Stufe hergibt. Machbare Wege:

* **HF PRO** (~9 $/Monat) → Docker-Space auf `cpu-basic` mit 16 GB. Ein
  passendes Dockerfile ist schnell geschrieben: `python:3.10-slim`, als uid
  1000 laufen, `HF_HOME` auf ein beschreibbares Verzeichnis, Streamlit mit
  `--server.address=0.0.0.0` und `app_port` im README-Frontmatter.
* **ZeroGPU** (ebenfalls PRO) wäre schneller, setzt aber Gradio statt
  Streamlit voraus — also ein Umbau der Oberfläche.

Der Inferenzcode dafür existiert unverändert in `goz_extract.inference` und
`goz_extract.retrieval`; die lokale `app.py` im Repo-Root benutzt ihn.
