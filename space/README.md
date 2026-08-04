# GOZ-Code-Extraktion: LoRA-Finetuning vs. RAG

> Deploy-Ziel ist **Streamlit Community Cloud**, nicht Hugging Face Spaces:
> HF hat das Streamlit-SDK abgeschafft und verlangt für Docker-/Gradio-Spaces
> auch auf `cpu-basic` ein PRO-Abo. Das frühere Space-Frontmatter steht in der
> Git-Historie dieser Datei, falls das je wieder in Frage kommt.
>
> Hauptdatei: `space/app.py` · Branch: `master`

Aus einer zahnärztlichen Behandlungsnotiz werden die passenden GOZ-Ziffern
extrahiert (Multi-Label, 10 Kern-Codes). Zwei Ansätze auf demselben
Basismodell (Llama-3.2-3B-Instruct):

* **RAG-Baseline** — BM25 + `multilingual-e5-base` liefern Kandidaten, das
  unveränderte Basismodell wählt daraus.
* **LoRA-Finetune** — Domänenwissen steckt in den Adaptergewichten, kein
  Nachschlagen zur Laufzeit.

| Ansatz | Precision | Recall | F1 | Exact Match |
|---|---|---|---|---|
| RAG-Baseline | 0,40 | 0,70 | 0,48 | 0,07 |
| LoRA-Finetune | 0,65 | 0,58 | 0,59 | 0,38 |

Gemessen über 81 unabhängige Testnotizen.

Die Demo zeigt standardmäßig **vorberechnete echte Ausgaben** dieses
Auswertungslaufs — sofort und ohne Modell im Speicher. Eigener Text löst
Live-Inferenz aus; auf der freien CPU-Stufe dauert das entsprechend.

Trainingsdaten sind vollständig synthetisch erzeugt, kein Abgleich mit realen
Praxisfällen.

Code: <https://github.com/marco-stang/medical-coding-extractor>
