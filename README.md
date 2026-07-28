# GOZ-Code-Extraktion: LoRA-Finetuning vs. RAG-Baseline

Portfolio-Projekt: Ein LoRA-feingetuntes Llama-3.2-3B-Instruct extrahiert
GOZ-Ziffern aus zahnärztlichen Behandlungsnotizen — verglichen gegen eine
RAG-Baseline auf demselben, unveränderten Basismodell. Ziel: eine konkrete,
messbare Antwort auf "schlägt Finetuning RAG?".

## Aufgabe

Aus einer Behandlungsnotiz (kann mehrere Behandlungsschritte einer Sitzung
beschreiben) werden alle zutreffenden GOZ-Ziffern extrahiert (Multi-Label),
aus einem Label-Space von 55 Codes (Kategorien "Allgemeine zahnärztliche
Leistungen" + "Konservierende Leistungen" der amtlichen Gebührenordnung).

## Zwei Wege, Domänenwissen einzubringen

- **RAG-Baseline:** BM25 + Embeddings (`multilingual-e5-base`) liefern
  Kandidaten-Codes, dasselbe Basismodell wählt daraus per Prompt.
- **LoRA-Finetune:** Domänenwissen steckt in den LoRA-Gewichten, kein
  Retrieval zur Inferenzzeit.

## Ergebnisse

⏳ Training läuft noch nicht — siehe `notebooks/train_and_infer.ipynb`.
Tabelle wird nach dem Colab-Lauf ergänzt.

<!-- Tabelle aus results/results.md einfügen, sobald Task 10+11 gelaufen sind -->

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
cp .env.example .env  # ANTHROPIC_API_KEY eintragen
.venv/Scripts/python.exe -m pytest tests/ -v
```

Trainingsdaten generieren und Codeliste kuratieren: siehe
`scripts/curate_codes.py`, `scripts/generate_data.py`,
`scripts/build_dataset.py`.

Training + Inferenz laufen auf Colab: `notebooks/train_and_infer.ipynb`
(braucht HF-Zustimmung zur Llama-3.2-Lizenz).

Demo starten (braucht die von Colab heruntergeladenen Artefakte unter
`adapters/`): `.venv/Scripts/python.exe -m streamlit run app.py`

## Datenherkunft

Nur die amtliche GOZ-Codeliste (öffentliche Gebührenordnung) und komplett
selbst generierte, synthetische Trainingsdaten. Details siehe
`docs/superpowers/specs/2026-07-27-goz-finetune-vs-rag-design.md`,
Abschnitt "Datenherkunft & IP-Abgrenzung".

## Limitierungen

- Label-Space auf 55 Alltags-Codes begrenzt (nicht die vollen 221 GOZ-Codes)
- Trainingsdaten sind synthetisch (LLM-generiert), kein Abgleich mit realen
  Praxisfällen im großen Stil
- RAG-Baseline nutzt eine vereinfachte Retrieval-Pipeline ohne die
  Segmentierungs- und Validierungsschritte eines Produktivsystems
