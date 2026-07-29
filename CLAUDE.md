# goz-finetune-vs-rag — Projektkontext

Vollständiger Plan: `docs/superpowers/plans/2026-07-27-goz-finetune-vs-rag-implementation.md`
Design-Spec: `docs/superpowers/specs/2026-07-27-goz-finetune-vs-rag-design.md`

## Was das hier ist

Portfolio-Projekt von Marco Stang (Schwesterprojekt zu `sql-agent`). Ziel:
PyTorch/LoRA-Finetuning-Lücke im Lebenslauf schließen und die Frage
"Finetuning vs. RAG" mit einem echten Experiment beantworten.

Thematisch an einer Alltagsaufgabe aus dem zahnärztlichen Praxisbetrieb
orientiert (Notiz → GOZ-Abrechnungscode) — verwendet aber **nur öffentliche
GOZ-Daten** (amtliche Codeliste) und komplett neue, selbst generierte
Trainingsdaten. Siehe Design-Spec, Abschnitt "Datenherkunft & IP-Abgrenzung",
für die genaue Abgrenzung.

## Wie hier gearbeitet wird

- Doku/Kommentare/Antworten auf Deutsch, Marco lernt aktiv mit (gleicher
  Stil wie `sql-agent`).
- GPU-Schritte (Training, Modell-Inferenz) laufen auf Colab, nicht lokal —
  siehe `notebooks/train_and_infer.ipynb`.

## Commands

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
cp .env.example .env  # ANTHROPIC_API_KEY eintragen

.venv/Scripts/python.exe -m pytest tests/ -v                    # komplette Test-Suite, kein GPU/Colab nötig
.venv/Scripts/python.exe -m pytest tests/test_dataset.py::test_x # einzelner Test

.venv/Scripts/python.exe scripts/curate_codes.py    # 1. GOZ-Codes kuratieren
.venv/Scripts/python.exe scripts/generate_data.py   # 2. synthetische Trainingsdaten generieren
.venv/Scripts/python.exe scripts/build_dataset.py   # 3. Dataset zusammenbauen
.venv/Scripts/python.exe scripts/run_eval.py        # Evaluation laufen lassen

.venv/Scripts/python.exe -m streamlit run app.py    # Demo-App (braucht Adapter aus Colab unter adapters/)
```

Kein Linter konfiguriert.

## Architektur

- `src/goz_extract/` — installierbares Package: `curate.py`, `data_generation.py`,
  `dataset.py`, `evaluate.py`, `inference.py`, `prompting.py`, `report.py`,
  `retrieval.py`, `schema.py`
- `scripts/` — CLI-Einstiegspunkte, die die Package-Funktionen aufrufen
  (Curation → Datengenerierung → Dataset-Assembly → Eval)
- `data/goz_codes.json` — kuratierte Label-Liste (Teilmenge der amtlichen
  GOZ-Codeliste), auf ein 10-Code-Kernset reduziert (siehe README für die
  Begründung)
- `notebooks/train_and_infer.ipynb` — Colab-only: LoRA-Training + Inferenz
  beider Ansätze (RAG-Baseline vs. LoRA-Finetune) auf Llama-3.2-3B-Instruct
- `app.py` — Streamlit-Demo (Root-Level, nicht unter `src/`)
- `tests/` — ein Testmodul pro Package-Modul, plus `test_dataset_integrity.py`
  und `test_smoke.py`

## Aktueller Stand

*Diesen Abschnitt aktuell halten.*

- ✅ Alle geplanten Tasks implementiert, `pytest` grün (kein GPU/Colab nötig
  für die lokale Test-Suite).
- ✅ Colab-Trainingslauf erfolgreich abgeschlossen: LoRA-Finetune schlägt
  die RAG-Baseline (F1 0.59 vs. 0.48, Exact Match 0.38 vs. 0.07) — siehe
  README, Abschnitt "Ergebnisse".
- ✅ Streamlit-Demo (`app.py`) lokal im Browser getestet, läuft.
- ⬜ Repo noch nicht auf GitHub gepusht.
