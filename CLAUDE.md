# goz-finetune-vs-rag — Projektkontext

Vollständiger Plan: `docs/superpowers/plans/2026-07-27-goz-finetune-vs-rag-implementation.md`
Design-Spec: `docs/superpowers/specs/2026-07-27-goz-finetune-vs-rag-design.md`

## Was das hier ist

Portfolio-Projekt von Marco Stang (Schwesterprojekt zu `sql-agent`). Ziel:
PyTorch/LoRA-Finetuning-Lücke im Lebenslauf schließen und die maika-Story
("Finetuning vs. RAG") mit einem echten Experiment untermauern.

Orientiert sich an einer Teilaufgabe von MAIKA (Notiz → GOZ-Codes), einem
Produktivsystem, an dem Marco bei ILI DIGITAL AG arbeitet — verwendet aber
**nur öffentliche GOZ-Daten** (amtliche Codeliste) und komplett neue,
selbst generierte Trainingsdaten. Siehe Design-Spec, Abschnitt
"Datenherkunft & IP-Abgrenzung", für die genaue Abgrenzung.

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
- `data/goz_codes.json` — kuratierte 55-Code-Label-Liste (Teilmenge der
  amtlichen GOZ-Codeliste)
- `notebooks/train_and_infer.ipynb` — Colab-only: LoRA-Training + Inferenz
  beider Ansätze (RAG-Baseline vs. LoRA-Finetune) auf Llama-3.2-3B-Instruct
- `app.py` — Streamlit-Demo (Root-Level, nicht unter `src/`)
- `tests/` — ein Testmodul pro Package-Modul, plus `test_dataset_integrity.py`
  und `test_smoke.py`

## Aktueller Stand

*Diesen Abschnitt aktuell halten. Für neue Sessions/Agenten: lies zuerst
`HANDOVER.md` im Repo-Root — die enthält den detaillierten Stand, bekannte
Fallstricke auf dieser Maschine und den genauen nächsten Schritt. Dieser
Abschnitt hier ist nur die Kurzfassung.*

- ✅ Alle 13 geplanten Tasks implementiert, einzeln review-clean, finale
  Whole-Branch-Review durchlaufen und deren Findings gefixt. In `master`
  gemergt, `pytest` grün (kein GPU/Colab nötig für die lokale Test-Suite).
- 🟡 **Colab-Trainingslauf läuft gerade / steht kurz bevor.** Der erste
  vollständige Durchlauf hatte zwei echte Qualitätsbugs (RAG-Baseline gab
  zu viele Kandidaten zurück, LoRA-Finetune kollabierte auf eine einzelne
  Vorhersage) — beide sind im Code gefixt (Completion-Only-Loss beim
  Training, schärfere Extraktions-Instruktion), aber **noch nicht mit
  einem echten Lauf verifiziert**. Details, exakte nächste Schritte und
  mehrere gelöste Colab-Umgebungsbugs (transformers-Versions-Inkompatibilität,
  torchao-Konflikt) stehen in `HANDOVER.md`.
- ⬜ Danach: `results/results.md` + README-Ergebnistabelle mit echten Zahlen
  befüllen, Streamlit-Demo (`app.py`) erstmals im Browser testen, IP-
  Disclosure-Frage zur Design-Spec klären, Repo auf GitHub pushen.
