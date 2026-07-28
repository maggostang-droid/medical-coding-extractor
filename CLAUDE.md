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

## Aktueller Stand

*Diesen Abschnitt aktuell halten, sobald ein Task aus dem Implementierungsplan
abgeschlossen ist.*

- ✅ Alle 13 geplanten Tasks implementiert und einzeln review-clean (Curation,
  Datengenerierung, Dataset-Assembly, Retrieval, Prompting, Evaluation,
  Inferenz-Loading, Colab-Notebook, Eval-Report, Streamlit-Demo, README).
  Zusätzlich eine finale Whole-Branch-Review durchlaufen und deren Findings
  gefixt (u.a. Golden-Fixtures außerhalb des 55-Code-Label-Space rausgefiltert,
  T4-taugliche dtypes, Speicher-Fix im Notebook und in `app.py`).
- ✅ `pytest` grün, kein GPU/Colab nötig für die lokale Test-Suite.
- ⬜ **Einziger noch offener Schritt:** `notebooks/train_and_infer.ipynb` auf
  Colab laufen lassen (Task 10-13 im Plan), um die echten Ergebnisse
  (RAG-Baseline vs. LoRA-Finetune) zu erzeugen und `results/results.md` +
  README-Ergebnistabelle zu befüllen.
