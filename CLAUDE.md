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

- ⬜ Projekt-Grundgerüst
