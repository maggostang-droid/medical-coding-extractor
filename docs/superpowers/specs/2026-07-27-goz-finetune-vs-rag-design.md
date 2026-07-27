# Design: GOZ-Code-Klassifikation per LoRA-Finetuning vs. RAG-Baseline

**Datum:** 2026-07-27
**Repo:** `goz-finetune-vs-rag` (neu, `02_Portfolio/goz-finetune-vs-rag`)
**Status:** Design approved, bereit für Implementierungsplan

## Kontext

Portfolio-Projekt von Marco Stang für Bewerbungen. Ziel: die PyTorch/Finetuning-Lücke
im Lebenslauf schließen (Ergänzung zu `sql-agent`, das die SQL/LangGraph-Lücke
schließt) und die maika-Story im Interview mit einem konkreten Experiment
untermauern: "Ich habe geprüft, ob Finetuning die RAG-Pipeline schlägt."

Marco arbeitet bei ILI DIGITAL AG am Produktivsystem **MAIKA** (Notiz →
GOZ/GOÄ-Abrechnung, Hybrid-Retrieval + LLM-Extraktion + ~50 Validierungsregeln).
Dieses Projekt ist **kein** Export/Fork von MAIKA, sondern ein bewusst schlankes,
eigenständiges Portfolio-Projekt, das sich thematisch an einer Teilaufgabe von
MAIKA orientiert (Retrieval + Extraktion, stark vereinfacht) und komplett neue,
selbst erzeugte Daten verwendet. Zeitbudget: 1-2 Tage.

## Datenherkunft & IP-Abgrenzung (wichtig)

Aus dem MAIKA-Repo (`github.com/maggostang-droid/dentist`, lokal unter
`Downloads/dentist-main`) wird **ausschließlich** verwendet:

- `data/databases/goz_database_v4.json`, und zwar **nur** die Felder `goz_nr`
  und `bezeichnung` — die amtliche Gebührenordnung für Zahnärzte (GOZ) ist ein
  Rechtsverordnungstext, öffentlich und unproblematisch.
- Die 5 `tests/fixtures/golden_single_v2/*synth_*.json`-Fixtures als Vorlage/
  Formatbeispiel für zusätzliche selbst generierte Multi-Code-Testfälle (nicht
  als exakte Kopie, sondern als Format-Referenz).

**Ausdrücklich NICHT verwendet:** Liebold-Kommentare, `kommentar_kurz`,
`aliases`, `synonyms.json`, die MAIKA-Embeddings (`goz_embeddings_vault.json`/
`v4.json` — abgeleitet aus proprietär angereichertem Text, zusätzlich
Wettbewerbsvorteil von ILI DIGITAL), die `real_*`-Fixtures (echte
Praxisfälle, vertraulich), sowie jeglicher MAIKA-Anwendungscode
(`maika/`, `frontend/`).

Begründung: Rechtstext ist öffentlich, alles andere ist entweder urheberrechtlich
geschütztes Drittmaterial (Liebold), Geschäfts-IP (Retrieval-/Embedding-Setup)
oder potenziell vertrauliche/personenbezogene Praxisdaten (`real_*`-Fixtures).

## Aufgabe

Klassifikation: kurzer Text eines zahnärztlichen Behandlungsschritts →
passende GOZ-Ziffer (Single-Label, aus einer festen Codeliste).

- **Label-Space:** ~40-60 Codes aus den Alltags-Kategorien Konservierende
  Leistungen, Chirurgische Leistungen, Allgemeine zahnärztliche Leistungen
  (aus den insgesamt 221 Codes in `goz_database_v4.json` — die volle Menge
  würde im Zeitbudget zu wenige Trainingsbeispiele pro seltener Klasse
  bedeuten, z.B. KFO/Schienen).
- **Trainingsdaten:** komplett synthetisch, per LLM-Generierungs-Skript
  erzeugt, ~20-30 Beispielsätze pro Code in drei Schwierigkeitsstufen:
  - leicht: nah an der amtlichen `bezeichnung`-Formulierung
  - mittel: umgangssprachliche Zahnarzt-Notiz, andere Wortwahl
  - schwer: Abkürzungen/implizite Formulierung ohne offensichtliche
    Stichwort-Überlappung mit der amtlichen Bezeichnung
  → ca. 1000-1500 Beispiele gesamt, stratifizierter 80/20 Train/Test-Split
  nach Code.
- **Bonus-Testset:** die 5 vorhandenen `synth_*`-Fixtures plus ca. 15-20
  weitere, selbst generierte Multi-Code-Beispiele im gleichen Format (ganze
  Notiz → mehrere GOZ-Codes). Dient als qualitativer Realismus-Check
  (andere Struktur als das Haupt-Testset: Mehrfach-Codes pro Notiz statt
  Einzelschritt-Klassifikation), nicht Teil der Haupt-Metrik.

## Modell & Training

- **Basismodell:** Llama 3.2 3B Instruct (gated auf Hugging Face, Meta-Lizenz
  muss akzeptiert werden — eingeplanter erster Schritt).
- **Methode:** QLoRA (4-bit) via Hugging Face + PEFT, Training auf
  Colab-T4-GPU.
- **Prompt-Format:** Instruktion ("Klassifiziere den folgenden
  Behandlungsschritt mit der passenden GOZ-Ziffer aus der Liste: ...") +
  Behandlungsschritt-Text → Ziffer als Output.
- **Reproduzierbarkeit:** Generierungs-Skript, Trainings-Skript und
  Hyperparameter werden versioniert; Datengenerierung nutzt eine LLM-API
  (Anthropic, analog zum bestehenden `sql-agent`-Setup mit
  `init_chat_model`), dokumentiert im README.

## RAG-Baseline

Eigene, schlanke Nachbildung von MAIKAs Hybrid-Retrieval-Prinzip — **nicht**
MAIKAs Code oder Daten, sondern selbst gebaut, nur mit öffentlichen Daten:

- **BM25** über die amtlichen `bezeichnung`-Texte der 40-60 Codes.
- **Embeddings:** selbst berechnet mit einem offenen multilingualen Modell
  (`multilingual-e5-base` o.ä.) über dieselben `bezeichnung`-Texte — bewusst
  nicht MAIKAs Embeddings oder OpenAI-Embeddings, um IP-Fragen zu vermeiden
  und zu zeigen, dass die Baseline selbst gebaut statt übernommen wurde.
- **Kombination:** Reciprocal Rank Fusion (RRF) der beiden Rankings.
- **Baseline-Vorhersage:** Top-1-Treffer aus dem kombinierten Ranking.

## Evaluation

- Top-1- und Top-3-Accuracy von Finetune vs. RAG-Baseline auf dem
  Haupt-Testset (gleicher Split für beide, fairer Vergleich).
- Confusion Matrix / Analyse der häufigsten Verwechslungen (pro Ansatz).
- Ergebnistabelle im README, im Stil von `sql-agent/evals/results.md`.
- Qualitativer Abschnitt zu den Multi-Code-Bonus-Testfällen (kein
  Accuracy-Wert, da andere Aufgabenstruktur — stattdessen Beispiel-Outputs
  beider Ansätze gegenübergestellt).

## Demo

Kleines Streamlit-Dashboard, im Stil von `sql-agent/src/app.py`:
Freitext-Eingabe eines Behandlungsschritts → Vorhersage von Finetune und
RAG-Baseline nebeneinander, inkl. Konfidenz/Top-3 je Ansatz.

## Out of Scope (bewusst weggelassen)

- Volle 221-Code-Abdeckung
- Multi-Code-Extraktion als trainiertes Verhalten (nur als Bonus-Testset,
  nicht als Trainingsziel)
- Echte Patienten-/Praxisdaten
- Übernahme von MAIKA-Code, Liebold-Inhalten oder MAIKA-Embeddings
- Auth, Deployment, Multi-Turn-Memory (analog zu den bewusst weggelassenen
  Punkten bei `sql-agent`)

## Offene Punkte für den Implementierungsplan

- Genaue Auswahl der 40-60 Codes (Liste fixieren)
- Exakte Prompt-Templates für die Datengenerierung (drei Schwierigkeitsstufen)
- Colab-Notebook-Struktur vs. lokale Skripte (Training braucht GPU, Rest kann
  lokal laufen)
- Konkrete Hyperparameter (LoRA-Rank, Lernrate, Epochen) — Startwerte im
  Plan, ggf. iterieren
