# Design: GOZ-Code-Extraktion per LoRA-Finetuning vs. RAG-Baseline

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
- Die 5 `tests/fixtures/golden_single_v2/*synth_*.json`-Fixtures direkt als
  Testbeispiele (Notiz + erwartete Codes) sowie als Formatvorlage für die
  zusätzlichen selbst generierten synthetischen Notizen.

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

Multi-Label-Extraktion: ganze zahnärztliche Behandlungsnotiz (kann mehrere
Behandlungsschritte einer Sitzung beschreiben) → Menge der zutreffenden
GOZ-Ziffern aus einer festen Codeliste. Das ist die vereinfachte Version von
MAIKAs Kernaufgabe (Retrieval + Extraktion, ohne die Segmentierungs-Stufe,
die ~50 Validierungsregeln und die Ä1/Ä3/Ä5-Sonderbehandlung).

- **Label-Space:** ~40-60 Codes aus den Alltags-Kategorien Konservierende
  Leistungen, Chirurgische Leistungen, Allgemeine zahnärztliche Leistungen
  (aus den insgesamt 221 Codes in `goz_database_v4.json` — die volle Menge
  würde im Zeitbudget zu wenige Trainingsbeispiele pro seltener Klasse
  bedeuten, z.B. KFO/Schienen).
- **Trainingsdaten:** komplett synthetisch, per LLM-Generierungs-Skript
  erzeugt, im Format der vorhandenen `synth_*`-Fixtures (Notiz-Text +
  Liste der zutreffenden Codes), in drei Schwierigkeitsstufen:
  - leicht: Notiz nah an den amtlichen `bezeichnung`-Formulierungen der
    enthaltenen Codes
  - mittel: umgangssprachliche Zahnarzt-Notiz, andere Wortwahl,
    typischerweise 2-3 Codes pro Notiz
  - schwer: Abkürzungen/implizite Formulierung, teils Codes ohne
    offensichtliche Stichwort-Überlappung zur amtlichen Bezeichnung
  → ca. 300-500 synthetische Notizen. Die 5 echten `synth_*`-Fixtures
  fließen als hochwertige Beispiele ins Test-Set ein (nicht als Vorlage
  für ein separates Bonus-Set). Split auf Notiz-Ebene (nicht auf
  Code-Ebene), damit keine Notiz gleichzeitig in Train und Test landet.

## Modell & Training

- **Basismodell:** Llama 3.2 3B Instruct (gated auf Hugging Face, Meta-Lizenz
  muss akzeptiert werden — eingeplanter erster Schritt).
- **Methode:** QLoRA (4-bit) via Hugging Face + PEFT, Training auf
  Colab-T4-GPU.
- **Prompt-Format:** Instruktion ("Extrahiere alle zutreffenden GOZ-Ziffern
  aus der folgenden Behandlungsnotiz. Antworte als Liste von Ziffern aus der
  Codeliste: ...") + Notiz-Text → Liste von Ziffern als Output. LoRA-Gewichte
  enthalten das Domänenwissen; **kein Retrieval zur Inferenzzeit**.
- **Reproduzierbarkeit:** Generierungs-Skript, Trainings-Skript und
  Hyperparameter werden versioniert; Datengenerierung nutzt eine LLM-API
  (Anthropic, analog zum bestehenden `sql-agent`-Setup mit
  `init_chat_model`), dokumentiert im README.

## RAG-Baseline

Echtes Retrieval-Augmented-Generation, selbst gebaut, nur mit öffentlichen
Daten — bewusst als fairer Vergleich zum Finetuning aufgesetzt: **dasselbe
Basismodell** (Llama 3.2 3B Instruct, ohne LoRA), aber mit Domänenwissen im
Prompt statt in den Gewichten.

1. **Retrieval:** Kandidaten-Codes für die Notiz ermitteln —
   - BM25 über die amtlichen `bezeichnung`-Texte der 40-60 Codes
   - Embeddings: selbst berechnet mit einem offenen multilingualen Modell
     (`multilingual-e5-base` o.ä.) über dieselben `bezeichnung`-Texte —
     bewusst nicht MAIKAs Embeddings oder OpenAI-Embeddings, um IP-Fragen zu
     vermeiden und zu zeigen, dass die Baseline selbst gebaut statt
     übernommen wurde
   - Kombination via Reciprocal Rank Fusion (RRF), Top-N Kandidaten
     (N deutlich größer als die erwartete Codeanzahl pro Notiz, z.B. 10-15)
2. **Generation:** Notiz-Text + die Top-N Kandidaten (Code + amtliche
   `bezeichnung`) werden dem unveränderten Basismodell als Kontext gegeben;
   das Modell wählt daraus die zutreffenden Codes und gibt sie als Liste aus
   — gleiches Antwortformat wie beim Finetune, damit die Auswertung
   identisch läuft.

## Evaluation

- Pro Notiz: Precision, Recall und F1 über die vorhergesagte vs.
  tatsächliche Code-Menge, gemittelt über das Test-Set (Finetune vs.
  RAG-Baseline, gleicher Split, gleiches Antwortformat).
- Exact-Match-Rate (Anteil Notizen mit exakt korrekter Code-Menge) als
  strengere Zusatzmetrik.
- Analyse der häufigsten Fehlertypen (fälschlich hinzugefügte vs. fehlende
  Codes) pro Ansatz.
- Ergebnistabelle im README, im Stil von `sql-agent/evals/results.md`.

## Demo

Kleines Streamlit-Dashboard, im Stil von `sql-agent/src/app.py`:
Freitext-Eingabe einer Behandlungsnotiz → vorhergesagte Code-Liste von
Finetune und RAG-Baseline nebeneinander, inkl. der von der RAG-Baseline
abgerufenen Kandidaten (macht den Unterschied "Wissen im Prompt vs. Wissen
in den Gewichten" sichtbar — ähnlich wie die Guardrail-Badges bei
`sql-agent`).

## Out of Scope (bewusst weggelassen)

- Volle 221-Code-Abdeckung
- Die Segmentierungs-Stufe, die ~50 Validierungsregeln und die
  Ä1/Ä3/Ä5-Sonderlogik aus MAIKA (deterministische Nachbearbeitung, kein
  ML-Bestandteil — würde die PyTorch/Finetuning-Kernaussage verwässern)
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
