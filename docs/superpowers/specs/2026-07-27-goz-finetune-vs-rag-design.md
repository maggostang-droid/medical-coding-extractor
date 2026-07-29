# Design: GOZ-Code-Extraktion per LoRA-Finetuning vs. RAG-Baseline

**Datum:** 2026-07-27
**Repo:** `goz-finetune-vs-rag` (neu, `02_Portfolio/goz-finetune-vs-rag`)
**Status:** Design approved, bereit für Implementierungsplan

## Kontext

Portfolio-Projekt von Marco Stang für Bewerbungen. Ziel: die PyTorch/Finetuning-Lücke
im Lebenslauf schließen (Ergänzung zu `sql-agent`, das die SQL/LangGraph-Lücke
schließt) und im Interview mit einem konkreten Experiment eine Frage
beantworten: "Schlägt Finetuning die RAG-Pipeline?"

Thematisch an einer Alltagsaufgabe aus dem zahnärztlichen Praxisbetrieb
orientiert (Notiz → GOZ/GOÄ-Abrechnung), wie sie in bestehenden
Produktivsystemen vorkommt — dieses Projekt ist aber ein bewusst schlankes,
eigenständiges Portfolio-Projekt (Retrieval + Extraktion, stark vereinfacht,
keine Segmentierungs-/Validierungsstufen) mit komplett neuen, selbst
erzeugten Daten. Zeitbudget: 1-2 Tage.

## Datenherkunft & IP-Abgrenzung (wichtig)

Aus einem Referenz-Repo eines bestehenden Produktivsystems wird
**ausschließlich** verwendet:

- `data/databases/goz_database_v4.json`, und zwar **nur** die Felder `goz_nr`
  und `bezeichnung` — die amtliche Gebührenordnung für Zahnärzte (GOZ) ist ein
  Rechtsverordnungstext, öffentlich und unproblematisch.
- 5 Golden-Fixture-Dateien direkt als Testbeispiele (Notiz + erwartete Codes)
  sowie als Formatvorlage für die zusätzlichen selbst generierten
  synthetischen Notizen.

**Ausdrücklich NICHT verwendet:** proprietäre Kommentar-/Alias-/Synonym-Daten
eines Drittanbieters, die dortigen Embeddings (abgeleitet aus proprietär
angereichertem Text, zusätzlich Wettbewerbsvorteil des Systembetreibers),
Fixtures mit echten Praxisfällen (vertraulich), sowie jeglicher
Anwendungscode des Referenzsystems.

Begründung: Rechtstext ist öffentlich, alles andere ist entweder urheberrechtlich
geschütztes Drittmaterial, Geschäfts-IP (Retrieval-/Embedding-Setup)
oder potenziell vertrauliche/personenbezogene Praxisdaten.

## Aufgabe

Multi-Label-Extraktion: ganze zahnärztliche Behandlungsnotiz (kann mehrere
Behandlungsschritte einer Sitzung beschreiben) → Menge der zutreffenden
GOZ-Ziffern aus einer festen Codeliste. Das ist eine deutlich vereinfachte
Version dessen, was ein Produktivsystem hier leisten müsste (Retrieval +
Extraktion, ohne Segmentierungs-Stufe, deterministische Validierungsregeln
oder Sonderfall-Behandlung einzelner Codes).

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
     bewusst keine proprietären oder OpenAI-Embeddings, um IP-Fragen zu
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
- Segmentierungs-Stufe und deterministische Post-Processing-/
  Validierungsregeln, wie sie ein Produktivsystem bräuchte (kein
  ML-Bestandteil — würde die PyTorch/Finetuning-Kernaussage verwässern)
- Echte Patienten-/Praxisdaten
- Übernahme von Fremdcode, proprietären Kommentar-Inhalten oder fremden
  Embeddings
- Auth, Deployment, Multi-Turn-Memory (analog zu den bewusst weggelassenen
  Punkten bei `sql-agent`)

## Offene Punkte für den Implementierungsplan

- Genaue Auswahl der 40-60 Codes (Liste fixieren)
- Exakte Prompt-Templates für die Datengenerierung (drei Schwierigkeitsstufen)
- Colab-Notebook-Struktur vs. lokale Skripte (Training braucht GPU, Rest kann
  lokal laufen)
- Konkrete Hyperparameter (LoRA-Rank, Lernrate, Epochen) — Startwerte im
  Plan, ggf. iterieren
