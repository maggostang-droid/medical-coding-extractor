# Handover — goz-finetune-vs-rag

**Stand:** 2026-07-29, nach Merge in `master` (aktuellsten Commit-Hash per
`git log -1` prüfen statt hier fest zu verdrahten - dieser Abschnitt wird
öfter geändert als der Hash nachgepflegt wird).

**Für wen:** Ein neuer Agent (oder du selbst in einer neuen Session), der hier
weitermacht, ohne den bisherigen Chatverlauf zu kennen.

Lies zuerst `CLAUDE.md` (Projektkontext, Arbeitsstil), dann diese Datei
(aktueller Stand, offene Baustellen, bekannte Fallstricke). Die Design-Spec
(`docs/superpowers/specs/2026-07-27-goz-finetune-vs-rag-design.md`) und der
Implementierungsplan (`docs/superpowers/plans/2026-07-27-goz-finetune-vs-rag-implementation.md`)
bleiben die Quelle für *warum* die Architektur so aussieht, wie sie aussieht.

**Hinweis zur Entstehung dieser Datei:** Ein Teil der unten beschriebenen
Fixes (Query-/Passage-Präfixe in `retrieval.py`, Sanity-Check-Zelle fürs
Training, `trl` aus `pyproject.toml` entfernt) kam von einem parallel
arbeitenden Agenten, nicht von der Session, die den Rest dieser Datei
geschrieben hat — beim Zusammenführen sind zwischenzeitlich zwei
Notebook-Zellen durch einen veralteten Zellen-Verweis kaputtgegangen
(Markdown-Zelle mit Code als Inhalt, altes Duplikat blieb stehen). Ist
repariert (Commit `c3f7467`), aber falls das Notebook nochmal komisch
aussieht: mit `python -c "import json; json.load(open(...))"` und einer
Zellen-Übersicht (cell_type + erste Zeile pro Zelle) gegenchecken, bevor du
weiter darauf aufbaust.

## Wo wir stehen

Alle 13 Tasks aus dem Implementierungsplan sind umgesetzt, einzeln reviewt
und in `master` gemergt. Anlauf 3 (manueller `Trainer` + Completion-Masking)
lief technisch fehlerfrei durch den Colab-GPU-Trainingslauf, inklusive
Sanity-Check-Bestätigung ("Completion-Masking verifiziert an 10
Stichproben"). **Trotzdem war das Ergebnis wieder unbrauchbar** — nur mit
einer anderen Diagnose als zuvor:

- **RAG-Baseline** (mit Cell-8/prompting-Fixes aus dem ersten Review):
  Precision 0.19, Recall 0.56, F1 0.26 — besser als der allererste Lauf
  (0.13/0.64/0.21), aber immer noch schwach (34% der Test-Notizen komplett
  falsch, nur 1% exakt getroffen).
- **LoRA-Finetune**: F1 0.05 — **kein Fortschritt ggü. dem allerersten,
  kaputten Lauf (F1 0.03)**. Root-Cause-Analyse (systematisches Debugging,
  siehe `superpowers:systematic-debugging`) ergab: **kein Bug mehr**. Die
  Trainings-Loss-Kurve aus dem echten Lauf (`trainer_state.json` im
  Adapter-Checkpoint) sah gesund aus (Loss 3.18→1.06, Token-Accuracy
  52%→82% über 63 Steps) — aber bei der freien Generierung kollabierte das
  Modell trotzdem auf eine fast konstante, inhaltsunabhängige Antwort
  (`0060, 2030` statt vorher `2300`, für 100% bzw. 58% der 82 Test-Notizen,
  unabhängig vom Inhalt). Das ist klassisches **Exposure Bias**: gute
  Teacher-Forced-Metriken bedeuten nicht, dass die freie Generierung
  robust ist, wenn das Lernsignal (63 Steps, LoRA nur auf
  Attention-Projektionen, 55 Klassen bei nur ~6 Beispielen/Klasse) zu
  schwach ist, um den Basis-Modell-Prior am ersten Antwort-Token zu
  überschreiben. Testdaten wurden dabei explizit auf Qualität geprüft
  (keine Train/Test-Überlappung, alle Test-Codes im Training vertreten,
  Stichproben manuell gegen die GOZ-Bezeichnungen verifiziert) — **die
  Daten waren nicht die Ursache.**

Auf Marcos Vorschlag hin wurde zusätzlich das MAIKA-Produktivsystem
(`C:\Users\Marco\Downloads\dentist-main\dentist-main`, externes
Referenz-Repo, nur Architektur-Vergleich — siehe IP-Abgrenzung in
`CLAUDE.md`) angeschaut, um zu verstehen, warum es dort deutlich besser
funktioniert. Kernunterschiede (konzeptionell, nicht 1:1 übernommen):
großes Closed-Source-Modell statt 3B-Llama, eine 9-stufige Pipeline mit 3
getrennten LLM-Calls statt einem einzigen Prompt, Segmentierung vor
Retrieval, strukturierter Output statt Prosa-Regex, ein
Post-Processing-Layer mit ~50 deterministischen Regeln, und **kein
Finetuning** (RAG + starkes Modell + Regel-Nachbearbeitung statt LoRA auf
wenigen hundert Beispielen).

## Anlauf 4: Architektur-Überarbeitung (2026-07-28)

Statt eines vierten blinden Fix-Versuchs am Training (3 gescheiterte
Anläufe an derselben Pipeline-Stelle sind laut Debugging-Prozess ein
Signal, die Architektur zu hinterfragen statt weiter zu patchen) wurden
vier von MAIKAs Techniken adaptiert (konzeptionell, ohne proprietären Code
zu übernehmen) plus eine gezielte Reduktion des Label-Space:

1. **Code-Space auf 10 Kern-Codes reduziert** (von 55): `data/goz_codes.json`
   enthält jetzt nur noch die 10 häufigsten Codes aus den bisherigen
   Trainingsdaten (`0065, 0070, 0090, 0100, 2030, 2040, 2060, 2290, 2350,
   2410`). Grund: bei 329 Beispielen auf 55 Klassen kamen nur ~6
   Beispiele/Klasse zusammen - zu wenig für ein 63-Step-LoRA-Finetune.
   `goz_extract.dataset.restrict_to_valid_codes()` (neu) **projiziert**
   `expected_codes` jedes generierten Beispiels auf den reduzierten Space
   (behält Teilmengen-Codes, verwirft nur bei leerer Schnittmenge) statt
   Beispiele komplett zu verwerfen, sobald ein einzelner Code außerhalb
   liegt (das hätte bei striktem Verwerfen nur 68 statt 156 Beispiele
   übrig gelassen). `scripts/build_dataset.py` wendet das jetzt auch auf
   `generated`-Notizen an (vorher nur auf `golden_synth`-Fixtures - war
   eine echte Lücke, jetzt konsistent). Neu gebaut: **125 Train- / 31
   Test-Notizen**, ~18 Beispiele/Klasse (vorher ~6) - dreifache Dichte.
   Bekannter Nebeneffekt: bei nur 31 Test-Notizen kommt Code `2030` durch
   den Zufalls-Split gerade in keinem einzigen Testbeispiel vor (Split ist
   nicht nach Code stratifiziert) - Recall für `2030` ist also nicht
   messbar, nur Precision. Nicht behoben (würde einen stratifizierten
   Multi-Label-Splitter brauchen, aufwändiger als für dieses Experiment
   gerechtfertigt), aber wissen, dass es da ist.
2. **Strukturierter JSON-Output statt Prosa+Regex** (`prompting.py`):
   Instruktion verlangt jetzt ein JSON-Array (`["1234", "5678"]`) statt
   einer kommagetrennten Liste. `parse_code_list_response` parst das Array
   strikt (nur `[...]`-Block, valide JSON-Liste aus Strings) und fällt nur
   zurück auf die alte Prosa-Regex, wenn das kein valides Array ergibt
   (lokale Modelle wie Llama-3.2-3B haben keine echte
   Grammar-Constrained-Decoding-Garantie wie MAIKAs
   `response_format`-Structured-Output über die OpenAI-API). Neue Funktion
   `restrict_to_candidates()`: im RAG-Modus werden Vorhersagen zusätzlich
   auf die tatsächlich angebotenen Kandidaten beschränkt (alles andere ist
   per Definition eine Fehlvorhersage) - angelehnt an MAIKAs
   Post-Processing-Prinzip. **Trainings-Completion-Format im Notebook
   (Abschnitt 3, `to_chat_example`) UND die Sanity-Check-Zelle wurden
   entsprechend von `", ".join(...)` auf `json.dumps(...)` umgestellt** -
   beide müssen zusammenpassen, sonst schlägt die Sanity-Check-Assertion
   fehl.
3. **Segmentierung vor Retrieval** (`retrieval.py`, neue Funktion
   `segment_note()`): Notiz wird grob per Satzzeichen in Behandlungsschritte
   gesplittet, BM25+Embedding-Retrieval läuft pro Segment statt einmal über
   die ganze Notiz, Ergebnisse werden per RRF fusioniert. MAIKA macht das
   über einen eigenen LLM-Call (Segmentierungs-Modell) - hier bewusst eine
   simple Regel statt eines dritten Modell-Calls, um die Architektur nicht
   unnötig zu verkomplizieren.

Alle vier Punkte sind lokal implementiert und getestet (`pytest`: 52
passed, 1 skipped, u.a. neue Tests `test_restrict_to_valid_codes_*`,
`test_parse_code_list_response_ignores_codes_mentioned_outside_the_json_array`,
`test_restrict_to_candidates_*`, `test_segment_note_*`). `goz-extract-src.zip`
ist neu gebaut mit reduzierter Codeliste + neuen Train/Test-Daten + allen
Code-Änderungen.

**Wichtig: `adapters/goz-extract-llama32-3b/` (lokaler Adapter) und alle
Dateien in `results/` sind jetzt komplett veraltet** - trainiert/generiert
auf dem alten 55-Code-Space mit dem alten Komma-Format. Beide sind
inhaltlich inkompatibel mit dem neuen Code (nicht nur "andere Zahlen",
sondern anderes Label-Space und anderes Antwortformat) und werden beim
nächsten Colab-Lauf komplett überschrieben.

### Anlauf-4-Ergebnisse (nach dem kompletten Neustart) und was daraus folgt

Der komplette Neustart lief durch (24 Trainings-Steps statt 63 - das
Dataset ist durch die Filterung auf 125 statt 329 Beispiele geschrumpft,
siehe unten warum das wichtig ist). Neue Zahlen:

| Ansatz | Precision | Recall | F1 | Exact Match |
|---|---|---|---|---|
| RAG-Baseline | 0.16 | 0.66 | 0.24 | 0.03 |
| LoRA-Finetune | 0.06 | 0.06 | 0.06 | 0.06 |

**LoRA-Finetune kollabiert immer noch** - jetzt auf `2060` (17/31 Notizen,
55%) und `2030` (13/31, 42%), pro Notiz wird IMMER genau 1 Code
vorhergesagt, komplett unabhängig vom Inhalt (z.B. `2060` für eine
Wurzelkanal-Notiz, eine Brückenanker-Notiz und einen Vitalitätstest
gleichermaßen). Root Cause diesmal eindeutig: `trainer_state.json` im
neuen Checkpoint zeigt nur **24 Gradienten-Schritte** (vs. 63 vorher) -
die Reduktion auf 10 Codes hat zwar die Beispiele/Klasse verdreifacht
(~18 statt ~6), aber weil wir bewusst nur gefiltert statt neu generiert
haben (Marcos Entscheidung damals, um Zeit/API-Calls zu sparen), ist die
**absolute Datenmenge von 329 auf 125 geschrumpft** - bei gleicher
`num_train_epochs=3` heißt das automatisch weniger Gesamt-Schritte, nicht
mehr. Der Dichte-Gewinn pro Klasse wurde vom Verlust an Gesamt-Trainingszeit
mehr als aufgefressen.

**Zusätzlicher, unabhängiger Fund beim Auswerten der RAG-Baseline:**
`retrieve_candidates(..., top_n=12)` war seit der Code-Reduktion ein
No-Op - es gibt nur noch 10 Codes total, `top_n=12` liefert also *immer
alle 10* als "Kandidaten" zurück. Retrieval hat de facto gar nichts mehr
eingegrenzt, und der Kandidaten-Validierungs-Layer aus Anlauf 4 konnte
dadurch auch nichts rausfiltern (alles war ja "Kandidat"). Erklärt, warum
Precision trotz aller Fixes nicht besser wurde (⌀ 5,7 von 10 Codes
zurückgegeben - mehr als die Hälfte des gesamten Label-Space pro Notiz).
**Gefixt:** `top_n=12` → `top_n=5` in Notebook-Zelle 9 und `app.py`
(Commit ausstehend). `goz-extract-src.zip` bereits neu gebaut.

**Empfehlung für den nächsten Schritt (Diskussion mit Marco nötig, kein
weiterer blinder Fix):** Die naheliegendste Erklärung ist jetzt "zu wenig
absolute Trainingsschritte", nicht mehr ein Pipeline-Bug. Marco hat sich für
**beide** Hebel gleichzeitig entschieden: mehr Daten UND mehr Epochen.

## Anlauf 5: Mehr Daten + mehr Trainingsschritte (2026-07-28)

1. **Zusätzliche synthetische Daten generiert**, gezielt nur für die 10
   Kern-Codes (nicht für die ursprüngliche 55er-Liste): `scripts/generate_data.py
   --codes data/goz_codes.json --out data/synthetic_notes_core10.jsonl
   --per-code 20` → 250 neue Notizen, alle 250 bereits vollständig
   innerhalb der 10-Code-Liste (keine Projektion/Verwerfung nötig, anders
   als beim alten 410er-Korpus, der gegen die volle 55er-Liste generiert
   wurde). `data/synthetic_notes_combined.jsonl` = altes 410er-Korpus (nach
   Projektion, siehe Anlauf 4) + neue 250 Notizen, in `build_dataset.py`
   als `--generated` reingegeben.
2. **Neu gebautes Dataset: 325 Train- / 81 Test-Notizen** (vorher 125/31),
   **~54 Beispiele/Klasse im Training** (vorher ~18, ursprünglich ~6) -
   fast verzehnfacht ggü. dem allerersten 55-Code-Lauf. Alle 10 Codes sind
   jetzt auch im Test-Set vertreten (die `2030`-Recall-Lücke aus Anlauf 4
   ist damit ebenfalls behoben, war reiner Zufalls-Split-Effekt bei kleiner
   Menge).
3. **`num_train_epochs` von 3 auf 10 erhöht** (Notebook Abschnitt 3,
   `TrainingArguments`) - bei 325 Beispielen, effektiver Batchgröße 16:
   ~20 Steps/Epoche × 10 = **~200 Steps** (vorher 63, davor 24). Dazu
   `save_total_limit=1` ergänzt: ohne das wären bei 10 Epochen 10
   Checkpoints à ~100MB (Adapter + Optimizer-State) im Adapter-Ordner
   gelandet (~1GB) - jetzt behält der Trainer automatisch nur den
   neuesten.

`goz-extract-src.zip` neu gebaut mit den größeren Train/Test-Dateien,
`pytest` weiterhin grün (52 passed, 1 skipped). **Kompletter Neustart in
Colab nötig** (wie bei Anlauf 4) - Notebook + `goz-extract-src.zip` neu
hochladen, von oben durchlaufen. **Erwartete Ausgabe bei Zelle 5:**
`10 325 81` (nicht mehr `10 125 31`). Der Trainingslauf dauert jetzt länger
(mehr Epochen), aber bei nur 325 Beispielen auf einer T4 immer noch im
Minutenbereich, nicht Stunden.

**Falls das Finetune danach immer noch kollabiert:** dann ist "zu wenig
Daten/Steps" widerlegt, und die verbleibenden Hebel sind LoRA-Kapazität
(`target_modules` um `gate_proj`/`up_proj`/`down_proj` erweitern, `r`
erhöhen) oder die Lernrate. Das wäre dann tatsächlich ein neuer,
unverbrauchter Hypothesentest, kein vierter Blindversuch am selben Punkt.

## Code-Beschreibungen angereichert (2026-07-28)

Marco fragte, ob wir die Code-Beschreibungen verbessern bzw. von MAIKA
übernehmen könnten. **Bewusst NICHT von MAIKA übernommen** — MAIKAs
reichhaltigere Code-Daten ("Liebold v6": Klinisch/Einheit/Nicht-neben/
Ausschluss-Codes) stammen aus einem kommerziellen, urheberrechtlich
geschützten GOZ-Kommentarwerk (Liebold/Raff/Wahl) — das ist nicht mal
MAIKAs eigenes IP, sondern das eines Drittverlags, und dürfte unabhängig
von der Arbeitgeber-Abgrenzung nicht in dieses Repo übernommen werden.
MAIKAs eigene Zusatzregeln sind zudem proprietäre ILI-DIGITAL-Geschäftslogik
(dieselbe Grenze wie in der Design-Spec).

**Stattdessen:** neues optionales Feld `erweiterte_beschreibung` auf
`GozCode` (`schema.py`), befüllt über einen neuen, eigenständigen
Claude-Call (`goz_extract.enrichment` + `scripts/enrich_codes.py`), der
**nur aus allgemeinem GOZ-Fachwissen** eine Klartext-Erklärung plus
gängige umgangssprachliche Begriffe/Abkürzungen pro Code generiert - kein
Bezug zu MAIKA oder Liebold, gleiches Muster wie die
Trainingsdaten-Generierung. `GozCode.format_for_prompt()` (neu, in
`schema.py`) hängt die erweiterte Beschreibung an, wenn vorhanden - genutzt
sowohl in `prompting.py` (RAG-Kandidatenliste im Prompt) als auch in
`data_generation.py` (Codeliste beim Generieren synthetischer Notizen).

`data/goz_codes.json` ist jetzt für alle 10 Kern-Codes angereichert (echter
API-Call bereits gelaufen, siehe Datei). `goz-extract-src.zip` neu gebaut.
`pytest`: 60 passed, 1 skipped (8 neue Tests: `test_enrichment.py` komplett
neu, plus `test_goz_code_format_for_prompt_*` in `test_schema.py`).

**Für den nächsten Colab-Lauf ändert sich nichts an den Instruktionen** -
die RAG-Baseline-Zelle nutzt automatisch die angereicherten Beschreibungen,
weil `code_by_nr`/`candidates` direkt aus `data/goz_codes.json` (im neuen
Zip) gebaut werden. Falls demnächst nochmal synthetische Daten generiert
werden (`generate_data.py`), profitiert auch das automatisch von den
saubereren Codebeschreibungen im Prompt.

## Retrieval + Trainingsdaten nochmal RAG-freundlicher gemacht (2026-07-29)

Marco fragte, ob wir es dem RAG noch etwas leichter machen können, indem
Begriffe aus Bezeichnung/erweiterter Beschreibung auch in den generierten
Notizen (`synthetic_notes_core10.jsonl`) auftauchen. Zwei zusammenhängende
Fixes:

1. **`BM25Index`/`EmbeddingIndex` indexieren jetzt `format_for_prompt()`**
   statt nur `bezeichnung` (`retrieval.py`) - die umgangssprachlichen
   Begriffe aus `erweiterte_beschreibung` (z.B. "Kofferdam" für 2040,
   "WK-Aufbereitung" für 2410) waren vorher komplett ungenutzt, obwohl sie
   schon seit dem Enrichment-Schritt in `data/goz_codes.json` liegen -
   BM25 konnte nur exakte Amtsbegriffe matchen. Neuer Test:
   `test_bm25_matches_umgangssprachliche_begriffe_from_erweiterte_beschreibung`.
   Stichprobe bestätigt: `"Kofferdam angelegt"` rankt jetzt `2040` an
   Position 1 (vorher hätte nur "Spanngummi" gematcht).
2. **`build_generation_prompt`** (`data_generation.py`) verlangt jetzt
   explizit mindestens einen Begriff aus Bezeichnung/Umgangssprachlich-Liste
   pro erwähntem Code, **auch bei Schwierigkeitsgrad "schwer"** (der Rest
   der Notiz bleibt dort weiterhin implizit/abgekürzt - nur nicht mehr
   komplett ohne Anker-Begriff). Das ist keine künstliche Vereinfachung:
   die Umgangssprachlich-Begriffe sind reale Fachausdrücke, die Zahnärzte
   tatsächlich in Notizen verwenden.

`data/synthetic_notes_core10.jsonl` mit dem neuen Prompt **neu generiert**
(gleicher Befehl, `--per-code 20`, 250 Notizen, alle weiterhin vollständig
im 10-Code-Space) und mit dem alten 410er-Korpus zu
`data/synthetic_notes_combined.jsonl` neu kombiniert. **Dataset-Größe
unverändert (325 Train / 81 Test)** - nur der Wortschatz in den neuen 250
core10-Notizen ist nutzbarer fürs Retrieval. `goz-extract-src.zip` neu
gebaut, `pytest`: 62 passed, 1 skipped (2 neue Tests).

**Für den nächsten Colab-Lauf:** keine Änderung an der Anleitung nötig -
alles läuft automatisch mit den neuen Daten/dem neuen Retrieval-Index,
sobald das neu gebaute `goz-extract-src.zip` hochgeladen wird.

## Sofort-nächster Schritt

**Kompletter Neustart in Colab nötig, kein Live-Patch der laufenden
Session** - Trainingsdaten, Epochenzahl und Save-Strategie haben sich seit
dem letzten Lauf (Anlauf 4) nochmal geändert (siehe "Anlauf 5" oben):

1. Laufzeit trennen und löschen, neue T4-GPU-Laufzeit verbinden.
2. `notebooks/train_and_infer.ipynb` frisch hochladen (alle Fixes bereits
   committed) + `goz-extract-src.zip` aus dem Repo-Root bei der
   Upload-Zelle hochladen (bereits mit den 325/81 Train-/Test-Daten neu
   gebaut).
3. Von oben durchlaufen. **Erwartete Ausgabe bei Zelle 5:** `10 325 81`
   (nicht mehr `10 125 31`).
4. **Bei der Sanity-Check-Zelle in Abschnitt 3 genau hinschauen** - falls
   die Assertion fehlschlägt, nicht überspringen (siehe Markdown-Zelle
   direkt danach).
5. Training dauert jetzt länger (10 statt 3 Epochen, ~200 statt ~24-63
   Steps) - bei nur 325 Beispielen auf einer T4 aber immer noch im
   Minutenbereich.
6. Nach dem kompletten Lauf: Ergebnisse herunterladen, lokal nach
   `results/`/`adapters/` legen (alte Dateien überschreiben - sind ohnehin
   veraltet), `scripts/run_eval.py --results-dir results/` laufen lassen.

### Was zu erwarten ist / wie du die neuen Zahlen einordnest

- **Das ist jetzt der erste Lauf, bei dem "zu wenig Daten/Steps" als
  Erklärung für einen weiteren Mode Collapse tatsächlich ausscheiden
  würde** - ~54 Beispiele/Klasse (vorher ~18, davor ~6) und ~200 Steps
  (vorher ~24, davor ~63 auf dem 55-Code-Space). Falls das Finetune
  *immer noch* auf 1-2 konstante Codes kollabiert: nicht nochmal an Daten
  oder Epochenzahl drehen, sondern LoRA-Kapazität (`target_modules` um
  `gate_proj`/`up_proj`/`down_proj` erweitern, `r` erhöhen) oder
  Lernrate als nächste, bisher unverbrauchte Hypothese testen.
- **RAG-Baseline sollte durch den `top_n=5`-Fix** (Retrieval grenzt jetzt
  tatsächlich ein statt immer alle 10 Codes zu zeigen) **spürbar
  präziser werden.** Falls nicht: rohen, ungeparsten Modell-Output
  ausgeben lassen (`print(generated)` vor dem
  `parse_code_list_response(...)`-Aufruf in `generate_codes`,
  `src/goz_extract/inference.py`).
- Mit 81 Test-Notizen (10-15 pro Code) ist die Eval-Varianz deutlich
  geringer als bei den vorherigen 31 - Zahlen sind diesmal aussagekräftiger.

## Kritischer Code-Review, Runde 1 (2026-07-28)

*Historisch — das war die Review-Runde vor Anlauf 4 oben. Diese Fixes sind
weiterhin gültig und im Code, aber die Ergebnis-Einordnung hier ist
überholt (siehe "Wo wir stehen" oben für den aktuellen Stand).*

Auf Marcos Bitte hin nochmal das ganze Projekt kritisch durchgegangen,
Schwerpunkt `notebooks/train_and_infer.ipynb`. Sechs Fixes umgesetzt,
`pytest` bleibt grün (42 passed, 1 skipped):

1. **Sanity-Check-Zelle fürs Completion-Masking** (neue Zelle in
   Notebook-Abschnitt 3, direkt nach der Tokenisierung): `prompt_len =
   min(len(prompt_ids), len(full_ids))` in `tokenize_with_completion_mask`
   geht davon aus, dass separates Tokenisieren des abgeschnittenen
   Prompt-Strings exakt ein Präfix der Tokenisierung des vollständigen
   Strings ergibt — bei BPE-Tokenizern an Konkatenationsgrenzen nicht
   garantiert. War bis jetzt an keiner Stelle verifiziert. Die neue Zelle
   decodiert für 10 Zufallsbeispiele die nicht-maskierten Labels und
   assert-t, dass sie exakt `", ".join(expected_codes)` entsprechen —
   bricht mit klarer Meldung ab, statt einen Trainingslauf mit falscher
   Maskierungsgrenze (= derselbe Mechanismus wie der ursprüngliche Mode
   Collapse) unbemerkt durchlaufen zu lassen. **Siehe "Sofort-nächster
   Schritt" oben — diese Zelle lief noch nicht gegen den bereits
   trainierten Adapter.**
2. **`prompting.py`, `_INSTRUCTION`**: ergänzt um "Erwähne keine anderen
   Ziffern aus der Kandidatenliste, auch nicht um zu begründen, warum sie
   nicht zutreffen." `parse_code_list_response` sammelt jede 3-4-stellige
   Zahl aus dem gesamten generierten Text ein (bewusst so, siehe
   `test_parse_code_list_response_extracts_from_prose`) — wenn das Modell
   die Kandidatenliste kommentiert statt nur die Auswahl zu nennen, zählt
   jede erwähnte Ziffer als Vorhersage. Passt auffällig gut zur
   beobachteten Baseline-Überprediction (9 von 12 Kandidaten). Der
   bisherige Fix (`b030681`) adressierte nur "keine Wiederholung der
   Notiz", nicht dieses Muster.
3. **`retrieval.py`, `EmbeddingIndex`**: nimmt jetzt optional
   `encode_query_fn` getrennt von `encode_fn` entgegen. Vorher wurde
   dieselbe Funktion (mit `"passage: "`-Präfix) für Korpus **und** Query
   benutzt — `intfloat/multilingual-e5-base` ist aber asymmetrisch
   trainiert (`"query: "` für Suchanfragen, `"passage: "` für Korpus-Texte);
   beide Seiten gleich zu präfixieren verletzt diese Konvention und
   verschlechtert die Embedding-Hälfte der RRF-Fusion. Notebook-Abschnitt 2
   und `app.py` nutzen jetzt `encode_passages`/`encode_query` getrennt.
   Neuer Test: `test_embedding_index_uses_separate_query_encoder_when_given`.
4. **Notebook Abschnitt 3, `collate_fn`**: `pad_id = tokenizer.pad_token_id
   or tokenizer.eos_token_id` → `... if tokenizer.pad_token_id is not None
   else ...`. Latenter Fallstrick, falls `pad_token_id` je `0` wäre (falsy
   in Python); aktuell unkritisch, da Llama-3-Tokenizer standardmäßig kein
   Pad-Token hat.
5. **`pyproject.toml`**: `trl`-Dependency entfernt — seit Anlauf 3
   (`7bc7cd7`, plain `Trainer` statt `SFTTrainer`) importiert nirgendwo im
   Code mehr etwas aus `trl`.
6. Validiert (kein Fund, nur Gegencheck): "2300" (das Mode-Collapse-Ziel im
   ersten Lauf) kommt nur 4 von 329 Mal in `data/train.jsonl` vor — spricht
   gegen Label-Imbalance als Ursache und für die bestehende Diagnose
   (Loss dominiert vom sich wiederholenden Prompt-Teil).

## Bekannte Stolperfalle beim Artefakte-Download

Der `adapter.zip` (gebaut via `!zip -r adapter.zip
adapters/goz-extract-llama32-3b`) entpackt sich relativ zum aktuellen
Ordner — wenn man ihn in einen bereits existierenden lokalen `adapters/`-
Ordner entpackt, landet er doppelt verschachtelt
(`adapters/adapters/goz-extract-llama32-3b/...`). Passiert ist das schon
einmal, gefixt durch `mv adapters/adapters/goz-extract-llama32-3b
adapters/` + `rmdir adapters/adapters`. Nach dem Entpacken kurz prüfen,
dass `adapters/goz-extract-llama32-3b/adapter_config.json` direkt (nicht
noch eine Ebene tiefer) existiert, bevor `run_eval.py`/`app.py` es laden.

## Andere im Colab-Debugging gefundene und gefixte Bugs

Diese sind bereits stabil im Code und sollten beim neuen Lauf nicht mehr
auftreten:

- **`generate_codes` in `src/goz_extract/inference.py`** (Commit `68d7e2d`):
  `tokenizer.apply_chat_template(..., return_tensors="pt")` lieferte je
  nach `transformers`-Version entweder einen reinen Tensor oder ein
  `BatchEncoding`-Objekt zurück — crashte `model.generate()` auf Colab mit
  einem kryptischen `KeyError: 'shape'`/`AttributeError`. Fix: über
  `tokenizer(text, return_tensors="pt")` gehen (liefert immer
  `BatchEncoding`), mit `**inputs` an `generate()` übergeben. Versions-
  unabhängig, sollte halten.
- **`peft`/`torchao`-Versionskonflikt** (Commit `476ac36`): Colabs
  vorinstalliertes `torchao` (0.10.0) ist zu alt für die von `!pip install
  trl` gezogene `peft`-Version (verlangt >0.16.0), `PeftModel.from_pretrained(...)`
  crasht sonst mit `ImportError`. Trat zweimal auf verschiedenen Runtimes
  auf — damit kein Rate-Pin mehr, sondern ein bestätigt reproduzierbarer
  Fix. `!pip install -q -U torchao` ist jetzt fest in Zelle 1 (Setup).
- **`notebook_login()`**: nimmt den Token NICHT als Argument entgegen
  (`notebook_login("hf_...")` wirft `TypeError`) — Zelle ohne Argument
  aufrufen, Token ins dann erscheinende Eingabefeld einfügen.
- **`device_map="auto"`-Offloading kollidiert mit `peft`s Adapter-Ladelogik**
  (Commit `c3f7467`): Nach erfolgreichem Training crashte das Laden von
  `finetuned_model` mit `KeyError:
  'base_model.model.model.model.layers.20.input_layernorm'` — ein
  `"model."`-Segment zu viel im Pfad, ausgelöst durch
  `peft.peft_model.PeftModel._update_offload`, das nur greift, wenn
  `device_map="auto"` tatsächlich Layer auf CPU/Platte auslagert (wenig
  freier VRAM zum Ladezeitpunkt, z.B. weil `base_model`/`train_model` vom
  vorherigen Schritt noch nicht vollständig freigegeben waren). Fix:
  `load_model()` hat jetzt einen `device_map`-Parameter (Default weiterhin
  `"auto"`); RAG-Baseline- und Finetune-Inferenz-Zellen im Notebook rufen
  ihn jetzt mit `device_map={"": 0}` auf (alles auf eine GPU, kein
  Offload) — Llama 3.2 3B passt in fp16 (~6GB) komfortabel auf eine T4
  ohne Auslagerung, wenn vorher aufgeräumt wurde.

## Bewusst nicht gefixt (Entscheidung, kein Vergessen)

- **Keine Versions-Pins im `!pip install`** in Zelle 1 des Notebooks. Der
  finale Review hatte das als Empfehlung genannt; ich habe bewusst
  dagegen entschieden, weil die auf Colab tatsächlich installierten
  Versionen (`transformers 5.14.1`, `torch 2.13.0`) deutlich neuer sind als
  alles, was zuverlässig bekannt ist (dieses Projekt läuft in einer Zukunft
  jenseits des Wissensstands) — ein geratener Pin hätte eher neue
  Inkompatibilitäten riskiert als welche gelöst. Das gilt weiterhin für
  `transformers`/`trl`/`torch` selbst; der `torchao`-Fix ist davon eine
  bewusste Ausnahme, weil er kein Rate-Pin mehr ist, sondern zweimal live
  reproduziert und bestätigt wurde (siehe oben, jetzt in Zelle 1
  eingebaut).

(Das `device_map="auto"`-Offloading-Problem, ursprünglich hier als "nicht
systematisch, nicht gefixt" vermerkt, ist inzwischen doch als echter,
reproduzierbarer Bug bestätigt und gefixt — siehe "Andere im
Colab-Debugging gefundene und gefixte Bugs" oben.)

## Lokales Setup — wichtige Fallstricke auf dieser Maschine

- **Voller `pip install -e ".[dev]"` (mit torch/transformers/peft/trl/
  sentence-transformers) hängt bzw. ist sehr langsam auf dieser
  Windows+OneDrive-synchronisierten Maschine** — ist im Rahmen dieser
  Session zweimal minutenlang hängengeblieben (mutmaßlich OneDrive-
  Dateisperren oder Antivirus-Scan bei den vielen kleinen Dateien). Bei
  einem Hänger: Prozess über `tasklist`/PowerShell `Get-Process python`
  identifizieren (nicht die MSYS-`ps`-PIDs verwenden, die stimmen mit den
  echten Windows-PIDs nicht überein), mit `Stop-Process -Force` killen,
  `.venv` löschen, neu versuchen — beim zweiten/dritten Versuch lief es
  bisher immer durch.
- **Für alles, was KEIN Torch/Transformers braucht** (Codes kuratieren,
  Daten generieren, Dataset bauen, Eval-Report), gibt es lokal ein
  schlankes venv: `.venv-data` (Python 3.13, nur `pydantic langchain
  langchain-anthropic python-dotenv rank-bm25 pytest` + `pip install
  --no-deps -e .`). Das ist deutlich schneller/zuverlässiger als das volle
  `pyproject.toml`-Set und reicht für `scripts/curate_codes.py`,
  `scripts/generate_data.py`, `scripts/build_dataset.py`,
  `scripts/run_eval.py` und die komplette lokale Test-Suite (kein Test
  braucht ein echtes Modell außer dem `RUN_MODEL_TESTS`-geskippten).
- **`.env` und die generierten `data/*.jsonl`-Dateien sind gitignored** und
  lebten ursprünglich nur im (inzwischen gelöschten) SDD-Worktree — sie
  wurden nach dem Merge neu generiert (`.env`: Anthropic-Key aus
  `../sql-agent/.env` kopiert, mit Marcos ausdrücklicher Erlaubnis; Daten:
  über die Skripte neu erzeugt). Drei generierte Notiz-Korpora liegen
  gitignored lokal: `data/synthetic_notes.jsonl` (410 Notizen, ursprünglich
  gegen die volle 55-Code-Liste generiert), `data/synthetic_notes_core10.jsonl`
  (250 neue Notizen aus Anlauf 5, gezielt gegen die reduzierte 10-Code-Liste
  generiert — `generate_data.py --codes data/goz_codes.json --out
  data/synthetic_notes_core10.jsonl --per-code 20`) und
  `data/synthetic_notes_combined.jsonl` (beide zusammen, das ist die Datei,
  die aktuell als `--generated`-Input für `build_dataset.py` dient).
  **Falls diese Dateien wieder fehlen** (z.B. nach einem `git clean`):
  1. `curate_codes.py` ausführen (erzeugt die volle 55-Code-Liste aus der
     amtlichen Quelle).
  2. `data/goz_codes.json` von Hand auf `["0065", "0070", "0090", "0100",
     "2030", "2040", "2060", "2290", "2350", "2410"]` filtern (die
     Kern-10-Reduktion aus Anlauf 4 — kein Script kennt diesen Schritt,
     er ist manuell).
  3. `generate_data.py --codes data/goz_codes.json --out
     data/synthetic_notes_core10.jsonl --per-code 20` (Anlauf-5-Notizen,
     die alten 410 aus dem ersten Lauf sind ohne den Original-Output nicht
     exakt reproduzierbar, da `generate_examples` mit `temperature=0.9`
     nicht-deterministisch ist - aber ein neuer Lauf mit ausreichend
     Notizen sollte ähnliche Dichte liefern).
  4. Beide Korpora zu `data/synthetic_notes_combined.jsonl` zusammenfügen,
     dann `build_dataset.py --generated data/synthetic_notes_combined.jsonl
     ...` (Befehl in `CLAUDE.md`/oben, Golden-Synth-Pfad siehe
     "Referenz: wichtige Pfade" unten).
- **`goz-extract-src.zip`** im Repo-Root (gitignored) ist das Colab-Upload-
  Bündel (`src/goz_extract/` + `data/goz_codes.json` + `data/train.jsonl` +
  `data/test.jsonl`). Nach jeder Änderung an einer dieser Dateien neu
  bauen: `zip -r goz-extract-src.zip src/goz_extract data/goz_codes.json
  data/train.jsonl data/test.jsonl -x "*__pycache__*"`. **Aktuell bereits
  neu gebaut** mit Stand Anlauf 5 (325/81 Train-/Test-Daten + alle
  Code-Änderungen) — bei weiteren Code-/Daten-Änderungen erneut ausführen.

## Aktueller Datei-Status (nicht committed, bewusst)

- `results/predictions_rag.jsonl`, `results/predictions_finetune.jsonl`,
  `results/results.md` — **von Anlauf 4** (10-Code-Space, JSON-Format,
  RAG F1 0.24 / Finetune F1 0.06, Finetune kollabiert auf `2060`/`2030` —
  siehe "Anlauf-4-Ergebnisse" oben), liegen lokal, sind NICHT committed.
  Durch Anlauf 5 (325/81 statt 125/31 Notizen, 10 statt 3 Epochen,
  `top_n=5` statt 12) erneut überholt. `results.md` ist nicht gitignored
  (soll später mit echten Zahlen committed werden), aktuell bewusst noch
  nicht eingecheckt. **Nach dem Anlauf-5-Colab-Lauf diese drei Dateien
  überschreiben, dann erst committen.**
- `adapters/goz-extract-llama32-3b/` — LoRA-Adapter von Anlauf 4, trainiert
  auf 125 Notizen über 24 Steps. Komplett gitignored (bleibt immer lokal),
  wird beim Anlauf-5-Lauf überschrieben (10 Epochen statt 3, `save_total_limit=1`
  jetzt gesetzt).

## Offene Punkte danach (nicht blockierend, aber nicht vergessen)

1. **README.md "## Ergebnisse"** ist noch ein Platzhalter-Kommentar —
   sobald echte, plausible Zahlen aus `results/results.md` vorliegen, dort
   einfügen.
2. **Streamlit-Demo (`app.py`) wurde noch nie im Browser getestet** — nur
   statisch verifiziert (AST-Parse, Signatur-Check). Sobald ein
   funktionierender Adapter da ist: `.venv-data` reicht nicht (braucht
   Torch/Transformers/Streamlit) — dafür bräuchte es doch das volle
   `pip install -e ".[dev]"` lokal (mit den obigen Hänger-Fallstricken im
   Hinterkopf), oder die Demo bewusst nur als "noch nicht verifiziert"
   kennzeichnen.
3. **IP-Disclosure-Frage zur Design-Spec** (noch unentschieden): Der finale
   Reviewer merkte an, dass `docs/superpowers/specs/…design.md` intern recht
   detailliert beschreibt, wie MAIKA bei ILI DIGITAL AG technisch
   aufgebaut ist (Name, Retrieval-Architektur, Validierungsregeln). Kein
   Verstoß gegen die vereinbarte Datengrenze (die echten IP-Grenzen — nur
   öffentliche GOZ-Codes + 5 synthetische Fixtures — sind mehrfach
   verifiziert eingehalten), aber bevor das Repo public geht, sollte Marco
   bewusst entscheiden, wie viel davon sichtbar bleiben soll.
4. **Repo ist noch nicht auf GitHub gepusht** (anders als `sql-agent`).
   Separater Schritt, wann immer Marco so weit ist.

## Referenz: wichtige Pfade

- Dieses Repo: `C:\Users\Marco\OneDrive\02_Portfolio\goz-finetune-vs-rag`
- MAIKA-Referenz-Repo (extern, nur für Kontext/Kuration, nicht Teil dieses
  Repos): `C:\Users\Marco\Downloads\dentist-main\dentist-main`
  (`data/databases/goz_database_v4.json` = Quelle für die kuratierte
  Codeliste; `tests/fixtures/golden_single_v2/*synth*.json` = die 5
  synthetischen Golden-Fixtures)
- Anthropic-API-Key liegt auch in `../sql-agent/.env` (Schwesterprojekt),
  falls `.env` hier nochmal neu aufgesetzt werden muss.
