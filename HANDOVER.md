# Handover — goz-finetune-vs-rag

**Stand:** 2026-07-28, nach Merge in `master`, aktuellster Commit `c3f7467`.

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
und in `master` gemergt (per Subagent-Driven Development, Branch
`implement-goz-extract` ist nach dem Merge gelöscht). Ein finaler
Whole-Branch-Review fand 1 Critical + 4 Important + 5 Minor Findings, alle
in einer Fix-Runde behoben (Commit `d496ef0`), per Scoped-Re-Review bestätigt.

**Der Colab-GPU-Trainingslauf (Anlauf 3, manueller `Trainer`) ist erfolgreich
durchgelaufen** (kein OOM, kein Fehler mehr). Danach beim Finetune-Inferenz-
Schritt (Adapter laden) noch zwei weitere Live-Bugs gefunden und gefixt
(`torchao`-Versionskonflikt erneut, plus ein `KeyError` durch
`device_map="auto"`-Offloading, das mit `peft`s Adapter-Lade-Logik
kollidierte — siehe unten). **Marco steht jetzt kurz davor, die
Finetune-Inferenz-Zelle mit allen Fixes zu wiederholen** — das ist der
Zweck der "Sofort-nächster-Schritt"-Anleitung unten.

Offen bleibt weiterhin die inhaltliche Frage: ob der Completion-Masking-Fix
den ursprünglichen Mode Collapse tatsächlich behoben hat, zeigt sich erst an
den neuen Predictions/Eval-Zahlen, nicht schon daran, dass der Lauf
technisch durchlief.

## Sofort-nächster Schritt

Marco ist mitten in einer Colab-Session: Training ist durch. Die
Finetune-Inferenz-Zelle (Abschnitt 4) ist gerade zweimal live fehlgeschlagen
(`torchao`-`ImportError`, dann ein `KeyError` durchs Offloading) — beide
Fixes sind jetzt im Notebook (Commit `c3f7467`): `torchao`-Upgrade in
Zelle 1, `device_map={"": 0}` beim Adapter-Laden in Abschnitt 4. **Er muss
die aktuelle Version der Abschnitt-4-Zelle (Speicher freigeben + Adapter
laden) aus `notebooks/train_and_infer.ipynb` in seine laufende Session
einfügen** (kein Neustart nötig, das Training-Ergebnis bleibt erhalten).

**Wichtig: die bereits gelaufene RAG-Baseline muss auch nochmal laufen**,
bevor `run_eval.py` aussagekräftige Zahlen liefert. Zwei Gründe:
1. `prompting.py` und `retrieval.py` haben seit dem RAG-Baseline-Lauf
   Fixes bekommen (schärfere Instruktion, getrennte Query-/Passage-Präfixe
   fürs Embedding-Retrieval — asymmetrische Encoder wie
   `multilingual-e5-base` brauchen das) — der gelaufene
   `results/predictions_rag.jsonl` spiegelt noch den alten Code wider.
2. Die RAG-Baseline-Zelle selbst hat jetzt auch `device_map={"": 0}` (war
   vorher `"auto"`, hätte beim erneuten Laden potenziell dasselbe
   Offload-Problem wie beim Finetune-Schritt).

Dafür: `goz-extract-src.zip` aus dem Repo-Root neu hochladen (schon mit
allen Fixes neu gebaut) + die RAG-Baseline-Zellen (Abschnitt 2) erneut
ausführen (Basismodell muss neu geladen werden, `base_model` wurde beim
Speicher-Freigeben-Schritt schon gelöscht).

**Neu seit dem parallelen Review:** eine Sanity-Check-Zelle direkt nach der
Tokenisierung in Abschnitt 3 (vor `TrainingArguments`/`trainer.train()`),
die die Completion-Masking-Grenze an 10 Stichproben verifiziert und mit
einer klaren Fehlermeldung abbricht, falls sie nicht exakt sitzt. **Die
bereits gelaufene Trainingssession hatte diese Zelle noch nicht** — die
Masking-Korrektheit für den bereits trainierten Adapter ist technisch noch
unverifiziert, auch wenn der Lauf selbst ohne Fehler durchging und die
Loss-Kurve plausibel aussah. Falls `tokenized_train` und `tokenizer` in der
laufenden Colab-Session noch im Speicher sind (wurden beim
Speicher-Freigeben-Schritt für Abschnitt 4 gelöscht — falls das schon
passiert ist, nicht mehr nachholbar ohne neu zu trainieren), lohnt sich ein
nachträgliches Ausführen dieser Zelle als Bestätigung. Falls nicht mehr
möglich: kein Blocker, nur ein fehlender zusätzlicher Vertrauensbeweis —
die Eval-Zahlen selbst sind der eigentliche Test.

**Wenn Finetune-Inferenz und die neue RAG-Baseline durch sind:** Ergebnisse
herunterladen (siehe Anleitung im Notebook, Abschnitt 5), lokal nach
`results/` bzw. `adapters/` legen (Downloads/Zip-Struktur vorher prüfen —
beim ersten Mal landete der Adapter-Ordner doppelt verschachtelt, siehe
"Bekannte Stolperfalle beim Artefakte-Download" unten), `scripts/run_eval.py
--results-dir results/` laufen lassen (nutze dafür `.venv-data`, siehe
unten — kein Torch nötig), und **die Zahlen kritisch prüfen** (siehe "Was
zu erwarten ist" unten) bevor du sie als Endergebnis behandelst oder ins
README schreibst.

## Was gerade live gefixt wurde (und warum)

Der erste komplette Trainingslauf (vor den Fixes unten) lieferte
offensichtlich kaputte Ergebnisse:

- **RAG-Baseline**: gab im Schnitt 9 von 12 Kandidaten-Codes pro Notiz
  zurück (statt gezielt auszuwählen) → Precision 0.13, Recall 0.64, F1 0.21
- **LoRA-Finetune**: sagte bei 60/82 (73%) Test-Notizen exakt denselben
  einzelnen Code `"2300"` voraus, unabhängig vom Inhalt (Mode Collapse) →
  F1 0.03 — deutlich schlechter als die Baseline

Committete Fixes dafür:

1. **`src/goz_extract/prompting.py`** (Commit `b030681`) — `_INSTRUCTION`
   verschärft: explizit "nur zutreffende Codes, nicht jeden Kandidaten,
   keine Erklärung, keine Wiederholung der Notiz". Zielt auf das
   RAG-Überprediction-Problem. **Noch nicht verifiziert.**
2. **Completion-Only-Loss fürs Training** — das war eine kleine Odyssee
   durch drei Anläufe, weil `trl` auf Colab in einer viel neueren Version
   installiert ist (1.9.2) als alles, was ich zuverlässig kenne:
   - Anlauf 1 (Commit `b030681`): `DataCollatorForCompletionOnlyLM` aus
     `trl` — **existiert in trl>=1.x nicht mehr** (`ImportError`, live auf
     Colab bestätigt).
   - Anlauf 2 (Commit `1cf8f77`): TRLs Nachfolge-Mechanismus
     `SFTConfig(assistant_only_loss=True)` + Dataset mit `"messages"`-Feld
     statt geflachtem `"text"`-String — **scheiterte ebenfalls**, weil
     dieses Feature `{% generation %}`-Marker im Jinja-Chat-Template
     braucht, die Llama 3.2s Standard-Template nicht hat, und TRL das
     Template nicht automatisch patchen kann (`ValueError`, live auf Colab
     bestätigt).
   - Anlauf 3 (Commit `7bc7cd7`, **aktueller Stand**): TRLs High-Level-API
     komplett umgangen. Labels werden jetzt manuell maskiert (Prompt-Teil
     inkl. Chat-Template-Boilerplate auf `-100` gesetzt, nur die
     Ziffern-Antwort trägt zum Loss bei) und mit dem einfacheren, stabilen
     `transformers.Trainer` + `peft.get_peft_model` trainiert statt
     `SFTTrainer`. Siehe `notebooks/train_and_infer.ipynb`, Abschnitt 3,
     für die Implementierung inkl. Kommentare.

   Die zugrundeliegende Diagnose bleibt unverändert: bei nur 63
   Trainingsschritten auf 329 fast identisch formulierten Prompts dominierte
   ohne Completion-Masking der Loss auf dem sich wiederholenden
   Prompt-Teil, das Lernsignal für die eigentliche Code-Auswahl ging unter.

**Anlauf 3 ist jetzt gegen einen echten Trainingslauf gelaufen** (siehe "Wo
wir stehen" oben) — technisch fehlerfrei, aber ob die Masking-Grenze dabei
tatsächlich korrekt saß, ist noch nicht per Sanity-Check bestätigt (siehe
"Sofort-nächster Schritt" oben, die neue Zelle kam erst danach dazu). Falls
bei einem künftigen Trainingslauf der manuelle `Trainer`-Ansatz an einer
weiteren Versions-Eigenheit scheitert (z.B. `prepare_model_for_kbit_training`/
`get_peft_model`-Signatur hat sich geändert): dieselbe Introspektions-Strategie
wie bei den TRL-Fehlern anwenden (`inspect.signature(...)`, `dir(...)`,
Docstring lesen) statt zu raten — hat bisher jedes Mal den echten Grund
gefunden.

### Was zu erwarten ist / wie du die neuen Zahlen einordnest

- Wenn `LoRA-Finetune` nach dem Fix immer noch stark kollabiert (z.B. eine
  Handvoll identischer Vorhersagen dominiert weiterhin): nächster Hebel ist
  vermutlich zu wenig Training — `num_train_epochs=3` (63 Steps) in der
  Trainings-Zelle ist knapp bemessen. Vor dem Erhöhen aber erst prüfen, ob
  die Completion-Masking-Fix überhaupt greift (z.B. Trainings-Loss-Kurve
  diesmal genauer anschauen, oder eine kleine Stichprobe roher —
  ungeparster — Modellausgaben ausgeben lassen).
- Wenn `RAG-Baseline` immer noch viele Kandidaten zurückgibt: als
  Diagnose-Schritt einen rohen (nicht geparsten) Modell-Output ausgeben
  lassen (in `generate_codes` in `src/goz_extract/inference.py` vor dem
  `parse_code_list_response(...)`-Aufruf `print(generated)` einfügen) — so
  lässt sich unterscheiden, ob das Modell tatsächlich zu viele Codes wählt,
  oder ob es nur ausschweift und `_CODE_PATTERN` in `prompting.py`
  (`\bÄ?\d{3,4}\b`) jede erwähnte Zahl einsammelt statt nur die "gewählten".
  Falls Letzteres: robusteren Parser bauen (z.B. nur erste Zeile der
  Antwort auswerten) statt weiter an der Instruktion zu drehen.
- Vergiss nicht: das Test-Set hat nur 82 Notizen (nach dem Label-Space-Fix,
  siehe unten) — bei so kleinen Zahlen schwankt F1 stark zwischen einzelnen
  Beispielen, nicht überinterpretieren.

## Kritischer Code-Review (2026-07-28)

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
  über die Skripte neu erzeugt, exakt dieselben Zahlen wie vorher: 410
  generierte Notizen, 329 Train / 82 Test nach dem Label-Space-Filter).
  **Falls diese Dateien wieder fehlen** (z.B. nach einem `git clean`),
  einfach die drei Skripte in der Reihenfolge curate → generate → build neu
  laufen lassen (Befehle in `CLAUDE.md` unter "Commands").
- **`goz-extract-src.zip`** im Repo-Root (gitignored) ist das Colab-Upload-
  Bündel (`src/goz_extract/` + `data/goz_codes.json` + `data/train.jsonl` +
  `data/test.jsonl`). Nach jeder Änderung an einer dieser Dateien neu
  bauen: `zip -r goz-extract-src.zip src/goz_extract data/goz_codes.json
  data/train.jsonl data/test.jsonl -x "*__pycache__*"`. **Stand nach dem
  Review vom 2026-07-28 (`prompting.py`, `retrieval.py` geändert) noch
  nicht neu gebaut** — vor dem nächsten Colab-Upload nachholen, siehe
  "Sofort-nächster Schritt" oben.

## Aktueller Datei-Status (nicht committed, bewusst)

- `results/predictions_rag.jsonl`, `results/predictions_finetune.jsonl`,
  `results/results.md` — **vom ERSTEN, noch fehlerhaften Trainingslauf**
  (Mode Collapse etc.), liegen lokal, sind NICHT committed. `results.md`
  ist nicht gitignored (soll später mit echten Zahlen committed werden),
  aktuell aber bewusst noch nicht eingecheckt, weil die Zahlen falsch sind.
  **Nach dem neuen Colab-Lauf diese drei Dateien überschreiben, dann erst
  committen.**
- `adapters/goz-extract-llama32-3b/` — LoRA-Adapter vom ersten,
  fehlerhaften Lauf. Komplett gitignored (bleibt immer lokal). Wird beim
  neuen Lauf einfach überschrieben.

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
