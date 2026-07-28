# Handover — goz-finetune-vs-rag

**Stand:** 2026-07-28, nach Merge in `master` (Commit `b030681`).
**Für wen:** Ein neuer Agent (oder du selbst in einer neuen Session), der hier
weitermacht, ohne den bisherigen Chatverlauf zu kennen.

Lies zuerst `CLAUDE.md` (Projektkontext, Arbeitsstil), dann diese Datei
(aktueller Stand, offene Baustellen, bekannte Fallstricke). Die Design-Spec
(`docs/superpowers/specs/2026-07-27-goz-finetune-vs-rag-design.md`) und der
Implementierungsplan (`docs/superpowers/plans/2026-07-27-goz-finetune-vs-rag-implementation.md`)
bleiben die Quelle für *warum* die Architektur so aussieht, wie sie aussieht.

## Wo wir stehen

Alle 13 Tasks aus dem Implementierungsplan sind umgesetzt, einzeln reviewt
und in `master` gemergt (per Subagent-Driven Development, Branch
`implement-goz-extract` ist nach dem Merge gelöscht). Ein finaler
Whole-Branch-Review fand 1 Critical + 4 Important + 5 Minor Findings, alle
in einer Fix-Runde behoben (Commit `d496ef0`), per Scoped-Re-Review bestätigt.

**Seitdem laufen wir gegen den echten Colab-GPU-Trainingslauf** — das ist der
Teil, der noch nicht fertig/verifiziert ist. Mehrere Live-Bugs wurden dabei
gefunden und im Code gefixt (siehe unten), aber der **erste vollständige
Trainingslauf mit allen Fixes steht noch aus**.

## Sofort-nächster Schritt

Marco startet (oder hat gerade gestartet) eine **frische Colab-Session**:

1. Bestehende Colab-Runtime trennen/löschen (die alte Session ist voller
   Live-Patches, die nicht mehr gebraucht werden)
2. `notebooks/train_and_infer.ipynb` frisch hochladen (enthält bereits alle
   Fixes, siehe unten)
3. Bei der Upload-Zelle: `goz-extract-src.zip` aus dem Repo-Root hochladen
   (ebenfalls bereits mit allen Fixes neu gebaut, gitignored, liegt lokal)
4. Zellen der Reihe nach durchlaufen (HF-Token erneut in `notebook_login()`
   einfügen — Session-scoped, die Meta-Lizenz-Freigabe selbst gilt weiter)

**Wenn der Lauf durch ist:** Ergebnisse herunterladen (siehe Anleitung im
Notebook, Abschnitt 5), lokal nach `results/` bzw. `adapters/` legen,
`scripts/run_eval.py --results-dir results/` laufen lassen (nutze dafür
`.venv-data`, siehe unten — kein Torch nötig), und **die Zahlen kritisch
prüfen** (siehe "Was zu erwarten ist" unten) bevor du sie als Endergebnis
behandelst oder ins README schreibst.

## Was gerade live gefixt wurde (und warum)

Der erste komplette Trainingslauf (vor den Fixes unten) lieferte
offensichtlich kaputte Ergebnisse:

- **RAG-Baseline**: gab im Schnitt 9 von 12 Kandidaten-Codes pro Notiz
  zurück (statt gezielt auszuwählen) → Precision 0.13, Recall 0.64, F1 0.21
- **LoRA-Finetune**: sagte bei 60/82 (73%) Test-Notizen exakt denselben
  einzelnen Code `"2300"` voraus, unabhängig vom Inhalt (Mode Collapse) →
  F1 0.03 — deutlich schlechter als die Baseline

Committete Fixes dafür (Commit `b030681`):

1. **`src/goz_extract/prompting.py`** — `_INSTRUCTION` verschärft: explizit
   "nur zutreffende Codes, nicht jeden Kandidaten, keine Erklärung, keine
   Wiederholung der Notiz". Zielt auf das RAG-Überprediction-Problem.
2. **`notebooks/train_and_infer.ipynb`, Trainings-Zelle (Abschnitt 3)** —
   `DataCollatorForCompletionOnlyLM` ergänzt, damit der Loss nur auf der
   Antwort berechnet wird, nicht auf der (bei allen 329 Beispielen fast
   identischen) Instruktion+Notiz davor. Das ist die wahrscheinlichste
   Ursache des Mode Collapse: bei nur 63 Trainingsschritten dominierte der
   Loss auf dem sich wiederholenden Prompt-Teil, das eigentliche Lernsignal
   für die Code-Auswahl ging unter.

**Diese Fixes wurden noch NICHT gegen einen echten Trainingslauf
verifiziert.** Das ist der Zweck des "Sofort-nächster-Schritt"-Laufs oben.

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
- **`peft`/`torchao`-Versionskonflikt**: Colabs vorinstalliertes `torchao`
  (0.10.0) war zu alt für die von `!pip install trl` gezogene `peft`-
  Version (verlangt >0.16.0), `PeftModel.from_pretrained(...)` crashte mit
  `ImportError`. Fix war nur ein Live-Kommando
  (`!pip install -q -U torchao`), **nicht** ins Notebook eingebaut (siehe
  "Bewusst nicht gefixt" unten, warum).
- **`notebook_login()`**: nimmt den Token NICHT als Argument entgegen
  (`notebook_login("hf_...")` wirft `TypeError`) — Zelle ohne Argument
  aufrufen, Token ins dann erscheinende Eingabefeld einfügen.

## Bewusst nicht gefixt (Entscheidung, kein Vergessen)

- **Keine Versions-Pins im `!pip install`** in Zelle 1 des Notebooks. Der
  finale Review hatte das als Empfehlung genannt; ich habe bewusst
  dagegen entschieden, weil die auf Colab tatsächlich installierten
  Versionen (`transformers 5.14.1`, `torch 2.13.0`) deutlich neuer sind als
  alles, was zuverlässig bekannt ist (dieses Projekt läuft in einer Zukunft
  jenseits des Wissensstands) — ein geratener Pin hätte eher neue
  Inkompatibilitäten riskiert als welche gelöst. Der `torchao`-Fix wurde
  aus demselben Grund nur live gemacht, nicht committed.
- **`device_map="auto"`-Offloading auf CPU** (einmal während des RAG-
  Baseline-Laufs aufgetreten, per manuellem `.to("cuda:0")` umgangen): trat
  vermutlich nur wegen Speicherdrucks durch mehrfaches Neuladen während des
  Live-Debuggings auf, nicht als systematischer Bug. Nicht in den
  committeten Code übernommen — bei einem sauberen Durchlauf sollte ein
  3B-Modell in fp16 (~6GB) auf einer T4 (~15GB) problemlos komplett auf die
  GPU passen.

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
  data/train.jsonl data/test.jsonl -x "*__pycache__*"` (aktuell mit Stand
  Commit `b030681` gebaut).

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
