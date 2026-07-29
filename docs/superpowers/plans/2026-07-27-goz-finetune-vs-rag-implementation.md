# GOZ-Code-Extraktion: LoRA-Finetuning vs. RAG-Baseline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ein Llama-3.2-3B-Instruct-Modell wird per QLoRA darauf trainiert, aus einer zahnärztlichen Behandlungsnotiz die zutreffenden GOZ-Ziffern zu extrahieren, und wird gegen eine selbstgebaute RAG-Baseline (gleiches Basismodell, aber mit retrieval-gestütztem Prompt-Kontext statt LoRA-Gewichten) verglichen — mit einer Streamlit-Demo und einer README-Ergebnistabelle als Portfolio-Artefakt.

**Architecture:** Reine, gut testbare Python-Bausteine (Schema, Retrieval, Prompting, Evaluation) leben in `src/goz_extract/` und werden sowohl vom lokalen Code als auch vom Colab-Notebook importiert. Alles, was eine GPU braucht (Training, Modell-Inferenz), läuft in einem Colab-Notebook und exportiert Ergebnis-Artefakte (Predictions als JSONL, LoRA-Adapter) zurück ins Repo; alles andere läuft lokal ohne GPU.

**Tech Stack:** Python 3.11, Pydantic, Hugging Face `transformers`/`peft`/`trl`/`datasets`, `sentence-transformers`, `rank-bm25`, LangChain (`langchain-anthropic`, für Datengenerierung — gleiches Muster wie im Schwesterprojekt `sql-agent`), Streamlit, pytest.

## Global Constraints

- Zeitbudget: 1-2 Tage — jede Aufgabe ist bewusst klein gehalten, kein Over-Engineering.
- Label-Space: GOZ-Codes der Kategorien `A__Allgemeine_zahnärztliche_Leistungen` und `C__Konservierende_Leistungen` (55 Codes gesamt — passt in den vereinbarten Rahmen von ~40-60).
- Aus einem Referenz-Repo eines bestehenden Produktivsystems wird **ausschließlich** verwendet: `goz_nr` + `bezeichnung` aus `goz_database_v4.json` (öffentliche Gebührenordnung) sowie 5 `*synth_*.json`-Golden-Fixtures aus `tests/fixtures/golden_single_v2/`. Keine proprietären Kommentare, Aliases, Synonyme, Embeddings, `real_*`-Fixtures oder Anwendungscode des Referenzsystems.
- Basismodell: `meta-llama/Llama-3.2-3B-Instruct` (gated auf Hugging Face — Zustimmung zur Meta-Lizenz ist vorausgesetzt, siehe Task 1).
- Finetune-Inferenz nutzt **kein Retrieval** — Domänenwissen steckt in den LoRA-Gewichten.
- RAG-Baseline nutzt **dasselbe unveränderte Basismodell**, aber mit BM25+Embeddings(RRF)-Kandidaten als Prompt-Kontext.
- Embeddings für die RAG-Baseline werden selbst berechnet (`intfloat/multilingual-e5-base`, offen, keine proprietären oder OpenAI-Embeddings).
- Metrik: Precision/Recall/F1 pro Notiz (gemittelt) + Exact-Match-Rate, für beide Ansätze auf identischem Test-Split.
- Doku, Kommentare und Prompts auf Deutsch (Marco lernt aktiv mit, gleicher Stil wie `sql-agent`).
- Training läuft auf Colab (T4-GPU) — dieses Environment hat keine GPU, daher sind alle GPU-Schritte als manuell auszuführende Colab-Zellen markiert, mit exaktem erwartetem Ergebnis zur Verifikation.

---

## Task 1: Projekt-Grundgerüst

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `CLAUDE.md`
- Create: `src/goz_extract/__init__.py`
- Create: `tests/__init__.py`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Produces: installierbares Package `goz_extract` (Version `0.1.0`), Test-Setup für alle folgenden Tasks.

- [ ] **Step 1: `pyproject.toml` anlegen**

```toml
[project]
name = "goz-extract"
version = "0.1.0"
description = "GOZ-Code-Extraktion aus Behandlungsnotizen: LoRA-Finetuning vs. RAG-Baseline"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.6",
    "langchain>=0.2",
    "langchain-anthropic>=0.1",
    "python-dotenv>=1.0",
    "rank-bm25>=0.2.2",
    "sentence-transformers>=3.0",
    "numpy>=1.26",
    "torch>=2.2",
    "transformers>=4.44",
    "peft>=0.12",
    "trl>=0.9",
    "datasets>=2.20",
    "streamlit>=1.37",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: `.gitignore` anlegen**

```
.venv/
__pycache__/
*.pyc
.env
data/synthetic_notes.jsonl
results/predictions_*.jsonl
*.ipynb_checkpoints/
adapters/
.pytest_cache/
```

(`data/synthetic_notes.jsonl` und die Predictions sind generierte Artefakte,
kein Quellcode — werden in Task 5/10 bewusst per Skript erzeugt, nicht von
Hand editiert. Falls sie später doch versioniert werden sollen, aus
`.gitignore` entfernen.)

- [ ] **Step 3: `.env.example` anlegen**

```
# Für die synthetische Datengenerierung (Task 4)
ANTHROPIC_API_KEY=
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-5
```

- [ ] **Step 4: `CLAUDE.md` anlegen**

```markdown
# goz-finetune-vs-rag — Projektkontext

Vollständiger Plan: `docs/superpowers/plans/2026-07-27-goz-finetune-vs-rag-implementation.md`
Design-Spec: `docs/superpowers/specs/2026-07-27-goz-finetune-vs-rag-design.md`

## Was das hier ist

Portfolio-Projekt von Marco Stang (Schwesterprojekt zu `sql-agent`). Ziel:
PyTorch/LoRA-Finetuning-Lücke im Lebenslauf schließen und die Frage
"Finetuning vs. RAG" mit einem echten Experiment beantworten.

Thematisch an einer Alltagsaufgabe aus dem zahnärztlichen Praxisbetrieb
orientiert (Notiz → GOZ-Codes) — verwendet aber **nur öffentliche
GOZ-Daten** (amtliche Codeliste) und komplett neue, selbst generierte
Trainingsdaten. Siehe Design-Spec, Abschnitt "Datenherkunft &
IP-Abgrenzung", für die genaue Abgrenzung.

## Wie hier gearbeitet wird

- Doku/Kommentare/Antworten auf Deutsch, Marco lernt aktiv mit (gleicher
  Stil wie `sql-agent`).
- GPU-Schritte (Training, Modell-Inferenz) laufen auf Colab, nicht lokal —
  siehe `notebooks/train_and_infer.ipynb`.

## Aktueller Stand

*Diesen Abschnitt aktuell halten, sobald ein Task aus dem Implementierungsplan
abgeschlossen ist.*

- ⬜ Projekt-Grundgerüst
```

- [ ] **Step 5: Package-Skelett anlegen**

`src/goz_extract/__init__.py`:
```python
__version__ = "0.1.0"
```

`tests/__init__.py`: leere Datei.

- [ ] **Step 6: Smoke-Test schreiben**

`tests/test_smoke.py`:
```python
import goz_extract


def test_package_importable():
    assert goz_extract.__version__ == "0.1.0"
```

- [ ] **Step 7: venv anlegen, Package installieren, Test laufen lassen**

Run:
```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/python.exe -m pytest tests/test_smoke.py -v
```
Expected: `1 passed`

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore .env.example CLAUDE.md src/ tests/
git commit -m "Set up project scaffolding for goz-extract package"
```

---

## Task 2: Datenschema

**Files:**
- Create: `src/goz_extract/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Produces:
  - `GozCode(goz_nr: str, bezeichnung: str)` — Pydantic-Model
  - `NoteExample(text: str, expected_codes: list[str], difficulty: str | None, source: str, split: str | None)` — Pydantic-Model
  - `Prediction(text: str, predicted_codes: list[str])` — Pydantic-Model

- [ ] **Step 1: Failing Test schreiben**

`tests/test_schema.py`:
```python
import pytest
from pydantic import ValidationError

from goz_extract.schema import GozCode, NoteExample, Prediction


def test_goz_code_requires_nr_and_bezeichnung():
    code = GozCode(goz_nr="0090", bezeichnung="Intraorale Infiltrationsanästhesie")
    assert code.goz_nr == "0090"
    assert code.bezeichnung.startswith("Intraorale")


def test_goz_code_rejects_missing_fields():
    with pytest.raises(ValidationError):
        GozCode(goz_nr="0090")


def test_note_example_defaults():
    note = NoteExample(
        text="Zahn 36: Infiltrationsanästhesie, Komposit-Füllung zweiflächig.",
        expected_codes=["0090", "2080"],
    )
    assert note.difficulty is None
    assert note.source == "generated"
    assert note.split is None


def test_prediction_holds_predicted_codes():
    pred = Prediction(text="...", predicted_codes=["0090", "2080"])
    assert pred.predicted_codes == ["0090", "2080"]
```

- [ ] **Step 2: Test laufen lassen, Fehlschlag verifizieren**

Run: `.venv/Scripts/python.exe -m pytest tests/test_schema.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'goz_extract.schema'`

- [ ] **Step 3: `schema.py` implementieren**

```python
"""Pydantic-Schemas für GOZ-Codes, Trainings-/Testbeispiele und Modell-Vorhersagen."""
from typing import Literal

from pydantic import BaseModel, Field

Difficulty = Literal["easy", "medium", "hard"]
Split = Literal["train", "test"]


class GozCode(BaseModel):
    """Ein einzelner GOZ-Code aus der amtlichen Gebührenordnung.

    Enthält bewusst nur die öffentlichen Felder goz_nr und bezeichnung —
    siehe Design-Spec, Abschnitt "Datenherkunft & IP-Abgrenzung".
    """

    goz_nr: str
    bezeichnung: str


class NoteExample(BaseModel):
    """Eine (synthetische oder aus den Golden-Fixtures übernommene)
    Behandlungsnotiz mit den erwarteten GOZ-Codes."""

    text: str
    expected_codes: list[str]
    difficulty: Difficulty | None = None
    source: str = "generated"
    split: Split | None = None


class Prediction(BaseModel):
    """Vorhersage eines Ansatzes (RAG-Baseline oder Finetune) für eine Notiz."""

    text: str
    predicted_codes: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Test laufen lassen, Erfolg verifizieren**

Run: `.venv/Scripts/python.exe -m pytest tests/test_schema.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/goz_extract/schema.py tests/test_schema.py
git commit -m "Add Pydantic schema for GOZ codes, note examples and predictions"
```

---

## Task 3: GOZ-Codeliste kuratieren

**Files:**
- Create: `src/goz_extract/curate.py`
- Create: `scripts/curate_codes.py`
- Test: `tests/test_curate.py`

**Interfaces:**
- Consumes: `GozCode` aus Task 2 (`src/goz_extract/schema.py`)
- Produces: `curate_codes(raw_entries: list[dict], allowed_prefixes: tuple[str, ...]) -> list[GozCode]`, genutzt von Task 5 (Datensatz), Task 6 (Retrieval), Task 7 (Prompting)
- Erzeugt (manuell auszuführen, siehe Step 6): `data/goz_codes.json`

- [ ] **Step 1: Failing Test schreiben**

`tests/test_curate.py`:
```python
from goz_extract.curate import curate_codes

RAW_ENTRIES = [
    {"goz_nr": "0010", "bezeichnung": "Eingehende Untersuchung", "kategorie": "A__Allgemeine_zahnärztliche_Leistungen"},
    {"goz_nr": "2080", "bezeichnung": "Kompositfüllung zweiflächig", "kategorie": "C__Konservierende_Leistungen"},
    {"goz_nr": "9010", "bezeichnung": "Implantatinsertion", "kategorie": "K__Implantologische_Leistungen"},
    {"goz_nr": "0500", "bezeichnung": "Osteotomie", "kategorie": "D__Chirurgische_Leistungen"},
]


def test_curate_keeps_only_allowed_categories():
    result = curate_codes(RAW_ENTRIES, allowed_prefixes=("A__", "C__"))
    codes = {c.goz_nr for c in result}
    assert codes == {"0010", "2080"}


def test_curate_only_exposes_public_fields():
    result = curate_codes(RAW_ENTRIES, allowed_prefixes=("A__",))
    assert len(result) == 1
    assert result[0].goz_nr == "0010"
    assert result[0].bezeichnung == "Eingehende Untersuchung"
    assert not hasattr(result[0], "kategorie")


def test_curate_raises_on_missing_field():
    import pytest

    broken = [{"goz_nr": "0010", "kategorie": "A__x"}]
    with pytest.raises(KeyError):
        curate_codes(broken, allowed_prefixes=("A__",))
```

- [ ] **Step 2: Test laufen lassen, Fehlschlag verifizieren**

Run: `.venv/Scripts/python.exe -m pytest tests/test_curate.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'goz_extract.curate'`

- [ ] **Step 3: `curate.py` implementieren**

```python
"""Filtert die volle GOZ-Codeliste auf die öffentlichen Felder der
gewünschten Kategorien (siehe Global Constraints im Implementierungsplan:
nur A__ und C__)."""
from goz_extract.schema import GozCode


def curate_codes(raw_entries: list[dict], allowed_prefixes: tuple[str, ...]) -> list[GozCode]:
    result = []
    for entry in raw_entries:
        kategorie = entry["kategorie"]
        if not kategorie.startswith(allowed_prefixes):
            continue
        result.append(GozCode(goz_nr=entry["goz_nr"], bezeichnung=entry["bezeichnung"]))
    return result
```

- [ ] **Step 4: Test laufen lassen, Erfolg verifizieren**

Run: `.venv/Scripts/python.exe -m pytest tests/test_curate.py -v`
Expected: `3 passed`

- [ ] **Step 5: CLI-Skript für die einmalige Kuration schreiben**

`scripts/curate_codes.py`:
```python
"""Einmaliges Skript: liest die volle GOZ-Datenbank aus einem lokalen
Checkout eines Referenz-Repos und schreibt die kuratierte, öffentliche
Teilmenge nach data/goz_codes.json. Pfad zum Checkout wird per Argument
übergeben, damit kein persönlicher absoluter Pfad im Repo landet.

Aufruf (einmalig, lokal, nicht Teil der automatisierten Tests):
    .venv/Scripts/python.exe scripts/curate_codes.py \
        --source "<pfad-zum-referenz-repo>/data/databases/goz_database_v4.json" \
        --out data/goz_codes.json
"""
import argparse
import json
from pathlib import Path

from goz_extract.curate import curate_codes

ALLOWED_PREFIXES = ("A__", "C__")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    raw_entries = json.loads(args.source.read_text(encoding="utf-8"))
    codes = curate_codes(raw_entries, allowed_prefixes=ALLOWED_PREFIXES)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps([c.model_dump() for c in codes], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"{len(codes)} Codes kuratiert -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Skript manuell einmal ausführen (lokal, mit echtem Referenz-Repo-Pfad)**

Run (Pfad an den tatsächlichen Download-Ordner anpassen):
```bash
.venv/Scripts/python.exe scripts/curate_codes.py --source "<pfad-zum-referenz-repo>/data/databases/goz_database_v4.json" --out data/goz_codes.json
```
Expected: Konsolenausgabe `55 Codes kuratiert -> data/goz_codes.json`, Datei
`data/goz_codes.json` existiert und enthält 55 Einträge mit ausschließlich
den Feldern `goz_nr` und `bezeichnung`.

- [ ] **Step 7: Commit**

```bash
git add src/goz_extract/curate.py scripts/curate_codes.py tests/test_curate.py data/goz_codes.json
git commit -m "Add GOZ code curation script, producing public 55-code subset"
```

---

## Task 4: Synthetische Datengenerierung

**Files:**
- Create: `src/goz_extract/data_generation.py`
- Create: `scripts/generate_data.py`
- Test: `tests/test_data_generation.py`

**Interfaces:**
- Consumes: `GozCode`, `NoteExample` aus Task 2
- Produces: `build_generation_prompt(codes: list[GozCode], difficulty: str, n_examples: int) -> str`, `parse_generation_response(raw_text: str) -> list[NoteExample]`, `generate_examples(chat_model, codes: list[GozCode], difficulty: str, n_examples: int) -> list[NoteExample]` — `generate_examples` wird von `scripts/generate_data.py` genutzt (Task 5 konsumiert die Ausgabedatei)

- [ ] **Step 1: Failing Test schreiben**

`tests/test_data_generation.py`:
```python
import json

from goz_extract.data_generation import (
    build_generation_prompt,
    generate_examples,
    parse_generation_response,
)
from goz_extract.schema import GozCode

CODES = [
    GozCode(goz_nr="0090", bezeichnung="Intraorale Infiltrationsanästhesie"),
    GozCode(goz_nr="2080", bezeichnung="Kompositfüllung, zweiflächig"),
]

VALID_RESPONSE = json.dumps(
    [
        {
            "text": "Zahn 36: Infiltrationsanästhesie, Karies excaviert, Kompositfüllung zweiflächig gelegt.",
            "expected_codes": ["0090", "2080"],
        },
        {
            "text": "Lokalanästhesie gesetzt, anschließend Füllung mit zwei Flächen in Komposit.",
            "expected_codes": ["0090", "2080"],
        },
    ]
)


def test_build_generation_prompt_mentions_all_codes_and_difficulty():
    prompt = build_generation_prompt(CODES, difficulty="medium", n_examples=2)
    assert "0090" in prompt
    assert "2080" in prompt
    assert "medium" in prompt or "mittel" in prompt
    assert "2" in prompt


def test_parse_generation_response_valid_json():
    examples = parse_generation_response(VALID_RESPONSE)
    assert len(examples) == 2
    assert examples[0].expected_codes == ["0090", "2080"]
    assert examples[0].source == "generated"


def test_parse_generation_response_handles_prose_wrapper():
    wrapped = f"Hier ist die Liste:\n```json\n{VALID_RESPONSE}\n```\nViele Grüße"
    examples = parse_generation_response(wrapped)
    assert len(examples) == 2


def test_parse_generation_response_rejects_empty():
    import pytest

    with pytest.raises(ValueError):
        parse_generation_response("Ich kann das nicht generieren.")


class _FakeChatModel:
    def invoke(self, prompt: str):
        class _Msg:
            content = VALID_RESPONSE

        return _Msg()


def test_generate_examples_uses_injected_chat_model_and_sets_difficulty():
    examples = generate_examples(_FakeChatModel(), CODES, difficulty="medium", n_examples=2)
    assert len(examples) == 2
    assert all(e.difficulty == "medium" for e in examples)
```

- [ ] **Step 2: Test laufen lassen, Fehlschlag verifizieren**

Run: `.venv/Scripts/python.exe -m pytest tests/test_data_generation.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'goz_extract.data_generation'`

- [ ] **Step 3: `data_generation.py` implementieren**

```python
"""Erzeugt synthetische Behandlungsnotizen samt erwarteten GOZ-Codes über
eine Chat-LLM-API. Prompt-Bau und Antwort-Parsing sind bewusst von der
tatsächlichen API-Anbindung getrennt, damit beides ohne Netzwerkzugriff
testbar ist (siehe generate_examples: nimmt ein beliebiges Objekt mit
.invoke(prompt) -> Objekt mit .content entgegen, kompatibel zu
LangChains BaseChatModel)."""
import json
import re
from typing import Protocol

from goz_extract.schema import GozCode, NoteExample

_DIFFICULTY_LABELS = {
    "easy": "leicht (nah an der amtlichen Bezeichnung formuliert)",
    "medium": "mittel (umgangssprachliche Zahnarzt-Notiz, andere Wortwahl)",
    "hard": "schwer (Abkürzungen, implizite Formulierung ohne offensichtliche Stichwortüberlappung)",
}


def build_generation_prompt(codes: list[GozCode], difficulty: str, n_examples: int) -> str:
    code_list = "\n".join(f"- {c.goz_nr}: {c.bezeichnung}" for c in codes)
    label = _DIFFICULTY_LABELS.get(difficulty, difficulty)
    return f"""Du erstellst synthetische Trainingsdaten für ein GOZ-Code-Extraktionsmodell.

Verfügbare GOZ-Codes (nur diese dürfen als expected_codes vorkommen):
{code_list}

Erzeuge {n_examples} realistische, kurze zahnärztliche Behandlungsnotizen
(Schwierigkeitsgrad: {label}). Jede Notiz kombiniert 1-3 der obigen Codes
plausibel (z.B. Anästhesie + Füllung, nicht rein zufällig).

Antworte ausschließlich mit einem JSON-Array, jedes Element:
{{"text": "<Notiz>", "expected_codes": ["<goz_nr>", ...]}}
"""


def parse_generation_response(raw_text: str) -> list[NoteExample]:
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"Keine JSON-Liste in der Antwort gefunden: {raw_text!r}")

    payload = json.loads(match.group(0))
    if not payload:
        raise ValueError("Generierte Liste ist leer")

    return [NoteExample(text=item["text"], expected_codes=item["expected_codes"]) for item in payload]


class _InvocableChatModel(Protocol):
    def invoke(self, prompt: str): ...


def generate_examples(
    chat_model: _InvocableChatModel, codes: list[GozCode], difficulty: str, n_examples: int
) -> list[NoteExample]:
    prompt = build_generation_prompt(codes, difficulty, n_examples)
    response = chat_model.invoke(prompt)
    examples = parse_generation_response(response.content)
    for example in examples:
        example.difficulty = difficulty
    return examples
```

- [ ] **Step 4: Test laufen lassen, Erfolg verifizieren**

Run: `.venv/Scripts/python.exe -m pytest tests/test_data_generation.py -v`
Expected: `5 passed`

- [ ] **Step 5: CLI-Skript für die echte Generierung schreiben**

`scripts/generate_data.py`:
```python
"""Erzeugt die synthetischen Trainingsnotizen über die echte Anthropic-API
und schreibt sie als JSONL. Braucht ANTHROPIC_API_KEY in .env (siehe
.env.example). Kostet echte API-Calls — nicht Teil der automatisierten Tests.

Aufruf:
    .venv/Scripts/python.exe scripts/generate_data.py \
        --codes data/goz_codes.json --out data/synthetic_notes.jsonl \
        --per-code 6
"""
import argparse
import json
import random
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from goz_extract.data_generation import generate_examples
from goz_extract.schema import GozCode

DIFFICULTIES = ["easy", "medium", "hard"]
CODES_PER_BATCH = 4
EXAMPLES_PER_BATCH = 5


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--per-code", type=int, default=6,
        help="Zielanzahl Batches pro Code (jeder Batch sampelt 4 Codes und erzeugt 5 Notizen)",
    )
    args = parser.parse_args()

    codes = [GozCode(**c) for c in json.loads(args.codes.read_text(encoding="utf-8"))]
    chat_model = init_chat_model(model="claude-sonnet-4-5", model_provider="anthropic", temperature=0.9)

    rng = random.Random(42)
    # Jeder Batch deckt CODES_PER_BATCH Codes ab -> um jeden Code im Schnitt
    # ~per_code mal in einem Batch zu haben, braucht es (len(codes)*per_code)/CODES_PER_BATCH
    # Batches. Bei 55 Codes und per_code=6: ~83 Batches x 5 Notizen = ~415 Notizen.
    total_batches = max(1, round(len(codes) * args.per_code / CODES_PER_BATCH))

    all_examples = []
    for _ in range(total_batches):
        batch_codes = rng.sample(codes, k=min(CODES_PER_BATCH, len(codes)))
        difficulty = rng.choice(DIFFICULTIES)
        examples = generate_examples(chat_model, batch_codes, difficulty, EXAMPLES_PER_BATCH)
        all_examples.extend(examples)
        print(f"+{len(examples)} Beispiele ({difficulty}), gesamt: {len(all_examples)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for example in all_examples:
            f.write(example.model_dump_json() + "\n")
    print(f"{len(all_examples)} Notizen geschrieben -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Skript manuell einmal ausführen (braucht ANTHROPIC_API_KEY in .env)**

Run:
```bash
.venv/Scripts/python.exe scripts/generate_data.py --codes data/goz_codes.json --out data/synthetic_notes.jsonl --per-code 6
```
Expected: Konsolenausgabe zeigt wachsende Gesamtzahl, am Ende ca.
380-420 Notizen geschrieben nach `data/synthetic_notes.jsonl` (83 Batches
x 5 Notizen, siehe Formel in `scripts/generate_data.py`). Stichprobenartig
2-3 Zeilen der Datei lesen und prüfen, dass `expected_codes` nur Codes aus
`data/goz_codes.json` enthält.

- [ ] **Step 7: Commit**

```bash
git add src/goz_extract/data_generation.py scripts/generate_data.py tests/test_data_generation.py
git commit -m "Add synthetic note generation via LLM, with prompt/parsing unit tests"
```

(Die generierte `data/synthetic_notes.jsonl` bleibt laut `.gitignore`
unversioniert — reproduzierbar über das Skript. Falls Marco das Artefakt
doch im Repo haben möchte, `.gitignore`-Eintrag entfernen und separat
committen.)

---

## Task 5: Datensatz zusammenstellen & splitten

**Files:**
- Create: `src/goz_extract/dataset.py`
- Create: `scripts/build_dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `NoteExample` aus Task 2
- Produces: `load_golden_synth_fixtures(fixture_dir: Path) -> list[NoteExample]`, `split_dataset(examples: list[NoteExample], test_fraction: float, seed: int) -> tuple[list[NoteExample], list[NoteExample]]` — Rückgabe `(train, test)`, wird von Task 10 (Colab-Notebook) und Task 11 (Eval-Report) konsumiert.

- [ ] **Step 1: Failing Test schreiben**

`tests/test_dataset.py`:
```python
import json

from goz_extract.dataset import load_golden_synth_fixtures, split_dataset
from goz_extract.schema import NoteExample


def test_load_golden_synth_fixtures(tmp_path):
    fixture = {
        "input": "Zahn 36: Anästhesie, Füllung zweiflächig.",
        "expected_codes": ["0090", "2080"],
    }
    (tmp_path / "002_synth_02_fuellung.json").write_text(json.dumps(fixture), encoding="utf-8")
    (tmp_path / "not_a_fixture.txt").write_text("ignore me", encoding="utf-8")

    examples = load_golden_synth_fixtures(tmp_path)

    assert len(examples) == 1
    assert examples[0].text == fixture["input"]
    assert examples[0].expected_codes == ["0090", "2080"]
    assert examples[0].source == "golden_synth"


def test_load_golden_synth_fixtures_ignores_real_fixtures(tmp_path):
    # Sicherheitsnetz: reale Praxisfall-Fixtures (real_*) dürfen NIEMALS
    # geladen werden, auch nicht versehentlich - siehe Design-Spec,
    # Abschnitt "Datenherkunft & IP-Abgrenzung".
    synth_fixture = {"input": "Synthetische Notiz.", "expected_codes": ["0090"]}
    real_fixture = {"input": "Echte Patientennotiz aus der Praxis.", "expected_codes": ["0090"]}
    (tmp_path / "005_synth_05_x.json").write_text(json.dumps(synth_fixture), encoding="utf-8")
    (tmp_path / "006_real_01_x.json").write_text(json.dumps(real_fixture), encoding="utf-8")

    examples = load_golden_synth_fixtures(tmp_path)

    assert len(examples) == 1
    assert examples[0].text == "Synthetische Notiz."


def _make_examples(n: int, source: str = "generated") -> list[NoteExample]:
    return [
        NoteExample(text=f"Notiz {i}", expected_codes=["0090"], source=source)
        for i in range(n)
    ]


def test_split_dataset_is_deterministic_for_same_seed():
    examples = _make_examples(20)
    train_a, test_a = split_dataset(examples, test_fraction=0.2, seed=42)
    train_b, test_b = split_dataset(examples, test_fraction=0.2, seed=42)
    assert [e.text for e in train_a] == [e.text for e in train_b]
    assert [e.text for e in test_a] == [e.text for e in test_b]


def test_split_dataset_respects_fraction():
    examples = _make_examples(20)
    train, test = split_dataset(examples, test_fraction=0.2, seed=42)
    assert len(test) == 4
    assert len(train) == 16


def test_split_dataset_always_puts_golden_synth_in_test():
    examples = _make_examples(16, source="generated") + _make_examples(4, source="golden_synth")
    train, test = split_dataset(examples, test_fraction=0.2, seed=42)
    golden_texts = {e.text for e in examples if e.source == "golden_synth"}
    test_golden_texts = {e.text for e in test if e.source == "golden_synth"}
    assert test_golden_texts == golden_texts
    assert all(e.split == "test" for e in test)
    assert all(e.split == "train" for e in train)
```

- [ ] **Step 2: Test laufen lassen, Fehlschlag verifizieren**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dataset.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'goz_extract.dataset'`

- [ ] **Step 3: `dataset.py` implementieren**

```python
"""Lädt die Golden-Synth-Fixtures, kombiniert sie mit den generierten
Notizen und teilt alles reproduzierbar in Train/Test. Die Golden-Synth-
Fixtures landen immer im Test-Set, weil sie die höchste Qualität haben
(siehe Design-Spec).

Das Glob-Pattern "*synth*.json" ist bewusst gewählt (nicht "*.json"): im
Fixture-Ordner des Referenzsystems liegen auch real_*.json-Dateien mit
echten Praxisfällen, die laut Design-Spec (Abschnitt "Datenherkunft &
IP-Abgrenzung") NIEMALS geladen werden dürfen - das Pattern verhindert das
schon auf Dateisystem-Ebene, statt sich auf nachträgliches Filtern zu
verlassen."""
import json
import random
from pathlib import Path

from goz_extract.schema import NoteExample


def load_golden_synth_fixtures(fixture_dir: Path) -> list[NoteExample]:
    examples = []
    for path in sorted(fixture_dir.glob("*synth*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        examples.append(
            NoteExample(
                text=raw["input"],
                expected_codes=raw["expected_codes"],
                source="golden_synth",
            )
        )
    return examples


def split_dataset(
    examples: list[NoteExample], test_fraction: float, seed: int
) -> tuple[list[NoteExample], list[NoteExample]]:
    golden = [e for e in examples if e.source == "golden_synth"]
    rest = [e for e in examples if e.source != "golden_synth"]

    rng = random.Random(seed)
    shuffled = rest.copy()
    rng.shuffle(shuffled)

    n_test_from_rest = max(0, round(len(examples) * test_fraction) - len(golden))
    test_rest, train_rest = shuffled[:n_test_from_rest], shuffled[n_test_from_rest:]

    for example in train_rest:
        example.split = "train"
    for example in test_rest + golden:
        example.split = "test"

    return train_rest, test_rest + golden
```

- [ ] **Step 4: Test laufen lassen, Erfolg verifizieren**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dataset.py -v`
Expected: `5 passed`

- [ ] **Step 5: CLI-Skript zum Zusammenbau schreiben**

`scripts/build_dataset.py`:
```python
"""Kombiniert die generierten synthetischen Notizen mit den Golden-Synth-
Fixtures, splittet und schreibt train/test als JSONL.

Aufruf:
    .venv/Scripts/python.exe scripts/build_dataset.py \
        --generated data/synthetic_notes.jsonl \
        --golden-synth-dir "<pfad-zum-referenz-repo>/tests/fixtures/golden_single_v2" \
        --out-dir data/
"""
import argparse
import json
from pathlib import Path

from goz_extract.dataset import load_golden_synth_fixtures, split_dataset
from goz_extract.schema import NoteExample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", required=True, type=Path)
    parser.add_argument("--golden-synth-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generated = [
        NoteExample.model_validate_json(line)
        for line in args.generated.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # load_golden_synth_fixtures filtert real_-Fixtures bereits beim Einlesen
    # aus (Glob-Pattern "*synth*.json") - siehe dataset.py.
    golden_synth = load_golden_synth_fixtures(args.golden_synth_dir)

    train, test = split_dataset(generated + golden_synth, args.test_fraction, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_examples in [("train.jsonl", train), ("test.jsonl", test)]:
        out_path = args.out_dir / name
        with out_path.open("w", encoding="utf-8") as f:
            for example in split_examples:
                f.write(example.model_dump_json() + "\n")
        print(f"{len(split_examples)} Notizen -> {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Skript manuell ausführen**

Run:
```bash
.venv/Scripts/python.exe scripts/build_dataset.py --generated data/synthetic_notes.jsonl --golden-synth-dir "<pfad-zum-referenz-repo>/tests/fixtures/golden_single_v2" --out-dir data/
```
Expected: `data/train.jsonl` und `data/test.jsonl` existieren, Test-Set
enthält alle 5 golden_synth-Beispiele plus den anteiligen Rest.

- [ ] **Step 7: Commit**

```bash
git add src/goz_extract/dataset.py scripts/build_dataset.py tests/test_dataset.py
git commit -m "Add dataset assembly: merge generated notes with golden-synth fixtures, split train/test"
```

---

## Task 6: Retrieval-Index (BM25 + Embeddings + RRF)

**Files:**
- Create: `src/goz_extract/retrieval.py`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `GozCode` aus Task 2
- Produces: `tokenize(text: str) -> list[str]`, `BM25Index(codes: list[GozCode])` mit `.rank(query: str) -> list[str]`, `EmbeddingIndex(codes: list[GozCode], encode_fn)` mit `.rank(query: str) -> list[str]`, `reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]`, `retrieve_candidates(note_text, bm25_index, embedding_index, top_n) -> list[str]` — wird von Task 7 (Prompting) und Task 10 (Colab-Notebook) konsumiert.

- [ ] **Step 1: Failing Test schreiben**

`tests/test_retrieval.py`:
```python
import numpy as np

from goz_extract.retrieval import (
    BM25Index,
    EmbeddingIndex,
    reciprocal_rank_fusion,
    retrieve_candidates,
    tokenize,
)
from goz_extract.schema import GozCode

CODES = [
    GozCode(goz_nr="0090", bezeichnung="Intraorale Infiltrationsanästhesie"),
    GozCode(goz_nr="2080", bezeichnung="Kompositfüllung, zweiflächig, Adhäsivtechnik"),
    GozCode(goz_nr="0010", bezeichnung="Eingehende Untersuchung zur Feststellung von Erkrankungen"),
]


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Kompositfüllung, zweiflächig!") == ["kompositfüllung", "zweiflächig"]


def test_bm25_ranks_exact_term_match_first():
    index = BM25Index(CODES)
    ranking = index.rank("Kompositfüllung zweiflächig")
    assert ranking[0] == "2080"


def _fake_encode_fn(texts: list[str]) -> np.ndarray:
    # Deterministische Fake-Embeddings: Vektor = Zeichenhäufigkeit von 'a','u','n'
    def vec(t: str) -> list[float]:
        t = t.lower()
        return [t.count("a"), t.count("u"), t.count("n")]

    return np.array([vec(t) for t in texts], dtype=float)


def test_embedding_index_ranks_by_cosine_similarity():
    index = EmbeddingIndex(CODES, encode_fn=_fake_encode_fn)
    ranking = index.rank("Untersuchung")
    assert set(ranking) == {"0090", "2080", "0010"}
    assert len(ranking) == 3


def test_reciprocal_rank_fusion_prefers_items_ranked_high_in_both():
    ranking_a = ["0090", "2080", "0010"]
    ranking_b = ["2080", "0090", "0010"]
    fused = reciprocal_rank_fusion([ranking_a, ranking_b])
    assert fused[0] in {"0090", "2080"}
    assert fused[-1] == "0010"


def test_retrieve_candidates_combines_bm25_and_embeddings():
    bm25_index = BM25Index(CODES)
    embedding_index = EmbeddingIndex(CODES, encode_fn=_fake_encode_fn)
    candidates = retrieve_candidates(
        "Füllung zweiflächig nach Anästhesie", bm25_index, embedding_index, top_n=2
    )
    assert len(candidates) == 2
    assert "2080" in candidates
```

- [ ] **Step 2: Test laufen lassen, Fehlschlag verifizieren**

Run: `.venv/Scripts/python.exe -m pytest tests/test_retrieval.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'goz_extract.retrieval'`

- [ ] **Step 3: `retrieval.py` implementieren**

```python
"""BM25- und Embedding-basiertes Retrieval über die kuratierte GOZ-Codeliste,
kombiniert per Reciprocal Rank Fusion (RRF) — die RAG-Baseline für den
Vergleich gegen das LoRA-Finetune. Bewusst mit injizierbarer encode_fn
gebaut, damit die Kern-Logik ohne ein echtes Embedding-Modell testbar ist."""
import re
from typing import Callable

import numpy as np
from rank_bm25 import BM25Okapi

from goz_extract.schema import GozCode


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    def __init__(self, codes: list[GozCode]) -> None:
        self._codes = codes
        self._corpus = [tokenize(c.bezeichnung) for c in codes]
        self._bm25 = BM25Okapi(self._corpus)

    def rank(self, query: str) -> list[str]:
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1]
        return [self._codes[i].goz_nr for i in order]


class EmbeddingIndex:
    def __init__(self, codes: list[GozCode], encode_fn: Callable[[list[str]], np.ndarray]) -> None:
        self._codes = codes
        self._encode_fn = encode_fn
        embeddings = encode_fn([c.bezeichnung for c in codes])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._embeddings = embeddings / norms

    def rank(self, query: str) -> list[str]:
        query_vec = self._encode_fn([query])[0]
        norm = np.linalg.norm(query_vec) or 1.0
        query_vec = query_vec / norm
        scores = self._embeddings @ query_vec
        order = np.argsort(scores)[::-1]
        return [self._codes[i].goz_nr for i in order]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, goz_nr in enumerate(ranking):
            scores[goz_nr] = scores.get(goz_nr, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda goz_nr: scores[goz_nr], reverse=True)


def retrieve_candidates(
    note_text: str, bm25_index: BM25Index, embedding_index: EmbeddingIndex, top_n: int
) -> list[str]:
    fused = reciprocal_rank_fusion([bm25_index.rank(note_text), embedding_index.rank(note_text)])
    return fused[:top_n]
```

- [ ] **Step 4: Test laufen lassen, Erfolg verifizieren**

Run: `.venv/Scripts/python.exe -m pytest tests/test_retrieval.py -v`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/goz_extract/retrieval.py tests/test_retrieval.py
git commit -m "Add BM25 + embedding retrieval with RRF fusion for the RAG baseline"
```

---

## Task 7: Prompting & Antwort-Parsing

**Files:**
- Create: `src/goz_extract/prompting.py`
- Test: `tests/test_prompting.py`

**Interfaces:**
- Consumes: `GozCode` aus Task 2
- Produces: `build_extraction_prompt(note_text: str, candidates: list[GozCode] | None = None) -> str`, `parse_code_list_response(raw_text: str, valid_codes: set[str]) -> list[str]` — beide werden von Task 9 (Inferenz) und Task 10 (Colab-Notebook) konsumiert; identisches Antwortformat für RAG-Baseline und Finetune ist Voraussetzung für Task 8 (Evaluation).

- [ ] **Step 1: Failing Test schreiben**

`tests/test_prompting.py`:
```python
from goz_extract.prompting import build_extraction_prompt, parse_code_list_response
from goz_extract.schema import GozCode

NOTE = "Zahn 36: Infiltrationsanästhesie, Karies excaviert, Kompositfüllung zweiflächig gelegt."
CANDIDATES = [
    GozCode(goz_nr="0090", bezeichnung="Intraorale Infiltrationsanästhesie"),
    GozCode(goz_nr="2080", bezeichnung="Kompositfüllung, zweiflächig"),
]


def test_finetune_prompt_has_no_candidate_list():
    prompt = build_extraction_prompt(NOTE, candidates=None)
    assert NOTE in prompt
    assert "0090" not in prompt


def test_rag_prompt_includes_candidates():
    prompt = build_extraction_prompt(NOTE, candidates=CANDIDATES)
    assert NOTE in prompt
    assert "0090" in prompt
    assert "2080" in prompt
    assert "Intraorale Infiltrationsanästhesie" in prompt


def test_parse_code_list_response_plain_list():
    result = parse_code_list_response("0090, 2080", valid_codes={"0090", "2080", "0010"})
    assert result == ["0090", "2080"]


def test_parse_code_list_response_filters_invalid_codes():
    result = parse_code_list_response("0090, 9999, 2080", valid_codes={"0090", "2080"})
    assert result == ["0090", "2080"]


def test_parse_code_list_response_dedupes_preserving_order():
    result = parse_code_list_response("2080, 0090, 2080", valid_codes={"0090", "2080"})
    assert result == ["2080", "0090"]


def test_parse_code_list_response_extracts_from_prose():
    text = "Die passenden Codes sind 0090 (Anästhesie) und 2080 (Füllung)."
    result = parse_code_list_response(text, valid_codes={"0090", "2080"})
    assert result == ["0090", "2080"]


def test_parse_code_list_response_empty_when_nothing_matches():
    result = parse_code_list_response("Keine passenden Codes gefunden.", valid_codes={"0090"})
    assert result == []
```

- [ ] **Step 2: Test laufen lassen, Fehlschlag verifizieren**

Run: `.venv/Scripts/python.exe -m pytest tests/test_prompting.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'goz_extract.prompting'`

- [ ] **Step 3: `prompting.py` implementieren**

```python
"""Gemeinsames Prompt-Format für RAG-Baseline und Finetune, damit beide
Ansätze identisch ausgewertet werden können. candidates=None -> Finetune-
Modus (Wissen steckt in den LoRA-Gewichten, kein Retrieval-Kontext).
candidates=[...] -> RAG-Modus (Retrieval-Kandidaten als Prompt-Kontext)."""
import re

from goz_extract.schema import GozCode

_INSTRUCTION = (
    "Extrahiere alle zutreffenden GOZ-Ziffern aus der folgenden zahnärztlichen "
    "Behandlungsnotiz. Antworte ausschließlich mit einer kommagetrennten Liste "
    "der Ziffern, ohne weiteren Text."
)

_CODE_PATTERN = re.compile(r"Ä?\d{3,4}")


def build_extraction_prompt(note_text: str, candidates: list[GozCode] | None = None) -> str:
    if candidates is None:
        return f"{_INSTRUCTION}\n\nNotiz:\n{note_text}"

    candidate_list = "\n".join(f"- {c.goz_nr}: {c.bezeichnung}" for c in candidates)
    return (
        f"{_INSTRUCTION}\n\n"
        f"Wähle ausschließlich aus den folgenden Kandidaten-Codes:\n{candidate_list}\n\n"
        f"Notiz:\n{note_text}"
    )


def parse_code_list_response(raw_text: str, valid_codes: set[str]) -> list[str]:
    found = _CODE_PATTERN.findall(raw_text)
    result: list[str] = []
    for code in found:
        if code in valid_codes and code not in result:
            result.append(code)
    return result
```

- [ ] **Step 4: Test laufen lassen, Erfolg verifizieren**

Run: `.venv/Scripts/python.exe -m pytest tests/test_prompting.py -v`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/goz_extract/prompting.py tests/test_prompting.py
git commit -m "Add shared extraction prompt building and response parsing"
```

---

## Task 8: Evaluations-Metriken

**Files:**
- Create: `src/goz_extract/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Produces: `precision_recall_f1(predicted: list[str], expected: list[str]) -> dict[str, float]`, `exact_match(predicted: list[str], expected: list[str]) -> bool`, `evaluate_predictions(pairs: list[tuple[list[str], list[str]]]) -> dict[str, float]` — wird von Task 11 (Eval-Report) konsumiert.

- [ ] **Step 1: Failing Test schreiben**

`tests/test_evaluate.py`:
```python
from goz_extract.evaluate import evaluate_predictions, exact_match, precision_recall_f1


def test_precision_recall_f1_perfect_match():
    result = precision_recall_f1(["0090", "2080"], ["0090", "2080"])
    assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_precision_recall_f1_partial_match():
    result = precision_recall_f1(["0090", "9999"], ["0090", "2080"])
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert round(result["f1"], 4) == 0.5


def test_precision_recall_f1_empty_prediction_and_empty_expected():
    result = precision_recall_f1([], [])
    assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_precision_recall_f1_empty_prediction_nonempty_expected():
    result = precision_recall_f1([], ["0090"])
    assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_exact_match_ignores_order():
    assert exact_match(["2080", "0090"], ["0090", "2080"]) is True
    assert exact_match(["0090"], ["0090", "2080"]) is False


def test_evaluate_predictions_averages_across_pairs():
    pairs = [
        (["0090", "2080"], ["0090", "2080"]),
        ([], ["0090"]),
    ]
    result = evaluate_predictions(pairs)
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["exact_match_rate"] == 0.5
```

- [ ] **Step 2: Test laufen lassen, Fehlschlag verifizieren**

Run: `.venv/Scripts/python.exe -m pytest tests/test_evaluate.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'goz_extract.evaluate'`

- [ ] **Step 3: `evaluate.py` implementieren**

```python
"""Precision/Recall/F1 und Exact-Match pro Notiz, gemittelt über ein
Test-Set - die Vergleichsmetrik zwischen RAG-Baseline und Finetune."""


def precision_recall_f1(predicted: list[str], expected: list[str]) -> dict[str, float]:
    predicted_set, expected_set = set(predicted), set(expected)

    if not predicted_set and not expected_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted_set or not expected_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    true_positives = len(predicted_set & expected_set)
    precision = true_positives / len(predicted_set)
    recall = true_positives / len(expected_set)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def exact_match(predicted: list[str], expected: list[str]) -> bool:
    return set(predicted) == set(expected)


def evaluate_predictions(pairs: list[tuple[list[str], list[str]]]) -> dict[str, float]:
    metrics = [precision_recall_f1(predicted, expected) for predicted, expected in pairs]
    n = len(metrics)
    exact_matches = sum(exact_match(predicted, expected) for predicted, expected in pairs)
    return {
        "precision": sum(m["precision"] for m in metrics) / n,
        "recall": sum(m["recall"] for m in metrics) / n,
        "f1": sum(m["f1"] for m in metrics) / n,
        "exact_match_rate": exact_matches / n,
    }
```

- [ ] **Step 4: Test laufen lassen, Erfolg verifizieren**

Run: `.venv/Scripts/python.exe -m pytest tests/test_evaluate.py -v`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/goz_extract/evaluate.py tests/test_evaluate.py
git commit -m "Add precision/recall/F1 and exact-match evaluation metrics"
```

---

## Task 9: Modell-Inferenz-Abstraktion

**Files:**
- Create: `src/goz_extract/inference.py`
- Test: `tests/test_inference.py`

**Interfaces:**
- Consumes: `build_extraction_prompt`, `parse_code_list_response` aus Task 7
- Produces: `load_model(model_id: str, adapter_path: str | None = None) -> tuple[model, tokenizer]`, `generate_codes(model, tokenizer, note_text: str, valid_codes: set[str], candidates: list[GozCode] | None = None) -> list[str]` — wird von Task 12 (Streamlit-Demo) konsumiert. Im Colab-Notebook (Task 10) wird dasselbe Modul importiert.

Dieses Modul braucht ein ~6 GB großes Modell und (für sinnvolle
Geschwindigkeit) eine GPU — nicht automatisiert in der lokalen Testsuite
lauffähig. Die Kernlogik (Prompt-Bau, Parsing) ist bereits in Task 7 über
Unit-Tests abgedeckt; hier wird nur die dünne Lade-/Generierungs-Schicht
ergänzt, mit einem übersprungenen Test als Dokumentation der erwarteten
Schnittstelle plus manueller Verifikation.

- [ ] **Step 1: Test mit Skip-Markierung schreiben**

`tests/test_inference.py`:
```python
"""generate_codes braucht ein reales (~6GB) Sprachmodell und wird deshalb
standardmäßig übersprungen - siehe Task 9 im Implementierungsplan für die
manuelle Verifikation auf Colab. Der Test dokumentiert die erwartete
Schnittstelle und läuft, wenn RUN_MODEL_TESTS=1 gesetzt ist."""
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="Braucht ein reales, ~6GB großes Sprachmodell - siehe Task 9 im Implementierungsplan",
)


def test_generate_codes_returns_valid_codes_only():
    from goz_extract.inference import generate_codes, load_model

    model, tokenizer = load_model("meta-llama/Llama-3.2-3B-Instruct")
    result = generate_codes(
        model,
        tokenizer,
        note_text="Zahn 36: Infiltrationsanästhesie, Kompositfüllung zweiflächig.",
        valid_codes={"0090", "2080", "0010"},
    )
    assert all(code in {"0090", "2080", "0010"} for code in result)
```

- [ ] **Step 2: Test laufen lassen, Skip verifizieren**

Run: `.venv/Scripts/python.exe -m pytest tests/test_inference.py -v`
Expected: `1 skipped`

- [ ] **Step 3: `inference.py` implementieren**

```python
"""Lädt Llama-3.2-3B-Instruct (optional mit LoRA-Adapter) und generiert
GOZ-Code-Listen. Gemeinsam genutzt vom Colab-Notebook (Training + Baseline-
/Finetune-Inferenz über das Test-Set) und der Streamlit-Demo (Einzelanfragen)."""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from goz_extract.prompting import build_extraction_prompt, parse_code_list_response
from goz_extract.schema import GozCode


def load_model(model_id: str, adapter_path: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate_codes(
    model,
    tokenizer,
    note_text: str,
    valid_codes: set[str],
    candidates: list[GozCode] | None = None,
    max_new_tokens: int = 64,
) -> list[str]:
    prompt = build_extraction_prompt(note_text, candidates=candidates)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = tokenizer.decode(output_ids[0][inputs.shape[1]:], skip_special_tokens=True)
    return parse_code_list_response(generated, valid_codes)
```

- [ ] **Step 4: Manuelle Verifikation auf Colab (siehe Task 10, gleiche Umgebung)**

Wird zusammen mit Task 10 verifiziert — `inference.py` wird dort importiert
und real gegen die geladenen Modelle getestet. Kein separater Schritt hier,
um doppelte Colab-Sessions zu vermeiden.

- [ ] **Step 5: Commit**

```bash
git add src/goz_extract/inference.py tests/test_inference.py
git commit -m "Add model loading and code-list generation, shared by Colab notebook and demo"
```

---

## Task 10: Colab-Notebook — Training & Inferenz

**Files:**
- Create: `notebooks/train_and_infer.ipynb`

**Interfaces:**
- Consumes: `GozCode`/`NoteExample` (Task 2), `retrieve_candidates` + `BM25Index`/`EmbeddingIndex` (Task 6), `build_extraction_prompt`/`parse_code_list_response` (Task 7), `load_model`/`generate_codes` (Task 9), `data/goz_codes.json` + `data/train.jsonl` + `data/test.jsonl` (Task 3/5)
- Produces: `results/predictions_rag.jsonl`, `results/predictions_finetune.jsonl` (je eine Zeile pro Test-Notiz: `{"text": ..., "expected_codes": [...], "predicted_codes": [...]}`), LoRA-Adapter-Ordner `adapters/goz-extract-llama32-3b/` — beide werden von Task 11 (Eval-Report) und Task 12 (Streamlit-Demo) konsumiert.

Dieser Task läuft komplett manuell auf Colab (T4-GPU) — kein automatisierter
Test in diesem Environment möglich. Jeder Schritt listet das erwartete
Ergebnis zur Verifikation.

- [ ] **Step 1: Notebook mit Setup- und Daten-Zellen anlegen**

`notebooks/train_and_infer.ipynb` (Struktur — als gültiges `.ipynb`-JSON
anlegen, jede Markdown-Zeile wird eine eigene Notebook-Zelle):

Zelle 1 (code):
```python
!pip install -q transformers peft trl bitsandbytes accelerate datasets sentence-transformers rank-bm25 pydantic
```

Zelle 2 (code):
```python
from huggingface_hub import notebook_login
notebook_login()  # HF-Token mit akzeptierter Llama-3.2-Lizenz eingeben
```

Zelle 3 (code):
```python
# Repo-Dateien hochladen: src/goz_extract/, data/goz_codes.json, data/train.jsonl, data/test.jsonl
from google.colab import files
uploaded = files.upload()  # als .zip hochladen und entpacken, siehe README-Setup-Abschnitt
!unzip -o goz-extract-src.zip -d .
import sys; sys.path.insert(0, "src")
```

Zelle 4 (code):
```python
import json
from pathlib import Path

from goz_extract.schema import GozCode, NoteExample

codes = [GozCode(**c) for c in json.loads(Path("data/goz_codes.json").read_text(encoding="utf-8"))]
train = [NoteExample.model_validate_json(l) for l in Path("data/train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
test = [NoteExample.model_validate_json(l) for l in Path("data/test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
valid_codes = {c.goz_nr for c in codes}
print(len(codes), len(train), len(test))
```
Erwartete Ausgabe: `55 <train-Anzahl> <test-Anzahl>` (Zahlen aus Task 5,
Step 6).

- [ ] **Step 2: RAG-Baseline-Zellen anlegen**

Zelle 5 (code):
```python
from sentence_transformers import SentenceTransformer

from goz_extract.retrieval import BM25Index, EmbeddingIndex, retrieve_candidates

embed_model = SentenceTransformer("intfloat/multilingual-e5-base")


def encode_fn(texts):
    return embed_model.encode([f"passage: {t}" for t in texts], normalize_embeddings=False)


bm25_index = BM25Index(codes)
embedding_index = EmbeddingIndex(codes, encode_fn=encode_fn)
code_by_nr = {c.goz_nr: c for c in codes}
```

Zelle 6 (code):
```python
from goz_extract.inference import generate_codes, load_model

base_model, tokenizer = load_model("meta-llama/Llama-3.2-3B-Instruct")

rag_predictions = []
for example in test:
    candidate_codes = [code_by_nr[nr] for nr in retrieve_candidates(example.text, bm25_index, embedding_index, top_n=12)]
    predicted = generate_codes(base_model, tokenizer, example.text, valid_codes, candidates=candidate_codes)
    rag_predictions.append({"text": example.text, "expected_codes": example.expected_codes, "predicted_codes": predicted})

print(rag_predictions[0])
```
Erwartetes Verhalten: läuft ohne Fehler über alle Test-Notizen durch
(Dauer: ca. 1-3 Minuten auf T4 für ~80-100 Notizen), `predicted_codes` in
jedem Eintrag ist eine Liste gültiger GOZ-Ziffern.

Zelle 7 (code):
```python
import json
Path("results").mkdir(exist_ok=True)
with open("results/predictions_rag.jsonl", "w", encoding="utf-8") as f:
    for row in rag_predictions:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
```

- [ ] **Step 3: LoRA-Trainings-Zellen anlegen**

Zelle 8 (code):
```python
from datasets import Dataset
from goz_extract.prompting import build_extraction_prompt

def to_chat_text(example):
    prompt = build_extraction_prompt(example.text, candidates=None)
    completion = ", ".join(example.expected_codes)
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

train_dataset = Dataset.from_list([to_chat_text(e) for e in train])
```

Zelle 9 (code):
```python
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from transformers import AutoModelForCausalLM

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
# Fürs Training brauchen wir eine 4-bit-quantisierte Kopie (QLoRA) - nicht
# über load_model() (das lädt in bfloat16 ohne Quantisierung, siehe Task 9),
# sondern direkt mit quantization_config.
train_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct", quantization_config=bnb_config, device_map="auto"
)

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

sft_config = SFTConfig(
    output_dir="adapters/goz-extract-llama32-3b",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
)

trainer = SFTTrainer(
    model=train_model,
    train_dataset=train_dataset,
    args=sft_config,
    peft_config=lora_config,
)
trainer.train()
trainer.save_model("adapters/goz-extract-llama32-3b")
```
Erwartetes Verhalten: Trainings-Loss sinkt sichtbar über die Logging-Schritte;
läuft ohne CUDA-OOM auf T4 durch (bei OOM: `per_device_train_batch_size`
auf 2 reduzieren, `gradient_accumulation_steps` auf 8 erhöhen).

- [ ] **Step 4: Finetune-Inferenz-Zellen anlegen**

Zelle 10 (code):
```python
finetuned_model, ft_tokenizer = load_model(
    "meta-llama/Llama-3.2-3B-Instruct", adapter_path="adapters/goz-extract-llama32-3b"
)

finetune_predictions = []
for example in test:
    predicted = generate_codes(finetuned_model, ft_tokenizer, example.text, valid_codes, candidates=None)
    finetune_predictions.append({"text": example.text, "expected_codes": example.expected_codes, "predicted_codes": predicted})

with open("results/predictions_finetune.jsonl", "w", encoding="utf-8") as f:
    for row in finetune_predictions:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(finetune_predictions[0])
```
Erwartetes Verhalten: läuft über alle Test-Notizen durch, `predicted_codes`
enthält gültige GOZ-Ziffern.

- [ ] **Step 5: Artefakte herunterladen und lokal ablegen**

Manuell in Colab: `results/predictions_rag.jsonl`,
`results/predictions_finetune.jsonl` und den Ordner
`adapters/goz-extract-llama32-3b/` herunterladen (Colab-Dateibrowser oder
`files.download(...)`), lokal nach
`goz-finetune-vs-rag/results/` bzw. `goz-finetune-vs-rag/adapters/` legen.

- [ ] **Step 6: Commit (Notebook + heruntergeladene Artefakte)**

```bash
git add notebooks/train_and_infer.ipynb results/predictions_rag.jsonl results/predictions_finetune.jsonl
git commit -m "Add Colab notebook for QLoRA training and RAG-baseline/finetune inference over the test set"
```

(LoRA-Adapter-Gewichte unter `adapters/` sind groß — vor dem Commit prüfen,
ob sie stattdessen per `.gitignore` ausgeschlossen und separat z.B. auf
Hugging Face Hub hochgeladen werden sollen; im README verlinken.)

---

## Task 11: Eval-Report

**Files:**
- Create: `src/goz_extract/report.py`
- Create: `scripts/run_eval.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `evaluate_predictions` aus Task 8, `results/predictions_rag.jsonl` + `results/predictions_finetune.jsonl` aus Task 10
- Produces: `render_results_table(metrics_by_approach: dict[str, dict[str, float]]) -> str` — Markdown-Tabelle für `results/results.md` und das README.

- [ ] **Step 1: Failing Test schreiben**

`tests/test_report.py`:
```python
from goz_extract.report import render_results_table


def test_render_results_table_contains_all_approaches_and_metrics():
    metrics = {
        "RAG-Baseline": {"precision": 0.70, "recall": 0.65, "f1": 0.674, "exact_match_rate": 0.40},
        "LoRA-Finetune": {"precision": 0.85, "recall": 0.80, "f1": 0.824, "exact_match_rate": 0.60},
    }
    table = render_results_table(metrics)
    assert "RAG-Baseline" in table
    assert "LoRA-Finetune" in table
    assert "0.70" in table or "0.700" in table
    assert "Precision" in table and "Recall" in table and "F1" in table and "Exact Match" in table
```

- [ ] **Step 2: Test laufen lassen, Fehlschlag verifizieren**

Run: `.venv/Scripts/python.exe -m pytest tests/test_report.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'goz_extract.report'`

- [ ] **Step 3: `report.py` implementieren**

```python
"""Rendert die Vergleichs-Metriken als Markdown-Tabelle, im Stil von
sql-agent/evals/results.md."""


def render_results_table(metrics_by_approach: dict[str, dict[str, float]]) -> str:
    header = "| Ansatz | Precision | Recall | F1 | Exact Match |\n"
    separator = "|---|---|---|---|---|\n"
    rows = ""
    for approach, metrics in metrics_by_approach.items():
        rows += (
            f"| {approach} | {metrics['precision']:.2f} | {metrics['recall']:.2f} | "
            f"{metrics['f1']:.2f} | {metrics['exact_match_rate']:.2f} |\n"
        )
    return header + separator + rows
```

- [ ] **Step 4: Test laufen lassen, Erfolg verifizieren**

Run: `.venv/Scripts/python.exe -m pytest tests/test_report.py -v`
Expected: `1 passed`

- [ ] **Step 5: CLI-Skript schreiben**

`scripts/run_eval.py`:
```python
"""Liest die von Colab exportierten Predictions, berechnet die Metriken
und schreibt die Ergebnistabelle.

Aufruf:
    .venv/Scripts/python.exe scripts/run_eval.py --results-dir results/
"""
import argparse
import json
from pathlib import Path

from goz_extract.evaluate import evaluate_predictions
from goz_extract.report import render_results_table

APPROACH_FILES = {
    "RAG-Baseline": "predictions_rag.jsonl",
    "LoRA-Finetune": "predictions_finetune.jsonl",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    args = parser.parse_args()

    metrics_by_approach = {}
    for approach, filename in APPROACH_FILES.items():
        path = args.results_dir / filename
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        pairs = [(row["predicted_codes"], row["expected_codes"]) for row in rows]
        metrics_by_approach[approach] = evaluate_predictions(pairs)

    table = render_results_table(metrics_by_approach)
    out_path = args.results_dir / "results.md"
    out_path.write_text(f"# Ergebnisse: LoRA-Finetune vs. RAG-Baseline\n\n{table}", encoding="utf-8")
    print(table)
    print(f"Geschrieben nach {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Skript manuell ausführen (braucht die Colab-Artefakte aus Task 10)**

Run:
```bash
.venv/Scripts/python.exe scripts/run_eval.py --results-dir results/
```
Expected: Tabelle wird auf der Konsole ausgegeben und nach
`results/results.md` geschrieben.

- [ ] **Step 7: Commit**

```bash
git add src/goz_extract/report.py scripts/run_eval.py tests/test_report.py results/results.md
git commit -m "Add eval report generation: precision/recall/F1/exact-match table"
```

---

## Task 12: Streamlit-Demo

**Files:**
- Create: `app.py`

**Interfaces:**
- Consumes: `data/goz_codes.json` (Task 3), `retrieve_candidates`/`BM25Index`/`EmbeddingIndex` (Task 6), `load_model`/`generate_codes` (Task 9), LoRA-Adapter aus `adapters/goz-extract-llama32-3b/` (Task 10)

Braucht das lokal vorhandene, trainierte Modell (aus Task 10) — kein
automatisierter Test, manuelle Verifikation im Browser, analog zum
Playwright-Vorgehen bei `sql-agent`.

- [ ] **Step 1: `app.py` schreiben**

```python
"""Streamlit-Demo: Behandlungsnotiz eingeben, Vorhersage von RAG-Baseline
und LoRA-Finetune nebeneinander vergleichen."""
import json
from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer

from goz_extract.inference import generate_codes, load_model
from goz_extract.retrieval import BM25Index, EmbeddingIndex, retrieve_candidates
from goz_extract.schema import GozCode

st.set_page_config(page_title="GOZ-Extraktion: Finetune vs. RAG", layout="wide")


@st.cache_resource
def load_resources():
    codes = [
        GozCode(**c)
        for c in json.loads(Path("data/goz_codes.json").read_text(encoding="utf-8"))
    ]
    code_by_nr = {c.goz_nr: c for c in codes}
    valid_codes = set(code_by_nr)

    embed_model = SentenceTransformer("intfloat/multilingual-e5-base")
    encode_fn = lambda texts: embed_model.encode([f"passage: {t}" for t in texts])
    bm25_index = BM25Index(codes)
    embedding_index = EmbeddingIndex(codes, encode_fn=encode_fn)

    base_model, base_tokenizer = load_model("meta-llama/Llama-3.2-3B-Instruct")
    finetuned_model, ft_tokenizer = load_model(
        "meta-llama/Llama-3.2-3B-Instruct", adapter_path="adapters/goz-extract-llama32-3b"
    )
    return code_by_nr, valid_codes, bm25_index, embedding_index, base_model, base_tokenizer, finetuned_model, ft_tokenizer


(code_by_nr, valid_codes, bm25_index, embedding_index,
 base_model, base_tokenizer, finetuned_model, ft_tokenizer) = load_resources()

st.title("GOZ-Code-Extraktion: LoRA-Finetuning vs. RAG-Baseline")
note_text = st.text_area(
    "Behandlungsnotiz",
    "Zahn 36: Infiltrationsanästhesie, Karies excaviert, Kompositfüllung zweiflächig gelegt.",
)

if st.button("Extrahieren"):
    candidate_nrs = retrieve_candidates(note_text, bm25_index, embedding_index, top_n=12)
    candidates = [code_by_nr[nr] for nr in candidate_nrs]

    col_rag, col_finetune = st.columns(2)

    with col_rag:
        st.subheader("RAG-Baseline")
        rag_codes = generate_codes(base_model, base_tokenizer, note_text, valid_codes, candidates=candidates)
        st.write([f"{nr}: {code_by_nr[nr].bezeichnung}" for nr in rag_codes])
        with st.expander("Retrieval-Kandidaten (Prompt-Kontext)"):
            st.write([f"{nr}: {code_by_nr[nr].bezeichnung}" for nr in candidate_nrs])

    with col_finetune:
        st.subheader("LoRA-Finetune")
        finetune_codes = generate_codes(finetuned_model, ft_tokenizer, note_text, valid_codes, candidates=None)
        st.write([f"{nr}: {code_by_nr[nr].bezeichnung}" for nr in finetune_codes])
        st.caption("Kein Retrieval zur Inferenzzeit — Wissen steckt in den LoRA-Gewichten.")
```

- [ ] **Step 2: Manuell starten und im Browser verifizieren**

Run:
```bash
.venv/Scripts/python.exe -m streamlit run app.py
```
Erwartetes Verhalten: Seite lädt (Erstladen dauert wegen Modell-Download/
-Laden mehrere Minuten), Eingabe der Beispielnotiz + Klick auf
"Extrahieren" zeigt zwei Spalten mit unterschiedlichen Code-Listen und die
Retrieval-Kandidaten im aufklappbaren Bereich.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "Add Streamlit demo comparing RAG-baseline and LoRA-finetune predictions"
```

---

## Task 13: README

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: `results/results.md` (Task 11) für die Ergebnistabelle

- [ ] **Step 1: README schreiben**

```markdown
# GOZ-Code-Extraktion: LoRA-Finetuning vs. RAG-Baseline

Portfolio-Projekt: Ein LoRA-feingetuntes Llama-3.2-3B-Instruct extrahiert
GOZ-Ziffern aus zahnärztlichen Behandlungsnotizen — verglichen gegen eine
RAG-Baseline auf demselben, unveränderten Basismodell. Ziel: eine konkrete,
messbare Antwort auf "schlägt Finetuning RAG?".

## Aufgabe

Aus einer Behandlungsnotiz (kann mehrere Behandlungsschritte einer Sitzung
beschreiben) werden alle zutreffenden GOZ-Ziffern extrahiert (Multi-Label),
aus einem Label-Space von 55 Codes (Kategorien "Allgemeine zahnärztliche
Leistungen" + "Konservierende Leistungen" der amtlichen Gebührenordnung).

## Zwei Wege, Domänenwissen einzubringen

- **RAG-Baseline:** BM25 + Embeddings (`multilingual-e5-base`) liefern
  Kandidaten-Codes, dasselbe Basismodell wählt daraus per Prompt.
- **LoRA-Finetune:** Domänenwissen steckt in den LoRA-Gewichten, kein
  Retrieval zur Inferenzzeit.

## Ergebnisse

<!-- Tabelle aus results/results.md einfügen, sobald Task 10+11 gelaufen sind -->

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
cp .env.example .env  # ANTHROPIC_API_KEY eintragen
.venv/Scripts/python.exe -m pytest tests/ -v
```

Trainingsdaten generieren und Codeliste kuratieren: siehe
`scripts/curate_codes.py`, `scripts/generate_data.py`,
`scripts/build_dataset.py`.

Training + Inferenz laufen auf Colab: `notebooks/train_and_infer.ipynb`
(braucht HF-Zustimmung zur Llama-3.2-Lizenz).

Demo starten (braucht die von Colab heruntergeladenen Artefakte unter
`adapters/`): `.venv/Scripts/python.exe -m streamlit run app.py`

## Datenherkunft

Nur die amtliche GOZ-Codeliste (öffentliche Gebührenordnung) und komplett
selbst generierte, synthetische Trainingsdaten. Details siehe
`docs/superpowers/specs/2026-07-27-goz-finetune-vs-rag-design.md`,
Abschnitt "Datenherkunft & IP-Abgrenzung".

## Limitierungen

- Label-Space auf 55 Alltags-Codes begrenzt (nicht die vollen 221 GOZ-Codes)
- Trainingsdaten sind synthetisch (LLM-generiert), kein Abgleich mit realen
  Praxisfällen im großen Stil
- RAG-Baseline nutzt eine vereinfachte Retrieval-Pipeline ohne die
  Segmentierungs- und Validierungsschritte eines Produktivsystems
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Add README with project overview, setup instructions and results placeholder"
```
