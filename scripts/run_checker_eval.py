"""Wertet den Checker-Node über das Testset aus.

Drei Backends, weil drei verschiedene Fragen dahinterstecken:

* ``--backend oracle`` — perfekte Auswahl aus dem Pool. Kein Modell, kein
  Netz. Muss die 0.80 aus ``analyze_merge_headroom.py`` reproduzieren; tut es
  das nicht, ist die Verdrahtung kaputt und jede Modellzahl wertlos.
* ``--backend jsonl`` — liest fertige Checker-Vorhersagen (z.B. aus dem
  Colab-Notebook, wo das Basismodell läuft). Das ist die **faire** Zeile für
  die Ergebnistabelle: gleiches Basismodell wie RAG und Finetune.
* ``--backend anthropic`` — Sonde mit einem API-Modell. Beantwortet nur, ob
  die Auswahlaufgabe überhaupt lösbar ist, und **gehört nicht in dieselbe
  Tabelle** wie die Zeilen auf Llama-3.2-3B: anderes Modell, andere
  Gewichtsklasse. Nützlich, um vor einer GPU-Sitzung zu wissen, ob sich der
  Aufwand lohnt.

Aufruf:
    python scripts/run_checker_eval.py --results-dir results/ --backend oracle
    python scripts/run_checker_eval.py --results-dir results/ --backend anthropic
    python scripts/run_checker_eval.py --results-dir results/ --backend jsonl \
        --checker-predictions results/predictions_graph_checker.jsonl
"""
import argparse
import json
import os
import sys
from pathlib import Path

from goz_extract.checker import build_checker_prompt, checker_predict, pool_candidates
from goz_extract.evaluate import evaluate_predictions
from goz_extract.graph_merge import graph_predict, load_valid_codes
from goz_extract.prompting import parse_code_list_response
from goz_extract.report import render_results_table
from goz_extract.schema import GozCode


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} fehlt.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_codes(path: Path) -> dict[str, GozCode]:
    codes = [GozCode(**c) for c in json.loads(path.read_text(encoding="utf-8"))]
    return {c.goz_nr: c for c in codes}


# --- Backends ---------------------------------------------------------------
def make_oracle_backend(expected_by_note: dict[str, list[str]]):
    """Wählt aus dem Pool genau das, was richtig ist. Obergrenze, kein Modell."""

    def select(note_text, candidates):
        expected = set(expected_by_note.get(note_text, []))
        return [c.goz_nr for c in candidates if c.goz_nr in expected]

    return select


def make_jsonl_backend(path: Path):
    """Liest fertige Auswahlen (Colab-Export) statt selbst zu generieren."""
    rows = load_rows(path)
    by_note = {row["text"]: row["predicted_codes"] for row in rows}
    if len(by_note) != len(rows):
        raise ValueError(f"{path}: doppelte Notiztexte, Join nicht eindeutig")

    def select(note_text, candidates):
        if note_text not in by_note:
            raise KeyError(f"Keine Checker-Vorhersage für: {note_text[:60]!r}")
        return by_note[note_text]

    return select


def make_anthropic_backend(model: str, code_by_nr: dict[str, GozCode], valid_codes: set[str]):
    """Sonde mit einem API-Modell — anderes Modell als die Tabellenzeilen."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover - Abhaengigkeit ist optional
        sys.exit("Paket 'anthropic' fehlt: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        env_file = Path(__file__).resolve().parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY nicht gesetzt (Umgebung oder .env).")

    client = anthropic.Anthropic(api_key=api_key)

    def select(note_text, candidates):
        prompt = build_checker_prompt(
            note_text, [c.goz_nr for c in candidates], code_by_nr
        )
        response = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(block.text for block in response.content if block.type == "text")
        return parse_code_list_response(raw, valid_codes)

    return select


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--codes", type=Path, default=Path("data/goz_codes.json"))
    parser.add_argument("--backend", choices=["oracle", "jsonl", "anthropic"], required=True)
    parser.add_argument("--checker-predictions", type=Path, help="nur für --backend jsonl")
    parser.add_argument("--model", default="claude-sonnet-4-5", help="nur für --backend anthropic")
    parser.add_argument("--limit", type=int, help="nur die ersten N Notizen (Probelauf)")
    parser.add_argument("--export", type=Path, help="Auswahl des Checkers als JSONL sichern")
    args = parser.parse_args()

    code_by_nr = load_codes(args.codes)
    valid_codes = load_valid_codes(args.codes)
    rag_rows = load_rows(args.results_dir / "predictions_rag.jsonl")
    lora_rows = load_rows(args.results_dir / "predictions_finetune.jsonl")
    by_text = {row["text"]: row for row in lora_rows}
    joined = [(r, by_text[r["text"]]) for r in rag_rows if r["text"] in by_text]
    if len(joined) != len(rag_rows):
        raise ValueError("Join über den Notiztext unvollständig — verschiedene Testsets?")
    if args.limit:
        joined = joined[: args.limit]

    if args.backend == "oracle":
        select_fn = make_oracle_backend({r["text"]: r["expected_codes"] for r, _ in joined})
        label = "Graph + Checker (Orakel — Obergrenze, kein Modell)"
    elif args.backend == "jsonl":
        if not args.checker_predictions:
            sys.exit("--checker-predictions fehlt")
        select_fn = make_jsonl_backend(args.checker_predictions)
        label = "Graph + Checker (Llama-3.2-3B, gleiche Basis)"
    else:
        select_fn = make_anthropic_backend(args.model, code_by_nr, valid_codes)
        label = f"Graph + Checker ({args.model} — SONDE, anderes Modell)"

    pairs, review, exported = [], 0, []
    pool_sizes = []
    for i, (rag_row, lora_row) in enumerate(joined, 1):
        result = checker_predict(
            rag_row["text"],
            rag_row["predicted_codes"],
            lora_row["predicted_codes"],
            code_by_nr,
            select_fn,
        )
        pairs.append((result.predicted_codes, rag_row["expected_codes"]))
        review += result.needs_review
        pool_sizes.append(len(pool_candidates(rag_row["predicted_codes"], lora_row["predicted_codes"])))
        exported.append({
            "text": rag_row["text"],
            "expected_codes": rag_row["expected_codes"],
            "predicted_codes": result.predicted_codes,
            "pooled": result.sources["gepoolt"],
        })
        if args.backend == "anthropic" and i % 10 == 0:
            print(f"  … {i}/{len(joined)}", file=sys.stderr)

    # Vergleichszeilen aus denselben Exporten, damit die Tabelle selbsterklärend ist.
    metrics = {
        "RAG-Baseline": evaluate_predictions([(r["predicted_codes"], r["expected_codes"]) for r, _ in joined]),
        "LoRA-Finetune": evaluate_predictions([(l["predicted_codes"], r["expected_codes"]) for r, l in joined]),
        "Graph (Merge + Verifier)": evaluate_predictions([
            (graph_predict(r["predicted_codes"], l["predicted_codes"], valid_codes).predicted_codes,
             r["expected_codes"]) for r, l in joined
        ]),
        label: evaluate_predictions(pairs),
    }

    n = len(pairs)
    print(render_results_table(metrics))
    print(f"Notizen:              {n}")
    print(f"Ø Kandidaten im Pool: {sum(pool_sizes) / n:.2f}")
    print(f"needs_review:         {review}/{n} ({review / n:.0%})")
    if args.backend == "anthropic":
        print("\nHINWEIS: Sonde mit einem anderen Modell als die übrigen Zeilen —")
        print("nicht als faire Vergleichszeile veröffentlichen.")

    if args.export:
        args.export.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in exported) + "\n",
            encoding="utf-8",
        )
        print(f"Auswahl gesichert: {args.export}")


if __name__ == "__main__":
    main()
