"""Berechnet die Graph-Zeile der Ergebnistabelle offline aus den bereits
exportierten Predictions beider Einzelpfade.

Kein Colab, keine GPU: RAG- und LoRA-Vorhersagen liegen als JSONL vor, der
Graph entscheidet nur noch darüber. Gibt zusätzlich die needs_review-Quote
aus — ohne die ist die Graph-Zeile nicht ehrlich lesbar.

Aufruf:
    .venv/Scripts/python.exe scripts/run_graph_eval.py --results-dir results/
    .venv/Scripts/python.exe scripts/run_graph_eval.py --results-dir results/ \
        --rag-only-policy accept        # Merge-Schwelle gelockert
"""
import argparse
import json
from pathlib import Path

from goz_extract.evaluate import evaluate_predictions
from goz_extract.graph_merge import (
    RAG_ONLY_ACCEPT,
    RAG_ONLY_UNCERTAIN,
    graph_predict,
    load_valid_codes,
)
from goz_extract.report import render_results_table

APPROACH_FILES = {
    "RAG-Baseline": "predictions_rag.jsonl",
    "LoRA-Finetune": "predictions_finetune.jsonl",
}


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} fehlt. Erst notebooks/train_and_infer.ipynb auf Colab laufen "
            f"lassen und die Predictions-Dateien nach {path.parent} herunterladen."
        )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def join_rows(rag_rows: list[dict], lora_rows: list[dict]) -> list[tuple[dict, dict]]:
    """Paart die Zeilen beider Exporte.

    Bevorzugt den Join über den Notiztext (stabil gegen Umsortierung), fällt
    auf die Zeilenreihenfolge zurück. In beiden Fällen wird geprüft, dass die
    erwarteten Codes übereinstimmen — sonst stammen die Exporte aus
    verschiedenen Testsets und jede Auswertung wäre Unsinn.
    """
    if len(rag_rows) != len(lora_rows):
        raise ValueError(
            f"Unterschiedlich viele Zeilen: RAG {len(rag_rows)}, LoRA {len(lora_rows)}"
        )

    by_text = {row["text"]: row for row in lora_rows}
    joined: list[tuple[dict, dict]] = []
    if len(by_text) == len(lora_rows) and all(row["text"] in by_text for row in rag_rows):
        joined = [(row, by_text[row["text"]]) for row in rag_rows]
    else:
        joined = list(zip(rag_rows, lora_rows))

    for rag_row, lora_row in joined:
        if set(rag_row["expected_codes"]) != set(lora_row["expected_codes"]):
            raise ValueError(
                "expected_codes stimmen nicht überein — die Exporte gehören zu "
                f"verschiedenen Testsets. Betroffen: {rag_row['text'][:60]!r}"
            )
    return joined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--codes", type=Path, default=Path("data/goz_codes.json"))
    parser.add_argument(
        "--rag-only-policy",
        choices=[RAG_ONLY_UNCERTAIN, RAG_ONLY_ACCEPT],
        default=RAG_ONLY_UNCERTAIN,
        help="Wie mit Codes umgehen, die nur der RAG-Pfad geliefert hat.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="results.md überschreiben (ohne Flag nur Ausgabe auf stdout).",
    )
    args = parser.parse_args()

    valid_codes = load_valid_codes(args.codes)
    rag_rows = load_rows(args.results_dir / APPROACH_FILES["RAG-Baseline"])
    lora_rows = load_rows(args.results_dir / APPROACH_FILES["LoRA-Finetune"])
    joined = join_rows(rag_rows, lora_rows)

    metrics_by_approach = {
        "RAG-Baseline": evaluate_predictions(
            [(r["predicted_codes"], r["expected_codes"]) for r in rag_rows]
        ),
        "LoRA-Finetune": evaluate_predictions(
            [(r["predicted_codes"], r["expected_codes"]) for r in lora_rows]
        ),
    }

    graph_pairs = []
    review_count = 0
    rejected_count = 0
    for rag_row, lora_row in joined:
        result = graph_predict(
            rag_row["predicted_codes"],
            lora_row["predicted_codes"],
            valid_codes,
            rag_only_policy=args.rag_only_policy,
        )
        graph_pairs.append((result.predicted_codes, rag_row["expected_codes"]))
        review_count += result.needs_review
        rejected_count += len(result.rejected)

    metrics_by_approach["Graph (RAG ∥ LoRA + Verifier)"] = evaluate_predictions(graph_pairs)

    n = len(graph_pairs)
    review_rate = review_count / n
    table = render_results_table(metrics_by_approach)

    print(table)
    print(f"Notizen im Testset:        {n}")
    print(f"needs_review:              {review_count}/{n} ({review_rate:.0%})")
    print(f"vom Verifier abgelehnt:    {rejected_count} Codes (nicht im Katalog)")
    print(f"rag-only-policy:           {args.rag_only_policy}")

    if args.write:
        out_path = args.results_dir / "results.md"
        out_path.write_text(
            "# Ergebnisse: LoRA-Finetune vs. RAG-Baseline vs. Graph\n\n"
            f"{table}\n"
            f"Graph-Zeile mit `rag-only-policy={args.rag_only_policy}`; "
            f"{review_count} von {n} Notizen ({review_rate:.0%}) sind als "
            "`needs_review` markiert und würden in der Praxis von einem Menschen "
            "geprüft. Unsichere Codes zählen nicht als Vorhersage — die Precision "
            "des Graphen ist deshalb nur zusammen mit dieser Quote zu lesen.\n",
            encoding="utf-8",
        )
        print(f"Geschrieben nach {out_path}")


if __name__ == "__main__":
    main()
