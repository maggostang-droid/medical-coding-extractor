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
