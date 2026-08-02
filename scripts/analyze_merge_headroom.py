"""Diagnose: lohnt sich ein Merge der beiden Extraktor-Pfade überhaupt?

Beantwortet drei Fragen aus den vorhandenen Prediction-Exporten — ohne GPU:

1. Wie verlässlich ist ein Code, je nachdem welcher Pfad ihn geliefert hat?
2. Was bringen die naheliegenden Mengen-Strategien (Union, Schnitt, ...)?
3. Wie hoch könnte ein Auswahl-Node überhaupt kommen (Obergrenze)?

Die dritte Frage ist die wichtigste: Wenn die erwarteten Codes ohnehin selten
in der Vereinigung beider Pfade stecken, kann keine noch so schlaue
Merge-Regel etwas retten — dann ist der Graph an dieser Stelle die falsche
Antwort.

Aufruf:
    .venv/Scripts/python.exe scripts/analyze_merge_headroom.py --results-dir results/
"""
import argparse
import json
from pathlib import Path

from goz_extract.evaluate import evaluate_predictions


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    args = parser.parse_args()

    rag = load(args.results_dir / "predictions_rag.jsonl")
    lora = load(args.results_dir / "predictions_finetune.jsonl")
    rows = list(zip(rag, lora))
    n = len(rows)

    print(f"## 1. Precision nach Herkunft eines Codes ({n} Notizen)\n")
    buckets = {"von beiden": [0, 0], "nur LoRA": [0, 0], "nur RAG": [0, 0]}
    for r, l in rows:
        R, L, E = set(r["predicted_codes"]), set(l["predicted_codes"]), set(r["expected_codes"])
        for name, codes in (("von beiden", R & L), ("nur LoRA", L - R), ("nur RAG", R - L)):
            for code in codes:
                buckets[name][0 if code in E else 1] += 1
    for name, (tp, fp) in buckets.items():
        total = tp + fp
        print(f"  {name:<12} {tp:>3}/{total:<4} = {tp / total:.2f}" if total else f"  {name:<12} n/a")

    print("\n## 2. Mengen-Strategien\n")
    strategies = {
        "nur RAG": lambda R, L: R,
        "nur LoRA": lambda R, L: L,
        "Union": lambda R, L: R | L,
        "Schnittmenge": lambda R, L: R & L,
        "Merge-Regel (beide + nur-LoRA)": lambda R, L: (R & L) | (L - R),
    }
    print("| Strategie | Precision | Recall | F1 | Exact Match |")
    print("|---|---|---|---|---|")
    for name, fn in strategies.items():
        pairs = [
            (sorted(fn(set(r["predicted_codes"]), set(l["predicted_codes"]))), r["expected_codes"])
            for r, l in rows
        ]
        m = evaluate_predictions(pairs)
        print(
            f"| {name} | {m['precision']:.2f} | {m['recall']:.2f} | "
            f"{m['f1']:.2f} | {m['exact_match_rate']:.2f} |"
        )

    print("\n## 3. Obergrenzen\n")
    oracle = sum(
        any(
            candidate == set(r["expected_codes"])
            for candidate in (
                set(r["predicted_codes"]),
                set(l["predicted_codes"]),
                set(r["predicted_codes"]) | set(l["predicted_codes"]),
                set(r["predicted_codes"]) & set(l["predicted_codes"]),
            )
        )
        for r, l in rows
    )
    subset = sum(
        set(r["expected_codes"]) <= (set(r["predicted_codes"]) | set(l["predicted_codes"]))
        for r, l in rows
    )
    print(f"  Orakel über die 4 fertigen Mengen:      EM {oracle}/{n} = {oracle / n:.2f}")
    print(f"  Perfekte Teilmenge aus der Vereinigung: EM {subset}/{n} = {subset / n:.2f}")
    print(
        "\n  Lesart: Die erste Zahl ist die Grenze für Mengen-Algebra, die zweite\n"
        "  die Grenze für einen Auswahl-/Checker-Node, der aus den gepoolten\n"
        "  Kandidaten die richtige Teilmenge zieht. Klaffen sie auseinander,\n"
        "  liegt der Hebel im Auswählen, nicht im Verrechnen."
    )


if __name__ == "__main__":
    main()
