"""Rendert die Vergleichs-Metriken als Markdown-Tabelle, im Stil von
sql-agent/evals/results.md."""


def render_results_table(metrics_by_approach: dict[str, dict[str, float]]) -> str:
    """Erzeugt eine Markdown-Tabelle aus Metriken-Dictionary.

    Args:
        metrics_by_approach: Dict mit Ansatz-Namen als Schlüssel und
            Dict mit Metriken als Wert. Jeder Metriken-Dict sollte
            'precision', 'recall', 'f1', 'exact_match_rate' enthalten.

    Returns:
        Markdown-formatted Tabelle als String.
    """
    header = "| Ansatz | Precision | Recall | F1 | Exact Match |\n"
    separator = "|---|---|---|---|---|\n"
    rows = ""
    for approach, metrics in metrics_by_approach.items():
        rows += (
            f"| {approach} | {metrics['precision']:.2f} | {metrics['recall']:.2f} | "
            f"{metrics['f1']:.2f} | {metrics['exact_match_rate']:.2f} |\n"
        )
    return header + separator + rows
