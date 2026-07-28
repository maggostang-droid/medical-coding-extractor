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
