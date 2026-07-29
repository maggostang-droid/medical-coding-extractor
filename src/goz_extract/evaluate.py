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
    if not pairs:
        raise ValueError("Keine Predictions zum Auswerten")
    metrics = [precision_recall_f1(predicted, expected) for predicted, expected in pairs]
    n = len(metrics)
    exact_matches = sum(exact_match(predicted, expected) for predicted, expected in pairs)
    return {
        "precision": sum(m["precision"] for m in metrics) / n,
        "recall": sum(m["recall"] for m in metrics) / n,
        "f1": sum(m["f1"] for m in metrics) / n,
        "exact_match_rate": exact_matches / n,
    }
