import pytest

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


def test_evaluate_predictions_raises_clear_error_on_empty_pairs():
    with pytest.raises(ValueError, match="Keine Predictions"):
        evaluate_predictions([])
