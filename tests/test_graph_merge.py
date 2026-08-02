"""Tests für die Fan-in-Logik des GOZ-Graphen."""

import pytest

from goz_extract.graph_merge import (
    RAG_ONLY_ACCEPT,
    graph_predict,
    merge_candidates,
    verify,
)

VALID = {"2060", "2030", "0065", "1040"}


def test_code_von_beiden_wird_akzeptiert():
    result = merge_candidates(["2060"], ["2060"])
    assert result.accepted == ["2060"]
    assert result.uncertain == []
    assert result.sources["beide"] == ["2060"]


def test_nur_lora_wird_akzeptiert():
    result = merge_candidates([], ["2030"])
    assert result.accepted == ["2030"]
    assert result.uncertain == []


def test_nur_rag_wird_unsicher():
    result = merge_candidates(["1040"], [])
    assert result.accepted == []
    assert result.uncertain == ["1040"]
    assert result.needs_review


def test_gelockerte_policy_akzeptiert_rag_only():
    result = merge_candidates(["1040"], [], rag_only_policy=RAG_ONLY_ACCEPT)
    assert result.accepted == ["1040"]
    assert not result.needs_review


def test_unbekannte_policy_fliegt_auf():
    with pytest.raises(ValueError):
        merge_candidates([], [], rag_only_policy="irgendwas")


def test_verifier_wirft_unbekannte_codes_raus():
    merged = merge_candidates(["9999"], ["9999"])
    result = verify(merged, VALID)
    assert result.accepted == []
    assert result.rejected == ["9999"]
    assert result.needs_review


def test_ausschlusspaar_wird_eskaliert_nicht_geraten():
    merged = merge_candidates(["2060", "2030"], ["2060", "2030"])
    result = verify(merged, VALID, mutually_exclusive={frozenset({"2060", "2030"})})
    assert result.accepted == []
    assert result.uncertain == ["2030", "2060"]
    assert result.conflicts == [("2030", "2060")]
    assert result.needs_review


def test_predicted_codes_enthaelt_nur_akzeptierte():
    result = graph_predict(["1040", "2060"], ["2060"], VALID)
    assert result.predicted_codes == ["2060"]
    assert result.uncertain == ["1040"]


def test_end_to_end_sauberer_fall():
    result = graph_predict(["2060", "0065"], ["2060", "0065"], VALID)
    assert result.predicted_codes == ["0065", "2060"]
    assert not result.needs_review


def test_leere_eingaben():
    result = graph_predict([], [], VALID)
    assert result.predicted_codes == []
    assert not result.needs_review
