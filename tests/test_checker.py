"""Tests für den Checker-Node — ohne Modell, über eine eingesetzte
Auswahlfunktion."""

import pytest

from goz_extract.checker import build_checker_prompt, checker_predict, pool_candidates
from goz_extract.schema import GozCode

CODES = [
    GozCode(goz_nr="2060", bezeichnung="Kompositfüllung, einflächig", erweiterte_beschreibung="Zahnfarbene Füllung."),
    GozCode(goz_nr="2030", bezeichnung="Besondere Massnahmen", erweiterte_beschreibung="Kofferdam."),
    GozCode(goz_nr="0065", bezeichnung="Optisch-elektronische Abformung", erweiterte_beschreibung="Intraoralscan."),
]
CODE_BY_NR = {c.goz_nr: c for c in CODES}


def select_none(note, candidates):
    return []


def select_all(note, candidates):
    return [c.goz_nr for c in candidates]


def test_pool_ist_die_vereinigung_sortiert():
    assert pool_candidates(["2060", "0065"], ["2030", "2060"]) == ["0065", "2030", "2060"]


def test_pool_bleibt_bei_leeren_eingaben_leer():
    assert pool_candidates([], []) == []


def test_prompt_enthaelt_notiz_und_alle_kandidaten():
    prompt = build_checker_prompt("Zahn 46 gefüllt.", ["2060", "2030"], CODE_BY_NR)
    assert "Zahn 46 gefüllt." in prompt
    assert "2060" in prompt and "2030" in prompt
    assert "0065" not in prompt  # nicht im Pool -> darf nicht angeboten werden


def test_prompt_ignoriert_unbekannte_ziffern_im_pool():
    prompt = build_checker_prompt("Notiz", ["2060", "9999"], CODE_BY_NR)
    assert "9999" not in prompt


def test_checker_waehlt_teilmenge_und_verwirft_den_rest():
    def select_only_2060(note, candidates):
        return ["2060"]

    result = checker_predict("Notiz", ["2060", "0065"], ["2060", "2030"], CODE_BY_NR, select_only_2060)
    assert result.predicted_codes == ["2060"]
    assert result.sources["gepoolt"] == ["0065", "2030", "2060"]
    assert result.sources["verworfen"] == ["0065", "2030"]


def test_verworfene_kandidaten_loesen_keine_pruefung_aus():
    """Der Kernunterschied zur Merge-Regel: Aussortieren ist die Aufgabe des
    Knotens, kein Fall für einen Menschen."""

    def select_only_2060(note, candidates):
        return ["2060"]

    result = checker_predict("Notiz", ["2060", "0065"], ["2060"], CODE_BY_NR, select_only_2060)
    assert not result.needs_review


def test_leere_auswahl_aus_vollem_pool_eskaliert():
    result = checker_predict("Notiz", ["2060"], ["2030"], CODE_BY_NR, select_none)
    assert result.predicted_codes == []
    assert result.needs_review
    assert result.uncertain == ["2030", "2060"]


def test_leerer_pool_eskaliert_nicht():
    result = checker_predict("Notiz", [], [], CODE_BY_NR, select_none)
    assert result.predicted_codes == []
    assert not result.needs_review


def test_auswahl_ausserhalb_des_pools_wird_verworfen():
    """Halluziniert das Modell eine Ziffer, die gar nicht angeboten wurde,
    zählt sie nicht — roher Output ist nie automatisch das Ergebnis."""

    def select_foreign(note, candidates):
        return ["2060", "0065"]

    result = checker_predict("Notiz", ["2060"], ["2030"], CODE_BY_NR, select_foreign)
    assert result.predicted_codes == ["2060"]


def test_doppelte_auswahl_wird_entdoppelt():
    def select_twice(note, candidates):
        return ["2060", "2060"]

    result = checker_predict("Notiz", ["2060"], ["2060"], CODE_BY_NR, select_twice)
    assert result.predicted_codes == ["2060"]


def test_ausschlusspaar_wird_auch_hier_eskaliert():
    result = checker_predict(
        "Notiz",
        ["2060", "2030"],
        ["2060", "2030"],
        CODE_BY_NR,
        select_all,
        mutually_exclusive={frozenset({"2060", "2030"})},
    )
    assert result.predicted_codes == []
    assert result.conflicts == [("2030", "2060")]
    assert result.needs_review


def test_select_fn_bekommt_gozcode_objekte_nicht_strings():
    """Die Auswahlfunktion braucht Bezeichnung und Beschreibung, sonst kann
    sie fachlich gar nicht entscheiden."""
    gesehen = {}

    def spy(note, candidates):
        gesehen["typen"] = {type(c) for c in candidates}
        gesehen["nrs"] = [c.goz_nr for c in candidates]
        return []

    checker_predict("Notiz", ["2060"], ["2030"], CODE_BY_NR, spy)
    assert gesehen["typen"] == {GozCode}
    assert gesehen["nrs"] == ["2030", "2060"]


@pytest.mark.parametrize("rag,lora,erwartet", [
    (["2060"], [], ["2060"]),
    ([], ["2030"], ["2030"]),
    (["2060"], ["2030"], ["2030", "2060"]),
])
def test_pool_deckt_beide_pfade_ab(rag, lora, erwartet):
    assert pool_candidates(rag, lora) == erwartet
