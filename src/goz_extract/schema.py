"""Pydantic-Schemas für GOZ-Codes, Trainings-/Testbeispiele und Modell-Vorhersagen."""
from typing import Literal

from pydantic import BaseModel, Field

Difficulty = Literal["easy", "medium", "hard"]
Split = Literal["train", "test"]


class GozCode(BaseModel):
    """Ein einzelner GOZ-Code aus der amtlichen Gebührenordnung.

    Enthält bewusst nur die öffentlichen Felder goz_nr und bezeichnung —
    siehe Design-Spec, Abschnitt "Datenherkunft & IP-Abgrenzung".
    """

    goz_nr: str
    bezeichnung: str


class NoteExample(BaseModel):
    """Eine (synthetische oder aus den Golden-Fixtures übernommene)
    Behandlungsnotiz mit den erwarteten GOZ-Codes."""

    text: str
    expected_codes: list[str]
    difficulty: Difficulty | None = None
    source: str = "generated"
    split: Split | None = None


class Prediction(BaseModel):
    """Vorhersage eines Ansatzes (RAG-Baseline oder Finetune) für eine Notiz."""

    text: str
    predicted_codes: list[str] = Field(default_factory=list)
