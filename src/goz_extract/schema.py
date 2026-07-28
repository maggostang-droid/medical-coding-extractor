"""Pydantic-Schemas für GOZ-Codes, Trainings-/Testbeispiele und Modell-Vorhersagen."""
from typing import Literal

from pydantic import BaseModel, Field

Difficulty = Literal["easy", "medium", "hard"]
Split = Literal["train", "test"]


class GozCode(BaseModel):
    """Ein einzelner GOZ-Code aus der amtlichen Gebührenordnung.

    Enthält bewusst nur öffentliche Felder — siehe Design-Spec, Abschnitt
    "Datenherkunft & IP-Abgrenzung". `erweiterte_beschreibung` ist eine
    optionale Klartext-Erklärung + gängige Begriffe/Abkürzungen, generiert
    aus allgemeinem GOZ-Fachwissen (siehe `goz_extract.enrichment`) - kein
    Bezug zu proprietären Kommentarwerken (z.B. Liebold/Raff/Wahl, die
    urheberrechtlich geschützt sind) oder MAIKAs eigener Geschäftslogik.
    """

    goz_nr: str
    bezeichnung: str
    erweiterte_beschreibung: str | None = None

    def format_for_prompt(self) -> str:
        """Formatiert Code + Bezeichnung für Prompts (RAG-Kandidatenliste,
        Datengenerierung) - hängt die erweiterte Beschreibung an, falls
        vorhanden, sonst nur die amtliche Bezeichnung."""
        if self.erweiterte_beschreibung:
            return f"{self.goz_nr}: {self.bezeichnung} ({self.erweiterte_beschreibung})"
        return f"{self.goz_nr}: {self.bezeichnung}"


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
