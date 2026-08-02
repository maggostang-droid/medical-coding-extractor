"""Gemeinsame UI-Bausteine für die Portfolio-Demos im MARCO.OS-Stil.

Diese Datei wird in jede Streamlit-Demo kopiert (bewusst dupliziert statt als
Paket, damit jedes Repo eigenständig lauffähig bleibt). Sie sorgt dafür, dass
Kopfbereich, Beispiel-Einstieg, Technik-Bereich und Fußzeile in allen Demos
identisch aussehen.

Aufbau jeder Demo (siehe DEMO-STYLEGUIDE.md):
    1. page_setup()      Seitentitel und Layout
    2. page_header()     Produktname, Claim, Rücklink ins Portfolio
    3. example_picker()  Null-Klick-Einstieg mit vorbereiteten Beispielen
    4. <die eigentliche Interaktion>
    5. under_the_hood()  technischer Beleg, eingeklappt
    6. portfolio_footer()ehrliche Einordnung, Repo- und Portfolio-Link
"""

from __future__ import annotations

import streamlit as st

PORTFOLIO_URL = "https://maggostang-droid.github.io/marco-os/"

# Cluster-Akzente, identisch zu CLUSTER_ACCENT in marco-os/assets/js/window-manager.js
ACCENT = {
    "agentic-ai": "#fbbf24",
    "cloud": "#5eead4",
    "full-stack": "#a78bfa",
}

# Farbtoken aus marco-os/assets/css/style.css
BG = "#0a0716"
PANEL = "#140f24"
TEXT = "#e7e4f5"
TEXT_DIM = "#c9c5e8"


def page_setup(title: str, layout: str = "centered") -> None:
    """Setzt den Browser-Tab-Titel einheitlich auf '<Produktname> · MARCO.OS'.

    Kein Emoji als page_icon: Der Produktname soll im Tab lesbar sein, und der
    Styleguide verbietet Emojis in Titeln.
    """
    st.set_page_config(page_title=f"{title} · MARCO.OS", layout=layout)


def page_header(title: str, claim: str, project_id: str, cluster: str) -> None:
    """Kopfbereich: Produktname, Ein-Satz-Claim, Rücklink auf den Planeten.

    `title` muss exakt der Projektname aus marco-os/data/projects.js sein, sonst
    wirkt der Sprung aus dem Portfolio wie ein falscher Link. `claim` ist derselbe
    Satz wie fett unter der README-Überschrift. `project_id` ist die LOKALE ID aus
    projects.js (z.B. 'sql-agent'), nur die löst der Deep-Link-Router auf.
    """
    accent = ACCENT.get(cluster, ACCENT["cloud"])
    # Der Titel kommt bewusst als Streamlit-eigene Ueberschrift, nicht als
    # HTML-Element im Markdown-Block: eigene Ueberschriften-Elemente bleiben je
    # nach Streamlit-Version unsichtbar (Platz wird reserviert, Text nicht
    # gezeichnet), was in Screenshots einen kopflosen Kopfbereich ergibt.
    st.title(title)
    st.markdown(
        f"""
        <div style="border-left:3px solid {accent};padding:.15rem 0 .35rem 1rem;margin:-.5rem 0 1.1rem">
          <div style="font-size:1.02rem;color:{TEXT_DIM};line-height:1.5">{claim}</div>
          <div style="font-family:monospace;font-size:.8rem;margin-top:.7rem;color:{TEXT_DIM};opacity:.75">
            <a href="{PORTFOLIO_URL}#{project_id}" target="_blank"
               style="color:{accent};text-decoration:none">▸ Teil von MARCO.OS</a>
            <span style="opacity:.6"> · Portfolio von Marco Stang</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def example_picker(intro: str, examples: dict[str, str], key: str = "example") -> str | None:
    """Null-Klick-Einstieg: vorbereitete Beispiele als Buttons.

    Der wichtigste Baustein für Besucher, die nicht wissen, was sie eingeben
    sollen. Gibt den Schlüssel des geklickten Beispiels zurück oder None.
    Bewusst Buttons statt Dropdown: ein Klick statt zwei, und alle Optionen sind
    sofort sichtbar.
    """
    st.markdown(f"**{intro}**")
    cols = st.columns(len(examples))
    clicked = None
    for col, (label, description) in zip(cols, examples.items()):
        with col:
            if st.button(label, key=f"{key}_{label}", use_container_width=True):
                clicked = label
            if description:
                st.caption(description)
    return clicked


def under_the_hood(label: str = "Unter der Haube"):
    """Einheitlich benannter Bereich für den technischen Beleg.

    Hier hinein gehört alles, was ein Tech-Lead sehen will und ein Recruiter
    nicht braucht: generiertes SQL, Prompts, Retrieval-Kandidaten, SHAP-Werte,
    Guardrail-Verlauf, Rohantworten. Immer eingeklappt, immer gleich benannt.

    Verwendung:
        with under_the_hood():
            st.code(sql, language="sql")
    """
    return st.expander(f"▸ {label}", expanded=False)


def portfolio_footer(repo: str, project_id: str, caveats: list[str], siblings: list[tuple[str, str]] | None = None) -> None:
    """Fußzeile: ehrliche Einordnung, Repo-Link, Rücklink ins Portfolio.

    `caveats` benennt, was diese Demo NICHT ist (Free-Tier-Kaltstart, synthetische
    Daten, Stichprobe). Das ist Teil des Produktversprechens, nicht Kleingedrucktes.
    """
    st.divider()
    if caveats:
        st.caption("**Was diese Demo nicht ist:** " + " · ".join(caveats))
    links = [
        f"[Quellcode auf GitHub](https://github.com/maggostang-droid/{repo})",
        f"[Dieses Projekt in MARCO.OS]({PORTFOLIO_URL}#{project_id})",
    ]
    if siblings:
        links += [f"[{name}](https://github.com/maggostang-droid/{slug})" for name, slug in siblings]
    st.caption(" · ".join(links))
    st.caption("Marco Stang · Dr.-Ing. · [LinkedIn](https://www.linkedin.com/in/marco-stang) · stang.marco@t-online.de")
