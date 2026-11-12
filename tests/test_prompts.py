"""Prompt système et few-shots : les exemples sont-ils réellement valides ?

Un few-shot invalide est pire qu'un few-shot absent : il apprend au modèle une
structure que le renderer rejettera. Ces tests garantissent que chaque exemple
respecte le protocole v1.0, le catalogue LPO et le contrat « aucune mesure en
clair » — donc qu'il serait accepté s'il était produit en production.
"""

from __future__ import annotations

import pytest

from lpo_a2ui.validation import catalog_id, new_renderer_state, validate_messages
from orchestrator.prompts import FEW_SHOTS, SYSTEM_PROMPT, few_shot_prompt

KIND_BY_SHOT = {
    "tendance sur 10 ans": "trend",
    "classement départemental": "ranking",
    "carte des observations": "map",
    "comparaison de deux espèces": "trend",
    "fiche espèce": "species",
    "hors périmètre": "empty",
}


def test_system_prompt_states_the_invariants():
    for invariant in (
        "Aucun chiffre inventé",
        "Catalogue fermé",
        "Structure / données séparées",
        "Repli texte systématique",
        "Confidentialité",
    ):
        assert invariant in SYSTEM_PROMPT
    # Le catalogue effectif est cité : le prompt ne peut pas viser un autre catalogue.
    assert catalog_id() in SYSTEM_PROMPT
    # Aucun composant hors catalogue n'est présenté comme disponible.
    for forbidden in ("Html", "Iframe", "VideoPlayer", "CodeBlock", "Embed"):
        assert forbidden not in SYSTEM_PROMPT


def test_few_shots_cover_the_three_interface_models_and_edge_cases():
    names = {shot.name for shot in FEW_SHOTS}
    assert names == set(KIND_BY_SHOT)
    kinds = {KIND_BY_SHOT[shot.name] for shot in FEW_SHOTS}
    assert {"trend", "ranking", "map", "species", "empty"} <= kinds
    # La comparaison est un cas de `trend` à deux séries.
    comparison = next(shot for shot in FEW_SHOTS if shot.name == "comparaison de deux espèces")
    series = comparison.messages[0]["createSurface"]["dataModel"]["chart"]["series"]
    assert len(series) == 2


@pytest.mark.parametrize("shot", FEW_SHOTS, ids=[shot.name for shot in FEW_SHOTS])
def test_every_few_shot_message_is_valid(shot):
    report = validate_messages(list(shot.messages))
    assert report.ok, report.errors
    message = shot.messages[0]
    assert message["version"] == "1.0"
    assert message["createSurface"]["catalogId"] == catalog_id()


@pytest.mark.parametrize("shot", FEW_SHOTS, ids=[shot.name for shot in FEW_SHOTS])
def test_few_shots_have_a_text_fallback_and_a_trace(shot):
    assert shot.text.strip()
    assert shot.question.strip()
    # Une réponse hors périmètre ne doit déclencher aucun appel d'agrégation.
    if KIND_BY_SHOT[shot.name] == "empty":
        assert shot.tool_calls == ()
    else:
        assert shot.tool_calls
        assert all(
            call.tool
            in {
                "get_dataset_overview",
                "list_species",
                "get_observations",
                "aggregate_observations",
                "get_species_info",
            }
            for call in shot.tool_calls
        )


@pytest.mark.parametrize("shot", FEW_SHOTS, ids=[shot.name for shot in FEW_SHOTS])
def test_few_shot_measures_live_in_the_data_model(shot):
    """Les chiffres ne doivent apparaître que dans `dataModel`, jamais dans les composants."""
    components = shot.messages[0]["createSurface"]["components"]
    for component in components:
        for key in ("series", "items", "points", "rows"):
            if key in component:
                assert isinstance(component[key], dict) and "@path" in component[key]


def test_few_shots_replay_without_recreating_surfaces():
    """Un exemple de filtre doit pouvoir s'appliquer sur la surface de son exemple."""
    state = new_renderer_state()
    for shot in FEW_SHOTS:
        assert validate_messages(list(shot.messages), state=state).ok
    # Les identifiants de surface sont distincts d'un exemple à l'autre.
    ids = [shot.messages[0]["createSurface"]["surfaceId"] for shot in FEW_SHOTS]
    assert len(ids) == len(set(ids))


def test_prompt_block_is_prompt_ready():
    prompt = few_shot_prompt()
    assert prompt.startswith(SYSTEM_PROMPT)
    assert prompt.count("### Exemple") == len(FEW_SHOTS)
    assert "Messages A2UI" in prompt
    assert "Texte de repli" in prompt
