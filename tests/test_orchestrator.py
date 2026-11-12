"""Tests de l'agent orchestrateur (sans LLM ni sous-processus MCP).

Le client MCP est remplacé par un adaptateur branché directement sur le service :
ce sont les mêmes noms d'outils, les mêmes arguments et les mêmes sorties
structurées. On vérifie surtout les garanties attendues de l'agent :

- choix de visualisation cohérent avec l'intention ;
- chiffres présents dans le texte et les données = chiffres MCP ;
- aucune surface A2UI invalide ;
- mises à jour de filtres en données seules ;
- état vide explicite quand la donnée manque.
"""

from __future__ import annotations

import asyncio
from datetime import date

import pytest

from orchestrator.agent import Conversation, LpoAssistant

TODAY = date(2026, 11, 12)


def ask(assistant, question, conversation=None):
    return asyncio.run(assistant.ask(question, conversation or Conversation()))


@pytest.fixture()
def assistant(service_client):
    return LpoAssistant(service_client, today=TODAY, use_llm_intent=False)


def test_trend_question_produces_valid_line_chart_surface(assistant):
    turn = ask(assistant, "Évolution des observations de Milan royal sur les 10 dernières années ?")
    assert turn.kind == "trend"
    assert turn.degraded is False
    assert turn.validation["ok"] is True
    assert turn.messages and "createSurface" in turn.messages[0]
    components = turn.messages[0]["createSurface"]["components"]
    chart = next(c for c in components if c["component"] == "LineChart")
    assert chart["series"] == {"@path": "/chart/series"}
    model = turn.messages[0]["createSurface"]["dataModel"]
    assert model["chart"]["series"]
    assert "Milan royal" in turn.text


def test_ranking_question_produces_bar_chart(assistant):
    turn = ask(assistant, "Top 10 des espèces observées dans le Finistère en 2025 ?")
    assert turn.kind == "ranking"
    assert turn.validation["ok"] is True
    chart = next(
        c for c in turn.messages[0]["createSurface"]["components"] if c["component"] == "BarChart"
    )
    assert chart["items"] == {"@path": "/chart/items"}
    assert len(turn.messages[0]["createSurface"]["dataModel"]["chart"]["items"]) <= 10


def test_map_question_produces_map_and_privacy_notice(assistant):
    turn = ask(assistant, "Carte des observations de Busard cendré ce printemps ?")
    assert turn.kind == "map"
    assert turn.validation["ok"] is True
    model = turn.messages[0]["createSurface"]["dataModel"]
    points = model["chart"]["points"]
    assert points
    assert all(round(p["lat"], 1) == p["lat"] for p in points)
    assert any("Localisation" in (n.get("title") or "") for n in model["notices"])


def test_comparison_question_produces_two_series(assistant):
    turn = ask(assistant, "Compare Milan royal et Busard cendré par mois.")
    assert turn.kind == "trend"
    series = turn.messages[0]["createSurface"]["dataModel"]["chart"]["series"]
    assert len(series) == 2


def test_species_question_produces_species_surface(assistant):
    turn = ask(assistant, "Fiche espèce du Héron cendré")
    assert turn.kind == "species"
    model = turn.messages[0]["createSurface"]["dataModel"]
    assert model["species"]["name"] == "Héron cendré"


def test_off_topic_question_yields_explicit_empty_state(assistant):
    turn = ask(assistant, "Quel temps fera-t-il demain ?")
    assert turn.kind == "empty"
    assert "inventer" in turn.text
    notices = turn.messages[0]["createSurface"]["dataModel"]["notices"]
    assert notices and notices[0]["tone"] == "info"
    # Aucun appel d'agrégation n'a été fait pour une question hors périmètre.
    tools = [t.tool for t in turn.trace]
    assert "aggregate_observations" not in tools


def test_every_number_in_text_comes_from_mcp(assistant, service_client):
    turn = ask(assistant, "Top 10 des espèces observées en Gironde en 2025 ?")
    aggregate_calls = [
        args for tool, args in service_client.calls if tool == "aggregate_observations"
    ]
    ventes = aggregate_calls[-1]
    assert ventes["department"] == "Gironde"
    payload = turn.messages[0]["createSurface"]["dataModel"]
    # Les valeurs du data model sont exactement celles de la série MCP.
    items = payload["chart"]["items"]
    total = int(
        payload["kpis"][0]["value"].replace("\u202f", "").replace("\u00a0", "").replace(" ", "")
    )
    measured = sum(item["value"] for item in items)
    # Le classement est tronqué au top 10 : la somme des parts ne peut pas
    # dépasser le total réel annoncé par le serveur MCP.
    assert 0 < measured <= total
    assert "Gironde" in turn.text


def test_filters_update_data_model_without_recreating_surface(assistant):
    conversation = Conversation(id="t")
    first = ask(assistant, "Top 10 des espèces en Gironde en 2025 ?", conversation)
    assert first.mode == "create"
    surface = conversation.surface_id

    turn = asyncio.run(
        assistant.act(
            {"name": "set_filter", "context": {"filter": "area", "value": "Finistère"}},
            conversation,
        )
    )
    assert turn.mode == "update"
    assert turn.surface_id == surface
    assert {list(m)[1] for m in turn.messages} == {"updateDataModel"}
    assert turn.validation["ok"] is True
    assert "Finistère" in turn.text
    # La structure n'est pas renvoyée : seule la donnée circule.
    assert all("components" not in m.get("updateDataModel", {}) for m in turn.messages)


def test_period_filter_resolves_presets(assistant):
    conversation = Conversation(id="t")
    ask(
        assistant,
        "Évolution des observations de Milan royal sur les 10 dernières années ?",
        conversation,
    )
    turn = asyncio.run(
        assistant.act(
            {"name": "set_filter", "context": {"filter": "period", "value": "spring"}},
            conversation,
        )
    )
    assert "printemps" in turn.text
    assert turn.mode == "update"


def test_species_filter_on_comparison_degrades_to_trend(assistant):
    conversation = Conversation(id="t")
    ask(assistant, "Compare Milan royal et Busard cendré par mois.", conversation)
    turn = asyncio.run(
        assistant.act(
            {"name": "set_filter", "context": {"filter": "species", "value": "Milan royal"}},
            conversation,
        )
    )
    assert turn.kind in {"trend", "empty"}
    assert turn.validation["ok"] is True


def test_degraded_mode_when_a2ui_is_invalid(assistant, monkeypatch):
    """Si la validation échoue, l'agent renvoie un texte seul (jamais une UI douteuse)."""

    def broken(*args, **kwargs):
        raise AssertionError("ne doit pas être appelé")

    monkeypatch.setattr(
        "orchestrator.agent.validate_messages",
        lambda messages, state=None: __import__(
            "lpo_a2ui.validation", fromlist=["ValidationReport"]
        ).ValidationReport(ok=False, errors=["test"], checked_messages=0, state=state),
    )
    turn = ask(assistant, "Évolution des observations de Milan royal sur les 10 dernières années ?")
    assert turn.degraded is True
    assert turn.messages == []
    assert "interface graphique" in turn.text


def test_repository_profile_yields_empty_map_not_fake_points(assistant_for_repo):
    turn = ask(assistant_for_repo, "Carte des observations de Busard cendré ce printemps ?")
    assert turn.kind == "empty"
    model = turn.messages[0]["createSurface"]["dataModel"]
    messages = " ".join(n.get("message", "") for n in model["notices"]).lower()
    # L'espèce n'existe pas dans l'extrait : refus explicite…
    assert "busard cendré" in messages and "n'est pas reconnu" in messages
    # …et la limite de schéma reste annoncée (aucune coordonnée dans la table).
    assert "coordonné" in messages
    assert "MapChart" not in {
        c["component"] for c in turn.messages[0]["createSurface"]["components"]
    }


def test_repository_profile_refuses_a_species_it_does_not_contain(assistant_for_repo):
    """L'extrait réel ne contient que 4 identifiants d'espèces, sans aucun nom.

    Répondre « 169 observations » à une question sur le Milan royal serait
    exact arithmétiquement et faux sur le fond : ces lignes ne le concernent pas.
    """
    turn = ask(
        assistant_for_repo,
        "Évolution des observations de Milan royal sur les 10 dernières années ?",
    )
    assert turn.kind == "empty"
    assert turn.validation["ok"] is True
    assert "Milan royal" in turn.text
    assert "aucun chiffre" in turn.text.lower()
    # Aucun outil de données n'est appelé : rien à mesurer sur une espèce absente.
    components = turn.messages[0]["createSurface"]["components"]
    assert "LineChart" not in {c["component"] for c in components}


@pytest.fixture()
def assistant_for_repo(repo_client):
    return LpoAssistant(repo_client, today=TODAY, use_llm_intent=False)


def test_repository_profile_counts_records_not_observations(assistant_for_repo):
    """L'extrait contient des zones, pas des observations : le texte doit le dire."""
    turn = ask(assistant_for_repo, "Évolution du nombre de zones par an")
    assert turn.kind == "trend"
    assert "enregistrements" in turn.text
    assert "observations au total" not in turn.text
