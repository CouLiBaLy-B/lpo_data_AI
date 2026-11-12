"""Client MCP de production (`orchestrator.mcp_client`) sur un vrai transport stdio.

Les autres tests de l'orchestrateur remplacent le client par un adaptateur
branché sur le service : rapide, mais cela ne prouve pas que la chaîne
`agent → SDK MCP → sous-processus serveur` fonctionne réellement. Ces tests-ci
lancent un vrai sous-processus `python -m lpo_data_mcp` et vérifient le contrat
observé par l'agent : noms d'outils, contenu structuré, propagation d'erreur,
transmission de l'environnement.

La session est ouverte et fermée **dans la tâche du test** (un `async with`) :
un `AsyncExitStack` du SDK MCP ne peut pas être fermé depuis une autre tâche,
donc pas de fixture asynchrone avec `yield`.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from orchestrator.mcp_client import McpDataClient, McpToolError

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_TOOLS = {
    "get_dataset_overview",
    "list_species",
    "get_observations",
    "aggregate_observations",
    "get_species_info",
}


def new_client(**kwargs) -> McpDataClient:
    return McpDataClient(command=sys.executable, args=["-m", "lpo_data_mcp"], cwd=ROOT, **kwargs)


@asynccontextmanager
async def open_client(**kwargs):
    mcp_client = new_client(**kwargs)
    try:
        await mcp_client.connect()
        yield mcp_client
    finally:
        await mcp_client.close()


@pytest.fixture()
def agent_env(monkeypatch, demo_db_path):
    """Environnement du sous-processus : base de démonstration + profil complet."""
    from src.database import engine

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{demo_db_path}")
    monkeypatch.setenv("LPO_PROFILE", "observations")
    engine.reset_engine()
    yield
    engine.reset_engine()


async def test_connect_lists_the_five_tools(agent_env):
    async with open_client() as client:
        assert set(client.tool_names) == EXPECTED_TOOLS


async def test_call_returns_structured_content(agent_env):
    async with open_client() as client:
        overview = await client.call("get_dataset_overview")
    assert overview["profile"] == "observations"
    assert overview["total_rows"] > 0
    assert overview["summary"]


async def test_environment_is_forwarded_to_the_server(agent_env):
    """Le sous-processus doit voir la base de test, pas la base par défaut."""
    async with open_client() as client:
        overview = await client.call("get_dataset_overview")
    assert overview["is_demo"] is True
    assert overview["total_rows"] == 900  # graine de `demo_db_path`


async def test_aggregate_returns_series_and_total(agent_env):
    async with open_client() as client:
        species = await client.call("list_species", {"limit": 5})
        label = species["species"][0]["label"]
        result = await client.call(
            "aggregate_observations",
            {"group_by": "year", "species": [label], "metric": "observations"},
        )
    assert result["points"]
    assert result["total"] == sum(point["value"] for point in result["points"])
    assert all(point["share"] >= 0 for point in result["points"])


async def test_tool_error_is_propagated_with_server_message(agent_env):
    """Une erreur d'outil doit remonter telle quelle, jamais un résultat vide."""
    async with open_client() as client:
        with pytest.raises(McpToolError) as excinfo:
            await client.call("aggregate_observations", {"group_by": "not-a-dimension"})
    assert "not-a-dimension" in str(excinfo.value)
    assert excinfo.value.tool == "aggregate_observations"


async def test_client_connects_lazily_on_first_call(agent_env):
    lazy = new_client()
    try:
        assert lazy.tool_names == []  # pas encore connecté
        result = await lazy.call("get_dataset_overview")
        assert result["profile"] == "observations"
        assert lazy.tool_names  # la connexion a eu lieu
    finally:
        await lazy.close()


async def test_unusable_command_fails_loudly(agent_env):
    broken = McpDataClient(command="lpo-command-that-does-not-exist", args=[], cwd=ROOT)
    with pytest.raises(Exception):
        await broken.connect()
    await broken.close()


async def test_close_is_idempotent_and_reconnectable(agent_env):
    client = new_client()
    await client.connect()
    await client.close()
    await client.close()
    # Après fermeture, un appel relance une session plutôt que d'échouer.
    assert (await client.call("get_dataset_overview"))["profile"] == "observations"
    await client.close()
