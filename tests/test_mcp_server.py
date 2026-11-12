"""Test d'intégration MCP : vrai serveur lancé en stdio, vrai client SDK.

Ce test vérifie le contrat de bout en bout côté données : noms d'outils, sortie
structurée typée, résumé texte, et erreurs explicites (sans SQL libre).
"""

from __future__ import annotations

import asyncio
import os
import sys

from mcp import Client
from mcp.client.stdio import StdioServerParameters

from tests.conftest import ROOT


def _run(coro):
    return asyncio.run(coro)


def test_stdio_server_exposes_read_only_tools(demo_db_path):
    async def scenario():
        env = dict(os.environ)
        env["DATABASE_URL"] = f"sqlite:///{demo_db_path}"
        env["LPO_PROFILE"] = "observations"
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "lpo_data_mcp"],
            cwd=str(ROOT),
            env=env,
        )
        async with Client(params) as client:
            tools = await client.list_tools()
            names = {tool.name for tool in tools.tools}
            assert names == {
                "get_dataset_overview",
                "list_species",
                "get_observations",
                "aggregate_observations",
                "get_species_info",
            }
            # Toutes les descriptions sont fournies au modèle.
            assert all(tool.description for tool in tools.tools)

            overview = await client.call_tool("get_dataset_overview", {})
            assert overview.is_error is False
            assert overview.structured_content["profile"] == "observations"
            assert overview.content and overview.content[0].text

            aggregated = await client.call_tool(
                "aggregate_observations", {"group_by": "year", "metric": "observations"}
            )
            assert aggregated.is_error is False
            payload = aggregated.structured_content
            assert payload["points"]
            assert {"key", "label", "value"} <= set(payload["points"][0])

            bad = await client.call_tool("aggregate_observations", {"group_by": "nope"})
            assert bad.is_error is True
            text = bad.content[0].text
            assert "group_by" in text and "year" in text

    _run(scenario())


def test_stdio_server_reports_missing_database(tmp_path):
    async def scenario():
        env = dict(os.environ)
        env["DATABASE_URL"] = f"sqlite:///{tmp_path / 'absent.db'}"
        env["LPO_PROFILE"] = "observations"
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "lpo_data_mcp"], cwd=str(ROOT), env=env
        )
        async with Client(params) as client:
            result = await client.call_tool("list_species", {})
            assert result.is_error is True
            message = result.content[0].text
            assert "identifiants SQL" not in message
            assert "n'existe pas" in message or "no such table" in message.lower()

    _run(scenario())
