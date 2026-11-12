"""Test d'intégration : application web réelle + serveur MCP réel (stdio).

Contrairement à `test_web_api.py` (client MCP remplacé), ce test ne substitue
rien : le `lifespan` de FastAPI démarre un vrai sous-processus
`python -m lpo_data_mcp`, et les requêtes HTTP traversent toute la chaîne

    HTTP → agent → client MCP → sous-processus → SQLite

C'est le chemin exact que l'utilisateur exécute (`uvicorn web.app:app`).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from web import app as web_app  # noqa: E402


@pytest.fixture()
def live_client(monkeypatch, demo_db_path):
    from src.database import engine

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{demo_db_path}")
    monkeypatch.setenv("LPO_PROFILE", "observations")
    monkeypatch.setenv("LPO_LLM_INTENT", "off")
    engine.reset_engine()
    with TestClient(web_app.app) as client:
        yield client
    engine.reset_engine()


def test_full_chain_trend_then_filter(live_client):
    health = live_client.get("/api/health").json()
    assert len(health["mcpTools"]) == 5

    first = live_client.post(
        "/api/ask",
        json={
            "question": "Évolution des observations de Milan royal sur les 10 dernières années ?"
        },
    )
    assert first.status_code == 200
    turn = first.json()
    assert turn["kind"] == "trend"
    assert turn["validation"]["ok"] is True
    assert turn["mode"] == "create"
    # La trace ne contient que les appels qui produisent la réponse : la
    # découverte du jeu de données (`get_dataset_overview`) a lieu une fois, à
    # l'initialisation de l'assistant, et n'est donc pas rejouée à chaque tour.
    tools = [entry["tool"] for entry in turn["trace"]]
    assert tools == ["aggregate_observations"]
    assert all(entry["ok"] for entry in turn["trace"])

    updated = live_client.post(
        "/api/action",
        json={
            "conversationId": turn["conversationId"],
            "surfaceId": turn["surfaceId"],
            "action": {"name": "set_filter", "context": {"filter": "period", "value": "12m"}},
        },
    )
    assert updated.status_code == 200
    body = updated.json()
    assert body["mode"] == "update"
    assert body["surfaceId"] == turn["surfaceId"]
    assert {list(m)[1] for m in body["messages"]} == {"updateDataModel"}
    assert body["validation"]["ok"] is True


def test_dataset_endpoint_comes_from_the_mcp_server(live_client):
    dataset = live_client.get("/api/dataset").json()
    assert dataset["profile"] == "observations"
    assert dataset["is_demo"] is True
    assert dataset["total_rows"] == 900
