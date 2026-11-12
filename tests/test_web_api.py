"""API web (FastAPI) — contrat HTTP exposé au client A2UI.

Le test remplace le client MCP par l'adaptateur branché sur le service (mêmes
outils, mêmes sorties) : on valide la plomberie HTTP, la validation des entrées,
l'état par conversation et le contenu des messages A2UI renvoyés.
"""

from __future__ import annotations

import contextlib
from datetime import date

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from orchestrator.agent import LpoAssistant  # noqa: E402
from web import app as web_app  # noqa: E402


@pytest.fixture()
def client(service_client):
    """TestClient dont `lifespan` est court-circuité (pas de sous-processus MCP)."""

    @contextlib.asynccontextmanager
    async def fake_lifespan(app):
        app.state.assistant = LpoAssistant(
            service_client, today=date(2026, 11, 12), use_llm_intent=False
        )
        app.state.conversations = {}
        app.state.client = service_client
        yield

    original = web_app.lifespan
    web_app.app.router.lifespan_context = fake_lifespan
    with TestClient(web_app.app) as test_client:
        yield test_client
    web_app.app.router.lifespan_context = original


def test_health_lists_mcp_tools(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert set(body["mcpTools"]) == {
        "get_dataset_overview",
        "list_species",
        "get_observations",
        "aggregate_observations",
        "get_species_info",
    }


def test_catalog_endpoint_is_the_single_source_of_truth(client):
    body = client.get("/api/catalog").json()
    assert body["protocolVersion"] == "1.0"
    assert body["catalogId"] == web_app.catalog_id()
    assert "LineChart" in body["components"] and "MapChart" in body["components"]
    assert body["catalog"]["components"]


def test_index_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "A2UI" in response.text


def test_dataset_endpoint_describes_profile(client):
    body = client.get("/api/dataset").json()
    assert body["profile"] == "observations"
    assert body["total_rows"] > 0
    assert body["capabilities"]


def test_ask_returns_validated_messages_and_fallback_text(client):
    response = client.post("/api/ask", json={"question": "Top 10 des espèces en Gironde en 2025 ?"})
    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "ranking"
    assert body["conversationId"]
    assert body["validation"]["ok"] is True
    assert body["text"]
    message = body["messages"][0]
    assert message["version"] == "1.0"
    assert "createSurface" in message
    assert body["trace"], "la trace des appels MCP doit être exposée"


def test_conversation_is_reused_and_filters_update_in_place(client):
    first = client.post(
        "/api/ask", json={"question": "Top 10 des espèces en Gironde en 2025 ?"}
    ).json()
    conversation_id = first["conversationId"]
    surface_id = first["surfaceId"]

    second = client.post(
        "/api/action",
        json={
            "conversationId": conversation_id,
            "surfaceId": surface_id,
            "action": {"name": "set_filter", "context": {"filter": "area", "value": "Finistère"}},
        },
    )
    assert second.status_code == 200
    body = second.json()
    assert body["mode"] == "update"
    assert body["surfaceId"] == surface_id
    assert {list(m)[1] for m in body["messages"]} == {"updateDataModel"}


def test_unknown_conversation_id_is_adopted_not_rejected(client):
    body = client.post(
        "/api/ask",
        json={"question": "Fiche espèce du Héron cendré", "conversationId": "client-side-id"},
    ).json()
    assert body["conversationId"] == "client-side-id"
    assert body["kind"] == "species"


def test_reset_forgets_the_conversation(client):
    first = client.post("/api/ask", json={"question": "Fiche espèce du Héron cendré"}).json()
    conversation_id = first["conversationId"]
    assert client.post("/api/reset", json={"conversationId": conversation_id}).json() == {
        "status": "reset"
    }
    again = client.post(
        "/api/ask",
        json={"question": "Fiche espèce du Héron cendré", "conversationId": conversation_id},
    ).json()
    assert again["conversationId"] == conversation_id
    assert again["mode"] == "create"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": "ab"},
        {"question": "x" * 501},
    ],
)
def test_ask_rejects_invalid_payloads(client, payload):
    assert client.post("/api/ask", json=payload).status_code == 422


def test_action_requires_a_conversation(client):
    response = client.post("/api/action", json={"action": {"name": "set_filter", "context": {}}})
    assert response.status_code == 422


def test_off_topic_question_is_answered_explicitly(client):
    body = client.post("/api/ask", json={"question": "Quel temps fera-t-il demain ?"}).json()
    assert body["kind"] == "empty"
    assert "inventer" in body["text"]
    assert body["messages"][0]["createSurface"]["dataModel"]["notices"]


def test_no_arbitrary_markup_leaks_into_messages(client):
    body = client.post(
        "/api/ask", json={"question": "Carte des observations de Busard cendré ?"}
    ).json()
    blob = repr(body["messages"])
    for forbidden in ("<script", "javascript:", "onclick=", "<style"):
        assert forbidden not in blob
