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

    original = web_app.app.router.lifespan_context
    web_app.app.router.lifespan_context = fake_lifespan
    with TestClient(web_app.app) as test_client:
        yield test_client
    web_app.app.router.lifespan_context = original
    # L'application est un objet module : ne pas laisser de configuration d'un
    # test au suivant (`test_web_settings.py` écrit ces attributs).
    for attribute in ("settings", "settings_lock", "mcp_status"):
        web_app.app.state.__dict__.pop(attribute, None)


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


# ── Jeton d'accès optionnel ─────────────────────────────────────────────────


def test_api_is_open_without_configured_token(client, monkeypatch):
    monkeypatch.delenv("LPO_WEB_TOKEN", raising=False)
    assert client.get("/api/dataset").status_code == 200
    assert (
        client.post("/api/ask", json={"question": "Fiche espèce du Héron cendré"}).status_code
        == 200
    )


def test_configured_token_is_enforced(client, monkeypatch):
    monkeypatch.setenv("LPO_WEB_TOKEN", "s3cret")
    anonymous = client.post("/api/ask", json={"question": "Fiche espèce du Héron cendré"})
    assert anonymous.status_code == 401
    assert anonymous.headers["www-authenticate"] == "Bearer"
    assert client.get("/api/dataset").status_code == 401

    wrong = client.post(
        "/api/ask",
        json={"question": "Fiche espèce du Héron cendré"},
        headers={"Authorization": "Bearer faux"},
    )
    assert wrong.status_code == 401

    allowed = client.post(
        "/api/ask",
        json={"question": "Fiche espèce du Héron cendré"},
        headers={"Authorization": "Bearer s3cret"},
    )
    assert allowed.status_code == 200
    assert allowed.json()["kind"] == "species"


def test_health_and_catalog_stay_public(client, monkeypatch):
    """Les sondes de disponibilité et le catalogue ne révèlent aucune donnée."""
    monkeypatch.setenv("LPO_WEB_TOKEN", "s3cret")
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/catalog").status_code == 200
