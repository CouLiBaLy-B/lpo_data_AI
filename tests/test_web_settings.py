"""API de configuration : modèle et serveur MCP modifiables depuis l'interface.

Le client MCP reste l'adaptateur branché sur le service (aucun sous-processus) ;
les changements de cible sont testés avec un faux client, ce qui permet de
vérifier la règle importante : une cible qui ne répond pas n'est **pas** adoptée.
"""

from __future__ import annotations

import contextlib
from datetime import date

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from orchestrator import settings as runtime  # noqa: E402
from orchestrator.agent import Conversation, LpoAssistant  # noqa: E402
from web import app as web_app  # noqa: E402

ENV_NAMES = (
    "LPO_LLM_INTENT",
    "AI_PROVIDER",
    "AI_MODEL",
    "LPO_MCP_URL",
    "LPO_MCP_COMMAND",
    "LPO_MCP_ARGS",
    "LPO_WEB_SETTINGS",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)


class FakeMcpClient:
    """Client MCP de test : connecte ou échoue, et répond à un aperçu."""

    def __init__(self, *, fail: bool = False, url: str | None = None):
        self.fail = fail
        self.url = url
        self.closed = False
        self.tool_names = ["get_dataset_overview", "aggregate_observations"]

    async def connect(self) -> None:
        if self.fail:
            raise RuntimeError("connexion refusée par le test")

    async def close(self) -> None:
        self.closed = True

    async def call(self, tool: str, arguments: dict | None = None) -> dict:
        # Comme le vrai client, `call` ouvre la session si nécessaire.
        if self.fail:
            raise RuntimeError("connexion refusée par le test")
        assert tool == "get_dataset_overview"
        return {
            "dataset_label": "Observations naturalistes (test)",
            "profile": "observations",
            "total_rows": 900,
            "records_label": "observations",
            "is_demo": True,
        }


@pytest.fixture()
def client(service_client, monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)

    @contextlib.asynccontextmanager
    async def fake_lifespan(app):
        settings = runtime.from_env()
        app.state.settings = settings
        app.state.client = service_client
        app.state.conversations = {}
        app.state.assistant = LpoAssistant(
            service_client, today=date(2026, 11, 12), use_llm_intent=False
        )
        app.state.mcp_status = {"ok": True, "tools": list(service_client.tool_names), "error": None}
        yield

    original = web_app.app.router.lifespan_context
    web_app.app.router.lifespan_context = fake_lifespan
    with TestClient(web_app.app) as test_client:
        yield test_client
    web_app.app.router.lifespan_context = original
    for attribute in ("settings", "settings_lock", "mcp_status"):
        web_app.app.state.__dict__.pop(attribute, None)


def test_settings_endpoint_describes_model_and_mcp_without_secrets(client):
    body = client.get("/api/settings").json()
    assert body["source"] == "env"
    assert body["editable"] is True
    assert body["llm"]["mode"] == "auto"
    assert body["llm"]["provider"] == "anthropic"
    assert body["llm"]["apiKeySet"] is False
    assert "apiKey" not in body["llm"] and "api_key" not in body["llm"]
    assert [provider["id"] for provider in body["llm"]["providers"]][:3] == [
        "anthropic",
        "openai",
        "google_genai",
    ]
    assert body["mcp"]["transport"] == "stdio"
    assert body["mcp"]["target"].endswith("python -m lpo_data_mcp")


def test_model_settings_can_be_changed_without_touching_the_dataset(client):
    conversation = Conversation()
    web_app.app.state.conversations[conversation.id] = conversation
    response = client.post(
        "/api/settings",
        json={"llm": {"mode": "on", "provider": "openai", "model": "gpt-4o-mini"}},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["restarted"] == ["llm"]
    assert body["settings"]["source"] == "ui"
    assert body["settings"]["llm"]["provider"] == "openai"
    assert web_app.app.state.assistant.use_llm_intent is True
    assert conversation.id in web_app.app.state.conversations  # structure inchangée


def test_invalid_settings_are_refused_with_a_readable_message(client):
    bad_provider = client.post("/api/settings", json={"llm": {"provider": "skynet"}})
    assert bad_provider.status_code == 422
    assert "Fournisseur inconnu" in bad_provider.json()["detail"]

    bad_url = client.post(
        "/api/settings", json={"mcp": {"transport": "http", "url": "ftp://exemple/mcp"}}
    )
    assert bad_url.status_code == 422
    assert "http://" in bad_url.json()["detail"]

    bad_key = client.post("/api/settings", json={"llm": {"providerr": "openai"}})
    assert bad_key.status_code == 422


def test_api_key_is_stored_in_memory_and_never_returned(client, monkeypatch):
    response = client.post(
        "/api/settings",
        json={"llm": {"provider": "openai", "apiKey": "sk-test-secret", "model": "gpt-4o-mini"}},
    )
    assert response.status_code == 200
    assert "sk-test-secret" not in response.text
    assert response.json()["settings"]["llm"]["apiKeySet"] is True
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # ne pas fuiter vers les autres tests
    public = client.get("/api/settings")
    assert "sk-test-secret" not in public.text


def test_reset_restores_environment_values(client, monkeypatch):
    client.post("/api/settings", json={"llm": {"provider": "openai", "mode": "off"}})
    response = client.post("/api/settings", json={"reset": "env"})
    assert response.status_code == 200
    body = response.json()
    assert body["settings"]["source"] == "env"
    assert body["settings"]["llm"]["provider"] == "anthropic"
    assert body["settings"]["llm"]["mode"] == "auto"


def test_mcp_target_is_adopted_only_when_it_answers(client, monkeypatch):
    adopted = FakeMcpClient(url="http://127.0.0.1:8765/mcp")
    monkeypatch.setattr(web_app.runtime, "build_client", lambda settings: adopted)
    web_app.app.state.conversations["conv-1"] = Conversation(id="conv-1")

    response = client.post(
        "/api/settings", json={"mcp": {"transport": "http", "url": "http://127.0.0.1:8765/mcp"}}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "mcp" in body["restarted"] and "conversations" in body["restarted"]
    assert web_app.app.state.client is adopted
    assert web_app.app.state.conversations == {}  # surfaces de l'ancien jeu de données
    assert body["settings"]["mcp"]["transport"] == "http"


def test_unreachable_mcp_target_keeps_the_previous_settings(client, monkeypatch):
    previous_client = web_app.app.state.client
    monkeypatch.setattr(
        web_app.runtime, "build_client", lambda settings: FakeMcpClient(fail=True)
    )
    response = client.post(
        "/api/settings", json={"mcp": {"transport": "http", "url": "http://127.0.0.1:9/mcp"}}
    )
    assert response.status_code == 502
    assert "Réglages précédents conservés" in response.json()["detail"]
    assert web_app.app.state.client is previous_client
    assert web_app.app.state.settings.mcp.transport == "stdio"


def test_settings_test_probes_a_target_without_saving_it(client, monkeypatch):
    monkeypatch.setattr(web_app.runtime, "build_client", lambda settings: FakeMcpClient())
    response = client.post(
        "/api/settings/test",
        json={"target": "mcp", "mcp": {"transport": "http", "url": "http://127.0.0.1:8765/mcp"}},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["dataset"]["totalRows"] == 900
    assert body["tools"] == ["get_dataset_overview", "aggregate_observations"]
    # La cible testée n'a pas été adoptée.
    assert web_app.app.state.settings.mcp.transport == "stdio"


def test_settings_test_reports_a_failure_instead_of_raising(client, monkeypatch):
    monkeypatch.setattr(
        web_app.runtime, "build_client", lambda settings: FakeMcpClient(fail=True)
    )
    body = client.post(
        "/api/settings/test",
        json={"target": "mcp", "mcp": {"transport": "http", "url": "http://127.0.0.1:9/mcp"}},
    ).json()
    assert body["ok"] is False
    assert "test" in body["error"]


def test_model_test_instantiates_the_model_without_calling_it(client, monkeypatch):
    class DummyModel:
        pass

    monkeypatch.setattr(web_app.runtime, "model_factory", lambda settings: lambda: DummyModel())
    body = client.post(
        "/api/settings/test",
        json={"target": "llm", "llm": {"mode": "on", "provider": "openai", "model": "gpt-x"}},
    ).json()
    assert body["ok"] is True
    assert "DummyModel" in body["detail"]

    def broken(settings):
        raise runtime.SettingsError("clé API manquante")

    monkeypatch.setattr(web_app.runtime, "model_factory", broken)
    failed = client.post(
        "/api/settings/test", json={"target": "llm", "llm": {"mode": "on", "provider": "openai"}}
    ).json()
    assert failed["ok"] is False
    assert "clé API manquante" in failed["error"]


def test_off_mode_freezes_the_configuration(client, monkeypatch):
    monkeypatch.setenv("LPO_WEB_SETTINGS", "off")
    assert client.get("/api/settings").json()["editable"] is False
    response = client.post("/api/settings", json={"llm": {"mode": "off"}})
    assert response.status_code == 403
    assert "LPO_WEB_SETTINGS" in response.json()["detail"]


def test_settings_require_the_token_when_configured(client, monkeypatch):
    monkeypatch.setenv("LPO_WEB_TOKEN", "secret-du-site")
    assert client.get("/api/settings").status_code == 401
    assert client.post("/api/settings", json={"llm": {"mode": "off"}}).status_code == 401
    headers = {"Authorization": "Bearer secret-du-site"}
    assert client.get("/api/settings", headers=headers).status_code == 200


def test_health_exposes_model_and_mcp_state(client):
    body = client.get("/api/health").json()
    assert body["mcp"]["ok"] is True
    assert body["mcp"]["transport"] == "stdio"
    assert body["mcp"]["tools"] == body["mcpTools"]
    assert body["llm"]["mode"] in {"auto", "on", "off"}
    assert body["settingsSource"] == "env"
