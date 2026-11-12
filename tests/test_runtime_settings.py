"""Réglages d'exécution : validation, expurgation et application.

Ces tests ne touchent ni au réseau ni à un modèle réel : on vérifie ce qui
compte pour un réglage modifiable depuis une interface — qu'un identifiant
inconnu soit refusé, qu'aucune clé ne ressorte, et qu'une cible MCP ne soit
adoptée que si elle répond.
"""

from __future__ import annotations

import pytest

from orchestrator import settings as runtime


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "LPO_LLM_INTENT",
        "AI_PROVIDER",
        "AI_MODEL",
        "LPO_MCP_URL",
        "LPO_MCP_COMMAND",
        "LPO_MCP_ARGS",
        "LPO_WEB_SETTINGS",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_from_env_reads_mode_provider_and_transport(clean_env):
    clean_env.setenv("LPO_LLM_INTENT", "off")
    clean_env.setenv("AI_PROVIDER", "openai")
    clean_env.setenv("AI_MODEL", "gpt-4o-mini")
    clean_env.setenv("LPO_MCP_URL", "http://127.0.0.1:8765/mcp")
    settings = runtime.from_env()
    assert settings.llm.mode == "off"
    assert settings.llm.provider == "openai"
    assert settings.llm.model == "gpt-4o-mini"
    assert settings.mcp.transport == "http"
    assert settings.mcp.target == "http://127.0.0.1:8765/mcp"


def test_unknown_provider_is_refused():
    current = runtime.from_env()
    with pytest.raises(runtime.SettingsError, match="Fournisseur inconnu"):
        runtime.merge(current, {"llm": {"provider": "skynet"}})


def test_invalid_model_name_is_refused():
    current = runtime.from_env()
    with pytest.raises(runtime.SettingsError, match="Nom de modèle invalide"):
        runtime.merge(current, {"llm": {"model": "modèle avec espaces"}})


def test_http_transport_requires_a_http_url():
    current = runtime.from_env()
    for url in ("ftp://exemple", "pas une url", "http://user:pass@exemple/mcp"):
        with pytest.raises(runtime.SettingsError):
            runtime.merge(current, {"mcp": {"transport": "http", "url": url}})


def test_stdio_requires_a_command():
    current = runtime.from_env()
    with pytest.raises(runtime.SettingsError, match="Commande"):
        runtime.merge(current, {"mcp": {"transport": "stdio", "command": "  "}})


def test_args_are_validated_and_stored_as_strings():
    current = runtime.from_env()
    merged = runtime.merge(
        current,
        {"mcp": {"transport": "stdio", "command": "python", "args": ["-m", "lpo_data_mcp"]}},
    )
    assert merged.mcp.args == ("-m", "lpo_data_mcp")
    with pytest.raises(runtime.SettingsError, match="Arguments"):
        runtime.merge(current, {"mcp": {"args": [1, 2]}})


def test_unknown_keys_are_reported():
    current = runtime.from_env()
    with pytest.raises(runtime.SettingsError, match="inconnu"):
        runtime.merge(current, {"llm": {"providerr": "openai"}})
    with pytest.raises(runtime.SettingsError, match="Clé inconnue"):
        runtime.merge(current, {"modele": {}})


def test_reset_returns_environment_values(clean_env):
    clean_env.setenv("LPO_LLM_INTENT", "off")
    current = runtime.from_env()
    changed = runtime.merge(current, {"llm": {"mode": "on", "provider": "openai"}})
    assert changed.source == "ui"
    back = runtime.merge(changed, {"reset": "env"})
    assert back.source == "env"
    assert back.llm.mode == "off"


def test_api_key_is_never_exposed_and_can_be_cleared(clean_env):
    current = runtime.from_env()
    merged = runtime.merge(current, {"llm": {"provider": "openai", "apiKey": "sk-test-123"}})
    public = runtime.redacted(merged)
    assert public["llm"]["apiKeySet"] is True
    assert "sk-test-123" not in str(public)
    assert merged.llm.api_key not in repr(merged)  # repr ne fuit pas la clé

    cleared = runtime.merge(merged, {"llm": {"clearApiKey": True}})
    assert cleared.llm.api_key == ""
    assert runtime.redacted(cleared)["llm"]["apiKeySet"] is False


def test_empty_api_key_means_unchanged(clean_env):
    current = runtime.from_env()
    with_key = runtime.merge(current, {"llm": {"provider": "openai", "apiKey": "sk-test"}})
    same = runtime.merge(with_key, {"llm": {"apiKey": ""}})
    assert same.llm.api_key == "sk-test"


def test_apply_api_key_exports_the_secret_to_the_environment(clean_env):
    merged = runtime.merge(runtime.from_env(), {"llm": {"provider": "openai", "apiKey": "sk-x"}})
    runtime.apply_api_key(merged.llm)
    import os

    assert os.environ["OPENAI_API_KEY"] == "sk-x"


def test_url_only_mode_forbids_local_commands(clean_env):
    clean_env.setenv("LPO_WEB_SETTINGS", "url-only")
    assert runtime.edit_mode() == "url-only"
    with pytest.raises(runtime.SettingsError, match="url-only"):
        runtime.merge(runtime.from_env(), {"mcp": {"transport": "stdio", "command": "python"}})
    # Le transport HTTP reste permis.
    merged = runtime.merge(
        runtime.from_env(), {"mcp": {"transport": "http", "url": "http://127.0.0.1:9/mcp"}}
    )
    assert merged.mcp.transport == "http"


def test_off_mode_is_reported_as_not_editable(clean_env):
    clean_env.setenv("LPO_WEB_SETTINGS", "off")
    assert runtime.edit_mode() == "off"
    assert runtime.redacted(runtime.from_env())["editable"] is False


def test_mask_secrets_removes_credentials_from_errors():
    assert runtime.mask_secrets("échec https://user:pwd@db:5432/x") == "échec https://db:5432/x"


def test_build_client_forces_the_configured_transport(clean_env):
    clean_env.setenv("LPO_MCP_URL", "http://127.0.0.1:8765/mcp")
    stdio = runtime.build_client(
        runtime.McpSettings(transport="stdio", command="python", args=("-m", "lpo_data_mcp"))
    )
    assert stdio.url is None  # l'URL de l'environnement ne reprend pas la main
    http = runtime.build_client(runtime.McpSettings(transport="http", url="http://hote/mcp"))
    assert http.url == "http://hote/mcp"
