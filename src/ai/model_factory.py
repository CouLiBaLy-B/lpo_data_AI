"""
Model factory — abstraction provider-agnostique.

Changer de fournisseur de LLM ne demande que de modifier AI_PROVIDER / AI_MODEL
dans .env. Tout le reste du code dépend de cette fabrique, jamais d'un SDK
particulier.

Providers supportés nativement par init_chat_model :
    anthropic, openai, google_genai, groq, mistralai, ollama, bedrock, ...
"""

from __future__ import annotations

import logging
import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from ..config.settings import settings

logger = logging.getLogger(__name__)

# Clé d'environnement attendue par chaque provider
_PROVIDER_ENV_KEY = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
}

# Mapping clé de settings → variable d'env, pour propager la clé saisie dans .env
_SETTINGS_KEY = {
    "anthropic": settings.ai.anthropic_api_key,
    "openai": settings.ai.openai_api_key,
    "google_genai": settings.ai.google_api_key,
}


class ModelConfigError(RuntimeError):
    """Configuration LLM invalide ou incomplète."""


def _ensure_api_key(provider: str) -> None:
    """Garantit que la clé du provider est présente dans l'environnement."""
    env_key = _PROVIDER_ENV_KEY.get(provider)
    if env_key is None:
        return  # provider local (ollama, etc.) : pas de clé requise

    value = os.environ.get(env_key) or _SETTINGS_KEY.get(provider, "")
    if not value or value.startswith("your_"):
        raise ModelConfigError(
            f"Clé API manquante pour le provider '{provider}'. "
            f"Renseignez {env_key} dans le fichier .env."
        )
    os.environ[env_key] = value


def create_model(
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """
    Construit une instance de chat model prête à l'emploi.

    Les valeurs par défaut viennent de settings (.env). Tout paramètre passé
    explicitement les surcharge — utile pour, par ex., un modèle plus puissant
    sur l'agent et un modèle léger ailleurs.
    """
    provider = provider or settings.ai.provider
    model = model or settings.ai.model
    temperature = settings.ai.temperature if temperature is None else temperature
    max_tokens = max_tokens or settings.ai.max_tokens

    _ensure_api_key(provider)

    try:
        llm = init_chat_model(
            model,
            model_provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.info("Modèle initialisé: %s:%s", provider, model)
        return llm
    except Exception as e:  # noqa: BLE001
        raise ModelConfigError(
            f"Impossible d'initialiser {provider}:{model} — {e}"
        ) from e


def available_providers() -> list[str]:
    """Providers pour lesquels une clé est configurée (UI / diagnostic)."""
    ready = []
    for provider, env_key in _PROVIDER_ENV_KEY.items():
        value = os.environ.get(env_key) or _SETTINGS_KEY.get(provider, "")
        if value and not value.startswith("your_"):
            ready.append(provider)
    return ready
