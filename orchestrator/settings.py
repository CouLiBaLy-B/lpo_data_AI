"""Réglages d'exécution : fournisseur de modèle et serveur MCP.

Deux sources, une seule vérité :

1. **l'environnement** (``.env``, variables d'exécution) au démarrage — c'est la
   source de production : rien ne dépend d'un clic dans un navigateur ;
2. **l'interface web** (``POST /api/settings``), qui surcharge ces valeurs pour
   la durée du processus.

Ce module ne fait que décrire, valider et sérialiser ces réglages. Il ne
construit ni client MCP ni modèle : ``build_client`` et ``model_factory``
renvoient des fabriques, ce qui permet de les remplacer dans les tests.

Règles de sécurité tenues ici :

- les clés d'API ne sont **jamais** renvoyées au client (``redacted`` ne publie
  qu'un booléen ``apiKeySet``) ni journalisées ;
- une commande ``stdio`` exécute un programme sur le serveur : le mode
  ``LPO_WEB_SETTINGS=url-only`` interdit cette voie, ``off`` ferme l'écriture ;
- les URL sont validées (http/https, sans identifiants dans l'URL) et les
  messages d'erreur sont expurgés avant d'être renvoyés.
"""

from __future__ import annotations

import os
import re
import shlex
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: ``LPO_WEB_SETTINGS`` : ``on`` (défaut), ``url-only`` ou ``off``.
EDIT_MODES = ("on", "url-only", "off")

#: Fournisseurs proposés par l'interface. Liste fermée : un identifiant inconnu
#: ferait échouer `init_chat_model` avec un message peu lisible.
PROVIDERS: tuple[dict[str, str], ...] = (
    {"id": "anthropic", "label": "Anthropic", "keyEnv": "ANTHROPIC_API_KEY"},
    {"id": "openai", "label": "OpenAI", "keyEnv": "OPENAI_API_KEY"},
    {"id": "google_genai", "label": "Google Gemini", "keyEnv": "GOOGLE_API_KEY"},
    {"id": "mistralai", "label": "Mistral AI", "keyEnv": "MISTRAL_API_KEY"},
    {"id": "groq", "label": "Groq", "keyEnv": "GROQ_API_KEY"},
    {"id": "ollama", "label": "Ollama (local, sans clé)", "keyEnv": ""},
)
PROVIDER_IDS = tuple(provider["id"] for provider in PROVIDERS)
_KEY_ENV = {provider["id"]: provider["keyEnv"] for provider in PROVIDERS}

MODES = ("auto", "on", "off")
TRANSPORTS = ("stdio", "http")

_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,79}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class SettingsError(ValueError):
    """Réglage refusé (message destiné à l'utilisateur, jamais à un journal brut)."""


# ── Modèle ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LlmSettings:
    """Fournisseur de modèle utilisé pour extraire l'intention."""

    mode: str = "auto"  # auto | on | off
    provider: str = "anthropic"
    model: str = ""
    temperature: float = 0.0
    #: Clé saisie dans l'interface. ``repr=False`` : une trace de débogage ne
    #: doit pas la contenir.
    api_key: str = field(default="", repr=False, compare=False)

    @property
    def key_env(self) -> str:
        return _KEY_ENV.get(self.provider, "")

    @property
    def api_key_set(self) -> bool:
        """Vrai si une clé est disponible (saisie ou variable d'environnement)."""
        if not self.key_env:  # fournisseur local : aucune clé attendue
            return True
        return bool(self.api_key or _usable_key(os.getenv(self.key_env, "")))

    @property
    def ready(self) -> bool:
        return bool(self.model) and self.api_key_set


def _usable_key(value: str | None) -> str:
    value = (value or "").strip()
    if not value or value.startswith("your_"):
        return ""
    return value


def model_factory(settings: LlmSettings) -> Callable[[], Any]:
    """Fabrique de modèle, importée paresseusement (``langchain`` est optionnel)."""

    def build() -> Any:
        from src.ai.model_factory import create_model

        apply_api_key(settings)
        return create_model(
            provider=settings.provider,
            model=settings.model or None,
            temperature=settings.temperature,
        )

    return build


def apply_api_key(settings: LlmSettings) -> None:
    """Place la clé saisie dans l'environnement du processus, en mémoire seulement.

    ``src.ai.model_factory`` lit l'environnement : c'est le seul point de
    passage commun entre la configuration par ``.env`` et celle par l'interface.
    """
    if settings.api_key and settings.key_env:
        os.environ[settings.key_env] = settings.api_key


def llm_enabled(settings: LlmSettings) -> bool:
    """``auto`` et ``on`` tentent le LLM ; l'échec retombe sur l'heuristique."""
    return settings.mode in {"auto", "on"}


# ── Serveur MCP ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class McpSettings:
    """Cible du client MCP : commande locale (stdio) ou serveur HTTP."""

    transport: str = "stdio"
    url: str = ""
    command: str = sys.executable
    args: tuple[str, ...] = ("-m", "lpo_data_mcp")
    cwd: str = str(PROJECT_ROOT)

    @property
    def target(self) -> str:
        """Résumé lisible de la cible, sans secret."""
        if self.transport == "http":
            return self.url
        return " ".join([Path(self.command).name, *self.args]).strip()


@dataclass(frozen=True)
class RuntimeSettings:
    llm: LlmSettings = field(default_factory=LlmSettings)
    mcp: McpSettings = field(default_factory=McpSettings)
    source: str = "env"  # env | ui


# ── Lecture de l'environnement ───────────────────────────────────────────────


def _env_llm_mode() -> str:
    raw = os.getenv("LPO_LLM_INTENT", "auto").strip().lower()
    if raw in {"0", "off", "false", "never"}:
        return "off"
    if raw in {"1", "on", "true", "always"}:
        return "on"
    return "auto"


def _env_provider_model() -> tuple[str, str]:
    provider = os.getenv("AI_PROVIDER", "").strip()
    model = os.getenv("AI_MODEL", "").strip()
    if provider and model:
        return provider, model
    try:  # ``.env`` chargé par pydantic-settings, sans dépendre de langchain
        from src.config.settings import settings as app_settings

        return provider or app_settings.ai.provider, model or app_settings.ai.model
    except Exception:  # noqa: BLE001 - configuration absente : valeurs neutres
        return provider or "anthropic", model


def from_env() -> RuntimeSettings:
    """Réglages par défaut, tels que fournis par l'environnement."""
    provider, model = _env_provider_model()
    if provider not in PROVIDER_IDS:
        provider = "anthropic"
    url = os.getenv("LPO_MCP_URL", "").strip()
    raw_args = os.getenv("LPO_MCP_ARGS", "").strip()
    args = tuple(shlex.split(raw_args)) if raw_args else ("-m", "lpo_data_mcp")
    mcp = McpSettings(
        transport="http" if url else "stdio",
        url=url,
        command=os.getenv("LPO_MCP_COMMAND", "").strip() or sys.executable,
        args=args,
        cwd=str(PROJECT_ROOT),
    )
    return RuntimeSettings(
        llm=LlmSettings(mode=_env_llm_mode(), provider=provider, model=model),
        mcp=mcp,
        source="env",
    )


# ── Validation d'un patch ────────────────────────────────────────────────────


def edit_mode() -> str:
    raw = os.getenv("LPO_WEB_SETTINGS", "on").strip().lower()
    return raw if raw in EDIT_MODES else "on"


def _clean_text(value: Any, *, label: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise SettingsError(f"{label} : texte attendu.")
    text = value.strip()
    if _CONTROL_RE.search(text):
        raise SettingsError(f"{label} : caractères de contrôle interdits.")
    if len(text) > max_length:
        raise SettingsError(f"{label} : {max_length} caractères maximum.")
    return text


def _validate_url(value: str) -> str:
    if not value:
        raise SettingsError("URL du serveur MCP obligatoire pour le transport HTTP.")
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"}:
        raise SettingsError("URL du serveur MCP : http:// ou https:// attendu.")
    if not parts.netloc:
        raise SettingsError("URL du serveur MCP : hôte manquant.")
    if "@" in parts.netloc:
        raise SettingsError(
            "URL du serveur MCP : identifiants interdits dans l'URL "
            "(utiliser un proxy ou des en-têtes côté serveur)."
        )
    return value


def _validate_mcp(current: McpSettings, patch: dict[str, Any]) -> McpSettings:
    unknown = set(patch) - {"transport", "url", "command", "args", "cwd"}
    if unknown:
        raise SettingsError(f"Réglage MCP inconnu : {', '.join(sorted(unknown))}.")
    transport = str(patch.get("transport", current.transport)).strip().lower()
    if transport not in TRANSPORTS:
        raise SettingsError("Transport MCP : « stdio » ou « http » attendu.")
    if transport == "stdio" and edit_mode() == "url-only":
        raise SettingsError(
            "Ce serveur n'accepte que le transport HTTP "
            "(LPO_WEB_SETTINGS=url-only) : une commande locale exécuterait un "
            "programme sur la machine."
        )

    url = current.url
    if "url" in patch:
        url = _validate_url(_clean_text(patch["url"], label="URL", max_length=500))
    command = current.command
    if "command" in patch:
        command = _clean_text(patch["command"], label="Commande", max_length=300)
        if not command:
            raise SettingsError("Commande du serveur MCP obligatoire.")
        if not isinstance(patch["command"], str) or "\n" in patch["command"]:
            raise SettingsError("Commande du serveur MCP : une seule ligne attendue.")
    args = current.args
    if "args" in patch:
        raw = patch["args"]
        if isinstance(raw, str):
            raw = shlex.split(raw)
        if not isinstance(raw, (list, tuple)) or not all(isinstance(a, str) for a in raw):
            raise SettingsError("Arguments du serveur MCP : liste de textes attendue.")
        if len(raw) > 12:
            raise SettingsError("Arguments du serveur MCP : 12 au maximum.")
        args = tuple(_clean_text(a, label="Argument", max_length=300) for a in raw)
    cwd = current.cwd
    if "cwd" in patch:
        cwd = _clean_text(patch["cwd"] or current.cwd, label="Répertoire", max_length=500)
        if not Path(cwd).is_dir():
            raise SettingsError("Répertoire de travail : dossier inexistant.")

    if transport == "http":
        url = _validate_url(url)
    elif not command:
        raise SettingsError("Commande du serveur MCP obligatoire.")
    return McpSettings(transport=transport, url=url, command=command, args=args, cwd=cwd)


def _validate_llm(current: LlmSettings, patch: dict[str, Any]) -> LlmSettings:
    unknown = set(patch) - {"mode", "provider", "model", "temperature", "apiKey", "clearApiKey"}
    if unknown:
        raise SettingsError(f"Réglage de modèle inconnu : {', '.join(sorted(unknown))}.")

    mode = str(patch.get("mode", current.mode)).strip().lower()
    if mode not in MODES:
        raise SettingsError("Mode d'intention : « auto », « on » ou « off » attendu.")

    provider = str(patch.get("provider", current.provider)).strip().lower()
    if provider not in PROVIDER_IDS:
        raise SettingsError(
            f"Fournisseur inconnu : {provider or '—'} (choix : {', '.join(PROVIDER_IDS)})."
        )

    model = current.model
    if "model" in patch:
        model = _clean_text(patch["model"], label="Modèle", max_length=80)
        if model and not _MODEL_RE.match(model):
            raise SettingsError(
                "Nom de modèle invalide : lettres, chiffres, « . _ : / - » uniquement."
            )

    temperature = current.temperature
    if "temperature" in patch:
        try:
            temperature = float(patch["temperature"])
        except (TypeError, ValueError) as exc:
            raise SettingsError("Température : nombre attendu.") from exc
        if not 0.0 <= temperature <= 1.0:
            raise SettingsError("Température : entre 0 et 1.")

    api_key = current.api_key
    if patch.get("clearApiKey"):
        api_key = ""
    elif "apiKey" in patch:
        value = patch["apiKey"]
        if not isinstance(value, str):
            raise SettingsError("Clé API : texte attendu.")
        value = value.strip()
        if value:  # une chaîne vide signifie « ne pas changer »
            api_key = _clean_text(value, label="Clé API", max_length=300)

    return LlmSettings(
        mode=mode,
        provider=provider,
        model=model,
        temperature=temperature,
        api_key=api_key,
    )


def merge(current: RuntimeSettings, patch: dict[str, Any]) -> RuntimeSettings:
    """Applique un patch validé à des réglages existants.

    ``{"reset": "env"}`` revient aux valeurs d'environnement ; sinon seules les
    clés ``llm`` et/ou ``mcp`` sont acceptées, et les champs absents sont
    conservés.
    """
    if not isinstance(patch, dict):
        raise SettingsError("Corps de requête : objet JSON attendu.")
    if str(patch.get("reset") or "").strip().lower() == "env":
        return from_env()
    unknown = set(patch) - {"llm", "mcp"}
    if unknown:
        raise SettingsError(f"Clé inconnue : {', '.join(sorted(unknown))}.")

    llm_patch = patch.get("llm") or {}
    mcp_patch = patch.get("mcp") or {}
    if not isinstance(llm_patch, dict) or not isinstance(mcp_patch, dict):
        raise SettingsError("« llm » et « mcp » doivent être des objets JSON.")

    llm = _validate_llm(current.llm, llm_patch) if llm_patch else current.llm
    mcp = _validate_mcp(current.mcp, mcp_patch) if mcp_patch else current.mcp
    return RuntimeSettings(llm=llm, mcp=mcp, source="ui")


# ── Sérialisation vers le client ─────────────────────────────────────────────


def redacted(settings: RuntimeSettings) -> dict[str, Any]:
    """Vue publique : aucune clé, aucun secret, seulement des booléens d'état."""
    llm = settings.llm
    return {
        "source": settings.source,
        "editable": edit_mode() != "off",
        "editMode": edit_mode(),
        "llm": {
            "mode": llm.mode,
            "provider": llm.provider,
            "model": llm.model,
            "temperature": llm.temperature,
            "apiKeySet": llm.api_key_set,
            "keyEnv": llm.key_env,
            "ready": llm.ready,
            "providers": [
                {
                    "id": provider["id"],
                    "label": provider["label"],
                    "keyEnv": provider["keyEnv"],
                    "ready": (
                        bool(_usable_key(os.getenv(provider["keyEnv"], "")))
                        if provider["keyEnv"]
                        else True
                    ),
                }
                for provider in PROVIDERS
            ],
        },
        "mcp": {
            "transport": settings.mcp.transport,
            "url": settings.mcp.url,
            "command": settings.mcp.command,
            "args": list(settings.mcp.args),
            "cwd": settings.mcp.cwd,
            "target": settings.mcp.target,
        },
        "environment": {
            "profile": os.getenv("LPO_PROFILE", "observations"),
            "table": os.getenv("LPO_OBS_TABLE", ""),
            "readOnly": os.getenv("DB_READ_ONLY", "true"),
        },
    }


def build_client(settings: McpSettings) -> Any:
    """Client MCP correspondant aux réglages (import local : dépendance lourde)."""
    from orchestrator.mcp_client import McpDataClient

    if settings.transport == "http":
        return McpDataClient(url=settings.url)
    return McpDataClient(
        url=None,
        command=settings.command,
        args=list(settings.args),
        cwd=settings.cwd,
    )


def with_source(settings: RuntimeSettings, source: str) -> RuntimeSettings:
    """Copie marquée « ui » (utile aux tests et à la traçabilité)."""
    return replace(settings, source=source)


def describe_llm_error(exc: Exception) -> str:
    """Message d'erreur compréhensible pour un réglage de modèle.

    Les cas fréquents sont traduits ; le reste est renvoyé tel quel, expurgé.
    """
    text = mask_secrets(str(exc) or exc.__class__.__name__)[:400]
    if isinstance(exc, ImportError) or "No module named" in text:
        return (
            "Dépendances du modèle absentes de cet environnement "
            f"({text}). Le mode heuristique reste utilisable."
        )
    if "Clé API manquante" in text or "api_key" in text.lower():
        return f"{text} Renseigner la clé du fournisseur ci-dessous."
    return text


def mask_secrets(text: str) -> str:
    """Retire les identifiants éventuellement présents dans un message d'erreur."""
    return re.sub(r"(https?://)[^/\s@]+@", r"\1", str(text))
