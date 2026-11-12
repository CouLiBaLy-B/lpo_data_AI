"""Application web de démonstration : client A2UI + agent orchestrateur.

Architecture :

    [Navigateur — renderer A2UI (JS, /static/renderer.js)]
        ⇅ HTTP/JSON (REST)
    [FastAPI — cet app]
        ⇅ client MCP (SDK officiel, stdio ou streamable-http)
    [Serveur MCP lpo-data]
        ⇅ SQLAlchemy lecture seule
    [SQLite / PostgreSQL]

Le client ne reçoit que des messages A2UI v1.0 validés et un résumé texte. Il
n'exécute aucun code fourni par le serveur.

Lancement :

    uvicorn web.app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from lpo_a2ui.validation import catalog_id, catalog_json
from orchestrator import settings as runtime
from orchestrator.agent import Conversation, LpoAssistant
from orchestrator.mcp_client import McpDataClient, McpToolError

logger = logging.getLogger("lpo.web")

STATIC_DIR = Path(__file__).parent / "static"
MAX_CONVERSATIONS = 200


def require_token(request: Request) -> None:
    """Jeton bearer optionnel (`LPO_WEB_TOKEN`).

    Sans variable d'environnement, l'API reste ouverte (démo locale). Dès qu'un
    jeton est configuré, tous les points d'entrée qui touchent aux données
    l'exigent : une instance exposée sur Internet ne peut pas être interrogée
    par n'importe qui. `/api/health` et `/api/catalog` restent publics (aucune
    donnée, utile aux sondes de disponibilité).
    """
    expected = os.getenv("LPO_WEB_TOKEN", "").strip()
    if not expected:
        return
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(token.strip(), expected):
        raise HTTPException(
            status_code=401,
            detail="Jeton d'accès requis (en-tête Authorization: Bearer …).",
            headers={"WWW-Authenticate": "Bearer"},
        )


def current_settings() -> runtime.RuntimeSettings:
    """Réglages effectifs (interface si elle a écrit, environnement sinon)."""
    settings = getattr(app.state, "settings", None)
    if settings is None:
        settings = runtime.from_env()
        app.state.settings = settings
    return settings


def build_assistant(client: McpDataClient, settings: runtime.RuntimeSettings) -> LpoAssistant:
    """Assistant configuré par les réglages courants."""
    enabled = runtime.llm_enabled(settings.llm)
    return LpoAssistant(
        client,
        use_llm_intent=enabled,
        llm_factory=runtime.model_factory(settings.llm) if enabled else None,
    )


async def connect_mcp(client: McpDataClient) -> dict[str, Any]:
    """Tente une connexion et renvoie un état sérialisable (jamais d'exception).

    Un serveur MCP injoignable ne doit pas empêcher l'interface de démarrer :
    c'est précisément dans ce cas que l'utilisateur ouvre **Configuration** pour
    corriger la cible.
    """
    started = time.perf_counter()
    try:
        await client.connect()
    except Exception as exc:  # noqa: BLE001 - erreur de transport, message expurgé
        detail = runtime.mask_secrets(str(exc) or exc.__class__.__name__)
        logger.warning("Connexion MCP impossible : %s", detail)
        return {
            "ok": False,
            "tools": [],
            "error": detail.splitlines()[0][:400],
            "durationMs": int((time.perf_counter() - started) * 1000),
        }
    return {
        "ok": True,
        "tools": client.tool_names,
        "error": None,
        "durationMs": int((time.perf_counter() - started) * 1000),
    }


async def close_quietly(client: Any) -> None:
    close = getattr(client, "close", None)
    if close is None:
        return
    try:
        await close()
    except Exception as exc:  # noqa: BLE001 - fermeture au mieux
        logger.info("Fermeture du client MCP incomplète : %s", runtime.mask_secrets(str(exc)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = runtime.from_env()
    runtime.apply_api_key(settings.llm)
    client = runtime.build_client(settings.mcp)
    status = await connect_mcp(client)
    app.state.settings = settings
    app.state.client = client
    app.state.mcp_status = status
    app.state.conversations = {}
    app.state.assistant = build_assistant(client, settings)
    if status["ok"]:
        logger.info(
            "Application prête (%s, outils MCP : %s)",
            settings.mcp.target,
            ", ".join(status["tools"]),
        )
    else:
        logger.warning(
            "Application démarrée sans serveur MCP (%s) : configurer la cible dans l'interface.",
            settings.mcp.target,
        )
    try:
        yield
    finally:
        await close_quietly(client)


app = FastAPI(
    title="LPO Data AI — agent A2UI",
    description=(
        "Agent conversationnel sur les données naturalistes LPO. Réponses "
        "graphiques via A2UI v1.0, données exposées par le serveur MCP lpo-data."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    conversationId: str | None = None


class ActionRequest(BaseModel):
    conversationId: str = Field(..., min_length=1)
    surfaceId: str | None = None
    action: dict[str, Any] = Field(..., description="Action A2UI {name, context}.")
    dataModel: dict[str, Any] | None = None


class SettingsPayload(BaseModel):
    """Patch de réglages : ``llm`` et/ou ``mcp``, ou ``reset: "env"``."""

    llm: dict[str, Any] | None = None
    mcp: dict[str, Any] | None = None
    reset: str | None = None


class SettingsTestPayload(BaseModel):
    """Test d'une cible **sans** l'enregistrer."""

    target: Literal["mcp", "llm"]
    llm: dict[str, Any] | None = None
    mcp: dict[str, Any] | None = None


def settings_lock() -> asyncio.Lock:
    """Sérialise les changements de configuration (un seul à la fois)."""
    lock = getattr(app.state, "settings_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        app.state.settings_lock = lock
    return lock


def mcp_failure(exc: Exception, *, action: str) -> HTTPException:
    """Traduit une panne MCP en 502 lisible, sans secret dans le message."""
    logger.warning("%s : %s", action, exc, exc_info=True)
    detail = runtime.mask_secrets(str(exc) or exc.__class__.__name__).splitlines()[0][:400]
    return HTTPException(
        status_code=502,
        detail=f"{action} : {detail}. Vérifier la cible MCP dans Configuration.",
    )


def _conversation(request_id: str | None) -> Conversation:
    store: dict[str, Conversation] = app.state.conversations
    if not request_id:
        conversation = Conversation()
        store[conversation.id] = conversation
    else:
        conversation = store.get(request_id)
        if conversation is None:
            conversation = Conversation(id=request_id)
            store[conversation.id] = conversation
    if len(store) > MAX_CONVERSATIONS:
        for key in list(store)[: len(store) - MAX_CONVERSATIONS]:
            if key != conversation.id:
                store.pop(key, None)
    return conversation


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    client: McpDataClient = app.state.client
    settings = current_settings()
    status = getattr(app.state, "mcp_status", None) or {}
    return {
        "status": "ok",
        "mcpTools": client.tool_names,
        "mcp": {
            "ok": bool(status.get("ok", bool(client.tool_names))),
            "transport": settings.mcp.transport,
            "target": settings.mcp.target,
            "tools": list(client.tool_names),
            "error": status.get("error"),
            "durationMs": status.get("durationMs"),
        },
        "llm": {
            "mode": settings.llm.mode,
            "provider": settings.llm.provider,
            "model": settings.llm.model,
            "ready": settings.llm.ready,
            "enabled": bool(getattr(app.state.assistant, "use_llm_intent", False)),
        },
        "settingsSource": settings.source,
    }


@app.get("/api/catalog")
def catalog() -> dict[str, Any]:
    """Catalogue A2UI servi au client (source de vérité des composants)."""
    payload = catalog_json()
    return {
        "protocolVersion": payload.get("protocolVersion"),
        "catalogId": catalog_id(),
        "title": payload.get("title"),
        "instructions": payload.get("instructions"),
        "components": sorted((payload.get("components") or {}).keys()),
        "catalog": payload,
        "protocolDocs": "https://a2ui.org/specification/v1.0-a2ui/",
    }


@app.get("/api/dataset", dependencies=[Depends(require_token)])
async def dataset() -> JSONResponse:
    assistant: LpoAssistant = app.state.assistant
    try:
        ctx = await assistant.ensure_context()
    except Exception as exc:  # noqa: BLE001 - panne de transport ou d'outil
        raise mcp_failure(exc, action="Aperçu du jeu de données indisponible") from exc
    return JSONResponse(ctx.overview)


@app.post("/api/ask", dependencies=[Depends(require_token)])
async def ask(payload: AskRequest) -> JSONResponse:
    assistant: LpoAssistant = app.state.assistant
    conversation = _conversation(payload.conversationId)
    try:
        turn = await assistant.ask(payload.question, conversation)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except McpToolError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - transport MCP coupé
        raise mcp_failure(exc, action="Question non traitée") from exc
    body = turn.as_dict()
    body["conversationId"] = conversation.id
    return JSONResponse(body)


@app.post("/api/action", dependencies=[Depends(require_token)])
async def action(payload: ActionRequest) -> JSONResponse:
    assistant: LpoAssistant = app.state.assistant
    conversation = _conversation(payload.conversationId)
    try:
        turn = await assistant.act(payload.action, conversation)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except McpToolError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - transport MCP coupé
        raise mcp_failure(exc, action="Action non appliquée") from exc
    body = turn.as_dict()
    body["conversationId"] = conversation.id
    return JSONResponse(body)


@app.get("/api/settings", dependencies=[Depends(require_token)])
async def read_settings() -> dict[str, Any]:
    """Réglages effectifs : modèle et cible MCP, sans aucun secret."""
    settings = current_settings()
    body = runtime.redacted(settings)
    body["mcp"]["status"] = getattr(app.state, "mcp_status", None)
    return body


@app.post("/api/settings", dependencies=[Depends(require_token)])
async def write_settings(payload: SettingsPayload) -> dict[str, Any]:
    """Applique un patch de réglages au processus courant.

    Un changement de cible MCP n'est retenu que si la nouvelle cible répond :
    on préfère conserver une configuration qui fonctionne plutôt que de laisser
    l'interface sans données. Les conversations sont alors oubliées (leurs
    surfaces décrivent l'ancien jeu de données).
    """
    if runtime.edit_mode() == "off":
        raise HTTPException(
            status_code=403,
            detail="Configuration figée par l'environnement (LPO_WEB_SETTINGS=off).",
        )
    async with settings_lock():
        current = current_settings()
        try:
            new = runtime.merge(current, payload.model_dump(exclude_none=True))
        except runtime.SettingsError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        runtime.apply_api_key(new.llm)
        restarted: list[str] = []
        if new.mcp != current.mcp:
            candidate = runtime.build_client(new.mcp)
            status = await connect_mcp(candidate)
            if not status["ok"]:
                await close_quietly(candidate)
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"Serveur MCP injoignable ({new.mcp.target}) : "
                        f"{status['error']}. Réglages précédents conservés."
                    ),
                )
            await close_quietly(app.state.client)
            app.state.client = candidate
            app.state.mcp_status = status
            app.state.conversations.clear()
            app.state.assistant = build_assistant(candidate, new)
            restarted.extend(["mcp", "conversations"])
        if not restarted and new.llm != current.llm:
            enabled = runtime.llm_enabled(new.llm)
            app.state.assistant.configure_llm(
                use_llm_intent=enabled,
                llm_factory=runtime.model_factory(new.llm) if enabled else None,
            )
            restarted.append("llm")

        app.state.settings = new
        logger.info(
            "Réglages appliqués (%s) : modèle %s:%s (%s), MCP %s",
            new.source,
            new.llm.provider,
            new.llm.model or "—",
            new.llm.mode,
            new.mcp.target,
        )
    return {
        "settings": runtime.redacted(new),
        "restarted": restarted,
        "mcp": app.state.mcp_status,
    }


@app.post("/api/settings/test", dependencies=[Depends(require_token)])
async def test_settings(payload: SettingsTestPayload) -> dict[str, Any]:
    """Vérifie une cible MCP ou l'instanciation du modèle, sans rien enregistrer."""
    if runtime.edit_mode() == "off":
        raise HTTPException(
            status_code=403,
            detail="Configuration figée par l'environnement (LPO_WEB_SETTINGS=off).",
        )
    current = current_settings()
    patch = {payload.target: payload.mcp if payload.target == "mcp" else payload.llm}
    try:
        candidate = runtime.merge(current, {key: value for key, value in patch.items() if value})
    except runtime.SettingsError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if payload.target == "llm":
        if not runtime.llm_enabled(candidate.llm):
            return {
                "target": "llm",
                "ok": True,
                "detail": "Mode « off » : l'heuristique déterministe est utilisée.",
            }
        if not candidate.llm.model:
            return {"target": "llm", "ok": False, "error": "Aucun nom de modèle renseigné."}
        try:
            model = runtime.model_factory(candidate.llm)()
        except Exception as exc:  # noqa: BLE001 - clé absente, dépendance manquante…
            return {
                "target": "llm",
                "ok": False,
                "error": runtime.describe_llm_error(exc),
            }
        return {
            "target": "llm",
            "ok": True,
            "detail": (
                f"Modèle instancié ({type(model).__name__}) — aucun appel réseau effectué "
                "à ce stade."
            ),
        }

    client = runtime.build_client(candidate.mcp)
    started = time.perf_counter()
    try:
        overview = await asyncio.wait_for(client.call("get_dataset_overview"), timeout=20)
    except asyncio.TimeoutError:
        return {"target": "mcp", "ok": False, "error": "Délai dépassé (20 s)."}
    except Exception as exc:  # noqa: BLE001 - dialogue MCP
        return {
            "target": "mcp",
            "ok": False,
            "error": runtime.mask_secrets(str(exc) or exc.__class__.__name__)[:400],
        }
    finally:
        await close_quietly(client)
    return {
        "target": "mcp",
        "ok": True,
        "transport": candidate.mcp.transport,
        "targetLabel": candidate.mcp.target,
        "tools": list(client.tool_names),
        "dataset": {
            "label": overview.get("dataset_label") or overview.get("description"),
            "profile": overview.get("profile"),
            "totalRows": overview.get("total_rows"),
            "recordsLabel": overview.get("records_label"),
            "isDemo": overview.get("is_demo"),
        },
        "durationMs": int((time.perf_counter() - started) * 1000),
    }


@app.post("/api/reset", dependencies=[Depends(require_token)])
def reset(payload: dict[str, Any] | None = None) -> dict[str, str]:
    conversation_id = (payload or {}).get("conversationId")
    if conversation_id:
        app.state.conversations.pop(conversation_id, None)
    return {"status": "reset"}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "web.app:app",
        host=os.getenv("LPO_WEB_HOST", "0.0.0.0"),
        port=int(os.getenv("LPO_WEB_PORT", "8080")),
    )
