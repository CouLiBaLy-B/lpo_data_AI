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

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from lpo_a2ui.validation import catalog_id, catalog_json
from orchestrator.agent import Conversation, LpoAssistant
from orchestrator.mcp_client import McpDataClient, McpToolError

logger = logging.getLogger("lpo.web")

STATIC_DIR = Path(__file__).parent / "static"
MAX_CONVERSATIONS = 200


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = McpDataClient()
    await client.connect()
    app.state.assistant = LpoAssistant(client)
    app.state.conversations = {}
    app.state.client = client
    logger.info("Application prête (outils MCP : %s)", ", ".join(client.tool_names))
    try:
        yield
    finally:
        await client.close()


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
    return {"status": "ok", "mcpTools": client.tool_names}


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


@app.get("/api/dataset")
async def dataset() -> dict[str, Any]:
    assistant: LpoAssistant = app.state.assistant
    ctx = await assistant.ensure_context()
    return ctx.overview


@app.post("/api/ask")
async def ask(payload: AskRequest) -> JSONResponse:
    assistant: LpoAssistant = app.state.assistant
    conversation = _conversation(payload.conversationId)
    try:
        turn = await assistant.ask(payload.question, conversation)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except McpToolError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    body = turn.as_dict()
    body["conversationId"] = conversation.id
    return JSONResponse(body)


@app.post("/api/action")
async def action(payload: ActionRequest) -> JSONResponse:
    assistant: LpoAssistant = app.state.assistant
    conversation = _conversation(payload.conversationId)
    try:
        turn = await assistant.act(payload.action, conversation)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except McpToolError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    body = turn.as_dict()
    body["conversationId"] = conversation.id
    return JSONResponse(body)


@app.post("/api/reset")
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
