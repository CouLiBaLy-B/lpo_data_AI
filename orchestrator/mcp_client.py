"""Client MCP côté agent (SDK officiel `mcp`).

Transport par défaut : **stdio** — l'orchestrateur lance
``python -m lpo_data_mcp`` en sous-processus, ce qui isole totalement l'accès aux
données de l'agent et de l'interface web. ``LPO_MCP_URL`` permet de basculer sur
un serveur `streamable-http` déjà déployé.

Le client maintient **une** session pour la durée de vie de l'application (le
sous-processus est coûteux à démarrer) et sérialise les appels.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import Client
from mcp.client.stdio import StdioServerParameters

logger = logging.getLogger("lpo_mcp.client")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Valeur par défaut de `url` : tant qu'elle est utilisée, `LPO_MCP_URL` décide.
#: `url=None` force au contraire le transport stdio (réglages de l'interface).
_ENV_URL = object()


class McpToolError(RuntimeError):
    """Appel d'outil MCP en erreur (message renvoyé par le serveur)."""

    def __init__(self, tool: str, message: str):
        self.tool = tool
        self.message = message
        super().__init__(f"{tool}: {message}")


class McpDataClient:
    """Client des outils `lpo-data`, en lecture seule."""

    def __init__(
        self,
        *,
        url: str | None | object = _ENV_URL,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: Path | str | None = None,
    ):
        if url is _ENV_URL:
            self.url = os.getenv("LPO_MCP_URL") or None
        else:
            self.url = (str(url).strip() if url else "") or None
        self.command = command or os.getenv("LPO_MCP_COMMAND") or sys.executable
        self.args = args or ["-m", "lpo_data_mcp"]
        self.cwd = Path(cwd or PROJECT_ROOT)
        self._stack: AsyncExitStack | None = None
        self._client: Client | None = None
        self._lock = asyncio.Lock()
        self._tools: list[str] = []

    async def connect(self) -> None:
        if self._client is not None:
            return
        stack = AsyncExitStack()
        try:
            if self.url:
                logger.info("Connexion MCP (streamable-http) : %s", self.url)
                client = await stack.enter_async_context(Client(self.url))
            else:
                params = StdioServerParameters(
                    command=self.command,
                    args=list(self.args),
                    cwd=str(self.cwd),
                    env=dict(os.environ),
                )
                logger.info("Connexion MCP (stdio) : %s %s", self.command, " ".join(self.args))
                client = await stack.enter_async_context(Client(params))
        except Exception:
            await stack.aclose()
            raise
        self._stack = stack
        self._client = client
        tools = await client.list_tools()
        self._tools = [t.name for t in tools.tools]
        logger.info("Outils MCP disponibles : %s", ", ".join(self._tools))

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._client = None

    async def __aenter__(self) -> McpDataClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools)

    async def call(self, tool: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Appelle un outil et renvoie son contenu structuré (dict)."""
        if self._client is None:
            await self.connect()
        assert self._client is not None

        async with self._lock:
            result = await self._client.call_tool(tool, arguments or {})

        if getattr(result, "is_error", False):
            text = _text_of(result)
            raise McpToolError(tool, text or "erreur inconnue")

        structured = getattr(result, "structured_content", None)
        if isinstance(structured, dict):
            # Le SDK enveloppe parfois les valeurs simples sous {"result": ...}.
            if set(structured) == {"result"} and isinstance(structured["result"], dict):
                return structured["result"]
            return structured
        text = _text_of(result)
        try:
            return json.loads(text)
        except (TypeError, ValueError):
            return {"summary": text}


def _text_of(result: Any) -> str:
    """Concatène les blocs texte d'un CallToolResult."""
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(str(text))
    return "\n".join(parts).strip()
