"""
Middleware d'audit SQL — defense-in-depth au niveau de l'agent.

S'AJOUTE au stack deepagents par défaut (planning, filesystem, subagents…).
Pour chaque appel d'outil SQL :
  - journalise la requête (traçabilité),
  - applique une garde lecture seule AVANT exécution (court-circuit l'outil et
    renvoie un message d'erreur que l'agent peut corriger).

La garde DB (`engine.assert_read_only`) reste la barrière de référence ; ce
middleware ajoute le log + un refus immédiat côté agent (réutilise la même règle,
donc pas de duplication de logique).
"""

from __future__ import annotations

import logging

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage

from ..database import engine

logger = logging.getLogger("lpo.sql_audit")

# Outils dont l'argument est du SQL à auditer
_SQL_TOOLS = {"run_sql_query", "check_sql"}


class SQLAuditMiddleware(AgentMiddleware):
    """Audit + garde lecture seule sur chaque appel d'outil SQL. Aucune I/O disque."""

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        name = request.tool_call.get("name", "")
        if name in _SQL_TOOLS:
            sql = (request.tool_call.get("args") or {}).get("sql", "") or ""
            logger.info("SQL[%s] %s", name, " ".join(sql.split())[:300])
            try:
                engine.assert_read_only(sql)
            except Exception as e:  # noqa: BLE001
                logger.warning("SQL refusé par l'audit: %s", e)
                return ToolMessage(
                    content=f"ERREUR (audit lecture seule): {e}",
                    tool_call_id=request.tool_call["id"],
                    name=name,
                )
        return handler(request)

    def before_model(self, state, runtime):
        logger.debug("tour modèle: %d message(s)", len(state.get("messages", [])))
        return None
