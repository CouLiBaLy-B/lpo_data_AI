"""
Agent SQL « deep » — assemblage deepagents.

L'agent reçoit une question en langage naturel et, en autonomie :
  1. découvre les tables (list_tables)
  2. lit le schéma des tables pertinentes (describe_tables)
  3. écrit une requête, la valide (check_sql), l'exécute (run_sql_query)
  4. s'auto-corrige en cas d'erreur, puis répond en français

Composants deepagents utilisés :
- system_prompt externalisé + skill `sql-lpo` versionnés git (progressive disclosure)
- SQLAuditMiddleware (audit + garde lecture seule au niveau agent)
- backend par défaut = StateBackend (en mémoire, zéro disque — adapté à Streamlit)
- exécution en STREAMING (stream_mode="updates") : un seul passage du graphe,
  les étapes sont émises au fil de l'eau pour l'UI.

Le schéma n'est jamais dans le prompt : l'agent l'explore. Le système est donc
indépendant du schéma et gère nativement les bases multi-tables (JOINs).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .middleware import SQLAuditMiddleware
from .model_factory import create_model
from .tools import SQL_TOOLS

logger = logging.getLogger(__name__)

# ── Prompt & skills versionnés (chargés au boot) ──────────────────────────────

_PROMPTS = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (_PROMPTS / "system_sql.md").read_text(encoding="utf-8")

# Skills (SKILL.md) fournis à l'agent via le state (StateBackend) à chaque appel.
# Contrainte deepagents 0.6.11 : le `name:` du frontmatter == nom du dossier parent.
SKILL_FILES: dict[str, dict[str, str]] = {}
_skills_root = _PROMPTS / "skills"
if _skills_root.is_dir():
    for _skill_dir in _skills_root.iterdir():
        _md = _skill_dir / "SKILL.md"
        if _md.is_file():
            SKILL_FILES[f"/skills/{_skill_dir.name}/SKILL.md"] = {
                "content": _md.read_text(encoding="utf-8")
            }


@dataclass
class AgentStep:
    tool: str
    args: dict[str, Any]
    result: str
    status: str  # "ok" | "error"


@dataclass
class AgentResult:
    answer: str
    sql: str | None = None
    steps: list[AgentStep] = field(default_factory=list)
    error: str | None = None


_agent = None


def get_agent():
    """Construit (paresseusement) et met en cache le graphe deepagents."""
    global _agent
    if _agent is None:
        from deepagents import create_deep_agent

        kwargs: dict[str, Any] = {
            "model": create_model(),
            "tools": SQL_TOOLS,
            "system_prompt": SYSTEM_PROMPT,
            "middleware": [SQLAuditMiddleware()],
        }
        if SKILL_FILES:
            kwargs["skills"] = ["/skills/"]
        _agent = create_deep_agent(**kwargs)
        logger.info("Agent deepagents construit (skills=%d).", len(SKILL_FILES))
    return _agent


def reset_agent() -> None:
    """Force la reconstruction (ex: après changement de provider)."""
    global _agent
    _agent = None


def _agent_input(question: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"messages": [{"role": "user", "content": question}]}
    if SKILL_FILES:
        payload["files"] = SKILL_FILES
    return payload


# ── Exécution en streaming (un seul passage de graphe) ────────────────────────

def stream_run(question: str, *, recursion_limit: int = 25) -> Iterator[AgentStep]:
    """
    Exécute l'agent en STREAMING : yield chaque AgentStep dès qu'elle se produit
    (pour un affichage live). Le résultat final complet (AgentResult) est renvoyé
    comme valeur de retour du générateur (récupérable via StopIteration.value, ou
    simplement via run() qui draine ce générateur).
    """
    steps: list[AgentStep] = []
    last_sql: str | None = None
    final_answer = "Analyse terminée."
    pending: dict[str, dict[str, Any]] = {}

    try:
        agent = get_agent()
        for chunk in agent.stream(
            _agent_input(question),
            config={"recursion_limit": recursion_limit},
            stream_mode="updates",
        ):
            for _node, update in chunk.items():
                # Des nœuds de middleware émettent des updates None : garde indispensable.
                if not isinstance(update, dict):
                    continue
                for msg in update.get("messages", []):
                    for tc in getattr(msg, "tool_calls", None) or []:
                        pending[tc["id"]] = tc

                    cls = msg.__class__.__name__
                    if cls == "ToolMessage":
                        tc = pending.get(getattr(msg, "tool_call_id", ""), {})
                        name = tc.get("name") or getattr(msg, "name", "outil")
                        content = str(getattr(msg, "content", ""))
                        step = AgentStep(
                            tool=name,
                            args=tc.get("args", {}) or {},
                            result=_summarize(content),
                            status="error" if _is_error(content) else "ok",
                        )
                        steps.append(step)
                        if name == "run_sql_query" and step.status == "ok" and "sql" in step.args:
                            last_sql = step.args["sql"]
                        yield step
                    elif cls == "AIMessage" and not getattr(msg, "tool_calls", None):
                        text = _final_text(msg)
                        if text:
                            final_answer = text
    except Exception as e:  # noqa: BLE001
        logger.error("Échec de l'agent: %s", e, exc_info=True)
        return AgentResult(answer="", error=str(e), steps=steps)

    return AgentResult(answer=final_answer, sql=last_sql, steps=steps)


def run(question: str, *, recursion_limit: int = 25) -> AgentResult:
    """Exécute l'agent et renvoie le résultat complet (draine le streaming)."""
    gen = stream_run(question, recursion_limit=recursion_limit)
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value or AgentResult(answer="Analyse terminée.")


# ── Helpers d'extraction ──────────────────────────────────────────────────────

def _is_error(content: str) -> bool:
    return content.startswith("ERREUR") or "INVALIDE" in content


def _final_text(msg: Any) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        content = " ".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return (content or "").strip()


def _summarize(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"
