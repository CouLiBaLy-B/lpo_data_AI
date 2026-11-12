"""Génère une trace bout en bout : question → appels MCP → messages A2UI → texte.

Usage (depuis la racine du dépôt) :

    python -m lpo_a2ui.examples.build_transcript
    python -m lpo_a2ui.examples.build_transcript --out /tmp/trace.md

Le script interroge réellement le service de données (branché en direct sur le
serveur MCP par le même contrat d'outils que `orchestrator.mcp_client`) et
applique la même validation A2UI que la production. Il ne fabrique aucune
sortie : tout ce qui apparaît dans la trace sort du pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any

from orchestrator.agent import Conversation, LpoAssistant

HERE = Path(__file__).parent
DEFAULT_OUT = HERE / "transcript.md"
JSON_OUT = HERE / "turns.json"
MAX_LIST = 3

SCENARIO = [
    ("Évolution des observations de Milan royal sur les 10 dernières années ?", None),
    ("Top 10 des espèces observées dans le Finistère en 2025 ?", None),
    (
        "Interaction sur la surface de classement : filtre « zone » → Gironde",
        {"name": "set_filter", "context": {"filter": "area", "value": "Gironde"}},
    ),
    (
        "Interaction sur la même surface : filtre « période » → 12 derniers mois",
        {"name": "set_filter", "context": {"filter": "period", "value": "12m"}},
    ),
    ("Carte des observations de Busard cendré ce printemps ?", None),
    ("Compare Milan royal et Busard cendré par mois.", None),
    ("Fiche espèce du Héron cendré", None),
    ("Quel temps fera-t-il demain ?", None),
]


class ServiceBackedClient:
    """Client MCP en cours de processus : mêmes noms d'outils, mêmes sorties."""

    METHODS = {
        "get_dataset_overview": "dataset_overview",
        "list_species": "list_species",
        "get_observations": "get_observations",
        "aggregate_observations": "aggregate_observations",
        "get_species_info": "get_species_info",
    }

    def __init__(self, service):
        self.service = service
        self.tool_names = list(self.METHODS)

    async def close(self):  # parité avec McpDataClient
        return None

    async def call(self, tool: str, arguments: dict | None = None) -> dict:
        result = getattr(self.service, self.METHODS.get(tool, tool))(**(arguments or {}))
        return result.model_dump() if hasattr(result, "model_dump") else dict(result)


# L'horodatage de calcul et les durées changent à chaque exécution : les figer
# rend la trace reproductible (une régénération sans changement de données ne
# produit aucun diff).
_STAMP = re.compile(r"calculé le [0-9T:+\-]+")


def _stable(value: Any) -> Any:
    if isinstance(value, str):
        return _STAMP.sub("calculé le <date de génération>", value)
    return value


def _sanitise(value: Any) -> Any:
    """Applique `_stable` à toute la structure (dicts, listes, chaînes)."""
    if isinstance(value, dict):
        return {key: _sanitise(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitise(item) for item in value]
    return _stable(value)


def _duration(ms: int) -> str:
    """Durée en tranches : stable d'une exécution à l'autre, informative."""
    if ms < 100:
        return "< 100 ms"
    if ms < 1000:
        return "< 1 s"
    return "≥ 1 s"


def _trim(value: Any, depth: int = 0) -> Any:
    """Tronque les longues listes de données pour garder la trace lisible."""
    if isinstance(value, list):
        head = [_trim(item, depth + 1) for item in value[:MAX_LIST]]
        if len(value) > MAX_LIST:
            head.append(f"… ({len(value) - MAX_LIST} élément(s) supplémentaire(s))")
        return head
    if isinstance(value, dict):
        return {key: _trim(item, depth + 1) for key, item in value.items()}
    return _stable(value)


def _format_trace(trace) -> list[str]:
    lines = []
    for entry in trace:
        status = "ok" if entry.ok else f"échec ({entry.error})"
        args = json.dumps(entry.arguments, ensure_ascii=False, sort_keys=True)
        summary = (entry.summary or "").replace("\n", " ").strip()
        if len(summary) > 120:
            summary = summary[:117] + "…"
        lines.append(
            f"| `{entry.tool}` | `{args}` | {status} | {_duration(entry.duration_ms)} | {summary} |"
        )
    return lines


def _format_messages(messages: list[dict]) -> str:
    blocks = []
    for message in messages:
        key = next(k for k in message if k != "version")
        payload = message[key]
        payload = _trim(payload)
        document = {"version": message["version"], key: payload}
        rendered = json.dumps(document, ensure_ascii=False, indent=2)
        blocks.append(f"```json\n{rendered}\n```")
    return "\n\n".join(blocks)


def build(out: Path) -> Path:
    database_url = os.getenv("DATABASE_URL", "sqlite:///data/lpo_demo.db")
    os.environ.setdefault("LPO_PROFILE", "observations")
    os.environ["DATABASE_URL"] = database_url

    from lpo_data_mcp.service import LpoDataService

    display_url = database_url
    root = Path.cwd()
    for prefix in (f"sqlite:///{root}/", "sqlite:///"):
        if display_url.startswith(prefix):
            display_url = display_url[len(prefix) :]
            break

    assistant = LpoAssistant(
        ServiceBackedClient(LpoDataService()), today=date(2026, 11, 12), use_llm_intent=False
    )
    conversation = Conversation(id="exemple-bout-en-bout")

    turns: list[dict[str, Any]] = []
    parts: list[str] = [
        "# Exemple bout en bout — question → MCP → A2UI → interface",
        "",
        "> **Trace générée automatiquement** par `python -m lpo_a2ui.examples.build_transcript`.",
        "> Les messages A2UI sont ceux réellement validés par "
        "`lpo_a2ui.validation.validate_messages`.",
        "",
        f"- Base : `{display_url}` (profil `{os.getenv('LPO_PROFILE', 'observations')}`)",
        "- Date de référence : 2026-11-12",
        "- Intention : heuristique déterministe (`LPO_LLM_INTENT=off`)",
        "- Renderer : `web/static/renderer.js` (catalogue LPO v1)",
        "",
        "Chaque tour présente : la question, les appels MCP réellement effectués,",
        "les messages A2UI émis, le résumé texte de repli, puis l'effet produit.",
        "",
    ]

    for index, (question, action) in enumerate(SCENARIO, start=1):
        parts.append(f"## Tour {index} — {question}")
        parts.append("")
        if action is None:
            turn = asyncio.run(assistant.ask(question, conversation))
            parts.append(f"**Question** : « {question} »")
        else:
            turn = asyncio.run(assistant.act(action, conversation))
            parts.append(f"**Action A2UI** : `{json.dumps(action, ensure_ascii=False)}`")
        parts.append("")

        parts.append("### 1. Appels MCP")
        parts.append("")
        if turn.trace:
            parts.append("| Outil | Arguments | État | Durée | Résumé renvoyé |")
            parts.append("| --- | --- | --- | --- | --- |")
            parts.extend(_format_trace(turn.trace))
        else:
            parts.append("_Aucun appel (question hors périmètre)._")
        parts.append("")

        parts.append("### 2. Messages A2UI (`agent → renderer`)")
        parts.append("")
        if turn.messages:
            parts.append(_format_messages(turn.messages))
        else:
            parts.append(f"_Aucun message : interface non produite ({turn.validation})._")
        parts.append("")

        parts.append("### 3. Résumé texte de repli")
        parts.append("")
        parts.append("```text")
        parts.append(turn.text.strip())
        parts.append("```")
        parts.append("")

        parts.append("### 4. Effet côté client")
        parts.append("")
        parts.append(
            f"- type de surface : `{turn.kind}` — mode : `{turn.mode}` — "
            f"surface : `{turn.surface_id or '—'}`"
        )
        parts.append(f"- validation : `{turn.validation}`")
        if turn.messages:
            components = [
                c["component"]
                for m in turn.messages
                for c in (m.get("createSurface", {}).get("components") or [])
            ]
            if components:
                parts.append(f"- composants instanciés : {', '.join(components)}")
            data_paths = [
                m["updateDataModel"]["path"] for m in turn.messages if "updateDataModel" in m
            ]
            if data_paths:
                parts.append(f"- data model mis à jour : {', '.join(f'`{p}`' for p in data_paths)}")
        parts.append("")

        turns.append(
            {
                "index": index,
                "question": question,
                "action": action,
                "kind": turn.kind,
                "mode": turn.mode,
                "surfaceId": turn.surface_id,
                "text": turn.text,
                "messages": turn.messages,
            }
        )

    out.write_text("\n".join(parts), encoding="utf-8")
    JSON_OUT.write_text(
        json.dumps(_sanitise({"turns": turns}), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    path = build(args.out)
    print(f"Trace écrite : {path} ({path.stat().st_size} octets)")


if __name__ == "__main__":
    main()
