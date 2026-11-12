"""Agent orchestrateur en ligne de commande (sans renderer).

Sert à vérifier la chaîne « question → MCP → messages A2UI → texte » sans
navigateur. Les messages A2UI sont affichés sous forme de résumé ; le rendu
graphique est fourni par `web/` (client A2UI) ou par
`lpo_a2ui.examples.render_snapshot`.

    python -m orchestrator                          # REPL interactif
    python -m orchestrator "Top 10 des espèces en Gironde en 2025"
    python -m orchestrator --json "…"               # réponse brute (debug)
    python -m orchestrator --demo                   # scénario complet
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

from .agent import AssistantTurn, Conversation, LpoAssistant
from .mcp_client import McpDataClient, McpToolError

DEMO = [
    "Évolution des observations de Milan royal sur les 10 dernières années ?",
    "Top 10 des espèces observées dans le Finistère en 2025 ?",
    "Carte des observations de Busard cendré ce printemps ?",
    "Compare Milan royal et Busard cendré par mois.",
    "Fiche espèce du Héron cendré",
    "Quel temps fera-t-il demain ?",
]


def _summary(turn: AssistantTurn) -> str:
    lines = [turn.text.strip(), ""]
    if turn.messages:
        described = ", ".join(
            f"{next(k for k in message if k != 'version')}" for message in turn.messages
        )
        lines.append(
            f"[A2UI] kind={turn.kind} mode={turn.mode} surface={turn.surface_id} "
            f"messages={len(turn.messages)} ({described})"
        )
    else:
        lines.append(f"[A2UI] aucune surface ({turn.kind})")
    if turn.trace:
        calls = ", ".join(
            f"{entry.tool}({entry.duration_ms} ms{' ✗' if not entry.ok else ''})"
            for entry in turn.trace
        )
        lines.append(f"[MCP] {calls}")
    lines.append(f"[validation] {turn.validation}")
    return "\n".join(lines)


async def _run(questions: list[str], *, as_json: bool) -> int:
    client = McpDataClient()
    await client.connect()
    try:
        assistant = LpoAssistant(client)
        conversation = Conversation()
        for question in questions:
            turn = await assistant.ask(question, conversation)
            if as_json:
                payload: dict[str, Any] = turn.as_dict()
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print(f"\n❯ {question}\n")
                print(_summary(turn))
        return 0
    finally:
        await client.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent LPO — question → MCP → A2UI")
    parser.add_argument("question", nargs="*", help="question à poser (sinon REPL)")
    parser.add_argument("--json", action="store_true", help="affiche la réponse brute")
    parser.add_argument("--demo", action="store_true", help="joue le scénario de démonstration")
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="affiche le prompt système et les few-shots A2UI, puis quitte",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")

    if args.prompt:
        from .prompts import few_shot_prompt

        print(few_shot_prompt())
        return 0

    try:
        if args.demo:
            return asyncio.run(_run(DEMO, as_json=args.json))
        if args.question:
            return asyncio.run(_run([" ".join(args.question)], as_json=args.json))
        return _repl()
    except McpToolError as exc:
        print(f"Serveur MCP indisponible : {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


def _repl() -> int:
    from .agent import Conversation, LpoAssistant

    async def loop() -> int:
        client = McpDataClient()
        await client.connect()
        try:
            assistant = LpoAssistant(client)
            conversation = Conversation()
            print("Agent LPO — tapez votre question, « /reset » ou « /quit ».")
            while True:
                try:
                    question = input("\n❯ ").strip()
                except EOFError:
                    return 0
                if not question:
                    continue
                if question in {"/quit", "/exit"}:
                    return 0
                if question == "/reset":
                    conversation = Conversation()
                    print("Conversation réinitialisée.")
                    continue
                turn = await assistant.ask(question, conversation)
                print(_summary(turn))
        finally:
            await client.close()

    return asyncio.run(loop())


if __name__ == "__main__":
    raise SystemExit(main())
