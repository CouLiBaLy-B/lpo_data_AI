"""Point d'entrée CLI du serveur MCP `lpo-data`.

Exemples :

    python -m lpo_data_mcp                                  # stdio (agent local)
    python -m lpo_data_mcp --transport streamable-http      # réseau, port 8765
"""

from __future__ import annotations

import argparse
import logging
import os

from .server import mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="Serveur MCP lpo-data (lecture seule)")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default=os.getenv("LPO_MCP_TRANSPORT", "stdio"),
        help="Transport MCP (défaut : stdio).",
    )
    parser.add_argument("--host", default=os.getenv("LPO_MCP_HOST", "127.0.0.1"), help="Hôte HTTP.")
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("LPO_MCP_PORT", "8765")), help="Port HTTP."
    )
    parser.add_argument(
        "--log-level", default=os.getenv("LPO_LOG_LEVEL", "INFO"), help="Niveau de log."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.transport == "stdio":
        # stdio : la sortie standard est le canal du protocole, les logs vont sur stderr.
        mcp.run("stdio")
    else:
        mcp.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
