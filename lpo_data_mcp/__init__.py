"""Serveur MCP de données naturalistes LPO (lecture seule).

Point d'entrée : ``python -m lpo_data_mcp``. Voir ``docs/01-architecture.md``.
"""

from .server import mcp

__all__ = ["mcp"]
__version__ = "0.1.0"
