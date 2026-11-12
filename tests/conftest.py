"""Fixtures de test.

Deux bases sont utilisées :

- une base **de démonstration** générée à la volée (graine fixe) pour tester le
  profil ``observations`` complet (coordonnées, effectifs, protection) ;
- la base **du dépôt** (``data/sensitive_areas.db``) pour vérifier le profil
  dégradé ``sensitive_areas`` (ni coordonnées, ni effectifs, ni noms d'espèces).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.seed_demo_db import seed  # noqa: E402


@pytest.fixture(scope="session")
def demo_db_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("lpo_db") / "demo.db"
    seed(path, rows=900, seed=1234)
    return path


@pytest.fixture()
def use_db(monkeypatch):
    """Bascule la base utilisée par le moteur (settings + reset du cache)."""

    def _use(db_path: Path | str, profile: str = "observations"):
        from src.config.settings import settings
        from src.database import engine

        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
        monkeypatch.setenv("LPO_PROFILE", profile)
        settings.database.url = f"sqlite:///{db_path}"
        engine.reset_engine()

    yield _use
    from src.database import engine

    engine.reset_engine()


@pytest.fixture()
def demo_service(use_db, demo_db_path):
    use_db(demo_db_path, "observations")
    from lpo_data_mcp.service import LpoDataService

    return LpoDataService()


@pytest.fixture()
def repo_service(use_db):
    """Profil réel du dépôt : `sensitive_areas`."""
    db = ROOT / "data" / "sensitive_areas.db"
    if not db.exists():  # pragma: no cover
        pytest.skip("data/sensitive_areas.db absent")
    use_db(db, "sensitive_areas")
    from lpo_data_mcp.service import LpoDataService

    return LpoDataService()


class ServiceBackedClient:
    """Client MCP factice, branché en direct sur `LpoDataService`.

    Permet de tester l'orchestrateur sans sous-processus ni transport, tout en
    conservant exactement le contrat des outils (mêmes noms, mêmes arguments).
    """

    def __init__(self, service):
        self.service = service
        self.calls: list[tuple[str, dict]] = []

    @property
    def tool_names(self) -> list[str]:
        return [
            "get_dataset_overview",
            "list_species",
            "get_observations",
            "aggregate_observations",
            "get_species_info",
        ]

    # Correspondance outil MCP → méthode du service (l'aperçu est un outil du
    # serveur, pas du service : son nom diffère).
    METHODS = {
        "get_dataset_overview": "dataset_overview",
        "list_species": "list_species",
        "get_observations": "get_observations",
        "aggregate_observations": "aggregate_observations",
        "get_species_info": "get_species_info",
    }

    async def call(self, tool: str, arguments: dict | None = None) -> dict:
        arguments = dict(arguments or {})
        self.calls.append((tool, arguments))
        method = getattr(self.service, self.METHODS.get(tool, tool))
        result = method(**arguments)
        if hasattr(result, "model_dump"):
            return result.model_dump()
        return dict(result)


@pytest.fixture()
def service_client(demo_service):
    return ServiceBackedClient(demo_service)


@pytest.fixture()
def repo_client(repo_service):
    return ServiceBackedClient(repo_service)
