"""Serveur MCP `lpo-data` — outils typés, en lecture seule.

Transport : `stdio` par défaut (recommandé pour un agent local), `streamable-http`
disponible pour un déploiement réseau. Aucun outil n'accepte de SQL : l'appelant
choisit des énumérations (`group_by`, `metric`) et des filtres typés ; les
valeurs passent par des paramètres liés, les identifiants viennent du profil de
schéma, et chaque outil applique un plafond de lignes + une pagination.

Lancer :

    python -m lpo_data_mcp                    # stdio
    python -m lpo_data_mcp --transport streamable-http --port 8765

Chaque outil renvoie un objet Pydantic qui contient un `summary` (résumé texte
en français), les données et des `meta` (profil, capacités, avertissements,
niveau de confidentialité).
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .models import (
    AggregateResult,
    DatasetOverview,
    ObservationPage,
    SpeciesDetail,
    SpeciesListResult,
)
from .service import get_service
from .sql import QueryError

logger = logging.getLogger("lpo_mcp.server")

INSTRUCTIONS = """\
Serveur de données naturalistes LPO (lecture seule).

Règles d'usage :
1. Appeler `get_dataset_overview` en premier : il indique le profil de schéma
   réellement disponible et ses capacités (coordonnées, effectifs, noms
   d'espèces, statut de protection).
2. Ne jamais inventer d'espèce : utiliser `list_species` pour obtenir les
   libellés exacts avant de filtrer.
3. `aggregate_observations` couvre tendances (group_by=year|month), classements
   (group_by=species|department|region) et cartes (group_by=grid).
4. Les coordonnées d'espèces sensibles sont floutées côté serveur ; ne pas
   tenter de les reconstituer.
5. Les résultats vides ou partiels sont explicités dans `meta.warnings` : les
   relayer à l'utilisateur plutôt que de combler les trous.
"""


def _guarded(fn):
    """Transforme une erreur prévisible en `ToolError` explicite pour le client.

    Sans cela, le SDK masque le message (« Error executing tool … ») et l'agent
    ne peut pas expliquer l'échec à l'utilisateur.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except (QueryError, ValueError) as exc:
            raise ToolError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Échec de l'outil %s", fn.__name__)
            raise ToolError(f"Erreur d'exécution ({fn.__name__}): {exc}") from exc

    return wrapper


mcp: MCPServer = MCPServer(
    name="lpo-data",
    title="LPO Data (observations naturalistes, lecture seule)",
    instructions=INSTRUCTIONS,
    version="0.1.0",
)


@mcp.tool(
    name="get_dataset_overview",
    description=(
        "Décrit le jeu de données exposé : profil de schéma, table, colonnes, "
        "volumétrie, étendue temporelle, capacités (coordinates, individual_counts, "
        "species_names, protection…) et limites connues. À appeler en premier."
    ),
)
@_guarded
def get_dataset_overview() -> DatasetOverview:
    return get_service().dataset_overview()


@mcp.tool(
    name="list_species",
    description=(
        "Liste les espèces présentes, avec nombre d'observations, effectifs, statut "
        "de protection et indicateur de sensibilité. `query` filtre sur le libellé "
        "(sous-chaîne, insensible à la casse). Pagination via limit/offset (max 500)."
    ),
)
@_guarded
def list_species(query: str | None = None, limit: int = 25, offset: int = 0) -> SpeciesListResult:
    return get_service().list_species(query=query, limit=limit, offset=offset)


@mcp.tool(
    name="get_observations",
    description=(
        "Renvoie des observations détaillées (date, espèce, effectif, département, "
        "région, coordonnées déjà floutées si nécessaire). Filtres : `species` "
        "(liste de libellés exacts), `start_date`/`end_date` (ISO AAAA-MM-JJ), "
        "`department`, `region`. Pagination limit/offset (max 500). `privacy_mode` : "
        "auto | exact | blurred | department."
    ),
)
@_guarded
def get_observations(
    species: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    department: str | None = None,
    region: str | None = None,
    limit: int = 50,
    offset: int = 0,
    privacy_mode: str = "auto",
    include_sensitive: bool = True,
) -> ObservationPage:
    return get_service().get_observations(
        species=species,
        start_date=start_date,
        end_date=end_date,
        department=department,
        region=region,
        limit=limit,
        offset=offset,
        privacy_mode=privacy_mode,
        include_sensitive=include_sensitive,
    )


@mcp.tool(
    name="aggregate_observations",
    description=(
        "Agrège les observations. `group_by` : year | month (tendances), "
        "species | department | region (classements), grid (carte : mailles "
        "géographiques arrondies). `metric` : observations (nombre de lignes) ou "
        "individuals (somme des effectifs, si disponible). Mêmes filtres que "
        "get_observations. Renvoie une série triée, le total, et la part de chaque "
        "point (`share`)."
    ),
)
@_guarded
def aggregate_observations(
    group_by: str = "year",
    metric: str = "observations",
    species: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    department: str | None = None,
    region: str | None = None,
    limit: int = 50,
    privacy_mode: str = "auto",
) -> AggregateResult:
    return get_service().aggregate_observations(
        group_by=group_by,
        metric=metric,
        species=species,
        start_date=start_date,
        end_date=end_date,
        department=department,
        region=region,
        limit=limit,
        privacy_mode=privacy_mode,
    )


@mcp.tool(
    name="get_species_info",
    description=(
        "Fiche d'une espèce : identité, statut de protection, sensibilité, "
        "volumétrie, première/dernière observation, départements concernés, série "
        "mensuelle et top départements. `species` doit être un libellé exact issu de "
        "list_species (ou un identifiant numérique)."
    ),
)
@_guarded
def get_species_info(species: str) -> SpeciesDetail:
    return get_service().get_species_info(species)


def tool_catalog() -> list[dict[str, Any]]:
    """Liste statique des outils (documentation / tests, sans transport MCP)."""
    return [
        {"name": "get_dataset_overview", "read_only": True},
        {"name": "list_species", "read_only": True},
        {"name": "get_observations", "read_only": True},
        {"name": "aggregate_observations", "read_only": True},
        {"name": "get_species_info", "read_only": True},
    ]
