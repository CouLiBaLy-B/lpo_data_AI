"""Schémas de sortie des outils MCP.

Chaque outil renvoie **un objet structuré** (donc un `structuredContent` MCP
typé) contenant systématiquement :

- ``summary`` : un résumé textuel en français, utilisable sans renderer A2UI ;
- les données (listes, séries, points) ;
- ``meta`` : profil utilisé, capacités du jeu de données, volumétrie,
  avertissements et niveau de confidentialité appliqué.

Ce triple contrat permet à l'agent de composer une surface A2UI **et** une
réponse texte de repli, sans jamais recalculer ni inventer de valeur.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


class ResultMeta(BaseModel):
    """Métadonnées communes à tous les résultats."""

    profile: str = Field(description="Profil de schéma utilisé (observations, sensitive_areas…).")
    table: str = Field(description="Table physique interrogée.")
    capabilities: list[str] = Field(
        default_factory=list,
        description="Capacités réellement disponibles (coordinates, individual_counts…).",
    )
    row_count: int = Field(default=0, description="Nombre de lignes renvoyées.")
    truncated: bool = Field(default=False, description="Vrai si un plafond a été atteint.")
    privacy: str = Field(default="auto", description="Mode de confidentialité appliqué.")
    warnings: list[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=_now)


class SpeciesInfo(BaseModel):
    """Espèce telle que la connaît le jeu de données."""

    id: str = Field(description="Identifiant stable à réutiliser dans les filtres.")
    label: str = Field(description="Libellé affichable.")
    latin_name: str | None = Field(default=None, description="Nom scientifique si disponible.")
    protection: str | None = Field(default=None, description="Statut de protection si disponible.")
    sensitive: bool = Field(default=False, description="Espèce à localisation sensible.")
    observation_count: int = Field(default=0, description="Nombre d'observations.")
    individual_count: float | None = Field(default=None, description="Somme des effectifs.")
    first_seen: str | None = Field(default=None, description="Première date observée.")
    last_seen: str | None = Field(default=None, description="Dernière date observée.")
    departments: list[str] = Field(default_factory=list, description="Départements concernés.")


class SpeciesListResult(BaseModel):
    summary: str
    query: str | None = None
    species: list[SpeciesInfo] = Field(default_factory=list)
    total: int = Field(default=0, description="Nombre total d'espèces correspondantes.")
    meta: ResultMeta


class Observation(BaseModel):
    """Une observation individuelle (coordonnées déjà dégradées si nécessaire)."""

    id: str | None = None
    #: NULL quand la ligne ne porte aucune espèce (zone purement réglementaire).
    #: La chaîne « None » serait une valeur affichée, donc une donnée inventée.
    species: str | None = None
    event_date: str | None = None
    count: float | None = None
    department: str | None = None
    region: str | None = None
    lat: float | None = Field(default=None, description="Latitude diffusée (voir precision).")
    lon: float | None = None
    protection: str | None = None
    sensitive: bool = False
    precision: str = Field(
        default="exact",
        description="exact | blurred | department — précision réellement diffusée.",
    )


class ObservationPage(BaseModel):
    summary: str
    observations: list[Observation] = Field(default_factory=list)
    total: int = Field(default=0, description="Nombre total de lignes correspondant aux filtres.")
    returned: int = 0
    offset: int = 0
    meta: ResultMeta


class SeriesPoint(BaseModel):
    """Point d'agrégat (période, espèce, département ou maille géographique)."""

    key: str = Field(description="Clé technique du point (année, mois, département…).")
    series: str | None = Field(
        default=None, description="Série d'appartenance (comparaison multi-espèces)."
    )
    label: str = Field(description="Libellé affichable.")
    value: float = Field(description="Valeur de la métrique demandée.")
    observations: int | None = Field(default=None, description="Nombre d'observations brutes.")
    individuals: float | None = Field(default=None, description="Somme des effectifs.")
    share: float | None = Field(default=None, description="Part du total (0-1).")
    lat: float | None = Field(default=None, description="Latitude de la maille (group_by=grid).")
    lon: float | None = None
    sensitive: bool = Field(default=False, description="Maille contenant une espèce sensible.")


class AggregateResult(BaseModel):
    summary: str
    group_by: str
    metric: str
    points: list[SeriesPoint] = Field(default_factory=list)
    total: float = Field(default=0, description="Total de la métrique sur le périmètre filtré.")
    total_observations: int = 0
    meta: ResultMeta


class DatasetOverview(BaseModel):
    summary: str
    profile: str
    table: str
    description: str
    dataset_label: str | None = Field(default=None, description="Libellé de provenance du jeu.")
    is_demo: bool = Field(
        default=False, description="Vrai pour un jeu de démonstration synthétique."
    )
    capabilities: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    total_rows: int
    #: Libellé métier d'une ligne (« observations », « enregistrements ») : le
    #: texte affiché ne doit pas appeler « observation » une zone réglementaire.
    records_label: str = "observations"
    date_min: str | None = None
    date_max: str | None = None
    species_count: int | None = None
    department_count: int | None = None
    departments: list[str] = Field(
        default_factory=list, description="Départements distincts connus."
    )
    regions: list[str] = Field(default_factory=list, description="Régions distinctes connues.")
    species_sample: list[str] = Field(
        default_factory=list, description="Quelques libellés d'espèces (découverte)."
    )
    notes: list[str] = Field(default_factory=list, description="Limites connues du jeu de données.")
    sample_species: list[str] = Field(default_factory=list)
    meta: ResultMeta


class SpeciesDetail(BaseModel):
    summary: str
    species: SpeciesInfo
    monthly: list[SeriesPoint] = Field(default_factory=list)
    top_departments: list[SeriesPoint] = Field(default_factory=list)
    meta: ResultMeta
