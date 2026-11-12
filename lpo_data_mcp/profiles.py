"""Profils de schéma : comment les concepts métier se projettent sur la base.

Le dépôt ne contient qu'**une** table métier (`sensitive_areas`, issue de l'API
biodiv-sports) alors que la cible fonctionnelle parle d'« observations »
naturalistes. Plutôt que d'inventer un schéma, on décrit la projection :

- ``observations``      → schéma cible (espèce, date, coordonnées, effectif…)
- ``sensitive_areas``   → schéma réellement présent dans le dépôt

Chaque profil déclare les colonnes physiques des concepts logiques et les
**capacités** réellement disponibles (coordonnées, effectifs, statut de
protection…). L'orchestrateur s'en sert pour choisir une visualisation honnête
et pour afficher un état vide explicite au lieu d'un graphique trompeur.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

Capability = Literal[
    "temporal",
    "species",
    "species_names",
    "individual_counts",
    "coordinates",
    "department",
    "region",
    "protection",
    "sensitivity",
    "structure",
]


@dataclass(frozen=True)
class SchemaProfile:
    """Projection des concepts logiques sur les colonnes d'une table."""

    name: str
    table: str
    description: str
    #: colonnes physiques (None = concept absent du schéma)
    species_key: str | None = None
    species_label: str | None = None
    event_date: str | None = None
    count: str | None = None
    lat: str | None = None
    lon: str | None = None
    department: str | None = None
    region: str | None = None
    protection: str | None = None
    sensitive: str | None = None
    structure: str | None = None
    id_column: str | None = None
    #: valeurs considérées comme « protégée » (insensible à la casse)
    protected_values: tuple[str, ...] = ("protégée", "protegee", "protégé", "sensible")
    #: note affichée dans les surfaces (limites connues du jeu de données)
    notes: list[str] = field(default_factory=list)

    @property
    def capabilities(self) -> set[Capability]:
        caps: set[Capability] = set()
        if self.event_date:
            caps.add("temporal")
        if self.species_key or self.species_label:
            caps.add("species")
        if self.species_label:
            caps.add("species_names")
        if self.count:
            caps.add("individual_counts")
        if self.lat and self.lon:
            caps.add("coordinates")
        if self.department:
            caps.add("department")
        if self.region:
            caps.add("region")
        if self.protection:
            caps.add("protection")
        if self.sensitive:
            caps.add("sensitivity")
        if self.structure:
            caps.add("structure")
        return caps

    @property
    def species_expr(self) -> str | None:
        """Expression SQL du libellé d'espèce (fallback : identifiant lisible)."""
        if self.species_label:
            return self.species_label
        if self.species_key:
            # `species_id` peut être NULL (zone purement réglementaire) : sans ce
            # CASE, la concaténation renvoie NULL et l'espèce devient « nan ».
            # CAST(CAST(...)) : `species_id` est stocké en REAL (10.0 → « 10 »).
            return (
                f"CASE WHEN {self.species_key} IS NULL THEN 'Zone réglementaire' "
                f"ELSE 'Espèce #' || CAST(CAST({self.species_key} AS INTEGER) AS TEXT) END"
            )
        return None


OBSERVATIONS = SchemaProfile(
    name="observations",
    table=os.getenv("LPO_OBS_TABLE", "observations"),
    description=(
        "Table d'observations naturalistes (schéma cible) : une ligne = une "
        "observation datée et géolocalisée."
    ),
    species_key="species",
    species_label="species",
    event_date="event_date",
    count="count",
    lat="lat",
    lon="lon",
    department="department",
    region="region",
    protection="protection_status",
    sensitive="sensitive",
    structure="source",
    id_column="id",
    notes=[],
)

SENSITIVE_AREAS = SchemaProfile(
    name="sensitive_areas",
    table="sensitive_areas",
    description=(
        "Table présente dans le dépôt (API biodiv-sports) : zones sensibles "
        "réglementaires ou liées à une espèce. Ni coordonnées, ni effectifs, ni "
        "noms d'espèces : seuls l'identifiant d'espèce, la date de création, la "
        "région et le département sont disponibles."
    ),
    species_key="species_id",
    species_label=None,
    event_date="create_datetime",
    count=None,
    lat=None,
    lon=None,
    department="department",
    region="region",
    protection=None,
    sensitive=None,
    structure="structure",
    id_column="id",
    notes=[
        "Le schéma réel ne contient pas de coordonnées : la carte n'est pas disponible.",
        "Les espèces ne sont connues que par leur identifiant numérique (pas de nom).",
        "Aucun effectif : la seule métrique est le nombre d'enregistrements.",
        "Les dates correspondent à la date de création de la zone, pas à une observation.",
    ],
)

PROFILES: dict[str, SchemaProfile] = {
    "observations": OBSERVATIONS,
    "sensitive_areas": SENSITIVE_AREAS,
}


def detect_profile(tables: list[str]) -> SchemaProfile:
    """Choisit un profil d'après les tables réellement présentes."""
    forced = os.getenv("LPO_PROFILE", "auto").strip().lower()
    if forced and forced != "auto":
        if forced not in PROFILES:
            raise ValueError(
                f"LPO_PROFILE='{forced}' inconnu. Valeurs : auto, " + ", ".join(sorted(PROFILES))
            )
        return PROFILES[forced]

    if OBSERVATIONS.table in tables:
        return OBSERVATIONS
    if SENSITIVE_AREAS.table in tables:
        return SENSITIVE_AREAS
    raise ValueError(
        "Aucune table exploitable : attendu "
        f"'{OBSERVATIONS.table}' (observations) ou '{SENSITIVE_AREAS.table}'. "
        f"Tables trouvées : {', '.join(tables) or 'aucune'}."
    )
