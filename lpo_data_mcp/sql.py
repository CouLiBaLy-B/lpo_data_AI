"""Construction de SQL paramétré et sûr pour le serveur MCP.

Principes (cf. README / docs) :

- **aucun SQL libre** exposé : l'appelant choisit des *énumérations* (group_by,
  metric, ordre) ; les valeurs passent par des paramètres liés ;
- **identifiants** (tables, colonnes) issus uniquement du profil de schéma, donc
  jamais d'une entrée utilisateur ; ils sont validés par expression régulière
  puis cités ;
- **liste blanche** pour tout ce qui finit dans le SQL structurel ;
- **plafonds** de lignes et pagination systématiques.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

GROUP_BY = (
    "year",
    "month",
    "department",
    "region",
    "species",
    "grid",
    "species_month",
    "species_year",
)
METRICS = ("observations", "individuals")


class QueryError(ValueError):
    """Paramètre de requête refusé (liste blanche, format, capacité absente)."""


def ident(name: str) -> str:
    """Valide puis cite un identifiant SQL (table ou colonne)."""
    if not _IDENT_RE.match(name):
        raise QueryError(f"Identifiant SQL refusé : {name!r}")
    return f'"{name}"'


def check_choice(value: str, allowed: tuple[str, ...], label: str) -> str:
    """Vérifie qu'une valeur appartient à une liste blanche."""
    normalized = (value or "").strip().lower()
    if normalized not in allowed:
        raise QueryError(f"{label}='{value}' invalide. Valeurs autorisées : {', '.join(allowed)}.")
    return normalized


@dataclass
class Where:
    """Accumulateur de clauses WHERE + paramètres liés."""

    clauses: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def add(self, clause: str, **params: Any) -> None:
        self.clauses.append(clause)
        self.params.update(params)

    def sql(self) -> str:
        return f" WHERE {' AND '.join(self.clauses)}" if self.clauses else ""


def parse_date(value: str | None, *, label: str) -> date | None:
    """Parse une date ISO (YYYY-MM-DD) et rejette tout le reste."""
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text[:10]).date()
    except ValueError as exc:
        raise QueryError(f"{label}='{value}' n'est pas une date ISO (attendu AAAA-MM-JJ).") from exc


def date_bounds(
    start: date | None, end: date | None, dialect: str, where: Where, column: str
) -> None:
    """Ajoute les bornes temporelles, avec une fin inclusive."""
    col = ident(column)
    if start is not None:
        where.add(
            f"{col} >= :start_date",
            start_date=_date_literal(start, dialect, end_of_day=False),
        )
    if end is not None:
        where.add(
            f"{col} <= :end_date",
            end_date=_date_literal(end, dialect, end_of_day=True),
        )


def _date_literal(value: date, dialect: str, *, end_of_day: bool) -> str:
    """Représentation comparable au stockage de la colonne selon le moteur."""
    suffix = " 23:59:59" if end_of_day else " 00:00:00"
    if dialect == "postgresql":
        return value.isoformat() + ("T23:59:59" if end_of_day else "T00:00:00")
    return value.isoformat() + suffix


def time_bucket(column: str, level: str, dialect: str) -> tuple[str, str]:
    """Expression de regroupement temporel et expression de libellé."""
    col = ident(column)
    level = check_choice(level, ("year", "month", "week", "day"), "niveau temporel")
    if dialect == "postgresql":
        fmt = {"year": "YYYY", "month": "YYYY-MM", "week": 'IYYY-"S"IW', "day": "YYYY-MM-DD"}[level]
        expr = f"to_char({col}, '{fmt}')"
    else:
        fmt = {"year": "%Y", "month": "%Y-%m", "week": "%Y-W%W", "day": "%Y-%m-%d"}[level]
        expr = f"strftime('{fmt}', {col})"
    return expr, expr


def round_expr(column: str, decimals: int, dialect: str) -> str:
    """Arrondi (floutage) d'une coordonnée, portable SQLite/PostgreSQL."""
    col = ident(column)
    if dialect == "postgresql":
        return f"ROUND(CAST({col} AS numeric), {int(decimals)})"
    return f"ROUND({col}, {int(decimals)})"


def lower_trim(column: str) -> str:
    """Normalisation de libellé pour les comparaisons insensibles à la casse."""
    return f"LOWER(TRIM({ident(column)}))"


def limit_offset(limit: int, offset: int, *, max_limit: int = 500) -> tuple[int, int]:
    """Borne la pagination pour protéger la base."""
    try:
        limit_i = int(limit)
        offset_i = int(offset)
    except (TypeError, ValueError) as exc:
        raise QueryError("limit et offset doivent être des entiers.") from exc
    if limit_i <= 0:
        raise QueryError("limit doit être strictement positif.")
    if offset_i < 0:
        raise QueryError("offset doit être positif ou nul.")
    return min(limit_i, max_limit), offset_i
