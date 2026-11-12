"""Crée les index recommandés sur la table exposée par le serveur MCP.

Les agrégations (`group_by=year|month|species|department|grid`) et les filtres
(`species`, `start_date`/`end_date`, `department`, `region`) portent sur les
mêmes colonnes. Sur une table de démonstration (12 000 lignes) l'index est
inutile ; sur une table réelle de plusieurs millions de lignes, il change tout.

Le script est **idempotent** et **non destructif** : il ne crée que des index
manquants, ne touche à aucune donnée, et n'exige qu'un droit DDL.

    python scripts/create_indexes.py --dry-run
    python scripts/create_indexes.py
    python scripts/create_indexes.py --db data/lpo_demo.db --profile observations
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lpo_data_mcp.profiles import PROFILES, SchemaProfile, detect_profile  # noqa: E402

#: motifs d'index, par ordre de priorité (les plus utiles d'abord)
INDEX_PATTERNS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("event_date", "idx_{table}_event_date", ("event_date",)),
    ("species", "idx_{table}_species", ("species",)),
    ("species_date", "idx_{table}_species_date", ("species", "event_date")),
    ("area", "idx_{table}_area", ("department", "region")),
)


def _column_for(profile: SchemaProfile, key: str) -> str | None:
    return {
        "event_date": profile.event_date,
        "species": profile.species_label or profile.species_key,
        "department": profile.department,
        "region": profile.region,
    }.get(key)


def plan_indexes(profile: SchemaProfile) -> list[tuple[str, tuple[str, ...]]]:
    """Liste (nom d'index, colonnes) pertinents pour ce profil."""
    plan: list[tuple[str, tuple[str, ...]]] = []
    for key, template, members in INDEX_PATTERNS:
        columns: list[str] = []
        for member in members:
            column = _column_for(profile, member)
            if column:
                columns.append(column)
        if not columns:
            continue
        # Un index redondant avec un index déjà prévu (même préfixe) est inutile.
        if any(tuple(columns) == existing for _, existing in plan):
            continue
        if key == "species" and any(
            len(existing) > 1 and existing[0] == columns[0] for _, existing in plan
        ):
            continue
        plan.append((template.format(table=profile.table), tuple(columns)))
    return plan


def create_indexes(profile: SchemaProfile, *, dry_run: bool = False) -> list[str]:
    """Applique le plan ; renvoie les instructions exécutées (ou simulées)."""
    from src.database import engine  # import tardif : configure le moteur

    existing = _existing_indexes(profile.table)
    statements: list[str] = []
    for name, columns in plan_indexes(profile):
        if name in existing:
            continue
        quoted = ", ".join(f'"{column}"' for column in columns)
        statements.append(f'CREATE INDEX IF NOT EXISTS "{name}" ON "{profile.table}" ({quoted})')
    if not dry_run:
        with engine.get_engine().begin() as connection:
            for statement in statements:
                connection.exec_driver_sql(statement)
    return statements


def _existing_indexes(table: str) -> set[str]:
    """Noms d'index déjà présents (SQLite et PostgreSQL)."""
    from sqlalchemy import inspect

    from src.database import engine

    try:
        inspector = inspect(engine.get_engine())
        return {index["name"] for index in inspector.get_indexes(table) if index.get("name")}
    except Exception:  # noqa: BLE001 - base non introspectionnable
        return set()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index recommandés pour le serveur MCP lpo-data.")
    parser.add_argument("--db", help="Chemin SQLite (sinon DATABASE_URL).")
    parser.add_argument("--profile", help="Profil de schéma (sinon LPO_PROFILE/auto).")
    parser.add_argument("--dry-run", action="store_true", help="Affiche le SQL sans l'exécuter.")
    args = parser.parse_args(argv)

    if args.db:
        os.environ["DATABASE_URL"] = f"sqlite:///{Path(args.db).resolve()}"
    if args.profile:
        os.environ["LPO_PROFILE"] = args.profile

    from src.config.settings import settings
    from src.database import engine

    # `settings` a pu être chargé avant la modification d'environnement.
    if args.db:
        settings.database.url = os.environ["DATABASE_URL"]
    engine.reset_engine()

    try:
        tables = engine.list_tables()
    except Exception as exc:  # noqa: BLE001
        print(f"Connexion impossible : {exc}", file=sys.stderr)
        return 2
    try:
        profile = PROFILES[args.profile] if args.profile else detect_profile(tables)
    except ValueError as exc:
        print(f"Profil indéterminé : {exc}", file=sys.stderr)
        return 2
    if profile.table not in tables:
        print(f"Table « {profile.table} » absente de la base.", file=sys.stderr)
        return 2

    statements = create_indexes(profile, dry_run=args.dry_run)
    if not statements:
        print(f"Aucun index à créer sur « {profile.table} » (déjà en place).")
        return 0
    for statement in statements:
        print(("-- " if args.dry_run else "") + statement)
    print(
        f"{len(statements)} index "
        + ("simulé(s)" if args.dry_run else "créé(s)")
        + f" sur « {profile.table} »."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
