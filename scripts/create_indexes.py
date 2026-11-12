"""Crée les index recommandés sur la table exposée par le serveur MCP.

Les agrégations (`group_by=year|month|species|department|grid`) et les filtres
(`species`, `start_date`/`end_date`, `department`, `region`) portent sur les
mêmes colonnes. Sur une table de démonstration (12 000 lignes) l'index est
inutile ; sur une table réelle de plusieurs millions de lignes, il change tout.

Le script est **idempotent** et **non destructif** : il ne crée que des index
manquants, ne touche à aucune donnée, et n'exige qu'un droit DDL.

Sur PostgreSQL (production), la création passe par ``CREATE INDEX
CONCURRENTLY`` : la table reste lisible et écriturable pendant l'indexation,
au prix d'un verrou plus faible et d'une durée plus longue. Ces instructions ne
pouvant pas s'exécuter dans une transaction, la connexion est ouverte en
autocommit et ``lock_timeout`` borne l'attente. Un index laissé « invalide » par
une interruption est signalé en fin d'exécution (``pg_index.indisvalid``).

    python scripts/create_indexes.py --dry-run      # SQL seul, aucun verrou
    python scripts/create_indexes.py
    python scripts/create_indexes.py --db data/lpo_demo.db --profile observations
    DATABASE_URL=postgresql://... python scripts/create_indexes.py --schema lpo
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lpo_data_mcp.profiles import PROFILES, SchemaProfile, detect_profile  # noqa: E402
from src.database import postgres  # noqa: E402

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


def _qualified(table: str, schema: str) -> str:
    return f'"{schema}"."{table}"' if schema else f'"{table}"'


def plan_statements(profile: SchemaProfile, dialect: str, *, schema: str = "") -> list[str]:
    """Instructions DDL du plan, pour le moteur visé.

    PostgreSQL : ``CONCURRENTLY`` évite de bloquer les écritures de l'ETL
    pendant l'indexation ; SQLite ne connaît pas cette option et n'en a pas
    besoin (base de développement, accès mono-processus).
    """
    target = _qualified(profile.table, schema)
    concurrent = dialect == "postgresql"
    prefix = (
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS"
        if concurrent
        else "CREATE INDEX IF NOT EXISTS"
    )
    statements: list[str] = []
    for name, columns in plan_indexes(profile):
        quoted = ", ".join(f'"{column}"' for column in columns)
        statements.append(f'{prefix} "{name}" ON {target} ({quoted})')
    return statements


def create_indexes(
    profile: SchemaProfile,
    *,
    dry_run: bool = False,
    schema: str | None = None,
) -> list[str]:
    """Applique le plan ; renvoie les instructions exécutées (ou simulées)."""
    from src.config.settings import settings
    from src.database import engine  # import tardif : configure le moteur

    target_schema = settings.database.db_schema if schema is None else schema
    target_schema = postgres.validate_schema(target_schema)

    # Le moteur d'exploitation est construit **en écriture** : le serveur MCP,
    # lui, ouvre ses connexions en `default_transaction_read_only`.
    admin = engine.get_admin_engine()
    try:
        dialect = admin.dialect.name
        existing = _existing_indexes(admin, profile.table, target_schema)
        statements = [
            statement
            for statement in plan_statements(profile, dialect, schema=target_schema)
            if _index_name(statement) not in existing
        ]
        if dry_run or not statements:
            return statements
        if dialect == "postgresql":
            # `CONCURRENTLY` est interdit dans un bloc de transaction.
            with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
                connection.exec_driver_sql("SET lock_timeout = '5s'")
                for statement in statements:
                    connection.exec_driver_sql(statement)
        else:
            with admin.begin() as connection:
                for statement in statements:
                    connection.exec_driver_sql(statement)
        return statements
    finally:
        admin.dispose()


def _index_name(statement: str) -> str:
    """Nom d'index cité dans une instruction ``CREATE INDEX``."""
    match = re.search(r'INDEX(?: CONCURRENTLY)? IF NOT EXISTS "([^"]+)"', statement)
    return match.group(1) if match else ""


def _existing_indexes(admin, table: str, schema: str = "") -> set[str]:
    """Noms d'index déjà présents (SQLite et PostgreSQL)."""
    from sqlalchemy import inspect

    try:
        inspector = inspect(admin)
        return {
            index["name"]
            for index in inspector.get_indexes(table, schema=schema or None)
            if index.get("name")
        }
    except Exception:  # noqa: BLE001 - base non introspectionnable
        return set()


def invalid_indexes(admin, schema: str = "") -> list[str]:
    """Index laissés invalides par une création interrompue (PostgreSQL).

    Un `CREATE INDEX CONCURRENTLY` interrompu laisse un index que le planificateur
    ignore ; comme son nom existe, une relance le sauterait. Mieux vaut le dire
    que le laisser passer pour un index opérationnel.
    """
    from sqlalchemy import text

    if admin.dialect.name != "postgresql":
        return []
    query = text(
        """
        SELECT c.relname
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indexrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE NOT i.indisvalid AND n.nspname = COALESCE(NULLIF(:schema, ''), current_schema())
        ORDER BY c.relname
        """
    )
    with admin.connect() as connection:
        return [row[0] for row in connection.execute(query, {"schema": schema})]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index recommandés pour le serveur MCP lpo-data.")
    parser.add_argument("--db", help="Chemin SQLite (sinon DATABASE_URL).")
    parser.add_argument("--profile", help="Profil de schéma (sinon LPO_PROFILE/auto).")
    parser.add_argument(
        "--schema", help="Schéma PostgreSQL (sinon DB_SCHEMA). Sans effet sur SQLite."
    )
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

    statements = create_indexes(profile, dry_run=args.dry_run, schema=args.schema)
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
    if not args.dry_run:
        admin = engine.get_admin_engine()
        try:
            broken = invalid_indexes(admin, args.schema or settings.database.db_schema)
        finally:
            admin.dispose()
        if broken:
            print(
                "Index invalides détectés (création interrompue) : "
                + ", ".join(broken)
                + ". Les supprimer puis relancer : DROP INDEX CONCURRENTLY <nom>;",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
