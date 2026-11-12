"""Support PostgreSQL de production — vérifié sans instance disponible.

La production tourne sur PostgreSQL, le développement sur SQLite. Ces tests
couvrent ce qui casse silencieusement lors d'un passage en production et qui
n'est **pas** visible sur SQLite :

1. normalisation du DSN (les hébergeurs fournissent souvent ``postgres://``,
   que SQLAlchemy ne sait pas résoudre) ;
2. expurgation du mot de passe dans les journaux, y compris s'il contient
   ``@`` ou des caractères encodés ;
3. paramètres de connexion (``application_name``, ``search_path`` borné) ;
4. garde-fous de session (timeouts, fuseau, lecture seule) ;
5. DDL d'indexation : ``CONCURRENTLY`` hors transaction côté PostgreSQL,
   forme historique côté SQLite ;
6. déclaration d'un pilote unique dans les dépendances.

Un test d'intégration facultatif s'exécute si ``LPO_TEST_PG_URL`` est défini :

    LPO_TEST_PG_URL=postgresql://lpo_ro:secret@localhost:5432/lpo \\
        .venv/bin/python -m pytest tests/test_postgres_support.py -q
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from lpo_data_mcp.profiles import OBSERVATIONS, SENSITIVE_AREAS
from scripts.create_indexes import plan_indexes, plan_statements
from src.config.settings import DatabaseSettings
from src.database import postgres

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = (ROOT / "requirements.txt").read_text(encoding="utf-8")
DDL = (ROOT / "docs" / "sql" / "sensitive_areas_postgres.sql").read_text(encoding="utf-8")
REAL_EXTRACT = ROOT / "data" / "sensitive_areas.db"


# ── DSN ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("postgres://u:p@h:5432/db", "postgresql+psycopg://u:p@h:5432/db"),
        ("postgresql://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
        ("postgresql+psycopg://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
        # Un pilote explicitement choisi n'est jamais réécrit.
        ("postgresql+psycopg2://u:p@h/db", "postgresql+psycopg2://u:p@h/db"),
        ("sqlite:///data/lpo_demo.db", "sqlite:///data/lpo_demo.db"),
        ("  postgres://u@h/db  ", "postgresql+psycopg://u@h/db"),
        ("", ""),
    ],
)
def test_database_urls_are_normalised(given: str, expected: str) -> None:
    assert postgres.normalize_database_url(given) == expected


@pytest.mark.parametrize(
    "url",
    [
        "postgres://u:p@h/db",
        "postgresql://u:p@h:5432/db?sslmode=require",
        "postgresql+psycopg://u:p@h/db",
    ],
)
def test_postgres_urls_are_recognised(url: str) -> None:
    assert postgres.is_postgres_url(url)


def test_sqlite_is_not_mistaken_for_postgres() -> None:
    assert not postgres.is_postgres_url("sqlite:///data/lpo_demo.db")


def test_password_never_reaches_the_logs() -> None:
    """Un mot de passe peut contenir « @ » : un découpage naïf le laisserait fuir."""
    url = "postgresql://lpo:mot@depasse:secret@db.interne:5432/lpo"
    redacted = postgres.redact_url(url)
    assert "mot@depasse" not in redacted
    assert "secret" not in redacted
    assert "db.interne:5432/lpo" in redacted


def test_redaction_handles_encoded_passwords_and_garbage() -> None:
    assert "p%40ss" not in postgres.redact_url("postgresql://u:p%40ss@h/db")
    assert postgres.redact_url("pas une url") == "(URL de base invalide)"


# ── Paramètres de connexion ─────────────────────────────────────────────────


def test_connection_parameters_identify_the_application_and_bound_the_wait() -> None:
    args = postgres.connect_args(
        schema="lpo", application_name="lpo-data-mcp", connect_timeout_s=7
    )
    assert args["application_name"] == "lpo-data-mcp"
    assert args["connect_timeout"] == 7
    # `public` reste en second : les extensions (PostGIS) y sont installées.
    assert args["options"] == "-c search_path=lpo,public"


def test_no_search_path_is_imposed_without_a_schema() -> None:
    args = postgres.connect_args(schema="", application_name="lpo-data-ai")
    assert "options" not in args


@pytest.mark.parametrize(
    "schema",
    ["lpo; DROP SCHEMA public", "lpo,public", "lpo schema", "-lpo", "1lpo", ""],
)
def test_schema_names_are_validated(schema: str) -> None:
    """`search_path` ne peut pas être un paramètre lié : on refuse tout le reste."""
    if schema == "":
        assert postgres.validate_schema(schema) == ""
        return
    with pytest.raises(ValueError):
        postgres.validate_schema(schema)


def test_schema_comes_from_the_environment() -> None:
    settings = DatabaseSettings(DB_SCHEMA="lpo", DB_APPLICATION_NAME="lpo-data-mcp")
    assert settings.db_schema == "lpo"
    assert settings.application_name == "lpo-data-mcp"


# ── Garde-fous de session ───────────────────────────────────────────────────


def test_read_only_session_is_enforced_server_side() -> None:
    statements = postgres.session_setup_statements(
        read_only=True, statement_timeout_ms=15_000, lock_timeout_ms=3_000
    )
    joined = " ; ".join(statements)
    assert "SET default_transaction_read_only = on" in statements
    assert "SET statement_timeout = 15000" in joined
    assert "SET lock_timeout = 3000" in joined
    assert "idle_in_transaction_session_timeout" in joined
    # Les années affichées ne doivent pas dépendre du fuseau de l'hôte.
    assert "SET TIME ZONE 'UTC'" in statements


def test_write_sessions_explicitly_clear_read_only() -> None:
    """Un rôle dont le défaut serait `on` bloquerait sinon la création d'index."""
    statements = postgres.session_setup_statements(read_only=False)
    assert "SET default_transaction_read_only = off" in statements


@pytest.mark.parametrize("value", [-1, "15000", None, True])
def test_timeouts_reject_non_positive_and_non_integer_values(value: object) -> None:
    with pytest.raises(ValueError):
        postgres.session_setup_statements(read_only=True, statement_timeout_ms=value)  # type: ignore[arg-type]


def test_timezone_is_validated() -> None:
    with pytest.raises(ValueError):
        postgres.session_setup_statements(read_only=True, timezone="UTC'; DROP DATABASE x")


# ── DDL d'indexation ────────────────────────────────────────────────────────


def test_postgres_indexes_are_built_concurrently() -> None:
    """`CONCURRENTLY` : la table reste disponible pendant l'indexation."""
    statements = plan_statements(OBSERVATIONS, "postgresql")
    assert statements
    for statement in statements:
        assert statement.startswith("CREATE INDEX CONCURRENTLY IF NOT EXISTS")
        assert ' ON "observations" ' in statement


def test_postgres_ddl_can_target_a_dedicated_schema() -> None:
    statements = plan_statements(OBSERVATIONS, "postgresql", schema="lpo")
    assert all(' ON "lpo"."observations" ' in statement for statement in statements)


def test_sqlite_ddl_is_unchanged() -> None:
    """La base de développement ne connaît pas `CONCURRENTLY`."""
    statements = plan_statements(OBSERVATIONS, "sqlite")
    assert statements
    for statement in statements:
        assert statement.startswith("CREATE INDEX IF NOT EXISTS")
        assert "CONCURRENTLY" not in statement


# ── Schéma de production dérivé de l'extrait réel ───────────────────────────


@pytest.mark.skipif(not REAL_EXTRACT.exists(), reason="extrait data/sensitive_areas.db absent")
def test_postgres_ddl_covers_every_column_of_the_real_extract() -> None:
    """Le DDL de production ne doit pas oublier une colonne de l'extrait.

    Une colonne manquante ne se voit qu'au moment du chargement, en production.
    """
    with sqlite3.connect(REAL_EXTRACT) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(sensitive_areas)")]
    assert columns
    for column in columns:
        # `Pays` devient `pays` : PostgreSQL replie les identifiants non cités.
        expected = column.lower()
        assert expected in DDL.lower(), f"colonne absente du DDL : {column}"


def test_postgres_ddl_declares_the_indexes_the_script_creates() -> None:
    for name, _columns in plan_indexes(SENSITIVE_AREAS):
        assert name in DDL, f"index attendu par le script absent du DDL : {name}"


def test_postgres_ddl_indexes_are_created_concurrently() -> None:
    """Sur une table déjà lue par l'ETL, un index bloquant arrête le service."""
    index_lines = [
        line.strip()
        for line in DDL.splitlines()
        if line.strip().startswith("CREATE INDEX")
    ]
    assert index_lines
    assert all("CONCURRENTLY" in line for line in index_lines)


def test_postgres_ddl_never_declares_species_id_as_real() -> None:
    """`species_id` est un identifiant entier ; l'extrait SQLite le stocke en REAL."""
    assert "species_id      integer" in DDL
    assert "species_id real" not in DDL


# ── Dépendances ─────────────────────────────────────────────────────────────


def test_a_single_postgres_driver_is_declared() -> None:
    """Deux pilotes installés, et `postgresql://` choisirait psycopg2 par défaut."""
    assert "psycopg[binary]" in REQUIREMENTS
    assert "psycopg2" not in REQUIREMENTS


# ── Test d'intégration facultatif ───────────────────────────────────────────


@pytest.mark.skipif(
    not os.environ.get("LPO_TEST_PG_URL"),
    reason="définir LPO_TEST_PG_URL pour interroger une vraie base PostgreSQL",
)
def test_live_postgres_session_is_read_only_and_bounded() -> None:
    """Contre une vraie base : la session refuse l'écriture, le timeout est posé."""
    from sqlalchemy import text

    from src.config.settings import settings
    from src.database import engine

    original = settings.database.url
    settings.database.url = os.environ["LPO_TEST_PG_URL"]
    engine.reset_engine()
    try:
        with engine.get_engine().connect() as connection:
            read_only = connection.execute(text("SHOW transaction_read_only")).scalar()
            timeout = connection.execute(text("SHOW statement_timeout")).scalar()
            application_name = connection.execute(text("SHOW application_name")).scalar()
        assert read_only == "on"
        assert timeout == f"{settings.database.statement_timeout_ms}ms"
        assert application_name == settings.database.application_name
        with pytest.raises(Exception):
            with engine.get_engine().connect() as connection:
                connection.execute(text("CREATE TABLE _refus_attendu (id int)"))
    finally:
        settings.database.url = original
        engine.reset_engine()
