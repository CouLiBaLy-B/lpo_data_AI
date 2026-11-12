"""
Couche d'accès base de données — agnostique au moteur et au schéma.

- Un seul moteur SQLAlchemy piloté par DATABASE_URL (SQLite en dev, Postgres en prod).
- Introspection dynamique : les tables et colonnes sont DÉCOUVERTES à l'exécution
  (jamais codées en dur). Le système fonctionne donc sur n'importe quelle base,
  mono ou multi-tables.
- LECTURE SEULE : seules les requêtes SELECT/WITH sont autorisées.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Any

import pandas as pd
import sqlparse
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine

from ..config.settings import settings
from ..core.exceptions import DatabaseException
from . import postgres

logger = logging.getLogger(__name__)

_engine_lock = threading.Lock()
_engine: Engine | None = None

# Mots-clés interdits en lecture seule (toute mutation / DDL)
_FORBIDDEN = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "REPLACE", "MERGE", "GRANT", "REVOKE", "ATTACH", "DETACH", "PRAGMA",
    "VACUUM", "REINDEX", "EXEC", "EXECUTE", "CALL",
}


def get_engine() -> Engine:
    """Retourne le moteur SQLAlchemy partagé (créé paresseusement)."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = _build_engine()
    return _engine


def _build_engine(*, read_only: bool | None = None) -> Engine:
    """Construit un moteur pour ``DATABASE_URL``.

    ``read_only`` force le mode de session (les scripts d'exploitation, qui
    créent des index, passent ``False``) ; par défaut on suit la configuration.
    """
    db = settings.database
    url = postgres.normalize_database_url(db.url)
    effective_read_only = db.read_only if read_only is None else read_only
    connect_args: dict[str, Any] = {}
    engine_kwargs: dict[str, Any] = {
        "pool_pre_ping": True,
        "future": True,
        "pool_timeout": 30,
    }

    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    else:
        # Tout ce qui n'est pas SQLite est traité comme un serveur distant :
        # pool borné et recyclage, quel que soit le moteur.
        engine_kwargs.update(
            pool_size=db.pool_size,
            max_overflow=db.max_overflow,
            pool_recycle=1800,
        )
        if postgres.is_postgres_url(url):
            # Paramètres libpq propres à PostgreSQL (`application_name`,
            # `search_path`, délai de connexion).
            connect_args.update(
                postgres.connect_args(
                    schema=db.db_schema,
                    application_name=db.application_name,
                    connect_timeout_s=db.connect_timeout_s,
                )
            )

    try:
        engine = create_engine(url, connect_args=connect_args, **engine_kwargs)
    except Exception as e:  # noqa: BLE001
        raise DatabaseException(_connection_error(e))

    if engine.dialect.name == "postgresql":
        _install_session_guardrails(engine, read_only=effective_read_only)

    try:
        # Test de connexion immédiat : on échoue tôt, et le message porte la
        # cause réelle (rôle absent, mot de passe, TLS) plutôt qu'une erreur
        # surgie trois couches plus loin.
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(
            "Moteur DB initialisé: %s (lecture seule: %s)",
            postgres.redact_url(url),
            effective_read_only,
        )
        return engine
    except Exception as e:  # noqa: BLE001
        engine.dispose()
        raise DatabaseException(_connection_error(e))


def _install_session_guardrails(engine: Engine, *, read_only: bool) -> None:
    """Applique les ``SET`` de session à chaque connexion PostgreSQL.

    L'exécution passe par le curseur du pilote, avant que SQLAlchemy n'ouvre
    une transaction : ``default_transaction_read_only`` ne s'applique qu'aux
    transactions **suivantes**, il doit donc être posé hors transaction.
    """
    db = settings.database
    statements = postgres.session_setup_statements(
        read_only=read_only,
        statement_timeout_ms=db.statement_timeout_ms,
        lock_timeout_ms=db.lock_timeout_ms,
        idle_in_transaction_timeout_ms=db.idle_in_transaction_timeout_ms,
    )

    @event.listens_for(engine, "connect")
    def _apply(dbapi_connection: Any, _record: Any) -> None:  # noqa: ANN401
        cursor = dbapi_connection.cursor()
        try:
            for statement in statements:
                cursor.execute(statement)
            dbapi_connection.commit()
        finally:
            cursor.close()


def _connection_error(error: Exception) -> str:
    """Message de connexion actionnable, sans divulguer le mot de passe."""
    message = str(error).strip() or error.__class__.__name__
    if "psycopg" in message.lower() and "import" in message.lower():
        return (
            "Pilote PostgreSQL absent : installez `psycopg[binary]` "
            f"(pip install -r requirements.txt). Détail : {message}"
        )
    return f"Connexion base de données impossible: {message}"


def get_admin_engine() -> Engine:
    """Moteur **en écriture**, non mis en cache : réservé aux scripts d'exploitation.

    Seul ``scripts/create_indexes.py`` l'utilise (DDL). Il n'est jamais importé
    par le serveur MCP ni par l'interface web, qui restent en lecture seule.
    """
    return _build_engine(read_only=False)


def reset_engine() -> None:
    """Réinitialise le moteur (utile après changement de DATABASE_URL)."""
    global _engine
    with _engine_lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        get_schema.cache_clear()
        list_tables.cache_clear()


# ── Introspection dynamique ───────────────────────────────────────────────────

@lru_cache(maxsize=1)
def list_tables() -> list[str]:
    """Liste toutes les tables ET vues de la base connectée."""
    insp = inspect(get_engine())
    names = list(insp.get_table_names()) + list(insp.get_view_names())
    return sorted(set(names))


@lru_cache(maxsize=64)
def get_schema(table: str) -> dict[str, Any]:
    """
    Retourne le schéma d'une table : colonnes, types, clés primaires,
    clés étrangères (pour permettre à l'agent de construire des JOINs).
    """
    insp = inspect(get_engine())
    if table not in list_tables():
        raise DatabaseException(f"Table inconnue: {table}")

    columns = [
        {"name": c["name"], "type": str(c["type"]), "nullable": c.get("nullable", True)}
        for c in insp.get_columns(table)
    ]
    try:
        pk = insp.get_pk_constraint(table).get("constrained_columns", [])
    except Exception:  # noqa: BLE001
        pk = []
    try:
        fks = [
            {
                "columns": fk["constrained_columns"],
                "references": f"{fk['referred_table']}({', '.join(fk['referred_columns'])})",
            }
            for fk in insp.get_foreign_keys(table)
        ]
    except Exception:  # noqa: BLE001
        fks = []

    return {"table": table, "columns": columns, "primary_key": pk, "foreign_keys": fks}


def schema_overview(max_tables: int = 50) -> str:
    """Représentation textuelle compacte de tout le schéma (pour l'agent/UX)."""
    lines: list[str] = []
    for tbl in list_tables()[:max_tables]:
        try:
            s = get_schema(tbl)
        except DatabaseException:
            continue
        cols = ", ".join(f"{c['name']} {c['type']}" for c in s["columns"])
        lines.append(f"• {tbl}({cols})")
        for fk in s["foreign_keys"]:
            lines.append(f"    FK {', '.join(fk['columns'])} → {fk['references']}")
    return "\n".join(lines) if lines else "(aucune table)"


def sample_rows(table: str, limit: int = 3) -> pd.DataFrame:
    """Quelques lignes d'exemple — aide l'agent à comprendre les valeurs réelles."""
    safe = _validate_identifier(table)
    return run_select(f'SELECT * FROM "{safe}" LIMIT {int(limit)}')


# ── Exécution en lecture seule ────────────────────────────────────────────────

def assert_read_only(sql: str) -> None:
    """
    Garde-fou : rejette tout ce qui n'est pas une lecture pure.
    Lève DatabaseException si la requête contient une mutation ou plusieurs
    instructions.
    """
    if not settings.database.read_only:
        return

    statements = [s for s in sqlparse.parse(sql) if str(s).strip()]
    if len(statements) != 1:
        raise DatabaseException("Une seule instruction SELECT est autorisée.")

    stmt = statements[0]
    stmt_type = stmt.get_type()  # SELECT, INSERT, UNKNOWN, ...
    if stmt_type not in ("SELECT", "UNKNOWN"):
        raise DatabaseException(f"Requête non autorisée ({stmt_type}). Lecture seule.")

    tokens = {t.value.upper() for t in stmt.flatten()}
    forbidden = tokens & _FORBIDDEN
    if forbidden:
        raise DatabaseException(
            f"Mot-clé interdit en lecture seule: {', '.join(sorted(forbidden))}"
        )

    first_kw = stmt.token_first(skip_cm=True)
    if first_kw is None or first_kw.value.upper() not in ("SELECT", "WITH"):
        raise DatabaseException("La requête doit commencer par SELECT ou WITH.")


def run_select(sql: str, max_rows: int | None = None) -> pd.DataFrame:
    """
    Exécute une requête de lecture et retourne un DataFrame.
    Applique la garde lecture seule, un timeout serveur (Postgres) et un cap de lignes.
    """
    assert_read_only(sql)
    cap = max_rows or settings.database.max_rows
    engine = get_engine()

    try:
        with engine.connect() as conn:
            # statement_timeout et lecture seule sont posés à l'ouverture de la
            # connexion (voir `_install_session_guardrails`) : aucune requête ne
            # peut les oublier, y compris celles exécutées hors de ce module.
            df = pd.read_sql_query(text(sql), conn)
    except DatabaseException:
        raise
    except Exception as e:  # noqa: BLE001
        raise DatabaseException(f"Erreur SQL: {e}")

    if len(df) > cap:
        logger.info("Résultat tronqué à %d lignes (sur %d).", cap, len(df))
        df = df.head(cap)
    return df


def run_select_params(
    sql: str,
    params: dict[str, Any] | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """
    Variante **paramétrée** de `run_select` (bind parameters SQLAlchemy).

    Utilisée par le serveur MCP : les valeurs venant de l'utilisateur/LLM
    passent exclusivement par `params`, jamais par concaténation de chaînes.
    La garde lecture seule et le plafond de lignes restent identiques.
    """
    assert_read_only(sql)
    cap = max_rows or settings.database.max_rows
    engine = get_engine()

    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(text(sql), conn, params=dict(params or {}))
    except DatabaseException:
        raise
    except Exception as e:  # noqa: BLE001
        raise DatabaseException(f"Erreur SQL: {e}")

    if len(df) > cap:
        logger.info("Résultat tronqué à %d lignes (sur %d).", cap, len(df))
        df = df.head(cap)
    return df


def explain_valid(sql: str) -> tuple[bool, str]:
    """Vérifie la validité d'une requête SANS l'exécuter (LIMIT 0 enveloppant)."""
    try:
        assert_read_only(sql)
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text(f"SELECT * FROM ({sql.rstrip(';')}) AS _probe LIMIT 0"))
        return True, "OK"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def _validate_identifier(name: str) -> str:
    """Empêche l'injection via un nom de table issu de l'introspection."""
    if name not in list_tables():
        raise DatabaseException(f"Table inconnue: {name}")
    return name
