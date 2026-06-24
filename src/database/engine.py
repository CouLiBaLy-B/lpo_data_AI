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
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from ..config.settings import settings
from ..core.exceptions import DatabaseException

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


def _build_engine() -> Engine:
    url = settings.database.url
    connect_args: dict[str, Any] = {}
    engine_kwargs: dict[str, Any] = {"pool_pre_ping": True, "future": True}

    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    else:
        # Postgres : pool raisonnable + timeout serveur appliqué par connexion
        engine_kwargs.update(pool_size=5, max_overflow=5, pool_recycle=1800)

    try:
        engine = create_engine(url, connect_args=connect_args, **engine_kwargs)
        # Test de connexion immédiat pour échouer tôt et clairement
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Moteur DB initialisé: %s", _safe_url(url))
        return engine
    except Exception as e:  # noqa: BLE001
        raise DatabaseException(f"Connexion base de données impossible: {e}")


def _safe_url(url: str) -> str:
    """Masque le mot de passe dans les logs."""
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        if "@" in rest:
            creds, host = rest.split("@", 1)
            user = creds.split(":", 1)[0]
            return f"{scheme}://{user}:***@{host}"
    return url


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
            if engine.dialect.name == "postgresql":
                conn.execute(
                    text(f"SET statement_timeout = {int(settings.database.statement_timeout_ms)}")
                )
            df = pd.read_sql_query(text(sql), conn)
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
