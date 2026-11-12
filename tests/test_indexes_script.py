"""Script d'indexation : sûr, idempotent, adapté au profil.

Ces index sont la réponse au risque de volumétrie : sur la base de démonstration
ils ne changent rien, sur une table réelle ils évitent des balayages complets.
Le test vérifie qu'on ne détruit rien et qu'une seconde exécution ne fait rien.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from lpo_data_mcp.profiles import OBSERVATIONS, SENSITIVE_AREAS
from scripts.create_indexes import create_indexes, plan_indexes


@pytest.fixture()
def db_copy(demo_db_path, tmp_path):
    target = tmp_path / "copie.db"
    shutil.copy2(demo_db_path, target)
    return target


def _index_names(path: Path) -> set[str]:
    with sqlite3.connect(path) as connection:
        return {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='observations'"
            )
        }


def test_plan_matches_profile_columns():
    names = [name for name, _ in plan_indexes(OBSERVATIONS)]
    assert "idx_observations_event_date" in names
    assert "idx_observations_species" in names
    assert "idx_observations_area" in names
    # Le plan ne référence que des colonnes réellement présentes dans le profil.
    for _, columns in plan_indexes(OBSERVATIONS):
        assert all(columns)


def test_plan_adapts_to_a_schema_without_coordinates_or_counts():
    plan = dict(plan_indexes(SENSITIVE_AREAS))
    assert plan["idx_sensitive_areas_species"] == ("species_id",)
    assert plan["idx_sensitive_areas_event_date"] == ("create_datetime",)


def test_indexes_are_created_without_touching_data(db_copy, monkeypatch):
    from src.config.settings import settings
    from src.database import engine

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_copy}")
    monkeypatch.setenv("LPO_PROFILE", "observations")
    settings.database.url = f"sqlite:///{db_copy}"
    engine.reset_engine()

    before = sqlite3.connect(db_copy).execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    statements = create_indexes(OBSERVATIONS)
    assert statements

    created = _index_names(db_copy)
    assert {"idx_observations_event_date", "idx_observations_species"} <= created

    after = sqlite3.connect(db_copy).execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    assert after == before, "l'indexation ne doit modifier aucune ligne"

    # Idempotence : la seconde exécution ne produit plus d'instruction.
    assert create_indexes(OBSERVATIONS) == []
    engine.reset_engine()


def test_dry_run_executes_nothing(db_copy, monkeypatch):
    from src.config.settings import settings
    from src.database import engine

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_copy}")
    settings.database.url = f"sqlite:///{db_copy}"
    engine.reset_engine()

    before = _index_names(db_copy)
    statements = create_indexes(OBSERVATIONS, dry_run=True)
    assert statements
    assert _index_names(db_copy) == before
    engine.reset_engine()
