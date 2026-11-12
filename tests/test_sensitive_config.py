"""Liste officielle des espèces sensibles fournie par configuration.

Le schéma réel du dépôt ne porte aucun indicateur de sensibilité : la liste
officielle doit donc pouvoir être fournie sans redémarrage ni changement de
code. Ces tests vérifient que la configuration est bien appliquée **côté
serveur** (SQL), et pas seulement cosmétiquement dans l'interface.
"""

from __future__ import annotations

import pytest

from lpo_data_mcp import privacy


def _non_sensitive_species(service) -> str:
    """Première espèce que la base ne marque pas comme sensible."""
    listing = service.list_species(limit=200)
    for item in listing.species:
        if not item.sensitive:
            return str(item.label)
    pytest.skip("aucune espèce non sensible dans la base de test")


def test_configuration_is_empty_by_default(monkeypatch):
    monkeypatch.delenv("LPO_SENSITIVE_SPECIES", raising=False)
    monkeypatch.delenv("LPO_SENSITIVE_SPECIES_FILE", raising=False)
    assert privacy.configured_sensitive_species() == ()
    assert privacy.sensitive_species_sql_list() == ""


def test_labels_are_cleaned_and_deduplicated(monkeypatch):
    monkeypatch.setenv("LPO_SENSITIVE_SPECIES", " Aigle royal ,, aigle royal, Busard cendré ")
    labels = privacy.configured_sensitive_species()
    assert labels == ("Aigle royal", "aigle royal", "Busard cendré")


def test_file_source_ignores_comments_and_blank_lines(monkeypatch, tmp_path):
    path = tmp_path / "sensibles.txt"
    path.write_text("# Liste officielle\nMilan royal\n\n  Busard cendré  \n", encoding="utf-8")
    monkeypatch.delenv("LPO_SENSITIVE_SPECIES", raising=False)
    monkeypatch.setenv("LPO_SENSITIVE_SPECIES_FILE", str(path))
    assert privacy.configured_sensitive_species() == ("Milan royal", "Busard cendré")


def test_missing_file_is_not_fatal(monkeypatch, tmp_path):
    monkeypatch.setenv("LPO_SENSITIVE_SPECIES_FILE", str(tmp_path / "absent.txt"))
    assert privacy.configured_sensitive_species() == ()


def test_sql_list_escapes_apostrophes(monkeypatch):
    """Un nom vernaculaire avec apostrophe ne doit pas casser la requête."""
    monkeypatch.setenv("LPO_SENSITIVE_SPECIES", "Couleuvre d'Esculape")
    assert privacy.sensitive_species_sql_list() == "'Couleuvre d''Esculape'"


def test_configured_species_becomes_sensitive_in_sql(monkeypatch, demo_service):
    """Une espèce non sensible le devient dès qu'elle figure dans la liste."""
    label = _non_sensitive_species(demo_service)
    before = demo_service.list_species(query=label, limit=5)
    assert before.species and before.species[0].sensitive is False

    monkeypatch.setenv("LPO_SENSITIVE_SPECIES", label)
    after = demo_service.list_species(query=label, limit=5)
    assert after.species[0].sensitive is True

    overview = demo_service.dataset_overview()
    assert any("LPO_SENSITIVE_SPECIES" in warning for warning in overview.meta.warnings)


def test_configured_species_coordinates_are_blurred(monkeypatch, demo_service):
    label = _non_sensitive_species(demo_service)
    monkeypatch.delenv("LPO_SENSITIVE_SPECIES", raising=False)
    exact = demo_service.get_observations(species=[label], limit=50)
    assert any(point.lat is not None for point in exact.observations)

    monkeypatch.setenv("LPO_SENSITIVE_SPECIES", label)
    blurred = demo_service.get_observations(species=[label], limit=50)
    assert blurred.observations
    assert all(
        point.lat is None or round(point.lat, 1) == point.lat for point in blurred.observations
    )
    # La dégradation est annoncée au client (résumé + méta), pas silencieuse.
    assert blurred.meta.privacy == "blurred"
    assert "flout" in blurred.summary.lower()


def test_configured_species_can_be_excluded(monkeypatch, demo_service):
    label = _non_sensitive_species(demo_service)
    monkeypatch.setenv("LPO_SENSITIVE_SPECIES", label)
    kept = demo_service.get_observations(species=[label], limit=50)
    excluded = demo_service.get_observations(species=[label], limit=50, include_sensitive=False)
    assert kept.observations
    assert excluded.observations == []
    assert excluded.meta.row_count == 0


def test_injection_attempt_in_configuration_returns_no_rows(monkeypatch, demo_service):
    """La liste est échappée : une tentative d'injection n'élargit pas la sélection."""
    monkeypatch.delenv("LPO_SENSITIVE_SPECIES", raising=False)
    baseline = {s.label for s in demo_service.list_species(limit=200).species if s.sensitive}

    monkeypatch.setenv("LPO_SENSITIVE_SPECIES", "x' OR 1=1 --")
    listing = demo_service.list_species(limit=200)
    # Aucune espèce ne s'appelle « x' OR 1=1 -- » : la sélection reste identique.
    assert {s.label for s in listing.species if s.sensitive} == baseline


def test_configuration_is_read_at_call_time(monkeypatch, demo_service):
    label = _non_sensitive_species(demo_service)
    assert demo_service.list_species(query=label, limit=5).species[0].sensitive is False
    monkeypatch.setenv("LPO_SENSITIVE_SPECIES", label)
    assert demo_service.list_species(query=label, limit=5).species[0].sensitive is True
    monkeypatch.delenv("LPO_SENSITIVE_SPECIES")
    assert demo_service.list_species(query=label, limit=5).species[0].sensitive is False
