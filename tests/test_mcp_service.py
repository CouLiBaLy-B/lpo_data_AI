"""Tests du service de données exposé par le serveur MCP."""

from __future__ import annotations

import pytest

from lpo_data_mcp.sql import QueryError

# ── Profil « observations » (jeu de démonstration) ───────────────────────────


def test_overview_describes_capabilities_and_provenance(demo_service):
    overview = demo_service.dataset_overview()
    assert overview.profile == "observations"
    assert overview.is_demo is True
    assert "coordinates" in overview.capabilities
    assert "individual_counts" in overview.capabilities
    assert overview.total_rows > 0
    assert overview.date_min and overview.date_max
    assert overview.departments
    assert (
        "démonstration" in overview.summary.lower()
        or "démonstration" in (overview.dataset_label or "").lower()
    )


def test_list_species_returns_labels_counts_and_sensitivity(demo_service):
    result = demo_service.list_species(limit=5)
    assert result.species
    assert result.total >= len(result.species)
    assert all(s.label and s.observation_count >= 0 for s in result.species)
    assert any(s.sensitive for s in demo_service.list_species(limit=200).species)
    assert "observations" in result.summary


def test_list_species_pagination_changes_results(demo_service):
    first = demo_service.list_species(limit=3, offset=0)
    second = demo_service.list_species(limit=3, offset=3)
    assert [s.label for s in first.species] != [s.label for s in second.species]


def test_aggregate_by_year_is_sorted_and_shares_sum_to_one(demo_service):
    result = demo_service.aggregate_observations(group_by="year", metric="observations")
    assert result.points
    keys = [int(p.key) for p in result.points]
    assert keys == sorted(keys)
    assert result.total == pytest.approx(sum(p.value for p in result.points), rel=0.001)
    assert sum(p.share or 0 for p in result.points) == pytest.approx(1.0, rel=0.01)


def test_aggregate_ranking_is_descending_with_shares(demo_service):
    result = demo_service.aggregate_observations(group_by="species", limit=5)
    values = [p.value for p in result.points]
    assert values == sorted(values, reverse=True)
    assert result.points[0].share == pytest.approx(values[0] / result.total, rel=0.001)


def test_aggregate_individuals_metric_uses_counts(demo_service):
    observations = demo_service.aggregate_observations(group_by="department", metric="observations")
    individuals = demo_service.aggregate_observations(group_by="department", metric="individuals")
    assert observations.points and individuals.points
    assert sum(p.value for p in individuals.points) >= sum(p.value for p in observations.points)


def test_aggregate_grid_blurs_coordinates_and_flags_sensitive(demo_service):
    result = demo_service.aggregate_observations(group_by="grid", limit=30)
    assert result.points
    assert all(p.lat is not None and p.lon is not None for p in result.points)
    # Une décimale : maille d'environ 11 km, jamais la coordonnée exacte.
    for point in result.points:
        assert point.lat == round(point.lat, 1)
        assert point.lon == round(point.lon, 1)
    assert any("maille" in w for w in result.meta.warnings)


def test_aggregate_comparison_returns_one_series_per_species(demo_service):
    species = [s.label for s in demo_service.list_species(limit=2).species]
    result = demo_service.aggregate_observations(
        group_by="species_month", species=species, limit=500
    )
    assert {p.series for p in result.points} == set(species)


def test_filters_by_species_and_department(demo_service):
    species = demo_service.list_species(limit=1).species[0].label
    department = demo_service.dataset_overview().departments[0]
    result = demo_service.aggregate_observations(
        group_by="year", species=[species], department=department
    )
    assert result.points
    observations = demo_service.get_observations(species=[species], department=department, limit=5)
    assert all(o.species == species for o in observations.observations)
    assert all(o.department == department for o in observations.observations)


def test_geolocation_is_blurred_for_sensitive_species(demo_service):
    sensitive = next(s.label for s in demo_service.list_species(limit=200).species if s.sensitive)
    page = demo_service.get_observations(species=[sensitive], limit=10)
    assert page.observations
    assert all(o.precision in {"blurred", "department"} for o in page.observations)
    assert all(o.lat == round(o.lat, 1) for o in page.observations if o.lat is not None)


def test_get_observations_pagination_and_totals(demo_service):
    first = demo_service.get_observations(limit=5, offset=0)
    second = demo_service.get_observations(limit=5, offset=5)
    assert first.total == second.total
    assert first.returned == 5
    assert {o.id for o in first.observations}.isdisjoint({o.id for o in second.observations})
    assert first.meta.truncated is True


def test_species_info_returns_monthly_series(demo_service):
    species = demo_service.list_species(limit=1).species[0].label
    detail = demo_service.get_species_info(species)
    assert detail.species.label == species
    assert detail.monthly
    assert detail.summary.startswith(species)


def test_unknown_species_is_reported_not_invented(demo_service):
    with pytest.raises(QueryError):
        demo_service.get_species_info("Dodo de Maurice")


def test_invalid_group_by_is_rejected(demo_service):
    with pytest.raises(QueryError):
        demo_service.aggregate_observations(group_by="sqlite_master")


def test_date_filter_excludes_out_of_range(demo_service):
    result = demo_service.aggregate_observations(
        group_by="year", start_date="2019-01-01", end_date="2019-12-31"
    )
    assert [p.key for p in result.points] == ["2019"]


def test_date_validation(demo_service):
    with pytest.raises(QueryError):
        demo_service.aggregate_observations(group_by="year", start_date="hier")


# ── Profil réel du dépôt : sensitive_areas ───────────────────────────────────


def test_repo_profile_is_degraded_but_honest(repo_service):
    overview = repo_service.dataset_overview()
    assert overview.profile == "sensitive_areas"
    assert "coordinates" not in overview.capabilities
    assert "individual_counts" not in overview.capabilities
    assert "species_names" not in overview.capabilities
    assert overview.notes  # limites explicitées


def test_repo_profile_cannot_map(repo_service):
    result = repo_service.aggregate_observations(group_by="grid")
    assert result.points == []
    assert "coordonnées" in result.summary.lower()
    assert any("coordinates" in w for w in result.meta.warnings)


def test_repo_profile_does_not_invent_a_species_for_regulatory_zones(repo_service):
    """131 des 169 lignes de l'extrait sont des zones sans espèce.

    Leur donner un libellé les faisait classer comme « l'espèce la plus
    observée » : un chiffre inventé. Elles sont désormais écartées et annoncées.
    """
    listing = repo_service.list_species(limit=50)
    assert len(listing.species) == 4
    assert listing.total == 4
    assert all(s.label.startswith("Espèce #") for s in listing.species)
    assert "Zone réglementaire" not in [s.label for s in listing.species]
    assert any("aucune espèce" in warning for warning in listing.meta.warnings)


def test_repo_profile_overview_counts_only_real_species(repo_service):
    overview = repo_service.dataset_overview()
    assert overview.total_rows == 169
    assert overview.species_count == 4
    assert all(label.startswith("Espèce #") for label in overview.sample_species)
    assert any(
        "131" in warning and "aucune espèce" in warning for warning in overview.meta.warnings
    )


def test_repo_profile_species_aggregation_excludes_rows_without_species(repo_service):
    result = repo_service.aggregate_observations(group_by="species")
    assert result.points
    assert all(point.label.startswith("Espèce #") for point in result.points)
    # 169 lignes − 131 zones réglementaires = 38 lignes rattachées à une espèce.
    assert sum(point.value for point in result.points) == 38


def test_repo_profile_rows_without_species_are_null_not_the_string_none(repo_service):
    page = repo_service.get_observations(limit=10)
    assert page.observations
    assert all(
        observation.species is None or observation.species.startswith("Espèce #")
        for observation in page.observations
    )
    assert any(observation.species is None for observation in page.observations)


def test_repo_profile_species_lookup_requires_an_exact_match(repo_service):
    detail = repo_service.get_species_info("10")
    assert detail.species.label == "Espèce #10"

    with pytest.raises(QueryError) as error:
        repo_service.get_species_info("Zone réglementaire")
    message = str(error.value)
    assert "aucune espèce" in message
    assert "Espèce #10" in message


def test_repo_profile_metric_falls_back_with_warning(repo_service):
    result = repo_service.aggregate_observations(group_by="year", metric="individuals")
    assert result.metric == "observations"
    assert any("Effectifs" in w for w in result.meta.warnings)


def test_protected_species_are_sensitive_even_with_an_explicit_column(demo_service):
    """Le statut de protection suffit : il ne doit pas être masqué par la colonne.

    Le bug corrigé ici : quand le schéma possède une colonne `sensitive`, elle
    était utilisée **seule** — une espèce protégée mais mal marquée (ou dont la
    colonne vaut 0) voyait ses coordonnées diffusées en clair.
    """
    listing = demo_service.list_species(limit=200)
    protected = [s for s in listing.species if (s.protection or "").lower().startswith("prot")]
    assert protected, "la base de démonstration doit contenir des espèces protégées"
    assert all(s.sensitive for s in protected), "toute espèce protégée est sensible"

    sample = protected[0]
    page = demo_service.get_observations(species=[sample.label], limit=20)
    assert page.observations
    assert all(point.lat is None or round(point.lat, 1) == point.lat for point in page.observations)
    assert page.meta.privacy == "blurred"
