"""Compréhension de l'intention (volet déterministe, sans LLM).

C'est le composant le plus critique pour la promesse « l'agent ne fabrique
jamais un chiffre » : il ne produit *que* des filtres et un type de
visualisation, jamais de valeur.
"""

from __future__ import annotations

from datetime import date

import pytest

from orchestrator.intent import (
    PERIOD_PRESETS,
    Intent,
    IntentParser,
    extract_area,
    extract_metric,
    extract_period,
    extract_species,
    extract_unknown_species,
    guess_kind,
    has_domain_signal,
    normalize,
    resolve_period,
)

TODAY = date(2026, 11, 12)
SPECIES = ["Milan royal", "Busard cendré", "Héron cendré", "Buse variable", "Milan noir"]
DEPARTMENTS = ["Gironde", "Finistère", "Loire-Atlantique"]
REGIONS = ["Bretagne", "Nouvelle-Aquitaine"]


@pytest.fixture()
def parser():
    return IntentParser(species_vocab=SPECIES, departments=DEPARTMENTS, regions=REGIONS)


def intent_of(parser, question):
    return parser.heuristic(question, today=TODAY)


# ── Normalisation ────────────────────────────────────────────────────────────


def test_normalize_strips_accents_and_case():
    assert normalize("Évolution  du BUSARD Cendré") == "evolution du busard cendre"


@pytest.mark.parametrize(
    "preset,expected_start,expected_end",
    [
        ("10y", "2017-01-01", "2026-11-12"),
        ("this_year", "2026-01-01", "2026-11-12"),
        ("last_year", "2025-01-01", "2025-12-31"),
        ("spring", "2026-03-01", "2026-06-30"),
        ("all", None, None),
    ],
)
def test_resolve_period_presets(preset, expected_start, expected_end):
    start, end, label = resolve_period(preset, TODAY)
    assert (start, end) == (expected_start, expected_end)
    assert label
    assert preset in PERIOD_PRESETS


def test_every_preset_is_resolvable():
    for preset in PERIOD_PRESETS:
        start, end, label = resolve_period(preset, TODAY)
        assert label
        if start and end:
            assert start <= end


# ── Extraction d'entités ─────────────────────────────────────────────────────


def test_extract_species_matches_accent_insensitive_labels():
    assert extract_species("évolution du busard cendre", SPECIES) == ["Busard cendré"]


def test_extract_species_respects_word_boundaries():
    # « Milan royal » ne doit pas matcher dans « Milanoiseau ».
    assert extract_species("milanoiseau", SPECIES) == []


def test_extract_species_prefers_longest_label():
    found = extract_species("compare le Milan royal et le Milan noir", SPECIES)
    assert found == ["Milan royal", "Milan noir"]


def test_extract_species_falls_back_on_distinctive_word():
    assert extract_species("et le heron ?", SPECIES) == ["Héron cendré"]


def test_extract_species_unknown_returns_empty():
    assert extract_species("le dodo de l'île Maurice", SPECIES) == []


def test_unknown_species_are_detected_instead_of_silently_dropped():
    """« Milan royal » absent du vocabulaire ne doit pas devenir « toutes espèces »."""
    assert extract_unknown_species(
        "Évolution des observations de Milan royal sur les 10 dernières années ?", []
    ) == ["Milan royal"]


def test_known_species_are_never_flagged_as_unknown():
    assert extract_unknown_species("évolution du Milan royal", ["Milan royal"]) == []
    assert extract_unknown_species("carte du Busard cendré", ["Busard cendré"]) == []


def test_known_areas_are_not_taken_for_species():
    assert (
        extract_unknown_species(
            "observations de Gironde en 2020", ["Milan royal"], areas=(["Gironde"], [])
        )
        == []
    )


def test_a_grammatical_word_never_extends_a_species_name():
    assert extract_unknown_species(
        "Évolution des observations de Milan royal sur 10 ans", []
    ) == ["Milan royal"]


def test_ordinary_questions_are_not_flagged(parser):
    """Une formulation sans nom d'espèce ne doit pas être bloquée."""
    for question in (
        "Évolution du nombre de zones par département",
        "Top 10 des espèces observées en 2025",
        "Quelles observations à Paris en 2020 ?",
    ):
        assert parser.heuristic(question).unknown_species == []


def test_parser_exposes_the_unknown_species(parser):
    # L'Aigle royal ne figure pas dans le vocabulaire de test.
    intent = parser.heuristic("Fiche espèce de l'Aigle royal")
    assert intent.unknown_species == ["Aigle royal"]
    assert intent.species == []


def test_known_species_mentioned_with_an_article_is_resolved(parser):
    intent = parser.heuristic("Fiche espèce du Héron cendré")
    assert intent.species == ["Héron cendré"]
    assert intent.unknown_species == []


@pytest.mark.parametrize(
    "text,preset",
    [
        ("sur les 10 dernières années", "10y"),
        ("ces 12 derniers mois", "12m"),
        ("en 2024", "custom"),
        ("au printemps", "spring"),
        ("cette année", "this_year"),
        ("l'an dernier", "last_year"),
    ],
)
def test_extract_period_detects_presets(text, preset):
    start, end, label, found = extract_period(text, TODAY)
    assert found == preset
    assert label


def test_extract_period_explicit_year_is_known_bounds():
    start, end, label, _ = extract_period("en 2024", TODAY)
    assert (start, end) == ("2024-01-01", "2024-12-31")
    assert "2024" in label


def test_extract_period_month_name():
    start, end, _, preset = extract_period("en mars 2025", TODAY)
    assert start.startswith("2025-03")
    assert preset == "custom"


def test_extract_period_defaults_to_all():
    assert extract_period("montre-moi les observations", TODAY)[3] == "all"


def test_extract_area_department_and_region():
    assert extract_area("en Gironde", DEPARTMENTS, REGIONS) == ("Gironde", None)
    assert extract_area("en Bretagne", DEPARTMENTS, REGIONS) == (None, "Bretagne")


def test_extract_area_absent():
    assert extract_area("partout", DEPARTMENTS, REGIONS) == (None, None)


def test_extract_metric():
    assert extract_metric("nombre d'individus observés") == "individuals"
    assert extract_metric("combien d'observations") == "observations"


# ── Choix de la visualisation ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "question,species,compare,kind",
    [
        ("évolution sur 10 ans", ["Milan royal"], False, "trend"),
        ("top 10 des espèces", [], False, "ranking"),
        ("carte des observations", [], False, "map"),
        ("où sont les observations", [], False, "map"),
        ("fiche espèce du Milan royal", ["Milan royal"], False, "species"),
        ("compare A et B", ["Milan royal", "Milan noir"], True, "comparison"),
    ],
)
def test_guess_kind(question, species, compare, kind):
    assert guess_kind(question, species, compare) == kind


def test_parser_detects_comparison_requires_two_species(parser):
    intent = intent_of(parser, "Compare Milan royal et Busard cendré par mois.")
    assert intent.kind == "comparison"
    assert intent.granularity == "month"
    assert intent.species == ["Milan royal", "Busard cendré"]


def test_parser_compare_with_one_species_falls_back_to_trend(parser):
    intent = intent_of(parser, "Compare le Milan royal avec l'an dernier")
    assert intent.kind == "trend"


@pytest.mark.parametrize(
    "question,kind",
    [
        ("Évolution des observations de Milan royal sur les 10 dernières années ?", "trend"),
        ("Top 10 des espèces observées dans le Finistère en 2025 ?", "ranking"),
        ("Carte des observations de Busard cendré ce printemps ?", "map"),
        ("Fiche espèce du Héron cendré", "species"),
        ("Compare Milan royal et Busard cendré par mois.", "comparison"),
        ("Quel temps fera-t-il demain ?", "unknown"),
        ("Écris-moi un poème sur la mer", "unknown"),
    ],
)
def test_reference_questions(parser, question, kind):
    assert intent_of(parser, question).kind == kind


def test_parser_extracts_area_and_period(parser):
    intent = intent_of(parser, "Top 10 des espèces en Gironde en 2025")
    assert intent.department == "Gironde"
    assert intent.start_date == "2025-01-01"


def test_parser_metric_defaults_to_observations(parser):
    assert intent_of(parser, "évolution du Milan royal").metric == "observations"


# ── Hors périmètre ───────────────────────────────────────────────────────────


def test_domain_signal_detection():
    assert has_domain_signal("carte des oiseaux", [], (None, None), "all") is True
    assert has_domain_signal("recette de cuisine", [], (None, None), "all") is False
    # Une espèce ou une zone connue suffit, même sans mot-clé du domaine.
    assert has_domain_signal("et le milan ?", ["Milan royal"], (None, None), "all") is True
    assert has_domain_signal("et en Gironde ?", [], ("Gironde", None), "all") is True


def test_unknown_kind_keeps_no_filter_hallucination(parser):
    intent = intent_of(parser, "Quel temps fera-t-il demain ?")
    assert intent.kind == "unknown"
    assert intent.species == []
    assert intent.start_date is None


def test_species_list_is_capped_at_three(parser):
    question = "Compare Milan royal, Milan noir, Busard cendré, Héron cendré et Buse variable"
    assert len(intent_of(parser, question).species) <= 3


# ── Contrat de sérialisation ─────────────────────────────────────────────────


def test_intent_serialises_to_dict():
    intent = Intent(kind="trend", species=["Milan royal"], period_preset="10y")
    payload = intent.as_dict() if hasattr(intent, "as_dict") else intent.__dict__
    assert payload["kind"] == "trend"
    assert payload["species"] == ["Milan royal"]


def test_intent_contains_no_measure_field():
    """Aucun champ de l'intention ne peut porter une valeur mesurée."""
    fields = set(Intent.__dataclass_fields__)
    for forbidden in ("value", "values", "count", "total", "series", "points"):
        assert forbidden not in fields
