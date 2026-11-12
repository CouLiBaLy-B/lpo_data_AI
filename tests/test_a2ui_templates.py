"""Catalogue A2UI et gabarits d'interface.

Vérifie que chaque surface produite est **valide** au sens du protocole v1.0
(schéma + catalogue + intégrité de séquence) et que la séparation
structure/données est respectée.
"""

from __future__ import annotations

import json

import pytest

from lpo_a2ui import templates, validation

SAMPLE_DATA = {
    "header": {"title": "Évolution — Milan royal", "subtitle": "2017–2026 · France"},
    "kpis": [{"label": "Observations", "value": "769", "hint": "10 ans", "tone": "neutral"}],
    "chart": {
        "title": "Observations par an",
        "series": [{"label": "Milan royal", "points": [{"x": "2020", "y": 12}]}],
        "items": [{"label": "Buse variable", "value": 318}],
        "points": [
            {"lat": 47.8, "lon": -4.6, "label": "47.8, -4.6", "value": 3, "sensitive": True}
        ],
        "legend": [{"label": "Maille faible", "value": "1"}],
        "xLabel": "Année",
        "yLabel": "Observations",
        "valueLabel": "Observations",
        "privacyMode": "blurred",
    },
    "table": {"title": "Détail", "rows": [{"period": "2020", "value": 12, "delta": 0.1}]},
    "notices": [{"tone": "info", "title": "Périmètre", "message": "Données de démonstration."}],
    "filters": {
        "species": {
            "value": "Milan royal",
            "options": [{"label": "Milan royal", "value": "Milan royal"}],
        },
        "period": {"value": "10y", "options": [{"label": "10 dernières années", "value": "10y"}]},
        "area": {"value": "", "options": [{"label": "Toute la France", "value": ""}]},
    },
    "species": {
        "name": "Milan royal",
        "latinName": "Milvus milvus",
        "protection": "protégée",
        "note": "",
    },
    "footer": {"source": "Source : MCP lpo-data"},
}

KINDS = ["trend", "ranking", "map", "species", "empty"]


def test_catalog_is_valid_and_declares_expected_components():
    catalog = validation.catalog_json()
    assert catalog["protocolVersion"] == "1.0"
    assert catalog["catalogId"] == validation.catalog_id()
    components = set(catalog["components"])
    assert {"Text", "Row", "Column", "List", "Card", "Button", "Divider"} <= components
    assert {
        "KpiCard",
        "LineChart",
        "BarChart",
        "MapChart",
        "DataTable",
        "Notice",
        "SelectField",
    } <= components
    # `Surface` est réservé par le protocole et ne doit pas être déclaré.
    assert "Surface" not in components
    for name, schema in catalog["components"].items():
        assert schema["properties"]["component"]["const"] == name


@pytest.mark.parametrize("kind", KINDS)
def test_surface_messages_are_protocol_valid(kind):
    messages = templates.surface_messages(kind, f"test-{kind}", SAMPLE_DATA)
    report = validation.validate_messages(messages)
    assert report.ok, report.errors
    assert report.checked_messages == 1
    message = messages[0]
    assert message["version"] == "1.0"
    assert message["createSurface"]["catalogId"] == validation.catalog_id()
    assert message["createSurface"]["sendDataModel"] is True


@pytest.mark.parametrize("kind", KINDS)
def test_structure_contains_no_hardcoded_measure(kind):
    """Aucun chiffre métier ne doit être écrit dans la structure des composants."""
    components = templates.surface_messages(kind, f"test-{kind}", SAMPLE_DATA)[0]["createSurface"][
        "components"
    ]
    for component in components:
        for key in ("series", "items", "points", "rows"):
            if key in component:
                assert component[key] == {"@path": f"/chart/{key}"} or component[key] == {
                    "@path": f"/table/{key}"
                }, f"{component['component']} expose {key} en littéral"
    # Les seules chaînes littérales sont des libellés d'interface.
    charts = [c for c in components if c["component"] in {"LineChart", "BarChart", "MapChart"}]
    for chart in charts:
        assert isinstance(chart["title"], dict) and "@path" in chart["title"]
    if kind in {"trend", "ranking", "map"}:
        assert len(charts) == 1


def test_update_messages_only_touch_data_model():
    messages = templates.update_messages("trend", "surface-1", SAMPLE_DATA)
    assert messages
    assert {list(message)[1] for message in messages} == {"updateDataModel"}
    paths = {message["updateDataModel"]["path"] for message in messages}
    assert "/chart/series" in paths and "/kpis" in paths


def test_update_requires_existing_surface():
    """Le protocole est *stateful* : un update sans surface préalable est invalide."""
    messages = templates.update_messages("trend", "surface-1", SAMPLE_DATA)
    report = validation.validate_messages(messages)
    assert report.ok is False
    assert "Surface not found" in report.errors[0]


def test_stateful_validation_accepts_incremental_updates():
    state = validation.new_renderer_state()
    created = templates.surface_messages("trend", "surface-1", SAMPLE_DATA)
    assert validation.validate_messages(created, state=state).ok
    assert validation.validate_messages(
        templates.update_messages("trend", "surface-1", SAMPLE_DATA), state=state
    ).ok


def test_delete_then_create_is_valid():
    messages = templates.delete_messages("surface-1") + templates.surface_messages(
        "ranking", "surface-2", SAMPLE_DATA
    )
    assert validation.validate_messages(messages).ok


def test_invalid_message_is_rejected_by_schema():
    broken = {
        "version": "1.0",
        "createSurface": {
            "surfaceId": "x",
            "catalogId": validation.catalog_id(),
            "components": [{"id": "root", "component": "Column", "children": ["chart"]}],
            "dataModel": {},
        },
    }
    report = validation.validate_messages([broken])
    assert report.ok is False


def test_unknown_component_is_rejected():
    message = {
        "version": "1.0",
        "createSurface": {
            "surfaceId": "x",
            "catalogId": validation.catalog_id(),
            "components": [{"id": "root", "component": "VideoPlayer", "url": "http://x"}],
            "dataModel": {},
        },
    }
    report = validation.validate_messages([message])
    assert report.ok is False


def test_literal_measures_are_rejected_by_lpo_contract():
    data = json.loads(json.dumps(SAMPLE_DATA))
    message = templates.surface_messages("trend", "surface-x", data)[0]
    chart = next(c for c in message["createSurface"]["components"] if c["component"] == "LineChart")
    chart["series"] = [{"label": "X", "points": [{"x": "2020", "y": 1}]}]
    report = validation.validate_messages([message])
    assert report.ok is False
    assert "littéral" in report.errors[0]


def test_normalise_data_provides_all_bound_paths():
    model = templates.normalise_data("map", None)
    for path in (
        "header/title",
        "kpis",
        "chart/points",
        "chart/legend",
        "table/rows",
        "notices",
        "filters/species/options",
        "footer/source",
    ):
        node = model
        for part in path.split("/"):
            assert part in node, f"chemin manquant : {path}"
            node = node[part]


def test_templates_use_only_catalog_components():
    catalog_components = set(validation.catalog_json()["components"])
    for kind in KINDS:
        components = templates.surface_messages(kind, f"s-{kind}", SAMPLE_DATA)[0]["createSurface"][
            "components"
        ]
        for component in components:
            assert component["component"] in catalog_components
