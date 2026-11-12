"""Modèles d'interface A2UI v1.0 pour l'analyse naturaliste.

Trois modèles demandés — **tendance**, **classement**, **carte** — plus une
**fiche espèce** et un **état vide** explicite. Chaque modèle produit la même
chose : une liste de messages agent → renderer.

    createSurface  (structure + data model initial, en un seul message v1.0)
    updateDataModel (mises à jour ultérieures : données seules)

Les valeurs numériques ne sont **jamais** écrites dans la structure : tout passe
par des liaisons ``{"@path": ...}`` vers le data model, lui-même construit à
partir des résultats MCP. C'est ce qui permet de ne renvoyer que des données
quand un filtre change.

Références : https://a2ui.org/specification/v1.0-a2ui/ (messages, `ChildList`,
liaisons `@path`), https://a2ui.org/concepts/data-binding/ et
https://a2ui.org/concepts/actions-and-functions/.
"""

from __future__ import annotations

from typing import Any, Literal

from .validation import catalog_id

SurfaceKind = Literal["trend", "ranking", "map", "species", "empty"]

PROTOCOL_VERSION = "1.0"

#: Clés du data model mises à jour en priorité lors d'une interaction.
UPDATE_PATHS: dict[str, list[str]] = {
    "trend": [
        "/header/subtitle",
        "/kpis",
        "/chart/series",
        "/chart/title",
        "/table/rows",
        "/table/caption",
        "/notices",
        "/footer/source",
    ],
    "ranking": [
        "/header/subtitle",
        "/kpis",
        "/chart/items",
        "/chart/title",
        "/table/rows",
        "/table/caption",
        "/notices",
        "/footer/source",
    ],
    "map": [
        "/header/subtitle",
        "/kpis",
        "/chart/points",
        "/chart/legend",
        "/chart/title",
        "/notices",
        "/footer/source",
    ],
    "species": [
        "/header/subtitle",
        "/kpis",
        "/species",
        "/table/rows",
        "/notices",
        "/footer/source",
    ],
    "empty": ["/header/subtitle", "/notices", "/footer/source"],
}


# ── Structure ────────────────────────────────────────────────────────────────


def _shell(kind: SurfaceKind, chart: dict, columns: list[dict]) -> list[dict]:
    """Tronc commun des surfaces : en-tête, KPI, zone principale, filtres."""
    components: list[dict] = [
        {
            "id": "root",
            "component": "Column",
            "children": [
                "header",
                "kpis",
                "main_card",
                "table_card",
                "notice_list",
                "filters_card",
                "footer",
            ],
        },
        {"id": "header", "component": "Card", "child": "header_col"},
        {
            "id": "header_col",
            "component": "Column",
            "children": ["header_title", "header_subtitle"],
        },
        {
            "id": "header_title",
            "component": "Text",
            "text": {"@path": "/header/title"},
        },
        {
            "id": "header_subtitle",
            "component": "Text",
            "text": {"@path": "/header/subtitle"},
            "variant": "caption",
        },
        {
            "id": "kpis",
            "component": "List",
            "direction": "horizontal",
            "children": {"path": "/kpis", "componentId": "kpi_tpl"},
        },
        {
            "id": "kpi_tpl",
            "component": "KpiCard",
            "label": {"@path": "label"},
            "value": {"@path": "value"},
            "hint": {"@path": "hint"},
            "trend": {"@path": "trend"},
            "tone": {"@path": "tone"},
        },
        {"id": "main_card", "component": "Card", "child": chart["id"]},
        chart,
        {"id": "table_card", "component": "Card", "child": "table"},
        {
            "id": "table",
            "component": "DataTable",
            "title": {"@path": "/table/title"},
            "caption": {"@path": "/table/caption"},
            "columns": columns,
            "rows": {"@path": "/table/rows"},
            "maxRows": 12,
            "emptyMessage": {"@path": "/table/emptyMessage"},
        },
        {
            "id": "notice_list",
            "component": "List",
            "children": {"path": "/notices", "componentId": "notice_tpl"},
        },
        {
            "id": "notice_tpl",
            "component": "Notice",
            "tone": {"@path": "tone"},
            "title": {"@path": "title"},
            "message": {"@path": "message"},
        },
        {"id": "filters_card", "component": "Card", "child": "filters_col"},
        {
            "id": "filters_col",
            "component": "Column",
            "children": ["filters_title", "filters_row"],
        },
        {"id": "filters_title", "component": "Text", "text": "Affiner", "variant": "caption"},
        {
            "id": "filters_row",
            "component": "Row",
            "children": ["filter_species", "filter_period", "filter_area"],
        },
        *_filter_components(kind),
        {
            "id": "footer",
            "component": "Text",
            "text": {"@path": "/footer/source"},
            "variant": "caption",
        },
    ]
    return components


def _filter_components(kind: SurfaceKind) -> list[dict]:
    """Trois filtres liés au data model, qui renvoient une action à l'agent."""

    def select(fid: str, key: str, label: str) -> dict:
        return {
            "id": fid,
            "component": "SelectField",
            "label": label,
            "value": {"@path": f"/filters/{key}/value"},
            "options": {"@path": f"/filters/{key}/options"},
            "action": {
                "event": {
                    "name": "set_filter",
                    "userMessage": f"Filtre {label} modifié",
                    "context": {
                        "filter": key,
                        "value": {"@path": f"/filters/{key}/value"},
                        "surfaceKind": kind,
                    },
                }
            },
        }

    return [
        select("filter_species", "species", "Espèce"),
        select("filter_period", "period", "Période"),
        select("filter_area", "area", "Zone"),
    ]


def _chart(kind: SurfaceKind) -> dict:
    """Composant principal, choisi par l'intention (jamais par le LLM)."""
    common = {
        "title": {"@path": "/chart/title"},
    }
    if kind == "trend":
        return {
            "id": "chart",
            "component": "LineChart",
            **common,
            "series": {"@path": "/chart/series"},
            "xLabel": {"@path": "/chart/xLabel"},
            "yLabel": {"@path": "/chart/yLabel"},
            "valueFormat": "integer",
            "height": 320,
            "emptyMessage": {"@path": "/chart/emptyMessage"},
        }
    if kind == "ranking":
        return {
            "id": "chart",
            "component": "BarChart",
            **common,
            "items": {"@path": "/chart/items"},
            "valueLabel": {"@path": "/chart/valueLabel"},
            "orientation": "horizontal",
            "maxItems": 10,
            "emptyMessage": {"@path": "/chart/emptyMessage"},
        }
    if kind == "map":
        return {
            "id": "chart",
            "component": "MapChart",
            **common,
            "points": {"@path": "/chart/points"},
            "legend": {"@path": "/chart/legend"},
            "privacyMode": {"@path": "/chart/privacyMode"},
            "height": 460,
            "emptyMessage": {"@path": "/chart/emptyMessage"},
        }
    raise ValueError(f"Type de graphique inconnu : {kind}")


_COLUMNS: dict[SurfaceKind, list[dict]] = {
    "trend": [
        {"key": "period", "label": "Période", "align": "left"},
        {"key": "value", "label": "Observations", "align": "right", "format": "integer"},
        {"key": "delta", "label": "Variation", "align": "right", "format": "percent"},
    ],
    "ranking": [
        {"key": "rank", "label": "#", "align": "right", "format": "integer"},
        {"key": "label", "label": "Espèce", "align": "left"},
        {"key": "value", "label": "Observations", "align": "right", "format": "integer"},
        {"key": "share", "label": "Part", "align": "right", "format": "percent"},
    ],
    "map": [
        {"key": "label", "label": "Point", "align": "left"},
        {"key": "value", "label": "Observations", "align": "right", "format": "integer"},
        {"key": "precision", "label": "Précision", "align": "left"},
    ],
    "species": [
        {"key": "period", "label": "Période", "align": "left"},
        {"key": "value", "label": "Observations", "align": "right", "format": "integer"},
        {"key": "individuals", "label": "Individus", "align": "right", "format": "integer"},
    ],
    "empty": [],
}


# ── Data model : valeurs par défaut sûres ────────────────────────────────────


def _kpi(label: str = "", value: str = "—", **extra: Any) -> dict:
    item = {"label": label, "value": value, "hint": "", "trend": "", "tone": "neutral"}
    item.update({k: v for k, v in extra.items() if v is not None})
    return item


def normalise_data(kind: SurfaceKind, data: dict | None) -> dict:
    """Garantit que chaque chemin lié existe (rendu progressif sans `undefined`)."""
    data = dict(data or {})
    header = {"title": "Analyse LPO", "subtitle": "", **data.get("header", {})}
    chart = {
        "title": "",
        "series": [],
        "items": [],
        "points": [],
        "legend": [],
        "xLabel": "",
        "yLabel": "",
        "valueLabel": "",
        "privacyMode": "exact",
        "emptyMessage": "",
        **data.get("chart", {}),
    }
    table = {
        "title": "",
        "caption": "",
        "rows": [],
        "emptyMessage": "Aucune ligne à afficher.",
        **data.get("table", {}),
    }
    filters = {
        "species": {"value": "", "options": []},
        "period": {"value": "", "options": []},
        "area": {"value": "", "options": []},
        **data.get("filters", {}),
    }
    return {
        "header": header,
        "kpis": data.get("kpis") or [_kpi("Observations", "—", hint="Aucune donnée")],
        "chart": chart,
        "table": table,
        "notices": data.get("notices") or [],
        "filters": filters,
        "species": data.get("species", {}),
        "footer": {
            "source": "Source : serveur MCP lpo-data (lecture seule)",
            **data.get("footer", {}),
        },
    }


# ── Fabriques de messages ────────────────────────────────────────────────────


def surface_messages(kind: SurfaceKind, surface_id: str, data: dict | None) -> list[dict]:
    """Messages de création d'une surface pour un modèle d'interface donné."""
    model = normalise_data(kind, data)
    if kind == "empty":
        components = _empty_components()
    elif kind == "species":
        components = _species_components()
    else:
        components = _shell(kind, _chart(kind), _COLUMNS[kind])

    return [
        {
            "version": PROTOCOL_VERSION,
            "createSurface": {
                "surfaceId": surface_id,
                "catalogId": catalog_id(),
                "sendDataModel": True,
                "components": components,
                "dataModel": model,
            },
        }
    ]


def update_messages(kind: SurfaceKind, surface_id: str, data: dict | None) -> list[dict]:
    """Mises à jour **données seules** sur une surface existante.

    Aucune structure n'est renvoyée : le renderer conserve ses composants et
    remplace les valeurs liées (cf. https://a2ui.org/concepts/messages/).
    """
    model = normalise_data(kind, data)
    messages: list[dict] = []
    for path in UPDATE_PATHS[kind]:
        value = _resolve(model, path)
        if value is None:
            continue
        messages.append(
            {
                "version": PROTOCOL_VERSION,
                "updateDataModel": {
                    "surfaceId": surface_id,
                    "path": path,
                    "value": value,
                },
            }
        )
    return messages


def delete_messages(surface_id: str) -> list[dict]:
    """Supprime une surface avant d'en créer une nouvelle (unicité des ids)."""
    return [
        {
            "version": PROTOCOL_VERSION,
            "deleteSurface": {"surfaceId": surface_id},
        }
    ]


# ── Surfaces spécialisées ────────────────────────────────────────────────────


def _empty_components() -> list[dict]:
    """Surface minimale : en-tête + explication de l'absence de données."""
    return [
        {
            "id": "root",
            "component": "Column",
            "children": ["header", "notice_list", "footer"],
        },
        {"id": "header", "component": "Card", "child": "header_col"},
        {
            "id": "header_col",
            "component": "Column",
            "children": ["header_title", "header_subtitle"],
        },
        {"id": "header_title", "component": "Text", "text": {"@path": "/header/title"}},
        {
            "id": "header_subtitle",
            "component": "Text",
            "text": {"@path": "/header/subtitle"},
            "variant": "caption",
        },
        {
            "id": "notice_list",
            "component": "List",
            "children": {"path": "/notices", "componentId": "notice_tpl"},
        },
        {
            "id": "notice_tpl",
            "component": "Notice",
            "tone": {"@path": "tone"},
            "title": {"@path": "title"},
            "message": {"@path": "message"},
        },
        {
            "id": "footer",
            "component": "Text",
            "text": {"@path": "/footer/source"},
            "variant": "caption",
        },
    ]


def _species_components() -> list[dict]:
    """Fiche espèce : identité, statut, chiffres clés et série mensuelle."""
    return [
        {
            "id": "root",
            "component": "Column",
            "children": ["header", "kpis", "species_card", "table_card", "notice_list", "footer"],
        },
        {"id": "header", "component": "Card", "child": "header_col"},
        {
            "id": "header_col",
            "component": "Column",
            "children": ["header_title", "header_subtitle"],
        },
        {"id": "header_title", "component": "Text", "text": {"@path": "/header/title"}},
        {
            "id": "header_subtitle",
            "component": "Text",
            "text": {"@path": "/header/subtitle"},
            "variant": "caption",
        },
        {
            "id": "kpis",
            "component": "List",
            "direction": "horizontal",
            "children": {"path": "/kpis", "componentId": "kpi_tpl"},
        },
        {
            "id": "kpi_tpl",
            "component": "KpiCard",
            "label": {"@path": "label"},
            "value": {"@path": "value"},
            "hint": {"@path": "hint"},
            "trend": {"@path": "trend"},
            "tone": {"@path": "tone"},
        },
        {"id": "species_card", "component": "Card", "child": "species_col"},
        {
            "id": "species_col",
            "component": "Column",
            "children": ["species_name", "species_latin", "species_status", "species_note"],
        },
        {"id": "species_name", "component": "Text", "text": {"@path": "/species/name"}},
        {
            "id": "species_latin",
            "component": "Text",
            "text": {"@path": "/species/latinName"},
            "variant": "caption",
        },
        {
            "id": "species_status",
            "component": "Text",
            "text": {"@path": "/species/protection"},
            "variant": "caption",
        },
        {
            "id": "species_note",
            "component": "Text",
            "text": {"@path": "/species/note"},
        },
        {"id": "table_card", "component": "Card", "child": "table"},
        {
            "id": "table",
            "component": "DataTable",
            "title": {"@path": "/table/title"},
            "caption": {"@path": "/table/caption"},
            "columns": _COLUMNS["species"],
            "rows": {"@path": "/table/rows"},
            "maxRows": 12,
            "emptyMessage": {"@path": "/table/emptyMessage"},
        },
        {
            "id": "notice_list",
            "component": "List",
            "children": {"path": "/notices", "componentId": "notice_tpl"},
        },
        {
            "id": "notice_tpl",
            "component": "Notice",
            "tone": {"@path": "tone"},
            "title": {"@path": "title"},
            "message": {"@path": "message"},
        },
        {
            "id": "footer",
            "component": "Text",
            "text": {"@path": "/footer/source"},
            "variant": "caption",
        },
    ]


def _resolve(model: dict, pointer: str) -> Any:
    """Résout un JSON Pointer simple (`/a/b`) dans le data model."""
    current: Any = model
    for part in pointer.strip("/").split("/"):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current
