"""Prompt système de l'agent et exemples few-shot A2UI **valides**.

Deux usages :

1. **Documentation exécutable du comportement** : `SYSTEM_PROMPT` énonce les
   invariants de l'agent (jamais de chiffre inventé, catalogue fermé, données
   séparées de la structure, repli texte, états vides explicites).
2. **Few-shots pour un LLM** : les exemples ci-dessous montrent la séquence
   complète *question → appels MCP → messages A2UI → texte de repli*. Ils sont
   construits avec les **mêmes gabarits** que la production
   (`lpo_a2ui.templates`) : ils ne peuvent donc pas dériver du catalogue, et
   `tests/test_prompts.py` vérifie qu'ils restent valides.

Les valeurs numériques des exemples sont **illustratives** et signalées comme
telles : ce sont des formes de réponse, jamais des données du jeu réel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lpo_a2ui import templates
from lpo_a2ui.validation import catalog_id

PROTOCOL_VERSION = "1.0"

SYSTEM_PROMPT = f"""\
Tu es l'orchestrateur de LPO Data AI. Tu réponds à des questions en français sur
des observations naturalistes (espèces, dates, communes, départements, effectifs,
statuts de protection). Tu ne produis jamais de texte libre destiné au client :
tu produis (1) une intention, (2) des appels d'outils MCP, (3) des messages A2UI
conformes au catalogue LPO, (4) un résumé texte de repli.

## Règles absolues

1. **Aucun chiffre inventé.** Toute valeur affichée (total, part, maximum,
   nombre d'observations, effectif, coordonnée) doit provenir d'un résultat
   d'outil MCP. Si une donnée manque, dis-le et affiche un état vide ; ne
   comble jamais un trou par une estimation.
2. **Outils typés uniquement.** Tu ne construis jamais de SQL. Tu appelles
   `get_dataset_overview` (une fois, pour connaître le profil et les
   capacités), `list_species` (libellés exacts), `aggregate_observations`
   (tendances, classements, cartes), `get_observations` (détail paginé) et
   `get_species_info` (fiche espèce).
3. **Capacités avant intention.** Si le profil n'expose pas de coordonnées, une
   question « carte » produit un état vide explicite. Idem pour les effectifs.
4. **Catalogue fermé.** Tu n'utilises que les composants déclarés dans
   {catalog_id()} : Text, Row, Column, List, Card, Button, Divider, KpiCard,
   LineChart, BarChart, MapChart, DataTable, Notice, SelectField. Tu n'émets
   jamais de HTML, CSS, JavaScript, ni de style.
5. **Structure / données séparées.** La structure (`components`) est figée pour
   un type de vue ; les valeurs vivent dans le `dataModel` et sont référencées
   par `{{"@path": "/…"}}`. Une valeur mesurée en clair dans un composant est un
   défaut.
6. **Interactions.** Un filtre (`SelectField`) produit une action
   `{{"name": "set_filter", "context": {{"filter": "species|period|area", "value": "…"}}}}`.
   L'action revient à toi : tu rappelles l'outil MCP puis tu envoies
   **uniquement** des `updateDataModel` sur la surface existante. Tu ne recrées
   une surface (`deleteSurface` + `createSurface`) que si le type de vue change.
7. **Validation.** Chaque message est validé (schéma, catalogue, séquences)
   avant envoi. Un message refusé n'est jamais envoyé : tu renvoies alors le
   résumé texte seul.
8. **Repli texte systématique.** Chaque réponse comporte un paragraphe
   autosuffisant (chiffres et période inclus), lisible sans interface.
9. **Confidentialité.** Les coordonnées des espèces sensibles arrivent déjà
   floutées par le serveur MCP : tu ne les reconstitues pas et tu signales la
   maille dans une `Notice`.

## Choix de la visualisation

| Intention | kind | Composant principal | Agrégation |
| --- | --- | --- | --- |
| évolution dans le temps | trend | LineChart | `group_by=year` (ou `month`) |
| top / classement d'espèces | ranking | BarChart | `group_by=species` |
| comptage / classement de zones | ranking | BarChart | `department`, `region` ou `category` |
| carte / localisation | map | MapChart | `group_by=grid` |
| comparaison de 2+ espèces | comparison | LineChart multi-séries | `group_by=species_month` |
| fiche espèce | species | KpiCard + Notice | `get_species_info` |
| hors périmètre / données absentes | empty | Notice | aucun appel d'agrégation |

L'axe d'un classement est déduit de la question, pas inventé : une question
qui parle de « zones » sans citer d'espèce (par exemple « combien de zones en
Haute-Savoie ? ») se classe par département ; « par catégorie » se classe sur
la colonne `category`, disponible seulement si l'aperçu du jeu de données
l'annonce dans ses capacités.

## Format de sortie

Pour chaque tour : `intent` (kind, filtres, période), `tool_calls` (nom + 
arguments), `messages` (A2UI v1.0 : `{{"version": "1.0", <une seule clé>}}`),
`text` (résumé de repli). Aucune autre clé.
"""


@dataclass(frozen=True)
class ToolCall:
    """Appel d'outil MCP tel qu'il apparaît dans un exemple."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FewShot:
    """Exemple complet : question → outils MCP → surface A2UI → texte."""

    name: str
    question: str
    intent: dict[str, Any]
    tool_calls: tuple[ToolCall, ...]
    messages: tuple[dict, ...]
    text: str

    def as_prompt_block(self) -> str:
        """Rend l'exemple au format attendu dans un prompt (JSON compact)."""
        import json

        calls = "\n".join(
            f"  - {call.tool}({json.dumps(call.arguments, ensure_ascii=False)})"
            for call in self.tool_calls
        )
        payload = json.dumps(list(self.messages), ensure_ascii=False)
        return (
            f"### Exemple : {self.name}\n"
            f"Question : {self.question}\n"
            f"Intention : {json.dumps(self.intent, ensure_ascii=False)}\n"
            f"Appels MCP :\n{calls or '  (aucun)'}\n"
            f"Messages A2UI : {payload}\n"
            f"Texte de repli : {self.text}\n"
        )


# ── Données illustratives (formes de réponse, pas des valeurs du jeu réel) ────

_DEMO_FILTERS = {
    "species": {
        "value": "Milan royal",
        "options": [
            {"label": "Toutes espèces", "value": ""},
            {"label": "Milan royal", "value": "Milan royal"},
            {"label": "Busard cendré", "value": "Busard cendré"},
        ],
    },
    "period": {
        "value": "10y",
        "options": [
            {"label": "10 dernières années", "value": "10y"},
            {"label": "12 derniers mois", "value": "12m"},
            {"label": "Toute la période", "value": "all"},
        ],
    },
    "area": {
        "value": "",
        "options": [
            {"label": "Toute la France", "value": ""},
            {"label": "Gironde", "value": "Gironde"},
            {"label": "Finistère", "value": "Finistère"},
        ],
    },
}


def _filters(species: str = "", period: str = "10y", area: str = "") -> dict[str, Any]:
    filters = {key: dict(value) for key, value in _DEMO_FILTERS.items()}
    filters["species"]["value"] = species
    filters["period"]["value"] = period
    filters["area"]["value"] = area
    return filters


def _footer() -> dict[str, str]:
    return {"source": "Source : serveur MCP lpo-data — données de démonstration"}


def _create(kind: str, surface_id: str, data: dict[str, Any]) -> tuple[dict, ...]:
    return tuple(templates.surface_messages(kind, surface_id, data))


# 1. Tendance ────────────────────────────────────────────────────────────────

_TREND = _create(
    "trend",
    "example-trend",
    {
        "header": {
            "title": "Évolution des observations — Milan royal",
            "subtitle": "2017–2026 · France entière",
        },
        "kpis": [
            {
                "label": "Observations cumulées",
                "value": "769",
                "hint": "2017–2026",
                "tone": "neutral",
            },
            {"label": "Pic", "value": "2026", "hint": "94 observations", "tone": "neutral"},
            {
                "label": "Évolution 2017 → 2026",
                "value": "+29 %",
                "hint": "73 → 94",
                "tone": "positive",
            },
        ],
        "chart": {
            "title": "Observations par année",
            "series": [
                {
                    "label": "Milan royal",
                    "points": [
                        {"x": "2017", "y": 73},
                        {"x": "2018", "y": 78},
                        {"x": "2026", "y": 94},
                    ],
                }
            ],
            "xLabel": "Année",
            "yLabel": "Observations",
        },
        "table": {
            "title": "Détail par année",
            "rows": [
                {"period": "2017", "value": 73, "share": 0.095, "delta": 0.0},
                {"period": "2018", "value": 78, "share": 0.101, "delta": 0.068},
                {"period": "2026", "value": 94, "share": 0.122, "delta": 0.205},
            ],
        },
        "notices": [
            {
                "tone": "info",
                "title": "Périmètre",
                "message": "Données de démonstration ; la série est tronquée à 3 points ici.",
            }
        ],
        "filters": _filters(species="Milan royal"),
        "footer": _footer(),
    },
)

# 2. Classement ──────────────────────────────────────────────────────────────

_RANKING = _create(
    "ranking",
    "example-ranking",
    {
        "header": {
            "title": "Top 3 des espèces — Finistère",
            "subtitle": "année 2025 · Finistère",
        },
        "kpis": [
            {
                "label": "Observations (toutes espèces)",
                "value": "94",
                "hint": "année 2025",
                "tone": "neutral",
            },
            {
                "label": "Espèce en tête",
                "value": "Buse variable",
                "hint": "17 observations",
                "tone": "neutral",
            },
            {
                "label": "Espèces classées",
                "value": "3",
                "hint": "sur 3 renvoyées",
                "tone": "neutral",
            },
        ],
        "chart": {
            "title": "Classement par observations",
            "items": [
                {"label": "Buse variable", "value": 17},
                {"label": "Chevreuil européen", "value": 12},
                {"label": "Renard roux", "value": 11},
            ],
            "valueLabel": "Observations",
        },
        "table": {
            "title": "Détail du classement",
            "rows": [
                {"label": "Buse variable", "value": 17, "share": 0.157},
                {"label": "Chevreuil européen", "value": 12, "share": 0.111},
                {"label": "Renard roux", "value": 11, "share": 0.102},
            ],
        },
        "notices": [
            {
                "tone": "info",
                "title": "Périmètre",
                "message": "Classement limité au top 3 dans cet exemple.",
            }
        ],
        "filters": _filters(period="this_year", area="Finistère"),
        "footer": _footer(),
    },
)

# 3. Carte ───────────────────────────────────────────────────────────────────

_MAP = _create(
    "map",
    "example-map",
    {
        "header": {
            "title": "Carte des observations — Busard cendré",
            "subtitle": "printemps 2026 · France entière",
        },
        "kpis": [
            {
                "label": "Mailles occupées",
                "value": "18",
                "hint": "printemps 2026",
                "tone": "neutral",
            },
            {
                "label": "Observations cartographiées",
                "value": "27",
                "hint": "cumul des mailles",
                "tone": "neutral",
            },
            {
                "label": "Mailles sensibles",
                "value": "18",
                "hint": "espèces protégées ou sensibles",
                "tone": "warning",
            },
        ],
        "chart": {
            "title": "Densité d'observations par maille",
            "points": [
                {"lat": 48.4, "lon": -4.5, "label": "48.4, -4.5", "value": 3, "sensitive": True},
                {"lat": 47.8, "lon": -4.6, "label": "47.8, -4.6", "value": 2, "sensitive": True},
                {"lat": 47.2, "lon": -4.7, "label": "47.2, -4.7", "value": 1, "sensitive": True},
            ],
            "legend": [
                {"label": "Maille à faible densité", "value": "1", "tone": "neutral"},
                {"label": "Densité maximale", "value": "3", "tone": "warning"},
            ],
            "privacyMode": "blurred",
            "emptyMessage": "Aucun point géolocalisé.",
        },
        "table": {
            "title": "Mailles les plus denses",
            "caption": "Coordonnées agrégées côté serveur.",
            "rows": [
                {"label": "48.4, -4.5", "value": 3, "share": 0.111},
                {"label": "47.8, -4.6", "value": 2, "share": 0.074},
                {"label": "47.2, -4.7", "value": 1, "share": 0.037},
            ],
        },
        "notices": [
            {
                "tone": "warning",
                "title": "Localisation protégée",
                "message": (
                    "Les coordonnées sont agrégées et arrondies par le serveur MCP "
                    "(espèces sensibles). Aucun point exact n'est transmis au client."
                ),
            }
        ],
        "filters": _filters(species="Busard cendré", period="spring"),
        "footer": _footer(),
    },
)

# 4. Comparaison (deux séries) ───────────────────────────────────────────────

_COMPARISON = _create(
    "trend",
    "example-comparison",
    {
        "header": {
            "title": "Évolution des observations — Milan royal et Busard cendré",
            "subtitle": "12 derniers mois · France entière",
        },
        "kpis": [
            {
                "label": "Observations cumulées",
                "value": "254",
                "hint": "12 derniers mois",
                "tone": "neutral",
            },
            {"label": "Séries comparées", "value": "2", "hint": "Milan royal, Busard cendré"},
        ],
        "chart": {
            "title": "Observations par mois",
            "series": [
                {
                    "label": "Milan royal",
                    "points": [{"x": "2026-04", "y": 22}, {"x": "2026-05", "y": 27}],
                },
                {
                    "label": "Busard cendré",
                    "points": [{"x": "2026-04", "y": 18}, {"x": "2026-05", "y": 31}],
                },
            ],
            "xLabel": "Mois",
            "yLabel": "Observations",
        },
        "table": {
            "title": "Détail mensuel",
            "rows": [
                {"period": "2026-04", "value": 40, "share": 0.157, "delta": 0.0},
                {"period": "2026-05", "value": 58, "share": 0.228, "delta": 0.45},
            ],
        },
        "notices": [
            {
                "tone": "info",
                "title": "Lecture",
                "message": (
                    "Chaque série suit une espèce ; les deux séries partagent la même échelle."
                ),
            }
        ],
        "filters": _filters(species="Milan royal", period="12m"),
        "footer": _footer(),
    },
)

# 5. Fiche espèce ────────────────────────────────────────────────────────────

_SPECIES = _create(
    "species",
    "example-species",
    {
        "header": {"title": "Fiche espèce — Héron cendré", "subtitle": "Ardea cinerea"},
        "kpis": [
            {
                "label": "Observations",
                "value": "812",
                "hint": "toute la période",
                "tone": "neutral",
            },
            {"label": "Départements", "value": "8", "hint": "où l'espèce est observée"},
            {"label": "Protection", "value": "protégée", "hint": "statut du jeu de données"},
        ],
        "species": {
            "name": "Héron cendré",
            "latinName": "Ardea cinerea",
            "protection": "protégée",
            "note": "Fiche construite à partir de get_species_info (aucune donnée externe).",
        },
        "notices": [
            {
                "tone": "info",
                "title": "Source",
                "message": "Identité, statut et volumétrie proviennent du serveur MCP.",
            }
        ],
        "filters": _filters(species="Héron cendré", period="all"),
        "footer": _footer(),
    },
)

# 6. Hors périmètre / données absentes ───────────────────────────────────────

_EMPTY = _create(
    "empty",
    "example-empty",
    {
        "header": {
            "title": "Question hors périmètre",
            "subtitle": "Aucune donnée naturaliste concernée",
        },
        "notices": [
            {
                "tone": "info",
                "title": "Je ne peux pas répondre",
                "message": (
                    "Cette question ne porte pas sur les observations naturalistes exposées "
                    "par le serveur MCP. Aucune donnée ne sera inventée : reformulez avec une "
                    "espèce, une période ou une zone."
                ),
            }
        ],
        "filters": _filters(period="all"),
        "footer": _footer(),
    },
)


FEW_SHOTS: tuple[FewShot, ...] = (
    FewShot(
        name="tendance sur 10 ans",
        question="Évolution des observations de Milan royal sur les 10 dernières années ?",
        intent={
            "kind": "trend",
            "species": ["Milan royal"],
            "start_date": "2017-01-01",
            "end_date": "2026-11-12",
            "period_preset": "10y",
            "granularity": "year",
            "metric": "observations",
        },
        tool_calls=(
            ToolCall(
                "aggregate_observations",
                {
                    "group_by": "year",
                    "metric": "observations",
                    "species": ["Milan royal"],
                    "start_date": "2017-01-01",
                    "end_date": "2026-11-12",
                },
            ),
        ),
        messages=_TREND,
        text=(
            "Milan royal : 769 observations entre 2017 et 2026. Pic en 2026 (94), "
            "soit +29 % par rapport à 2017 (73)."
        ),
    ),
    FewShot(
        name="classement départemental",
        question="Top 10 des espèces observées dans le Finistère en 2025 ?",
        intent={
            "kind": "ranking",
            "department": "Finistère",
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "period_preset": "custom",
            "metric": "observations",
        },
        tool_calls=(
            ToolCall(
                "aggregate_observations",
                {
                    "group_by": "species",
                    "metric": "observations",
                    "department": "Finistère",
                    "start_date": "2025-01-01",
                    "end_date": "2025-12-31",
                    "limit": 10,
                },
            ),
        ),
        messages=_RANKING,
        text=(
            "Sur la zone « Finistère » (année 2025), Buse variable arrive en tête avec "
            "17 observations (15,7 % du total), devant Chevreuil européen et Renard roux. "
            "Total : 94 observations."
        ),
    ),
    FewShot(
        name="carte des observations",
        question="Carte des observations de Busard cendré ce printemps ?",
        intent={
            "kind": "map",
            "species": ["Busard cendré"],
            "start_date": "2026-03-01",
            "end_date": "2026-06-30",
            "period_preset": "spring",
            "granularity": "month",
        },
        tool_calls=(
            ToolCall(
                "aggregate_observations",
                {
                    "group_by": "grid",
                    "metric": "observations",
                    "species": ["Busard cendré"],
                    "start_date": "2026-03-01",
                    "end_date": "2026-06-30",
                    "privacy_mode": "auto",
                },
            ),
        ),
        messages=_MAP,
        text=(
            "27 observations géolocalisées réparties sur 18 mailles pour Busard cendré "
            "(printemps 2026, France entière). Les mailles contenant une espèce sensible "
            "sont agrégées : la précision spatiale est volontairement dégradée."
        ),
    ),
    FewShot(
        name="comparaison de deux espèces",
        question="Compare Milan royal et Busard cendré par mois.",
        intent={
            "kind": "comparison",
            "species": ["Milan royal", "Busard cendré"],
            "period_preset": "12m",
            "granularity": "month",
        },
        tool_calls=(
            ToolCall(
                "aggregate_observations",
                {
                    "group_by": "species_month",
                    "metric": "observations",
                    "species": ["Milan royal", "Busard cendré"],
                    "start_date": "2025-11-12",
                    "end_date": "2026-11-12",
                },
            ),
        ),
        messages=_COMPARISON,
        text=(
            "Comparaison Milan royal / Busard cendré sur 12 mois : 254 observations au "
            "total. En mai 2026, Busard cendré (31) dépasse Milan royal (27)."
        ),
    ),
    FewShot(
        name="fiche espèce",
        question="Fiche espèce du Héron cendré",
        intent={"kind": "species", "species": ["Héron cendré"], "period_preset": "all"},
        tool_calls=(ToolCall("get_species_info", {"species": "Héron cendré"}),),
        messages=_SPECIES,
        text=(
            "Héron cendré (Ardea cinerea) : espèce protégée, 812 observations sur toute la "
            "période disponible, présente dans 8 départements."
        ),
    ),
    FewShot(
        name="hors périmètre",
        question="Quel temps fera-t-il demain ?",
        intent={"kind": "unknown", "period_preset": "all"},
        tool_calls=(),
        messages=_EMPTY,
        text=(
            "Cette question ne porte pas sur les observations naturalistes exposées par le "
            "serveur MCP : je ne peux pas y répondre sans inventer de données. Précisez une "
            "espèce, une période ou une zone."
        ),
    ),
)


def few_shot_prompt() -> str:
    """Bloc de prompt complet : système + exemples, prêt à être envoyé à un LLM."""
    blocks = [SYSTEM_PROMPT, "\n## Exemples\n"]
    blocks.extend(shot.as_prompt_block() for shot in FEW_SHOTS)
    return "\n".join(blocks)


__all__ = [
    "FEW_SHOTS",
    "PROTOCOL_VERSION",
    "SYSTEM_PROMPT",
    "FewShot",
    "ToolCall",
    "few_shot_prompt",
]
