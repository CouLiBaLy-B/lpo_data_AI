# Exemple bout en bout — question → MCP → A2UI → interface

> **Trace générée automatiquement** par `python -m lpo_a2ui.examples.build_transcript`.
> Les messages A2UI sont ceux réellement validés par `lpo_a2ui.validation.validate_messages`.

- Base : `data/lpo_demo.db` (profil `observations`)
- Date de référence : 2026-11-12
- Intention : heuristique déterministe (`LPO_LLM_INTENT=off`)
- Renderer : `web/static/renderer.js` (catalogue LPO v1)

Chaque tour présente : la question, les appels MCP réellement effectués,
les messages A2UI émis, le résumé texte de repli, puis l'effet produit.

## Tour 1 — Évolution des observations de Milan royal sur les 10 dernières années ?

**Question** : « Évolution des observations de Milan royal sur les 10 dernières années ? »

### 1. Appels MCP

| Outil | Arguments | État | Durée | Résumé renvoyé |
| --- | --- | --- | --- | --- |
| `aggregate_observations` | `{"end_date": "2026-11-12", "group_by": "year", "limit": 120, "metric": "observations", "species": ["Milan royal"], "start_date": "2017-01-01"}` | ok | < 100 ms | 769 observations au total, par an (10 point(s)). Maximum : 2026 (94). |

### 2. Messages A2UI (`agent → renderer`)

```json
{
  "version": "1.0",
  "createSurface": {
    "surfaceId": "lpo-exemple-bout-en-bout-1",
    "catalogId": "https://lpo-data-ai.example/a2ui/catalogs/lpo/v1/catalog.json",
    "sendDataModel": true,
    "components": [
      {
        "id": "root",
        "component": "Column",
        "children": [
          "header",
          "kpis",
          "main_card",
          "… (4 élément(s) supplémentaire(s))"
        ]
      },
      {
        "id": "header",
        "component": "Card",
        "child": "header_col"
      },
      {
        "id": "header_col",
        "component": "Column",
        "children": [
          "header_title",
          "header_subtitle"
        ]
      },
      "… (18 élément(s) supplémentaire(s))"
    ],
    "dataModel": {
      "header": {
        "title": "Évolution des observations — Milan royal",
        "subtitle": "2017–2026 · France entière"
      },
      "kpis": [
        {
          "label": "Observations cumulées",
          "value": "769",
          "hint": "2017–2026",
          "tone": "neutral"
        },
        {
          "label": "Pic",
          "value": "2026",
          "hint": "94 observations",
          "tone": "neutral"
        },
        {
          "label": "Évolution 2017 → 2026",
          "value": "+29 %",
          "hint": "73 → 94",
          "tone": "positive"
        }
      ],
      "chart": {
        "title": "Observations par année",
        "series": [
          {
            "label": "Milan royal",
            "points": [
              {
                "x": "2017",
                "y": 73.0
              },
              {
                "x": "2018",
                "y": 86.0
              },
              {
                "x": "2019",
                "y": 74.0
              },
              "… (7 élément(s) supplémentaire(s))"
            ]
          }
        ],
        "items": [],
        "points": [],
        "legend": [],
        "xLabel": "Année",
        "yLabel": "Observations",
        "valueLabel": "",
        "privacyMode": "exact",
        "emptyMessage": "Aucune donnée sur la période."
      },
      "table": {
        "title": "Détail par période",
        "caption": "Observations et variation par rapport à la période précédente.",
        "rows": [
          {
            "period": "2017",
            "value": 73.0,
            "delta": null
          },
          {
            "period": "2018",
            "value": 86.0,
            "delta": 0.1780821917808219
          },
          {
            "period": "2019",
            "value": 74.0,
            "delta": -0.13953488372093023
          },
          "… (7 élément(s) supplémentaire(s))"
        ],
        "emptyMessage": "Aucune période à afficher."
      },
      "notices": [
        {
          "tone": "warning",
          "title": "Jeu de démonstration",
          "message": "Jeu de démonstration synthétique (observations naturalistes) — aucune valeur écologique réelle. Les chiffres servent à valider la chaîne MCP → A2UI, pas à documenter la biodiversité."
        }
      ],
      "filters": {
        "species": {
          "value": "Milan royal",
          "options": [
            {
              "label": "Toutes les espèces",
              "value": ""
            },
            {
              "label": "Busard cendré",
              "value": "Busard cendré"
            },
            {
              "label": "Buse variable",
              "value": "Buse variable"
            },
            "… (12 élément(s) supplémentaire(s))"
          ]
        },
        "period": {
          "value": "10y",
          "options": [
            {
              "label": "10 dernières années",
              "value": "10y"
            },
            {
              "label": "12 derniers mois",
              "value": "12m"
            },
            {
              "label": "Cette année",
              "value": "this_year"
            },
            "… (3 élément(s) supplémentaire(s))"
          ]
        },
        "area": {
          "value": "",
          "options": [
            {
              "label": "Toute la France",
              "value": ""
            },
            {
              "label": "Bas-Rhin",
              "value": "Bas-Rhin"
            },
            {
              "label": "Bouches-du-Rhône",
              "value": "Bouches-du-Rhône"
            },
            "… (12 élément(s) supplémentaire(s))"
          ]
        }
      },
      "species": {},
      "footer": {
        "source": "Source : serveur MCP lpo-data (lecture seule) · profil observations · table observations · calculé le <date de génération>"
      }
    }
  }
}
```

### 3. Résumé texte de repli

```text
Milan royal : 769 observations au total (2017–2026, France entière). Maximum en 2026 avec 94 observations. Évolution 2017 → 2026 : +29 %.
```

### 4. Effet côté client

- type de surface : `trend` — mode : `create` — surface : `lpo-exemple-bout-en-bout-1`
- validation : `{'ok': True, 'checked': 1, 'errors': []}`
- composants instanciés : Column, Card, Column, Text, Text, List, KpiCard, Card, LineChart, Card, DataTable, List, Notice, Card, Column, Text, Row, SelectField, SelectField, SelectField, Text

## Tour 2 — Top 10 des espèces observées dans le Finistère en 2025 ?

**Question** : « Top 10 des espèces observées dans le Finistère en 2025 ? »

### 1. Appels MCP

| Outil | Arguments | État | Durée | Résumé renvoyé |
| --- | --- | --- | --- | --- |
| `aggregate_observations` | `{"department": "Finistère", "end_date": "2025-12-31", "group_by": "species", "limit": 10, "metric": "observations", "start_date": "2025-01-01"}` | ok | < 100 ms | 94 observations au total, par espèce (10 point(s)). Maximum : Milan royal (10). |

### 2. Messages A2UI (`agent → renderer`)

```json
{
  "version": "1.0",
  "deleteSurface": {
    "surfaceId": "lpo-exemple-bout-en-bout-1"
  }
}
```

```json
{
  "version": "1.0",
  "createSurface": {
    "surfaceId": "lpo-exemple-bout-en-bout-2",
    "catalogId": "https://lpo-data-ai.example/a2ui/catalogs/lpo/v1/catalog.json",
    "sendDataModel": true,
    "components": [
      {
        "id": "root",
        "component": "Column",
        "children": [
          "header",
          "kpis",
          "main_card",
          "… (4 élément(s) supplémentaire(s))"
        ]
      },
      {
        "id": "header",
        "component": "Card",
        "child": "header_col"
      },
      {
        "id": "header_col",
        "component": "Column",
        "children": [
          "header_title",
          "header_subtitle"
        ]
      },
      "… (18 élément(s) supplémentaire(s))"
    ],
    "dataModel": {
      "header": {
        "title": "Top 10 des espèces — Finistère",
        "subtitle": "année 2025 · Finistère"
      },
      "kpis": [
        {
          "label": "Observations (toutes espèces)",
          "value": "94",
          "hint": "année 2025",
          "tone": "neutral"
        },
        {
          "label": "Espèce en tête",
          "value": "Milan royal",
          "hint": "10 observations",
          "tone": "neutral"
        },
        {
          "label": "Espèces classées",
          "value": "10",
          "hint": "sur 10 renvoyées",
          "tone": "neutral"
        }
      ],
      "chart": {
        "title": "Classement par observations",
        "series": [],
        "items": [
          {
            "label": "Milan royal",
            "value": 10.0,
            "hint": "10,6 %"
          },
          {
            "label": "Héron cendré",
            "value": 10.0,
            "hint": "10,6 %"
          },
          {
            "label": "Faucon crécerelle",
            "value": 10.0,
            "hint": "10,6 %"
          },
          "… (7 élément(s) supplémentaire(s))"
        ],
        "points": [],
        "legend": [],
        "xLabel": "",
        "yLabel": "",
        "valueLabel": "Observations",
        "privacyMode": "exact",
        "emptyMessage": "Aucune espèce à classer."
      },
      "table": {
        "title": "Détail du classement",
        "caption": "Part calculée sur le total du périmètre filtré.",
        "rows": [
          {
            "rank": 1,
            "label": "Milan royal",
            "value": 10.0,
            "share": 0.10638297872340426
          },
          {
            "rank": 2,
            "label": "Héron cendré",
            "value": 10.0,
            "share": 0.10638297872340426
          },
          {
            "rank": 3,
            "label": "Faucon crécerelle",
            "value": 10.0,
            "share": 0.10638297872340426
          },
          "… (7 élément(s) supplémentaire(s))"
        ],
        "emptyMessage": "Aucune ligne."
      },
      "notices": [
        {
          "tone": "warning",
          "title": "Jeu de démonstration",
          "message": "Jeu de démonstration synthétique (observations naturalistes) — aucune valeur écologique réelle. Les chiffres servent à valider la chaîne MCP → A2UI, pas à documenter la biodiversité."
        },
        {
          "tone": "warning",
          "title": "Résultat tronqué",
          "message": "Le serveur MCP a atteint sa limite de lignes : la vue montre un sous-ensemble. Affiner la période ou la zone pour voir l'ensemble."
        }
      ],
      "filters": {
        "species": {
          "value": "",
          "options": [
            {
              "label": "Toutes les espèces",
              "value": ""
            },
            {
              "label": "Busard cendré",
              "value": "Busard cendré"
            },
            {
              "label": "Buse variable",
              "value": "Buse variable"
            },
            "… (12 élément(s) supplémentaire(s))"
          ]
        },
        "period": {
          "value": "custom",
          "options": [
            {
              "label": "10 dernières années",
              "value": "10y"
            },
            {
              "label": "12 derniers mois",
              "value": "12m"
            },
            {
              "label": "Cette année",
              "value": "this_year"
            },
            "… (3 élément(s) supplémentaire(s))"
          ]
        },
        "area": {
          "value": "Finistère",
          "options": [
            {
              "label": "Toute la France",
              "value": ""
            },
            {
              "label": "Bas-Rhin",
              "value": "Bas-Rhin"
            },
            {
              "label": "Bouches-du-Rhône",
              "value": "Bouches-du-Rhône"
            },
            "… (12 élément(s) supplémentaire(s))"
          ]
        }
      },
      "species": {},
      "footer": {
        "source": "Source : serveur MCP lpo-data (lecture seule) · profil observations · table observations · calculé le <date de génération>"
      }
    }
  }
}
```

### 3. Résumé texte de repli

```text
Sur la zone « Finistère » (année 2025), Milan royal arrive en tête avec 10 observations (10,6 % du total), devant Héron cendré, Faucon crécerelle. Total : 94 observations.
```

### 4. Effet côté client

- type de surface : `ranking` — mode : `create` — surface : `lpo-exemple-bout-en-bout-2`
- validation : `{'ok': True, 'checked': 2, 'errors': []}`
- composants instanciés : Column, Card, Column, Text, Text, List, KpiCard, Card, BarChart, Card, DataTable, List, Notice, Card, Column, Text, Row, SelectField, SelectField, SelectField, Text

## Tour 3 — Interaction sur la surface de classement : filtre « zone » → Gironde

**Action A2UI** : `{"name": "set_filter", "context": {"filter": "area", "value": "Gironde"}}`

### 1. Appels MCP

| Outil | Arguments | État | Durée | Résumé renvoyé |
| --- | --- | --- | --- | --- |
| `aggregate_observations` | `{"department": "Gironde", "end_date": "2025-12-31", "group_by": "species", "limit": 10, "metric": "observations", "start_date": "2025-01-01"}` | ok | < 100 ms | 108 observations au total, par espèce (10 point(s)). Maximum : Buse variable (17). |

### 2. Messages A2UI (`agent → renderer`)

```json
{
  "version": "1.0",
  "updateDataModel": {
    "surfaceId": "lpo-exemple-bout-en-bout-2",
    "path": "/header/subtitle",
    "value": "année 2025 · Gironde"
  }
}
```

```json
{
  "version": "1.0",
  "updateDataModel": {
    "surfaceId": "lpo-exemple-bout-en-bout-2",
    "path": "/kpis",
    "value": [
      {
        "label": "Observations (toutes espèces)",
        "value": "108",
        "hint": "année 2025",
        "tone": "neutral"
      },
      {
        "label": "Espèce en tête",
        "value": "Buse variable",
        "hint": "17 observations",
        "tone": "neutral"
      },
      {
        "label": "Espèces classées",
        "value": "10",
        "hint": "sur 10 renvoyées",
        "tone": "neutral"
      }
    ]
  }
}
```

```json
{
  "version": "1.0",
  "updateDataModel": {
    "surfaceId": "lpo-exemple-bout-en-bout-2",
    "path": "/chart/items",
    "value": [
      {
        "label": "Buse variable",
        "value": 17.0,
        "hint": "15,7 %"
      },
      {
        "label": "Chevreuil européen",
        "value": 12.0,
        "hint": "11,1 %"
      },
      {
        "label": "Renard roux",
        "value": 11.0,
        "hint": "10,2 %"
      },
      "… (7 élément(s) supplémentaire(s))"
    ]
  }
}
```

```json
{
  "version": "1.0",
  "updateDataModel": {
    "surfaceId": "lpo-exemple-bout-en-bout-2",
    "path": "/chart/title",
    "value": "Classement par observations"
  }
}
```

```json
{
  "version": "1.0",
  "updateDataModel": {
    "surfaceId": "lpo-exemple-bout-en-bout-2",
    "path": "/table/rows",
    "value": [
      {
        "rank": 1,
        "label": "Buse variable",
        "value": 17.0,
        "share": 0.1574074074074074
      },
      {
        "rank": 2,
        "label": "Chevreuil européen",
        "value": 12.0,
        "share": 0.1111111111111111
      },
      {
        "rank": 3,
        "label": "Renard roux",
        "value": 11.0,
        "share": 0.10185185185185185
      },
      "… (7 élément(s) supplémentaire(s))"
    ]
  }
}
```

```json
{
  "version": "1.0",
  "updateDataModel": {
    "surfaceId": "lpo-exemple-bout-en-bout-2",
    "path": "/table/caption",
    "value": "Part calculée sur le total du périmètre filtré."
  }
}
```

```json
{
  "version": "1.0",
  "updateDataModel": {
    "surfaceId": "lpo-exemple-bout-en-bout-2",
    "path": "/notices",
    "value": [
      {
        "tone": "warning",
        "title": "Jeu de démonstration",
        "message": "Jeu de démonstration synthétique (observations naturalistes) — aucune valeur écologique réelle. Les chiffres servent à valider la chaîne MCP → A2UI, pas à documenter la biodiversité."
      },
      {
        "tone": "warning",
        "title": "Résultat tronqué",
        "message": "Le serveur MCP a atteint sa limite de lignes : la vue montre un sous-ensemble. Affiner la période ou la zone pour voir l'ensemble."
      }
    ]
  }
}
```

```json
{
  "version": "1.0",
  "updateDataModel": {
    "surfaceId": "lpo-exemple-bout-en-bout-2",
    "path": "/footer/source",
    "value": "Source : serveur MCP lpo-data (lecture seule) · profil observations · table observations · calculé le <date de génération>"
  }
}
```

### 3. Résumé texte de repli

```text
Filtre appliqué (zone = Gironde). Sur la zone « Gironde » (année 2025), Buse variable arrive en tête avec 17 observations (15,7 % du total), devant Chevreuil européen, Renard roux. Total : 108 observations.
```

### 4. Effet côté client

- type de surface : `ranking` — mode : `update` — surface : `lpo-exemple-bout-en-bout-2`
- validation : `{'ok': True, 'checked': 8, 'errors': []}`
- data model mis à jour : `/header/subtitle`, `/kpis`, `/chart/items`, `/chart/title`, `/table/rows`, `/table/caption`, `/notices`, `/footer/source`

## Tour 4 — Interaction sur la même surface : filtre « période » → 12 derniers mois

**Action A2UI** : `{"name": "set_filter", "context": {"filter": "period", "value": "12m"}}`

### 1. Appels MCP

| Outil | Arguments | État | Durée | Résumé renvoyé |
| --- | --- | --- | --- | --- |
| `aggregate_observations` | `{"department": "Gironde", "end_date": "2026-11-12", "group_by": "species", "limit": 10, "metric": "observations", "start_date": "2025-11-12"}` | ok | < 100 ms | 147 observations au total, par espèce (10 point(s)). Maximum : Buse variable (26). |

### 2. Messages A2UI (`agent → renderer`)

```json
{
  "version": "1.0",
  "updateDataModel": {
    "surfaceId": "lpo-exemple-bout-en-bout-2",
    "path": "/header/subtitle",
    "value": "12 derniers mois · Gironde"
  }
}
```

```json
{
  "version": "1.0",
  "updateDataModel": {
    "surfaceId": "lpo-exemple-bout-en-bout-2",
    "path": "/kpis",
    "value": [
      {
        "label": "Observations (toutes espèces)",
        "value": "147",
        "hint": "12 derniers mois",
        "tone": "neutral"
      },
      {
        "label": "Espèce en tête",
        "value": "Buse variable",
        "hint": "26 observations",
        "tone": "neutral"
      },
      {
        "label": "Espèces classées",
        "value": "10",
        "hint": "sur 10 renvoyées",
        "tone": "neutral"
      }
    ]
  }
}
```

```json
{
  "version": "1.0",
  "updateDataModel": {
    "surfaceId": "lpo-exemple-bout-en-bout-2",
    "path": "/chart/items",
    "value": [
      {
        "label": "Buse variable",
        "value": 26.0,
        "hint": "17,7 %"
      },
      {
        "label": "Chevreuil européen",
        "value": 21.0,
        "hint": "14,3 %"
      },
      {
        "label": "Renard roux",
        "value": 17.0,
        "hint": "11,6 %"
      },
      "… (7 élément(s) supplémentaire(s))"
    ]
  }
}
```

```json
{
  "version": "1.0",
  "updateDataModel": {
    "surfaceId": "lpo-exemple-bout-en-bout-2",
    "path": "/chart/title",
    "value": "Classement par observations"
  }
}
```

```json
{
  "version": "1.0",
  "updateDataModel": {
    "surfaceId": "lpo-exemple-bout-en-bout-2",
    "path": "/table/rows",
    "value": [
      {
        "rank": 1,
        "label": "Buse variable",
        "value": 26.0,
        "share": 0.17687074829931973
      },
      {
        "rank": 2,
        "label": "Chevreuil européen",
        "value": 21.0,
        "share": 0.14285714285714285
      },
      {
        "rank": 3,
        "label": "Renard roux",
        "value": 17.0,
        "share": 0.11564625850340136
      },
      "… (7 élément(s) supplémentaire(s))"
    ]
  }
}
```

```json
{
  "version": "1.0",
  "updateDataModel": {
    "surfaceId": "lpo-exemple-bout-en-bout-2",
    "path": "/table/caption",
    "value": "Part calculée sur le total du périmètre filtré."
  }
}
```

```json
{
  "version": "1.0",
  "updateDataModel": {
    "surfaceId": "lpo-exemple-bout-en-bout-2",
    "path": "/notices",
    "value": [
      {
        "tone": "warning",
        "title": "Jeu de démonstration",
        "message": "Jeu de démonstration synthétique (observations naturalistes) — aucune valeur écologique réelle. Les chiffres servent à valider la chaîne MCP → A2UI, pas à documenter la biodiversité."
      },
      {
        "tone": "warning",
        "title": "Résultat tronqué",
        "message": "Le serveur MCP a atteint sa limite de lignes : la vue montre un sous-ensemble. Affiner la période ou la zone pour voir l'ensemble."
      }
    ]
  }
}
```

```json
{
  "version": "1.0",
  "updateDataModel": {
    "surfaceId": "lpo-exemple-bout-en-bout-2",
    "path": "/footer/source",
    "value": "Source : serveur MCP lpo-data (lecture seule) · profil observations · table observations · calculé le <date de génération>"
  }
}
```

### 3. Résumé texte de repli

```text
Filtre appliqué (période = 12 derniers mois). Sur la zone « Gironde » (12 derniers mois), Buse variable arrive en tête avec 26 observations (17,7 % du total), devant Chevreuil européen, Renard roux. Total : 147 observations.
```

### 4. Effet côté client

- type de surface : `ranking` — mode : `update` — surface : `lpo-exemple-bout-en-bout-2`
- validation : `{'ok': True, 'checked': 8, 'errors': []}`
- data model mis à jour : `/header/subtitle`, `/kpis`, `/chart/items`, `/chart/title`, `/table/rows`, `/table/caption`, `/notices`, `/footer/source`

## Tour 5 — Carte des observations de Busard cendré ce printemps ?

**Question** : « Carte des observations de Busard cendré ce printemps ? »

### 1. Appels MCP

| Outil | Arguments | État | Durée | Résumé renvoyé |
| --- | --- | --- | --- | --- |
| `aggregate_observations` | `{"end_date": "2026-06-30", "group_by": "grid", "limit": 200, "metric": "observations", "species": ["Busard cendré"], "start_date": "2026-03-01"}` | ok | < 100 ms | 16 observations au total, par maille (16 point(s)). Maximum : 43.5, 3.2 (1). |

### 2. Messages A2UI (`agent → renderer`)

```json
{
  "version": "1.0",
  "deleteSurface": {
    "surfaceId": "lpo-exemple-bout-en-bout-2"
  }
}
```

```json
{
  "version": "1.0",
  "createSurface": {
    "surfaceId": "lpo-exemple-bout-en-bout-3",
    "catalogId": "https://lpo-data-ai.example/a2ui/catalogs/lpo/v1/catalog.json",
    "sendDataModel": true,
    "components": [
      {
        "id": "root",
        "component": "Column",
        "children": [
          "header",
          "kpis",
          "main_card",
          "… (4 élément(s) supplémentaire(s))"
        ]
      },
      {
        "id": "header",
        "component": "Card",
        "child": "header_col"
      },
      {
        "id": "header_col",
        "component": "Column",
        "children": [
          "header_title",
          "header_subtitle"
        ]
      },
      "… (18 élément(s) supplémentaire(s))"
    ],
    "dataModel": {
      "header": {
        "title": "Carte des observations — Busard cendré",
        "subtitle": "printemps 2026 · France entière"
      },
      "kpis": [
        {
          "label": "Mailles occupées",
          "value": "16",
          "hint": "printemps 2026",
          "tone": "neutral"
        },
        {
          "label": "Observations cartographiées",
          "value": "16",
          "hint": "cumul des mailles",
          "tone": "neutral"
        },
        {
          "label": "Mailles sensibles",
          "value": "16",
          "hint": "espèces protégées ou sensibles",
          "tone": "warning"
        }
      ],
      "chart": {
        "title": "Densité d'observations par maille",
        "series": [],
        "items": [],
        "points": [
          {
            "lat": 43.5,
            "lon": 3.2,
            "label": "43.5, 3.2",
            "value": 1.0,
            "sensitive": true
          },
          {
            "lat": 43.6,
            "lon": 3.8,
            "label": "43.6, 3.8",
            "value": 1.0,
            "sensitive": true
          },
          {
            "lat": 43.7,
            "lon": 3.4,
            "label": "43.7, 3.4",
            "value": 1.0,
            "sensitive": true
          },
          "… (13 élément(s) supplémentaire(s))"
        ],
        "legend": [
          {
            "label": "Maille à faible densité",
            "value": "1",
            "tone": "neutral"
          },
          {
            "label": "Densité moyenne",
            "value": "0",
            "tone": "neutral"
          },
          {
            "label": "Densité maximale",
            "value": "1",
            "tone": "warning"
          }
        ],
        "xLabel": "",
        "yLabel": "",
        "valueLabel": "",
        "privacyMode": "blurred",
        "emptyMessage": "Aucun point géolocalisé."
      },
      "table": {
        "title": "Mailles les plus denses",
        "caption": "Coordonnées agrégées côté serveur.",
        "rows": [
          {
            "label": "43.5, 3.2",
            "value": 1.0,
            "precision": "floutée"
          },
          {
            "label": "43.6, 3.8",
            "value": 1.0,
            "precision": "floutée"
          },
          {
            "label": "43.7, 3.4",
            "value": 1.0,
            "precision": "floutée"
          },
          "… (9 élément(s) supplémentaire(s))"
        ],
        "emptyMessage": "Aucune maille."
      },
      "notices": [
        {
          "tone": "warning",
          "title": "Jeu de démonstration",
          "message": "Jeu de démonstration synthétique (observations naturalistes) — aucune valeur écologique réelle. Les chiffres servent à valider la chaîne MCP → A2UI, pas à documenter la biodiversité."
        },
        {
          "tone": "info",
          "title": "À savoir",
          "message": "Coordonnées agrégées sur une maille d'environ 11 km (protection des espèces sensibles)."
        },
        {
          "tone": "warning",
          "title": "Localisation protégée",
          "message": "Les coordonnées sont agrégées et arrondies par le serveur MCP (espèces sensibles). Aucun point exact n'est transmis au client."
        }
      ],
      "filters": {
        "species": {
          "value": "Busard cendré",
          "options": [
            {
              "label": "Toutes les espèces",
              "value": ""
            },
            {
              "label": "Busard cendré",
              "value": "Busard cendré"
            },
            {
              "label": "Buse variable",
              "value": "Buse variable"
            },
            "… (12 élément(s) supplémentaire(s))"
          ]
        },
        "period": {
          "value": "spring",
          "options": [
            {
              "label": "10 dernières années",
              "value": "10y"
            },
            {
              "label": "12 derniers mois",
              "value": "12m"
            },
            {
              "label": "Cette année",
              "value": "this_year"
            },
            "… (3 élément(s) supplémentaire(s))"
          ]
        },
        "area": {
          "value": "",
          "options": [
            {
              "label": "Toute la France",
              "value": ""
            },
            {
              "label": "Bas-Rhin",
              "value": "Bas-Rhin"
            },
            {
              "label": "Bouches-du-Rhône",
              "value": "Bouches-du-Rhône"
            },
            "… (12 élément(s) supplémentaire(s))"
          ]
        }
      },
      "species": {},
      "footer": {
        "source": "Source : serveur MCP lpo-data (lecture seule) · profil observations · table observations · confidentialité : blurred · calculé le <date de génération>"
      }
    }
  }
}
```

### 3. Résumé texte de repli

```text
16 observations géolocalisées réparties sur 16 mailles pour Busard cendré (printemps 2026, France entière). 16 maille(s) contiennent une espèce sensible : la précision spatiale est volontairement dégradée.
```

### 4. Effet côté client

- type de surface : `map` — mode : `create` — surface : `lpo-exemple-bout-en-bout-3`
- validation : `{'ok': True, 'checked': 2, 'errors': []}`
- composants instanciés : Column, Card, Column, Text, Text, List, KpiCard, Card, MapChart, Card, DataTable, List, Notice, Card, Column, Text, Row, SelectField, SelectField, SelectField, Text

## Tour 6 — Compare Milan royal et Busard cendré par mois.

**Question** : « Compare Milan royal et Busard cendré par mois. »

### 1. Appels MCP

| Outil | Arguments | État | Durée | Résumé renvoyé |
| --- | --- | --- | --- | --- |
| `aggregate_observations` | `{"group_by": "species_month", "limit": 500, "metric": "observations", "species": ["Milan royal", "Busard cendré"]}` | ok | < 100 ms | 1 351 observations au total, par mois et par espèce (254 point(s)). Maximum : 2017-08 (12). |

### 2. Messages A2UI (`agent → renderer`)

```json
{
  "version": "1.0",
  "deleteSurface": {
    "surfaceId": "lpo-exemple-bout-en-bout-3"
  }
}
```

```json
{
  "version": "1.0",
  "createSurface": {
    "surfaceId": "lpo-exemple-bout-en-bout-4",
    "catalogId": "https://lpo-data-ai.example/a2ui/catalogs/lpo/v1/catalog.json",
    "sendDataModel": true,
    "components": [
      {
        "id": "root",
        "component": "Column",
        "children": [
          "header",
          "kpis",
          "main_card",
          "… (4 élément(s) supplémentaire(s))"
        ]
      },
      {
        "id": "header",
        "component": "Card",
        "child": "header_col"
      },
      {
        "id": "header_col",
        "component": "Column",
        "children": [
          "header_title",
          "header_subtitle"
        ]
      },
      "… (18 élément(s) supplémentaire(s))"
    ],
    "dataModel": {
      "header": {
        "title": "Comparaison — Milan royal vs Busard cendré",
        "subtitle": "toute la période disponible · France entière"
      },
      "kpis": [
        {
          "label": "Milan royal",
          "value": "841",
          "hint": "observations cumulées",
          "tone": "neutral"
        },
        {
          "label": "Busard cendré",
          "value": "510",
          "hint": "observations cumulées",
          "tone": "neutral"
        }
      ],
      "chart": {
        "title": "Observations par mois",
        "series": [
          {
            "label": "Milan royal",
            "color": "#1f4e79",
            "points": [
              {
                "x": "2016-01",
                "y": 4.0
              },
              {
                "x": "2016-02",
                "y": 7.0
              },
              {
                "x": "2016-03",
                "y": 7.0
              },
              "… (127 élément(s) supplémentaire(s))"
            ]
          },
          {
            "label": "Busard cendré",
            "color": "#2e7d32",
            "points": [
              {
                "x": "2016-01",
                "y": 5.0
              },
              {
                "x": "2016-02",
                "y": 4.0
              },
              {
                "x": "2016-03",
                "y": 3.0
              },
              "… (121 élément(s) supplémentaire(s))"
            ]
          }
        ],
        "items": [],
        "points": [],
        "legend": [],
        "xLabel": "Mois",
        "yLabel": "Observations",
        "valueLabel": "",
        "privacyMode": "exact",
        "emptyMessage": "Aucune série."
      },
      "table": {
        "title": "Totaux comparés",
        "caption": "Somme des observations sur la période.",
        "rows": [
          {
            "period": "Milan royal",
            "value": 841.0,
            "delta": null
          },
          {
            "period": "Busard cendré",
            "value": 510.0,
            "delta": null
          }
        ],
        "emptyMessage": "Aucune donnée."
      },
      "notices": [
        {
          "tone": "warning",
          "title": "Jeu de démonstration",
          "message": "Jeu de démonstration synthétique (observations naturalistes) — aucune valeur écologique réelle. Les chiffres servent à valider la chaîne MCP → A2UI, pas à documenter la biodiversité."
        }
      ],
      "filters": {
        "species": {
          "value": "",
          "options": [
            {
              "label": "Toutes les espèces",
              "value": ""
            },
            {
              "label": "Busard cendré",
              "value": "Busard cendré"
            },
            {
              "label": "Buse variable",
              "value": "Buse variable"
            },
            "… (12 élément(s) supplémentaire(s))"
          ]
        },
        "period": {
          "value": "all",
          "options": [
            {
              "label": "10 dernières années",
              "value": "10y"
            },
            {
              "label": "12 derniers mois",
              "value": "12m"
            },
            {
              "label": "Cette année",
              "value": "this_year"
            },
            "… (3 élément(s) supplémentaire(s))"
          ]
        },
        "area": {
          "value": "",
          "options": [
            {
              "label": "Toute la France",
              "value": ""
            },
            {
              "label": "Bas-Rhin",
              "value": "Bas-Rhin"
            },
            {
              "label": "Bouches-du-Rhône",
              "value": "Bouches-du-Rhône"
            },
            "… (12 élément(s) supplémentaire(s))"
          ]
        }
      },
      "species": {},
      "footer": {
        "source": "Source : serveur MCP lpo-data (lecture seule) · profil observations · table observations · calculé le <date de génération>"
      }
    }
  }
}
```

### 3. Résumé texte de repli

```text
Comparaison toute la période disponible : Milan royal totalise 841 observations, contre 510 pour Busard cendré (écart 331).
```

### 4. Effet côté client

- type de surface : `trend` — mode : `create` — surface : `lpo-exemple-bout-en-bout-4`
- validation : `{'ok': True, 'checked': 2, 'errors': []}`
- composants instanciés : Column, Card, Column, Text, Text, List, KpiCard, Card, LineChart, Card, DataTable, List, Notice, Card, Column, Text, Row, SelectField, SelectField, SelectField, Text

## Tour 7 — Fiche espèce du Héron cendré

**Question** : « Fiche espèce du Héron cendré »

### 1. Appels MCP

| Outil | Arguments | État | Durée | Résumé renvoyé |
| --- | --- | --- | --- | --- |
| `get_species_info` | `{"species": "Héron cendré"}` | ok | < 100 ms | Héron cendré : 938 observation(s), 3 048 individu(s), de 2016-01-15 à 2026-10-30, 10 département(s) concerné(s). |

### 2. Messages A2UI (`agent → renderer`)

```json
{
  "version": "1.0",
  "deleteSurface": {
    "surfaceId": "lpo-exemple-bout-en-bout-4"
  }
}
```

```json
{
  "version": "1.0",
  "createSurface": {
    "surfaceId": "lpo-exemple-bout-en-bout-5",
    "catalogId": "https://lpo-data-ai.example/a2ui/catalogs/lpo/v1/catalog.json",
    "sendDataModel": true,
    "components": [
      {
        "id": "root",
        "component": "Column",
        "children": [
          "header",
          "kpis",
          "species_card",
          "… (3 élément(s) supplémentaire(s))"
        ]
      },
      {
        "id": "header",
        "component": "Card",
        "child": "header_col"
      },
      {
        "id": "header_col",
        "component": "Column",
        "children": [
          "header_title",
          "header_subtitle"
        ]
      },
      "… (15 élément(s) supplémentaire(s))"
    ],
    "dataModel": {
      "header": {
        "title": "Fiche espèce — Héron cendré",
        "subtitle": "toute la période disponible · France entière"
      },
      "kpis": [
        {
          "label": "Observations",
          "value": "938",
          "hint": "toute la période disponible",
          "tone": "neutral"
        },
        {
          "label": "Individus",
          "value": "3 048",
          "hint": "somme des effectifs",
          "tone": "neutral"
        },
        {
          "label": "Départements",
          "value": "10",
          "hint": "Gironde, Finistère, Hérault",
          "tone": "neutral"
        }
      ],
      "chart": {
        "title": "",
        "series": [],
        "items": [],
        "points": [],
        "legend": [],
        "xLabel": "",
        "yLabel": "",
        "valueLabel": "",
        "privacyMode": "exact",
        "emptyMessage": ""
      },
      "table": {
        "title": "Observations par mois",
        "caption": "Série mensuelle issue du serveur MCP.",
        "rows": [
          {
            "period": "2024-01",
            "value": 8.0,
            "individuals": 15.0
          },
          {
            "period": "2024-02",
            "value": 11.0,
            "individuals": 39.0
          },
          {
            "period": "2024-03",
            "value": 11.0,
            "individuals": 52.0
          },
          "… (21 élément(s) supplémentaire(s))"
        ],
        "emptyMessage": "Aucune série mensuelle disponible."
      },
      "notices": [
        {
          "tone": "warning",
          "title": "Jeu de démonstration",
          "message": "Jeu de démonstration synthétique (observations naturalistes) — aucune valeur écologique réelle. Les chiffres servent à valider la chaîne MCP → A2UI, pas à documenter la biodiversité."
        },
        {
          "tone": "info",
          "title": "À savoir",
          "message": "Espèce sensible : les localisations précises ne doivent pas être diffusées."
        },
        {
          "tone": "warning",
          "title": "Espèce sensible",
          "message": "La localisation précise n'est pas diffusée ; seules les échelles agrégées sont disponibles."
        }
      ],
      "filters": {
        "species": {
          "value": "Héron cendré",
          "options": [
            {
              "label": "Toutes les espèces",
              "value": ""
            },
            {
              "label": "Busard cendré",
              "value": "Busard cendré"
            },
            {
              "label": "Buse variable",
              "value": "Buse variable"
            },
            "… (12 élément(s) supplémentaire(s))"
          ]
        },
        "period": {
          "value": "all",
          "options": [
            {
              "label": "10 dernières années",
              "value": "10y"
            },
            {
              "label": "12 derniers mois",
              "value": "12m"
            },
            {
              "label": "Cette année",
              "value": "this_year"
            },
            "… (3 élément(s) supplémentaire(s))"
          ]
        },
        "area": {
          "value": "",
          "options": [
            {
              "label": "Toute la France",
              "value": ""
            },
            {
              "label": "Bas-Rhin",
              "value": "Bas-Rhin"
            },
            {
              "label": "Bouches-du-Rhône",
              "value": "Bouches-du-Rhône"
            },
            "… (12 élément(s) supplémentaire(s))"
          ]
        }
      },
      "species": {
        "name": "Héron cendré",
        "latinName": "",
        "protection": "protégée",
        "note": "Données observées de 2016-01-15 à 2026-10-30. Départements : Gironde, Finistère, Hérault, Loire-Atlantique, Côtes-d'Armor, Bas-Rhin, Charente-Maritime, Indre-et-Loire, Morbihan, Vendée."
      },
      "footer": {
        "source": "Source : serveur MCP lpo-data (lecture seule) · profil observations · table observations · calculé le <date de génération>"
      }
    }
  }
}
```

### 3. Résumé texte de repli

```text
Héron cendré : 938 observation(s), 3 048 individu(s), de 2016-01-15 à 2026-10-30, dans 10 département(s). Statut : protégée
```

### 4. Effet côté client

- type de surface : `species` — mode : `create` — surface : `lpo-exemple-bout-en-bout-5`
- validation : `{'ok': True, 'checked': 2, 'errors': []}`
- composants instanciés : Column, Card, Column, Text, Text, List, KpiCard, Card, Column, Text, Text, Text, Text, Card, DataTable, List, Notice, Text

## Tour 8 — Quel temps fera-t-il demain ?

**Question** : « Quel temps fera-t-il demain ? »

### 1. Appels MCP

_Aucun appel (question hors périmètre)._

### 2. Messages A2UI (`agent → renderer`)

```json
{
  "version": "1.0",
  "deleteSurface": {
    "surfaceId": "lpo-exemple-bout-en-bout-5"
  }
}
```

```json
{
  "version": "1.0",
  "createSurface": {
    "surfaceId": "lpo-exemple-bout-en-bout-6",
    "catalogId": "https://lpo-data-ai.example/a2ui/catalogs/lpo/v1/catalog.json",
    "sendDataModel": true,
    "components": [
      {
        "id": "root",
        "component": "Column",
        "children": [
          "header",
          "notice_list",
          "footer"
        ]
      },
      {
        "id": "header",
        "component": "Card",
        "child": "header_col"
      },
      {
        "id": "header_col",
        "component": "Column",
        "children": [
          "header_title",
          "header_subtitle"
        ]
      },
      "… (5 élément(s) supplémentaire(s))"
    ],
    "dataModel": {
      "header": {
        "title": "Question hors périmètre",
        "subtitle": "toute la période disponible · France entière"
      },
      "kpis": [
        {
          "label": "Observations",
          "value": "—",
          "hint": "Aucune donnée",
          "trend": "",
          "tone": "neutral"
        }
      ],
      "chart": {
        "title": "",
        "series": [],
        "items": [],
        "points": [],
        "legend": [],
        "xLabel": "",
        "yLabel": "",
        "valueLabel": "",
        "privacyMode": "exact",
        "emptyMessage": ""
      },
      "table": {
        "title": "",
        "caption": "",
        "rows": [],
        "emptyMessage": "Aucune ligne à afficher."
      },
      "notices": [
        {
          "tone": "info",
          "title": "Aucune donnée à afficher",
          "message": "Cette question ne porte pas sur les observations naturalistes exposées par le serveur MCP. Je ne peux pas y répondre sans inventer de données. Essayez par exemple : « évolution des observations de Milan royal depuis 2016 », « top 10 des espèces en Gironde en 2025 » ou « carte du Busard cendré ce printemps »."
        }
      ],
      "filters": {
        "species": {
          "value": "",
          "options": []
        },
        "period": {
          "value": "",
          "options": []
        },
        "area": {
          "value": "",
          "options": []
        }
      },
      "species": {},
      "footer": {
        "source": "Source : serveur MCP lpo-data (lecture seule) · profil observations · table ? · calculé à la demande"
      }
    }
  }
}
```

### 3. Résumé texte de repli

```text
Cette question ne porte pas sur les observations naturalistes exposées par le serveur MCP. Je ne peux pas y répondre sans inventer de données. Essayez par exemple : « évolution des observations de Milan royal depuis 2016 », « top 10 des espèces en Gironde en 2025 » ou « carte du Busard cendré ce printemps ». Question reformulée : Quel temps fera-t-il demain ?
```

### 4. Effet côté client

- type de surface : `empty` — mode : `create` — surface : `lpo-exemple-bout-en-bout-6`
- validation : `{'ok': True, 'checked': 2, 'errors': []}`
- composants instanciés : Column, Card, Column, Text, Text, List, Notice, Text
