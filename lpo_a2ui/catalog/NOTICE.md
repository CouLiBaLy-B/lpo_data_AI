# Catalogues A2UI vendorisés

| Fichier | Source | Version |
| --- | --- | --- |
| `basic_catalog_v1.json` | [a2ui-project/a2ui](https://github.com/a2ui-project/a2ui) — `catalogs/basic/v1/catalog.json` | commit `8bf644a37d680a227296b8c0849a7f4b328b8b73` |
| `common_types.json` | idem — `specification/v1_0/json/common_types.json` | même commit |
| `agent_to_renderer.json` | idem — `specification/v1_0/json/agent_to_renderer.json` | même commit |

Licence Apache-2.0 (voir le dépôt amont). Ces copies servent de **référence et de
repli de validation** ; la validation de référence passe par le SDK officiel
`a2ui-agent-sdk` (voir `lpo_a2ui/validation.py`).

`lpo_catalog.json` est notre catalogue personnalisé (composants de base retenus +
composants métier `KpiCard`, `LineChart`, `BarChart`, `MapChart`, `DataTable`,
`Notice`, `SelectField`).
