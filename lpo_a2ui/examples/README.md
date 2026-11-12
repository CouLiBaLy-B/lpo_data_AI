# Exemple bout en bout (livrable 6)

Ce dossier prouve la chaîne complète **question → appels MCP → messages A2UI →
rendu**, avec des données réelles et sans intervention manuelle.

| Fichier | Contenu | Généré par |
| --- | --- | --- |
| `transcript.md` | trace lisible : question, appels MCP (outil, arguments, durée, résumé renvoyé), messages A2UI, résumé texte, effet client | `build_transcript.py` |
| `turns.json` | mêmes tours au format machine (messages complets), consommé par le rendu | `build_transcript.py` |
| `rendered.html` | page autonome : chaque surface effectivement construite par le renderer, avec le texte de repli | `render_snapshot.mjs` |
| `rendered.report.json` | contrôle du rendu (composants, nœuds DOM, erreurs) | `render_snapshot.mjs` |

## Régénérer la trace

```bash
python scripts/seed_demo_db.py --rows 12000          # si data/lpo_demo.db est absent
DATABASE_URL=sqlite:///$(pwd)/data/lpo_demo.db \
LPO_PROFILE=observations LPO_LLM_INTENT=off \
  python -m lpo_a2ui.examples.build_transcript
```

Le script utilise le **vrai** `LpoAssistant` et le **vrai** validateur A2UI ; il
n'écrit rien qui ne sorte du pipeline.

## Régénérer le rendu DOM

Le rendu s'appuie sur le renderer de production (`web/static/renderer.js`)
exécuté dans `jsdom` — c'est le même code que dans le navigateur.

```bash
mkdir -p .snapshot && cd .snapshot && npm init -y >/dev/null && npm i jsdom && cd ..
NODE_PATH=$PWD/.snapshot/node_modules node lpo_a2ui/examples/render_snapshot.mjs
```

Sans `NODE_PATH`, on peut créer un lien symbolique temporaire
`lpo_a2ui/examples/node_modules` vers un dossier contenant `jsdom`.

## Scénario couvert

1. **Tendance** — évolution d'une espèce sur 10 ans (`LineChart`, 3 KPI).
2. **Classement** — top 10 d'un département (`BarChart`, `DataTable`).
3. **Interaction** — filtre « zone » → Gironde : `updateDataModel` seul, la
   surface de classement **n'est pas recréée** (`mode: update`).
4. **Interaction** — filtre « période » → 12 derniers mois : idem, sans nouvelle
   structure.
5. **Carte** — observations au printemps, points floutés + `Notice` de
   confidentialité (`MapChart`).
6. **Comparaison** — deux espèces par mois (deux séries dans le même
   `LineChart`).
7. **Fiche espèce** — `get_species_info` (`KpiCard`, `Notice`).
8. **Hors périmètre** — « Quel temps fera-t-il demain ? » : **aucun appel
   d'agrégation**, surface `empty` explicite, texte qui dit qu'aucune donnée ne
   sera inventée.
