---
title: Lpo Ai
emoji: 👁
colorFrom: pink
colorTo: green
sdk: streamlit
sdk_version: 1.58.0
app_file: main.py
pinned: false
---

# LPO AI — Assistant SQL

Application Streamlit qui répond à des questions métier en **langage naturel** sur
une base SQL. Un **agent deepagents (LangChain)** explore le schéma de la base
**dynamiquement** (il ne le connaît pas à l'avance), génère le SQL en lecture
seule, s'auto-corrige, puis l'application produit une visualisation interactive.

![aperçu](image.png)

## Points clés

- 🧠 **Agent deepagents** — découvre les tables et colonnes via des outils
  (`list_tables`, `describe_tables`, `check_sql`, `run_sql_query`). Le schéma
  n'est **jamais** codé dans le prompt → fonctionne sur n'importe quelle base,
  **mono ou multi-tables** (JOINs automatiques).
- 🔌 **Indépendant du provider** — `model_factory` via `init_chat_model`.
  Changer de LLM = modifier `AI_PROVIDER` / `AI_MODEL` dans `.env`
  (Anthropic, OpenAI, Google, Ollama…).
- 🗄️ **Indépendant du moteur DB** — SQLAlchemy + `DATABASE_URL`
  (**SQLite** en dev, **Postgres** en prod).
- 🔒 **Lecture seule** — toute requête de modification (INSERT/UPDATE/DELETE/DDL)
  est rejetée.
- 🧹 **Cleaning à la lecture** — normalisation générique (trim, valeurs nulles,
  types, dates, déduplication).
- 📊 **Visualisation déterministe** — figure Plotly choisie d'après la forme des
  données (pas d'exécution de code généré).

## Prérequis

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (gestion des dépendances)

## Installation

```bash
uv sync
```

Créer un fichier `.env` (voir `.env` fourni) :

```bash
# Authentification app
USERNAME=admin
PASSWORD=admin
SECRET_KEY=change_me

# Base de données
DATABASE_URL=sqlite:///data/sensitive_areas.db      # dev
# DATABASE_URL=postgresql://user:pwd@host:5432/db   # prod
DB_READ_ONLY=true

# IA (changement de provider ici)
AI_PROVIDER=anthropic
AI_MODEL=claude-haiku-4-5-20251001
ANTHROPIC_API_KEY=...
```

## Lancement

```bash
uv run streamlit run main.py
```

## Structure

| Élément | Rôle |
| --- | --- |
| `main.py` | Entrée Streamlit : authentification + navigation |
| `src/ai/agent.py` | Assemblage de l'agent deepagents + `run()` |
| `src/ai/tools.py` | Outils d'introspection schema-agnostic (lecture seule) |
| `src/ai/model_factory.py` | Création du LLM, provider-agnostique |
| `src/ai/cleaning.py` | Nettoyage générique des résultats |
| `src/ai/viz_engine.py` | Visualisation Plotly déterministe |
| `src/database/engine.py` | Moteur SQLAlchemy + introspection + garde lecture seule |
| `src/ui/pages/` | Pages Streamlit (dashboard, chat) |
| `data_retrieve/` | ETL biodiv-sports → SQLite (**dev uniquement**) |

Voir [architecture.md](architecture.md) pour le détail.

---

# Chaîne conversationnelle MCP + A2UI (nouvelle interface)

En plus de l'application Streamlit ci-dessus, le dépôt expose une chaîne
complète **question en langage naturel → outils MCP → interface graphique
A2UI** :

```
[Utilisateur] → [Client avec renderer A2UI] ⇄ [Agent orchestrateur] ⇄ [Serveur MCP lpo-data]
```

- L'agent ne fabrique **aucun chiffre** : toutes les valeurs viennent du serveur
  MCP, mis en forme (jamais recalculées).
- L'interface est décrite en **messages A2UI v1.0** validés contre le catalogue
  avant envoi. Le client n'exécute **ni HTML, ni CSS, ni JS** fourni par le
  serveur.
- Les filtres (espèce, période, zone) remontent à l'agent, qui rappelle le
  serveur MCP puis ne renvoie qu'un **`updateDataModel`** : la surface n'est pas
  recréée.
- Chaque réponse comporte un **résumé texte de repli** (clients sans renderer).

## 1. Installation

```bash
uv sync                      # ou : python -m venv .venv && .venv/bin/pip install -e .
```

Dépendances ajoutées : `mcp`, `a2ui-core`, `a2ui-agent-sdk`, `fastapi`,
`uvicorn`, `httpx` (déclarées dans `pyproject.toml`).

## 2. Base de démonstration

Le dépôt ne contient pas de table d'observations (voir
[docs/00-analyse-depot.md](docs/00-analyse-depot.md)) : une base de
démonstration **explicitement étiquetée** est générée localement.

```bash
python scripts/seed_demo_db.py --rows 12000
# → data/lpo_demo.db (12 000 observations, 2016-01-01 → 2026-10-31)
```

La base réelle du dépôt (`data/sensitive_areas.db`, 169 lignes, sans
coordonnées ni effectifs) reste utilisable via un autre **profil** ; l'interface
dégrade alors explicitement au lieu d'inventer.

## 3. Lancer le serveur MCP seul

```bash
# stdio (défaut, utilisé par l'agent local)
LPO_PROFILE=observations DATABASE_URL=sqlite:///$(pwd)/data/lpo_demo.db \
  python -m lpo_data_mcp

# streamable-http (service réseau)
LPO_PROFILE=observations DATABASE_URL=sqlite:///$(pwd)/data/lpo_demo.db \
  python -m lpo_data_mcp --transport streamable-http --port 8765
```

Outils exposés (lecture seule, sorties structurées + résumé texte) :
`get_dataset_overview`, `list_species`, `get_observations`,
`aggregate_observations`, `get_species_info`.

Inspection rapide avec l'inspecteur MCP officiel :

```bash
npx @modelcontextprotocol/inspector python -m lpo_data_mcp
```

## 4. Lancer l'agent en ligne de commande

```bash
LPO_PROFILE=observations DATABASE_URL=sqlite:///$(pwd)/data/lpo_demo.db \
  python -m orchestrator
```

L'agent démarre lui-même le serveur MCP en `stdio` (`LPO_MCP_COMMAND`) ou se
connecte à un serveur HTTP existant (`LPO_MCP_URL`).

## 5. Lancer l'interface web (client A2UI)

```bash
LPO_PROFILE=observations DATABASE_URL=sqlite:///$(pwd)/data/lpo_demo.db \
  LPO_LLM_INTENT=off \
  .venv/bin/python -m uvicorn web.app:app --host 0.0.0.0 --port 8080
```

Puis ouvrir <http://localhost:8080>. C'est un **renderer A2UI** : il reçoit des
messages JSON validés (`/api/ask`, `/api/action`) et construit le DOM avec des
API sûres (`textContent`, SVG). Le catalogue est servi par `/api/catalog`.

| Variable | Effet |
| --- | --- |
| `LPO_PROFILE` | `observations` (défaut) ou `sensitive_areas` (base du dépôt) |
| `DATABASE_URL` | base interrogée (lecture seule) |
| `LPO_LLM_INTENT` | `auto` (défaut) / `off` — extraction d'intention par LLM ou heuristique |
| `LPO_MCP_URL` | URL d'un serveur MCP `streamable-http` existant |
| `LPO_MCP_COMMAND` | commande du serveur MCP `stdio` (défaut : `python -m lpo_data_mcp`) |
| `LPO_PRIVACY_MODE` | `auto` (défaut) / `exact` / `blurred` / `department` |
| `LPO_BLUR_DECIMALS` | mailles de floutage : `1` (défaut, ~11 km) à `0` (~111 km) |
| `LPO_ALLOW_EXACT_COORDINATES` | `1` pour autoriser les coordonnées exactes d'espèces sensibles |
| `LPO_SENSITIVE_SPECIES` | liste des espèces sensibles, séparées par des virgules |
| `LPO_SENSITIVE_SPECIES_FILE` | même liste dans un fichier (un libellé par ligne, `#` = commentaire) |
| `LPO_WEB_TOKEN` | jeton bearer exigé par l'API (laisser vide = API ouverte, mode démo) |

## 6. Tests

```bash
.venv/bin/python -m pytest            # 201 tests
```

Afficher le prompt système et les exemples few-shot utilisés par l'agent :

```bash
python -m orchestrator --prompt
```

Couverture : garde-fous SQL, confidentialité (floutage), contrat des outils MCP,
serveur MCP réel en `stdio`, client MCP de production sur transport réel,
catalogue et gabarits A2UI, compréhension de l'intention, orchestrateur
(create/update, dégradation, états vides), prompt système et few-shots A2UI,
chemin LLM de l'intention (faux modèle), liste de sensibilité configurée,
transport MCP `streamable-http` réel, jeton d'accès de l'API, script d'index,
API web, et **chaîne complète** (HTTP → agent → sous-processus MCP → SQLite) via
`tests/test_web_e2e.py`.

## 6 bis. Index de performance (base réelle)

```bash
python scripts/create_indexes.py --dry-run   # affiche le SQL
python scripts/create_indexes.py             # crée les index manquants
```

Idempotent, non destructif, adapté au profil détecté (`(species, event_date)`,
`(department, region)`, `(species_id)`, …). Sans effet sur la base de
démonstration, décisif sur une table de plusieurs millions de lignes.

## 6 ter. Protéger l'API et configurer la sensibilité

```bash
# API protégée par jeton (le client accepte /?token=… une seule fois)
LPO_WEB_TOKEN="$(openssl rand -hex 24)" uvicorn web.app:app --port 8080

# Liste officielle des espèces sensibles, sans redémarrage
LPO_SENSITIVE_SPECIES="Milan royal,Busard cendré,Aigle royal" \
  python -m lpo_data_mcp
```

Détails (transport HTTP en production, reverse-proxy, volumétrie, journaux) :
[docs/02-exploitation.md](docs/02-exploitation.md).

## 7. Exemple bout en bout

```bash
python -m lpo_a2ui.examples.build_transcript   # → lpo_a2ui/examples/transcript.md
```

La trace contient, pour chaque question : les **appels MCP** (outil, arguments,
durée, résumé), les **messages A2UI** réellement validés, le **résumé texte** et
l'effet côté client. Le rendu DOM correspondant est dans
`lpo_a2ui/examples/rendered.html` (voir le README du dossier pour le
régénérer).

## 8. Structure ajoutée

| Élément | Rôle |
| --- | --- |
| `lpo_data_mcp/` | Serveur MCP : outils typés, SQL paramétré, profils, confidentialité |
| `lpo_a2ui/catalog/` | Catalogue A2UI v1 (14 composants) + schémas vendorisés |
| `lpo_a2ui/validation.py` | Validation schéma + catalogue + séquences avant envoi |
| `lpo_a2ui/templates.py` | Gabarits des surfaces (trend, ranking, map, species, empty) |
| `orchestrator/intent.py` | Compréhension de l'intention (LLM ou heuristique) |
| `orchestrator/mcp_client.py` | Client MCP (stdio / streamable-http) |
| `orchestrator/views.py` | Données MCP → modèle de vue (aucun calcul métier) |
| `orchestrator/agent.py` | Boucle question → outils → surface, actions, trace |
| `orchestrator/prompts.py` | Prompt système + 6 few-shots A2UI valides (spécification exécutable) |
| `web/` | API FastAPI + client A2UI (`renderer.js`, `app.js`, `styles.css`) |
| `lpo_a2ui/examples/` | Trace bout en bout + rendu DOM |
| `docs/00-analyse-depot.md` | Rapport d'analyse (Étape 0) et questions ouvertes |
| `docs/01-architecture.md` | Architecture, choix techniques, conformité A2UI v1.0 |
| `docs/02-exploitation.md` | Configuration, déploiement, confidentialité, volumétrie, limites |
| `scripts/create_indexes.py` | Index recommandés pour une base réelle (idempotent) |

## 9. Compatibilité

L'application Streamlit d'origine (`uv run streamlit run main.py`) continue de
fonctionner : la seule modification apportée au code existant est l'ajout de
`run_select_params()` dans `src/database/engine.py` (mêmes garde-fous, requêtes
paramétrées).
