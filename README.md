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
  (**SQLite** en dev, **PostgreSQL** en prod : DSN normalisé, rôle en lecture
  seule, `search_path` borné, timeouts posés à chaque connexion).
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

# Base de données — dev SQLite / prod PostgreSQL
DATABASE_URL=sqlite:///data/sensitive_areas.db      # dev
# DATABASE_URL=postgresql://lpo_ro:pwd@db.interne:5432/lpo   # prod
# DB_SCHEMA=lpo                 # schéma métier (search_path)
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
dégrade alors explicitement au lieu d'inventer. Les questions y portent sur les
**zones**, pas sur les espèces : le classement suit l'axe demandé
(`department`, `region`, `category`) et les textes parlent d'« enregistrements ».
Exemples vérifiés sur l'extrait : « combien de zones en Haute-Savoie ? » → 28 ;
« classement des zones par catégorie » → 131 « zone reglementaire » / 38 « Espece ».

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

Le bouton **Configuration** de l'en-tête permet de changer, sans redémarrage :

- le **fournisseur de modèle** (Anthropic, OpenAI, Gemini, Mistral, Groq, Ollama),
  le modèle et sa clé (écrite en mémoire seulement, jamais renvoyée), ainsi que
  le mode d'extraction d'intention (`auto` / `on` / `off`) ;
- le **serveur MCP** : URL `streamable-http` ou commande locale `stdio`, avec un
  bouton « Tester la connexion » qui affiche les outils et le jeu de données
  réellement atteints avant d'enregistrer.

Une cible MCP qui ne répond pas n'est jamais adoptée (les réglages précédents
sont conservés), et un changement de cible réinitialise la conversation : ses
surfaces décrivaient l'ancien jeu de données.

| Variable | Effet |
| --- | --- |
| `LPO_PROFILE` | `observations` (défaut) ou `sensitive_areas` (base du dépôt) |
| `DATABASE_URL` | base interrogée (lecture seule) |
| `LPO_LLM_INTENT` | `auto` (défaut) / `off` — extraction d'intention par LLM ou heuristique |
| `LPO_MCP_URL` | URL d'un serveur MCP `streamable-http` existant |
| `LPO_MCP_COMMAND` | commande du serveur MCP `stdio` (défaut : `python -m lpo_data_mcp`) |
| `LPO_MCP_ARGS` | arguments de cette commande (défaut : `-m lpo_data_mcp`) |
| `LPO_WEB_SETTINGS` | `on` (défaut) / `url-only` / `off` — ce que l'interface peut modifier |
| `LPO_PRIVACY_MODE` | `auto` (défaut) / `exact` / `blurred` / `department` |
| `LPO_BLUR_DECIMALS` | mailles de floutage : `1` (défaut, ~11 km) à `0` (~111 km) |
| `LPO_ALLOW_EXACT_COORDINATES` | `1` pour autoriser les coordonnées exactes d'espèces sensibles |
| `LPO_SENSITIVE_SPECIES` | liste des espèces sensibles, séparées par des virgules |
| `LPO_SENSITIVE_SPECIES_FILE` | même liste dans un fichier (un libellé par ligne, `#` = commentaire) |
| `LPO_WEB_TOKEN` | jeton bearer exigé par l'API (laisser vide = API ouverte, mode démo) |
| `DB_SCHEMA` | schéma PostgreSQL exposé (`search_path` borné ; sans effet sur SQLite) |
| `DB_APPLICATION_NAME` | nom lisible dans `pg_stat_activity` (défaut `lpo-data-ai`) |
| `DB_STATEMENT_TIMEOUT_MS` | coupure serveur des requêtes lentes (défaut 15 000) |
| `DB_LOCK_TIMEOUT_MS` | attente maximale d'un verrou (défaut 3 000) |
| `DB_POOL_SIZE` / `DB_MAX_OVERFLOW` | pool PostgreSQL par processus (5 / 5) |

## 6. Tests

```bash
.venv/bin/python -m pytest            # 362 tests (1 ignoré : intégration PostgreSQL)
```

Afficher le prompt système et les exemples few-shot utilisés par l'agent :

```bash
python -m orchestrator --prompt
```

Le workflow CI proposé (`docs/ci/python-app.yml`) exécute `ruff check .` puis
`pytest` sur Python 3.11, sans clé LLM ni base externe. Il n'a pas pu être
installé automatiquement (permission `workflows` manquante sur le jeton de
push) : voir `docs/ci/README.md`, qui documente aussi les deux défauts du
workflow actuel (Python 3.10 et affichage d'une clé API dans les journaux).

Couverture : garde-fous SQL, confidentialité (floutage), contrat des outils MCP,
serveur MCP réel en `stdio`, client MCP de production sur transport réel,
catalogue et gabarits A2UI, compréhension de l'intention, orchestrateur,
qualité de l'interface (contrastes AA, focus, états vides)
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

## 6 quater. PostgreSQL en production

Le détail (rôle, garde-fous, diagnostic) est dans
[`docs/02-exploitation.md`](docs/02-exploitation.md) §1. Résumé :

```bash
DATABASE_URL=postgresql://lpo_ro:motdepasse@db.interne:5432/lpo \
DB_SCHEMA=lpo \
DB_STATEMENT_TIMEOUT_MS=15000 \
  python -m lpo_data_mcp --transport streamable-http --port 8765
```

- `postgres://` et `postgresql://` sont normalisés vers psycopg 3
  (`postgresql+psycopg://`), seul pilote déclaré — `src/database/postgres.py`.
- Chaque connexion applique `statement_timeout`, `lock_timeout`,
  `idle_in_transaction_session_timeout`, `TIME ZONE 'UTC'` et, en lecture seule,
  `default_transaction_read_only = on`.
- Ce dernier réglage est un **garde-fou**, pas une frontière : le rôle
  PostgreSQL doit n'avoir que `SELECT` (SQL de création du rôle dans la doc).
- `DB_SCHEMA` borne le `search_path` à `lpo,public` : l'introspection ne voit pas
  les autres applications du cluster.
- Les index se créent en `CONCURRENTLY` (table disponible pendant l'indexation),
  avec détection des index laissés invalides par une interruption.

Le schéma PostgreSQL attendu est fourni dans
[`docs/sql/sensitive_areas_postgres.sql`](docs/sql/sensitive_areas_postgres.sql),
dérivé de l'extrait réel `data/sensitive_areas.db` (169 lignes, 50 zones,
131 lignes sans espèce) : types, commentaires, index `CONCURRENTLY` aux mêmes
noms que le script d'indexation, exemple de chargement.

Les tests unitaires de ce support (`tests/test_postgres_support.py`, 37 tests)
tournent **sans serveur** ; le test d'intégration s'active avec
`LPO_TEST_PG_URL=postgresql://…` et vérifie, sur une vraie base, que la session
refuse l'écriture et que les timeouts sont posés.

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
| `src/database/postgres.py` | Règles PostgreSQL : DSN, `search_path`, garde-fous de session |
| `scripts/create_indexes.py` | Index recommandés pour une base réelle (idempotent, `CONCURRENTLY` en prod) |

## 9. Compatibilité

L'application Streamlit d'origine (`uv run streamlit run main.py`) continue de
fonctionner : la seule modification apportée au code existant est l'ajout de
`run_select_params()` dans `src/database/engine.py` (mêmes garde-fous, requêtes
paramétrées).
