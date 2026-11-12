# Étape 0 — Analyse du dépôt `lpo_data_AI`

> **Règle appliquée** : aucune hypothèse. Chaque affirmation ci-dessous renvoie à
> un fichier du dépôt ou à une requête exécutée sur la base livrée ; tout ce qui
> n'a pas pu être établi est listé en fin de document sous **Questions ouvertes**.

---

## 1. Objectif visé

Permettre à un utilisateur de poser une question en langage naturel sur des
observations naturalistes et de recevoir une réponse **graphique et
interactive**, rendue via **A2UI**, les données étant exposées par un
**serveur MCP** :

```
[Utilisateur] → [Client avec renderer A2UI] ⇄ [Agent orchestrateur] ⇄ [Serveur(s) MCP de données]
```

L'état de départ du dépôt ne contient ni serveur MCP, ni agent orchestrateur
« données → interface », ni couche A2UI. Ce rapport établit ce qui existe, ce
qui est réutilisable, et ce qui manque.

---

## 2. Stack actuelle

| Élément | Constat | Source |
|---|---|---|
| Langage | Python ≥ 3.11 ; `requires-python = ">=3.11"` | `pyproject.toml` |
| Gestionnaire | `uv` (`uv.lock` présent) + `requirements.txt` en doublon | `pyproject.toml`, `requirements.txt`, `uv.lock` |
| Front | **Streamlit** (`main.py`, pages `src/ui/pages/`) | `main.py`, `src/ui/pages/dashboard.py`, `src/ui/pages/query_interface.py` |
| Agent | `deepagents` + `langchain` (Anthropic & OpenAI) | `src/ai/agent.py`, `src/ai/model_factory.py` |
| Accès données | SQLAlchemy 2 (`create_engine`), pandas | `src/database/engine.py` |
| Visualisation | Plotly (bar / line / scatter), décision de type dans `src/ai/viz_engine.py` | `src/ai/viz_engine.py`, `src/core/models.py` |
| Config | `pydantic-settings` (`DatabaseSettings`, `AISettings`, `SecuritySettings`, `AppSettings`) | `src/config/settings.py` |
| Auth | JWT maison + hachage de mot de passe | `src/core/security.py` |
| Tests / lint | `pytest`, `ruff` déclarés **sans aucun test dans le dépôt d'origine** | `pyproject.toml` |
| CI | workflow GitHub Actions | `.github/workflows/` |
| Base livrée | SQLite `data/sensitive_areas.db` (169 lignes) | voir §3 |

Volume de code métier d'origine : ~2 100 lignes Python (`main.py` + `src/` +
`data_retrieve/`).

---

## 3. Inventaire des sources de données

### 3.1 Base réellement livrée : `data/sensitive_areas.db`

```
sensitive_areas (169 lignes)
  create_datetime, id, description, name, structure, species_id,
  practices, update_datetime, region, department, Pays, category
```

Constats mesurés (requêtes `PRAGMA table_info` + `COUNT(*)`) :

- **169 lignes**, 12 colonnes, aucune clé étrangère.
- **Pas de latitude/longitude** → impossible de produire une carte de points.
- **Pas d'effectifs** (`count`) ni de dates d'observation (`event_date`) : ce
  n'est pas une table d'observations mais un **référentiel de zones sensibles**
  (nom, structure porteuse, pratiques, catégorie, département/région).
- `species_id` est numérique **et parfois NULL** (typé `REAL` côté SQLite car
  du texte y a été inséré : une valeur remonte en `nan` sans cast explicite).
- Le fichier contient aussi des résidus `sensitive_areas 2.db-shm`, `3.db-wal`
  versionnés dans `data/` (artefacts d'un checkpoint SQLite).

**Conséquence** : ce schéma ne permet de répondre à *aucune* des quatre
questions cibles (évolution, top 10, carte, comparaison). Il permet en revanche
de répondre à des questions de type « quelles zones sensibles en Gironde ? ».

### 3.2 Base de démonstration générée : `data/lpo_demo.db`

Faute de table d'observations dans le dépôt, une base de démonstration
**explicitement étiquetée comme telle** a été produite par
`scripts/seed_demo_db.py` (déterministe, graine `20261112`) :

```
observations (12 000 lignes)
  id, species, latin_name, event_date, count, lat, lon,
  department, region, protection_status, sensitive, source
dataset_metadata (7 lignes : key, value)
```

- Période couverte : **2016-01-01 → 2026-10-31**.
- 12 espèces, 8 départements, effectifs renseignés, statut de protection et
  indicateur `sensitive` par ligne, coordonnées réelles (donc floutables).
- `dataset_metadata` porte la provenance (`dataset_label`, `is_demo`) afin que
  l'interface **annonce toujours** qu'il s'agit de données de démonstration.

Ce point est un **choix de conception** : il est préférable d'exposer une base
de démonstration clairement identifiée, avec le bon schéma, plutôt que de
bloquer la démonstration. Le mécanisme de **profils** (§5.1) permet de basculer
sur les données réelles sans toucher au code lorsque celles-ci seront fournies.

### 3.3 Base explorée par le pipeline existant : `.env` / `DATABASE_URL`

`src/config/settings.py` lit `DATABASE_URL` (PostgreSQL par défaut via
`psycopg2`), et `.env` est présent dans le dépôt. Aucune table d'observations
n'y est décrite dans le dépôt ; l'accès reste piloté par le LLM via SQL libre.

---

## 4. Logique existante réutilisable

| Brique existante | Emplacement | Réutilisation retenue |
|---|---|---|
| `assert_read_only(sql)` | `src/database/engine.py` | **Réutilisé tel quel** par la couche SQL du serveur MCP (refus des `INSERT/UPDATE/DDL`, commentaires neutralisés). |
| `run_select(sql, max_rows)` | `idem` | Étendu en **`run_select_params(sql, params, max_rows)`** : mêmes garde-fous + requêtes **paramétrées** (nouvelle signature rétrocompatible). |
| `_validate_identifier`, `list_tables`, `get_schema`, `schema_overview` | `idem` | Réutilisés pour construire le **profil de schéma** et la liste blanche d'identifiants du serveur MCP. |
| `create_model()` / `available_providers()` | `src/ai/model_factory.py` | Réutilisé comme **seul** point d'accès LLM de l'orchestrateur (extraction d'intention en sortie structurée), provider-agnostique. |
| `infer_chart_type` / `create_figure` | `src/ai/viz_engine.py` | **Logique de choix de visualisation réutilisée conceptuellement** : le mapping « données → type de graphique » alimente le choix de gabarit A2UI (`trend → LineChart`, `ranking → BarChart`, `map → MapChart`). Les figures Plotly ne sont **pas** envoyées : A2UI ne transporte pas de HTML/JS. |
| `DatabaseSettings`, `AppSettings` | `src/config/settings.py` | Source unique de configuration (URL de base, clés) ; les nouveaux modules lisent `os.getenv` compatible avec ces réglages. |
| `ChatMessage`, `QueryResult`, `ChartType` | `src/core/models.py` | Convention de nommage (Pydantic) reprise pour les modèles de sortie MCP. |
| Pages Streamlit | `src/ui/pages/` | **Conservées** (application historique) ; la nouvelle interface web A2UI est ajoutée à côté, sans régression. |

Ce qui n'a **pas** été réutilisé, volontairement :

- `src/ai/tools.py` (`run_sql_query`) et l'agent `deepagents` : ils exposent du
  **SQL libre produit par le modèle**. C'est exactement ce que la consigne
  interdit côté MCP. L'agent conservé est `deepagents`, mais la nouvelle chaîne
  passe par des **outils typés**.
- `src/core/security.py` (JWT maison) : hors périmètre de cette itération ;
  signalé comme point d'intégration futur.

---

## 5. Écarts à combler

| # | Écart | Traitement |
|---|---|---|
| 1 | Aucun serveur MCP dans le dépôt | **Lot 3** : `lpo_data_mcp/` (5 outils typés, lecture seule, stdio + streamable-http) |
| 2 | Aucune couche A2UI, donc aucun renderer | **Lot 4** : catalogue `lpo_a2ui/catalog/lpo_catalog.json` (14 composants) + renderer JS `web/static/renderer.js` |
| 3 | Aucun agent « question → outils → UI » | **Lot 5** : `orchestrator/` (intention, client MCP, vues, agent) |
| 4 | Données : pas de table d'observations | Profils de schéma (`lpo_data_mcp/profiles.py`) : `observations` (démo) et `sensitive_areas` (base réelle) ; l'interface dégrade explicitement quand une capacité manque |
| 5 | Aucune protection des espèces sensibles | `lpo_data_mcp/privacy.py` : floutage à la maille ~11 km, ou repli au département, ou exact sur autorisation explicite |
| 6 | Aucun test | **138 tests** `pytest` (privacy, SQL, service, serveur MCP réel en stdio, catalogue A2UI, intention, orchestrateur, API web) |
| 7 | Aucune trace des appels d'outils | `TraceEntry` exposée par l'agent (`trace` dans chaque réponse HTTP) |
| 8 | Le front Streamlit n'est pas un renderer A2UI | Nouvelle application web FastAPI + client A2UI (`web/`) ; Streamlit conservé |

### 5.1 Décision structurante : les profils de schéma

Le serveur MCP ne suppose **pas** un schéma unique. `lpo_data_mcp/profiles.py`
déclare, pour chaque profil :

- la table et les colonnes disponibles ;
- les **capacités** (`coordinates`, `individual_counts`, `species_names`,
  `protection`, `event_dates`, `sensitive`) ;
- comment chaque colonne se mappe sur la notion métier (espèce, date, effectif,
  lat/lon, département, région).

L'outil `get_dataset_overview` publie ce profil. L'orchestrateur **lit les
capacités** avant de construire une vue : sans coordonnées, une question
« carte » produit un **état vide explicite** (« aucune coordonnée exploitable »)
plutôt qu'une carte inventée. Ce comportement est testé
(`tests/test_orchestrator.py::test_repository_profile_yields_empty_map_not_fake_points`).

---

## 6. Questions ouvertes

Ces points **ne peuvent pas être tranchés depuis le dépôt** et nécessitent une
réponse du commanditaire ; chacun a une valeur par défaut explicite en attendant.

1. **Source réelle des observations** — où se trouve la table d'observations
   (schéma, volumétrie, colonnes, SGBD) ? *Défaut : base de démonstration
   `data/lpo_demo.db`, annoncée comme telle dans l'interface.*
2. **Référentiel taxonomique** — libellés officiels (TaxRef ? nom vernaculaire
   vs scientifique ? code espèce ?). *Défaut : `species` + `latin_name` en clair,
   vocabulaire servi par `list_species`.*
3. **Renderer A2UI cible** — Streamlit n'en est pas un. Faut-il une application
   React/Lit existante, une application mobile (Flutter/SwiftUI), ou le client
   de démonstration livré ici ? *Défaut : client web autonome
   `web/static/renderer.js` (JavaScript, sans dépendance) qui implémente le
   catalogue LPO.*
4. **Transport MCP en production** — stdio (agent local) ou streamable-http
   (service réseau) ? Authentification ? *Défaut : stdio, HTTP disponible par
   option `--transport`.*
5. **Format d'inférence A2UI** — JSON canonique validé côté agent (choix retenu)
   ou format compact « Express » encadré par `<a2ui>` ? *Défaut : JSON canonique,
   validé contre le catalogue avant émission.*
6. **Clé LLM** — aucune clé n'est fournie. L'extraction d'intention fonctionne
   donc en **heuristique déterministe** (`LPO_LLM_INTENT=off`) ; le chemin LLM
   est implémenté et testé « à blanc ». Quel provider/clé utiliser ?
7. **Politique de sensibilité** — quelles espèces considérer comme sensibles ?
   Le dépôt fournit un indicateur par ligne ; la liste officielle n'y est pas.
   *Défaut : colonne `sensitive` + `protection_status`, floutage ~11 km.*
8. **Volumétrie cible et pagination** — au-delà de quelques millions de lignes,
   l'agrégation SQL devra être complétée par des vues matérialisées. *Défaut :
   plafond de 500 lignes par outil, agrégats côté base.*
9. **Authentification / multi-tenant** — aucune notion d'utilisateur dans le
   nouveau flux. *Défaut : mono-utilisateur, conversations identifiées par un
   `conversationId` opaque.*
10. **Accessibilité** — le protocole v1.0 **refuse** l'extension `@accessibility`
    (vérifié : le SDK lève `A2uiCatalogError`). L'accessibilité repose donc sur
    le thème natif du renderer et sur le **résumé texte de repli** systématique.
