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
