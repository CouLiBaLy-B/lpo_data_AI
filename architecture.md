# Architecture — LPO AI Assistant SQL

Application Streamlit qui répond à des questions métier en langage naturel sur
une base SQL, via un **agent deepagents (LangChain)** qui découvre le schéma
dynamiquement. Indépendante du provider LLM et du moteur de base de données.

## Composants deepagents utilisés (0.6.11)

- **Outils d'introspection** : `list_tables`, `describe_tables`, `check_sql`,
  `run_sql_query` — le schéma est découvert, jamais codé dans le prompt.
- **Middleware** : `SQLAuditMiddleware` (audit + garde lecture seule au niveau
  agent, en plus de la garde DB) — s'ajoute au stack par défaut (planning, etc.).
- **Backend** : défaut `StateBackend` (filesystem virtuel en mémoire, zéro disque)
  — adapté à Streamlit multi-sessions.
- **Versioning de prompt** : `system_sql.md` + skill `sql-lpo/SKILL.md` versionnés
  git, chargés en progressive disclosure (`skills=["/skills/"]` + `invoke(files=…)`).
- **Streaming** : `stream_mode="updates"` → étapes émises au fil de l'eau (un seul
  passage de graphe), affichées en direct dans l'UI.

## Principes

1. **Schéma jamais dans le prompt** — l'agent explore la base avec des outils
   (`list_tables` → `describe_tables` → `check_sql` → `run_sql_query`). Il
   fonctionne donc sur n'importe quelle base, **mono ou multi-tables** (JOINs
   construits à partir des clés étrangères découvertes).
2. **Lecture seule** — seules les requêtes `SELECT`/`WITH` passent ; toute
   mutation/DDL est rejetée (la donnée prod est alimentée par un système externe).
3. **Provider-agnostic** — `model_factory` via `init_chat_model`. Changer de LLM =
   `AI_PROVIDER` + `AI_MODEL` dans `.env`.
4. **DB-agnostic** — un seul moteur SQLAlchemy piloté par `DATABASE_URL`
   (SQLite en dev, Postgres en prod).
5. **Cleaning à la lecture** — normalisation générique du DataFrame (trim, nulls,
   types, dates, dédup), sans hypothèse sur les noms de colonnes.
6. **Visualisation déterministe** — figure Plotly construite d'après la forme des
   données. Pas d'`exec()` de code généré.

## Flux d'une question

```text
Question (FR)
   │
   ▼
Agent deepagents ── list_tables → describe_tables → check_sql → run_sql_query
   │                              (auto-correction si erreur SQL)
   ▼
SQL final ──► ré-exécution lecture seule ──► cleaning ──► viz_engine (Plotly)
   │
   ▼
Réponse FR + graphique + données + raisonnement repliable
```

## Structure

```text
main.py                      # entrée Streamlit (auth + navigation)
src/
├── config/settings.py       # config typée (.env) : DB, IA, sécurité
├── core/
│   ├── models.py            # modèles Pydantic (ChartType, ...)
│   ├── exceptions.py        # exceptions applicatives
│   └── security.py          # auth JWT + anti-bruteforce
├── database/
│   └── engine.py            # moteur SQLAlchemy, introspection, garde lecture seule
├── ai/
│   ├── model_factory.py     # création LLM provider-agnostique (init_chat_model)
│   ├── tools.py             # outils agent : list_tables/describe_tables/check_sql/run_sql_query
│   ├── middleware.py        # SQLAuditMiddleware (audit + garde lecture seule)
│   ├── agent.py             # assemblage deepagents + run()/stream_run() → {answer, sql, steps}
│   ├── cleaning.py          # cleaning générique à la lecture
│   ├── viz_engine.py        # visualisation Plotly déterministe
│   └── prompts/             # system_sql.md + skills/sql-lpo/SKILL.md (versionnés git)
└── ui/pages/
    ├── dashboard.py         # tableau de bord résilient au schéma
    └── query_interface.py   # chat (réponse + graphe + raisonnement repliable)

data_retrieve/               # ETL biodiv-sports → SQLite (DEV uniquement)
```

## Configuration (.env)

```bash
DATABASE_URL=sqlite:///data/sensitive_areas.db      # dev
# DATABASE_URL=postgresql://user:pwd@host:5432/db   # prod
DB_READ_ONLY=true

AI_PROVIDER=anthropic            # anthropic | openai | google_genai | ollama ...
AI_MODEL=claude-haiku-4-5-20251001
ANTHROPIC_API_KEY=...            # clé du provider actif
```

## Lancement

```bash
uv sync
uv run streamlit run main.py
```
