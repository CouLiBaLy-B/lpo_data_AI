"""
Outils LangChain de l'agent SQL — schema-agnostic et lecture seule.

L'agent ne reçoit JAMAIS le schéma dans son prompt : il le découvre via ces
outils. C'est ce qui rend le système indépendant du schéma et capable de
travailler sur une base multi-tables (il enchaîne list_tables → describe_tables
→ run_sql_query, et construit les JOINs nécessaires).
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from ..database import engine

# Aperçu maximum de lignes renvoyées au LLM (le rendu complet se fait côté UI)
_PREVIEW_ROWS = 20


@tool
def list_tables() -> str:
    """Liste toutes les tables et vues disponibles dans la base de données.
    À appeler EN PREMIER pour découvrir les sources de données disponibles."""
    tables = engine.list_tables()
    if not tables:
        return "Aucune table dans la base."
    return "Tables disponibles:\n" + "\n".join(f"- {t}" for t in tables)


@tool
def describe_tables(tables: str) -> str:
    """Donne le schéma détaillé d'une ou plusieurs tables : colonnes, types,
    clés primaires, clés étrangères (pour construire les JOINs) et lignes d'exemple.
    `tables` est une liste de noms séparés par des virgules, ex: "commandes, clients".
    À appeler avant d'écrire une requête, pour connaître les vrais noms de colonnes."""
    names = [t.strip() for t in tables.split(",") if t.strip()]
    if not names:
        return "Préciser au moins un nom de table."

    blocks: list[str] = []
    for name in names:
        try:
            schema = engine.get_schema(name)
        except Exception as e:  # noqa: BLE001
            blocks.append(f"[{name}] erreur: {e}")
            continue

        cols = "\n".join(
            f"  - {c['name']}: {c['type']}" + ("" if c["nullable"] else " NOT NULL")
            for c in schema["columns"]
        )
        parts = [f"TABLE {name}", cols]
        if schema["primary_key"]:
            parts.append(f"  PK: {', '.join(schema['primary_key'])}")
        for fk in schema["foreign_keys"]:
            parts.append(f"  FK: {', '.join(fk['columns'])} → {fk['references']}")

        try:
            sample = engine.sample_rows(name, limit=3)
            if not sample.empty:
                parts.append("  Exemples:\n" + sample.to_string(index=False, max_colwidth=30))
        except Exception:  # noqa: BLE001
            pass

        blocks.append("\n".join(parts))

    return "\n\n".join(blocks)


@tool
def check_sql(sql: str) -> str:
    """Valide une requête SQL SELECT SANS l'exécuter (lecture seule + syntaxe).
    À appeler avant run_sql_query pour détecter erreurs et colonnes invalides."""
    ok, message = engine.explain_valid(sql)
    return "Requête valide ✓" if ok else f"Requête INVALIDE: {message}"


@tool
def run_sql_query(sql: str) -> str:
    """Exécute une requête SQL **SELECT** et retourne un aperçu des résultats (JSON).
    Refuse toute requête de modification (INSERT/UPDATE/DELETE/DDL).
    Utiliser des noms de colonnes/tables réels découverts via describe_tables."""
    try:
        df = engine.run_select(sql)
    except Exception as e:  # noqa: BLE001
        return f"ERREUR d'exécution: {e}\nVérifier les noms de colonnes via describe_tables."

    if df.empty:
        return "Requête exécutée : 0 ligne retournée."

    preview = df.head(_PREVIEW_ROWS)
    payload = {
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "preview_rows": json.loads(preview.to_json(orient="records", date_format="iso")),
        "truncated": len(df) > _PREVIEW_ROWS,
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


# Liste passée à create_deep_agent
SQL_TOOLS = [list_tables, describe_tables, check_sql, run_sql_query]
