"""
Tableau de bord — résilient au schéma.

- Vue d'ensemble générique : liste des tables + nombre de lignes + explorateur,
  fonctionne sur n'importe quelle base (mono ou multi-tables).
- Si la table métier `sensitive_areas` est présente, affiche en plus le
  dashboard LPO dédié (chaque graphique est protégé selon les colonnes réellement
  disponibles).
"""

from __future__ import annotations

import logging

import plotly.express as px
import streamlit as st

from ...ai.cleaning import clean_dataframe
from ...database import engine

logger = logging.getLogger(__name__)

COLORS = px.colors.qualitative.Set3
MAIN_TABLE = "sensitive_areas"


def _safe_select(sql: str):
    try:
        return clean_dataframe(engine.run_select(sql))
    except Exception as e:  # noqa: BLE001
        logger.warning("Dashboard query échouée: %s", e)
        return None


def _columns_of(table: str) -> set[str]:
    try:
        return {c["name"] for c in engine.get_schema(table)["columns"]}
    except Exception:  # noqa: BLE001
        return set()


class DashboardPage:
    def render(self):
        st.title("📊 Tableau de bord")

        try:
            tables = engine.list_tables()
        except Exception as e:  # noqa: BLE001
            st.error(f"Connexion base de données impossible : {e}")
            st.info("Vérifie `DATABASE_URL` dans le fichier `.env`.")
            return

        if not tables:
            st.warning("La base est connectée mais ne contient aucune table.")
            return

        self._overview(tables)
        st.divider()

        if MAIN_TABLE in tables:
            self._lpo_dashboard()
        else:
            self._table_explorer(tables)

    # ── Vue d'ensemble générique ────────────────────────────────────────────────

    def _overview(self, tables: list[str]):
        st.subheader("🗃️ Vue d'ensemble de la base")
        rows = []
        for t in tables:
            df = _safe_select(f'SELECT COUNT(*) AS n FROM "{t}"')
            n = int(df["n"].iloc[0]) if df is not None and not df.empty else 0
            rows.append({"Table": t, "Lignes": n, "Colonnes": len(_columns_of(t))})

        c1, c2, c3 = st.columns(3)
        c1.metric("Tables / vues", len(tables))
        c2.metric("Total lignes", f"{sum(r['Lignes'] for r in rows):,}".replace(",", " "))
        c3.metric("Connexion", engine.get_engine().dialect.name)

        st.dataframe(rows, use_container_width=True, hide_index=True)

    def _table_explorer(self, tables: list[str]):
        st.subheader("🔎 Explorateur de tables")
        table = st.selectbox("Choisir une table", tables)
        if not table:
            return
        schema = engine.get_schema(table)
        cols = ", ".join(f"{c['name']} ({c['type']})" for c in schema["columns"])
        st.caption(f"**Colonnes :** {cols}")
        df = _safe_select(f'SELECT * FROM "{table}" LIMIT 200')
        if df is not None:
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Dashboard LPO dédié (si sensitive_areas existe) ─────────────────────────

    def _lpo_dashboard(self):
        st.subheader("🦅 Zones sensibles LPO")
        cols = _columns_of(MAIN_TABLE)

        self._lpo_metrics(cols)

        c1, c2 = st.columns(2)
        with c1:
            if "category" in cols:
                self._chart_categories()
            if {"create_datetime"} & cols:
                self._chart_timeline()
        with c2:
            if "region" in cols:
                self._chart_regions()
            if "structure" in cols:
                self._chart_structures()

    def _lpo_metrics(self, cols: set[str]):
        metrics = []
        total = _safe_select(f"SELECT COUNT(*) AS n FROM {MAIN_TABLE}")
        metrics.append(("🗺 Zones", int(total["n"].iloc[0]) if total is not None else "—"))
        if "region" in cols:
            r = _safe_select(
                f"SELECT COUNT(DISTINCT region) AS n FROM {MAIN_TABLE} WHERE region IS NOT NULL"
            )
            metrics.append(("🌍 Régions", int(r["n"].iloc[0]) if r is not None else "—"))
        if "structure" in cols:
            s = _safe_select(
                f"SELECT COUNT(DISTINCT structure) AS n FROM {MAIN_TABLE} "
                "WHERE structure IS NOT NULL"
            )
            metrics.append(("🏢 Structures", int(s["n"].iloc[0]) if s is not None else "—"))
        if "category" in cols:
            e = _safe_select(f"SELECT COUNT(*) AS n FROM {MAIN_TABLE} WHERE category = 'Espece'")
            metrics.append(("🦅 Espèces", int(e["n"].iloc[0]) if e is not None else "—"))

        for col, (label, value) in zip(st.columns(len(metrics)), metrics):
            col.metric(label, value)

    def _chart_categories(self):
        df = _safe_select(
            f"SELECT category, COUNT(*) AS nb FROM {MAIN_TABLE} GROUP BY category ORDER BY nb DESC"
        )
        if df is None or df.empty:
            return
        fig = px.pie(df, values="nb", names="category", title="Répartition par catégorie",
                     color_discrete_sequence=COLORS, template="plotly_white")
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    def _chart_regions(self):
        df = _safe_select(
            f"SELECT region, COUNT(*) AS nb FROM {MAIN_TABLE} "
            f"WHERE region IS NOT NULL GROUP BY region ORDER BY nb DESC LIMIT 15"
        )
        if df is None or df.empty:
            return
        fig = px.bar(df, x="nb", y="region", orientation="h", title="Top 15 régions",
                     color="nb", color_continuous_scale="Greens", template="plotly_white")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    def _chart_timeline(self):
        # Expression de troncature mensuelle adaptée au moteur (SQLite vs Postgres)
        if engine.get_engine().dialect.name == "postgresql":
            month_expr = "to_char(create_datetime, 'YYYY-MM')"
        else:
            month_expr = "strftime('%Y-%m', create_datetime)"
        df = _safe_select(
            f"SELECT {month_expr} AS mois, COUNT(*) AS nb "
            f"FROM {MAIN_TABLE} WHERE create_datetime IS NOT NULL GROUP BY mois ORDER BY mois"
        )
        if df is None or df.empty:
            return
        fig = px.area(df, x="mois", y="nb", title="Créations par mois",
                      color_discrete_sequence=["#2E8B57"], template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    def _chart_structures(self):
        df = _safe_select(
            f"SELECT structure, COUNT(*) AS nb FROM {MAIN_TABLE} "
            f"WHERE structure IS NOT NULL GROUP BY structure ORDER BY nb DESC LIMIT 12"
        )
        if df is None or df.empty:
            return
        fig = px.bar(df, x="nb", y="structure", orientation="h", title="Top 12 structures",
                     color_discrete_sequence=["#1f77b4"], template="plotly_white")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)


dashboard_page = DashboardPage()
