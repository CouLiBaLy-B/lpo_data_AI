# src/ui/pages/dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

from src.services.data_service import data_service
from src.services.cache_service import cache_service
from src.core.exceptions import DatabaseException
from src.ui.utils import StreamlitUtils, UIComponents, format_number

logger = logging.getLogger(__name__)


class DashboardPage:
    """Page de tableau de bord principal"""

    def __init__(self):
        self.data_service = data_service
        self.cache_service = cache_service

    def render(self):
        """Affiche la page du tableau de bord"""
        st.title("📊 Tableau de Bord - Zones Sensibles")

        # Actualisation des données
        self._render_refresh_section()

        # Métriques principales
        self._render_key_metrics()

        # Graphiques principaux
        col1, col2 = st.columns(2)

        with col1:
            self._render_category_chart()
            self._render_timeline_chart()

        with col2:
            self._render_region_chart()
            self._render_structure_chart()

        # Tableaux détaillés
        self._render_detailed_tables()

        # Informations système
        with st.expander("ℹ️ Informations système"):
            self._render_system_info()

    def _render_refresh_section(self):
        """Section d'actualisation des données"""
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.subheader("🔄 Actualisation des données")

        with col2:
            if st.button("🔄 Actualiser", type="primary"):
                self._refresh_data()

        with col3:
            # Statut de la dernière actualisation
            summary = self.data_service.get_data_summary()
            last_update = summary.get("last_update")

            if last_update:
                time_diff = datetime.now() - last_update
                if time_diff < timedelta(hours=1):
                    UIComponents.create_status_badge("Récent", "success")
                elif time_diff < timedelta(hours=24):
                    UIComponents.create_status_badge("Acceptable", "warning")
                else:
                    UIComponents.create_status_badge("Ancien", "error")

                st.caption(f"Dernière MAJ: {last_update.strftime('%d/%m/%Y %H:%M')}")
            else:
                UIComponents.create_status_badge("Inconnue", "info")

    def _refresh_data(self):
        """Actualise les données"""
        try:
            with st.spinner("Actualisation des données en cours..."):
                result = self.data_service.refresh_data()

                if result["success"]:
                    StreamlitUtils.display_success_message(
                        result["message"],
                        {
                            "records_processed": result.get("records_processed", 0),
                            "last_update": result.get("last_update", "N/A"),
                        },
                    )
                    st.rerun()
                else:
                    st.error(
                        f"Erreur lors de l'actualisation: {result.get('message', 'Erreur inconnue')}"
                    )

        except Exception as e:
            logger.error(f"Erreur actualisation données: {e}")
            StreamlitUtils.display_error(e, "Actualisation des données")

    def _render_key_metrics(self):
        """Affiche les métriques clés"""
        try:
            stats = self.data_service.get_quick_stats()

            metrics = {
                "🏞️ Total Zones": format_number(stats.get("total_zones", 0)),
                "🗺️ Régions": format_number(stats.get("total_regions", 0)),
                "🏢 Structures": format_number(stats.get("total_structures", 0)),
                "📈 Taux de croissance": "N/A",  # À calculer avec données historiques
            }

            StreamlitUtils.display_metrics(metrics)

        except Exception as e:
            logger.error(f"Erreur métriques clés: {e}")
            st.error("Erreur lors du chargement des métriques")

    def _render_category_chart(self):
        """Graphique des zones par catégorie"""
        try:
            st.subheader("📊 Répartition par catégorie")

            query = "SELECT category, COUNT(*) as count FROM sensitive_areas GROUP BY category"
            result = self.data_service.execute_query(query)

            if result.data:
                df = pd.DataFrame(result.data)

                fig = px.pie(
                    df,
                    values="count",
                    names="category",
                    title="Distribution des zones sensibles",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                )

                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig.update_layout(height=400)

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune donnée disponible")

        except Exception as e:
            logger.error(f"Erreur graphique catégories: {e}")
            st.error("Erreur lors du chargement du graphique des catégories")

    def _render_region_chart(self):
        """Graphique des zones par région"""
        try:
            st.subheader("🗺️ Top 10 des régions")

            query = """
                SELECT region, COUNT(*) as count
                FROM sensitive_areas
                WHERE region != ''
                GROUP BY region
                ORDER BY count DESC
                LIMIT 10
            """
            result = self.data_service.execute_query(query)

            if result.data:
                df = pd.DataFrame(result.data)

                fig = px.bar(
                    df,
                    x="count",
                    y="region",
                    orientation="h",
                    title="Zones par région",
                    color="count",
                    color_continuous_scale="Blues",
                )

                fig.update_layout(
                    height=400, yaxis={"categoryorder": "total ascending"}
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune donnée disponible")

        except Exception as e:
            logger.error(f"Erreur graphique régions: {e}")
            st.error("Erreur lors du chargement du graphique des régions")

    def _render_structure_chart(self):
        """Graphique des zones par structure"""
        try:
            st.subheader("🏢 Top 10 des structures")

            query = """
                SELECT structure, COUNT(*) as count
                FROM sensitive_areas
                WHERE structure != ''
                GROUP BY structure
                ORDER BY count DESC
                LIMIT 10
            """
            result = self.data_service.execute_query(query)

            if result.data:
                df = pd.DataFrame(result.data)

                fig = px.bar(
                    df,
                    x="structure",
                    y="count",
                    title="Zones par structure",
                    color="count",
                    color_continuous_scale="Greens",
                )

                fig.update_layout(
                    height=400,
                    xaxis_tickangle=-45,
                    xaxis={"categoryorder": "total descending"},
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune donnée disponible")

        except Exception as e:
            logger.error(f"Erreur graphique structures: {e}")
            st.error("Erreur lors du chargement du graphique des structures")

    def _render_timeline_chart(self):
        """Graphique de la timeline de création"""
        try:
            st.subheader("📅 Timeline de création")

            query = """
                SELECT
                    strftime('%Y-%m', create_datetime) as month,
                    COUNT(*) as count
                FROM sensitive_areas
                WHERE create_datetime IS NOT NULL
                GROUP BY strftime('%Y-%m', create_datetime)
                ORDER BY month
                LIMIT 24
            """
            result = self.data_service.execute_query(query)

            if result.data:
                df = pd.DataFrame(result.data)
                df["month"] = pd.to_datetime(df["month"])

                fig = px.line(
                    df,
                    x="month",
                    y="count",
                    title="Évolution des créations de zones",
                    markers=True,
                )

                fig.update_layout(height=400)
                fig.update_xaxis(title="Mois")
                fig.update_yaxis(title="Nombre de zones créées")

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune donnée temporelle disponible")

        except Exception as e:
            logger.error(f"Erreur graphique timeline: {e}")
            st.error("Erreur lors du chargement de la timeline")

    def _render_detailed_tables(self):
        """Tables détaillées"""
        st.subheader("📋 Données détaillées")

        tab1, tab2, tab3 = st.tabs(
            ["🏞️ Par région", "🏢 Par structure", "🎯 Par pratique"]
        )

        with tab1:
            self._render_region_table()

        with tab2:
            self._render_structure_table()

        with tab3:
            self._render_practice_table()

    def _render_region_table(self):
        """Table des statistiques par région"""
        try:
            query = """
                SELECT
                    region,
                    COUNT(*) as total_zones,
                    COUNT(CASE WHEN category = 'Espece' THEN 1 END) as zones_especes,
                    COUNT(CASE WHEN category = 'zone reglementaire' THEN 1 END) as zones_reglementaires,
                    COUNT(DISTINCT structure) as nb_structures
                FROM sensitive_areas
                WHERE region != ''
                GROUP BY region
                ORDER BY total_zones DESC
            """
            result = self.data_service.execute_query(query)

            if result.data:
                df = pd.DataFrame(result.data)
                StreamlitUtils.display_dataframe_with_controls(df, "regions")
            else:
                st.info("Aucune donnée disponible")

        except Exception as e:
            logger.error(f"Erreur table régions: {e}")
            st.error("Erreur lors du chargement des données régionales")

    def _render_structure_table(self):
        """Table des statistiques par structure"""
        try:
            query = """
                SELECT
                    structure,
                    COUNT(*) as total_zones,
                    COUNT(DISTINCT region) as nb_regions,
                    COUNT(CASE WHEN category = 'Espece' THEN 1 END) as zones_especes,
                    COUNT(CASE WHEN category = 'zone reglementaire' THEN 1 END) as zones_reglementaires
                FROM sensitive_areas
                WHERE structure != ''
                GROUP BY structure
                ORDER BY total_zones DESC
            """
            result = self.data_service.execute_query(query)

            if result.data:
                df = pd.DataFrame(result.data)
                StreamlitUtils.display_dataframe_with_controls(df, "structures")
            else:
                st.info("Aucune donnée disponible")

        except Exception as e:
            logger.error(f"Erreur table structures: {e}")
            st.error("Erreur lors du chargement des données des structures")

    def _render_practice_table(self):
        """Table des statistiques par pratique"""
        try:
            query = """
                SELECT
                    practices,
                    COUNT(*) as nb_zones,
                    COUNT(DISTINCT region) as nb_regions,
                    COUNT(DISTINCT structure) as nb_structures
                FROM sensitive_areas
                WHERE practices IS NOT NULL AND practices != ''
                GROUP BY practices
                ORDER BY nb_zones DESC
                LIMIT 20
            """
            result = self.data_service.execute_query(query)

            if result.data:
                df = pd.DataFrame(result.data)
                StreamlitUtils.display_dataframe_with_controls(df, "practices")
            else:
                st.info("Aucune donnée disponible")

        except Exception as e:
            logger.error(f"Erreur table pratiques: {e}")
            st.error("Erreur lors du chargement des données des pratiques")

    def _render_system_info(self):
        """Informations système"""
        try:
            # Informations sur les données
            summary = self.data_service.get_data_summary()

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Données")
                if summary.get("data_stats"):
                    stats = summary["data_stats"]
                    st.write(
                        f"**Total d'enregistrements:** {stats.get('total_records', 'N/A')}"
                    )
                    st.write(
                        f"**Régions uniques:** {stats.get('total_regions', 'N/A')}"
                    )
                    st.write(
                        f"**Structures uniques:** {stats.get('total_structures', 'N/A')}"
                    )

                    if stats.get("last_calculated"):
                        st.write(
                            f"**Calculé le:** {stats['last_calculated'].strftime('%d/%m/%Y %H:%M')}"
                        )

            with col2:
                st.subheader("💾 Cache")
                cache_stats = self.cache_service.get_stats()

                if cache_stats.get("enabled"):
                    st.write("**Statut:** Activé")
                    st.write(
                        f"**Taille:** {cache_stats.get('size', 0)}/{cache_stats.get('max_size', 0)}"
                    )
                    st.write(
                        f"**Taux de réussite:** {cache_stats.get('hit_rate', 0):.1%}"
                    )
                    st.write(
                        f"**Total requêtes:** {cache_stats.get('total_requests', 0)}"
                    )
                else:
                    st.write("**Statut:** Désactivé")

            # Informations de base de données
            st.subheader("🗄️ Base de données")
            db_info = self.data_service.get_database_info()

            if db_info:
                st.write(f"**Chemin:** {db_info.get('path', 'N/A')}")
                st.write(f"**Taille:** {db_info.get('size', 'N/A')}")
                st.write(f"**Tables:** {len(db_info.get('tables', []))}")

                if db_info.get("last_backup"):
                    st.write(f"**Dernière sauvegarde:** {db_info['last_backup']}")
                else:
                    st.write("**Dernière sauvegarde:** Aucune")

        except Exception as e:
            logger.error(f"Erreur informations système: {e}")
            st.error("Erreur lors du chargement des informations système")


# Instance de la page
dashboard_page = DashboardPage()
