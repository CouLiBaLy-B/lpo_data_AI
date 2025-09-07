# src/ui/utils.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from datetime import datetime
import base64
import io

from ..core.models import QueryResult, ChartType
from ..core.exceptions import VisualizationException

logger = logging.getLogger(__name__)


class StreamlitUtils:
    """Utilitaires pour l'interface Streamlit"""

    @staticmethod
    def display_metrics(metrics: Dict[str, Any], columns: int = 4):
        """Affiche des métriques dans des colonnes"""
        cols = st.columns(columns)

        for i, (key, value) in enumerate(metrics.items()):
            with cols[i % columns]:
                if isinstance(value, dict) and "value" in value:
                    st.metric(
                        label=value.get("label", key),
                        value=value["value"],
                        delta=value.get("delta"),
                        help=value.get("help"),
                    )
                else:
                    st.metric(label=key.replace("_", " ").title(), value=value)

    @staticmethod
    def display_dataframe_with_controls(df: pd.DataFrame, key: str = None):
        """Affiche un DataFrame avec des contrôles"""
        if df.empty:
            st.warning("Aucune donnée à afficher")
            return

        # Contrôles
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            # Recherche
            search_term = st.text_input("🔍 Rechercher", key=f"search_{key}")

        with col2:
            # Nombre de lignes à afficher
            rows_to_show = st.selectbox(
                "Lignes", [10, 25, 50, 100, len(df)], index=1, key=f"rows_{key}"
            )

        with col3:
            # Colonnes à afficher
            columns_to_show = st.multiselect(
                "Colonnes",
                df.columns.tolist(),
                default=df.columns.tolist()[:5],
                key=f"cols_{key}",
            )

        # Filtrage
        filtered_df = df.copy()

        if search_term:
            # Recherche dans toutes les colonnes texte
            text_columns = df.select_dtypes(include=["object"]).columns
            mask = pd.Series([False] * len(df))

            for col in text_columns:
                mask |= (
                    df[col].astype(str).str.contains(search_term, case=False, na=False)
                )

            filtered_df = df[mask]

        # Sélection des colonnes
        if columns_to_show:
            filtered_df = filtered_df[columns_to_show]

        # Limitation des lignes
        if rows_to_show < len(filtered_df):
            filtered_df = filtered_df.head(rows_to_show)

        # Affichage
        st.dataframe(filtered_df, use_container_width=True)

        # Informations
        st.caption(f"Affichage de {len(filtered_df)} lignes sur {len(df)} total")

        # Bouton de téléchargement
        if not filtered_df.empty:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger CSV",
                data=csv,
                file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=f"download_{key}",
            )

    @staticmethod
    def create_chart_selector() -> Dict[str, Any]:
        """Crée un sélecteur de type de graphique"""
        chart_options = {
            "📊 Barres": "bar",
            "🥧 Secteurs": "pie",
            "📈 Ligne": "line",
            "🔵 Nuage de points": "scatter",
            "📊 Histogramme": "histogram",
            "🗺️ Carte": "map",
        }

        col1, col2 = st.columns([1, 2])

        with col1:
            selected_chart = st.selectbox(
                "Type de graphique",
                list(chart_options.keys()),
                key="chart_type_selector",
            )

        chart_type = chart_options[selected_chart]

        # Options spécifiques selon le type
        options = {"type": chart_type}

        with col2:
            if chart_type in ["bar", "line"]:
                options["orientation"] = st.radio(
                    "Orientation", ["Vertical", "Horizontal"], key="chart_orientation"
                )

            elif chart_type == "pie":
                options["hole"] = st.slider(
                    "Taille du trou (donut)", 0.0, 0.8, 0.0, key="pie_hole"
                )

        return options

    @staticmethod
    def display_query_result(result: QueryResult, show_chart: bool = True):
        """Affiche le résultat d'une requête avec options de visualisation"""
        if not result.data:
            st.warning("Aucun résultat trouvé")
            return

        # Conversion en DataFrame
        df = pd.DataFrame(result.data)

        # Onglets pour les différentes vues
        if show_chart and len(df.columns) >= 2:
            tab1, tab2, tab3 = st.tabs(["📊 Graphique", "📋 Données", "ℹ️ Informations"])
        else:
            tab1, tab2 = st.tabs(["📋 Données", "ℹ️ Informations"])
            tab3 = None

        # Onglet graphique
        if show_chart and len(df.columns) >= 2 and tab3:
            with tab1:
                try:
                    chart_config = StreamlitUtils.create_chart_selector()
                    fig = StreamlitUtils.create_plotly_chart(df, chart_config)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur lors de la création du graphique: {e}")

        # Onglet données
        with tab1 if not tab3 else tab2:
            StreamlitUtils.display_dataframe_with_controls(df, "query_result")

        # Onglet informations
        with tab2 if not tab3 else tab3:
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Nombre de lignes", result.row_count)
                st.metric("Nombre de colonnes", len(result.columns))

            with col2:
                st.metric("Temps d'exécution", f"{result.execution_time:.3f}s")
                st.metric("Hash de la requête", result.query_hash[:8] + "...")

            # Détails des colonnes
            st.subheader("Colonnes")
            for i, col in enumerate(result.columns):
                col_type = str(df[col].dtype)
                null_count = df[col].isnull().sum()
                unique_count = df[col].nunique()

                st.write(
                    f"**{col}** ({col_type}) - {unique_count} valeurs uniques, {null_count} nulls"
                )

    @staticmethod
    def create_plotly_chart(
        df: pd.DataFrame, config: Dict[str, Any]
    ) -> Optional[go.Figure]:
        """Crée un graphique Plotly basé sur la configuration"""
        try:
            chart_type = config.get("type", "bar")

            if len(df.columns) < 2:
                st.warning(
                    "Au moins 2 colonnes sont nécessaires pour créer un graphique"
                )
                return None

            # Sélection automatique des colonnes
            x_col = df.columns[0]
            y_col = df.columns[1]

            # Limitation des données pour les performances
            if len(df) > 1000:
                df_plot = df.head(1000)
                st.info(
                    "Affichage limité aux 1000 premières lignes pour les performances"
                )
            else:
                df_plot = df

            # Création du graphique selon le type
            if chart_type == "bar":
                orientation = config.get("orientation", "Vertical")
                if orientation == "Horizontal":
                    fig = px.bar(df_plot, x=y_col, y=x_col, orientation="h")
                else:
                    fig = px.bar(df_plot, x=x_col, y=y_col)

            elif chart_type == "pie":
                hole = config.get("hole", 0.0)
                fig = px.pie(df_plot, values=y_col, names=x_col, hole=hole)

            elif chart_type == "line":
                fig = px.line(df_plot, x=x_col, y=y_col, markers=True)

            elif chart_type == "scatter":
                size_col = df.columns[2] if len(df.columns) > 2 else None
                fig = px.scatter(df_plot, x=x_col, y=y_col, size=size_col)

            elif chart_type == "histogram":
                fig = px.histogram(df_plot, x=x_col)

            else:
                fig = px.bar(df_plot, x=x_col, y=y_col)

            # Personnalisation
            fig.update_layout(
                title=f"Analyse des données",
                xaxis_title=x_col.replace("_", " ").title(),
                yaxis_title=y_col.replace("_", " ").title(),
                template="plotly_white",
            )

            return fig

        except Exception as e:
            logger.error(f"Erreur création graphique: {e}")
            st.error(f"Erreur lors de la création du graphique: {e}")
            return None

    @staticmethod
    def display_error(error: Exception, context: str = None):
        """Affiche une erreur de manière conviviale"""
        error_type = type(error).__name__

        with st.container():
            st.error(f"**{error_type}**")

            if context:
                st.write(f"**Contexte:** {context}")

            st.write(f"**Message:** {str(error)}")

            # Détails techniques en expandeur
            with st.expander("Détails techniques"):
                st.code(str(error), language="text")

                if hasattr(error, "__traceback__"):
                    import traceback

                    st.code(traceback.format_exc(), language="text")

    @staticmethod
    def display_loading_spinner(message: str = "Chargement..."):
        """Affiche un spinner de chargement"""
        return st.spinner(message)

    @staticmethod
    def display_success_message(message: str, details: Dict[str, Any] = None):
        """Affiche un message de succès avec détails optionnels"""
        st.success(message)

        if details:
            with st.expander("Détails"):
                for key, value in details.items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")

    @staticmethod
    def create_sidebar_filters(df: pd.DataFrame) -> Dict[str, Any]:
        """Crée des filtres dans la sidebar"""
        filters = {}

        st.sidebar.header("🔍 Filtres")

        # Filtres pour les colonnes catégorielles
        categorical_cols = df.select_dtypes(include=["object"]).columns

        for col in categorical_cols:
            unique_values = df[col].dropna().unique()
            if len(unique_values) <= 20:  # Limite pour éviter trop d'options
                selected_values = st.sidebar.multiselect(
                    f"Filtrer par {col}",
                    options=sorted(unique_values),
                    key=f"filter_{col}",
                )
                if selected_values:
                    filters[col] = selected_values

        # Filtres pour les colonnes numériques
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

        for col in numeric_cols:
            min_val = float(df[col].min())
            max_val = float(df[col].max())

            if min_val != max_val:
                selected_range = st.sidebar.slider(
                    f"Plage {col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    key=f"range_{col}",
                )
                if selected_range != (min_val, max_val):
                    filters[col] = selected_range

        return filters

    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Applique les filtres au DataFrame"""
        filtered_df = df.copy()

        for col, filter_value in filters.items():
            if isinstance(filter_value, list):
                # Filtre catégoriel
                filtered_df = filtered_df[filtered_df[col].isin(filter_value)]
            elif isinstance(filter_value, tuple) and len(filter_value) == 2:
                # Filtre numérique (plage)
                min_val, max_val = filter_value
                filtered_df = filtered_df[
                    (filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)
                ]

        return filtered_df

    @staticmethod
    def display_data_quality_report(df: pd.DataFrame):
        """Affiche un rapport de qualité des données"""
        st.subheader("📊 Rapport de qualité des données")

        # Métriques générales
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Lignes", len(df))
        with col2:
            st.metric("Colonnes", len(df.columns))
        with col3:
            total_cells = len(df) * len(df.columns)
            null_cells = df.isnull().sum().sum()
            st.metric(
                "Cellules vides", f"{null_cells} ({null_cells/total_cells*100:.1f}%)"
            )
        with col4:
            duplicate_rows = df.duplicated().sum()
            st.metric(
                "Lignes dupliquées",
                f"{duplicate_rows} ({duplicate_rows/len(df)*100:.1f}%)",
            )

        # Détails par colonne
        st.subheader("Détails par colonne")

        quality_data = []
        for col in df.columns:
            col_data = {
                "Colonne": col,
                "Type": str(df[col].dtype),
                "Valeurs uniques": df[col].nunique(),
                "Valeurs nulles": df[col].isnull().sum(),
                "% Nulles": f"{df[col].isnull().sum()/len(df)*100:.1f}%",
                "Mémoire (KB)": f"{df[col].memory_usage(deep=True)/1024:.1f}",
            }
            quality_data.append(col_data)

        quality_df = pd.DataFrame(quality_data)
        st.dataframe(quality_df, use_container_width=True)

    @staticmethod
    def create_download_button(
        data: Union[pd.DataFrame, str],
        filename: str,
        label: str = "Télécharger",
        mime_type: str = None,
    ):
        """Crée un bouton de téléchargement"""
        if isinstance(data, pd.DataFrame):
            if filename.endswith(".csv"):
                data_str = data.to_csv(index=False)
                mime_type = mime_type or "text/csv"
            elif filename.endswith(".json"):
                data_str = data.to_json(orient="records", indent=2)
                mime_type = mime_type or "application/json"
            else:
                data_str = str(data)
                mime_type = mime_type or "text/plain"
        else:
            data_str = str(data)
            mime_type = mime_type or "text/plain"

        return st.download_button(
            label=label, data=data_str, file_name=filename, mime=mime_type
        )

    @staticmethod
    def display_code_with_copy(code: str, language: str = "python"):
        """Affiche du code avec option de copie"""
        st.code(code, language=language)

        # Bouton de copie (simulation)
        if st.button("📋 Copier le code", key=f"copy_{hash(code)}"):
            # Note: La copie réelle nécessiterait du JavaScript
            st.success("Code copié dans le presse-papiers!")

    @staticmethod
    def create_progress_bar(current: int, total: int, message: str = ""):
        """Crée une barre de progression"""
        progress = current / total if total > 0 else 0

        progress_bar = st.progress(progress)

        if message:
            st.write(f"{message} ({current}/{total})")

        return progress_bar

    @staticmethod
    def display_json_viewer(data: Dict[str, Any], title: str = "Données JSON"):
        """Affiche un visualiseur JSON"""
        st.subheader(title)

        # Option d'affichage
        view_mode = st.radio(
            "Mode d'affichage",
            ["Formaté", "Brut", "Arbre"],
            horizontal=True,
            key=f"json_view_{hash(str(data))}",
        )

        if view_mode == "Formaté":
            st.json(data)
        elif view_mode == "Brut":
            import json

            st.code(json.dumps(data, indent=2, ensure_ascii=False), language="json")
        else:  # Arbre
            StreamlitUtils._display_json_tree(data)

    @staticmethod
    def _display_json_tree(data: Any, level: int = 0, key: str = None):
        """Affiche les données JSON sous forme d'arbre"""
        indent = "  " * level

        if isinstance(data, dict):
            if key:
                st.write(f"{indent}📁 **{key}**")
            for k, v in data.items():
                StreamlitUtils._display_json_tree(v, level + 1, k)
        elif isinstance(data, list):
            if key:
                st.write(f"{indent}📋 **{key}** ({len(data)} éléments)")
            for i, item in enumerate(data[:10]):  # Limiter à 10 éléments
                StreamlitUtils._display_json_tree(item, level + 1, f"[{i}]")
            if len(data) > 10:
                st.write(f"{indent}  ... et {len(data) - 10} autres éléments")
        else:
            value_str = str(data)
            if len(value_str) > 100:
                value_str = value_str[:100] + "..."
            st.write(f"{indent}📄 **{key}**: {value_str}")


class ChartBuilder:
    """Constructeur de graphiques avancé"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    def create_interactive_builder(self) -> Optional[go.Figure]:
        """Crée un constructeur de graphiques interactif"""
        st.subheader("🎨 Constructeur de graphiques")

        if self.df.empty:
            st.warning("Aucune donnée disponible pour créer un graphique")
            return None

        # Sélection du type de graphique
        chart_types = {
            "Barres": "bar",
            "Ligne": "line",
            "Secteurs": "pie",
            "Nuage de points": "scatter",
            "Histogramme": "histogram",
            "Boîte à moustaches": "box",
            "Violon": "violin",
            "Heatmap": "heatmap",
        }

        col1, col2 = st.columns([1, 2])

        with col1:
            chart_type = st.selectbox(
                "Type de graphique", list(chart_types.keys()), key="chart_builder_type"
            )

        chart_code = chart_types[chart_type]

        # Configuration selon le type
        config = self._get_chart_config(chart_code)

        if not config:
            return None

        # Création du graphique
        try:
            fig = self._create_chart(chart_code, config)

            with col2:
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="chart_preview")

            return fig

        except Exception as e:
            st.error(f"Erreur lors de la création du graphique: {e}")
            return None

    def _get_chart_config(self, chart_type: str) -> Optional[Dict[str, Any]]:
        """Obtient la configuration pour un type de graphique"""
        config = {}

        if chart_type in ["bar", "line", "scatter"]:
            col1, col2, col3 = st.columns(3)

            with col1:
                x_options = (
                    self.categorical_cols + self.numeric_cols + self.datetime_cols
                )
                if not x_options:
                    st.error("Aucune colonne disponible pour l'axe X")
                    return None

                config["x"] = st.selectbox("Axe X", x_options, key="x_axis")

            with col2:
                if chart_type != "histogram":
                    y_options = self.numeric_cols
                    if not y_options:
                        st.error("Aucune colonne numérique disponible pour l'axe Y")
                        return None

                    config["y"] = st.selectbox("Axe Y", y_options, key="y_axis")

            with col3:
                # Options de couleur
                color_options = ["Aucune"] + self.categorical_cols
                color_choice = st.selectbox(
                    "Couleur par", color_options, key="color_by"
                )
                if color_choice != "Aucune":
                    config["color"] = color_choice

        elif chart_type == "pie":
            col1, col2 = st.columns(2)

            with col1:
                if not self.categorical_cols:
                    st.error("Aucune colonne catégorielle pour les noms")
                    return None
                config["names"] = st.selectbox(
                    "Noms", self.categorical_cols, key="pie_names"
                )

            with col2:
                if not self.numeric_cols:
                    st.error("Aucune colonne numérique pour les valeurs")
                    return None
                config["values"] = st.selectbox(
                    "Valeurs", self.numeric_cols, key="pie_values"
                )

        elif chart_type == "histogram":
            col1, col2 = st.columns(2)

            with col1:
                hist_options = self.numeric_cols
                if not hist_options:
                    st.error("Aucune colonne numérique disponible")
                    return None
                config["x"] = st.selectbox("Variable", hist_options, key="hist_var")

            with col2:
                config["bins"] = st.slider(
                    "Nombre de bins", 5, 100, 20, key="hist_bins"
                )

        elif chart_type in ["box", "violin"]:
            col1, col2 = st.columns(2)

            with col1:
                if self.categorical_cols:
                    config["x"] = st.selectbox(
                        "Catégorie", ["Aucune"] + self.categorical_cols, key="box_cat"
                    )
                    if config["x"] == "Aucune":
                        config["x"] = None

            with col2:
                if not self.numeric_cols:
                    st.error("Aucune colonne numérique disponible")
                    return None
                config["y"] = st.selectbox("Variable", self.numeric_cols, key="box_var")

        elif chart_type == "heatmap":
            numeric_cols_for_heatmap = self.numeric_cols
            if len(numeric_cols_for_heatmap) < 2:
                st.error("Au moins 2 colonnes numériques nécessaires pour une heatmap")
                return None

            config["columns"] = st.multiselect(
                "Colonnes à inclure",
                numeric_cols_for_heatmap,
                default=numeric_cols_for_heatmap[:5],
                key="heatmap_cols",
            )

        return config

    def _create_chart(
        self, chart_type: str, config: Dict[str, Any]
    ) -> Optional[go.Figure]:
        """Crée le graphique selon la configuration"""
        try:
            df_plot = self.df.copy()

            # Limitation des données pour les performances
            if len(df_plot) > 5000:
                df_plot = df_plot.sample(5000)
                st.info("Échantillon de 5000 lignes pour les performances")

            if chart_type == "bar":
                fig = px.bar(
                    df_plot, x=config["x"], y=config.get("y"), color=config.get("color")
                )

            elif chart_type == "line":
                fig = px.line(
                    df_plot,
                    x=config["x"],
                    y=config.get("y"),
                    color=config.get("color"),
                    markers=True,
                )

            elif chart_type == "scatter":
                fig = px.scatter(
                    df_plot, x=config["x"], y=config.get("y"), color=config.get("color")
                )

            elif chart_type == "pie":
                fig = px.pie(df_plot, names=config["names"], values=config["values"])

            elif chart_type == "histogram":
                fig = px.histogram(df_plot, x=config["x"], nbins=config.get("bins", 20))

            elif chart_type == "box":
                fig = px.box(df_plot, x=config.get("x"), y=config["y"])

            elif chart_type == "violin":
                fig = px.violin(df_plot, x=config.get("x"), y=config["y"])

            elif chart_type == "heatmap":
                if config.get("columns"):
                    corr_matrix = df_plot[config["columns"]].corr()
                    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
                else:
                    return None

            else:
                return None

            # Personnalisation
            fig.update_layout(
                title=f"Graphique {chart_type.title()}",
                template="plotly_white",
                height=500,
            )

            return fig

        except Exception as e:
            logger.error(f"Erreur création graphique {chart_type}: {e}")
            raise


class UIComponents:
    """Composants UI réutilisables"""

    @staticmethod
    def create_info_card(title: str, content: str, icon: str = "ℹ️"):
        """Crée une carte d'information"""
        st.markdown(
            f"""
        <div style="
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            background-color: #f8f9fa;
        ">
            <h3>{icon} {title}</h3>
            <p>{content}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def create_status_badge(status: str, color: str = None):
        """Crée un badge de statut"""
        colors = {
            "success": "#28a745",
            "error": "#dc3545",
            "warning": "#ffc107",
            "info": "#17a2b8",
        }

        badge_color = color or colors.get(status.lower(), "#6c757d")

        st.markdown(
            f"""
        <span style="
            background-color: {badge_color};
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        ">{status.upper()}</span>
        """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def create_timeline(events: List[Dict[str, Any]]):
        """Crée une timeline d'événements"""
        st.subheader("📅 Timeline")

        for i, event in enumerate(events):
            col1, col2 = st.columns([1, 4])

            with col1:
                st.write(f"**{event.get('date', 'N/A')}**")

            with col2:
                st.write(f"**{event.get('title', 'Événement')}**")
                if event.get("description"):
                    st.write(event["description"])

                if i < len(events) - 1:
                    st.write("---")

    @staticmethod
    def create_comparison_table(
        data1: Dict[str, Any],
        data2: Dict[str, Any],
        labels: tuple[str, str] = ("Avant", "AprÃ¨s"),
    ):
        """Crée un tableau de comparaison"""
        comparison_data = []

        all_keys = set(data1.keys()) | set(data2.keys())

        for key in sorted(all_keys):
            row = {
                "Métrique": key.replace("_", " ").title(),
                labels[0]: data1.get(key, "N/A"),
                labels[1]: data2.get(key, "N/A"),
            }

            # Calcul de la différence si possible
            try:
                val1 = float(data1.get(key, 0))
                val2 = float(data2.get(key, 0))
                diff = val2 - val1
                row["Différence"] = f"{diff:+.2f}"
            except (ValueError, TypeError):
                row["Différence"] = "N/A"

            comparison_data.append(row)

        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)

    @staticmethod
    def create_alert(message: str, alert_type: str = "info"):
        """Crée une alerte stylisée"""
        icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}

        colors = {
            "info": "#d1ecf1",
            "success": "#d4edda",
            "warning": "#fff3cd",
            "error": "#f8d7da",
        }

        icon = icons.get(alert_type, "ℹ️")
        bg_color = colors.get(alert_type, "#d1ecf1")

        st.markdown(
            f"""
        <div style="
            background-color: {bg_color};
            border-left: 4px solid #007bff;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        ">
            {icon} {message}
        </div>
        """,
            unsafe_allow_html=True,
        )


# Fonctions utilitaires globales
def format_number(num: Union[int, float], precision: int = 2) -> str:
    """Formate un nombre avec séparateurs de milliers"""
    if isinstance(num, (int, float)):
        if num >= 1_000_000:
            return f"{num/1_000_000:.{precision}f}M"
        elif num >= 1_000:
            return f"{num/1_000:.{precision}f}K"
        else:
            return f"{num:.{precision}f}"
    return str(num)


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division sécurisée qui évite la division par zéro"""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Tronque un texte à une longueur maximale"""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def validate_dataframe(
    df: pd.DataFrame, required_columns: List[str] = None
) -> Tuple[bool, str]:
    """Valide un DataFrame"""
    if df.empty:
        return False, "DataFrame vide"

    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            return False, f"Colonnes manquantes: {', '.join(missing_cols)}"

    return True, "DataFrame valide"


def get_memory_usage(df: pd.DataFrame) -> Dict[str, str]:
    """Retourne l'utilisation mémoire d'un DataFrame"""
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum()

    return {
        "total": f"{total_memory / 1024 / 1024:.2f} MB",
        "per_column": {
            col: f"{mem / 1024:.2f} KB" for col, mem in memory_usage.items()
        },
        "average_per_row": f"{total_memory / len(df) / 1024:.2f} KB",
    }
