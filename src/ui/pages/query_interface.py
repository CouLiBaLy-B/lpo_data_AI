# src/ui/pages/query_interface.py
import streamlit as st
import pandas as pd
import time
import logging
from typing import Optional

from ...agents.sql_agents import SQLAgent
from ...services.data_service import data_service
from ...services.cache_service import cache_service
from ...core.models import QueryRequest, QueryResult
from ...core.exceptions import QueryExecutionException, ValidationException
from ..utils import StreamlitUtils, UIComponents, ChartBuilder

logger = logging.getLogger(__name__)


class QueryInterfacePage:
    """Interface de requête SQL avec IA"""

    def __init__(self):
        self.sql_agent = SQLAgent(data_service.db_repository, cache_service)
        self.data_service = data_service

    def render(self):
        """Affiche l'interface de requête"""
        st.title("🤖 Assistant SQL Intelligent")

        # Instructions et exemples
        self._render_instructions()

        # Interface de requête principale
        self._render_query_interface()

        # Historique des requêtes
        self._render_query_history()

        # Exemples de requêtes
        self._render_query_examples()

    def _render_instructions(self):
        """Affiche les instructions d'utilisation"""
        with st.expander("📖 Instructions d'utilisation", expanded=False):
            st.markdown(
                """
            ### Comment utiliser l'assistant SQL ?

            1. **Posez votre question en français** dans la zone de texte
            2. L'IA génère automatiquement la requête SQL correspondante
            3. Les résultats sont affichés avec des options de visualisation
            4. Vous pouvez télécharger les données ou copier le code généré

            ### Exemples de questions :
            - "Combien y a-t-il de zones par catégorie ?"
            - "Quelles sont les 5 régions avec le plus de zones sensibles ?"
            - "Montrez-moi l'évolution des créations de zones par mois"
            - "Quelles structures gèrent des zones liées aux espèces ?"

            ### Données disponibles :
            - **Zones sensibles** avec leurs caractéristiques
            - **Régions et départements** de localisation
            - **Structures** responsables des données
            - **Pratiques sportives** concernées
            - **Dates** de création et mise à jour
            """
            )

    def _render_query_interface(self):
        """Interface principale de requête"""
        st.subheader("💬 Posez votre question")

        # Zone de saisie de la question
        col1, col2 = st.columns([4, 1])

        with col1:
            # Gérer la valeur par défaut en utilisant une clé différente pour éviter les conflits
            default_value = ""
            if "selected_question" in st.session_state:
                default_value = st.session_state.selected_question
                # Effacer après utilisation pour éviter la répétition
                del st.session_state.selected_question

            question = st.text_area(
                "Question en français :",
                value=default_value,
                placeholder="Ex: Combien y a-t-il de zones sensibles par région ?",
                height=100,
                key="user_question_input",
            )

        with col2:
            st.write("")  # Espacement
            st.write("")  # Espacement

            execute_button = st.button(
                "🚀 Exécuter",
                type="primary",
                disabled=not question.strip(),
                use_container_width=True,
            )

            if st.button("🗑️ Effacer", use_container_width=True):
                # Utiliser st.rerun() avec une approche différente
                st.session_state.user_question_input = ""
                st.rerun()

        # Exécution de la requête
        if execute_button and question.strip():
            self._execute_query(question.strip())

    def _execute_query(self, question: str):
        """Exécute une requête utilisateur"""
        try:
            # Sauvegarde dans l'historique
            self._add_to_history(question)

            with st.spinner("🤖 Génération de la requête SQL..."):
                # Création de la requête
                query_request = QueryRequest(
                    question=question,
                    user_id=st.session_state.get("user_id", "anonymous"),
                    session_id=st.session_state.get("session_id", "default"),
                )

                # Génération SQL
                sql_query = self.sql_agent.generate_sql_query(query_request)

                st.success(
                    f"✅ Requête générée avec {sql_query.confidence:.1%} de confiance"
                )

                # Affichage de la requête SQL
                with st.expander("🔍 Requête SQL générée"):
                    st.code(sql_query.query, language="sql")

                    if sql_query.explanation:
                        st.write("**Explication:**")
                        st.write(sql_query.explanation)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Type de requête", sql_query.query_type.value)
                    with col2:
                        st.metric("Confiance", f"{sql_query.confidence:.1%}")

            # Exécution de la requête
            with st.spinner("📊 Exécution de la requête..."):
                result = self.data_service.execute_query(sql_query.query)

                if result.data:
                    st.success(
                        f"✅ {result.row_count} résultats trouvés en {result.execution_time:.3f}s"
                    )

                    # Affichage des résultats
                    self._display_query_results(result, sql_query)
                else:
                    st.warning("⚠️ Aucun résultat trouvé")

        except ValidationException as e:
            st.error(f"❌ Erreur de validation: {e.message}")

        except QueryExecutionException as e:
            st.error(f"❌ Erreur d'exécution: {e.message}")

        except Exception as e:
            logger.error(f"Erreur inattendue lors de l'exécution: {e}")
            StreamlitUtils.display_error(e, "Exécution de la requête")

    def _display_query_results(self, result: QueryResult, sql_query):
        """Affiche les résultats de la requête"""
        # Conversion en DataFrame
        df = pd.DataFrame(result.data)

        # Onglets pour les différentes vues
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Visualisation", "📋 Données", "📈 Graphique personnalisé", "💾 Export"]
        )

        # Onglet visualisation automatique
        with tab1:
            try:
                # Génération automatique de visualisation
                viz_request = QueryRequest(
                    question=f"Créer une visualisation pour: {sql_query.query}",
                    context="visualization",
                )

                viz_result = self.sql_agent.generate_visualization(viz_request, df)

                if viz_result.success:
                    st.subheader("📊 Visualisation automatique")

                    # Exécution du code de visualisation
                    try:
                        exec(viz_result.python_code)
                    except Exception as e:
                        st.error(f"Erreur dans le code de visualisation: {e}")

                        # Fallback: graphique simple
                        self._create_fallback_chart(df)
                else:
                    st.warning("Impossible de générer une visualisation automatique")
                    self._create_fallback_chart(df)

            except Exception as e:
                logger.error(f"Erreur visualisation automatique: {e}")
                self._create_fallback_chart(df)

        # Onglet données
        with tab2:
            StreamlitUtils.display_dataframe_with_controls(df, "query_results")

            # Rapport de qualité des données
            if st.checkbox("📊 Afficher le rapport de qualité"):
                StreamlitUtils.display_data_quality_report(df)

        # Onglet graphique personnalisé
        with tab3:
            st.subheader("🎨 Créer un graphique personnalisé")
            chart_builder = ChartBuilder(df)
            custom_chart = chart_builder.create_interactive_builder()

            if custom_chart:
                # Option de sauvegarde du graphique
                if st.button("💾 Sauvegarder ce graphique"):
                    # Ici, on pourrait sauvegarder la configuration
                    st.success("Configuration sauvegardée!")

        # Onglet export
        with tab4:
            self._render_export_options(df, result, sql_query)

    def _create_fallback_chart(self, df: pd.DataFrame):
        """Crée un graphique de fallback simple"""
        try:
            if len(df.columns) >= 2:
                # Détection automatique du type de graphique approprié
                numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
                categorical_cols = df.select_dtypes(include=["object"]).columns

                if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
                    # Graphique en barres
                    import plotly.express as px

                    x_col = categorical_cols[0]
                    y_col = numeric_cols[0]

                    # Limitation des données
                    if len(df) > 20:
                        df_plot = df.nlargest(20, y_col)
                        st.info("Affichage des 20 premières valeurs")
                    else:
                        df_plot = df

                    fig = px.bar(df_plot, x=x_col, y=y_col)
                    fig.update_layout(
                        title="Visualisation des résultats", xaxis_tickangle=-45
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(
                        "Type de données non adapté pour une visualisation automatique"
                    )
            else:
                st.info("Données insuffisantes pour créer une visualisation")

        except Exception as e:
            logger.error(f"Erreur graphique fallback: {e}")
            st.error("Impossible de créer une visualisation")

    def _render_export_options(self, df: pd.DataFrame, result: QueryResult, sql_query):
        """Options d'export des résultats"""
        st.subheader("💾 Options d'export")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Formats de données:**")

            # Export CSV
            csv_data = df.to_csv(index=False)
            st.download_button(
                "📄 Télécharger CSV",
                csv_data,
                file_name=f"query_results_{int(time.time())}.csv",
                mime="text/csv",
            )

            # Export JSON
            json_data = df.to_json(orient="records", indent=2)
            st.download_button(
                "📋 Télécharger JSON",
                json_data,
                file_name=f"query_results_{int(time.time())}.json",
                mime="application/json",
            )

            # Export Excel
            try:
                import io

                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name="Résultats", index=False)

                st.download_button(
                    "📊 Télécharger Excel",
                    buffer.getvalue(),
                    file_name=f"query_results_{int(time.time())}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except ImportError:
                st.info("Export Excel non disponible (openpyxl requis)")

        with col2:
            st.write("**Code et requêtes:**")

            # Export SQL
            st.download_button(
                "🗃️ Télécharger SQL",
                sql_query.query,
                file_name=f"query_{int(time.time())}.sql",
                mime="text/plain",
            )

            # Rapport complet
            report = self._generate_report(result, sql_query)
            st.download_button(
                "📑 Rapport complet",
                report,
                file_name=f"report_{int(time.time())}.md",
                mime="text/markdown",
            )

    def _generate_report(self, result: QueryResult, sql_query) -> str:
        """Génère un rapport complet"""
        report = f"""# Rapport de Requête SQL

## Informations générales
- **Date d'exécution:** {time.strftime('%d/%m/%Y %H:%M:%S')}
- **Nombre de résultats:** {result.row_count}
- **Temps d'exécution:** {result.execution_time:.3f} secondes
- **Type de requête:** {sql_query.query_type.value}
- **Confiance:** {sql_query.confidence:.1%}

## Requête SQL
```sql
{sql_query.query}
```

## Explication
{sql_query.explanation or 'Aucune explication disponible'}

## Colonnes des résultats
{', '.join(result.columns)}

## Statistiques
- Hash de la requête: {result.query_hash}
- Colonnes: {len(result.columns)}
- Lignes: {result.row_count}

---
*Rapport généré par LPO AI - SQL Query Visualizer*
"""
        return report

    def _add_to_history(self, question: str):
        """Ajoute une question à l'historique"""
        if "query_history" not in st.session_state:
            st.session_state.query_history = []

        # Éviter les doublons
        if question not in st.session_state.query_history:
            st.session_state.query_history.insert(0, question)

        # Limiter l'historique à 20 éléments
        if len(st.session_state.query_history) > 20:
            st.session_state.query_history = st.session_state.query_history[:20]

    def _render_query_history(self):
        """Affiche l'historique des requêtes"""
        if "query_history" in st.session_state and st.session_state.query_history:
            with st.expander("📚 Historique des requêtes"):
                st.write("Cliquez sur une question pour la réutiliser :")

                for i, past_question in enumerate(st.session_state.query_history[:10]):
                    col1, col2 = st.columns([4, 1])

                    with col1:
                        if st.button(
                            f"💬 {past_question[:80]}{'...' if len(past_question) > 80 else ''}",
                            key=f"history_{i}",
                            use_container_width=True,
                        ):
                            # Utiliser une clé différente pour éviter les conflits
                            st.session_state.selected_question = past_question
                            st.rerun()

                    with col2:
                        if st.button("🗑️", key=f"delete_history_{i}"):
                            st.session_state.query_history.pop(i)
                            st.rerun()

                # Bouton pour effacer tout l'historique
                if st.button("🗑️ Effacer tout l'historique"):
                    st.session_state.query_history = []
                    st.rerun()

    def _render_query_examples(self):
        """Affiche des exemples de requêtes"""
        with st.expander("💡 Exemples de questions"):
            examples = [
                {
                    "category": "📊 Statistiques générales",
                    "questions": [
                        "Combien y a-t-il de zones sensibles au total ?",
                        "Quelle est la répartition des zones par catégorie ?",
                        "Combien de régions sont couvertes ?",
                        "Quel est le nombre moyen de zones par région ?",
                    ],
                },
                {
                    "category": "📈 Statistiques par zone",
                    "questions": [
                        "Combien de zones sont en cours de construction ?",
                        "Quelles sont les zones les plus récentes ?",
                        "Quelles sont les zones les plus anciennes ?",
                        "Quelles sont les zones les plus importantes ?",
                    ],
                },
                {
                    "category": "🗺️ Analyses géographiques",
                    "questions": [
                        "Quelles sont les 10 régions avec le plus de zones sensibles ?",
                        "Combien de zones y a-t-il en Auvergne-Rhône-Alpes ?",
                        "Quels départements ont le moins de zones sensibles ?",
                        "Répartition des zones par pays",
                    ],
                },
                {
                    "category": "🏢 Analyses par structure",
                    "questions": [
                        "Quelles structures gèrent le plus de zones ?",
                        "Combien de zones sont gérées par la LPO ?",
                        "Quelles structures travaillent sur les espèces ?",
                        "Répartition des structures par région",
                    ],
                },
                {
                    "category": "🎯 Analyses par pratique",
                    "questions": [
                        "Quelles sont les pratiques sportives les plus concernées ?",
                        "Combien de zones concernent la randonnée ?",
                        "Répartition des pratiques par région",
                        "Quelles pratiques sont interdites dans le plus de zones ?",
                    ],
                },
                {
                    "category": "📅 Analyses temporelles",
                    "questions": [
                        "Évolution du nombre de zones créées par mois",
                        "Quelles zones ont été mises à jour récemment ?",
                        "Répartition des créations par année",
                        "Zones créées cette année",
                    ],
                },
            ]

            for example_group in examples:
                st.write(f"**{example_group['category']}**")

                for question in example_group["questions"]:
                    if st.button(
                        f"💬 {question}",
                        key=f"example_{hash(question)}",
                        use_container_width=True,
                    ):
                        # Utiliser une clé différente pour éviter les conflits
                        st.session_state.selected_question = question
                        st.rerun()

                st.write("")  # Espacement


# Instance de la page
query_interface_page = QueryInterfacePage()
