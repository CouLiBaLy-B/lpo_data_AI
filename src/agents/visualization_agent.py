# src/agents/visualization_agent.py
import re
import ast
from typing import Dict, Any, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
import logging

from .base_agent import AIAgent, CachedAgent
from ..core.models import (
    QueryResult,
    VisualizationConfig,
    ChartType,
    VisualizationResult,
)
from ..core.exceptions import VisualizationException, AIModelException
from ..config.settings import settings

logger = logging.getLogger(__name__)


class VisualizationAgent(CachedAgent, AIAgent):
    """Agent spécialisé dans la génération de visualisations"""

    def __init__(self, cache_service=None):
        CachedAgent.__init__(self, "VisualizationAgent", cache_service)
        AIAgent.__init__(self, "VisualizationAgent", settings.ai.chart_model)
        self._initialize_llm()
        self.chart_templates = self._load_chart_templates()

    def _initialize_llm(self):
        """Initialise le modèle de langage pour la visualisation"""
        try:
            self.llm = HuggingFaceHub(
                repo_id=self.model_name,
                model_kwargs={
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_tokens,
                    "do_sample": True,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                },
                huggingfacehub_api_token=settings.ai.huggingface_api_key,
            )
        except Exception as e:
            raise AIModelException(
                f"Erreur lors de l'initialisation du LLM de visualisation: {e}"
            )

    def _load_chart_templates(self) -> Dict[str, str]:
        """Charge les templates de graphiques"""
        return {
            "bar": """
def create_plot(data):
    import plotly.express as px
    fig = px.bar(data, x='{x}', y='{y}', title='{title}', color='{color}')
    fig.update_layout(xaxis_title='{x_title}', yaxis_title='{y_title}')
    return fig
            """,
            "pie": """
def create_plot(data):
    import plotly.express as px
    fig = px.pie(data, values='{values}', names='{names}', title='{title}')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig
            """,
            "line": """
def create_plot(data):
    import plotly.express as px
    fig = px.line(data, x='{x}', y='{y}', title='{title}', markers=True)
    fig.update_layout(xaxis_title='{x_title}', yaxis_title='{y_title}')
    return fig
            """,
            "scatter": """
def create_plot(data):
    import plotly.express as px
    fig = px.scatter(data, x='{x}', y='{y}', title='{title}', 
                    color='{color}', size='{size}')
    return fig
            """,
            "histogram": """
def create_plot(data):
    import plotly.express as px
    fig = px.histogram(data, x='{x}', title='{title}', nbins=20)
    return fig
            """,
        }

    def process(self, request: Dict[str, Any]) -> VisualizationResult:
        """Traite une demande de visualisation"""
        try:
            query_result = request.get("query_result")
            question = request.get("question", "")

            if not query_result or not query_result.data:
                raise VisualizationException("Aucune donnée à visualiser")

            # Analyser les données pour déterminer le type de graphique optimal
            chart_config = self._analyze_data_for_chart(query_result, question)

            # Générer le code Python
            python_code = self._generate_visualization_code(query_result, chart_config)

            # Valider le code généré
            if not self._validate_python_code(python_code):
                raise VisualizationException("Code Python généré invalide")

            return VisualizationResult(
                chart_config=chart_config, python_code=python_code, success=True
            )

        except Exception as e:
            logger.error(f"Erreur lors de la génération de visualisation: {e}")
            return VisualizationResult(
                chart_config=VisualizationConfig(
                    chart_type=ChartType.BAR, title="Erreur"
                ),
                python_code="",
                success=False,
                error_message=str(e),
            )

    def _analyze_data_for_chart(
        self, query_result: QueryResult, question: str
    ) -> VisualizationConfig:
        """Analyse les données pour déterminer le type de graphique optimal"""
        df = pd.DataFrame(query_result.data)
        columns = query_result.columns

        # Analyse basée sur la question
        question_lower = question.lower()

        # Détection du type de graphique basé sur les mots-clés
        if any(
            word in question_lower
            for word in ["répartition", "proportion", "pourcentage"]
        ):
            chart_type = ChartType.PIE
        elif any(
            word in question_lower for word in ["évolution", "temps", "mois", "année"]
        ):
            chart_type = ChartType.LINE
        elif any(word in question_lower for word in ["corrélation", "relation"]):
            chart_type = ChartType.SCATTER
        elif any(word in question_lower for word in ["distribution", "histogramme"]):
            chart_type = ChartType.HISTOGRAM
        else:
            chart_type = ChartType.BAR

        # Déterminer les colonnes à utiliser
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Configuration par défaut
        config = VisualizationConfig(
            chart_type=chart_type,
            title=self._generate_chart_title(question, chart_type),
        )

        # Configuration spécifique selon le type
        if (
            chart_type == ChartType.PIE
            and len(numeric_columns) > 0
            and len(categorical_columns) > 0
        ):
            config.y_axis = numeric_columns[0]  # values
            config.x_axis = categorical_columns[0]  # names
        elif chart_type == ChartType.BAR and len(columns) >= 2:
            config.x_axis = columns[0]
            config.y_axis = columns[1] if len(numeric_columns) > 0 else columns[1]
        elif chart_type == ChartType.LINE and len(columns) >= 2:
            config.x_axis = columns[0]
            config.y_axis = columns[1]

        # Couleur si plus de 2 colonnes
        if len(categorical_columns) > 1:
            config.color_column = categorical_columns[1]

        return config

    def _generate_chart_title(self, question: str, chart_type: ChartType) -> str:
        """Génère un titre pour le graphique"""
        if len(question) > 60:
            return question[:57] + "..."
        return question

    def _generate_visualization_code(
        self, query_result: QueryResult, config: VisualizationConfig
    ) -> str:
        """Génère le code Python pour la visualisation"""
        try:
            # Utiliser l'IA pour générer un code plus sophistiqué
            prompt = self._create_visualization_prompt(query_result, config)
            ai_code = self._generate_ai_code(prompt)

            if ai_code and self._validate_python_code(ai_code):
                return ai_code

            # Fallback: utiliser les templates
            return self._generate_template_code(query_result, config)

        except Exception as e:
            logger.warning(f"Erreur génération IA, utilisation template: {e}")
            return self._generate_template_code(query_result, config)

    def _create_visualization_prompt(
        self, query_result: QueryResult, config: VisualizationConfig
    ) -> str:
        """Crée le prompt pour la génération de code de visualisation"""
        df_info = f"Colonnes: {', '.join(query_result.columns)}\nNombre de lignes: {query_result.row_count}"

        prompt = f"""
        Générez du code Python avec Plotly pour créer une visualisation de données.
        
        DONNÉES:
        {df_info}
        
        CONFIGURATION:
        - Type de graphique: {config.chart_type.value}
        - Titre: {config.title}
        - Axe X: {config.x_axis}
        - Axe Y: {config.y_axis}
        - Couleur: {config.color_column}
        
        INSTRUCTIONS:
        1. Utilisez uniquement plotly.express ou plotly.graph_objects
        2. Le DataFrame est déjà disponible dans la variable 'data'
        3. Créez une fonction create_plot(data) qui retourne la figure
        4. Ajoutez des titres d'axes appropriés
        5. Utilisez une palette de couleurs professionnelle
        
        FORMAT DE SORTIE:
        
        def create_plot(data):
            import plotly.express as px
            # Votre code ici
            return fig
        
        
        Code Python:
        """

        return prompt

    def _generate_ai_code(self, prompt: str) -> str:
        """Génère le code avec l'IA"""
        try:
            response = self.llm(prompt)

            # Extraire le code Python du markdown
            code_match = re.search(r"\n(.*?)\n", response, re.DOTALL)
            if code_match:
                return code_match.group(1)

            # Si pas de markdown, chercher la fonction directement
            if "def create_plot" in response:
                return response

            return None

        except Exception as e:
            logger.error(f"Erreur génération IA: {e}")
            return None

    def _generate_template_code(
        self, query_result: QueryResult, config: VisualizationConfig
    ) -> str:
        """Génère le code à partir des templates"""
        chart_type = config.chart_type.value
        template = self.chart_templates.get(chart_type, self.chart_templates["bar"])

        # Remplacer les placeholders
        replacements = {
            "title": config.title or "Graphique",
            "x": config.x_axis or query_result.columns[0],
            "y": config.y_axis
            or (
                query_result.columns[1]
                if len(query_result.columns) > 1
                else query_result.columns[0]
            ),
            "color": config.color_column or "None",
            "values": (
                config.y_axis or query_result.columns[1]
                if len(query_result.columns) > 1
                else query_result.columns[0]
            ),
            "names": config.x_axis or query_result.columns[0],
            "size": "None",
            "x_title": (config.x_axis or query_result.columns[0])
            .replace("_", " ")
            .title(),
            "y_title": (
                config.y_axis or query_result.columns[1]
                if len(query_result.columns) > 1
                else query_result.columns[0]
            )
            .replace("_", " ")
            .title(),
        }

        code = template
        for key, value in replacements.items():
            code = code.replace(f"{{{key}}}", str(value))

        # Ajouter le code complet avec imports et gestion des données
        full_code = f"""
def get_data(sql_query):
    import sqlite3
    import pandas as pd
    conn = sqlite3.connect('data/sensitive_areas.db')
    df = pd.read_sql_query(sql_query, conn)
    conn.close()
    return df

{code}

# Exécution
import streamlit as st
sql_query = '''-- La requête SQL sera insérée ici'''
data = get_data(sql_query)

col1, col2 = st.tabs(["📊 Graphique", "📋 Données"])
with col1:
    if not data.empty:
        fig = create_plot(data)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Aucune donnée à afficher")

with col2:
    st.dataframe(data, use_container_width=True)
    if not data.empty:
        st.download_button(
            label="📥 Télécharger CSV",
            data=data.to_csv(index=False),
            file_name="data.csv",
            mime="text/csv"
        )
        """

        return full_code.strip()

    def _validate_python_code(self, code: str) -> bool:
        """Valide le code Python généré"""
        try:
            # Vérifications de sécurité
            forbidden_keywords = [
                "import os",
                "import sys",
                "exec",
                "eval",
                "__import__",
                "open(",
            ]
            for keyword in forbidden_keywords:
                if keyword in code:
                    return False

            # Vérifier la syntaxe
            ast.parse(code)

            # Vérifier la présence de la fonction create_plot
            if "def create_plot" not in code:
                return False

            return True

        except SyntaxError:
            return False
        except Exception:
            return False

    def create_quick_visualization(
        self, data: pd.DataFrame, chart_type: ChartType, title: str = None, **kwargs
    ) -> go.Figure:
        """Crée rapidement une visualisation sans IA"""
        try:
            if chart_type == ChartType.BAR:
                return self._create_bar_chart(data, title, **kwargs)
            elif chart_type == ChartType.PIE:
                return self._create_pie_chart(data, title, **kwargs)
            elif chart_type == ChartType.LINE:
                return self._create_line_chart(data, title, **kwargs)
            elif chart_type == ChartType.SCATTER:
                return self._create_scatter_chart(data, title, **kwargs)
            elif chart_type == ChartType.HISTOGRAM:
                return self._create_histogram(data, title, **kwargs)
            else:
                return self._create_bar_chart(data, title, **kwargs)

        except Exception as e:
            raise VisualizationException(f"Erreur création graphique: {e}")

    def _create_bar_chart(self, data: pd.DataFrame, title: str, **kwargs) -> go.Figure:
        """Crée un graphique en barres"""
        x_col = kwargs.get("x", data.columns[0])
        y_col = kwargs.get(
            "y", data.columns[1] if len(data.columns) > 1 else data.columns[0]
        )

        fig = px.bar(
            data, x=x_col, y=y_col, title=title, color_discrete_sequence=["#2E8B57"]
        )
        fig.update_layout(
            xaxis_title=x_col.replace("_", " ").title(),
            yaxis_title=y_col.replace("_", " ").title(),
        )
        return fig

    def _create_pie_chart(self, data: pd.DataFrame, title: str, **kwargs) -> go.Figure:
        """Crée un graphique en secteurs"""
        values_col = kwargs.get(
            "values", data.columns[1] if len(data.columns) > 1 else data.columns[0]
        )
        names_col = kwargs.get("names", data.columns[0])

        fig = px.pie(data, values=values_col, names=names_col, title=title)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        return fig

    def _create_line_chart(self, data: pd.DataFrame, title: str, **kwargs) -> go.Figure:
        """Crée un graphique linéaire"""
        x_col = kwargs.get("x", data.columns[0])
        y_col = kwargs.get(
            "y", data.columns[1] if len(data.columns) > 1 else data.columns[0]
        )

        fig = px.line(data, x=x_col, y=y_col, title=title, markers=True)
        return fig

    def _create_scatter_chart(
        self, data: pd.DataFrame, title: str, **kwargs
    ) -> go.Figure:
        """Crée un nuage de points"""
        x_col = kwargs.get("x", data.columns[0])
        y_col = kwargs.get(
            "y", data.columns[1] if len(data.columns) > 1 else data.columns[0]
        )

        fig = px.scatter(data, x=x_col, y=y_col, title=title)
        return fig

    def _create_histogram(self, data: pd.DataFrame, title: str, **kwargs) -> go.Figure:
        """Crée un histogramme"""
        x_col = kwargs.get("x", data.columns[0])

        fig = px.histogram(data, x=x_col, title=title, nbins=20)
        return fig
