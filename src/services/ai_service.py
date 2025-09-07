# src/services/ai_service.py
import asyncio
import time
from typing import Dict, Any, List, Optional, Union
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import logging

from ..core.models import QueryRequest, SQLQuery, QueryType, VisualizationResult
from ..core.exceptions import AIModelException, ValidationException, TimeoutException
from ..config.settings import settings
from .cache_service import CacheService

logger = logging.getLogger(__name__)


class AIService:
    """Service principal pour les fonctionnalités IA"""

    def __init__(self, cache_service: CacheService = None):
        self.cache_service = cache_service
        self.sql_llm = None
        self.viz_llm = None
        self.embeddings = None
        self.knowledge_base = None
        self.request_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0

        self._initialize_models()
        self._build_knowledge_base()

    def _initialize_models(self):
        """Initialise les modèles IA"""
        try:
            # Modèle pour SQL
            self.sql_llm = HuggingFaceHub(
                repo_id=settings.ai.sql_model,
                model_kwargs={
                    "temperature": settings.ai.temperature,
                    "max_new_tokens": settings.ai.max_tokens,
                    "do_sample": True,
                    "top_p": 0.95,
                    "repetition_penalty": 1.2,
                },
                huggingfacehub_api_token=settings.ai.huggingface_api_key,
            )

            # Modèle pour visualisation
            self.viz_llm = HuggingFaceHub(
                repo_id=settings.ai.chart_model,
                model_kwargs={
                    "temperature": settings.ai.temperature,
                    "max_new_tokens": settings.ai.max_tokens,
                    "do_sample": True,
                    "top_p": 0.9,
                },
                huggingfacehub_api_token=settings.ai.huggingface_api_key,
            )

            # Embeddings pour RAG
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            logger.info("Modèles IA initialisés avec succès")

        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des modèles: {e}")
            raise AIModelException(f"Impossible d'initialiser les modèles IA: {e}")

    def _build_knowledge_base(self):
        """Construit la base de connaissances RAG"""
        try:
            documents = []

            # Schéma de base de données
            schema_doc = Document(
                page_content=self._get_database_schema(),
                metadata={"type": "schema", "importance": "high"},
            )
            documents.append(schema_doc)

            # Exemples de requêtes
            for example in self._get_sql_examples():
                doc = Document(
                    page_content=example,
                    metadata={"type": "example", "importance": "medium"},
                )
                documents.append(doc)

            # Bonnes pratiques
            for practice in self._get_best_practices():
                doc = Document(
                    page_content=practice,
                    metadata={"type": "practice", "importance": "low"},
                )
                documents.append(doc)

            # Créer le vectorstore
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50
            )

            split_docs = text_splitter.split_documents(documents)
            self.knowledge_base = FAISS.from_documents(split_docs, self.embeddings)

            logger.info("Base de connaissances RAG construite")

        except Exception as e:
            logger.error(f"Erreur construction base de connaissances: {e}")
            raise AIModelException(
                f"Impossible de construire la base de connaissances: {e}"
            )

    def _get_database_schema(self) -> str:
        """Retourne le schéma de la base de données"""
        return """
        BASE DE DONNÉES: sensitive_areas.db
        
        TABLE: sensitive_areas
        Colonnes:
        - id (INTEGER, PRIMARY KEY): Identifiant unique
        - name (TEXT): Nom de la zone sensible
        - description (TEXT): Description détaillée
        - structure (TEXT): Organisation responsable
        - species_id (INTEGER): ID espèce (NULL pour zones réglementaires)
        - practices (TEXT): Activités sportives concernées
        - create_datetime (TIMESTAMP): Date de création
        - update_datetime (TIMESTAMP): Date de mise à jour
        - region (TEXT): Région géographique
        - departement (TEXT): Département
        - pays (TEXT): Pays
        - category (TEXT): 'zone reglementaire' ou 'Espece'
        
        RELATIONS:
        - category dérivée de species_id (NULL = 'zone reglementaire')
        - Index sur region, structure, species_id, category
        """

    def _get_sql_examples(self) -> str:
        """Retourne des exemples de requêtes SQL"""
        return """
EXEMPLE 1:
Question: Combien de zones par catégorie?
SQL: SELECT category, COUNT(*) as count FROM sensitive_areas GROUP BY category;

EXEMPLE 2:
Question: Top 5 des régions avec le plus de zones?
SQL: SELECT region, COUNT(*) as nb_zones FROM sensitive_areas WHERE region != '' GROUP BY region ORDER BY nb_zones DESC LIMIT 5;

EXEMPLE 3:
Question: Zones créées par mois?
SQL: SELECT strftime('%Y-%m', create_datetime) as mois, COUNT(*) as nb FROM sensitive_areas GROUP BY mois ORDER BY mois;

EXEMPLE 4:
Question: Structures gérant des espèces?
SQL: SELECT DISTINCT structure FROM sensitive_areas WHERE category = 'Espece' AND structure != '';
"""

    def _parse_sql_response(self, response: str) -> str:
        """Parse la réponse du modèle pour extraire le SQL"""
        # Nettoyage de la réponse
        sql_query = response.strip()

        # Suppression des balises markdown si présentes
        if "" in sql_query:
            sql_query = sql_query.split("sql")[1].split("")[0].strip()
        elif "" in sql_query:
            sql_query = sql_query.split("")[1].strip()

        # Suppression des préfixes courants
        prefixes_to_remove = [
            "SQL:",
            "REQUÊTE SQL:",
            "Query:",
            "SELECT:",
            "La requête SQL est:",
            "Voici la requête:",
        ]

        for prefix in prefixes_to_remove:
            if sql_query.upper().startswith(prefix.upper()):
                sql_query = sql_query[len(prefix) :].strip()

        # Nettoyage final
        sql_query = sql_query.replace("\n", " ").strip()

        # Ajout du point-virgule si manquant
        if not sql_query.endswith(";"):
            sql_query += ";"

        return sql_query

    def _validate_sql_query(self, sql_query: str) -> Tuple[bool, str]:
        """Valide une requête SQL"""
        try:
            import sqlparse

            # Parse SQL
            parsed = sqlparse.parse(sql_query)
            if not parsed:
                return False, "Requête SQL vide ou invalide"

            # Vérifications de sécurité
            forbidden_keywords = [
                "DROP",
                "DELETE",
                "UPDATE",
                "INSERT",
                "ALTER",
                "CREATE",
                "TRUNCATE",
            ]
            tokens = [token.value.upper() for token in parsed[0].flatten()]

            for keyword in forbidden_keywords:
                if keyword in tokens:
                    return False, f"Mot-clé interdit: {keyword}"

            # Vérification SELECT
            if not any(token.upper() == "SELECT" for token in tokens):
                return False, "Seules les requêtes SELECT sont autorisées"

            return True, "Requête valide"

        except Exception as e:
            return False, f"Erreur de validation: {str(e)}"

    def _estimate_sql_confidence(self, question: str, sql_query: str) -> float:
        """Estime la confiance dans la requête générée"""
        confidence = 0.5  # Base

        question_lower = question.lower()
        sql_lower = sql_query.lower()

        # Correspondance des mots-clés
        keyword_matches = 0
        important_keywords = [
            "count",
            "sum",
            "group",
            "order",
            "where",
            "région",
            "catégorie",
        ]

        for keyword in important_keywords:
            if keyword in question_lower and keyword in sql_lower:
                keyword_matches += 1

        confidence += (keyword_matches / len(important_keywords)) * 0.3

        # Bonus pour les requêtes bien formées
        if "GROUP BY" in sql_query.upper():
            confidence += 0.1
        if "ORDER BY" in sql_query.upper():
            confidence += 0.1
        if "LIMIT" in sql_query.upper():
            confidence += 0.05

        return min(confidence, 1.0)

    def _determine_query_type(self, sql_query: str) -> QueryType:
        """Détermine le type de requête"""
        sql_upper = sql_query.upper()

        if "COUNT(" in sql_upper or "SUM(" in sql_upper or "AVG(" in sql_upper:
            return QueryType.AGGREGATE
        elif "JOIN" in sql_upper:
            return QueryType.JOIN
        elif "COUNT(*)" in sql_upper:
            return QueryType.COUNT
        else:
            return QueryType.SELECT

    def _generate_sql_explanation(self, sql_query: str, question: str) -> str:
        """Génère une explication de la requête SQL"""
        try:
            explanation_parts = []
            sql_upper = sql_query.upper()

            # Analyse des clauses principales
            if "SELECT" in sql_upper:
                explanation_parts.append("Sélection des colonnes demandées")

            if "FROM" in sql_upper:
                explanation_parts.append("depuis la table des zones sensibles")

            if "WHERE" in sql_upper:
                explanation_parts.append("avec filtrage des données")

            if "GROUP BY" in sql_upper:
                explanation_parts.append("regroupement par catégorie")

            if "ORDER BY" in sql_upper:
                explanation_parts.append("tri des résultats")

            if "LIMIT" in sql_upper:
                explanation_parts.append("limitation du nombre de résultats")

            return "Cette requête effectue une " + ", ".join(explanation_parts) + "."

        except Exception:
            return "Requête SQL pour répondre à votre question."

    def _estimate_result_size(self, sql_query: str) -> Optional[int]:
        """Estime la taille des résultats"""
        try:
            sql_upper = sql_query.upper()

            if "LIMIT" in sql_upper:
                # Extraire la valeur LIMIT
                import re

                limit_match = re.search(r"LIMIT\s+(\d+)", sql_upper)
                if limit_match:
                    return int(limit_match.group(1))

            if "GROUP BY" in sql_upper:
                return 50  # Estimation pour les regroupements

            if "DISTINCT" in sql_upper:
                return 100  # Estimation pour les valeurs distinctes

            return None  # Taille inconnue

        except Exception:
            return None

    def _get_best_practices(self) -> List[str]:
        """Retourne les bonnes pratiques SQL"""
        return [
            "Utilisez toujours des alias pour les colonnes calculées (COUNT(*) as count)",
            "Préférez les jointures explicites aux jointures implicites",
            "Utilisez LIMIT pour éviter les résultats trop volumineux",
            "Indexez les colonnes utilisées dans WHERE et GROUP BY",
            "Utilisez strftime() pour les opérations sur les dates",
        ]

    async def generate_sql_async(self, request: QueryRequest) -> SQLQuery:
        """Génère une requête SQL de manière asynchrone"""
        return await asyncio.to_thread(self.generate_sql, request)

    def generate_sql(self, request: QueryRequest) -> SQLQuery:
        """Génère une requête SQL à partir d'une question"""
        start_time = time.time()
        self.request_count += 1

        try:
            # Vérification du cache
            if self.cache_service:
                cache_key = f"sql_{hash(request.question)}"
                cached_result = self.cache_service.get(cache_key)
                if cached_result:
                    logger.info("Résultat SQL trouvé dans le cache")
                    return SQLQuery(**cached_result)

            # Récupération du contexte RAG
            context = self._get_relevant_context(request.question)

            # Génération de la requête
            sql_query = self._generate_sql_with_llm(request.question, context)

            # Validation
            if not self._validate_sql(sql_query):
                raise ValidationException("Requête SQL générée invalide")

            # Détermination du type et de la confiance
            query_type = self._determine_query_type(sql_query)
            confidence = self._calculate_confidence(request.question, sql_query)

            result = SQLQuery(
                query=sql_query,
                query_type=query_type,
                confidence=confidence,
                explanation=self._generate_explanation(request.question, sql_query),
            )

            # Mise en cache
            if self.cache_service:
                self.cache_service.set(cache_key, result.dict())

            # Métriques
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time

            logger.info(f"Requête SQL générée en {processing_time:.3f}s")
            return result

        except Exception as e:
            self.error_count += 1
            logger.error(f"Erreur génération SQL: {e}")
            raise AIModelException(f"Erreur lors de la génération SQL: {e}")

    def _get_relevant_context(self, question: str) -> str:
        """Récupère le contexte pertinent via RAG"""
        try:
            if not self.knowledge_base:
                return ""

            relevant_docs = self.knowledge_base.similarity_search(question, k=3)
            return "\n\n".join([doc.page_content for doc in relevant_docs])

        except Exception as e:
            logger.warning(f"Erreur récupération contexte RAG: {e}")
            return ""

    def _generate_sql_with_llm(self, question: str, context: str) -> str:
        """Génère la requête SQL avec le LLM"""
        prompt_template = PromptTemplate(
            input_variables=["question", "context"],
            template="""
            Vous êtes un expert en SQL spécialisé dans l'analyse de données environnementales.
            
            CONTEXTE DE LA BASE DE DONNÉES:
            {context}
            
            QUESTION DE L'UTILISATEUR:
            {question}
            
            INSTRUCTIONS:
            1. Générez UNIQUEMENT une requête SQL SELECT valide
            2. Utilisez uniquement la table 'sensitive_areas'
            3. Respectez exactement les noms de colonnes du schéma
            4. Utilisez des alias pour les colonnes calculées
            5. Ajoutez ORDER BY et LIMIT si approprié
            6. Ne générez que le SQL, sans explication
            
            REQUÊTE SQL:
            """,
        )

        chain = LLMChain(llm=self.sql_llm, prompt=prompt_template)
        response = chain.run(question=question, context=context)

        # Nettoyer la réponse
        sql_query = response.strip()
        if sql_query.startswith(""):
            sql_query = sql_query[6:]
        if sql_query.endswith(""):
            sql_query = sql_query[:-3]

        return sql_query.strip()

    def _validate_sql(self, sql_query: str) -> bool:
        """Valide une requête SQL"""
        try:
            # Vérifications de sécurité
            forbidden_keywords = [
                "DROP",
                "DELETE",
                "UPDATE",
                "INSERT",
                "ALTER",
                "CREATE",
                "TRUNCATE",
            ]
            sql_upper = sql_query.upper()

            for keyword in forbidden_keywords:
                if keyword in sql_upper:
                    return False

            # Vérifier que c'est une requête SELECT
            if not sql_upper.strip().startswith("SELECT"):
                return False

            # Vérifier la présence de la table autorisée
            if "sensitive_areas" not in sql_query.lower():
                return False

            return True

        except Exception:
            return False

    def _determine_query_type(self, sql_query: str) -> QueryType:
        """Détermine le type de requête"""
        sql_upper = sql_query.upper()

        if "COUNT(" in sql_upper or "SUM(" in sql_upper or "AVG(" in sql_upper:
            return QueryType.AGGREGATE
        elif "JOIN" in sql_upper:
            return QueryType.JOIN
        elif "COUNT(*)" in sql_upper:
            return QueryType.COUNT
        else:
            return QueryType.SELECT

    def _calculate_confidence(self, question: str, sql_query: str) -> float:
        """Calcule la confiance dans la requête générée"""
        confidence = 0.5  # Base

        question_lower = question.lower()
        sql_lower = sql_query.lower()

        # Correspondance des mots-clés
        keywords = ["count", "sum", "group", "order", "where", "région", "catégorie"]
        matches = sum(1 for kw in keywords if kw in question_lower and kw in sql_lower)
        confidence += (matches / len(keywords)) * 0.3

        # Bonus pour les requêtes bien formées
        if "GROUP BY" in sql_query.upper():
            confidence += 0.1
        if "ORDER BY" in sql_query.upper():
            confidence += 0.1
        if any(
            alias in sql_query.lower() for alias in ["as count", "as nb", "as total"]
        ):
            confidence += 0.1

        return min(confidence, 1.0)

    def _generate_explanation(self, question: str, sql_query: str) -> str:
        """Génère une explication de la requête"""
        try:
            prompt = f"""
            Expliquez en français simple cette requête SQL:
            
            Question: {question}
            SQL: {sql_query}
            
            Explication (maximum 2 phrases):
            """

            response = self.sql_llm(prompt)
            return response.strip()

        except Exception:
            return "Requête SQL générée pour répondre à votre question."

    def generate_visualization_code(
        self, query_result: Dict[str, Any], question: str
    ) -> VisualizationResult:
        """Génère le code de visualisation"""
        try:
            # Analyser les données pour déterminer le type de graphique
            chart_type = self._determine_chart_type(query_result, question)

            # Générer le code avec le LLM
            code = self._generate_viz_code_with_llm(query_result, question, chart_type)

            # Valider le code
            if not self._validate_python_code(code):
                code = self._generate_fallback_code(query_result, chart_type)

            return VisualizationResult(
                chart_config=None,  # À implémenter si nécessaire
                python_code=code,
                success=True,
            )

        except Exception as e:
            logger.error(f"Erreur génération visualisation: {e}")
            return VisualizationResult(
                chart_config=None, python_code="", success=False, error_message=str(e)
            )

    def _determine_chart_type(self, query_result: Dict[str, Any], question: str) -> str:
        """Détermine le type de graphique optimal"""
        question_lower = question.lower()

        if any(
            word in question_lower
            for word in ["répartition", "proportion", "pourcentage"]
        ):
            return "pie"
        elif any(
            word in question_lower for word in ["évolution", "temps", "mois", "année"]
        ):
            return "line"
        elif any(word in question_lower for word in ["corrélation", "relation"]):
            return "scatter"
        else:
            return "bar"

    def _generate_viz_code_with_llm(
        self, query_result: Dict[str, Any], question: str, chart_type: str
    ) -> str:
        """Génère le code de visualisation avec le LLM"""
        columns = query_result.get("columns", [])

        prompt = f"""
        Générez du code Python avec Plotly pour visualiser ces données:
        
        Question: {question}
        Colonnes: {', '.join(columns)}
        Type de graphique suggéré: {chart_type}
        
        Instructions:
        1. Utilisez plotly.express
        2. Le DataFrame est dans la variable 'data'
        3. Créez une fonction create_plot(data) qui retourne la figure
        4. Ajoutez un titre approprié
        5. Gérez le cas où les données sont vides
        
        Code Python:
        """

        response = self.viz_llm(prompt)

        # Nettoyer la réponse
        code = response.strip()
        if "" in code:
            code = code.split("python")[1].split("")[0]

        return code.strip()

    def _validate_python_code(self, code: str) -> bool:
        """Valide le code Python généré"""
        try:
            import ast

            # Vérifications de sécurité
            forbidden = ["import os", "import sys", "exec", "eval", "__import__"]
            for forbidden_item in forbidden:
                if forbidden_item in code:
                    return False

            # Vérifier la syntaxe
            ast.parse(code)

            # Vérifier la présence de la fonction
            if "def create_plot" not in code:
                return False

            return True

        except:
            return False

    def _generate_fallback_code(
        self, query_result: Dict[str, Any], chart_type: str
    ) -> str:
        """Génère un code de fallback simple"""
        columns = query_result.get("columns", [])

        if len(columns) < 2:
            return """
def create_plot(data):
    import plotly.express as px
    if data.empty:
        return px.bar(title="Aucune donnée disponible")
    return px.bar(data, title="Données")
            """

        x_col = columns[0]
        y_col = columns[1]

        if chart_type == "pie":
            return f"""
def create_plot(data):
    import plotly.express as px
    if data.empty:
        return px.pie(title="Aucune donnée disponible")
    return px.pie(data, values='{y_col}', names='{x_col}', title="Répartition")
            """
        else:
            return f"""
def create_plot(data):
    import plotly.express as px
    if data.empty:
        return px.bar(title="Aucune donnée disponible")
    return px.bar(data, x='{x_col}', y='{y_col}', title="Analyse des données")
            """

    def get_service_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du service"""
        avg_time = (
            self.total_processing_time / self.request_count
            if self.request_count > 0
            else 0
        )
        error_rate = (
            self.error_count / self.request_count if self.request_count > 0 else 0
        )

        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "avg_processing_time": avg_time,
            "total_processing_time": self.total_processing_time,
            "models_loaded": {
                "sql_llm": self.sql_llm is not None,
                "viz_llm": self.viz_llm is not None,
                "embeddings": self.embeddings is not None,
                "knowledge_base": self.knowledge_base is not None,
            },
        }

    def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé du service"""
        try:
            # Test simple avec le modèle SQL
            test_response = self.sql_llm("SELECT 1")

            return {
                "status": "healthy",
                "models_responsive": True,
                "last_check": time.time(),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "models_responsive": False,
                "last_check": time.time(),
            }


# Instance globale du service IA
ai_service = AIService()
