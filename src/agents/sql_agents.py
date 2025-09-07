# src/agents/sql_agent.py
import re
import ast
import sqlparse
import traceback
from typing import List, Tuple, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import logging

# Alternatives pour le LLM
try:
    from langchain_huggingface import HuggingFaceEndpoint

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from langchain_openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_community.llms import Ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from src.core.models import QueryRequest, SQLQuery, QueryType, QueryResult
from src.core.exceptions import (
    AIModelException,
    QueryExecutionException,
    ValidationException,
)
from src.config.settings import settings
from src.database.repositories import DatabaseRepository
from src.services.cache_service import CacheService

logger = logging.getLogger(__name__)


class SQLAgent:
    """Agent SQL avancé avec RAG et validation intelligente"""

    def __init__(self, db_repository: DatabaseRepository, cache_service: CacheService):
        self.db_repository = db_repository
        self.cache_service = cache_service
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.knowledge_base = None
        self.llm = None
        self._initialize_llm()
        self._build_knowledge_base()

    def _initialize_llm(self):
        """Initialise le modèle de langage avec fallback sur plusieurs providers"""
        try:
            # Option 1: Essayer HuggingFace Endpoint avec configuration améliorée
            if (
                HF_AVAILABLE
                and hasattr(settings.ai, "huggingface_api_key")
                and settings.ai.huggingface_api_key
            ):
                try:
                    logger.info(
                        "Tentative d'initialisation avec HuggingFace Endpoint..."
                    )

                    # Configuration plus robuste pour HuggingFace
                    self.llm = HuggingFaceEndpoint(
                        repo_id=getattr(
                            settings.ai, "sql_model", "mistralai/Mistral-7B-v0.13"
                        ),
                        temperature=getattr(settings.ai, "temperature", 0.1),
                        max_new_tokens=getattr(settings.ai, "max_tokens", 512),
                        top_k=40,
                        top_p=0.95,
                        repetition_penalty=1.03,
                        huggingfacehub_api_token=settings.ai.huggingface_api_key,
                        return_full_text=False,
                    )
                    logger.info("HuggingFace Endpoint initialisé avec succès")
                    return
                except Exception as hf_error:
                    logger.warning(f"Échec HuggingFace Endpoint: {hf_error}")

            # Option 2: Fallback vers OpenAI
            if (
                OPENAI_AVAILABLE
                and hasattr(settings.ai, "openai_api_key")
                and settings.ai.openai_api_key
            ):
                try:
                    logger.info("Tentative d'initialisation avec OpenAI...")
                    self.llm = OpenAI(
                        openai_api_key=settings.ai.openai_api_key,
                        model_name=getattr(
                            settings.ai, "openai_model", "gpt-3.5-turbo-instruct"
                        ),
                        temperature=getattr(settings.ai, "temperature", 0.1),
                        max_tokens=getattr(settings.ai, "max_tokens", 512),
                    )
                    logger.info("OpenAI initialisé avec succès")
                    return
                except Exception as openai_error:
                    logger.warning(f"Échec OpenAI: {openai_error}")

            # Option 3: Fallback vers Ollama (local)
            if OLLAMA_AVAILABLE:
                try:
                    logger.info("Tentative d'initialisation avec Ollama...")
                    self.llm = Ollama(
                        model=getattr(settings.ai, "ollama_model", "llama2"),
                        temperature=getattr(settings.ai, "temperature", 0.1),
                    )
                    # Test de connexion
                    test_response = self.llm.invoke("Test")
                    logger.info("Ollama initialisé avec succès")
                    return
                except Exception as ollama_error:
                    logger.warning(f"Échec Ollama: {ollama_error}")

            # Option 4: LLM simulé pour développement
            logger.warning("Tous les LLM ont échoué, utilisation du LLM simulé")
            self.llm = MockLLM()

        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du LLM: {e}")
            logger.error(f"Traceback complet: {traceback.format_exc()}")
            # Utiliser le LLM simulé en dernier recours
            self.llm = MockLLM()

    def _build_knowledge_base(self):
        """Construit la base de connaissances RAG"""
        try:
            # Récupération du schéma de la base de données
            schema_info = self._get_database_schema()

            # Exemples de requêtes SQL avec leurs explications
            sql_examples = self._get_sql_examples()

            # Créer des documents pour la base de connaissances
            documents = []

            # Ajouter le schéma
            documents.append(
                Document(
                    page_content=schema_info,
                    metadata={"type": "schema", "importance": "high"},
                )
            )

            # Ajouter les exemples
            for example in sql_examples:
                documents.append(
                    Document(
                        page_content=example,
                        metadata={"type": "example", "importance": "medium"},
                    )
                )

            # Créer le vectorstore
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50
            )

            split_docs = text_splitter.split_documents(documents)
            self.knowledge_base = FAISS.from_documents(split_docs, self.embeddings)

            logger.info("Base de connaissances RAG construite avec succès")

        except Exception as e:
            logger.error(
                f"Erreur lors de la construction de la base de connaissances: {e}"
            )
            logger.error(f"Traceback complet: {traceback.format_exc()}")
            # Ne pas lever d'exception ici, continuer sans RAG
            logger.warning("Continuation sans base de connaissances RAG")

    def _get_database_schema(self) -> str:
        """Récupère le schéma de la base de données"""
        try:
            schema_info = self.db_repository.get_schema_info()

            schema_text = f"""
            BASE DE DONNÉES: sensitive_areas

            TABLES ET COLONNES:

            Table: sensitive_areas
            - id (INTEGER, PRIMARY KEY): Identifiant unique de la zone sensible
            - name (TEXT): Nom de la zone sensible
            - description (TEXT): Description détaillée de la zone
            - structure (TEXT): Organisation responsable des données
            - species_id (INTEGER): Identifiant de l'espèce (NULL pour zones réglementaires)
            - practices (TEXT): Activités sportives concernées
            - create_datetime (TIMESTAMP): Date de création
            - update_datetime (TIMESTAMP): Date de dernière mise à jour
            - region (TEXT): Région géographique
            - departement (TEXT): Département
            - pays (TEXT): Pays
            - category (TEXT): Catégorie ('zone reglementaire' ou 'Espece')

            RELATIONS:
            - species_id peut être NULL (zones réglementaires)
            - category est dérivée de species_id (NULL = 'zone reglementaire', NOT NULL = 'Espece')

            EXEMPLES DE DONNÉES:
            - Régions: Auvergne-Rhône-Alpes, Provence-Alpes-Côte d'Azur, etc.
            - Structures: Parc national des Écrins, LPO, etc.
            - Practices: Randonnée pédestre, VTT, Escalade, etc.

            {schema_info}
            """

            return schema_text

        except Exception as e:
            logger.error(f"Erreur lors de la récupération du schéma: {e}")
            return "Schéma non disponible"

    def _get_sql_examples(self) -> List[str]:
        """Retourne des exemples de requêtes SQL avec explications"""
        return [
            """
            Question: Combien y a-t-il de zones par catégorie?
            SQL: SELECT category, COUNT(*) as count FROM sensitive_areas GROUP BY category;
            Explication: Utilise GROUP BY pour compter les zones par catégorie.
            """,
            """
            Question: Quelles sont les 5 régions avec le plus de zones sensibles?
            SQL: SELECT region, COUNT(*) as nb_zones FROM sensitive_areas GROUP BY region ORDER BY nb_zones DESC LIMIT 5;
            Explication: GROUP BY région, tri décroissant et limitation à 5 résultats.
            """,
            """
            Question: Quelles structures ont créé des zones liées aux espèces?
            SQL: SELECT DISTINCT structure FROM sensitive_areas WHERE category = 'Espece';
            Explication: Filtre sur category = 'Espece' pour les zones liées aux espèces.
            """,
            """
            Question: Répartition des zones par mois de création?
            SQL: SELECT strftime('%Y-%m', create_datetime) as mois, COUNT(*) as nb FROM sensitive_areas GROUP BY mois ORDER BY mois;
            Explication: Extraction du mois avec strftime pour grouper par période.
            """,
        ]

    def _get_relevant_context(self, question: str) -> str:
        """Récupère le contexte pertinent via RAG"""
        try:
            if self.knowledge_base is None:
                return self._get_fallback_context()

            # Recherche de similarité
            relevant_docs = self.knowledge_base.similarity_search(
                question, k=3  # Top 3 documents les plus pertinents
            )

            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            return context

        except Exception as e:
            logger.error(f"Erreur lors de la récupération du contexte: {e}")
            return self._get_fallback_context()

    def _get_fallback_context(self) -> str:
        """Contexte de fallback si RAG non disponible"""
        return """
        BASE DE DONNÉES: sensitive_areas
        Table: sensitive_areas
        Colonnes: id, name, description, structure, species_id, practices, create_datetime, update_datetime, region, departement, pays, category
        """

    def _validate_sql_query(self, sql_query: str) -> Tuple[bool, str]:
        """Valide une requête SQL"""
        try:
            if not sql_query or not sql_query.strip():
                return False, "Requête SQL vide"

            # Parse SQL
            parsed = sqlparse.parse(sql_query)[0]

            # Vérifications de sécurité
            forbidden_keywords = [
                "DROP",
                "DELETE",
                "UPDATE",
                "INSERT",
                "ALTER",
                "CREATE",
                "TRUNCATE",
                "EXEC",
            ]
            tokens = [token.value.upper() for token in parsed.flatten()]

            for keyword in forbidden_keywords:
                if keyword in tokens:
                    return False, f"Mot-clé interdit détecté: {keyword}"

            # Vérification de la syntaxe
            if not any(token.upper() == "SELECT" for token in tokens):
                return False, "Seules les requêtes SELECT sont autorisées"

            return True, "Requête valide"

        except Exception as e:
            return False, f"Erreur de validation: {str(e)}"

    def _determine_query_type(self, sql_query: str) -> QueryType:
        """Détermine le type de requête SQL"""
        sql_upper = sql_query.upper()

        if "COUNT(" in sql_upper or "SUM(" in sql_upper or "AVG(" in sql_upper:
            return QueryType.AGGREGATE
        elif "JOIN" in sql_upper:
            return QueryType.JOIN
        elif "COUNT(*)" in sql_upper:
            return QueryType.COUNT
        else:
            return QueryType.SELECT

    def _parse_llm_response(self, response_text: str) -> dict:
        """Parse la réponse du LLM pour extraire les informations structurées"""
        try:
            # Recherche des patterns dans la réponse
            sql_patterns = [
                r"(?:Requête SQL|SQL|Query):\s*(.+?)(?:\n|Explication|$)",
                r"SELECT\s+.+?(?:\n|$)",
            ]

            sql_query = ""
            for pattern in sql_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                if match:
                    sql_query = match.group(1) if match.lastindex else match.group(0)
                    sql_query = sql_query.strip().strip(";").strip()
                    if sql_query:
                        break

            # Si aucune requête trouvée, essayer de générer une requête simple
            if not sql_query:
                sql_query = "SELECT * FROM sensitive_areas LIMIT 10"
                logger.warning(
                    "Aucune requête SQL trouvée dans la réponse, utilisation d'une requête par défaut"
                )

            explanation = "Requête générée automatiquement"
            confidence = 0.7
            query_type = self._determine_query_type(sql_query)
            estimated_rows = 10

            return {
                "query": sql_query,
                "explanation": explanation,
                "confidence": confidence,
                "query_type": query_type,
                "estimated_rows": estimated_rows,
                "raw_response": response_text,
            }

        except Exception as e:
            logger.error(f"Erreur lors du parsing de la réponse LLM: {e}")
            # Retour par défaut en cas d'erreur
            return {
                "query": "SELECT * FROM sensitive_areas LIMIT 5",
                "explanation": "Requête par défaut suite à une erreur de parsing",
                "confidence": 0.5,
                "query_type": QueryType.SELECT,
                "estimated_rows": 5,
                "raw_response": response_text,
            }

    def generate_sql_query(self, request: QueryRequest) -> SQLQuery:
        """Génère une requête SQL à partir d'une question"""
        try:
            # Vérification du cache
            cache_key = f"sql_query_{hash(request.question)}"
            cached_result = self.cache_service.get(cache_key)

            if cached_result:
                logger.info(f"Résultat trouvé dans le cache pour: {request.question}")
                return SQLQuery(**cached_result)

            # Récupération du contexte RAG
            context = self._get_relevant_context(request.question)

            # Création du prompt simplifié
            prompt = f"""
Vous êtes un expert SQL. Générez une requête SELECT pour la table 'sensitive_areas'.

Contexte: {context}

Question: {request.question}

Répondez avec une requête SQL SELECT valide uniquement:
"""

            # Générer la réponse
            logger.info(f"Génération SQL pour: {request.question}")

            try:
                logger.info(f"Prompt LLM: {prompt}")
                response_text = self.llm.invoke(prompt)
                logger.info(f"Réponse brute du LLM: {response_text}")
            except Exception as llm_error:
                logger.error(f"Erreur LLM: {llm_error}")
                # Utiliser une requête par défaut
                response_text = f"SELECT * FROM sensitive_areas WHERE name LIKE '%{request.question}%' LIMIT 10"

            # Parser la réponse
            parsed_response = self._parse_llm_response(response_text)

            # Validation de la requête
            is_valid, validation_message = self._validate_sql_query(
                parsed_response["query"]
            )
            if not is_valid:
                logger.warning(f"Requête invalide: {validation_message}")
                # Utiliser une requête par défaut
                parsed_response["query"] = "SELECT * FROM sensitive_areas LIMIT 10"
                parsed_response["explanation"] = (
                    f"Requête par défaut (erreur: {validation_message})"
                )

            # Créer l'objet SQLQuery
            sql_query = SQLQuery(
                query=parsed_response["query"],
                explanation=parsed_response["explanation"],
                confidence=parsed_response["confidence"],
                query_type=parsed_response["query_type"],
                estimated_rows=parsed_response["estimated_rows"],
                source="llm_generated",
            )

            # Mise en cache
            try:
                self.cache_service.set(cache_key, sql_query.dict())
            except Exception as cache_error:
                logger.warning(f"Erreur de mise en cache: {cache_error}")

            logger.info(f"Requête SQL générée: {sql_query.query}")
            return sql_query

        except Exception as e:
            logger.error(f"Erreur lors de la génération de la requête SQL: {str(e)}")
            logger.error(f"Type d'erreur: {type(e).__name__}")
            logger.error(f"Traceback complet: {traceback.format_exc()}")

            # Retourner une requête par défaut au lieu de lever une exception
            return SQLQuery(
                query="SELECT * FROM sensitive_areas LIMIT 5",
                explanation=f"Requête par défaut suite à une erreur: {str(e)}",
                confidence=0.1,
                query_type=QueryType.SELECT,
                estimated_rows=5,
                source="fallback",
            )


class MockLLM:
    """LLM simulé pour développement/test"""

    def invoke(self, prompt: str) -> str:
        """Simule une réponse de LLM basique"""
        prompt_lower = prompt.lower()

        if "count" in prompt_lower or "combien" in prompt_lower:
            return "SELECT category, COUNT(*) as count FROM sensitive_areas GROUP BY category;"
        elif "région" in prompt_lower or "region" in prompt_lower:
            return "SELECT region, COUNT(*) as nb_zones FROM sensitive_areas GROUP BY region ORDER BY nb_zones DESC LIMIT 10;"
        elif "structure" in prompt_lower:
            return "SELECT DISTINCT structure FROM sensitive_areas LIMIT 10;"
        else:
            return "SELECT * FROM sensitive_areas LIMIT 10;"
