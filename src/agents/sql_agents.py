# src/agents/sql_agent.py
import re
import ast
import sqlparse
from typing import List, Dict, Any, Optional, Tuple
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
import logging

from ..core.models import QueryRequest, SQLQuery, QueryType, QueryResult
from ..core.exceptions import AIModelException, QueryExecutionException, ValidationException
from ..config.settings import settings
from ..database.repositories import DatabaseRepository
from ..services.cache_service import CacheService

logger = logging.getLogger(__name__)

class SQLAgent:
    """Agent SQL avancé avec RAG et validation intelligente"""
    
    def __init__(self, db_repository: DatabaseRepository, cache_service: CacheService):
        self.db_repository = db_repository
        self.cache_service = cache_service
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.knowledge_base = None
        self.llm = None
        self._initialize_llm()
        self._build_knowledge_base()
    
    def _initialize_llm(self):
        """Initialise le modèle de langage"""
        try:
            self.llm = HuggingFaceHub(
                repo_id=settings.ai.sql_model,
                model_kwargs={
                    "temperature": settings.ai.temperature,
                    "max_new_tokens": settings.ai.max_tokens,
                    "do_sample": True,
                    "top_p": 0.95,
                    "top_k": 50,
                    "repetition_penalty": 1.2
                },
                huggingfacehub_api_token=settings.ai.huggingface_api_key
            )
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du LLM: {e}")
            raise AIModelException(f"Impossible d'initialiser le modèle: {e}")
    
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
            documents.append(Document(
                page_content=schema_info,
                metadata={"type": "schema", "importance": "high"}
            ))
            
            # Ajouter les exemples
            for example in sql_examples:
                documents.append(Document(
                    page_content=example,
                    metadata={"type": "example", "importance": "medium"}
                ))
            
            # Créer le vectorstore
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            
            split_docs = text_splitter.split_documents(documents)
            self.knowledge_base = FAISS.from_documents(split_docs, self.embeddings)
            
            logger.info("Base de connaissances RAG construite avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de la construction de la base de connaissances: {e}")
            raise AIModelException(f"Impossible de construire la base de connaissances: {e}")
    
    def _get_database_schema(self) -> str:
        """Récupère le schéma de la base de données"""
        try:
            schema_info = self.db_repository.get_schema_info()
            
            schema_text = """
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
            """
        ]
    
    def _get_relevant_context(self, question: str) -> str:
        """Récupère le contexte pertinent via RAG"""
        try:
            if self.knowledge_base is None:
                return ""
            
            # Recherche de similarité
            relevant_docs = self.knowledge_base.similarity_search(
                question, 
                k=3  # Top 3 documents les plus pertinents
            )
            
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            return context
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du contexte: {e}")
            return ""
    
    def _validate_sql_query(self, sql_query: str) -> Tuple[bool, str]:
        """Valide une requête SQL"""
        try:
            # Parse SQL
            parsed = sqlparse.parse(sql_query)[0]
            
            # Vérifications de sécurité
            forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
            tokens = [token.value.upper() for token in parsed.flatten()]
            
            for keyword in forbidden_keywords:
                if keyword in tokens:
                    return False, f"Mot-clé interdit détecté: {keyword}"
            
            # Vérification de la syntaxe
            if not any(token.upper() == 'SELECT' for token in tokens):
                return False, "Seules les requêtes SELECT sont autorisées"
            
            # Vérification des noms de tables
            table_names = self._extract_table_names(sql_query)
            allowed_tables = ['sensitive_areas']
            
            for table in table_names:
                if table not in allowed_tables:
                    return False, f"Table non autorisée: {table}"
            
            return True, "Requête valide"
            
        except Exception as e:
            return False, f"Erreur de validation: {str(e)}"
    
    def _extract_table_names(self, sql_query: str) -> List[str]:
        """Extrait les noms de tables d'une requête SQL"""
        try:
            parsed = sqlparse.parse(sql_query)[0]
            table_names = []
            
            from_seen = False
            for token in parsed.flatten():
                if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'FROM':
                    from_seen = True
                elif from_seen and token.ttype is None and token.value.strip():
                    table_names.append(token.value.strip())
                    from_seen = False
            
            return table_names
            
        except Exception:
            return []
    
    def _determine_query_type(self, sql_query: str) -> QueryType:
        """Détermine le type de requête SQL"""
        sql_upper = sql_query.upper()
        
        if 'COUNT(' in sql_upper or 'SUM(' in sql_upper or 'AVG(' in sql_upper:
            return QueryType.AGGREGATE
        elif 'JOIN' in sql_upper:
            return QueryType.JOIN
        elif 'COUNT(*)' in sql_upper:
            return QueryType.COUNT
        else:
            return QueryType.SELECT
    
    def _estimate_query_confidence(self, question: str, sql_query: str) -> float:
        """Estime la confiance dans la requête générée"""
        # Logique d'estimation de confiance basée sur:
        # - Correspondance des mots-clés
        # - Complexité de la requête
        # - Présence de toutes les tables nécessaires
        
        confidence = 0.5  # Base
        
        # Mots-clés importants présents
        question_lower = question.lower()
        sql_lower = sql_query.lower()
        
        keyword_matches = 0
        important_keywords = ['count', 'sum', 'group', 'order', 'where', 'région', 'catégorie', 'espèce']
        
        for keyword in important_keywords:
            if keyword in question_lower and keyword in sql_lower:
                keyword_matches += 1
        
        confidence += (keyword_matches / len(important_keywords)) * 0.3
        
        # Bonus pour les requêtes bien formées
        if 'GROUP BY' in sql_query.upper():
            confidence += 0.1
        if 'ORDER BY' in sql_query.upper():
            confidence += 0.1
        
        return min(confidence, 1.0)
    
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
            
            # Création du prompt amélioré
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
                3. Respectez exactement les noms de colon