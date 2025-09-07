# src/database/repositories.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime

from .connection import db_connection
from ..core.exceptions import DatabaseException
from ..core.models import QueryResult

logger = logging.getLogger(__name__)


class BaseRepository(ABC):
    """Repository de base avec méthodes communes"""

    def __init__(self, db_connection_instance=None):
        self.db = db_connection_instance or db_connection

    @abstractmethod
    def get_table_name(self) -> str:
        """Retourne le nom de la table"""
        pass

    def find_by_id(self, id_value: Any) -> Optional[Dict[str, Any]]:
        """Trouve un enregistrement par ID"""
        try:
            query = f"SELECT * FROM {self.get_table_name()} WHERE id = ?"
            results = self.db.execute_query(query, (id_value,))
            return results[0] if results else None
        except Exception as e:
            raise DatabaseException(f"Erreur lors de la recherche par ID: {e}")

    def find_all(self, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """Trouve tous les enregistrements"""
        try:
            query = f"SELECT * FROM {self.get_table_name()}"
            if limit:
                query += f" LIMIT {limit} OFFSET {offset}"
            return self.db.execute_query(query)
        except Exception as e:
            raise DatabaseException(f"Erreur lors de la recherche: {e}")

    def count(self, where_clause: str = None, params: tuple = None) -> int:
        """Compte les enregistrements"""
        try:
            query = f"SELECT COUNT(*) as count FROM {self.get_table_name()}"
            if where_clause:
                query += f" WHERE {where_clause}"

            result = self.db.execute_query(query, params)
            return result[0]["count"] if result else 0
        except Exception as e:
            raise DatabaseException(f"Erreur lors du comptage: {e}")

    def exists(self, where_clause: str, params: tuple) -> bool:
        """Vérifie si un enregistrement existe"""
        return self.count(where_clause, params) > 0


class SensitiveAreasRepository(BaseRepository):
    """Repository pour les zones sensibles"""

    def get_table_name(self) -> str:
        return "sensitive_areas"

    def find_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Trouve les zones par catégorie"""
        try:
            query = """
                SELECT * FROM sensitive_areas 
                WHERE category = ? 
                ORDER BY name
            """
            return self.db.execute_query(query, (category,))
        except Exception as e:
            raise DatabaseException(f"Erreur lors de la recherche par catégorie: {e}")

    def find_by_region(self, region: str) -> List[Dict[str, Any]]:
        """Trouve les zones par région"""
        try:
            query = """
                SELECT * FROM sensitive_areas 
                WHERE region = ? 
                ORDER BY name
            """
            return self.db.execute_query(query, (region,))
        except Exception as e:
            raise DatabaseException(f"Erreur lors de la recherche par région: {e}")

    def get_categories_stats(self) -> List[Dict[str, Any]]:
        """Obtient les statistiques par catégorie"""
        try:
            query = """
                SELECT 
                    category,
                    COUNT(*) as count,
                    COUNT(DISTINCT region) as regions_count,
                    COUNT(DISTINCT structure) as structures_count
                FROM sensitive_areas 
                GROUP BY category
                ORDER BY count DESC
            """
            return self.db.execute_query(query)
        except Exception as e:
            raise DatabaseException(f"Erreur lors de la récupération des stats: {e}")

    def get_regions_stats(self) -> List[Dict[str, Any]]:
        """Obtient les statistiques par région"""
        try:
            query = """
                SELECT 
                    region,
                    COUNT(*) as total_zones,
                    COUNT(CASE WHEN category = 'Espece' THEN 1 END) as especes_zones,
                    COUNT(CASE WHEN category = 'zone reglementaire' THEN 1 END) as reglementaires_zones
                FROM sensitive_areas 
                GROUP BY region
                ORDER BY total_zones DESC
            """
            return self.db.execute_query(query)
        except Exception as e:
            raise DatabaseException(
                f"Erreur lors de la récupération des stats régions: {e}"
            )

    def get_structures_stats(self) -> List[Dict[str, Any]]:
        """Obtient les statistiques par structure"""
        try:
            query = """
                SELECT 
                    structure,
                    COUNT(*) as zones_count,
                    COUNT(DISTINCT region) as regions_covered
                FROM sensitive_areas 
                GROUP BY structure
                ORDER BY zones_count DESC
            """
            return self.db.execute_query(query)
        except Exception as e:
            raise DatabaseException(
                f"Erreur lors de la récupération des stats structures: {e}"
            )

    def get_practices_stats(self) -> List[Dict[str, Any]]:
        """Obtient les statistiques par pratique"""
        try:
            query = """
                SELECT 
                    practices,
                    COUNT(*) as zones_count
                FROM sensitive_areas 
                WHERE practices IS NOT NULL 
                GROUP BY practices
                ORDER BY zones_count DESC
            """
            return self.db.execute_query(query)
        except Exception as e:
            raise DatabaseException(
                f"Erreur lors de la récupération des stats pratiques: {e}"
            )

    def get_creation_timeline(self) -> List[Dict[str, Any]]:
        """Obtient la timeline de création des zones"""
        try:
            query = """
                SELECT 
                    DATE(create_datetime) as date,
                    COUNT(*) as zones_created
                FROM sensitive_areas 
                GROUP BY DATE(create_datetime)
                ORDER BY date
            """
            return self.db.execute_query(query)
        except Exception as e:
            raise DatabaseException(
                f"Erreur lors de la récupération de la timeline: {e}"
            )

    def search_zones(
        self,
        search_term: str = None,
        category: str = None,
        region: str = None,
        structure: str = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Recherche avancée de zones"""
        try:
            conditions = []
            params = []

            if search_term:
                conditions.append("(name LIKE ? OR description LIKE ?)")
                params.extend([f"%{search_term}%", f"%{search_term}%"])

            if category:
                conditions.append("category = ?")
                params.append(category)

            if region:
                conditions.append("region = ?")
                params.append(region)

            if structure:
                conditions.append("structure = ?")
                params.append(structure)

            query = "SELECT * FROM sensitive_areas"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += f" ORDER BY name LIMIT {limit}"

            return self.db.execute_query(query, tuple(params))
        except Exception as e:
            raise DatabaseException(f"Erreur lors de la recherche: {e}")


class DatabaseRepository:
    """Repository principal pour les opérations de base de données"""

    def __init__(self, db_connection_instance=None):
        self.db = db_connection_instance or db_connection
        self.sensitive_areas = SensitiveAreasRepository(self.db)

    def execute_raw_query(self, query: str, params: tuple = None) -> QueryResult:
        """Exécute une requête SQL brute et retourne un QueryResult"""
        try:
            import time
            import hashlib

            start_time = time.time()
            results = self.db.execute_query(query, params)
            execution_time = time.time() - start_time

            # Créer un hash de la requête
            query_hash = hashlib.md5(query.encode()).hexdigest()

            # Extraire les colonnes
            columns = list(results[0].keys()) if results else []

            return QueryResult(
                data=results,
                columns=columns,
                row_count=len(results),
                execution_time=execution_time,
                query_hash=query_hash,
            )
        except Exception as e:
            raise DatabaseException(f"Erreur lors de l'exécution de la requête: {e}")

    def get_schema_info(self) -> Dict[str, Any]:
        """Obtient les informations de schéma"""
        try:
            return self.db.get_schema_info()
        except Exception as e:
            raise DatabaseException(f"Erreur lors de la récupération du schéma: {e}")

    def get_table_list(self) -> List[str]:
        """Obtient la liste des tables"""
        try:
            return self.db.get_table_list()
        except Exception as e:
            raise DatabaseException(f"Erreur lors de la récupération des tables: {e}")

    def validate_query(self, query: str) -> bool:
        """Valide une requête SQL"""
        try:
            # Vérifications de base
            query_upper = query.upper().strip()

            # Vérifier que c'est une requête SELECT
            if not query_upper.startswith("SELECT"):
                return False

            # Vérifier les mots-clés interdits
            forbidden_keywords = [
                "DROP",
                "DELETE",
                "UPDATE",
                "INSERT",
                "ALTER",
                "CREATE",
                "TRUNCATE",
            ]
            for keyword in forbidden_keywords:
                if keyword in query_upper:
                    return False

            # Tentative d'exécution avec LIMIT 0 pour vérifier la syntaxe
            test_query = f"SELECT * FROM ({query}) LIMIT 0"
            self.db.execute_query(test_query)

            return True
        except Exception:
            return False

    def get_query_stats(self, query: str) -> Dict[str, Any]:
        """Obtient les statistiques d'une requête"""
        try:
            # Estimer le nombre de lignes
            count_query = f"SELECT COUNT(*) as count FROM ({query})"
            count_result = self.db.execute_query(count_query)
            estimated_rows = count_result[0]["count"] if count_result else 0

            # Analyser la complexité
            complexity = self._analyze_query_complexity(query)

            return {
                "estimated_rows": estimated_rows,
                "complexity": complexity,
                "query_type": self._get_query_type(query),
            }
        except Exception as e:
            logger.warning(f"Erreur lors de l'analyse de la requête: {e}")
            return {
                "estimated_rows": 0,
                "complexity": "unknown",
                "query_type": "unknown",
            }

    def _analyze_query_complexity(self, query: str) -> str:
        """Analyse la complexité d'une requête"""
        query_upper = query.upper()

        complexity_score = 0

        # Mots-clés qui augmentent la complexité
        if "JOIN" in query_upper:
            complexity_score += 2
        if "GROUP BY" in query_upper:
            complexity_score += 1
        if "ORDER BY" in query_upper:
            complexity_score += 1
        if "HAVING" in query_upper:
            complexity_score += 2
        if "UNION" in query_upper:
            complexity_score += 3
        if "DISTINCT" in query_upper:
            complexity_score += 1

        # Compter les sous-requêtes
        subquery_count = query_upper.count("(SELECT")
        complexity_score += subquery_count * 2

        if complexity_score <= 1:
            return "simple"
        elif complexity_score <= 3:
            return "moderate"
        else:
            return "complex"

    def _get_query_type(self, query: str) -> str:
        """Détermine le type de requête"""
        query_upper = query.upper()

        if "COUNT(" in query_upper:
            return "count"
        elif (
            "SUM(" in query_upper
            or "AVG(" in query_upper
            or "MAX(" in query_upper
            or "MIN(" in query_upper
        ):
            return "aggregate"
        elif "GROUP BY" in query_upper:
            return "group"
        elif "JOIN" in query_upper:
            return "join"
        else:
            return "select"


# Instance globale
database_repository = DatabaseRepository()
