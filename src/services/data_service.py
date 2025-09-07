# src/services/data_service.py
import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
from pathlib import Path

from src.database.repositories import DatabaseRepository
from src.core.models import QueryResult
from src.core.exceptions import DatabaseException, DataExtractionException
from src.services.cache_service import cache_service
from data_retrieve.data_retrieve import DataProcessor

logger = logging.getLogger(__name__)


class DataService:
    """Service de gestion des données"""

    def __init__(self):
        self.db_repository = DatabaseRepository()
        self.data_processor = DataProcessor()
        self.cache_service = cache_service

    def execute_query(self, query: str, params: tuple = None) -> QueryResult:
        """Exécute une requête SQL et retourne les résultats"""
        try:
            # Vérification du cache
            cache_key = f"query_{hash(query + str(params or ''))}"
            cached_result = self.cache_service.get(cache_key)

            if cached_result:
                logger.info("Résultat trouvé dans le cache")
                return QueryResult(**cached_result)

            # Validation de la requête
            if not self.db_repository.validate_query(query):
                raise DatabaseException("Requête SQL invalide ou non autorisée")

            # Exécution
            result = self.db_repository.execute_raw_query(query, params)

            # Mise en cache
            self.cache_service.set(cache_key, result.dict(), ttl=1800)  # 30 minutes

            logger.info(
                f"Requête exécutée: {result.row_count} résultats en {result.execution_time:.3f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Erreur exécution requête: {e}")
            raise DatabaseException(f"Erreur lors de l'exécution: {e}")

    def get_quick_stats(self) -> Dict[str, Any]:
        """Retourne des statistiques rapides"""
        try:
            cache_key = "quick_stats"
            cached_stats = self.cache_service.get(cache_key)

            if cached_stats:
                return cached_stats

            stats = {}

            # Total des zones
            total_query = "SELECT COUNT(*) as count FROM sensitive_areas"
            total_result = self.db_repository.execute_raw_query(total_query)
            stats["total_zones"] = (
                total_result.data[0]["count"] if total_result.data else 0
            )

            # Total des régions
            regions_query = "SELECT COUNT(DISTINCT region) as count FROM sensitive_areas WHERE region != ''"
            regions_result = self.db_repository.execute_raw_query(regions_query)
            stats["total_regions"] = (
                regions_result.data[0]["count"] if regions_result.data else 0
            )

            # Total des structures
            structures_query = "SELECT COUNT(DISTINCT structure) as count FROM sensitive_areas WHERE structure != ''"
            structures_result = self.db_repository.execute_raw_query(structures_query)
            stats["total_structures"] = (
                structures_result.data[0]["count"] if structures_result.data else 0
            )

            # Répartition par catégorie
            category_query = "SELECT category, COUNT(*) as count FROM sensitive_areas GROUP BY category"
            category_result = self.db_repository.execute_raw_query(category_query)
            stats["category_distribution"] = (
                {row["category"]: row["count"] for row in category_result.data}
                if category_result.data
                else {}
            )

            # Mise en cache pour 10 minutes
            self.cache_service.set(cache_key, stats, ttl=600)

            return stats

        except Exception as e:
            logger.error(f"Erreur statistiques rapides: {e}")
            return {}

    def get_data_summary(self) -> Dict[str, Any]:
        """Retourne un résumé complet des données"""
        try:
            summary = {
                "data_stats": self.get_quick_stats(),
                "last_update": self._get_last_update_time(),
                "database_info": self.get_database_info(),
                "quality_metrics": self._get_data_quality_metrics(),
            }

            return summary

        except Exception as e:
            logger.error(f"Erreur résumé données: {e}")
            return {}

    def _get_last_update_time(self) -> Optional[datetime]:
        """Retourne la date de dernière mise à jour"""
        try:
            query = "SELECT MAX(update_datetime) as last_update FROM sensitive_areas"
            result = self.db_repository.execute_raw_query(query)

            if result.data and result.data[0]["last_update"]:
                return pd.to_datetime(result.data[0]["last_update"])

            return None

        except Exception as e:
            logger.error(f"Erreur dernière mise à jour: {e}")
            return None

    def get_database_info(self) -> Dict[str, Any]:
        """Retourne les informations de la base de données"""
        try:
            db_path = Path("data/sensitive_areas.db")

            info = {
                "path": str(db_path),
                "exists": db_path.exists(),
                "size": None,
                "tables": [],
                "last_backup": None,
            }

            if db_path.exists():
                # Taille du fichier
                size_bytes = db_path.stat().st_size
                if size_bytes > 1024 * 1024:
                    info["size"] = f"{size_bytes / (1024 * 1024):.1f} MB"
                else:
                    info["size"] = f"{size_bytes / 1024:.1f} KB"

                # Liste des tables
                info["tables"] = self.db_repository.get_table_list()

            return info

        except Exception as e:
            logger.error(f"Erreur informations base de données: {e}")
            return {}

    def _get_data_quality_metrics(self) -> Dict[str, Any]:
        """Calcule les métriques de qualité des données"""
        try:
            metrics = {}

            # Pourcentage de valeurs nulles par colonne
            null_query = """
                SELECT 
                    SUM(CASE WHEN name IS NULL OR name = '' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as name_null_pct,
                    SUM(CASE WHEN description IS NULL OR description = '' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as description_null_pct,
                    SUM(CASE WHEN region IS NULL OR region = '' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as region_null_pct,
                    SUM(CASE WHEN structure IS NULL OR structure = '' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as structure_null_pct
                FROM sensitive_areas
            """

            result = self.db_repository.execute_raw_query(null_query)
            if result.data:
                metrics["null_percentages"] = result.data[0]

            # Doublons potentiels
            duplicates_query = """
                SELECT COUNT(*) - COUNT(DISTINCT name || region || structure) as potential_duplicates
                FROM sensitive_areas
            """

            dup_result = self.db_repository.execute_raw_query(duplicates_query)
            if dup_result.data:
                metrics["potential_duplicates"] = dup_result.data[0][
                    "potential_duplicates"
                ]

            return metrics

        except Exception as e:
            logger.error(f"Erreur métriques qualité: {e}")
            return {}

    def refresh_data(self) -> Dict[str, Any]:
        """Actualise les données depuis la source"""
        try:
            logger.info("Début de l'actualisation des données")

            # Exécution du processeur de données
            self.data_processor.run()

            # Invalidation du cache
            self.cache_service.clear()

            # Nouvelles statistiques
            new_stats = self.get_quick_stats()

            result = {
                "success": True,
                "message": "Données actualisées avec succès",
                "records_processed": new_stats.get("total_zones", 0),
                "last_update": datetime.now(),
                "stats": new_stats,
            }

            logger.info(
                f"Actualisation terminée: {result['records_processed']} enregistrements"
            )

            return result

        except Exception as e:
            logger.error(f"Erreur actualisation données: {e}")
            return {
                "success": False,
                "message": f"Erreur lors de l'actualisation: {str(e)}",
                "error": str(e),
            }

    def export_data(
        self, format_type: str = "csv", query: str = None
    ) -> Dict[str, Any]:
        """Exporte les données dans différents formats"""
        try:
            # Requête par défaut
            if not query:
                query = "SELECT * FROM sensitive_areas"

            # Exécution de la requête
            result = self.execute_query(query)

            if not result.data:
                return {"success": False, "message": "Aucune donnée à exporter"}

            # Conversion en DataFrame
            df = pd.DataFrame(result.data)

            # Export selon le format
            if format_type.lower() == "csv":
                data = df.to_csv(index=False)
                mime_type = "text/csv"
                extension = "csv"
            elif format_type.lower() == "json":
                data = df.to_json(orient="records", indent=2)
                mime_type = "application/json"
                extension = "json"
            elif format_type.lower() == "excel":
                import io

                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name="Data", index=False)
                data = buffer.getvalue()
                mime_type = (
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                extension = "xlsx"
            else:
                raise ValueError(f"Format non supporté: {format_type}")

            return {
                "success": True,
                "data": data,
                "mime_type": mime_type,
                "extension": extension,
                "row_count": len(df),
                "filename": f'export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{extension}',
            }

        except Exception as e:
            logger.error(f"Erreur export données: {e}")
            return {"success": False, "message": f"Erreur lors de l'export: {str(e)}"}


# Instance globale
data_service = DataService()
