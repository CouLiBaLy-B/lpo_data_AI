# src/database/migrations.py
import sqlite3
import logging
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

from .connection import db_connection
from ..core.exceptions import DatabaseException

logger = logging.getLogger(__name__)


class Migration:
    """Classe de base pour les migrations"""

    def __init__(self, version: str, description: str):
        self.version = version
        self.description = description
        self.created_at = datetime.now()

    def up(self, connection: sqlite3.Connection):
        """Applique la migration"""
        raise NotImplementedError("La méthode up() doit être implémentée")

    def down(self, connection: sqlite3.Connection):
        """Annule la migration"""
        raise NotImplementedError("La méthode down() doit être implémentée")


class CreateSensitiveAreasTable(Migration):
    """Migration pour créer la table sensitive_areas"""

    def __init__(self):
        super().__init__("001", "Création de la table sensitive_areas")

    def up(self, connection: sqlite3.Connection):
        """Crée la table sensitive_areas"""
        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sensitive_areas (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                structure TEXT,
                species_id INTEGER,
                practices TEXT,
                create_datetime TIMESTAMP,
                update_datetime TIMESTAMP,
                region TEXT,
                departement TEXT,
                pays TEXT
            )
        """
        )

        # Index pour améliorer les performances
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_sensitive_areas_region ON sensitive_areas(region)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_sensitive_areas_structure ON sensitive_areas(structure)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_sensitive_areas_species ON sensitive_areas(species_id)"
        )

        connection.commit()
        logger.info("Table sensitive_areas créée avec succès")

    def down(self, connection: sqlite3.Connection):
        """Supprime la table sensitive_areas"""
        cursor = connection.cursor()
        cursor.execute("DROP TABLE IF EXISTS sensitive_areas")
        connection.commit()
        logger.info("Table sensitive_areas supprimée")


class AddCategoryColumn(Migration):
    """Migration pour ajouter la colonne category"""

    def __init__(self):
        super().__init__("002", "Ajout de la colonne category")

    def up(self, connection: sqlite3.Connection):
        """Ajoute la colonne category"""
        cursor = connection.cursor()

        try:
            cursor.execute("ALTER TABLE sensitive_areas ADD COLUMN category TEXT")

            # Mise à jour des valeurs existantes
            cursor.execute(
                """
                UPDATE sensitive_areas 
                SET category = CASE 
                    WHEN species_id IS NULL THEN 'zone reglementaire'
                    ELSE 'Espece' 
                END
            """
            )

            # Index sur la nouvelle colonne
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_sensitive_areas_category ON sensitive_areas(category)"
            )

            connection.commit()
            logger.info("Colonne category ajoutée avec succès")

        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                logger.info("Colonne category existe déjà")
            else:
                raise

    def down(self, connection: sqlite3.Connection):
        """Supprime la colonne category (non supporté par SQLite)"""
        logger.warning("Suppression de colonne non supportée par SQLite")


class CreateMigrationsTable(Migration):
    """Migration pour créer la table de suivi des migrations"""

    def __init__(self):
        super().__init__("000", "Création de la table migrations")

    def up(self, connection: sqlite3.Connection):
        """Crée la table migrations"""
        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS migrations (
                version TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN DEFAULT TRUE
            )
        """
        )

        connection.commit()
        logger.info("Table migrations créée")

    def down(self, connection: sqlite3.Connection):
        """Supprime la table migrations"""
        cursor = connection.cursor()
        cursor.execute("DROP TABLE IF EXISTS migrations")
        connection.commit()


class OptimizeDatabase(Migration):
    """Migration pour optimiser la base de données"""

    def __init__(self):
        super().__init__("003", "Optimisation de la base de données")

    def up(self, connection: sqlite3.Connection):
        """Optimise la base de données"""
        cursor = connection.cursor()

        # Configuration SQLite pour de meilleures performances
        cursor.execute("PRAGMA journal_mode = WAL")
        cursor.execute("PRAGMA synchronous = NORMAL")
        cursor.execute("PRAGMA cache_size = 10000")
        cursor.execute("PRAGMA temp_store = MEMORY")
        cursor.execute("PRAGMA mmap_size = 268435456")  # 256MB

        # Analyse des statistiques pour l'optimiseur
        cursor.execute("ANALYZE")

        connection.commit()
        logger.info("Base de données optimisée")

    def down(self, connection: sqlite3.Connection):
        """Restaure les paramètres par défaut"""
        cursor = connection.cursor()
        cursor.execute("PRAGMA journal_mode = DELETE")
        cursor.execute("PRAGMA synchronous = FULL")
        connection.commit()


class MigrationManager:
    """Gestionnaire des migrations de base de données"""

    def __init__(self, db_connection_instance=None):
        self.db = db_connection_instance or db_connection
        self.migrations = [
            CreateMigrationsTable(),
            CreateSensitiveAreasTable(),
            AddCategoryColumn(),
            OptimizeDatabase(),
        ]

    def get_applied_migrations(self) -> List[str]:
        """Récupère la liste des migrations appliquées"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT version FROM migrations ORDER BY version")
                return [row[0] for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            # Table migrations n'existe pas encore
            return []

    def record_migration(self, migration: Migration, success: bool = True):
        """Enregistre une migration dans la table de suivi"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO migrations (version, description, applied_at, success)
                    VALUES (?, ?, ?, ?)
                """,
                    (migration.version, migration.description, datetime.now(), success),
                )
                conn.commit()
        except Exception as e:
            logger.error(
                f"Erreur lors de l'enregistrement de la migration {migration.version}: {e}"
            )

    def migrate(self, target_version: str = None):
        """Applique les migrations jusqu'à la version cible"""
        applied_migrations = self.get_applied_migrations()

        for migration in self.migrations:
            if migration.version in applied_migrations:
                logger.info(f"Migration {migration.version} déjà appliquée")
                continue

            if target_version and migration.version > target_version:
                break

            try:
                logger.info(
                    f"Application de la migration {migration.version}: {migration.description}"
                )

                with self.db.get_connection() as conn:
                    migration.up(conn)

                self.record_migration(migration, True)
                logger.info(f"Migration {migration.version} appliquée avec succès")

            except Exception as e:
                logger.error(
                    f"Erreur lors de l'application de la migration {migration.version}: {e}"
                )
                self.record_migration(migration, False)
                raise DatabaseException(
                    f"Échec de la migration {migration.version}: {e}"
                )

    def rollback(self, target_version: str):
        """Annule les migrations jusqu'à la version cible"""
        applied_migrations = self.get_applied_migrations()

        # Trier les migrations en ordre inverse
        migrations_to_rollback = []
        for migration in reversed(self.migrations):
            if (
                migration.version in applied_migrations
                and migration.version > target_version
            ):
                migrations_to_rollback.append(migration)

        for migration in migrations_to_rollback:
            try:
                logger.info(f"Annulation de la migration {migration.version}")

                with self.db.get_connection() as conn:
                    migration.down(conn)

                # Supprimer l'enregistrement de la migration
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "DELETE FROM migrations WHERE version = ?", (migration.version,)
                    )
                    conn.commit()

                logger.info(f"Migration {migration.version} annulée avec succès")

            except Exception as e:
                logger.error(
                    f"Erreur lors de l'annulation de la migration {migration.version}: {e}"
                )
                raise DatabaseException(
                    f"Échec de l'annulation de la migration {migration.version}: {e}"
                )

    def get_migration_status(self) -> Dict[str, Any]:
        """Retourne le statut des migrations"""
        applied_migrations = self.get_applied_migrations()

        status = {
            "total_migrations": len(self.migrations),
            "applied_count": len(applied_migrations),
            "pending_count": len(self.migrations) - len(applied_migrations),
            "migrations": [],
        }

        for migration in self.migrations:
            is_applied = migration.version in applied_migrations
            status["migrations"].append(
                {
                    "version": migration.version,
                    "description": migration.description,
                    "applied": is_applied,
                    "created_at": migration.created_at,
                }
            )

        return status

    def reset_database(self):
        """Remet à zéro la base de données (ATTENTION: supprime toutes les données)"""
        logger.warning(
            "Remise à zéro de la base de données - TOUTES LES DONNÉES SERONT PERDUES"
        )

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # Supprimer toutes les tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()

                for table in tables:
                    if not table[0].startswith("sqlite_"):
                        cursor.execute(f"DROP TABLE IF EXISTS {table[0]}")

                conn.commit()

            logger.info("Base de données remise à zéro")

        except Exception as e:
            raise DatabaseException(f"Erreur lors de la remise à zéro: {e}")


# Instance globale du gestionnaire de migrations
migration_manager = MigrationManager()
