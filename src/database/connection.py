# src/database/connection.py
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path

from ..config.settings import settings
from ..core.exceptions import DatabaseException

logger = logging.getLogger(__name__)

class DatabaseConnection:
    """Gestionnaire de connexion à la base de données avec pool de connexions"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(settings.database.db_path)
        self.pool_size = settings.database.connection_pool_size
        self.connections = []
        self.lock = threading.Lock()
        self.connection_count = 0
        
        # Créer le répertoire si nécessaire
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialiser le pool
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialise le pool de connexions"""
        try:
            for _ in range(self.pool_size):
                conn = self._create_connection()
                self.connections.append(conn)
            logger.info(f"Pool de connexions initialisé avec {self.pool_size} connexions")
        except Exception as e:
            raise DatabaseException(f"Erreur lors de l'initialisation du pool: {e}")
    
    def _create_connection(self) -> sqlite3.Connection:
        """Crée une nouvelle connexion à la base de données"""
        try:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            
            # Configuration de la connexion
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute("PRAGMA cache_size = 10000")
            
            return conn
        except Exception as e:
            raise DatabaseException(f"Erreur lors de la création de la connexion: {e}")
    
    @contextmanager
    def get_connection(self):
        """Context manager pour obtenir une connexion du pool"""
        conn = None
        try:
            with self.lock:
                if self.connections:
                    conn = self.connections.pop()
                else:
                    conn = self._create_connection()
                    self.connection_count += 1
            
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise DatabaseException(f"Erreur lors de l'utilisation de la connexion: {e}")
        finally:
            if conn:
                try:
                    conn.commit()
                    with self.lock:
                        if len(self.connections) < self.pool_size:
                            self.connections.append(conn)
                        else:
                            conn.close()
                            self.connection_count -= 1
                except Exception as e:
                    logger.error(f"Erreur lors de la fermeture de la connexion: {e}")
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Exécute une requête SELECT et retourne les résultats"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                start_time = time.time()
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                results = [dict(row) for row in cursor.fetchall()]
                execution_time = time.time() - start_time
                
                logger.info(f"Requête exécutée en {execution_time:.3f}s: {query[:100]}...")
                return results
                
        except Exception as e:
            raise DatabaseException(f"Erreur lors de l'exécution de la requête: {e}")
    
    def execute_script(self, script: str, params: tuple = None) -> int:
        """Exécute un script SQL (INSERT, UPDATE, DELETE) et retourne le nombre de lignes affectées"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if params:
                    cursor.execute(script, params)
                else:
                    cursor.execute(script)
                
                return cursor.rowcount
                
        except Exception as e:
            raise DatabaseException(f"Erreur lors de l'exécution du script: {e}")
    
    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """Exécute une requête avec plusieurs paramètres"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(query, params_list)
                return cursor.rowcount
                
        except Exception as e:
            raise DatabaseException(f"Erreur lors de l'exécution multiple: {e}")
    
    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """Obtient les informations sur une table"""
        try:
            query = f"PRAGMA table_info({table_name})"
            return self.execute_query(query)
        except Exception as e:
            raise DatabaseException(f"Erreur lors de la récupération des infos table: {e}")
    
    def get_table_list(self) -> List[str]:
        """Obtient la liste des tables"""
        try:
            query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            results = self.execute_query(query)
            return [row['name'] for row in results]
        except Exception as e:
            raise DatabaseException(f"Erreur lors de la récupération de la liste des tables: {e}")
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Obtient les informations complètes du schéma"""
        try:
            schema_info = {}
            tables = self.get_table_list()
            
            for table in tables:
                table_info = self.get_table_info(table)
                schema_info[table] = {
                    'columns': table_info,
                    'indexes': self._get_table_indexes(table),
                    'row_count': self._get_table_row_count(table)
                }
            
            return schema_info
            
        except Exception as e:
            raise DatabaseException(f"Erreur lors de la récupération du schéma: {e}")
    
    def _get_table_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        """Obtient les index d'une table"""
        try:
            query = f"PRAGMA index_list({table_name})"
            return self.execute_query(query)
        except Exception as e:
            logger.warning(f"Erreur lors de la récupération des index pour {table_name}: {e}")
            return []
    
    def _get_table_row_count(self, table_name: str) -> int:
        """Obtient le nombre de lignes d'une table"""
        try:
            query = f"SELECT COUNT(*) as count FROM {table_name}"
            result = self.execute_query(query)
            return result[0]['count'] if result else 0
        except Exception as e:
            logger.warning(f"Erreur lors du comptage des lignes pour {table_name}: {e}")
            return 0
    
    def close_all_connections(self):
        """Ferme toutes les connexions du pool"""
        with self.lock:
            for conn in self.connections:
                try:
                    conn.close()
                except Exception as e:
                    logger.error(f"Erreur lors de la fermeture de connexion: {e}")
            self.connections.clear()
            self.connection_count = 0
        
        logger.info("Toutes les connexions fermées")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all_connections()

# Instance globale de connexion
db_connection = DatabaseConnection()