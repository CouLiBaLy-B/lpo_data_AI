# src/services/cache_service.py
import time
import json
import hashlib
import threading
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
import logging
from collections import OrderedDict

from ..config.settings import settings
from ..core.exceptions import CacheException

logger = logging.getLogger(__name__)


class CacheEntry:
    """Entrée de cache avec métadonnées"""

    def __init__(self, key: str, value: Any, ttl: int = None):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl or settings.cache.ttl
        self.access_count = 0
        self.last_accessed = self.created_at

    def is_expired(self) -> bool:
        """Vérifie si l'entrée a expiré"""
        if self.ttl <= 0:
            return False
        return time.time() - self.created_at > self.ttl

    def access(self) -> Any:
        """Accède à la valeur et met à jour les métadonnées"""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.value

    def size_estimate(self) -> int:
        """Estime la taille de l'entrée en bytes"""
        try:
            return len(json.dumps(self.value, default=str))
        except:
            return len(str(self.value))


class InMemoryCache:
    """Cache en mémoire avec LRU et TTL"""

    def __init__(self, max_size: int = None, default_ttl: int = None):
        self.max_size = max_size or settings.cache.max_size
        self.default_ttl = default_ttl or settings.cache.ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "expired_removals": 0}

    def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache"""
        with self.lock:
            if key not in self.cache:
                self.stats["misses"] += 1
                return None

            entry = self.cache[key]

            # Vérifier l'expiration
            if entry.is_expired():
                del self.cache[key]
                self.stats["expired_removals"] += 1
                self.stats["misses"] += 1
                return None

            # Déplacer vers la fin (LRU)
            self.cache.move_to_end(key)
            self.stats["hits"] += 1

            return entry.access()

    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Stocke une valeur dans le cache"""
        try:
            with self.lock:
                # Créer l'entrée
                entry = CacheEntry(key, value, ttl or self.default_ttl)

                # Supprimer l'ancienne entrée si elle existe
                if key in self.cache:
                    del self.cache[key]

                # Ajouter la nouvelle entrée
                self.cache[key] = entry

                # Vérifier la taille maximale
                self._evict_if_needed()

                return True

        except Exception as e:
            logger.error(f"Erreur lors de la mise en cache: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Supprime une entrée du cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def clear(self):
        """Vide le cache"""
        with self.lock:
            self.cache.clear()
            logger.info("Cache vidé")

    def _evict_if_needed(self):
        """Évince les entrées si nécessaire (LRU)"""
        while len(self.cache) > self.max_size:
            # Supprimer l'entrée la moins récemment utilisée
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.stats["evictions"] += 1

    def cleanup_expired(self):
        """Nettoie les entrées expirées"""
        with self.lock:
            expired_keys = []
            for key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                del self.cache[key]
                self.stats["expired_removals"] += 1

            if expired_keys:
                logger.info(
                    f"Nettoyage: {len(expired_keys)} entrées expirées supprimées"
                )

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache"""
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_rate": hit_rate,
                "total_requests": total_requests,
                **self.stats,
            }

    def get_cache_info(self) -> List[Dict[str, Any]]:
        """Retourne des informations détaillées sur le cache"""
        with self.lock:
            info = []
            for key, entry in self.cache.items():
                info.append(
                    {
                        "key": key,
                        "size_estimate": entry.size_estimate(),
                        "created_at": datetime.fromtimestamp(entry.created_at),
                        "last_accessed": datetime.fromtimestamp(entry.last_accessed),
                        "access_count": entry.access_count,
                        "ttl": entry.ttl,
                        "expires_at": (
                            datetime.fromtimestamp(entry.created_at + entry.ttl)
                            if entry.ttl > 0
                            else None
                        ),
                        "is_expired": entry.is_expired(),
                    }
                )
            return info


class CacheService:
    """Service de cache principal avec différentes stratégies"""

    def __init__(self):
        self.enabled = settings.cache.enabled
        self.cache = InMemoryCache() if self.enabled else None
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()

    def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache"""
        if not self.enabled or not self.cache:
            return None

        try:
            # Nettoyage périodique
            self._periodic_cleanup()

            return self.cache.get(self._normalize_key(key))

        except Exception as e:
            logger.error(f"Erreur récupération cache: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Stocke une valeur dans le cache"""
        if not self.enabled or not self.cache:
            return False

        try:
            return self.cache.set(self._normalize_key(key), value, ttl)

        except Exception as e:
            logger.error(f"Erreur stockage cache: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Supprime une entrée du cache"""
        if not self.enabled or not self.cache:
            return False

        try:
            return self.cache.delete(self._normalize_key(key))

        except Exception as e:
            logger.error(f"Erreur suppression cache: {e}")
            return False

    def clear_all(self):
        """Vide tout le cache"""
        if self.enabled and self.cache:
            self.cache.clear()

    def clear_pattern(self, pattern: str):
        """Supprime les entrées correspondant à un motif"""
        if not self.enabled or not self.cache:
            return

        try:
            import re

            regex = re.compile(pattern)

            with self.cache.lock:
                keys_to_delete = []
                for key in self.cache.cache.keys():
                    if regex.search(key):
                        keys_to_delete.append(key)

                for key in keys_to_delete:
                    del self.cache.cache[key]

                logger.info(
                    f"Suppression par motif '{pattern}': {len(keys_to_delete)} entrées"
                )

        except Exception as e:
            logger.error(f"Erreur suppression par motif: {e}")

    def _normalize_key(self, key: str) -> str:
        """Normalise une clé de cache"""
        # Créer un hash pour les clés trop longues
        if len(key) > 200:
            return hashlib.md5(key.encode()).hexdigest()

        # Nettoyer la clé
        return key.replace(" ", "_").replace("\n", "").replace("\r", "")

    def _periodic_cleanup(self):
        """Nettoyage périodique des entrées expirées"""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            self.cache.cleanup_expired()
            self.last_cleanup = current_time

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache"""
        if not self.enabled or not self.cache:
            return {"enabled": False}

        stats = self.cache.get_stats()
        stats["enabled"] = True
        stats["cleanup_interval"] = self.cleanup_interval
        stats["last_cleanup"] = datetime.fromtimestamp(self.last_cleanup)

        return stats

    def get_detailed_info(self) -> Dict[str, Any]:
        """Retourne des informations détaillées sur le cache"""
        if not self.enabled or not self.cache:
            return {"enabled": False}

        return {
            "enabled": True,
            "stats": self.get_stats(),
            "entries": self.cache.get_cache_info(),
        }

    def warm_up(self, queries: List[str]):
        """Préchauffe le cache avec des requêtes communes"""
        if not self.enabled:
            return

        logger.info(f"Préchauffage du cache avec {len(queries)} requêtes")

        # Cette méthode serait appelée avec des requêtes prédéfinies
        # pour améliorer les performances au démarrage
        for query in queries:
            try:
                # Ici, on exécuterait la requête et stockerait le résultat
                # Cette implémentation dépend du service de données
                pass
            except Exception as e:
                logger.warning(f"Erreur préchauffage pour '{query}': {e}")

    def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé du service de cache"""
        if not self.enabled:
            return {"status": "disabled"}

        try:
            # Test simple
            test_key = f"health_check_{int(time.time())}"
            test_value = "test"

            # Test d'écriture
            write_success = self.set(test_key, test_value, ttl=60)

            # Test de lecture
            read_value = self.get(test_key)
            read_success = read_value == test_value

            # Nettoyage
            self.delete(test_key)

            return {
                "status": "healthy" if write_success and read_success else "unhealthy",
                "write_success": write_success,
                "read_success": read_success,
                "stats": self.get_stats(),
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}


# Instance globale du service de cache
cache_service = CacheService()
