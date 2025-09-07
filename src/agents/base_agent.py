# src/agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import logging
import time
from datetime import datetime

from src.core.models import QueryRequest, APIResponse
from src.core.exceptions import AIModelException, ValidationException
from src.config.settings import settings

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Agent de base avec fonctionnalités communes"""

    def __init__(self, name: str):
        self.name = name
        self.created_at = datetime.now()
        self.request_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
        self.last_request_time = None

    @abstractmethod
    def process(self, request: Any) -> Any:
        """Traite une requête - à implémenter par les agents spécialisés"""
        pass

    def execute(self, request: Any) -> APIResponse:
        """Exécute une requête avec gestion d'erreurs et métriques"""
        start_time = time.time()
        self.request_count += 1
        self.last_request_time = datetime.now()

        try:
            # Validation de base
            self._validate_request(request)

            # Traitement
            result = self.process(request)

            # Métriques
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time

            logger.info(
                f"Agent {self.name} - Requête traitée en {processing_time:.3f}s"
            )

            return APIResponse(
                success=True,
                message=f"Requête traitée avec succès par {self.name}",
                data=result,
            )

        except Exception as e:
            self.error_count += 1
            processing_time = time.time() - start_time

            logger.error(
                f"Agent {self.name} - Erreur: {str(e)} (temps: {processing_time:.3f}s)"
            )
            return APIResponse(
                success=False,
                message=f"Erreur lors du traitement par {self.name}: {str(e)}",
                error_code=getattr(e, "error_code", "PROCESSING_ERROR"),
            )

    def _validate_request(self, request: Any):
        """Validation de base des requêtes"""
        if request is None:
            raise ValidationException("Requête vide", "EMPTY_REQUEST")

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de l'agent"""
        avg_processing_time = (
            self.total_processing_time / self.request_count
            if self.request_count > 0
            else 0
        )

        return {
            "name": self.name,
            "created_at": self.created_at,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": (
                self.error_count / self.request_count if self.request_count > 0 else 0
            ),
            "avg_processing_time": avg_processing_time,
            "last_request_time": self.last_request_time,
            "uptime": (datetime.now() - self.created_at).total_seconds(),
        }

    def reset_stats(self):
        """Remet à zéro les statistiques"""
        self.request_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
        self.last_request_time = None


class AIAgent(BaseAgent):
    """Agent IA de base avec fonctionnalités communes aux agents IA"""

    def __init__(self, name: str, model_name: str = None):
        super().__init__(name)
        self.model_name = model_name or settings.ai.model_name
        self.temperature = settings.ai.temperature
        self.max_tokens = settings.ai.max_tokens
        self.timeout = settings.ai.timeout
        self.llm = None

    def _initialize_llm(self):
        """Initialise le modèle de langage - à implémenter par les sous-classes"""
        pass

    def _create_prompt(self, request: Any) -> str:
        """Crée le prompt pour le modèle - à implémenter par les sous-classes"""
        pass

    def _post_process_response(self, response: str) -> Any:
        """Post-traite la réponse du modèle - à implémenter par les sous-classes"""
        return response

    def _validate_ai_response(self, response: str) -> bool:
        """Valide la réponse du modèle IA"""
        if not response or not response.strip():
            return False
        return True


class CachedAgent(BaseAgent):
    """Agent avec support de cache"""

    def __init__(self, name: str, cache_service=None):
        super().__init__(name)
        self.cache_service = cache_service
        self.cache_hits = 0
        self.cache_misses = 0

    def execute(self, request: Any) -> APIResponse:
        """Exécute avec vérification du cache"""
        if self.cache_service:
            cache_key = self._generate_cache_key(request)
            cached_result = self.cache_service.get(cache_key)

            if cached_result:
                self.cache_hits += 1
                logger.info(f"Agent {self.name} - Cache hit pour la clé: {cache_key}")
                return APIResponse(
                    success=True,
                    message=f"Résultat du cache pour {self.name}",
                    data=cached_result,
                )
            else:
                self.cache_misses += 1

        # Exécution normale
        response = super().execute(request)

        # Mise en cache du résultat si succès
        if response.success and self.cache_service:
            cache_key = self._generate_cache_key(request)
            self.cache_service.set(cache_key, response.data)

        return response

    def _generate_cache_key(self, request: Any) -> str:
        """Génère une clé de cache pour la requête"""
        import hashlib

        request_str = str(request)
        return f"{self.name}_{hashlib.md5(request_str.encode()).hexdigest()}"

    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de cache"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "total_cached_requests": total_requests,
        }


class AgentManager:
    """Gestionnaire d'agents"""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.created_at = datetime.now()

    def register_agent(self, agent: BaseAgent):
        """Enregistre un agent"""
        self.agents[agent.name] = agent
        logger.info(f"Agent {agent.name} enregistré")

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Récupère un agent par nom"""
        return self.agents.get(name)

    def list_agents(self) -> List[str]:
        """Liste tous les agents enregistrés"""
        return list(self.agents.keys())

    def get_all_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques de tous les agents"""
        stats = {
            "manager_created_at": self.created_at,
            "total_agents": len(self.agents),
            "agents": {},
        }

        for name, agent in self.agents.items():
            agent_stats = agent.get_stats()
            if isinstance(agent, CachedAgent):
                agent_stats.update(agent.get_cache_stats())
            stats["agents"][name] = agent_stats

        return stats

    def reset_all_stats(self):
        """Remet à zéro les statistiques de tous les agents"""
        for agent in self.agents.values():
            agent.reset_stats()
        logger.info("Statistiques de tous les agents remises à zéro")


# Instance globale du gestionnaire d'agents
agent_manager = AgentManager()
