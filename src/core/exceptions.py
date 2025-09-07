# src/core/exceptions.py
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LPOAIException(Exception):
    """Exception de base pour l'application"""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

        # Log l'exception
        logger.error(
            f"Exception: {self.__class__.__name__} - {message}",
            extra={"error_code": error_code, "details": details},
        )


class DatabaseException(LPOAIException):
    """Exception pour les erreurs de base de données"""

    pass


class AIModelException(LPOAIException):
    """Exception pour les erreurs de modèle IA"""

    pass


class ValidationException(LPOAIException):
    """Exception pour les erreurs de validation"""

    pass


class AuthenticationException(LPOAIException):
    """Exception pour les erreurs d'authentification"""

    pass


class CacheException(LPOAIException):
    """Exception pour les erreurs de cache"""

    pass


class QueryExecutionException(LPOAIException):
    """Exception pour les erreurs d'exécution de requête"""

    pass


class VisualizationException(LPOAIException):
    """Exception pour les erreurs de visualisation"""

    pass


class DataExtractionException(LPOAIException):
    """Exception pour les erreurs d'extraction de données"""

    pass


class APIException(LPOAIException):
    """Exception pour les erreurs d'API"""

    pass


# Exceptions spécifiques avec codes d'erreur
class InvalidQueryException(QueryExecutionException):
    """Exception pour les requêtes SQL invalides"""

    def __init__(self, message: str, sql_query: str = None):
        super().__init__(message, "INVALID_QUERY", {"sql_query": sql_query})


class TimeoutException(LPOAIException):
    """Exception pour les timeouts"""

    def __init__(self, message: str, timeout_value: int = None):
        super().__init__(message, "TIMEOUT", {"timeout": timeout_value})


class RateLimitException(LPOAIException):
    """Exception pour les limitations de débit"""

    def __init__(self, message: str, retry_after: int = None):
        super().__init__(message, "RATE_LIMIT", {"retry_after": retry_after})


class ConfigurationException(LPOAIException):
    """Exception pour les erreurs de configuration"""

    def __init__(self, message: str, config_key: str = None):
        super().__init__(message, "CONFIG_ERROR", {"config_key": config_key})


class SecurityException(LPOAIException):
    """Exception pour les erreurs de sécurité"""

    def __init__(self, message: str, security_level: str = "HIGH"):
        super().__init__(message, "SECURITY_VIOLATION", {"level": security_level})


# Gestionnaire d'exceptions global
class ExceptionHandler:
    """Gestionnaire centralisé des exceptions"""

    @staticmethod
    def handle_exception(exception: Exception, context: str = None) -> Dict[str, Any]:
        """Gère une exception et retourne un dictionnaire d'erreur standardisé"""

        error_response = {
            "success": False,
            "error_type": type(exception).__name__,
            "message": str(exception),
            "context": context,
        }

        if isinstance(exception, LPOAIException):
            error_response.update(
                {"error_code": exception.error_code, "details": exception.details}
            )

        # Log selon le type d'exception
        if isinstance(exception, SecurityException):
            logger.critical(
                f"Security violation: {exception.message}", extra={"context": context}
            )
        elif isinstance(exception, DatabaseException):
            logger.error(
                f"Database error: {exception.message}", extra={"context": context}
            )
        elif isinstance(exception, AIModelException):
            logger.warning(
                f"AI model error: {exception.message}", extra={"context": context}
            )
        else:
            logger.error(
                f"Unhandled exception: {exception}", extra={"context": context}
            )

        return error_response

    @staticmethod
    def is_retryable(exception: Exception) -> bool:
        """Détermine si une exception peut être retentée"""
        retryable_exceptions = [AIModelException, APIException, TimeoutException]

        return any(isinstance(exception, exc_type) for exc_type in retryable_exceptions)


# Décorateur pour la gestion d'exceptions
def handle_exceptions(context: str = None, reraise: bool = False):
    """Décorateur pour gérer automatiquement les exceptions"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_response = ExceptionHandler.handle_exception(e, context)
                if reraise:
                    raise
                return error_response

        return wrapper

    return decorator
