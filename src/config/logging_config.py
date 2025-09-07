import logging
import logging.handlers
from pathlib import Path
from .settings import settings


def setup_logging():
    """Configure le système de logging"""

    # Créer le dossier logs si nécessaire
    if settings.logging.file_path:
        log_dir = Path(settings.logging.file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)

    # Configuration du logger principal
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, settings.logging.level.upper()))

    # Handler pour la console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Handler pour les fichiers avec rotation
    if settings.logging.file_path:
        file_handler = logging.handlers.RotatingFileHandler(
            settings.logging.file_path, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        file_handler.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(settings.logging.format)
    console_handler.setFormatter(formatter)
    if settings.logging.file_path:
        file_handler.setFormatter(formatter)

    # Ajouter les handlers
    logger.addHandler(console_handler)
    if settings.logging.file_path:
        logger.addHandler(file_handler)

    return logger
