# src/config/settings.py
from pydantic import BaseSettings, Field, validator
from typing import Optional, List
import os
from pathlib import Path

class DatabaseSettings(BaseSettings):
    """Configuration de la base de données"""
    db_path: Path = Field(default="data/sensitive_areas.db")
    connection_pool_size: int = Field(default=5)
    echo: bool = Field(default=False)
    
    @validator('db_path')
    def validate_db_path(cls, v):
        if not v.parent.exists():
            v.parent.mkdir(parents=True, exist_ok=True)
        return v

class AISettings(BaseSettings):
    """Configuration IA"""
    huggingface_api_key: str = Field(..., env="HUGGINGFACE_API_KEY")
    model_name: str = Field(default="microsoft/DialoGPT-medium")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    timeout: int = Field(default=30)
    
    # Modèles spécialisés
    sql_model: str = Field(default="defog/sqlcoder-34b-alpha")
    chart_model: str = Field(default="microsoft/DialoGPT-medium")
    
    class Config:
        env_prefix = "AI_"

class SecuritySettings(BaseSettings):
    """Configuration de sécurité"""
    username: str = Field(..., env="USERNAME")
    password: str = Field(..., env="PASSWORD")
    secret_key: str = Field(..., env="SECRET_KEY")
    session_timeout: int = Field(default=3600)  # 1 heure
    max_login_attempts: int = Field(default=3)
    
    class Config:
        env_prefix = "SECURITY_"

class CacheSettings(BaseSettings):
    """Configuration du cache"""
    enabled: bool = Field(default=True)
    ttl: int = Field(default=3600)  # 1 heure
    max_size: int = Field(default=100)
    
    class Config:
        env_prefix = "CACHE_"

class LoggingSettings(BaseSettings):
    """Configuration des logs"""
    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_path: Optional[Path] = Field(default="logs/app.log")
    
    class Config:
        env_prefix = "LOG_"

class AppSettings(BaseSettings):
    """Configuration principale de l'application"""
    app_name: str = Field(default="LPO AI - SQL Query Visualizer")
    debug: bool = Field(default=False)
    environment: str = Field(default="production")
    
    # Sous-configurations
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    ai: AISettings = Field(default_factory=AISettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Instance globale
settings = AppSettings()

# src/config/logging_config.py
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
            settings.logging.file_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
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