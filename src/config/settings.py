# src/config/settings.py
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class DatabaseSettings(BaseSettings):
    """Configuration de la base de données"""

    db_path: Path = Field(default="data/sensitive_areas.db")
    connection_pool_size: int = Field(default=5)
    echo: bool = Field(default=False)

    @field_validator("db_path")
    @classmethod
    def validate_db_path(cls, v):
        if not v.parent.exists():
            v.parent.mkdir(parents=True, exist_ok=True)
        return v


class AISettings(BaseSettings):
    """Configuration IA"""

    huggingface_api_key: str = Field(...)
    model_name: str = Field(default="mistralai/Mistral-7B-v0.13")
    sql_model: str = Field(default="mistralai/Mistral-7B-v0.13")
    chart_model: str = Field(default="mistralai/Mistral-7B-v0.13")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    timeout: int = Field(default=30)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",  # Ignore les variables extra du .env
    }


class SecuritySettings(BaseSettings):
    """Configuration de sécurité"""

    username: str = Field(...)
    password: str = Field(...)
    secret_key: str = Field(...)
    session_timeout: int = Field(default=3600)
    max_login_attempts: int = Field(default=3)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class CacheSettings(BaseSettings):
    """Configuration du cache"""

    enabled: bool = Field(default=True)
    ttl: int = Field(default=3600)
    max_size: int = Field(default=100)

    model_config = {"extra": "ignore"}


class LoggingSettings(BaseSettings):
    """Configuration des logs"""

    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_path: Optional[Path] = Field(default="logs/app.log")

    model_config = {"extra": "ignore"}


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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Instance globale
settings = AppSettings()
