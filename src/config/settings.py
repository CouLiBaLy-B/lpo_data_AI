from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()

# Racine du projet (pour résoudre le chemin SQLite par défaut de façon absolue)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SQLITE = PROJECT_ROOT / "data" / "sensitive_areas.db"


class DatabaseSettings(BaseSettings):
    """
    Connexion base de données pilotée par DATABASE_URL.

    - dev  : sqlite:///data/sensitive_areas.db (défaut)
    - prod : postgresql://user:pwd@host:5432/dbname

    L'application travaille en LECTURE SEULE : seules les requêtes SELECT
    sont autorisées (la donnée est alimentée par un système externe).
    """

    url: str = Field(default=f"sqlite:///{_DEFAULT_SQLITE}", alias="DATABASE_URL")
    read_only: bool = Field(default=True, alias="DB_READ_ONLY")
    statement_timeout_ms: int = Field(default=15000, alias="DB_STATEMENT_TIMEOUT_MS")
    max_rows: int = Field(default=5000, alias="DB_MAX_ROWS")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


class AISettings(BaseSettings):
    """
    Configuration IA indépendante du provider.

    `provider` + `model` sont combinés par le model_factory via
    init_chat_model (ex: provider='anthropic', model='claude-haiku-4-5-...').
    Changer de provider = changer ces deux variables, rien d'autre.
    """

    provider: str = Field(default="anthropic", alias="AI_PROVIDER")
    model: str = Field(default="claude-haiku-4-5-20251001", alias="AI_MODEL")
    temperature: float = Field(default=0.0, alias="AI_TEMPERATURE")
    max_tokens: int = Field(default=2048, alias="AI_MAX_TOKENS")

    # Clés par provider (seule celle du provider actif est requise)
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


class SecuritySettings(BaseSettings):
    username: str = Field(default="admin", alias="USERNAME")
    password: str = Field(default="admin", alias="PASSWORD")
    secret_key: str = Field(default="lpo_ai_secret_default", alias="SECRET_KEY")
    session_timeout: int = Field(default=3600)
    max_login_attempts: int = Field(default=3)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


class AppSettings(BaseSettings):
    app_name: str = Field(default="LPO AI - Assistant SQL")
    debug: bool = Field(default=False, alias="DEBUG")

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    ai: AISettings = Field(default_factory=AISettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = AppSettings()
