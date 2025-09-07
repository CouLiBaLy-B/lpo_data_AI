# src/core/models.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum


class QueryType(str, Enum):
    """Types de requêtes supportées"""

    SELECT = "select"
    COUNT = "count"
    AGGREGATE = "aggregate"
    JOIN = "join"


class ChartType(str, Enum):
    """Types de graphiques supportés"""

    BAR = "bar"
    PIE = "pie"
    LINE = "line"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    MAP = "map"
    HEATMAP = "heatmap"


class QueryRequest(BaseModel):
    """Modèle pour les requêtes utilisateur"""

    question: str = Field(..., min_length=5, max_length=500)
    context: Optional[str] = Field(None, max_length=1000)
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    @validator("question")
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


class SQLQuery(BaseModel):
    """Modèle pour les requêtes SQL générées"""

    query: str = Field(..., min_length=10)
    query_type: QueryType
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: Optional[str] = None
    estimated_rows: Optional[int] = None

    @validator("query")
    def validate_sql(cls, v):
        # Validation basique SQL
        forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE"]
        upper_query = v.upper()
        for keyword in forbidden_keywords:
            if keyword in upper_query:
                raise ValueError(f"Forbidden SQL keyword: {keyword}")
        return v


class VisualizationConfig(BaseModel):
    """Configuration pour les visualisations"""

    chart_type: ChartType
    title: str
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    color_column: Optional[str] = None
    size_column: Optional[str] = None
    facet_column: Optional[str] = None
    theme: str = Field(default="plotly")


class QueryResult(BaseModel):
    """Résultat d'une requête"""

    data: List[Dict[str, Any]]
    columns: List[str]
    row_count: int
    execution_time: float
    query_hash: str


class VisualizationResult(BaseModel):
    """Résultat d'une visualisation"""

    chart_config: VisualizationConfig
    python_code: str
    success: bool
    error_message: Optional[str] = None


class UserSession(BaseModel):
    """Session utilisateur"""

    user_id: str
    session_id: str
    created_at: datetime
    last_activity: datetime
    is_authenticated: bool = False
    query_history: List[str] = Field(default_factory=list)


class APIResponse(BaseModel):
    """Réponse API standardisée"""

    success: bool
    message: str
    data: Optional[Any] = None
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# src/core/exceptions.py
class LPOAIException(Exception):
    """Exception de base pour l'application"""

    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


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
