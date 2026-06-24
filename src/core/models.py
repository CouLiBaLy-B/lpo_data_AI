from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ChartType(str, Enum):
    BAR = "bar"
    PIE = "pie"
    LINE = "line"
    SCATTER = "scatter"
    TABLE = "table"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    user_id: Optional[str] = None

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        return v.strip()


class SQLResult(BaseModel):
    """Résultat structuré retourné par le SQL agent (Claude tool use)."""
    sql: str
    explanation: str
    chart_type: ChartType = ChartType.BAR
    success: bool = True
    error: Optional[str] = None

    @field_validator("sql")
    @classmethod
    def validate_select_only(cls, v: str) -> str:
        forbidden = {"DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE"}
        upper = v.upper()
        for kw in forbidden:
            if kw in upper:
                raise ValueError(f"Mot-clé SQL interdit: {kw}")
        return v.strip()


class QueryResult(BaseModel):
    data: List[Dict[str, Any]]
    columns: List[str]
    row_count: int
    execution_time: float
    query_hash: str


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    sql_result: Optional[SQLResult] = None
    query_result: Optional[QueryResult] = None


class AppStats(BaseModel):
    total_zones: int = 0
    total_regions: int = 0
    total_structures: int = 0
    category_distribution: Dict[str, int] = Field(default_factory=dict)
