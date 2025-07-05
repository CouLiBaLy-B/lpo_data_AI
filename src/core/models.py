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
    
    @validator('question')
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
    
    @validator('query')
    def validate_sql(cls, v):
        # Validation basique SQL
        forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
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

# src/core/security.py
import hashlib
import hmac
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from functools import wraps
import streamlit as st

from .exceptions import AuthenticationException
from ..config.settings import settings

class SecurityManager:
    """Gestionnaire de sécurité avancé"""
    
    def __init__(self):
        self.secret_key = settings.security.secret_key
        self.session_timeout = settings.security.session_timeout
        self.max_attempts = settings.security.max_login_attempts
        self.failed_attempts: Dict[str, int] = {}
        self.blocked_users: Dict[str, datetime] = {}
    
    def hash_password(self, password: str, salt: str = None) -> tuple:
        """Hash sécurisé du mot de passe avec salt"""
        if salt is None:
            salt = secrets.token_hex(32)
        
        # Utilisation de PBKDF2 pour le hashing
        key = hashlib.pbkdf2_hmac('sha256', 
                                 password.encode('utf-8'), 
                                 salt.encode('utf-8'), 
                                 100000)  # 100k iterations
        return key.hex(), salt
    
    def verify_password(self, password: str, hashed_password: str, salt: str) -> bool:
        """Vérifie le mot de passe"""
        key, _ = self.hash_password(password, salt)
        return hmac.compare_digest(key, hashed_password)
    
    def create_jwt_token(self, user_id: str, additional_claims: Dict[str, Any] = None) -> str:
        """Crée un token JWT"""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(seconds=self.session_timeout),
            'iat': datetime.utcnow(),
            'jti': secrets.token_hex(16)  # Token ID unique
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Vérifie et décode un token JWT"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationException("Token expiré", "TOKEN_EXPIRED")
        except jwt.InvalidTokenError:
            raise AuthenticationException("Token invalide", "INVALID_TOKEN")
    
    def is_user_blocked(self, user_id: str) -> bool:
        """Vérifie si l'utilisateur est bloqué"""
        if user_id in self.blocked_users:
            block_time = self.blocked_users[user_id]
            if datetime.now() - block_time > timedelta(minutes=15):
                # Débloquer après 15 minutes
                del self.blocked_users[user_id]
                self.failed_attempts[user_id] = 0
                return False
            return True
        return False
    
    def record_failed_attempt(self, user_id: str):
        """Enregistre une tentative de connexion échouée"""
        self.failed_attempts[user_id] = self.failed_attempts.get(user_id, 0) + 1
        
        if self.failed_attempts[user_id] >= self.max_attempts:
            self.blocked_users[user_id] = datetime.now()
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authentifie un utilisateur et retourne un token JWT"""
        if self.is_user_blocked(username):
            raise AuthenticationException("Compte temporairement bloqué", "USER_BLOCKED")
        
        # Vérification des credentials (à adapter selon votre système)
        if username == settings.security.username:
            # Dans un vrai système, récupérer le hash et salt depuis la DB
            stored_hash, salt = self.hash_password(settings.security.password)
            
            if self.verify_password(password, stored_hash, salt):
                # Reset failed attempts on successful login
                self.failed_attempts[username] = 0
                return self.create_jwt_token(username)
        
        self.record_failed_attempt(username)
        raise AuthenticationException("Identifiants invalides", "INVALID_CREDENTIALS")

def require_auth(func):
    """Décorateur pour les fonctions nécessitant une authentification"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'auth_token' not in st.session_state:
            raise AuthenticationException("Authentification requise", "AUTH_REQUIRED")
        
        security_manager = SecurityManager()
        try:
            payload = security_manager.verify_jwt_token(st.session_state.auth_token)
            st.session_state.user_id = payload['user_id']
            return func(*args, **kwargs)
        except AuthenticationException:
            del st.session_state.auth_token
            raise
    
    return wrapper