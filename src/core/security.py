# src/core/security.py
import hashlib
import hmac
import secrets
import jwt  # PyJWT
from datetime import datetime, timedelta, timezone
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
        key = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000
        )  # 100k iterations
        return key.hex(), salt

    def verify_password(self, password: str, hashed_password: str, salt: str) -> bool:
        """Vérifie le mot de passe"""
        key, _ = self.hash_password(password, salt)
        return hmac.compare_digest(key, hashed_password)

    def create_jwt_token(
        self, user_id: str, additional_claims: Dict[str, Any] = None
    ) -> str:
        """Crée un token JWT"""
        payload = {
            "user_id": user_id,
            "exp": datetime.now(timezone.utc) + timedelta(seconds=self.session_timeout),
            "iat": datetime.now(timezone.utc),
            "jti": secrets.token_hex(16),  # Token ID unique
        }

        if additional_claims:
            payload.update(additional_claims)

        try:
            return jwt.encode(payload, self.secret_key, algorithm="HS256")
        except Exception as e:
            raise AuthenticationException(
                f"Erreur création token: {str(e)}", "TOKEN_CREATION_ERROR"
            )

    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Vérifie et décode un token JWT"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationException("Token expiré", "TOKEN_EXPIRED")
        except jwt.InvalidTokenError:
            raise AuthenticationException("Token invalide", "INVALID_TOKEN")
        except Exception as e:
            raise AuthenticationException(
                f"Erreur vérification token: {str(e)}", "TOKEN_VERIFICATION_ERROR"
            )

    def is_user_blocked(self, user_id: str) -> bool:
        """Vérifie si l'utilisateur est bloqué"""
        if user_id in self.blocked_users:
            block_time = self.blocked_users[user_id]
            if datetime.now(timezone.utc) - block_time > timedelta(minutes=15):
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
            self.blocked_users[user_id] = datetime.now(timezone.utc)

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authentifie un utilisateur et retourne un token JWT"""
        if self.is_user_blocked(username):
            raise AuthenticationException(
                "Compte temporairement bloqué", "USER_BLOCKED"
            )

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
        if "auth_token" not in st.session_state:
            raise AuthenticationException("Authentification requise", "AUTH_REQUIRED")

        security_manager = SecurityManager()
        try:
            payload = security_manager.verify_jwt_token(st.session_state.auth_token)
            st.session_state.user_id = payload["user_id"]
            return func(*args, **kwargs)
        except AuthenticationException:
            if "auth_token" in st.session_state:
                del st.session_state.auth_token
            raise

    return wrapper


# Fonction utilitaire pour vérifier l'installation de PyJWT
def check_jwt_installation():
    """Vérifie que PyJWT est correctement installé"""
    try:
        import jwt

        # Test d'encodage/décodage simple
        test_payload = {"test": "data"}
        test_secret = "test_secret"
        token = jwt.encode(test_payload, test_secret, algorithm="HS256")
        decoded = jwt.decode(token, test_secret, algorithms=["HS256"])
        return True
    except Exception as e:
        print(f"Erreur PyJWT: {e}")
        return False
