#!/usr/bin/env python3
"""
Script de diagnostic pour identifier le problème avec les settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv


def check_env_file():
    """Vérifier le fichier .env"""
    load_dotenv()
    env_path = Path(".env")

    print("=== DIAGNOSTIC ENVIRONNEMENT ===")
    print(f"Fichier .env existe: {env_path.exists()}")
    print(f"Chemin complet: {env_path.absolute()}")

    if env_path.exists():
        print("\nContenu du fichier .env:")
        print("-" * 40)
        with open(env_path, "r") as f:
            content = f.read()
            print(content)
        print("-" * 40)

    print("\nVariables d'environnement pertinentes:")
    env_vars = [
        "HUGGINGFACE_API_KEY",
        "AI_HUGGINGFACE_API_KEY",
        "USERNAME",
        "SECURITY_USERNAME",
        "PASSWORD",
        "SECURITY_PASSWORD",
        "SECRET_KEY",
        "SECURITY_SECRET_KEY",
    ]

    for var in env_vars:
        value = os.getenv(var)
        if value:
            # Masquer les clés sensibles
            if "KEY" in var or "PASSWORD" in var:
                display_value = value[:10] + "..." if len(value) > 10 else "***"
            else:
                display_value = value
            print(f"  {var}: {display_value}")
        else:
            print(f"  {var}: NON DÉFINI")


def test_pydantic_settings():
    """Tester les settings Pydantic étape par étape"""
    print("\n=== TEST PYDANTIC SETTINGS ===")

    try:
        from pydantic_settings import BaseSettings
        from pydantic import Field

        # Test 1: Configuration simple sans préfixe
        print("\n1. Test configuration simple:")

        class SimpleSettings(BaseSettings):
            huggingface_api_key: str = Field(...)

            class Config:
                env_file = ".env"
                extra = "ignore"

        try:
            SimpleSettings()
            print("  ✅ Configuration simple OK")
        except Exception as e:
            print(f"  ❌ Configuration simple échouée: {e}")

        # Test 2: Configuration avec préfixe
        print("\n2. Test configuration avec préfixe:")

        class PrefixSettings(BaseSettings):
            huggingface_api_key: str = Field(...)

            class Config:
                env_prefix = "AI_"
                env_file = ".env"
                extra = "ignore"

        try:
            PrefixSettings()
            print("  ✅ Configuration avec préfixe OK")
        except Exception as e:
            print(f"  ❌ Configuration avec préfixe échouée: {e}")

        # Test 3: Configuration avec valeur par défaut
        print("\n3. Test configuration avec défaut:")

        class DefaultSettings(BaseSettings):
            huggingface_api_key: str = Field(default="test_key")

            class Config:
                env_prefix = "AI_"
                env_file = ".env"
                extra = "ignore"

        try:
            default = DefaultSettings()
            print(
                f"  ✅ Configuration avec défaut OK: {default.huggingface_api_key[:10]}..."
            )
        except Exception as e:
            print(f"  ❌ Configuration avec défaut échouée: {e}")

    except ImportError as e:
        print(f"❌ Erreur import: {e}")


if __name__ == "__main__":
    check_env_file()
    test_pydantic_settings()
