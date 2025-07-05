"""
Architecture Sophistiquée pour SQL Query Visualizer
==================================================

Structure du projet recommandée :

src/
├── config/
│   ├── __init__.py
│   ├── settings.py          # Configuration centralisée
│   └── logging_config.py    # Configuration des logs
├── core/
│   ├── __init__.py
│   ├── exceptions.py        # Exceptions personnalisées
│   ├── models.py           # Modèles Pydantic
│   └── security.py        # Sécurité et authentification
├── agents/
│   ├── __init__.py
│   ├── base_agent.py       # Agent de base
│   ├── sql_agent.py        # Agent SQL amélioré
│   └── visualization_agent.py # Agent de visualisation
├── database/
│   ├── __init__.py
│   ├── connection.py       # Gestionnaire de connexion
│   ├── repositories.py     # Pattern Repository
│   └── migrations.py       # Migrations de base
├── services/
│   ├── __init__.py
│   ├── data_service.py     # Service de données
│   ├── ai_service.py       # Service IA
│   └── cache_service.py    # Service de cache
├── ui/
│   ├── __init__.py
│   ├── components/         # Composants UI réutilisables
│   ├── pages/             # Pages Streamlit
│   └── utils.py           # Utilitaires UI
├── utils/
│   ├── __init__.py
│   ├── validators.py       # Validation des données
│   └── helpers.py         # Fonctions utilitaires
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── main.py                 # Point d'entrée
├── requirements.txt
└── pyproject.toml         # Configuration moderne du projet
"""