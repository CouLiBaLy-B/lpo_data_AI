---
title: Lpo Ai
emoji: 👁
colorFrom: pink
colorTo: green
sdk: streamlit
sdk_version: 1.33.0
app_file: app.py
pinned: false
---

# SQL Query Visualizer

Ce projet est une application web construite avec Python et Streamlit. Il permet aux utilisateurs de poser des questions en langage naturel sur une base de données SQLite et d'obtenir une visualisation interactive des données à l'aide de la bibliothèque Plotly.

## Fonctionnalités

- Poser des questions en langage naturel et obtenir des requêtes SQL générées automatiquement
- Visualiser les données sous forme de graphiques et de cartes interactives à l'aide de Plotly
- Authentification des utilisateurs avec un nom d'utilisateur et un mot de passe
- Mettre à jour la base de données avec les données les plus récentes provenant d'une API

## Prérequis

- Python 3.x
- Streamlit
- LangChain
- Plotly
- GeoPy
- Sqlite3
- Pandas
- Requests
- BeautifulSoup4

## Installation

1. Clonez le dépôt GitHub
2. Créez un fichier `.env` à la racine du projet et ajoutez les variables d'environnement suivantes :
    - `USERNAME` : votre nom d'utilisateur
    - `PASSWORD` : votre mot de passe
    - `huggingface_api_key` : votre clé API pour HuggingFace
3. Installez les dépendances avec `pip install -r requirements.txt`

## Utilisation

1. Exécutez l'application avec `streamlit run app.py`
2. Connectez-vous avec vos identifiants
3. Posez une question dans la barre de texte
4. Cliquez sur "Run" pour générer la requête SQL et visualiser les données

## Structure du projet

- `app.py` : Le point d'entrée de l'application Streamlit
- `sql_agent.py` : Contient la logique pour générer les requêtes SQL et le code Plotly à partir des questions de l'utilisateur
- `data_retrieve.py` : Contient la logique pour extraire et transformer les données de l'API vers la base de données SQLite

## Contribuer

Les contributions sont les bienvenues! Veuillez ouvrir une nouvelle issue ou soumettre une pull request pour les améliorations proposées.
