#!/usr/bin/env python3
"""
LPO AI - SQL Query Visualizer
Application principale avec interface Streamlit
"""

import streamlit as st
import logging
import sys
import os
from pathlib import Path

from src.config.settings import settings
from src.config.logging_config import setup_logging
from src.core.security import SecurityManager, require_auth
from src.core.exceptions import ExceptionHandler, AuthenticationException
from src.ui.pages.dashboard import dashboard_page
from src.ui.pages.query_interface import query_interface_page
from src.ui.utils import StreamlitUtils, UIComponents
from src.services.data_service import data_service

# Configuration de la page Streamlit
st.set_page_config(
    page_title=settings.app_name,
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configuration du logging
logger = setup_logging()


class LPOAIApp:
    """Application principale LPO AI"""

    def __init__(self):
        self.security_manager = SecurityManager()
        self.pages = {
            "🏠 Tableau de bord": dashboard_page,
            "🤖 Assistant SQL": query_interface_page,
        }

        # Initialisation de la session
        self._initialize_session()

    def _initialize_session(self):
        """Initialise les variables de session"""
        if "initialized" not in st.session_state:
            st.session_state.initialized = True
            st.session_state.authenticated = False
            st.session_state.user_id = None
            st.session_state.login_attempts = 0
            st.session_state.current_page = "🏠 Tableau de bord"

            logger.info("Session initialisée")

    def run(self):
        """Lance l'application"""
        try:
            # Vérification de l'authentification
            if not st.session_state.authenticated:
                self._render_login_page()
            else:
                self._render_main_app()

        except Exception as e:
            logger.error(f"Erreur application principale: {e}")
            st.error("Une erreur inattendue s'est produite")

            if settings.debug:
                st.exception(e)

    def _render_login_page(self):
        """Affiche la page de connexion"""
        st.markdown(
            """
        <div style="text-align: center; padding: 2rem;">
            <h1>🦅 LPO AI - SQL Query Visualizer</h1>
            <p style="font-size: 1.2em; color: #666;">
                Analysez les données des zones sensibles avec l'intelligence artificielle
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Formulaire de connexion
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                st.markdown("### 🔐 Connexion")

                with st.form("login_form"):
                    username = st.text_input("👤 Nom d'utilisateur")
                    password = st.text_input("🔒 Mot de passe", type="password")

                    col_login, col_info = st.columns([1, 1])

                    with col_login:
                        login_button = st.form_submit_button(
                            "🚀 Se connecter", type="primary", use_container_width=True
                        )

                    with col_info:
                        if st.form_submit_button(
                            "ℹ️ Informations", use_container_width=True
                        ):
                            st.info(
                                """
                            **Fonctionnalités principales:**
                            - 🤖 Génération automatique de requêtes SQL à partir de questions en français
                            - 📊 Visualisations interactives des données
                            - 🗺️ Analyses géographiques des zones sensibles
                            - 📈 Tableaux de bord en temps réel
                            - 💾 Export des données et rapports
            
                            **Technologies utilisées:**
                            - Streamlit pour l'interface utilisateur
                            - Hugging Face pour l'IA
                            - Plotly pour les visualisations
                            - SQLite pour la base de données
            
                            **Version:** 1.0.0  
                            **Développé par:** LPO Data Team
                            """
                            )

                # Traitement de la connexion
                if login_button:
                    self._handle_login(username, password)

                # Affichage des tentatives de connexion
                if st.session_state.login_attempts > 0:
                    attempts_left = 3 - st.session_state.login_attempts
                    if attempts_left > 0:
                        st.warning(f"⚠️ {attempts_left} tentative(s) restante(s)")
                    else:
                        st.error("🚫 Trop de tentatives. Veuillez réessayer plus tard.")

        # Informations sur l'application
        with st.expander("ℹ️ À propos de l'application"):
            st.markdown(
                """
            ### LPO AI - SQL Query Visualizer
            
            Cette application permet d'analyser les données des zones sensibles 
            de la Ligue pour la Protection des Oiseaux (LPO) en utilisant 
            l'intelligence artificielle.
            
            **Fonctionnalités principales:**
            - 🤖 Génération automatique de requêtes SQL à partir de questions en français
            - 📊 Visualisations interactives des données
            - 🗺️ Analyses géographiques des zones sensibles
            - 📈 Tableaux de bord en temps réel
            - 💾 Export des données et rapports
            
            **Technologies utilisées:**
            - Streamlit pour l'interface utilisateur
            - Hugging Face pour l'IA
            - Plotly pour les visualisations
            - SQLite pour la base de données
            
            **Version:** 1.0.0  
            **Développé par:** LPO Data Team
            """
            )

    def _handle_login(self, username: str, password: str):
        """Gère la tentative de connexion"""
        try:
            if not username or not password:
                st.error("❌ Veuillez saisir vos identifiants")
                return

            # Tentative d'authentification
            token = self.security_manager.authenticate_user(username, password)

            if token:
                st.session_state.authenticated = True
                st.session_state.user_id = username
                st.session_state.auth_token = token
                st.session_state.login_attempts = 0

                logger.info(f"Connexion réussie pour l'utilisateur: {username}")
                st.success("✅ Connexion réussie!")
                st.rerun()

        except AuthenticationException as e:
            st.session_state.login_attempts += 1
            logger.warning(f"Tentative de connexion échouée: {username}")

            if e.error_code == "USER_BLOCKED":
                st.error("🚫 Compte temporairement bloqué")
            else:
                st.error("❌ Identifiants incorrects")

        except Exception as e:
            logger.error(f"Erreur lors de la connexion: {e}")
            st.error("❌ Erreur de connexion")

    def _render_main_app(self):
        """Affiche l'application principale"""
        # En-tête
        self._render_header()

        # Barre latérale
        self._render_sidebar()

        # Contenu principal
        self._render_main_content()

        # Pied de page
        self._render_footer()

    def _render_header(self):
        """Affiche l'en-tête de l'application"""
        st.markdown(
            """
        <div style="background: linear-gradient(90deg, #1f4e79 0%, #2e7d32 100%); 
                    padding: 1rem; margin: -1rem -1rem 2rem -1rem; color: white;">
            <h1 style="margin: 0; display: flex; align-items: center;">
                🦅 LPO AI - SQL Query Visualizer
                <span style="font-size: 0.5em; margin-left: auto; opacity: 0.8;">
                    v1.0.0
                </span>
            </h1>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                Analyse intelligente des zones sensibles
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    def _render_sidebar(self):
        """Affiche la barre latérale"""
        with st.sidebar:
            # Informations utilisateur
            st.markdown(
                f"""
            <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                <h4 style="margin: 0;">👤 {st.session_state.user_id}</h4>
                <small>Connecté depuis {self._get_session_duration()}</small>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Navigation
            st.markdown("### 🧭 Navigation")

            for page_name in self.pages.keys():
                if st.button(
                    page_name,
                    key=f"nav_{page_name}",
                    use_container_width=True,
                    type=(
                        "primary"
                        if st.session_state.current_page == page_name
                        else "secondary"
                    ),
                ):
                    st.session_state.current_page = page_name
                    st.rerun()

            st.divider()

            # Actions rapides
            st.markdown("### ⚡ Actions rapides")

            if st.button("🔄 Actualiser les données", use_container_width=True):
                with st.spinner("Actualisation en cours..."):
                    result = data_service.refresh_data()
                    if result["success"]:
                        st.success("✅ Données actualisées!")
                    else:
                        st.error(f"❌ Erreur: {result['message']}")

            if st.button("🗑️ Vider le cache", use_container_width=True):
                from src.services.cache_service import cache_service

                cache_service.clear()
                st.success("✅ Cache vidé!")

            st.divider()

            # Statistiques rapides
            st.markdown("### 📊 Statistiques")
            try:
                stats = data_service.get_quick_stats()

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Zones", stats.get("total_zones", 0))
                    st.metric("Régions", stats.get("total_regions", 0))

                with col2:
                    st.metric("Structures", stats.get("total_structures", 0))

                    # Répartition par catégorie
                    category_dist = stats.get("category_distribution", {})
                    if category_dist:
                        especes = category_dist.get("Espece", 0)
                        reglementaires = category_dist.get("zone reglementaire", 0)
                        st.metric("Espèces/Régl.", f"{especes}/{reglementaires}")

            except Exception as e:
                st.error("Erreur chargement stats")
                logger.error(f"Erreur stats sidebar: {e}")

            st.divider()

            # Déconnexion
            if st.button(
                "🚪 Se déconnecter", use_container_width=True, type="secondary"
            ):
                self._handle_logout()

    def _render_main_content(self):
        """Affiche le contenu principal"""
        try:
            current_page = st.session_state.current_page

            if current_page in self.pages:
                page_instance = self.pages[current_page]
                page_instance.render()
            else:
                st.error(f"Page non trouvée: {current_page}")

        except Exception as e:
            logger.error(f"Erreur affichage page: {e}")
            st.error("Erreur lors de l'affichage de la page")

            if settings.debug:
                st.exception(e)

    def _render_footer(self):
        """Affiche le pied de page"""
        st.markdown(
            """
        <div style="margin-top: 3rem; padding: 2rem; background: #f8f9fa; 
                    border-top: 1px solid #dee2e6; text-align: center;">
            <p style="margin: 0; color: #6c757d;">
                © 2024 LPO (Ligue pour la Protection des Oiseaux) - 
                Tous droits réservés
            </p>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9em; color: #6c757d;">
                Développé avec ❤️ pour la protection de la biodiversité
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    def _get_session_duration(self) -> str:
        """Retourne la durée de la session"""
        # Simplification - dans un vrai système, on stockerait l'heure de connexion
        return "quelques minutes"

    def _handle_logout(self):
        """Gère la déconnexion"""
        try:
            user_id = st.session_state.get("user_id", "unknown")

            # Nettoyage de la session
            for key in list(st.session_state.keys()):
                del st.session_state[key]

            logger.info(f"Déconnexion de l'utilisateur: {user_id}")
            st.success("✅ Déconnexion réussie")
            st.rerun()

        except Exception as e:
            logger.error(f"Erreur lors de la déconnexion: {e}")
            st.error("Erreur lors de la déconnexion")


def main():
    """Point d'entrée principal"""
    try:
        # Création et lancement de l'application
        app = LPOAIApp()
        app.run()

    except Exception as e:
        logger.critical(f"Erreur critique application: {e}")
        st.error("❌ Erreur critique de l'application")

        if settings.debug:
            st.exception(e)


if __name__ == "__main__":
    main()
