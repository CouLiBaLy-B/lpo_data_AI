#!/usr/bin/env python3
"""
LPO AI — SQL Query Visualizer
Point d'entrée principal de l'application Streamlit.
"""

import logging

import streamlit as st

from src.core.exceptions import AuthenticationException
from src.core.security import SecurityManager
from src.ui.pages.dashboard import dashboard_page
from src.ui.pages.query_interface import query_interface_page

st.set_page_config(
    page_title="LPO AI — Assistant SQL",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/app.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

PAGES = {
    "🏠 Tableau de bord": dashboard_page,
    "🤖 Assistant SQL": query_interface_page,
}


def _init_session():
    defaults = {
        "authenticated": False,
        "user_id": None,
        "current_page": "🏠 Tableau de bord",
        "messages": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _login_page():
    st.markdown(
        """
        <div style="text-align:center; padding:2rem 0 1rem;">
            <h1>🦅 LPO AI — Assistant SQL</h1>
            <p style="color:#666; font-size:1.1em;">
                Analysez les zones sensibles de la LPO grâce à l'intelligence artificielle
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.form("login_form"):
            st.subheader("🔐 Connexion")
            username = st.text_input("Identifiant")
            password = st.text_input("Mot de passe", type="password")
            submitted = st.form_submit_button(
                "Se connecter", type="primary", use_container_width=True
            )

        if submitted:
            if not username or not password:
                st.error("Veuillez saisir vos identifiants.")
                return
            try:
                sm = SecurityManager()
                token = sm.authenticate_user(username, password)
                st.session_state.authenticated = True
                st.session_state.user_id = username
                st.session_state.auth_token = token
                logger.info(f"Connexion: {username}")
                st.rerun()
            except AuthenticationException as e:
                blocked = e.error_code == "USER_BLOCKED"
                st.error("Compte bloqué (15 min)." if blocked else "Identifiants incorrects.")
            except Exception as e:
                st.error(f"Erreur de connexion : {e}")


def _main_app():
    # En-tête
    st.markdown(
        """
        <div style="background:linear-gradient(90deg,#1f4e79,#2e7d32);
                    padding:.8rem 1.5rem; margin:-1rem -1rem 1.5rem; color:white;
                    display:flex; align-items:center;">
            <span style="font-size:1.6rem; font-weight:700;">🦅 LPO AI</span>
            <span style="margin-left:.5rem; opacity:.8; font-size:.9rem;">
                — Analyse des zones sensibles
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Barre latérale
    with st.sidebar:
        st.markdown(f"**👤 {st.session_state.user_id}**")
        st.divider()
        st.markdown("### Navigation")
        for name in PAGES:
            is_active = st.session_state.current_page == name
            if st.button(name, key=f"nav_{name}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state.current_page = name
                st.rerun()

        st.divider()

        # Infos connexion
        from src.database import engine as db_engine
        try:
            st.caption(f"🗄️ Base : **{db_engine.get_engine().dialect.name}** · lecture seule")
        except Exception as e:  # noqa: BLE001
            st.caption(f"🗄️ Base : non connectée ({e})")

        # ETL d'ingestion : réservé au dev (SQLite). En prod la base est externe.
        try:
            is_sqlite = db_engine.get_engine().dialect.name == "sqlite"
        except Exception:  # noqa: BLE001
            is_sqlite = False
        if is_sqlite and st.button("🔄 Recharger (dev, SQLite)", use_container_width=True):
            from data_retrieve.data_retrieve import DataProcessor
            with st.spinner("Récupération des données…"):
                try:
                    DataProcessor().run()
                    db_engine.reset_engine()
                    st.success("Données rechargées !")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Erreur : {e}")

        st.divider()
        if st.button("🚪 Déconnexion", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Page courante
    page = PAGES.get(st.session_state.current_page)
    if page:
        page.render()
    else:
        st.error("Page introuvable.")


def main():
    _init_session()
    if not st.session_state.authenticated:
        _login_page()
    else:
        _main_app()


if __name__ == "__main__":
    main()
