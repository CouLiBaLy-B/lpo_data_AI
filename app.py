import hashlib
import streamlit as st
from dotenv import load_dotenv
import os

from sql_agent import SQLAgent
from data_retrieve.data_retrieve import DataProcessor

load_dotenv()

DB_PATH = "data/sensitive_areas.db"

USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
users = {USERNAME: hashlib.sha256(PASSWORD.encode()).hexdigest()}


def authenticate(username, password, users=users):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    return username in users and users[username] == hashed_password


@st.cache_resource()
def login_cache():
    return {"authenticated": False}


def login():
    st.markdown("<h3>Login</h3>", unsafe_allow_html=True)
    form = st.form("login_form")
    username = form.text_input("Username")
    password = form.text_input("Password", type="password")
    if form.form_submit_button("Login"):
        if authenticate(username, password):
            login_cache()["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid username or password")


def home():
    if not login_cache()["authenticated"]:
        login()
    else:
        question = st.text_input(
            "Enter Question:", "what is the total of zones for each category?"
        )
        if st.button("Run"):
            sql_agent = SQLAgent()
            sql_agent.execute(question=question)
            with st.expander("Voir le code Python"):
                st.code(sql_agent.generate(question=question))


if __name__ == "__main__":
    st.set_page_config(page_title="LPO App", page_icon=":bird:", layout="wide")

    st.markdown("<h2>Welcome to SQL Query Visualizer </h2>", unsafe_allow_html=True)
    st.sidebar.title("Navigation")
    with st.sidebar:
        st.markdown(
            """
        <footer style="position: fixed; bottom: 0; text-align: center;
        background-color: #cecece; border-top: 1px solid #ddd;">
          &copy; 2023 LPO. All rights reserved.
        </footer>
        """,
            unsafe_allow_html=True,
        )

        if st.button("Refresh Data", key="refresh_data"):
            data_processor = DataProcessor()
            data_processor.run()
            st.success("Data refreshed successfully!")

    home()
