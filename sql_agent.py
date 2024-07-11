from langchain_community.utilities import SQLDatabase
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_sql_query_chain

import streamlit as st
import textwrap
import os
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACE_HUB_API_TOKEN = os.getenv("huggingface_api_key")


class SQLAgent:
    def __init__(self):
        self.llm = HuggingFaceEndpoint(
            repo_id="mistralai/MixTraL-8x7B-Instruct-v0.1",
            temperature=0.001,
            repetition_penalty=1.2,
            max_length=500,
            max_new_tokens=1024,
            huggingfacehub_api_token=HUGGINGFACE_HUB_API_TOKEN,
        )
        self.db = SQLDatabase.from_uri("sqlite:///data/sensitive_areas.db")

    def question_to_sql(self, question: str) -> str:
        prompt = f"""
        Vous êtes un assistant d'analyse de données spécialisé dans la
        création de requêtes SQL à partir de questions en langage naturel.

        Tâche : Convertir la question suivante en une requête SQL valide pour
        la base de données 'sensitive_areas.db'.

        Contexte de la base de données :
        'sensitive_areas.db' contient des informations sur les zones sensibles,
        incluant leur emplacement, les espèces associées, les activités
        sportives, et les organisations fournissant les données.

        Colonnes principales :
        1. `id` (INTEGER) : Identifiant unique de la zone sensible.
        2. `name` (TEXT) : Nom de la zone sensible.
        3. `description` (TEXT) : Description de la zone sensible.
        4. `practices` (INTEGER) : Activités sportives liées.
        5. `structure` (TEXT) : Organisation fournissant les données.
        6. `species_id` (INTEGER) : Identifiant de l'espèce associée.
        7. `update_datetime` (TIMESTAMP) : Date et heure de la dernière
        mise à jour.
        8. `create_datetime` (TIMESTAMP) : Date et heure de création.
        9. `region` (TEXT) : Région de la zone sensible.
        10. `departement` (TEXT) : Département de la zone sensible.
        11. `pays` (TEXT) : Pays de la zone sensible.

        Question : {question}

        Votre requête SQL :
        """
        chain = create_sql_query_chain(self.llm, self.db)
        response = chain.invoke({"question": prompt.format(question=question)})
        return response

    def sql_to_plotly(self, sql: str) -> str:
        prompt = f"""
        Vous êtes un visualisateur scientifique de données expérimenté.

        Tâche : Générer du code Python avec Plotly et streamlit pour
        visualiser les données de la requête SQL fournie.

        Requête SQL :
        {sql}

        Instructions :
        1. Assurez-vous que le code Python généré est syntaxiquement
        correct et respecte les bonnes pratiques.

        2. Utilisez la bibliothèque sqlite3 pour interagir avec
        'sensitive_areas.db' dans le dossier 'data'.

        3. Structurez votre code de manière claire sans commentaires inutiles.
        4. Respectez le format de sortie spécifié ci-dessous.

        format:
        def get_data(sql_query):
            import sqlite3
            import pandas as pd
            conn = sqlite3.connect('data/sensitive_areas.db')
            df = pd.read_sql_query(sql_query, conn)
            return df

        def create_plot(data):
            import plotly.express as px

            fig = px.pie(data,
                         values='nb',
                         names="structure",
                         color="structure")
            fig.update_layout(
                title='''Nombre de zones sensibles par structure
                      dans les zones règlementées''')

            return fig

        sql_query = '''la requête SQL originale sans la modifier'''
        data = get_data(sql_query)

        import streamlit as st
        col1, col2 = st.tabs(["Data", "Plot"])
        with col1:
            st.dataframe(data)
        with col2:
            st.plotly_chart(create_plot(data))

        Code Python :
        """
        prompt_template = PromptTemplate(
            template=prompt,
            input_variables=["sql"]
        )
        chain = prompt_template | self.llm
        result = chain.invoke({"sql": sql})
        return (
            result.replace("```", "")
            .replace("python", "")
            .replace("-", "")
            .replace('"""', "")
        )

    def generate(self, question: str) -> str:
        sql = self.question_to_sql(question)
        plotly_code = self.sql_to_plotly(sql)
        return plotly_code

    def execute(self, question: str):
        python_graph = self.generate(question)
        try:
            python_graph = textwrap.dedent(python_graph)
            exec(python_graph)
        except Exception as e:
            st.write(f"Erreur lors de l'exécution du code généré : {e}")
