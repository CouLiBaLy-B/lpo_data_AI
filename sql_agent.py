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
        prompt = f"""Vous êtes un assistant d'analyse de données spécialisé
        dans la création de requêtes SQL à partir de questions en langage
        naturel. Votre tâche consiste à convertir la question suivante en
        une requête SQL valide pouvant être exécutée sur la base de
        données 'sensitive_areas.db'.

        Contexte de la base de données :
        La base de données 'sensitive_areas.db' contient des informations sur
        les zones sensibles, y compris leur emplacement, les espèces associées,
        les activités sportives concernées, et les organisations qui ont fourni
        les données. Voici un aperçu des colonnes de la table principale :

        1. `id` (INTEGER) : Identifiant unique de la zone sensible et de
        l'enregistrement.

        2. `name` (TEXT) : Nom de la zone sensible.

        3. `description` (TEXT) : Description de la zone sensible.

        4. `practices` (INTEGER) : Activités sportives liées à la
        zone sensible.

        5. `structure` (TEXT) : Nom ou acronyme de l'organisation qui a
        fourni les données pour cette zone sensible.

        6. `species_id` (INTEGER) : Identifiant de l'espèce associée à la
        zone sensible, ou NULL s'il s'agit d'une zone réglementaire.

        7. `update_datetime` (TIMESTAMP) : Date et heure de la dernière mise
        à jour de la zone sensible.

        8. `create_datetime` (TIMESTAMP) : Date et heure de enregistrement
        de la zone sensible.

        9. `region` (TEXT) : Région où se trouve la zone sensible.

        10. `departement` (TEXT) : Département où se trouve la zone sensible.

        11. `pays` (TEXT) : Pays où se trouve la zone sensible.

        Question : {question}

        Votre requête SQL :
        """
        chain = create_sql_query_chain(self.llm, self.db)
        response = chain.invoke({"question": prompt.format(question=question)})
        return response

    def sql_to_plotly(self, sql: str) -> str:
        prompt = f"""Vous êtes un visualisateur scientifique de données
        expérimenté. Votre tâche consiste à générer du code Python utilisant
        la bibliothèque Plotly pour créer une représentation visuelle
        pertinente et informative à partir de la requête SQL fournie.

        Vous appliquerez vos connaissances approfondies sur les principes de
        la science des données et les techniques de visualisation pour
        développer des graphiques et des cartes efficaces, capables
        de communiquer clairement les tendances temporelles ou
        géographiques présentes dans les données.

        Requête SQL :
        {sql}

        Suit a la lettre les instructions suivantes :

        1. Portez une attention particulière à la qualité du code, à la
        lisibilité et à la pertinence des noms de variables. Suivez
        rigoureusement les bonnes pratiques de programmation en Python.

        2. Assurez-vous que le code Python généré est syntaxiquement correct
        et ne contient pas d'erreurs. Le graphique produit doit être
        correctement affiché et représenter fidèlement les données de
        la requête SQL.

        3. Utilisez toujours la bibliothèque sqlite3 pour interagir avec
        la base de données 'sensitive_areas.db' située dans le dossier 'data'.

        4. N'incluez pas d'instructions d'installation des bibliothèques
        requises (plotly, sqlite3, etc.).

        5. Structurez votre code de manière organisée, sans commentaires.

        6. Respectez strictement le format de sortie suivant avec une
        attention particulière à ne modifié pas la requête SQL:

        ```python
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
        ```

        7. Dans la fonction create_plot, générez un graphique Plotly pertinent
        pour visualiser les données récupérées par la requête SQL.
        Choisissez un type de graphique approprié (par exemple, histogramme,
        diagramme circulaire, cartogramme, etc.) en fonction de la nature
        des données.

        8. Vérifiez soigneusement que votre code est correct avant de
        le soumettre.

        Code Python :
        """
        prompt_template = PromptTemplate(template=prompt,
                                         input_variables=["sql"])
        chain = prompt_template | self.llm
        result = chain.invoke({"sql": sql})
        return result.replace(
                            '```', ''
                            ).replace(
                                'python', ''
                            ).replace(
                                '-', ''
                            ).replace(
                                '"""', ""
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
