from langchain_community.llms import HuggingFaceHub
from langchain.chains import SequentialChain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import os
import hashlib

load_dotenv()


HUGGINGFACE_HUB_API_TOKEN = os.getenv("HUGGINGFACE_HUB_API_TOKEN")


SCHEMA_STR = """ Table Name: sensitive_areas

!important : data_path = data/biodiv_sports.db

Columns:

1. `id` (INTEGER): Unique identifier for the sensitive area and record.
2. `name` (TEXT): Name of the sensitive area.
3. `description` (TEXT): Description of the sensitive area.
4. `practices` (INTEGER): Sporting activities related to the sensitive area.
5. `structure` (TEXT): Name or acronym of the organization that provided the data for this sensitive area.
6. `species_id` (INTEGER): Identifier for the species associated with the sensitive area, or NULL if it is a regulatory zone.
7. `update_datetime` (TIMESTAMP): Date and time when the sensitive area was last updated.
8. `create_datetime` (TIMESTAMP): Date and time when the sensitive area was created.
9. `region` (TEXT): Region where the sensitive area is located.
10. `departement` (TEXT): Department where the sensitive area is located.
11. `pays` (TEXT): Country where the sensitive area is located.
12. `months` (TEXT): Month during which the species is present in the sensitive area.

Other information:
- category='spece' if species_id else'regulatory zone'
category is not a column name.

- Each id represents a single record.
"""


class SqlToPlotlyStreamlit:
    @staticmethod
    def chat_prompt_template() -> list:
        first_prompt = ChatPromptTemplate.from_template(
            """Generate the optimal SQL query considering the following table schema and the user's question.
           !IMPORTANT: Generate exactly one query and ensure that the query can be used directly without further  \
            modification with sqlite3. Ensure that the columns used in the generated query match the table schema."""
            "!Important: if there are aggregations arroundi the result to 2 decimal places always"
            """Table schema:\n{schema}\nQuestion: {question}"""
        )

        second_prompt = ChatPromptTemplate.from_template(
            "Convert the provided SQL query {query} into a Python best graph code snippet. Use the plotly library otherwise seaborn with error handling for visualization."
            "!IMPORTANT : Provide detailed instructions about loading necessary libraries very important, setting up credentials,"
            "and configuring parameters needed for connecting to the database, creating the figure, and showing the plot."
            "the graph will part of an streamlit app so import streamlit, pandas and all librairie that will be use."
            "Follow the following step:"
            "1 - import all neccessary librairies"
            "2 - settting up database connection with sqlite.connect"
            "3 - Use pandas read_sql_query to retrieve data from database"
            "4 - close database connection"
            "5 - try Plotly to build the graph if error use seaborn"
            "6 - Use distinct color in the graph"
            "7 -  Display the dataframe in table format"
            "8 - Never use statment if __name__ == '__main__' "
        )

        third_prompt = ChatPromptTemplate.from_template(
            """Return a JSON object containing the original {question}, extracted SQL query {query}, and extracted corresponding Python code {python_graph} snippet separated by two newlines ('\n')."""
        )

        fourth_prompt = ChatPromptTemplate.from_template(
            """Generate another type of graphic for the {query} different from {python_graph}."""
        )

        return [first_prompt, second_prompt, third_prompt, fourth_prompt]

    @staticmethod
    def generate_anwers(llm) -> list:
        first_prompt, second_prompt, third_prompt, fourth_prompt = (
            SqlToPlotlyStreamlit.chat_prompt_template()
        )

        chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="query")
        chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key="python_graph")
        chain_three = LLMChain(llm=llm, prompt=third_prompt, output_key="json")
        chain_four = LLMChain(
            llm=llm, prompt=fourth_prompt, output_key="reformulated_query"
        )

        return [chain_one, chain_two, chain_three, chain_four]

    @staticmethod
    def format_python_code_string(code_str: str) -> str:
        import re

        # Supprimer les espaces inutiles
        code_str = re.sub(r"^\s+", "", code_str, flags=re.MULTILINE)
        code_str = re.sub(r"\s+", " ", code_str)
        code_str = re.sub(r"\s+$", "", code_str, flags=re.MULTILINE)

        # Ajouter des espaces avant et après les opérateurs
        code_str = re.sub(r"(\+|\-|\*|\/|\=|\(|\))", r" \1 ", code_str)
        code_str = re.sub(r"(\w+)\s*(\w+)", r"\1 \2", code_str)

        # Ajouter des espaces après les virgules
        code_str = re.sub(r",\s*", ", ", code_str)

        # Ajouter des espaces après les deux-points
        code_str = re.sub(r":\s*", ": ", code_str)

        # Ajouter des espaces avant les crochets et accolades
        code_str = re.sub(r"(\w+)\s*(\{|\[)", r"\1 \2", code_str)
        code_str = re.sub(r"(\}|\])", r" \1", code_str)

        # Aligner les indentations
        lines = code_str.split("\n")
        indentation_levels = []
        for line in lines:
            indentation_level = len(line) - len(line.lstrip())
            indentation_levels.append(indentation_level)
        max_indentation_level = max(indentation_levels)
        indentation_spaces = " " * max_indentation_level
        code_str = "\n".join(indentation_spaces + line.lstrip() for line in lines)

        # Ajouter des espaces entre les mots-clés et les identifiants
        code_str = re.sub(r"(\b(and|or|not|is|in)\b)\s*(\w+)", r"\1 \3", code_str)

        return code_str

    @staticmethod
    def code_execution(code_str):

        code_str = SqlToPlotlyStreamlit.format_python_code_string(code_str)
        # Exécuter le code dans un environnement isolé
        namespace = {}
        exec(code_str, namespace)

        # Récupérer la figure du namespace
        fig = namespace.get("fig")

        if isinstance(fig, go.Figure):

            st.plotly_chart(fig)

    def __init__(self, schema: str, question: str):
        self.schema = schema
        self.question = question
        self.llm = HuggingFaceHub(
            repo_id="mistralai/MixTraL-8x7B-Instruct-v0.1",
            model_kwargs={
                "temperature": 0.001,
                "max_length": 5000,
                "max_new_tokens": 1024,
            },
            huggingfacehub_api_token=HUGGINGFACE_HUB_API_TOKEN,
        )

    def execute_overall_flow(self):
        error_occurred = False

        try:
            overall_chain = SequentialChain(
                chains=list(SqlToPlotlyStreamlit.generate_anwers(self.llm)),
                input_variables=["schema", "question"],
                output_variables=["query", "python_graph", "json"],
                verbose=True,
            )
            result = overall_chain({"schema": self.schema, "question": self.question})
            self.generated_python_code = (
                result["json"].split("```python")[-1].split("```")[0].replace("```", "")
            )
            # self.code_execution(self.generated_python_code)

        except Exception as e:
            error_message = f"\nAttempt: An exception occurred while executing the generated code:{e}"
            st.write(error_message)
            error_occurred = True

        if not error_occurred:
            st.write("## Generated Python")
            st.code(self.generated_python_code)


def main():
    st.set_page_config(layout="wide")
    st.title("SQL Query Visualizer")

    schema = SCHEMA_STR
    question = st.text_input(
        "Enter Question:", "what is the total of zones for each category ?"
    )
    if st.button("Run"):
        obj = SqlToPlotlyStreamlit(schema, question)
        obj.execute_overall_flow()
        exec(obj.generated_python_code)
