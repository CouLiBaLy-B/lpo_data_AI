import streamlit as st
import sqlite3


class DatabaseManager:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def get_distinct_values(self, column):
        query = f"SELECT DISTINCT {column} FROM sensitive_areas"
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        return [row[0] for row in results]

    def close_connection(self):
        self.conn.close()


class SidebarFilters:
    def __init__(self, db_manager):
        self.db_manager = db_manager

    def get_species_data(self):
        species_ids = self.db_manager.get_distinct_values("species_id")
        selected = st.sidebar.multiselect("Select a species ID", species_ids)
        return selected

    def get_practices_data(self):
        practices = self.db_manager.get_distinct_values("practices")
        selected = st.sidebar.multiselect("Select a practice", practices)
        return selected

    def get_regions_data(self):
        regions = self.db_manager.get_distinct_values("region")
        selected = st.sidebar.multiselect("Select a region", regions)
        return selected

    def get_departments_data(self):
        departments = self.db_manager.get_distinct_values("department")
        selected = st.sidebar.multiselect("Select a department", departments)
        return selected

    def get_countries_data(self):
        countries = self.db_manager.get_distinct_values("Pays")
        selected = st.sidebar.multiselect("Select a country", countries)
        return selected

    def get_months_data(self):
        months = self.db_manager.get_distinct_values("months")
        selected = st.sidebar.multiselect("Select a month", months)
        return selected

    def get_category_data(_self):
        selected = st.sidebar.multiselect(
            "Select a category", ["Espèces", "Réglementaire"]
        )
        return selected
