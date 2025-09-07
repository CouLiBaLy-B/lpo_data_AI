from datetime import datetime
import sqlite3
import pandas as pd
import polars as pl
import requests
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim


class DataExtractor:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="myApplication")
        self.API_URL = "https://biodiv-sports.fr/api/v2/sensitivearea/"
        self.session = requests.Session()

    def _convertir_date(self, date_string):
        try:
            date_obj = datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S.%f%z")
            formatted_date = date_obj.strftime("%Y-%m-%d")
            return formatted_date
        except ValueError:
            return """Format de date invalide. Assurez-vous que la chaîne est
                      au format 'aaaa-mm-jjThh:mm:ss.ssssss+hh:mm'."""

    def _remove_html_tags(self, text):
        soup = BeautifulSoup(text, "html.parser")
        cleaned_text = soup.get_text()
        return cleaned_text

    def _get_location_info(self, latitude, longitude):
        location = self.geolocator.reverse((latitude, longitude), exactly_one=True)
        address = location.raw["address"]
        return {
            "region": address.get("state", ""),
            "department": address.get("county", ""),
            "country": address.get("country", ""),
        }

    def _type_conversion(self, df):
        str_cols = [
            "id",
            "description",
            "name",
            "structure",
            "region",
            "department",
            "Pays",
        ]
        date_cols = ["update_datetime", "create_datetime"]

        df[str_cols] = df[str_cols].apply(lambda col: col.astype(str))
        df[date_cols] = df[date_cols].apply(lambda col: pd.to_datetime(col))

        return df

    def _data_extract(self, api_url):
        response = self.session.get(api_url)
        res = response.json()
        res = res["results"]
        return res

    def _data_transformation(self, res):
        df = pl.DataFrame(
            [
                {
                    "create_datetime": self._convertir_date(item["create_datetime"]),
                    "id": item["id"],
                    "description": self._remove_html_tags(item["description"]["fr"]),
                    "name": item["name"]["fr"],
                    "structure": item["structure"],
                    "species_id": item["species_id"],
                    "practices": item["practices"],
                    "lat": (
                        item["geometry"]["coordinates"][0][0][0][1]
                        if item["geometry"]["type"] == "MultiPolygon"
                        else item["geometry"]["coordinates"][0][0][1]
                    ),
                    "lon": (
                        item["geometry"]["coordinates"][0][0][0][0]
                        if item["geometry"]["type"] == "MultiPolygon"
                        else item["geometry"]["coordinates"][0][0][0]
                    ),
                    "update_datetime": self._convertir_date(item["update_datetime"]),
                }
                for item in res
            ]
        )

        df = df.with_columns(
            pl.struct(["lat", "lon"])
            .apply(lambda row: self._get_location_info(row["lat"], row["lon"]))
            .alias("location_info")
        )

        df = df.with_columns(
            pl.col("location_info").struct.field("region").alias("region"),
            pl.col("location_info").struct.field("department").alias("department"),
            pl.col("location_info").struct.field("country").alias("Pays"),
        )

        df = df.drop("location_info")
        df = df.with_columns(
            pl.col("create_datetime").str.strptime(pl.Date, "%Y-%m-%d"),
            pl.col("update_datetime").str.strptime(pl.Date, "%Y-%m-%d"),
            pl.col("id").cast(pl.Int64),
            pl.col("species_id").cast(pl.Int64),
        )

        df = df.explode("practices")

        df = df.drop(["lat", "lon"])

        return df

    def extract_and_transform(self):
        res = self._data_extract(self.API_URL)
        data = self._data_transformation(res)
        return data


class DatabaseManager:
    def __init__(self):
        self.conn = None

    def connect(self, db_file="data/sensitive_areas.db"):
        self.conn = sqlite3.connect(db_file)

    def close(self):
        if self.conn is not None:
            self.conn.close()

    def execute_query(self, query):
        cursor = self.conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        return result

    def execute_script(self, query, args=None):
        try:
            cursor = self.conn.cursor()
            if args:
                cursor.execute(query, args)
            else:
                cursor.execute(query)
            self.conn.commit()
        except Exception as e:
            print(f"Erreur pendant l'exécution du script SQL : {e}")

    def save_dataframe(self, df, table_name):
        df_pd = df.to_pandas()
        df_pd.to_sql(name=table_name, con=self.conn, if_exists="replace", index=False)

    def load_dataframe(self, table_name):
        return pd.read_sql_query(f"SELECT * FROM {table_name}", self.conn)


class DataProcessor:
    def __init__(self):
        self.database_manager = DatabaseManager()
        self.data_extractor = DataExtractor()

    def run(self):
        self.database_manager.connect()

        # Extrait et transforme les données
        data = self.data_extractor.extract_and_transform()

        # Sauvegardez les données dans la base de données
        self.save_data(data)

        # Fermez la connexion à la base de données
        self.database_manager.close()

    def save_data(self, data):
        # Schema de la table
        table_schema = """
            CREATE TABLE IF NOT EXISTS sensitive_areas (
                id TEXT PRIMARY KEY,
                description TEXT,
                name TEXT,
                structure TEXT,
                species_id INTEGER,
                practices TEXT,
                create_datetime TIMESTAMP,
                update_datetime TIMESTAMP,
                région TEXT,
                département TEXT,
                pays TEXT,
            );
        """

        # Envoyez le schéma de table en toute sécurité
        self.database_manager.execute_script(table_schema)

        # Transférez les données en toute sécurité
        self.database_manager.save_dataframe(data, "sensitive_areas")
        # Après avoir connecté la base de données
        self.database_manager.execute_script(
            """ALTER TABLE sensitive_areas
                                             ADD COLUMN category TEXT;"""
        )
        self.database_manager.execute_script(
            """UPDATE sensitive_areas
                                             SET category = CASE
                                             WHEN species_id IS NULL THEN
                                             'zone reglementaire'
                                             ELSE 'Espece' END;"""
        )


if __name__ == "__main__":
    data_processor = DataProcessor()
    data_processor.run()
