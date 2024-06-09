import datetime
import sqlite3
import pandas as pd
import requests
from bs4 import BeautifulSoup
import geopy
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
API_URL = os.getenv("BIODIV_API_URL")
DATABASE_PATH = "data/biodiv_sports.db"
USER_AGENT = os.getenv("USER_AGENT", "myApplication")


class DataExtractor:
    def __init__(self):
        self.geolocator = geopy.Nominatim(user_agent=USER_AGENT)
        self.session = requests.Session()

    def _convertir_date(self, date_string):
        try:
            return datetime.datetime.strptime(date_string,
                                              "%Y-%m-%dT%H:%M:%S.%f%z")
        except ValueError:
            return None

    def _get_location_info(self, latitude, longitude):
        if latitude and longitude:
            location = self.geolocator.reverse((latitude, longitude),
                                               exactly_one=True)
            if location:
                address = location.raw["address"]
                region = address.get("state", "")
                department = address.get("county", "")
                country = address.get("country", "")
                return region, department, country
        return "", "", ""

    def _add_months_names(self, df):
        months_list = [
            "janvier",
            "fevrier",
            "mars",
            "avril",
            "mai",
            "juin",
            "juillet",
            "août",
            "septembre",
            "octobre",
            "novembre",
            "decembre",
        ]
        df_with_months = pd.DataFrame()
        for month, value in enumerate(months_list, start=1):
            df_filtered = df[df["period"].str.contains(value, na=False)]
            df_with_months = pd.concat([df_with_months, df_filtered])
        return df_with_months

    def _type_conversion(self, df):
        str_cols = [
            "id",
            "name",
            "structure",
            "region",
            "department",
            "Pays",
        ]
        date_cols = ["update_datetime", "create_datetime", "months"]
        df[str_cols] = df[str_cols].apply(lambda col: col.astype(str))
        df[date_cols] = df[date_cols].apply(
            lambda col: pd.to_datetime(col, errors="ignore")
        )
        return df

    def extract_and_transform(self):
        res = self._data_extract(API_URL)
        data = self._data_transformation(res)
        return data

    def _data_extract(self, api_url):
        response = self.session.get(api_url)
        if response.status_code == 200:
            res = response.json()["results"]
            if res:
                data = pd.DataFrame(res)
                return data
        return pd.DataFrame()

    def _data_transformation(self, res):
        data = res.apply(
            lambda row: pd.Series(
                [
                    self._convertir_date(row["create_datetime"]),
                    row["id"],
                    row["name"],
                    row["structure"],
                    row["species_id"],
                    row["period"],
                    self._convertir_date(row["update_datetime"]),
                ]
            ),
            axis=1,
        )
        data = data.rename(
            columns={
                0: "create_datetime",
                1: "id",
                2: "name",
                3: "structure",
                4: "species_id",
                5: "period",
                6: "update_datetime",
            }
        )
        data = data.reset_index(drop=True)
        df = pd.DataFrame(data)
        df["practices"] = df.apply(
            lambda x: [y for y in x["structure"].split(",") if y], axis=1
        )
        df["practices"] = df["practices"].astype(str)
        df = df.join(
            df["practices"]
            .str.split(",", expand=True)
            .stack()
            .str.strip()
            .rename("practices")
        )
        df = self._add_months_names(df)
        df = self._type_conversion(df)
        df = df.explode("practices")
        return df


class DataRetriever:
    def __init__(self):
        self.conn = None
        self.data_extractor = DataExtractor()

    def run(self):
        with st.spinner("Connecting to database..."):
            self.conn = sqlite3.connect(DATABASE_PATH)

        res = None
        with st.spinner("Extracting and transforming data..."):
            res = self.data_extractor.extract_and_transform()

        if res is not None:
            with st.spinner("Saving data to database..."):
                res.to_sql(
                    "sensitive_areas", self.conn, if_exists="replace",
                    index=False
                )
            st.success("Done!")
        else:
            st.error(
                """Une erreur s'est produite lors de l'extraction
                et de la transformation des données."""
            )

        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
