import pandas as pd
import streamlit as st
import sqlite3
import plotly.express as px
from query.requetes import (query_zone_catg_prop,
                           query_mon_wise_prop,
                           query_practice_mon_wise,
                           query_zone_per_species
                           )
# Database connection
conn = sqlite3.connect("data/biodiv_sports.db")


def dashboard():
    conn = sqlite3.connect("data/biodiv_sports.db")

    # Read query results into Pandas DataFrames
    df_zone_catg_prop = pd.read_sql_query(query_zone_catg_prop, conn)
    df_mon_wise_prop = pd.read_sql_query(query_mon_wise_prop, conn)
    df_practice_mon_wise = pd.read_sql_query(query_practice_mon_wise, conn)
    df_zone_per_species = pd.read_sql_query(query_zone_per_species, conn)

    # Map month names to their corresponding numbers
    MONTHS = [
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
    MONTH_MAPPING = {k: v + 1 for v, k in enumerate(MONTHS)}

    # Add a numeric_months column based on MONTH_MAPPING
    df_practice_mon_wise["numeric_months"] = df_practice_mon_wise["months"].map(
        MONTH_MAPPING
    )
    df_mon_wise_prop["numeric_months"] = df_mon_wise_prop["months"]
    .map(MONTH_MAPPING)

    # Sort by numeric_months and reset index before dropping it
    df_practice_mon_wise = df_practice_mon_wise.sort_values(
        ["numeric_months", "species_id"]
    ).reset_index(drop=True)
    df_practice_mon_wise.drop("numeric_months", axis=1, inplace=True)

    df_mon_wise_prop = df_mon_wise_prop.sort_values(
        ["numeric_months", "proportion"]
    ).reset_index(drop=True)
    df_mon_wise_prop.drop("numeric_months", axis=1, inplace=True)

    st.markdown(
        """<h1 style='text-align: center'>Visualization of Biodiv-Sport
                Database</h1>""",
        unsafe_allow_html=True,
    )

    # Charts
    fig_pie = px.pie(df_zone_catg_prop, values="proportion", names="zone_category")
    fig_pie.update_layout(
        paper_bgcolor="#002b36",
        plot_bgcolor="#586e75",
        font_color="#fafafa",
        title={"text": "<b>Proportions of Zone Categories</b>", "x": 0.0, "y": 1},
    )

    fig_bar = px.bar(df_zone_per_species, x="name", y="number_of_zones")
    fig_bar.update_layout(
        paper_bgcolor="#002b36",
        plot_bgcolor="#586e75",
        font_color="#fafafa",
        title={"text": "<b>Total Number of Zones Per Species</b>", "x": 0.0, "y": 1},
    )

    fig_line = px.line(df_mon_wise_prop, x="months", y="proportion", color="species_id")
    fig_line.update_layout(
        paper_bgcolor="#002b36",
        plot_bgcolor="#586e75",
        font_color="#fafafa",
        title={
            "text": "<b>Monthly Proportion Distribution Across Year</b>",
            "x": 0.0,
            "y": 1,
        },
    )

    fig_heatmap = px.density_heatmap(
        df_practice_mon_wise,
        x="months",
        y="practices",
        z="nombre_zone",
        color_continuous_scale="OrRd",
        nbinsx=12,  # Set the number of bins in the x direction (months)
        nbinsy=10,
    )
    fig_heatmap.update_layout(
        xaxis={
            "tickmode": "array",
            "tickvals": list(MONTH_MAPPING.keys()),
        },
        paper_bgcolor="#002b36",
        plot_bgcolor="#586e75",
        font_color="#fafafa",
        title={
            "text": "Number of Zones According to Activity Practiced and By Month",
            "x": 0.0,
            "y": 1.0,
        },
    )

    st.plotly_chart(fig_pie)

    st.plotly_chart(fig_bar)

    st.plotly_chart(fig_line)

    st.plotly_chart(fig_heatmap)
