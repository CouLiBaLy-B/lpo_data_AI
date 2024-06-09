import pandas as pd
import streamlit as st
import plotly.express as px

from query.requetes import SQLQueries


class DashboardApp:
    def __init__(self, db_path):
        self.sql_queries = SQLQueries(db_path)
        self.months = [
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
        self.month_mapping = {k: v + 1 for v, k in enumerate(self.months)}

    def run(
        self,
        list_category,
        list_months,
        list_regions,
        list_department,
        list_species,
        list_practices,
    ):
        df_zone_catg_prop = self.sql_queries.query_zone_catg_prop(
            list_category
            )
        df_mon_wise_prop = self.sql_queries.query_mon_wise_prop(
            list_species, list_months
        )
        df_practice_mon_wise = self.sql_queries.query_practice_mon_wise(
            list_practices, list_months
        )
        df_zone_per_species = self.sql_queries.query_zone_per_species(
            list_species, list_regions, list_department
        )

        df_practice_mon_wise["numeric_months"] = df_practice_mon_wise["months"].map(
            self.month_mapping
        )
        df_mon_wise_prop["numeric_months"] = df_mon_wise_prop["months"].map(
            self.month_mapping
        )

        df_practice_mon_wise = df_practice_mon_wise.sort_values(
            ["numeric_months", "species_id"]
        ).reset_index(drop=True)
        df_practice_mon_wise.drop("numeric_months", axis=1, inplace=True)

        df_mon_wise_prop = df_mon_wise_prop.sort_values(
            ["numeric_months", "proportion"]
        ).reset_index(drop=True)
        df_mon_wise_prop.drop("numeric_months", axis=1, inplace=True)

        self.display_dashboard(
            df_zone_catg_prop,
            df_zone_per_species,
            df_mon_wise_prop,
            df_practice_mon_wise,
        )

    def display_dashboard(
        self,
        df_zone_catg_prop,
        df_zone_per_species,
        df_mon_wise_prop,
        df_practice_mon_wise,
    ):
        st.markdown(
            """<h1 style='text-align: center'>Visualization of
            Biodiv-Sport Database</h1>""",
            unsafe_allow_html=True,
        )

        fig_pie = px.pie(
            df_zone_catg_prop, values="total_proportion", names="zone_category"
        )
        fig_pie.update_layout(
            paper_bgcolor="#002b36",
            plot_bgcolor="#586e75",
            font_color="#fafafa",
            title={"text": "<b>Proportions of Zone Categories</b>",
                   "x": 0.0, "y": 1},
        )

        fig_bar = px.bar(df_zone_per_species, x="region", y="number_of_zones")
        fig_bar.update_layout(
            paper_bgcolor="#002b36",
            plot_bgcolor="#586e75",
            font_color="#fafafa",
            title={
                "text": "<b>Total Number of Zones Per Species</b>",
                "x": 0.0,
                "y": 1,
            },
        )

        fig_line = px.line(
            df_mon_wise_prop, x="months", y="proportion", color="species_id"
        )
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
            nbinsx=12,
            nbinsy=10,
        )
        fig_heatmap.update_layout(
            xaxis={
                "tickmode": "array",
                "tickvals": list(self.month_mapping.keys()),
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


if __name__ == "__main__":
    app = DashboardApp("data/biodiv_sports.db")
    app.run(
        list_category=["Species"],
        list_months=None,
        list_regions=None,
        list_department=None,
        list_species=None,
        list_practices=None,
    )
