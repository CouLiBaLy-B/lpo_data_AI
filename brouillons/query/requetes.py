from typing import List, Optional
import sqlite3
import pandas as pd


class SQLQueries:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)

    def query_zone_catg_prop(self,
                             list_categories: Optional[List[str]] = None):
        conditions = []

        if list_categories is not None:
            conditions.append(
                'zone_category IN ("' + '","'.join(list_categories) + '")'
            )

        condition_str = " AND ".join(conditions)
        query = f"""
            SELECT
              zone_category,
              SUM(proportion) as total_proportion
            FROM (
              SELECT
                category as zone_category,
                ROUND(COUNT(DISTINCT id)*1.0/(
                    SELECT
                      COUNT(DISTINCT id)
                    FROM
                      sensitive_areas
                  ), 2) AS proportion
              FROM
                sensitive_areas
              WHERE 1=1 {' and ' + condition_str if len(conditions) >= 1 else condition_str}
              GROUP BY
                months, species_id
            )
            GROUP BY
              zone_category;
            """
        result = pd.read_sql_query(query, self.conn)

        return result

    def query_mon_wise_prop(
        self,
        list_species_id: Optional[List[float]] = None,
        list_months: Optional[List[str]] = None,
    ):
        conditions = []

        if list_months is not None:
            conditions.append('months IN ("' + '","'.join(list_months) + '")')

        if list_species_id is not None:
            _list_species_id = [str(i) for i in list_species_id]
            conditions.append('species_id IN ("' + '","'.join(_list_species_id) + '")')

        condition_str = " AND ".join(conditions)
        query = f"""
            SELECT
              months,
              species_id,
              ROUND(COUNT(id)*1.0/(
                          SELECT
                            COUNT(id)
                          FROM
                            sensitive_areas
                      ), 2) AS proportion
            FROM
              sensitive_areas
            WHERE 1=1  {' and ' + condition_str if len(conditions) >= 1 else condition_str}
            GROUP BY
              months, species_id
            ORDER BY
              months;
            """
        result = pd.read_sql_query(query, self.conn)

        return result

    def query_practice_mon_wise(
        self,
        list_practices: Optional[List[float]] = None,
        list_months: Optional[List[str]] = None,
        list_species_id: Optional[List[float]] = None,
    ):
        conditions = []

        if list_practices is not None:
            _list_practices = [str(i) for i in list_practices]
            conditions.append('practices IN ("' + '","'.join(_list_practices) + '")')

        if list_months is not None:
            conditions.append('months IN ("' + '","'.join(list_months) + '")')

        if list_species_id is not None:
            _list_species_id = [str(i) for i in list_species_id]
            conditions.append('species_id IN ("' + '","'.join(_list_species_id) + '")')

        condition_str = " AND ".join(conditions)
        query = f"""
        SELECT
          practices,
          months,
          species_id,
          COUNT(DISTINCT id) AS nombre_zone
        FROM
          sensitive_areas
        WHERE 1=1 {' AND ' + condition_str if len(conditions) >= 1 else condition_str}
        GROUP BY
          1,
          2,
          3
        ORDER BY months;
        """
        result = pd.read_sql_query(query, self.conn)
        return result

    def query_zone_per_species(
        self,
        list_species_id: Optional[List[float]] = None,
        list_region: Optional[List[str]] = None,
        list_department: Optional[List[str]] = None,
    ):
        conditions = []

        if list_species_id is not None:
            _list_species_id = [str(i) for i in list_species_id]
            conditions.append('species_id IN ("' + '","'.join(_list_species_id) + '")')

        if list_region is not None:
            conditions.append('region IN ("' + '","'.join(list_region) + '")')

        if list_department is not None:
            conditions.append('department IN ("' + '","'.join(list_department) + '")')

        condition_str = " AND ".join(conditions)
        query = f"""
        SELECT
        region,
        department,
        COUNT(DISTINCT id)/12.0 AS number_of_zones
        FROM
        sensitive_areas
        WHERE 1=1 {' AND ' + condition_str if len(conditions) >= 1 else condition_str}
        GROUP BY
        1,2
        ORDER BY number_of_zones DESC;
        """
        result = pd.read_sql_query(query, self.conn)
        return result

    def close_connection(self):
        self.conn.close()
