"""
Moteur de visualisation déterministe.

Aucun LLM, aucun exec() — la figure Plotly est construite directement
à partir de la forme du DataFrame (colonnes numériques vs catégorielles,
cardinalité, etc.).

Techniques utilisées:
- Analyse de la forme des données (LLM4Vis / LIDA pattern)
- Règles heuristiques pour le choix du graphique
- Plotly Express pour un rendu professionnel
"""

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..core.models import ChartType

COLORS = px.colors.qualitative.Set3
TEMPLATE = "plotly_white"


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def _cat_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(exclude="number").columns.tolist()


def create_figure(
    df: pd.DataFrame,
    chart_type: ChartType,
    title: str,
) -> go.Figure:
    """
    Crée une figure Plotly de façon entièrement déterministe.

    Sélection des axes:
    - axe X → première colonne catégorielle (ou première colonne tout court)
    - axe Y → première colonne numérique
    - couleur → deuxième colonne catégorielle si elle existe

    Dégradation gracieuse: si le type demandé ne correspond pas aux données,
    on bascule sur le graphique en barres.
    """
    if df is None or df.empty:
        return px.bar(title=f"Aucune donnée — {title}", template=TEMPLATE)

    num = _numeric_cols(df)
    cat = _cat_cols(df)
    all_cols = df.columns.tolist()

    x = cat[0] if cat else all_cols[0]
    y = num[0] if num else (all_cols[1] if len(all_cols) > 1 else all_cols[0])
    color = cat[1] if len(cat) > 1 else None

    fig: go.Figure

    if chart_type == ChartType.PIE:
        if num and cat:
            fig = px.pie(
                df, values=y, names=x, title=title,
                color_discrete_sequence=COLORS,
                template=TEMPLATE,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
        else:
            fig = _bar(df, x, y, color, title)

    elif chart_type == ChartType.LINE:
        if len(all_cols) >= 2:
            fig = px.line(
                df, x=x, y=y, title=title,
                markers=True,
                color_discrete_sequence=COLORS,
                template=TEMPLATE,
            )
        else:
            fig = _bar(df, x, y, color, title)

    elif chart_type == ChartType.SCATTER:
        if len(num) >= 2:
            fig = px.scatter(
                df, x=num[0], y=num[1], title=title,
                color=cat[0] if cat else None,
                size=num[2] if len(num) > 2 else None,
                color_discrete_sequence=COLORS,
                template=TEMPLATE,
            )
        else:
            fig = _bar(df, x, y, color, title)

    else:
        fig = _bar(df, x, y, color, title)

    fig.update_layout(
        title={"text": title, "x": 0.05, "font": {"size": 16}},
        margin={"t": 60, "l": 40, "r": 30, "b": 60},
        xaxis_title=x.replace("_", " ").title() if chart_type != ChartType.PIE else None,
        yaxis_title=y.replace("_", " ").title() if chart_type != ChartType.PIE else None,
    )

    return fig


def _bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str],
    title: str,
) -> go.Figure:
    return px.bar(
        df, x=x, y=y, color=color,
        title=title,
        color_discrete_sequence=COLORS,
        template=TEMPLATE,
        text_auto=True,
    )


def auto_chart_type(df: pd.DataFrame, hint: ChartType) -> ChartType:
    """
    Vérifie si le type suggéré est compatible avec les données.
    Sinon, retourne le meilleur type par défaut.
    """
    num = _numeric_cols(df)
    cat = _cat_cols(df)

    if hint == ChartType.PIE and not (num and cat):
        return ChartType.BAR
    if hint == ChartType.SCATTER and len(num) < 2:
        return ChartType.BAR
    if hint == ChartType.LINE and len(df.columns) < 2:
        return ChartType.BAR

    return hint


def _is_temporal(df: pd.DataFrame) -> bool:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return True
        name = col.lower()
        if any(tok in name for tok in ("date", "mois", "month", "année", "annee", "year", "jour")):
            return True
    return False


def infer_chart_type(df: pd.DataFrame) -> ChartType:
    """
    Choisit un type de graphique uniquement d'après la forme des données
    nettoyées (aucune dépendance au LLM ni au schéma).
    """
    if df is None or df.empty or len(df.columns) < 2:
        return ChartType.TABLE

    num = _numeric_cols(df)
    cat = _cat_cols(df)

    if not num:
        return ChartType.TABLE
    if _is_temporal(df):
        return ChartType.LINE
    if len(num) >= 2 and not cat:
        return ChartType.SCATTER
    if cat and len(num) == 1 and len(df) <= 8:
        return ChartType.PIE
    return ChartType.BAR
