"""
Cleaning à la lecture — générique et schema-agnostic.

La donnée de production est alimentée par un système externe : on ne la modifie
pas en base, on la NORMALISE au moment de l'afficher. Aucune hypothèse sur les
noms de colonnes : les règles s'appliquent par type de colonne détecté.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Marqueurs textuels considérés comme "vide"
_NULLISH = {"", "none", "null", "nan", "n/a", "na", "-", "undefined", "<na>"}


def clean_dataframe(df: pd.DataFrame, *, drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Nettoie un DataFrame issu de n'importe quelle requête :

    - trim des chaînes + espaces multiples réduits
    - valeurs "nullish" ('', 'NULL', 'N/A'…) → NA
    - colonnes numériques stockées en texte → converties
    - colonnes de dates (heuristique sur le nom) → datetime
    - déduplication des lignes strictement identiques

    Retourne une COPIE ; ne mute pas l'entrée.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    for col in out.columns:
        if _is_textual(out[col]):
            out[col] = _clean_text_series(out[col])
            out[col] = _maybe_numeric(out[col])
            if _looks_like_date(col):
                out[col] = _maybe_datetime(out[col])

    if drop_duplicates:
        before = len(out)
        out = out.drop_duplicates().reset_index(drop=True)
        if before != len(out):
            logger.info("Cleaning: %d doublon(s) supprimé(s).", before - len(out))

    return out


def _is_textual(s: pd.Series) -> bool:
    """Vrai pour les colonnes texte, quel que soit le dtype pandas
    (object historique ou nouveau dtype `str` de pandas 3.0)."""
    if s.dtype == object:
        return True
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return False
    return pd.api.types.is_string_dtype(s)


def _clean_text_series(s: pd.Series) -> pd.Series:
    cleaned = (
        s.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    # Remplace les marqueurs vides par NA (insensible à la casse)
    mask = cleaned.str.lower().isin(_NULLISH)
    return cleaned.mask(mask, other=pd.NA)


def _maybe_numeric(s: pd.Series) -> pd.Series:
    """Convertit en numérique si la quasi-totalité des valeurs non nulles le permettent."""
    non_null = s.dropna()
    if non_null.empty:
        return s
    converted = pd.to_numeric(non_null, errors="coerce")
    if converted.notna().mean() >= 0.95:
        return pd.to_numeric(s, errors="coerce")
    return s


def _maybe_datetime(s: pd.Series) -> pd.Series:
    non_null = s.dropna()
    if non_null.empty:
        return s
    converted = pd.to_datetime(non_null, errors="coerce", format="mixed")
    if converted.notna().mean() >= 0.8:
        return pd.to_datetime(s, errors="coerce", format="mixed")
    return s


def _looks_like_date(column_name: str) -> bool:
    name = column_name.lower()
    return any(tok in name for tok in ("date", "datetime", "_at", "time", "jour", "mois"))


def normalize_label(value: str) -> str:
    """Normalise un libellé pour regroupement (casse de titre, espaces propres)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return value
    return " ".join(str(value).strip().split()).title()
