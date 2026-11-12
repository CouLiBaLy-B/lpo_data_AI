"""Floutage de la localisation des espèces sensibles ou protégées.

Trois niveaux, appliqués **côté serveur MCP** (jamais dans le renderer, qui ne
reçoit donc que des coordonnées déjà dégradées) :

- ``exact``       : coordonnées telles quelles (espèces non sensibles) ;
- ``blurred``     : arrondi sur une grille (~11 km par défaut, 1 décimale) ;
- ``department``  : coordonnées supprimées, agrégation au département.

``auto`` (défaut) applique ``blurred`` aux lignes marquées sensibles ou
protégées, ``exact`` aux autres. Le mode ``exact`` peut être interdit par
configuration (``LPO_ALLOW_EXACT_COORDINATES=false``) pour un déploiement
public.

Liste des espèces sensibles
---------------------------

Le schéma exposé peut ne pas porter d'indicateur de sensibilité fiable (le
dépôt réel ne l'a pas). Deux variables permettent alors de fournir la liste
officielle **sans toucher au code** :

- ``LPO_SENSITIVE_SPECIES`` : libellés séparés par des virgules ;
- ``LPO_SENSITIVE_SPECIES_FILE`` : fichier texte, un libellé par ligne
  (les lignes vides et commençant par ``#`` sont ignorées).

Ces libellés sont comparés, insensiblement à la casse, au libellé d'espèce de
chaque ligne : une espèce listée est traitée comme sensible même si la colonne
``sensitive`` vaut 0. La liste est lue à chaque appel (pas de cache), ce qui
permet de la changer sans redémarrer le serveur.
"""

from __future__ import annotations

import math
import os
from typing import Literal

PrivacyMode = Literal["auto", "exact", "blurred", "department"]
ResolvedPrivacy = Literal["exact", "blurred", "department"]

_DEFAULT_DECIMALS = 1
_DEFAULT_ALLOW_EXACT = False  # par défaut on ne diffuse pas de coordonnée exacte


def blur_decimals() -> int:
    """Nombre de décimales conservées en mode flouté (0 = ~111 km, 1 = ~11 km)."""
    try:
        value = int(os.getenv("LPO_BLUR_DECIMALS", str(_DEFAULT_DECIMALS)))
    except ValueError:
        return _DEFAULT_DECIMALS
    return max(0, min(4, value))


def _clean_labels(values: list[str]) -> tuple[str, ...]:
    """Normalise une liste de libellés (déduplication, ordre conservé)."""
    seen: dict[str, None] = {}
    for value in values:
        label = " ".join(str(value).split()).strip()
        if not label or label.startswith("#"):
            continue
        seen.setdefault(label[:120], None)
    return tuple(seen)


def _labels_from_file(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8") as handle:
            return [line.strip() for line in handle]
    except OSError:
        return []


def configured_sensitive_species() -> tuple[str, ...]:
    """Liste de sensibilité fournie par la configuration (vide par défaut)."""
    labels: list[str] = []
    raw = os.getenv("LPO_SENSITIVE_SPECIES", "")
    if raw:
        labels.extend(raw.split(","))
    path = os.getenv("LPO_SENSITIVE_SPECIES_FILE", "").strip()
    if path:
        labels.extend(_labels_from_file(path))
    return _clean_labels(labels)


def _sql_literal(label: str) -> str:
    """Littéral SQL échappé (une apostrophe ne doit rien casser ni injecter)."""
    return "'" + label.replace("'", "''") + "'"


def sensitive_species_sql_list() -> str:
    """Liste SQL échappée des libellés configurés (chaîne vide si aucun)."""
    return ", ".join(_sql_literal(label) for label in configured_sensitive_species())


def sensitive_species_sql_clause(column: str) -> str | None:
    """Clause SQL « l'espèce de cette ligne est-elle dans la liste ? ».

    `column` doit déjà être un identifiant validé (échappé par l'appelant).

    La comparaison est faite deux fois — sur la valeur brute et sur sa version
    minuscule — parce que `LOWER()` de SQLite ne traite que l'ASCII : sans cela,
    « Épervier d'Europe » ne serait pas reconnu à cause du « É ».
    """
    labels = configured_sensitive_species()
    if not labels:
        return None
    raw = ", ".join(_sql_literal(label) for label in labels)
    lowered = ", ".join(_sql_literal(label.lower()) for label in labels)
    return (
        f"(LOWER(TRIM({column})) IN ({lowered}) OR TRIM({column}) IN ({raw}))"
    )


def allow_exact_coordinates() -> bool:
    raw = os.getenv("LPO_ALLOW_EXACT_COORDINATES", str(_DEFAULT_ALLOW_EXACT)).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def is_sensitive(value: object, *, protected_values: tuple[str, ...]) -> bool:
    """Vrai si la ligne correspond à une espèce sensible/protégée."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"", "0", "false", "non", "no", "nan", "none"}:
        return False
    if text in {"1", "true", "oui", "yes"}:
        return True
    return text in {v.lower() for v in protected_values}


def resolve_mode(requested: str | None, *, sensitive: bool) -> ResolvedPrivacy:
    """Détermine le mode effectif pour une ligne."""
    mode = (requested or "auto").strip().lower()
    if mode not in {"auto", "exact", "blurred", "department"}:
        raise ValueError(f"Mode de confidentialité inconnu : {requested!r}")

    if mode == "auto":
        mode = "blurred" if sensitive else "exact"

    if mode == "exact" and sensitive and not allow_exact_coordinates():
        return "blurred"
    return mode  # type: ignore[return-value]


def blur_point(lat: float, lon: float, *, decimals: int | None = None) -> tuple[float, float]:
    """Arrondit un point sur une grille régulière (floutage irréversible)."""
    d = blur_decimals() if decimals is None else decimals
    return round(float(lat), d), round(float(lon), d)


def apply_privacy(
    *,
    lat: object,
    lon: object,
    sensitive: bool,
    mode: ResolvedPrivacy,
) -> tuple[float | None, float | None]:
    """Renvoie les coordonnées diffusables (ou ``None`` si non diffusables)."""
    if lat is None or lon is None:
        return None, None
    try:
        lat_f, lon_f = float(lat), float(lon)
    except (TypeError, ValueError):
        return None, None
    if math.isnan(lat_f) or math.isnan(lon_f):
        return None, None

    if mode == "exact":
        return lat_f, lon_f
    if mode == "blurred":
        return blur_point(lat_f, lon_f)
    return None, None


def accuracy_label(mode: ResolvedPrivacy) -> str:
    """Libellé lisible de la précision spatiale, pour l'affichage."""
    return {
        "exact": "coordonnées exactes",
        "blurred": f"localisation floutée (~{grid_size_km()} km)",
        "department": "département uniquement",
    }[mode]


def grid_size_km(decimals: int | None = None) -> int:
    """Taille approximative de la maille de floutage, en kilomètres."""
    d = blur_decimals() if decimals is None else decimals
    return max(1, int(round(111 * 10**-d)))


def privacy_summary(mode: ResolvedPrivacy, *, sensitive_rows: int) -> str:
    """Phrase de transparence insérée dans les réponses."""
    if mode == "exact":
        return "Coordonnées exactes (aucune espèce sensible dans ce résultat)."
    if mode == "blurred":
        return (
            f"{sensitive_rows} observation(s) d'espèce(s) sensible(s) : "
            f"localisation floutée sur une maille d'environ {grid_size_km()} km."
        )
    return (
        f"{sensitive_rows} observation(s) d'espèce(s) sensible(s) : "
        "seule l'échelle départementale est diffusée."
    )
