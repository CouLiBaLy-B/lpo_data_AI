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
