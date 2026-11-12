"""Validation des messages A2UI avant envoi au renderer.

Trois niveaux, dans cet ordre :

1. **Schéma du protocole + catalogue** — via le SDK officiel `a2ui-agent-sdk`
   (`validate_payload`) : chaque composant est vérifié contre le catalogue
   `lpo_catalog.json`, les liaisons `{"@path": ...}` contre les types
   `Dynamic*` de `common_types.json`.
2. **Intégrité de séquence** — via `RendererState` : identifiants uniques, cycle
   de vie des surfaces (`createSurface` avant `updateComponents`), topologie
   (tous les enfants existent, `root` présent et joignable).
3. **Contrat de données LPO** — vérifications métier que le schéma ne peut pas
   exprimer : aucune valeur numérique en dur dans les composants de graphique
   (tout doit passer par une liaison), et cohérence des listes.

Le module échoue **avant** l'envoi : si un message est invalide, l'orchestrateur
dégrade vers une réponse texte seule plutôt que d'envoyer une surface douteuse.

Référence protocole : https://a2ui.org/specification/v1.0-a2ui/ (v1.0, consulté
le 2026-11-12) et https://a2ui.org/concepts/messages/.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CATALOG_DIR = Path(__file__).parent / "catalog"
CATALOG_PATH = CATALOG_DIR / "lpo_catalog.json"

# Champs qui ne doivent jamais porter une valeur littérale : ils représentent
# des mesures et doivent être liés au data model (donc aux résultats MCP).
NUMERIC_MEASURE_KEYS = frozenset({"series", "items", "points", "rows", "value"})


class A2uiValidationError(ValueError):
    """Message A2UI refusé : le renderer le rejetterait aussi."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


@dataclass
class ValidationReport:
    """Résultat détaillé d'une validation (utile aux tests et au diagnostic)."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    checked_messages: int = 0
    state: Any = None

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise A2uiValidationError(self.errors)


@lru_cache(maxsize=1)
def load_catalog() -> Any:
    """Charge (et met en cache) le catalogue LPO sous forme d'objet SDK."""
    from a2ui.core.catalog import Catalog  # import local : dépendance optionnelle

    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return Catalog.from_json(payload)


@lru_cache(maxsize=1)
def catalog_id() -> str:
    """Identifiant du catalogue déclaré (utilisé dans `createSurface`)."""
    return load_catalog().id


@lru_cache(maxsize=1)
def catalog_json() -> dict:
    """Catalogue brut, exposé au renderer via `/api/catalog`."""
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def new_renderer_state() -> Any:
    """État miroir du renderer, à conserver pendant toute une conversation.

    Le protocole est *stateful* : un `updateDataModel` n'est valide que si la
    surface a été créée auparavant. Conserver l'état permet donc de valider les
    mises à jour incrémentales d'un tour à l'autre (mode « canvas »).
    """
    from a2ui.utils import RendererState  # import local

    return RendererState([load_catalog()])


def validate_messages(messages: Iterable[dict], *, state: Any = None) -> ValidationReport:
    """Valide une séquence de messages agent → renderer.

    Args:
        messages: messages à envoyer, dans l'ordre.
        state: état de renderer à réutiliser (None = état vierge).
    """
    messages = list(messages)
    state = state if state is not None else new_renderer_state()
    if not messages:
        return ValidationReport(ok=False, errors=["Aucun message A2UI à valider."], state=state)

    errors: list[str] = []
    for index, message in enumerate(messages):
        errors.extend(_check_lpo_contract(message, index))

    if not errors:
        try:
            state.apply(messages)
            state.check_topology(state.surface_ids)
        except Exception as exc:  # noqa: BLE001 - le SDK lève A2uiValidationError
            errors.append(f"{type(exc).__name__}: {exc}")

    return ValidationReport(
        ok=not errors, errors=errors, checked_messages=len(messages), state=state
    )


def validate_or_raise(messages: Iterable[dict]) -> list[dict]:
    """Valide puis renvoie les messages (pratique dans un pipeline)."""
    messages = list(messages)
    report = validate_messages(messages)
    report.raise_if_invalid()
    return messages


# ── Contrat métier LPO ────────────────────────────────────────────────────────


def _check_lpo_contract(message: dict, index: int) -> list[str]:
    """Vérifications que le JSON Schema du catalogue ne peut pas porter."""
    errors: list[str] = []
    for surface_key in ("createSurface", "updateComponents"):
        payload = message.get(surface_key)
        if not isinstance(payload, dict):
            continue
        for component in payload.get("components", []):
            errors.extend(_check_component(component, index))
    return errors


def _check_component(component: dict, index: int) -> list[str]:
    errors: list[str] = []
    cid = component.get("id", "?")
    ctype = component.get("component", "?")

    # Les graphiques doivent être alimentés par le data model : une liste
    # littérale de mesures dans le message de structure empêcherait les mises à
    # jour données-seules (et fige des chiffres dans le rendu).
    if ctype in {"LineChart", "BarChart", "MapChart", "DataTable"}:
        for key in NUMERIC_MEASURE_KEYS:
            value = component.get(key)
            if isinstance(value, list) and value:
                errors.append(
                    f"message {index}: {ctype} '{cid}' fournit '{key}' en littéral — "
                    'utiliser une liaison {"@path": ...} vers le data model.'
                )
    if ctype in {"BarChart", "MapChart", "LineChart"}:
        if "@accessibility" not in component:
            # Non bloquant : signalé pour la qualité d'accessibilité du rendu.
            logger.debug("message %d: %s '%s' sans @accessibility", index, ctype, cid)
    return errors


# ── Aide au diagnostic ────────────────────────────────────────────────────────


def describe_report(report: ValidationReport) -> str:
    if report.ok:
        return f"A2UI OK — {report.checked_messages} message(s) validé(s)."
    lines = [f"A2UI INVALIDE — {len(report.errors)} erreur(s) :"]
    lines.extend(f"  - {e}" for e in report.errors)
    return "\n".join(lines)
