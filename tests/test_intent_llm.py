"""Chemin LLM de l'extraction d'intention, sans clé ni appel réseau.

Le modèle est remplacé par un faux objet qui renvoie l'objet structuré attendu.
Cela permet de tester ce qui compte vraiment : les **garde-fous** appliqués à la
sortie du modèle (vocabulaire fermé, dates validées, périmètre, cohérence du
type de vue) et le repli sur l'heuristique en cas d'échec.

Aucun test ne dépend d'une clé API : la suite reste verte hors ligne.
"""

from __future__ import annotations

import sys
import types
from datetime import date

import pytest

from orchestrator.intent import IntentModel, IntentParser

TODAY = date(2026, 11, 12)
SPECIES = ["Milan royal", "Busard cendré", "Héron cendré"]
DEPARTMENTS = ["Gironde", "Finistère"]
REGIONS = ["Bretagne"]


class FakeStructuredModel:
    """Imite `model.with_structured_output(IntentModel).invoke(...)`."""

    def __init__(self, payload, *, raise_error: bool = False):
        self.payload = payload
        self.raise_error = raise_error
        self.calls: list[list] = []

    @property
    def prompts(self) -> list[str]:
        return [str(messages) for messages in self.calls]

    def with_structured_output(self, schema):
        outer = self
        assert schema is IntentModel, "le schéma de sortie doit être IntentModel"

        class Structured:
            def invoke(self, messages):
                outer.calls.append(messages)
                if outer.raise_error:
                    raise RuntimeError("quota dépassé")
                return outer.payload

        return Structured()


def make_parser(monkeypatch, payload, *, raise_error: bool = False, enabled: bool = True):
    """Remplace `src.ai.model_factory` par un module factice dans `sys.modules`.

    On n'importe **pas** la vraie fabrique : elle dépend de `langchain`, qui ne
    fait pas partie des dépendances de test. C'est aussi la preuve que le chemin
    LLM est optionnel — sans lui, l'agent fonctionne en heuristique.
    """
    fake = FakeStructuredModel(payload, raise_error=raise_error)

    def create_model(temperature: float = 0.0, **kwargs):
        return fake

    module = types.ModuleType("src.ai.model_factory")
    module.create_model = create_model
    module.available_providers = lambda: ["anthropic", "openai"]
    monkeypatch.setitem(sys.modules, "src.ai.model_factory", module)
    if enabled:
        monkeypatch.setenv("LPO_LLM_INTENT", "auto")
    else:
        monkeypatch.setenv("LPO_LLM_INTENT", "off")
    parser = IntentParser(species_vocab=SPECIES, departments=DEPARTMENTS, regions=REGIONS)
    return parser, fake


def test_valid_llm_output_is_used_as_is(monkeypatch):
    payload = IntentModel(
        kind="trend",
        species=["Milan royal"],
        start_date="2017-01-01",
        end_date="2026-11-12",
        period_label="2017–2026",
        period_preset="10y",
        granularity="year",
        rationale="Évolution demandée sur 10 ans.",
    )
    parser, fake = make_parser(monkeypatch, payload)
    intent = parser.parse("Évolution du Milan royal sur 10 ans ?", today=TODAY)

    assert intent.kind == "trend"
    assert intent.species == ["Milan royal"]
    assert intent.start_date == "2017-01-01"
    assert intent.rationale == "Évolution demandée sur 10 ans."
    assert fake.calls, "le modèle doit avoir été appelé"
    # Le prompt contient le vocabulaire fermé et interdit les chiffres.
    prompt = str(fake.calls[0])
    assert "Milan royal" in prompt
    assert "aucun chiffre" in prompt.lower()


def test_species_outside_vocabulary_are_dropped(monkeypatch):
    payload = IntentModel(
        kind="trend",
        species=["Milan royal", "Dragon de Komodo", "Tigre"],
        period_preset="all",
    )
    parser, _ = make_parser(monkeypatch, payload)
    intent = parser.parse("évolution du milan royal", today=TODAY)
    assert intent.species == ["Milan royal"]


def test_comparison_with_one_species_falls_back_to_trend(monkeypatch):
    payload = IntentModel(kind="comparison", species=["Milan royal"], period_preset="all")
    parser, _ = make_parser(monkeypatch, payload)
    assert parser.parse("compare milan royal", today=TODAY).kind == "trend"


def test_species_kind_requires_exactly_one_species(monkeypatch):
    payload = IntentModel(kind="species", species=[], period_preset="all")
    parser, _ = make_parser(monkeypatch, payload)
    assert parser.parse("fiche espèce", today=TODAY).kind == "ranking"


def test_unknown_department_is_ignored(monkeypatch):
    payload = IntentModel(kind="ranking", department="Atlantide", period_preset="all")
    parser, _ = make_parser(monkeypatch, payload)
    assert parser.parse("classement en Atlantide", today=TODAY).department is None


def test_malformed_dates_are_discarded(monkeypatch):
    payload = IntentModel(
        kind="trend",
        species=["Milan royal"],
        start_date="hier",
        end_date="2026-13-45",
        period_preset="all",
    )
    parser, _ = make_parser(monkeypatch, payload)
    intent = parser.parse("évolution récente du milan royal", today=TODAY)
    assert intent.start_date is None
    assert intent.end_date is None


def test_off_topic_question_cannot_become_a_chart(monkeypatch):
    """Un modèle trop zélé ne doit pas transformer une question étrangère en analyse."""
    payload = IntentModel(kind="trend", species=[], period_preset="all")
    parser, _ = make_parser(monkeypatch, payload)
    intent = parser.parse("Quel temps fera-t-il demain ?", today=TODAY)
    assert intent.kind == "unknown"


def test_model_failure_falls_back_to_heuristic(monkeypatch):
    parser, _ = make_parser(monkeypatch, None, raise_error=True)
    intent = parser.parse("Top 10 des espèces en Gironde en 2025 ?", today=TODAY)
    assert intent.kind == "ranking"
    assert intent.department == "Gironde"
    assert "Heuristique" in intent.rationale


def test_llm_disabled_short_circuits_the_model(monkeypatch):
    parser, fake = make_parser(
        monkeypatch, IntentModel(kind="trend", period_preset="all"), enabled=False
    )
    intent = parser.parse("Évolution du Milan royal sur 10 ans ?", today=TODAY)
    assert intent.kind == "trend"  # heuristique
    assert fake.calls == [], "le modèle ne doit pas être appelé quand LPO_LLM_INTENT=off"


def test_missing_dependency_falls_back_without_raising(monkeypatch):
    """Sans dépendances IA installées, on retombe proprement sur l'heuristique."""
    import builtins

    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.startswith("src.ai.model_factory"):
            raise ImportError("dépendances IA absentes")
        return real_import(name, *args, **kwargs)

    monkeypatch.setenv("LPO_LLM_INTENT", "auto")
    monkeypatch.setattr(builtins, "__import__", guard)
    parser = IntentParser(species_vocab=SPECIES, departments=DEPARTMENTS, regions=REGIONS)
    intent = parser.parse("Fiche espèce du Héron cendré", today=TODAY)
    assert intent.kind == "species"


@pytest.mark.parametrize(
    "question,kind",
    [
        ("Évolution du Milan royal sur 10 ans ?", "trend"),
        ("Top 10 des espèces en Gironde en 2025 ?", "ranking"),
        ("Fiche espèce du Héron cendré", "species"),
    ],
)
def test_llm_and_heuristic_agree_on_clear_questions(monkeypatch, question, kind):
    """Sur les questions non ambiguës, les deux chemins convergent."""
    heuristic = IntentParser(
        species_vocab=SPECIES, departments=DEPARTMENTS, regions=REGIONS
    ).heuristic(question, today=TODAY)
    payload = IntentModel(
        kind=heuristic.kind,
        species=heuristic.species,
        start_date=heuristic.start_date,
        end_date=heuristic.end_date,
        period_label=heuristic.period_label,
        period_preset=heuristic.period_preset,
        department=heuristic.department,
        region=heuristic.region,
        granularity=heuristic.granularity,
    )
    parser, _ = make_parser(monkeypatch, payload)
    assert parser.parse(question, today=TODAY).kind == kind
