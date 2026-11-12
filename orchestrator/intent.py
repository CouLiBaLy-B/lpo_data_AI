"""Compréhension de l'intention utilisateur.

Deux implémentations, dans cet ordre :

1. **LLM** (`ChatModel.with_structured_output(Intent)`) via la fabrique de modèles
   existante du dépôt (`src.ai.model_factory`) — provider-agnostique. Le modèle
   ne produit **que** l'intention (espèce, période, zone, métrique, type de
   visualisation) : aucun chiffre, aucune donnée.
2. **Heuristique déterministe** (repli) : mots-clés, vocabulaire d'espèces
   fourni par le serveur MCP et départements connus. C'est ce qui tourne sans
   clé API (démo, tests, CI).

Le type de visualisation est choisi ici, à partir de l'intention — jamais par un
texte libre du modèle — puis les gabarits A2UI correspondants sont appliqués.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

Kind = Literal["trend", "ranking", "map", "comparison", "species", "unknown"]
Preset = Literal["10y", "12m", "this_year", "last_year", "spring", "all", "custom"]
Granularity = Literal["year", "month"]
Metric = Literal["observations", "individuals"]


class IntentModel(BaseModel):
    """Schéma de sortie structurée du LLM (et format interne)."""

    kind: Kind = Field(description="Type de réponse attendue.")
    species: list[str] = Field(default_factory=list, description="Libellés exacts d'espèces.")
    start_date: str | None = Field(default=None, description="Date ISO de début (AAAA-MM-JJ).")
    end_date: str | None = Field(default=None, description="Date ISO de fin (AAAA-MM-JJ).")
    period_label: str = Field(default="", description="Période en clair pour l'affichage.")
    period_preset: Preset = "all"
    department: str | None = None
    region: str | None = None
    metric: Metric = "observations"
    granularity: Granularity = "year"
    rationale: str = Field(default="", description="Une phrase justifiant le choix.")


@dataclass
class Intent:
    """Intention enrichie (dataclass exploitable par l'orchestrateur)."""

    kind: Kind = "trend"
    species: list[str] = field(default_factory=list)
    start_date: str | None = None
    end_date: str | None = None
    period_label: str = ""
    period_preset: str = "all"
    department: str | None = None
    region: str | None = None
    metric: Metric = "observations"
    granularity: Granularity = "year"
    rationale: str = ""
    raw_question: str = ""
    filters: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_model(cls, model: IntentModel, *, question: str = "", **overrides: Any) -> Intent:
        data = model.model_dump()
        data.update(overrides)
        return cls(raw_question=question, **data)

    @property
    def species_label(self) -> str:
        return " et ".join(self.species) if self.species else "toutes espèces"

    @property
    def area_label(self) -> str:
        return self.department or self.region or "France entière"

    @property
    def area_phrase(self) -> str:
        """Formulation en prose, neutre grammaticalement.

        Les noms de départements imposent des articles variables (« en Gironde »,
        « dans le Finistère », « dans les Landes ») : on évite l'accord en
        nommant la zone comme une zone.
        """
        if self.department:
            return f"sur la zone « {self.department} »"
        if self.region:
            return f"sur la région « {self.region} »"
        return "sur toute la France"

    def model_dump_safe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "species": list(self.species),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "period_label": self.period_label,
            "period_preset": self.period_preset,
            "department": self.department,
            "region": self.region,
            "metric": self.metric,
            "granularity": self.granularity,
            "rationale": self.rationale,
            "raw_question": self.raw_question,
            "filters": dict(self.filters),
        }


# ── Presets de période ───────────────────────────────────────────────────────

PERIOD_PRESETS: dict[str, str] = {
    "10y": "10 dernières années",
    "12m": "12 derniers mois",
    "this_year": "Cette année",
    "last_year": "Année dernière",
    "spring": "Ce printemps",
    "all": "Toute la période",
}


def resolve_period(preset: str, today: date | None = None) -> tuple[str | None, str | None, str]:
    """Traduit un preset en bornes ISO + libellé."""
    today = today or date.today()
    if preset == "10y":
        start = date(today.year - 9, 1, 1)
        return start.isoformat(), today.isoformat(), f"{start.year}–{today.year}"
    if preset == "12m":
        start = today - timedelta(days=365)
        return start.isoformat(), today.isoformat(), "12 derniers mois"
    if preset == "this_year":
        start = date(today.year, 1, 1)
        return start.isoformat(), today.isoformat(), f"année {today.year}"
    if preset == "last_year":
        year = today.year - 1
        return date(year, 1, 1).isoformat(), date(year, 12, 31).isoformat(), f"année {year}"
    if preset == "spring":
        start = date(today.year, 3, 1)
        end = date(today.year, 6, 30)
        return start.isoformat(), end.isoformat(), f"printemps {today.year}"
    return None, None, "toute la période disponible"


# ── Normalisation / vocabulaire ──────────────────────────────────────────────


def normalize(text: str) -> str:
    """Minuscules, sans accents, espaces compactés."""
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_species(question: str, vocabulary: list[str]) -> list[str]:
    """Retrouve les espèces citées, dans l'ordre où elles apparaissent.

    On teste les libellés du plus long au plus court pour éviter qu'un libellé
    court ne masque un libellé long (« Milan » vs « Milan royal »), puis on
    rétablit l'ordre du texte : les séries d'un graphique de comparaison et sa
    légende suivent ainsi la formulation de l'utilisateur.
    """
    normalized_question = normalize(question)
    positions: dict[str, int] = {}
    for label in sorted(vocabulary, key=len, reverse=True):
        key = normalize(label)
        if not key or label in positions:
            continue
        match = re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", normalized_question)
        if match:
            positions[label] = match.start()
    if positions:
        return sorted(positions, key=lambda label: positions[label])

    # Repli : mot distinctif unique (ex. « milan » → Milan royal)
    found: list[str] = []
    tokens = [t for t in normalized_question.split() if len(t) >= 4]
    for label in vocabulary:
        key_tokens = normalize(label).split()
        for token in tokens:
            matches = [candidate for candidate in vocabulary if token in normalize(candidate)]
            if len(matches) == 1 and matches[0] == label and token in key_tokens:
                if label not in found:
                    found.append(label)
    return found


_MONTHS = {
    "janvier": 1,
    "fevrier": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "aout": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "decembre": 12,
}


def extract_period(text: str, today: date | None = None) -> tuple[str | None, str | None, str, str]:
    """Renvoie (start, end, libellé, preset) à partir du texte de la question."""
    today = today or date.today()
    q = normalize(text)

    if re.search(r"printemps", q):
        start, end, label = resolve_period("spring", today)
        return start, end, label, "spring"
    if re.search(r"10 derniere|dix derniere|decennie", q):
        start, end, label = resolve_period("10y", today)
        return start, end, label, "10y"
    if re.search(r"12 derniers mois|douze derniers mois|un an|derniere annee ecoulee", q):
        start, end, label = resolve_period("12m", today)
        return start, end, label, "12m"
    if re.search(r"cette annee|annee en cours", q):
        start, end, label = resolve_period("this_year", today)
        return start, end, label, "this_year"
    if re.search(r"annee derniere|an dernier", q):
        start, end, label = resolve_period("last_year", today)
        return start, end, label, "last_year"

    span = re.search(r"(?:entre|de)\s+((?:19|20)\d{2})\s+(?:et|a|-)\s+((?:19|20)\d{2})", q)
    if span:
        y1, y2 = sorted((int(span.group(1)), int(span.group(2))))
        return date(y1, 1, 1).isoformat(), date(y2, 12, 31).isoformat(), f"{y1}–{y2}", "custom"

    month_match = re.search(r"\b(" + "|".join(_MONTHS) + r")\s+((?:19|20)\d{2})", q)
    if month_match:
        month = _MONTHS[month_match.group(1)]
        year = int(month_match.group(2))
        last_day = (date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)).day
        return (
            date(year, month, 1).isoformat(),
            date(year, month, last_day).isoformat(),
            f"{month_match.group(1)} {year}",
            "custom",
        )

    years = re.findall(r"\b((?:19|20)\d{2})\b", q)
    if years:
        year = int(years[-1])
        return (
            date(year, 1, 1).isoformat(),
            date(year, 12, 31).isoformat(),
            f"année {year}",
            "custom",
        )

    start, end, label = resolve_period("all", today)
    return start, end, label, "all"


def extract_area(
    text: str, departments: list[str], regions: list[str]
) -> tuple[str | None, str | None]:
    """Détecte un département ou une région nommés explicitement."""
    q = normalize(text)
    for dept in sorted(departments, key=len, reverse=True):
        if normalize(dept) and re.search(rf"(?<![a-z]){re.escape(normalize(dept))}(?![a-z])", q):
            return dept, None
    for region in sorted(regions, key=len, reverse=True):
        if normalize(region) and re.search(
            rf"(?<![a-z]){re.escape(normalize(region))}(?![a-z])", q
        ):
            return None, region
    return None, None


def extract_metric(text: str) -> str:
    q = normalize(text)
    if re.search(r"individu|effectif|nombre d'animaux|compter les animaux", q):
        return "individuals"
    return "observations"


_DOMAIN_SIGNAL = re.compile(
    r"espece|observation|biodivers|naturalis|faune|flore|oiseau|mammifere|amphibien|"
    r"protection|sensible|zone|niche|migration|effectif|individu|departement|region|"
    r"carte|top|classement|compar|evolution|tendance|fiche"
)


def has_domain_signal(
    text: str, species: list[str], area: tuple[str | None, str | None], preset: str
) -> bool:
    """Vrai si la question porte plausiblement sur le jeu de données exposé."""
    if species:
        return True
    if area[0] or area[1]:
        return True
    if preset != "all":
        return True
    return bool(_DOMAIN_SIGNAL.search(normalize(text)))


def guess_kind(text: str, species: list[str], wants_compare: bool) -> str:
    """Choisit la visualisation adaptée à l'intention exprimée."""
    q = normalize(text)
    if (
        wants_compare
        or re.search(r"\bcompar|\bversus\b|\bvs\b|par mois pour", q)
        and len(species) >= 2
    ):
        return "comparison"
    if re.search(r"\bcarte\b|localis|geograph|ou (?:a-t-on|sont)|repartition spatiale|points?", q):
        return "map"
    if re.search(r"\btop\b|classement|les plus|palmares|quelles especes|principales especes", q):
        return "ranking"
    if re.search(r"evolution|tendance|depuis|par an|par annee|progression|augment", q):
        return "trend"
    if len(species) == 1 and re.search(r"fiche|informations|statut|protection|description", q):
        return "species"
    if len(species) == 1:
        return "trend"
    return "ranking"


# ── Parseur ──────────────────────────────────────────────────────────────────


class IntentParser:
    """Extrait l'intention : LLM si disponible, heuristique sinon."""

    def __init__(self, *, species_vocab: list[str], departments: list[str], regions: list[str]):
        self.species_vocab = species_vocab
        self.departments = departments
        self.regions = regions

    # — heuristique
    def heuristic(self, question: str, *, today: date | None = None) -> Intent:
        species = extract_species(question, self.species_vocab)
        start, end, label, preset = extract_period(question, today)
        department, region = extract_area(question, self.departments, self.regions)
        metric = extract_metric(question)
        compare = (
            bool(re.search(r"compar|versus|\bvs\b|entre", normalize(question)))
            and len(species) >= 2
        )
        kind = guess_kind(question, species, compare)
        if kind in {"trend", "ranking"} and not has_domain_signal(
            question, species, (department, region), preset
        ):
            kind = "unknown"
        granularity: Granularity = (
            "month"
            if kind in {"comparison"} or re.search(r"par mois|mensuel", normalize(question))
            else "year"
        )
        if kind == "comparison" and len(species) < 2:
            kind = "trend"
        return Intent(
            kind=kind,  # type: ignore[arg-type]
            species=species[:3],
            start_date=start,
            end_date=end,
            period_label=label,
            period_preset=preset,  # type: ignore[arg-type]
            department=department,
            region=region,
            metric=metric,  # type: ignore[arg-type]
            granularity=granularity,
            rationale="Heuristique déterministe (mots-clés + vocabulaire MCP).",
            raw_question=question,
        )

    # — LLM (optionnel)
    def llm(self, question: str, *, today: date | None = None) -> Intent | None:
        if os.getenv("LPO_LLM_INTENT", "auto").strip().lower() in {"0", "off", "false"}:
            return None
        try:
            from src.ai.model_factory import create_model
        except Exception as exc:  # noqa: BLE001 - dépendances IA optionnelles
            logger.info("Intention LLM indisponible (import): %s", exc)
            return None

        today = today or date.today()
        prompt = (
            "Tu extrais l'intention d'une question sur des observations naturalistes LPO.\n"
            f"Date du jour : {today.isoformat()}.\n"
            f"Espèces connues (utiliser EXACTEMENT ces libellés) : {self.species_vocab[:200]}\n"
            f"Départements connus : {self.departments}\n"
            f"Régions connues : {self.regions}\n"
            "Choisis `kind` : trend (évolution temporelle), ranking (top/classement), "
            "map (carte/localisation), comparison (comparer 2+ espèces), species (fiche).\n"
            "Ne produis aucun chiffre issu des données : seulement l'intention."
        )
        try:
            model = create_model(temperature=0.0)
            structured = model.with_structured_output(IntentModel)
            result = structured.invoke([("system", prompt), ("human", question)])
            payload = result if isinstance(result, IntentModel) else IntentModel(**result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Extraction LLM de l'intention impossible (%s) — heuristique.", exc)
            return None

        # Garde-fous : libellés d'espèces obligatoirement présents dans le vocabulaire
        species = [s for s in payload.species if s in self.species_vocab][:3]
        start, end = payload.start_date, payload.end_date
        if not start and payload.period_preset:
            start, end, label = resolve_period(payload.period_preset, today)
            if not payload.period_label:
                payload.period_label = label
        if payload.department and payload.department not in self.departments:
            payload.department = None
        if payload.region and payload.region not in self.regions:
            payload.region = None
        if payload.kind == "comparison" and len(species) < 2:
            payload.kind = "trend"
        if payload.kind == "species" and len(species) != 1:
            payload.kind = "trend" if species else "ranking"
        return Intent.from_model(
            payload,
            question=question,
            species=species,
            start_date=start,
            end_date=end,
            rationale=payload.rationale or "Intention extraite par le LLM.",
        )

    def parse(
        self, question: str, *, today: date | None = None, use_llm: bool | None = None
    ) -> Intent:
        if use_llm is None:
            use_llm = os.getenv("LPO_LLM_INTENT", "auto").strip().lower() not in {
                "0",
                "off",
                "false",
            }
        if use_llm:
            intent = self.llm(question, today=today)
            if intent is not None:
                return intent
        return self.heuristic(question, today=today)
