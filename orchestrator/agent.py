"""Agent orchestrateur : question → outils MCP → surface A2UI.

Rôle exact de l'agent (et limites volontaires) :

1. **Comprendre l'intention** (LLM si disponible, heuristique sinon) : espèce,
   période, zone, métrique, type de visualisation. Aucune donnée n'est produite
   à ce stade.
2. **Appeler les outils MCP** typés (`aggregate_observations`, `list_species`,
   `get_species_info`…) et conserver la trace des appels.
3. **Construire la vue** à partir des seuls résultats MCP (`views.py`) puis les
   **messages A2UI** via les gabarits (`lpo_a2ui/templates.py`).
4. **Valider** chaque message contre le catalogue et l'état du renderer avant
   envoi. En cas d'échec : repli texte, jamais de surface douteuse.
5. **Traiter les actions** utilisateur (filtres) en rappelant les outils MCP puis
   en n'envoyant qu'un `updateDataModel` sur la surface existante — la structure
   n'est pas recréée (sauf changement de type de vue).

L'agent **ne calcule pas** de biodiversité : il ne fait que mettre en forme des
valeurs renvoyées par le serveur MCP.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

from lpo_a2ui import templates
from lpo_a2ui.validation import (
    ValidationReport,
    describe_report,
    new_renderer_state,
    validate_messages,
)

from . import views as view_builder
from .intent import Intent, IntentParser, resolve_period
from .mcp_client import McpDataClient, McpToolError

logger = logging.getLogger("lpo.orchestrator")


@dataclass
class TraceEntry:
    """Trace d'un appel MCP (affichée dans l'UI et utile au diagnostic)."""

    tool: str
    arguments: dict[str, Any]
    ok: bool
    summary: str
    error: str | None = None
    duration_ms: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "ok": self.ok,
            "summary": self.summary,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class AssistantTurn:
    """Réponse complète : surface A2UI + texte de repli + trace."""

    text: str
    messages: list[dict] = field(default_factory=list)
    surface_id: str | None = None
    kind: str = "empty"
    mode: str = "create"  # create | update | none
    trace: list[TraceEntry] = field(default_factory=list)
    intent: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    degraded: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "messages": self.messages,
            "surfaceId": self.surface_id,
            "kind": self.kind,
            "mode": self.mode,
            "trace": [t.as_dict() for t in self.trace],
            "intent": self.intent,
            "validation": self.validation,
            "degraded": self.degraded,
        }


@dataclass
class Conversation:
    """État d'une conversation (une surface à la fois)."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    surface_id: str | None = None
    last_intent: Intent | None = None
    last_kind: str | None = None
    turn: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    #: Miroir du renderer côté agent : permet de valider les `updateDataModel`.
    renderer_state: Any = None

    def state(self) -> Any:
        if self.renderer_state is None:
            self.renderer_state = new_renderer_state()
        return self.renderer_state

    def next_surface_id(self) -> str:
        self.turn += 1
        return f"lpo-{self.id}-{self.turn}"


class LpoAssistant:
    """Orchestrateur données → interface."""

    def __init__(
        self,
        client: McpDataClient | None = None,
        *,
        today: date | None = None,
        use_llm_intent: bool | None = None,
        llm_factory: Callable[[], Any] | None = None,
    ):
        self.client = client or McpDataClient()
        self.today = today
        #: ``None`` = suivre `LPO_LLM_INTENT` ; imposé par les réglages d'exécution
        #: quand l'interface web a configuré le fournisseur de modèle.
        self.use_llm_intent = use_llm_intent
        self.llm_factory = llm_factory
        self._context: view_builder.ViewContext | None = None
        self._parser: IntentParser | None = None
        self._context_lock = asyncio.Lock()

    # ── Contexte (découverte du jeu de données) ──────────────────────────

    async def ensure_context(self) -> view_builder.ViewContext:
        """Charge (une fois) l'aperçu MCP, le vocabulaire d'espèces et les zones."""
        if self._context is not None:
            return self._context
        async with self._context_lock:
            if self._context is not None:
                return self._context
            overview = await self.client.call("get_dataset_overview")
            try:
                species = await self.client.call("list_species", {"limit": 200})
                labels = [str(s["label"]) for s in species.get("species") or []]
            except McpToolError:
                labels = [str(s) for s in overview.get("species_sample") or []]
            self._context = view_builder.ViewContext(
                overview=overview,
                species_options=labels,
                department_options=[str(d) for d in overview.get("departments") or []],
            )
            self._parser = IntentParser(
                species_vocab=labels,
                departments=[str(d) for d in overview.get("departments") or []],
                regions=[str(r) for r in overview.get("regions") or []],
                llm_factory=self.llm_factory,
                llm_mode=self.use_llm_intent,
            )
            return self._context

    def configure_llm(self, *, use_llm_intent: bool, llm_factory: Callable[[], Any] | None) -> None:
        """Change le fournisseur de modèle **sans** perdre le contexte MCP.

        Le vocabulaire (espèces, départements, régions) vient des outils : le
        relire à chaque changement de modèle serait un appel réseau inutile.
        """
        self.use_llm_intent = use_llm_intent
        self.llm_factory = llm_factory
        if self._parser is not None:
            self._parser = IntentParser(
                species_vocab=self._parser.species_vocab,
                departments=self._parser.departments,
                regions=self._parser.regions,
                llm_factory=llm_factory,
                llm_mode=use_llm_intent,
            )

    @property
    def parser(self) -> IntentParser:
        if self._parser is None:
            raise RuntimeError("Contexte non initialisé : appeler ensure_context().")
        return self._parser

    # ── Appels MCP ───────────────────────────────────────────────────────

    async def _call(
        self, tool: str, arguments: dict[str, Any], trace: list[TraceEntry]
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            payload = await self.client.call(tool, arguments)
        except McpToolError as exc:
            trace.append(
                TraceEntry(
                    tool=tool,
                    arguments=arguments,
                    ok=False,
                    summary="",
                    error=exc.message,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
            )
            raise
        trace.append(
            TraceEntry(
                tool=tool,
                arguments=arguments,
                ok=True,
                summary=str(payload.get("summary") or ""),
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        )
        return payload

    def _filters(self, intent: Intent) -> dict[str, Any]:
        filters: dict[str, Any] = {"limit": 200}
        if intent.species:
            filters["species"] = list(intent.species)
        if intent.start_date:
            filters["start_date"] = intent.start_date
        if intent.end_date:
            filters["end_date"] = intent.end_date
        if intent.department:
            filters["department"] = intent.department
        if intent.region:
            filters["region"] = intent.region
        return filters

    async def _fetch(self, intent: Intent, trace: list[TraceEntry]) -> dict[str, Any]:
        """Récupère les données nécessaires à l'intention, via MCP uniquement."""
        base = self._filters(intent)
        if intent.kind == "ranking":
            return {
                "agg": await self._call(
                    "aggregate_observations",
                    {
                        **base,
                        "group_by": intent.rank_by,
                        "metric": intent.metric,
                        "limit": 10,
                    },
                    trace,
                )
            }
        if intent.kind == "map":
            return {
                "agg": await self._call(
                    "aggregate_observations",
                    {**base, "group_by": "grid", "metric": intent.metric, "limit": 200},
                    trace,
                )
            }
        if intent.kind == "comparison":
            group = "species_month" if intent.granularity == "month" else "species_year"
            return {
                "agg": await self._call(
                    "aggregate_observations",
                    {**base, "group_by": group, "metric": intent.metric, "limit": 500},
                    trace,
                )
            }
        if intent.kind == "species" and len(intent.species) == 1:
            return {
                "detail": await self._call(
                    "get_species_info", {"species": intent.species[0]}, trace
                )
            }
        return {
            "agg": await self._call(
                "aggregate_observations",
                {**base, "group_by": intent.granularity, "metric": intent.metric, "limit": 120},
                trace,
            )
        }

    # ── Rendu d'une vue ──────────────────────────────────────────────────

    def _build_view(self, intent: Intent, payloads: dict[str, Any]) -> view_builder.BuiltView:
        ctx = self._context
        assert ctx is not None
        if intent.kind == "ranking":
            return view_builder.build_ranking(intent, payloads["agg"], ctx)
        if intent.kind == "map":
            return view_builder.build_map(intent, payloads["agg"], ctx)
        if intent.kind == "comparison":
            return view_builder.build_comparison(intent, payloads["agg"], ctx)
        if intent.kind == "species" and "detail" in payloads:
            return view_builder.build_species(intent, payloads["detail"], ctx)
        return view_builder.build_trend(intent, payloads["agg"], ctx)

    def _messages(
        self,
        conversation: Conversation,
        view: view_builder.BuiltView,
        *,
        replace_surface: bool,
    ) -> tuple[list[dict], str]:
        """Fabrique les messages A2UI et indique le mode utilisé."""
        surface_id = conversation.surface_id or conversation.next_surface_id()
        messages: list[dict] = []
        if replace_surface:
            if conversation.surface_id:
                messages.extend(templates.delete_messages(conversation.surface_id))
                surface_id = conversation.next_surface_id()
            messages.extend(
                templates.surface_messages(
                    view.kind,
                    surface_id,
                    view.data,
                    columns=templates.table_columns(
                        view.kind,
                        unit_label=str(view.data.get("unit") or "observations"),
                        dimension=view.dimension,
                    ),
                )
            )
            conversation.surface_id = surface_id
            conversation.last_kind = view.kind
            mode = "create"
        else:
            messages.extend(templates.update_messages(view.kind, surface_id, view.data))
            mode = "update"
        return messages, mode

    def _finalise(
        self,
        conversation: Conversation,
        view: view_builder.BuiltView,
        text: str,
        trace: list[TraceEntry],
        *,
        replace_surface: bool,
    ) -> AssistantTurn:
        """Valide les messages ; dégrade en texte seul si nécessaire."""
        messages, mode = self._messages(conversation, view, replace_surface=replace_surface)
        report: ValidationReport = validate_messages(messages, state=conversation.state())
        if not report.ok and mode == "update":
            # Une mise à jour refusée (surface perdue, incohérence) : on reconstruit
            # proprement la surface plutôt que de laisser l'utilisateur sans interface.
            logger.warning(
                "Mise à jour A2UI refusée, recréation de la surface : %s", report.errors[:2]
            )
            conversation.surface_id = None
            conversation.renderer_state = None
            messages, mode = self._messages(conversation, view, replace_surface=True)
            report = validate_messages(messages, state=conversation.state())
        conversation.renderer_state = report.state
        turn = AssistantTurn(
            text=text,
            messages=messages,
            surface_id=conversation.surface_id,
            kind=view.kind,
            mode=mode,
            trace=trace,
            intent=conversation.last_intent.model_dump_safe() if conversation.last_intent else {},
            validation={
                "ok": report.ok,
                "checked": report.checked_messages,
                "errors": report.errors,
            },
        )
        if not report.ok:
            logger.error("A2UI invalide — repli texte :\n%s", describe_report(report))
            turn.messages = []
            turn.mode = "none"
            turn.degraded = True
            turn.text = (
                f"{text}\n\n"
                "⚠️ L'interface graphique n'a pas pu être générée pour cette réponse "
                "(message A2UI invalide, rejeté avant envoi). Le résumé ci-dessus reste "
                "fiable : il provient des mêmes données."
            )
        return turn

    # ── Point d'entrée : question ────────────────────────────────────────

    async def ask(self, question: str, conversation: Conversation | None = None) -> AssistantTurn:
        conversation = conversation or Conversation()
        question = (question or "").strip()
        if not question:
            raise ValueError("Question vide.")
        if len(question) > 500:
            raise ValueError("Question trop longue (500 caractères maximum).")

        async with conversation.lock:
            trace: list[TraceEntry] = []
            try:
                ctx = await self.ensure_context()
            except McpToolError as exc:
                conversation.last_intent = None
                view = view_builder.BuiltView(
                    "empty",
                    {
                        "header": {"title": "Données indisponibles", "subtitle": exc.tool},
                        "notices": [
                            view_builder.notice(
                                "error",
                                "Serveur MCP injoignable",
                                f"L'outil `{exc.tool}` a échoué : {exc.message}",
                            )
                        ],
                        "footer": {"source": "Source : aucune — serveur MCP indisponible"},
                    },
                    (
                        "Le serveur de données MCP est indisponible : je ne peux pas répondre "
                        "sans inventer de chiffres. Vérifier la configuration de la base "
                        "(`DATABASE_URL`) puis relancer."
                    ),
                )
                turn = self._finalise(conversation, view, view.text, trace, replace_surface=True)
                return turn

            intent = self.parser.parse(question, today=self.today, use_llm=self.use_llm_intent)
            intent = self._refine(intent)
            conversation.last_intent = intent

            if intent.unknown_species and intent.kind != "unknown":
                # Aucun outil n'est appelé : on ne peut pas répondre sur une espèce
                # que le jeu de données ne contient pas, et les données des autres
                # espèces ne sont pas une approximation acceptable.
                view = view_builder.build_unknown_term(intent, ctx, intent.unknown_species)
                return self._finalise(conversation, view, view.text, trace, replace_surface=True)

            if intent.kind == "unknown":
                view = view_builder.build_unknown(
                    intent,
                    ctx,
                    reason=(
                        "Cette question ne porte pas sur les observations naturalistes exposées "
                        "par le serveur MCP. Je ne peux pas y répondre sans inventer de données. "
                        "Essayez par exemple : « évolution des observations de Milan royal depuis "
                        "2016 », « top 10 des espèces en Gironde en 2025 » ou « carte du Busard "
                        "cendré ce printemps »."
                    ),
                )
                return self._finalise(conversation, view, view.text, trace, replace_surface=True)

            try:
                payloads = await self._fetch(intent, trace)
            except McpToolError as exc:
                view = view_builder.BuiltView(
                    "empty",
                    {
                        "header": {
                            "title": f"Requête impossible — {intent.species_label}",
                            "subtitle": intent.area_label,
                        },
                        "notices": [
                            view_builder.notice(
                                "error", "L'outil de données a refusé la requête", exc.message
                            )
                        ],
                        "footer": {"source": ctx.source_line()},
                    },
                    (
                        f"Je ne peux pas répondre à cette question : l'outil MCP "
                        f"`{exc.tool}` a renvoyé une erreur ({exc.message}). "
                        "Reformuler avec une espèce, une période ou une zone reconnues."
                    ),
                )
                return self._finalise(conversation, view, view.text, trace, replace_surface=True)

            view = self._build_view(intent, payloads)
            return self._finalise(conversation, view, view.text, trace, replace_surface=True)

    def _refine(self, intent: Intent) -> Intent:
        """Ajustements déterministes avant l'appel MCP (granularité, cohérence)."""
        if (
            intent.kind in {"trend", "comparison"}
            and intent.start_date
            and intent.end_date
            and intent.start_date[:4] == intent.end_date[:4]
            and intent.granularity == "year"
        ):
            # Une seule année : le pas mensuel est plus informatif qu'un point unique.
            intent.granularity = "month"
        if intent.kind == "comparison" and len(intent.species) < 2:
            intent.kind = "trend"
            intent.granularity = "year"
        return intent

    # ── Point d'entrée : action utilisateur ──────────────────────────────

    async def act(
        self,
        action: dict[str, Any],
        conversation: Conversation,
    ) -> AssistantTurn:
        """Traite une action A2UI (filtre) et met à jour la surface existante."""
        async with conversation.lock:
            trace: list[TraceEntry] = []
            await self.ensure_context()
            intent = conversation.last_intent
            if intent is None:
                intent = self.parser.parse("top 10 des espèces observées", today=self.today)
                conversation.last_intent = intent

            name = str(action.get("name") or action.get("action") or "")
            context = action.get("context") or {}
            if name != "set_filter":
                logger.info("Action A2UI non gérée : %s", name)
                view = view_builder.build_unknown(
                    intent,
                    self._context,  # type: ignore[arg-type]
                    reason=f"L'action « {name} » n'est pas gérée par cet agent.",
                )
                return self._finalise(conversation, view, view.text, trace, replace_surface=True)

            new_intent = self._apply_filter(intent, context)
            conversation.last_intent = new_intent
            try:
                payloads = await self._fetch(new_intent, trace)
            except McpToolError as exc:
                view = view_builder.BuiltView(
                    "empty",
                    {
                        "header": {
                            "title": "Filtre impossible",
                            "subtitle": new_intent.area_label,
                        },
                        "notices": [
                            view_builder.notice(
                                "error", "L'outil de données a refusé la requête", exc.message
                            )
                        ],
                        "footer": {"source": self._context.source_line() if self._context else ""},
                    },
                    f"Le filtre n'a pas pu être appliqué : {exc.message}",
                )
                return self._finalise(conversation, view, view.text, trace, replace_surface=True)

            view = self._build_view(new_intent, payloads)
            same_kind = view.kind == conversation.last_kind and conversation.surface_id
            text = f"Filtre appliqué ({self._describe_filter(context)}). " + view.text
            return self._finalise(
                conversation, view, text, trace, replace_surface=not bool(same_kind)
            )

    # ── Filtres ──────────────────────────────────────────────────────────

    def _apply_filter(self, intent: Intent, context: dict[str, Any]) -> Intent:
        filter_name = str(context.get("filter") or "")
        value = str(context.get("value") or "")
        updated = Intent(**intent.model_dump_safe())
        updated.filters = dict(intent.filters)
        updated.filters[filter_name] = value

        if filter_name == "species":
            updated.species = [value] if value else []
        elif filter_name == "period":
            if value:
                start, end, label = resolve_period(value, self.today)
                updated.start_date, updated.end_date = start, end
                updated.period_label = label
            else:
                updated.start_date = updated.end_date = None
                updated.period_label = "toute la période disponible"
            updated.period_preset = value or "all"
            if value in {"12m", "spring"}:
                updated.granularity = "month"
        elif filter_name == "area":
            departments = self._context.department_options if self._context else []
            regions = (
                [str(r) for r in (self._context.overview.get("regions") or [])]
                if self._context
                else []
            )
            updated.department = value if value in departments else None
            updated.region = value if value in regions and value not in departments else None
        # Une comparaison n'a de sens qu'avec deux espèces.
        if updated.kind == "comparison" and len(updated.species) < 2:
            updated.kind = "trend"
            updated.granularity = "year"
        return updated

    def _describe_filter(self, context: dict[str, Any]) -> str:
        names = {"species": "espèce", "period": "période", "area": "zone"}
        filter_name = str(context.get("filter") or "?")
        value = str(context.get("value") or "")
        if filter_name == "period" and value:
            _start, _end, label = resolve_period(value, self.today)
            value = label
        return f"{names.get(filter_name, filter_name)} = {value or 'toutes'}"
