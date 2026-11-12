"""Construction des vues : résultats MCP → data model A2UI + résumé texte.

Ce module est **le seul endroit** où des nombres sont mis en forme. Ils
proviennent exclusivement des objets renvoyés par le serveur MCP ; aucune valeur
n'est calculée à partir d'autre chose que ces résultats (pas d'estimation, pas de
comblement de trous). Si une donnée manque, on produit un ``Notice`` explicite.

Chaque vue renvoie un data model conforme au contrat de
`lpo_a2ui/templates.py` (mêmes chemins `/header`, `/kpis`, `/chart`, `/table`,
`/notices`, `/filters`, `/footer`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from orchestrator.intent import PERIOD_PRESETS, Intent


@dataclass
class BuiltView:
    """Surface prête à être transformée en messages A2UI."""

    kind: str
    data: dict
    text: str
    notes: list[str] = field(default_factory=list)


# ── Formatage ────────────────────────────────────────────────────────────────


def fmt_int(value: Any) -> str:
    try:
        return f"{int(round(float(value))):,}".replace(",", "\u202f")
    except (TypeError, ValueError):
        return "—"


def fmt_num(value: Any, decimals: int = 1) -> str:
    try:
        return f"{float(value):,.{decimals}f}".replace(",", "\u202f").replace(".", ",")
    except (TypeError, ValueError):
        return "—"


def fmt_pct(ratio: Any, decimals: int = 0) -> str:
    try:
        return f"{float(ratio) * 100:.{decimals}f} %".replace(".", ",")
    except (TypeError, ValueError):
        return "—"


def fmt_date(value: str | None) -> str:
    if not value:
        return "—"
    return str(value)[:10]


def _variation(first: float, last: float) -> tuple[str, str]:
    """Variation relative + tonalité associée (positive/négative)."""
    if not first:
        return "—", "neutral"
    delta = (last - first) / abs(first)
    tone = "positive" if delta > 0.02 else ("warning" if delta < -0.02 else "neutral")
    sign = "+" if delta > 0 else ""
    return f"{sign}{fmt_pct(delta)}", tone


def notice(tone: str, title: str, message: str) -> dict:
    return {"tone": tone, "title": title, "message": message}


# ── Contexte commun ──────────────────────────────────────────────────────────


@dataclass
class ViewContext:
    """Éléments partagés par toutes les vues (provenance, filtres, limites)."""

    overview: dict
    species_options: list[str] = field(default_factory=list)
    department_options: list[str] = field(default_factory=list)

    @property
    def profile(self) -> str:
        return str(self.overview.get("profile") or "?")

    @property
    def records_unit(self) -> str:
        """Ce que compte une ligne dans ce jeu de données (au pluriel)."""
        return str(self.overview.get("records_label") or "observations")

    @property
    def notes(self) -> list[str]:
        return [str(n) for n in self.overview.get("notes") or []]

    def dataset_notices(self) -> list[dict]:
        notices: list[dict] = []
        if self.overview.get("is_demo"):
            notices.append(
                notice(
                    "warning",
                    "Jeu de démonstration",
                    str(self.overview.get("dataset_label") or "Données synthétiques")
                    + " — aucune valeur écologique réelle. Les chiffres servent à valider "
                    "la chaîne MCP → A2UI, pas à documenter la biodiversité.",
                )
            )
        for note in self.notes:
            notices.append(notice("info", "Limite du jeu de données", note))
        if not self.overview.get("capabilities"):
            notices.append(
                notice("error", "Aucune capacité détectée", "Le schéma exposé est vide.")
            )
        return notices

    def warnings(self, *payloads: dict) -> list[dict]:
        seen: set[str] = set()
        out: list[dict] = []
        for payload in payloads:
            if (payload.get("meta") or {}).get("truncated"):
                out.append(
                    notice(
                        "warning",
                        "Résultat tronqué",
                        "Le serveur MCP a atteint sa limite de lignes : la vue montre un "
                        "sous-ensemble. Affiner la période ou la zone pour voir l'ensemble.",
                    )
                )
            for warning in (payload.get("meta") or {}).get("warnings") or []:
                text = str(warning)
                if text in seen:
                    continue
                seen.add(text)
                tone = "warning" if "absent" in text or "impossible" in text else "info"
                out.append(notice(tone, "À savoir", text))
        return out

    def source_line(self, *payloads: dict, privacy: str | None = None) -> str:
        metas = [p.get("meta") or {} for p in payloads if p]
        generated = next((m.get("generated_at") for m in metas if m.get("generated_at")), "")
        parts = [
            "Source : serveur MCP lpo-data (lecture seule)",
            f"profil {self.profile}",
            f"table {metas[0].get('table') if metas else '?'}",
        ]
        if privacy and privacy not in {"", "auto"}:
            parts.append(f"confidentialité : {privacy}")
        parts.append(f"calculé le {generated}" if generated else "calculé à la demande")
        return " · ".join(parts)

    def filters(self, intent: Intent) -> dict:
        species_value = intent.species[0] if len(intent.species) == 1 else ""
        area_value = intent.department or intent.region or ""
        species_opts = [{"label": "Toutes les espèces", "value": ""}]
        species_opts += [{"label": s, "value": s} for s in sorted(self.species_options)]
        area_opts = [{"label": "Toute la France", "value": ""}]
        area_opts += [{"label": d, "value": d} for d in sorted(self.department_options)]
        return {
            "species": {"value": species_value, "options": species_opts},
            "period": {
                "value": intent.period_preset,
                "options": [
                    {"label": label, "value": key} for key, label in PERIOD_PRESETS.items()
                ],
            },
            "area": {"value": area_value, "options": area_opts},
        }


def _header(intent: Intent, title: str, *, extra: str | None = None) -> dict:
    subtitle = f"{intent.period_label or 'toute la période'} · {intent.area_label}"
    if extra:
        subtitle += f" · {extra}"
    return {"title": title, "subtitle": subtitle}


def _points_to_x(points: Iterable[dict]) -> list[dict]:
    return [
        {"x": str(p.get("label") or p.get("key") or ""), "y": float(p.get("value") or 0)}
        for p in points
    ]


def _delta_rows(points: list[dict]) -> list[dict]:
    rows: list[dict] = []
    previous: float | None = None
    for point in points:
        value = float(point.get("value") or 0)
        delta = None if previous in (None, 0) else (value - previous) / abs(previous)
        rows.append(
            {
                "period": str(point.get("label") or point.get("key") or ""),
                "value": value,
                "delta": delta,
            }
        )
        previous = value
    return rows


# ── Tendance ─────────────────────────────────────────────────────────────────


def build_trend(
    intent: Intent,
    agg: dict,
    ctx: ViewContext,
    *,
    extra_series: list[dict] | None = None,
) -> BuiltView:
    points = list(agg.get("points") or [])
    privacy = (agg.get("meta") or {}).get("privacy") or "auto"
    base_notices = ctx.dataset_notices() + ctx.warnings(agg)

    if not points:
        return _empty_view(
            intent,
            ctx,
            title=f"Évolution — {intent.species_label}",
            reason=(
                "Aucune observation ne correspond à cette question sur la période et la zone "
                "demandées. Aucun graphique n'est affiché plutôt qu'une courbe vide."
            ),
            extra_notices=base_notices,
        )

    metric = str(agg.get("metric") or intent.metric)
    unit = "individus" if metric == "individuals" else ctx.records_unit
    # Accord au pluriel : « observations cumulées » mais « individus cumulés ».
    cumul = "Individus cumulés" if metric == "individuals" else "Observations cumulées"
    series = extra_series or [{"label": intent.species_label, "points": _points_to_x(points)}]
    sorted_points = sorted(points, key=lambda p: str(p.get("label") or p.get("key")))
    first, last = sorted_points[0], sorted_points[-1]
    trend_label, trend_tone = _variation(
        float(first.get("value") or 0), float(last.get("value") or 0)
    )
    peak = max(sorted_points, key=lambda p: float(p.get("value") or 0))

    kpis = [
        {
            "label": cumul,
            "value": fmt_int(agg.get("total")),
            "hint": intent.period_label or "toute la période",
            "tone": "neutral",
        },
        {
            "label": "Pic",
            "value": str(peak.get("label") or peak.get("key")),
            "hint": f"{fmt_int(peak.get('value'))} {unit}",
            "tone": "neutral",
        },
    ]
    if len(sorted_points) > 1:
        kpis.append(
            {
                "label": f"Évolution {first.get('label')} → {last.get('label')}",
                "value": trend_label,
                "hint": f"{fmt_int(first.get('value'))} → {fmt_int(last.get('value'))}",
                "tone": trend_tone,
            }
        )
    else:
        kpis.append(
            {
                "label": "Période couverte",
                "value": str(first.get("label") or first.get("key")),
                "hint": "une seule période dans les données",
                "tone": "neutral",
            }
        )
    if agg.get("total_observations") and metric == "individuals":
        kpis.append(
            {
                "label": "Observations",
                "value": fmt_int(agg.get("total_observations")),
                "hint": "lignes brutes",
                "tone": "neutral",
            }
        )

    text = (
        f"{intent.species_label} : {fmt_int(agg.get('total'))} {unit} au total "
        f"({intent.period_label or 'toute la période'}, {intent.area_label}). "
        f"Maximum en {peak.get('label')} avec {fmt_int(peak.get('value'))} {unit}."
    )
    if len(sorted_points) > 1:
        text += f" Évolution {first.get('label')} → {last.get('label')} : {trend_label}."
    if any("flout" in str(n.get("message", "")).lower() for n in base_notices):
        text += " Localisation des espèces sensibles non diffusée."

    period_unit = "mois" if intent.granularity == "month" else "année"
    data = {
        "header": _header(intent, f"Évolution des {unit} — {intent.species_label}"),
        "kpis": kpis,
        "chart": {
            "title": f"{unit.capitalize()} par {period_unit}",
            "series": series,
            "xLabel": "Mois" if intent.granularity == "month" else "Année",
            "yLabel": unit.capitalize(),
            "emptyMessage": "Aucune donnée sur la période.",
        },
        "table": {
            "title": "Détail par période",
            "caption": f"{unit.capitalize()} et variation par rapport à la période précédente.",
            "rows": _delta_rows(sorted_points),
            "emptyMessage": "Aucune période à afficher.",
        },
        "notices": base_notices,
        "filters": ctx.filters(intent),
        "footer": {"source": ctx.source_line(agg, privacy=privacy)},
    }
    return BuiltView("trend", data, text)


# ── Classement ───────────────────────────────────────────────────────────────


def build_ranking(intent: Intent, agg: dict, ctx: ViewContext) -> BuiltView:
    points = sorted(
        (p for p in (agg.get("points") or [])),
        key=lambda p: float(p.get("value") or 0),
        reverse=True,
    )[:10]
    base_notices = ctx.dataset_notices() + ctx.warnings(agg)

    if not points:
        return _empty_view(
            intent,
            ctx,
            title=f"Classement des espèces — {intent.area_label}",
            reason=(
                "Aucune espèce observée ne correspond aux filtres. Vérifier la période, "
                "la zone ou le libellé d'espèce."
            ),
            extra_notices=base_notices,
        )

    metric = str(agg.get("metric") or "observations")
    unit = "individus" if metric == "individuals" else ctx.records_unit
    items = [
        {
            "label": str(p.get("label") or p.get("key")),
            "value": float(p.get("value") or 0),
            "hint": fmt_pct(p.get("share") or 0, 1),
        }
        for p in points
    ]
    rows = [
        {
            "rank": index + 1,
            "label": str(p.get("label") or p.get("key")),
            "value": float(p.get("value") or 0),
            "share": p.get("share") or 0,
        }
        for index, p in enumerate(points)
    ]
    leader = points[0]
    kpis = [
        {
            "label": f"{unit.capitalize()} (toutes espèces)",
            "value": fmt_int(agg.get("total")),
            "hint": intent.period_label or "toute la période",
            "tone": "neutral",
        },
        {
            "label": "Espèce en tête",
            "value": str(leader.get("label")),
            "hint": f"{fmt_int(leader.get('value'))} {unit}",
            "tone": "neutral",
        },
        {
            "label": "Espèces classées",
            "value": fmt_int(len(points)),
            "hint": (
                f"sur {fmt_int((agg.get('meta') or {}).get('row_count', len(points)))} renvoyées"
            ),
            "tone": "neutral",
        },
    ]
    text = (
        f"Sur la zone « {intent.area_label} » ({intent.period_label or 'toute la période'}), "
        f"{leader.get('label')} arrive en tête avec {fmt_int(leader.get('value'))} {unit} "
        f"({fmt_pct(leader.get('share') or 0, 1)} du total)"
    )
    runners_up = [str(p.get("label")) for p in points[1:3]]
    if runners_up:
        text += ", devant " + ", ".join(runners_up)
    text += f". Total : {fmt_int(agg.get('total'))} {unit}."

    data = {
        "header": _header(intent, f"Top {len(points)} des espèces — {intent.area_label}"),
        "kpis": kpis,
        "chart": {
            "title": f"Classement par {unit}",
            "items": items,
            "valueLabel": unit.capitalize(),
            "emptyMessage": "Aucune espèce à classer.",
        },
        "table": {
            "title": "Détail du classement",
            "caption": "Part calculée sur le total du périmètre filtré.",
            "rows": rows,
            "emptyMessage": "Aucune ligne.",
        },
        "notices": base_notices,
        "filters": ctx.filters(intent),
        "footer": {"source": ctx.source_line(agg)},
    }
    return BuiltView("ranking", data, text)


# ── Carte ────────────────────────────────────────────────────────────────────


def build_map(intent: Intent, agg: dict, ctx: ViewContext) -> BuiltView:
    points = list(agg.get("points") or [])
    privacy = str((agg.get("meta") or {}).get("privacy") or "auto")
    base_notices = ctx.dataset_notices() + ctx.warnings(agg)

    if not points:
        capabilities = ctx.overview.get("capabilities") or []
        reason = (
            "Aucune coordonnée exploitable n'est disponible pour cette question : le schéma "
            "exposé ne contient pas de géolocalisation."
            if "coordinates" not in capabilities
            else "Aucune observation géolocalisée ne correspond aux filtres."
        )
        return _empty_view(
            intent,
            ctx,
            title=f"Carte — {intent.species_label}",
            reason=reason,
            extra_notices=base_notices,
        )

    map_points = [
        {
            "lat": float(p.get("lat")),
            "lon": float(p.get("lon")),
            "label": str(p.get("label") or p.get("key")),
            "value": float(p.get("value") or 0),
            "sensitive": bool(p.get("sensitive")),
        }
        for p in points
        if p.get("lat") is not None and p.get("lon") is not None
    ]
    if not map_points:
        return _empty_view(
            intent,
            ctx,
            title=f"Carte — {intent.species_label}",
            reason="Les points renvoyés ne contiennent pas de coordonnées diffusables.",
            extra_notices=base_notices,
        )

    total = sum(p["value"] for p in map_points)
    sensitive = sum(1 for p in map_points if p["sensitive"])
    max_value = max(p["value"] for p in map_points)
    legend = [
        {"label": "Maille à faible densité", "value": "1", "tone": "neutral"},
        {"label": "Densité moyenne", "value": fmt_int(round(max_value / 2)), "tone": "neutral"},
        {"label": "Densité maximale", "value": fmt_int(max_value), "tone": "warning"},
    ]
    if privacy in {"blurred", "department"}:
        base_notices.append(
            notice(
                "warning",
                "Localisation protégée",
                "Les coordonnées sont agrégées et arrondies par le serveur MCP (espèces "
                "sensibles). Aucun point exact n'est transmis au client.",
            )
        )

    kpis = [
        {
            "label": "Mailles occupées",
            "value": fmt_int(len(map_points)),
            "hint": intent.period_label or "toute la période",
            "tone": "neutral",
        },
        {
            "label": "Observations cartographiées",
            "value": fmt_int(total),
            "hint": "cumul des mailles",
            "tone": "neutral",
        },
        {
            "label": "Mailles sensibles",
            "value": fmt_int(sensitive),
            "hint": "espèces protégées ou sensibles",
            "tone": "warning" if sensitive else "neutral",
        },
    ]
    text = (
        f"{fmt_int(total)} observations géolocalisées réparties sur {fmt_int(len(map_points))} "
        f"mailles pour {intent.species_label} ({intent.period_label or 'toute la période'}, "
        f"{intent.area_label}). "
        + (
            f"{fmt_int(sensitive)} maille(s) contiennent une espèce sensible : la précision "
            "spatiale est volontairement dégradée."
            if sensitive
            else "Aucune espèce sensible dans ce résultat."
        )
    )

    data = {
        "header": _header(intent, f"Carte des observations — {intent.species_label}"),
        "kpis": kpis,
        "chart": {
            "title": "Densité d'observations par maille",
            "points": map_points,
            "legend": legend,
            "privacyMode": privacy if privacy in {"exact", "blurred", "department"} else "blurred",
            "emptyMessage": "Aucun point géolocalisé.",
        },
        "table": {
            "title": "Mailles les plus denses",
            "caption": "Coordonnées agrégées côté serveur.",
            "rows": [
                {
                    "label": p["label"],
                    "value": p["value"],
                    "precision": "floutée" if p["sensitive"] else "agrégée",
                }
                for p in map_points[:12]
            ],
            "emptyMessage": "Aucune maille.",
        },
        "notices": base_notices,
        "filters": ctx.filters(intent),
        "footer": {"source": ctx.source_line(agg, privacy=privacy)},
    }
    return BuiltView("map", data, text)


# ── Comparaison ──────────────────────────────────────────────────────────────


def build_comparison(intent: Intent, agg: dict, ctx: ViewContext) -> BuiltView:
    points = list(agg.get("points") or [])
    names = list(intent.species)
    base_notices = ctx.dataset_notices() + ctx.warnings(agg)

    if not points or len(names) < 2:
        return _empty_view(
            intent,
            ctx,
            title="Comparaison d'espèces",
            reason=(
                "La comparaison nécessite deux espèces explicitement nommées et des données "
                "sur la période demandée."
            ),
            extra_notices=base_notices,
        )

    by_series: dict[str, list[dict]] = {name: [] for name in names}
    for point in points:
        series_name = str(point.get("series") or "")
        if series_name not in by_series:
            by_series[series_name] = []
        by_series[series_name].append(point)

    palette = ["#1f4e79", "#2e7d32", "#b26a00", "#7b1fa2"]
    series = [
        {
            "label": name,
            "color": palette[index % len(palette)],
            "points": _points_to_x(
                sorted(by_series.get(name, []), key=lambda p: str(p.get("key")))
            ),
        }
        for index, name in enumerate(names)
    ]
    series = [s for s in series if s["points"]]
    if not series:
        return _empty_view(
            intent,
            ctx,
            title="Comparaison d'espèces",
            reason="Aucune série exploitable pour les espèces demandées.",
            extra_notices=base_notices,
        )

    totals = {s["label"]: sum(p["y"] for p in s["points"]) for s in series}
    ranking = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    kpis = [
        {
            "label": label,
            "value": fmt_int(total),
            "hint": "observations cumulées",
            "tone": "neutral",
        }
        for label, total in ranking
    ]
    leader, leader_total = ranking[0]
    second = ranking[1] if len(ranking) > 1 else None
    text = (
        f"Comparaison {intent.period_label or 'sur toute la période'} : {leader} totalise "
        f"{fmt_int(leader_total)} observations"
        + (
            f", contre {fmt_int(second[1])} pour {second[0]} "
            f"(écart {fmt_int(leader_total - second[1])})."
            if second
            else "."
        )
    )
    data = {
        "header": _header(intent, f"Comparaison — {' vs '.join(s['label'] for s in series)}"),
        "kpis": kpis,
        "chart": {
            "title": "Observations par mois",
            "series": series,
            "xLabel": "Mois",
            "yLabel": "Observations",
            "emptyMessage": "Aucune série.",
        },
        "table": {
            "title": "Totaux comparés",
            "caption": "Somme des observations sur la période.",
            "rows": [{"period": label, "value": total, "delta": None} for label, total in ranking],
            "emptyMessage": "Aucune donnée.",
        },
        "notices": base_notices,
        "filters": ctx.filters(intent),
        "footer": {"source": ctx.source_line(agg)},
    }
    return BuiltView("trend", data, text)


# ── Fiche espèce ─────────────────────────────────────────────────────────────


def build_species(intent: Intent, detail: dict, ctx: ViewContext) -> BuiltView:
    species = detail.get("species") or {}
    monthly = list(detail.get("monthly") or [])
    base_notices = ctx.dataset_notices() + ctx.warnings(detail)

    label = str(species.get("label") or intent.species_label)
    protection = (
        species.get("protection") or "Statut de protection non renseigné dans ce jeu de données."
    )
    if species.get("sensitive"):
        base_notices.append(
            notice(
                "warning",
                "Espèce sensible",
                "La localisation précise n'est pas diffusée ; seules les échelles agrégées "
                "sont disponibles.",
            )
        )
    if not monthly:
        base_notices.append(
            notice(
                "info",
                "Aucune série mensuelle",
                "Le jeu de données ne permet pas de détailler les observations par mois.",
            )
        )

    kpis = [
        {
            "label": "Observations",
            "value": fmt_int(species.get("observation_count")),
            "hint": intent.period_label or "toute la période disponible",
            "tone": "neutral",
        },
        {
            "label": "Individus",
            "value": fmt_int(species.get("individual_count"))
            if species.get("individual_count") is not None
            else "—",
            "hint": "somme des effectifs",
            "tone": "neutral",
        },
        {
            "label": "Départements",
            "value": fmt_int(len(species.get("departments") or [])),
            "hint": ", ".join((species.get("departments") or [])[:3]) or "—",
            "tone": "neutral",
        },
    ]
    first, last = fmt_date(species.get("first_seen")), fmt_date(species.get("last_seen"))
    text = (
        f"{label} : {fmt_int(species.get('observation_count'))} observation(s)"
        + (
            f", {fmt_int(species.get('individual_count'))} individu(s)"
            if species.get("individual_count") is not None
            else ""
        )
        + f", de {first} à {last}, dans "
        f"{fmt_int(len(species.get('departments') or []))} département(s). "
        f"Statut : {protection}"
    )
    data = {
        "header": _header(
            intent, f"Fiche espèce — {label}", extra=str(species.get("latin_name") or "")
        ),
        "kpis": kpis,
        "species": {
            "name": label,
            "latinName": species.get("latin_name") or "",
            "protection": protection,
            "note": (
                f"Données observées de {first} à {last}. "
                f"Départements : {', '.join(species.get('departments') or []) or 'non renseignés'}."
            ),
        },
        "table": {
            "title": "Observations par mois",
            "caption": "Série mensuelle issue du serveur MCP.",
            "rows": [
                {
                    "period": str(p.get("label")),
                    "value": p.get("value"),
                    "individuals": p.get("individuals"),
                }
                for p in monthly[-24:]
            ],
            "emptyMessage": "Aucune série mensuelle disponible.",
        },
        "notices": base_notices,
        "filters": ctx.filters(intent),
        "footer": {"source": ctx.source_line(detail)},
    }
    return BuiltView("species", data, text)


# ── État vide ────────────────────────────────────────────────────────────────


def _empty_view(
    intent: Intent,
    ctx: ViewContext,
    *,
    title: str,
    reason: str,
    extra_notices: list[dict] | None = None,
) -> BuiltView:
    notices = [notice("info", "Aucune donnée à afficher", reason)]
    notices += extra_notices or []
    data = {
        "header": _header(intent, title),
        "notices": notices,
        "footer": {"source": ctx.source_line()},
    }
    return BuiltView("empty", data, f"{reason} Question reformulée : {intent.raw_question}")


def build_unknown(intent: Intent, ctx: ViewContext, *, reason: str) -> BuiltView:
    """Question hors périmètre : on explique au lieu de deviner."""
    return _empty_view(intent, ctx, title="Question hors périmètre", reason=reason)


def build_unknown_term(intent: Intent, ctx: ViewContext, terms: list[str]) -> BuiltView:
    """Terme cité (espèce, zone) absent du jeu de données : on refuse de répondre.

    Constat tiré de l'extrait réel du dépôt : « évolution des observations de
    Milan royal » y produisait une courbe de 169 lignes — celles de toutes les
    zones, dont 131 sans aucune espèce. Le chiffre était exact, la réponse
    fausse : elle attribuait au Milan royal des données qui ne le concernent pas.
    """
    quoted = ", ".join(f"« {term} »" for term in terms)
    known = ctx.species_options
    if known:
        sample = ", ".join(known[:5]) + ("…" if len(known) > 5 else "")
        available = f"Espèces présentes dans le jeu de données : {sample}."
    else:
        available = "Le jeu de données exposé ne contient aucun nom d'espèce."
    reason = (
        f"{quoted} n'est pas reconnu dans le jeu de données exposé — ni espèce, ni zone "
        f"connue de ce jeu. {available} Aucun chiffre n'est affiché : attribuer à ce terme "
        "les données d'une autre espèce ou d'une autre zone serait faux."
    )
    notices = [
        notice("warning", "Terme non reconnu", reason),
        *ctx.dataset_notices(),
    ]
    data = {
        "header": _header(intent, "Réponse impossible en l'état"),
        "notices": notices,
        "footer": {"source": ctx.source_line()},
    }
    text = f"{reason} Précisez le libellé ou la zone : « {intent.raw_question} »."
    return BuiltView("empty", data, text)
