"""Logique métier du serveur MCP `lpo-data` (lecture seule).

La couche est volontairement **séparée du serveur MCP** : `service.py` contient
les requêtes et les règles (whitelist, pagination, confidentialité), `server.py`
ne fait qu'exposer des outils typés. Les tests peuvent donc appeler le service
directement, sans transport MCP.

Toutes les requêtes passent par ``src.database.engine.run_select_params`` :
garde lecture seule du dépôt + paramètres liés + plafond de lignes.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

from src.database import engine

from . import privacy as privacy_rules
from .models import (
    AggregateResult,
    DatasetOverview,
    Observation,
    ObservationPage,
    ResultMeta,
    SeriesPoint,
    SpeciesDetail,
    SpeciesInfo,
    SpeciesListResult,
)
from .profiles import NO_SPECIES_LABEL, SchemaProfile, detect_profile
from .sql import (
    GROUP_BY,
    METRICS,
    QueryError,
    Where,
    check_choice,
    date_bounds,
    ident,
    limit_offset,
    lower_trim,
    parse_date,
    round_expr,
    time_bucket,
)

logger = logging.getLogger("lpo_mcp.service")

MAX_LIMIT = 500
DEFAULT_SPECIES_LIMIT = 25
DEFAULT_ROW_LIMIT = 50


def _num(value: float | int | None, decimals: int = 0) -> str:
    """Formatage français : 1 234 (espace fine insécable remplacée par une espace)."""
    if value is None:
        return "—"
    if decimals:
        return f"{float(value):,.{decimals}f}".replace(",", " ")
    return f"{int(round(float(value))):,}".replace(",", " ")


class LpoDataService:
    """Accès en lecture seule aux données naturalistes, via un profil de schéma."""

    def __init__(self, profile: SchemaProfile | None = None):
        self._profile = profile
        #: lignes sans espèce (zones purement réglementaires), calculées une fois
        #: par processus : ce compte ne sert qu'aux messages, jamais aux chiffres.
        self._excluded_species: int | None = None

    # ── Profil & introspection ────────────────────────────────────────────

    @property
    def profile(self) -> SchemaProfile:
        if self._profile is None:
            tables = engine.list_tables()
            self._profile = detect_profile(tables)
            logger.info(
                "Profil MCP sélectionné: %s (table %s)", self._profile.name, self._profile.table
            )
        return self._profile

    def reset(self) -> None:
        """Oublie le profil détecté (utile si la base a été rechargée)."""
        self._profile = None
        self._excluded_species = None

    @property
    def dialect(self) -> str:
        return engine.get_engine().dialect.name

    def _meta(
        self,
        *,
        row_count: int,
        warnings: Sequence[str] = (),
        truncated: bool = False,
        privacy: str = "auto",
    ) -> ResultMeta:
        profile = self.profile
        return ResultMeta(
            profile=profile.name,
            table=profile.table,
            capabilities=sorted(profile.capabilities),
            row_count=row_count,
            truncated=truncated,
            privacy=privacy,
            warnings=list(warnings),
        )

    def _query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        df = engine.run_select_params(sql, params, max_rows=MAX_LIMIT * 4)
        return df.to_dict(orient="records")

    def _table(self) -> str:
        return ident(self.profile.table)

    # ── Filtres ───────────────────────────────────────────────────────────

    def _species_clause(self, where: Where, species: Sequence[str] | None) -> None:
        profile = self.profile
        expr = profile.species_expr
        if not species:
            return
        if expr is None:
            raise QueryError("Ce jeu de données ne contient pas d'espèce.")
        label_clauses: list[str] = []
        key_clauses: list[str] = []
        for index, raw in enumerate([s for s in species if str(s).strip()]):
            value = str(raw).strip()
            label_clauses.append(f"LOWER({expr}) = :sp{index}")
            where.params[f"sp{index}"] = value.lower()
            # Identifiants numériques : on accepte « 10 », « 10.0 » ou « Espèce #10 ».
            if profile.species_key and not profile.species_label:
                numeric = re.findall(r"\d+(?:\.\d+)?", value)
                if numeric:
                    key_clauses.append(
                        f"CAST(CAST({ident(profile.species_key)} AS INTEGER) AS TEXT) = :spk{index}"
                    )
                    where.params[f"spk{index}"] = str(int(float(numeric[0])))
        clauses = label_clauses + key_clauses
        if clauses:
            where.add("(" + " OR ".join(clauses) + ")")

    def _area_clause(self, where: Where, department: str | None, region: str | None) -> None:
        profile = self.profile
        if department:
            if not profile.department:
                raise QueryError("Filtre département indisponible dans ce profil.")
            where.add(
                f"{lower_trim(profile.department)} = LOWER(:department)",
                department=department.strip(),
            )
        if region:
            if not profile.region:
                raise QueryError("Filtre région indisponible dans ce profil.")
            where.add(f"{lower_trim(profile.region)} = LOWER(:region)", region=region.strip())

    def _sensitive_expr(self) -> str | None:
        """Expression SQL « cette ligne est-elle sensible ? ».

        Trois sources, combinées par OU : la colonne dédiée du schéma, le statut
        de protection, et la **liste configurée** (`LPO_SENSITIVE_SPECIES` /
        `LPO_SENSITIVE_SPECIES_FILE`) comparée au libellé d'espèce. Cette
        dernière permet d'appliquer une liste officielle sur un schéma qui ne
        porte aucun indicateur de sensibilité.
        """
        profile = self.profile
        parts: list[str] = []
        if profile.sensitive:
            parts.append(f"({ident(profile.sensitive)})")
        if profile.protection:
            values = ", ".join(f"'{v.lower()}'" for v in profile.protected_values)
            parts.append(f"(LOWER(TRIM({ident(profile.protection)})) IN ({values}))")
        species_column = profile.species_label or profile.species_key
        if species_column:
            clause = privacy_rules.sensitive_species_sql_clause(ident(species_column))
            if clause:
                parts.append(clause)
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else "(" + " OR ".join(parts) + ")"

    def _base_where(
        self,
        *,
        species: Sequence[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        department: str | None = None,
        region: str | None = None,
        include_sensitive: bool = True,
    ) -> Where:
        profile = self.profile
        where = Where()
        self._species_clause(where, species)
        self._area_clause(where, department, region)
        if start_date or end_date:
            if not profile.event_date:
                raise QueryError("Ce jeu de données ne contient pas de date.")
            date_bounds(
                parse_date(start_date, label="start_date"),
                parse_date(end_date, label="end_date"),
                self.dialect,
                where,
                profile.event_date,
            )
        if not include_sensitive:
            sensitive_expr = self._sensitive_expr()
            if sensitive_expr:
                where.add(f"NOT ({sensitive_expr})")
        return where

    def _unspecified_species_count(self) -> int:
        """Nombre de lignes qui ne portent aucune espèce.

        Sur l'extrait réel du dépôt, 131 des 169 lignes sont des zones
        réglementaires sans `species_id`. Elles ne sont pas des espèces : elles
        sont donc exclues des listes, classements et fiches, et **annoncées**
        (sinon l'utilisateur croirait le jeu de données complet).
        """
        profile = self.profile
        if not profile.species_where:
            return 0
        if self._excluded_species is None:
            try:
                rows = self._query(
                    f"SELECT COUNT(*) AS n FROM {self._table()} "
                    f"WHERE NOT ({profile.species_where})"
                )
                self._excluded_species = int(rows[0]["n"]) if rows else 0
            except Exception:  # noqa: BLE001 - un compteur ne doit pas casser l'outil
                return 0
        return self._excluded_species

    def _exclusion_note(self, total: int | None = None) -> str | None:
        """Phrase d'explication des lignes écartées, ou None s'il n'y en a pas."""
        excluded = self._unspecified_species_count()
        if not excluded:
            return None
        scope = f" sur {_num(total)}" if total else ""
        return (
            f"{_num(excluded)} enregistrement(s){scope} ne portent aucune espèce "
            f"({NO_SPECIES_LABEL}) : exclus des décomptes par espèce."
        )

    # ── Outil : aperçu du jeu de données ─────────────────────────────────

    def dataset_overview(self) -> DatasetOverview:
        profile = self.profile
        table = self._table()
        warnings: list[str] = []
        try:
            agg = self._query(
                "SELECT COUNT(*) AS n"
                + (
                    f", MIN({ident(profile.event_date)}) AS dmin,"
                    f" MAX({ident(profile.event_date)}) AS dmax"
                    if profile.event_date
                    else ""
                )
                + (
                    # Les lignes sans espèce ne doivent pas gonfler le décompte :
                    # `COUNT(DISTINCT NULL)` les ignore déjà, mais le CASE rend la
                    # règle explicite et identique sur SQLite et PostgreSQL.
                    f", COUNT(DISTINCT CASE WHEN {profile.species_where} "
                    f"THEN {profile.species_expr} END) AS n_species"
                    if profile.species_expr and profile.species_where
                    else ""
                )
                + (
                    f", COUNT(DISTINCT {ident(profile.department)}) AS n_dept"
                    if profile.department
                    else ""
                )
                + f" FROM {table}"
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Lecture impossible de {profile.table}: {exc}") from exc

        row = agg[0] if agg else {}
        total = int(row.get("n") or 0)
        sample: list[str] = []
        if profile.species_expr:
            try:
                sample = [
                    str(r["label"])
                    for r in self._query(
                        f"SELECT {profile.species_expr} AS label, COUNT(*) AS n FROM {table} "
                        f"WHERE {profile.species_where} "
                        "GROUP BY 1 ORDER BY n DESC LIMIT 5"
                    )
                ]
            except Exception:  # noqa: BLE001
                sample = []

        dataset_label, is_demo, provenance_notes = self._dataset_metadata()
        warnings.extend(provenance_notes)
        columns = [c["name"] for c in engine.get_schema(profile.table)["columns"]]
        summary = (
            f"Jeu de données « {profile.name} » : {_num(total)} ligne(s) dans la table "
            f"{profile.table}. Capacités disponibles : "
            f"{', '.join(sorted(profile.capabilities)) or 'aucune'}."
        )
        if dataset_label:
            summary = f"{dataset_label}. " + summary
        if profile.notes:
            summary += " Limites : " + " ".join(profile.notes)
        exclusion_note = self._exclusion_note(total)
        if exclusion_note:
            warnings.append(exclusion_note)
            summary += " " + exclusion_note
        configured_species = privacy_rules.configured_sensitive_species()
        if configured_species:
            warnings.append(
                f"{len(configured_species)} espèce(s) marquée(s) sensible(s) par configuration "
                "(LPO_SENSITIVE_SPECIES) : leur localisation est floutée."
            )

        return DatasetOverview(
            summary=summary,
            profile=profile.name,
            table=profile.table,
            description=profile.description,
            dataset_label=dataset_label,
            is_demo=is_demo,
            capabilities=sorted(profile.capabilities),
            columns=columns,
            total_rows=total,
            records_label=profile.records_label,
            date_min=str(row["dmin"]) if row.get("dmin") else None,
            date_max=str(row["dmax"]) if row.get("dmax") else None,
            species_count=int(row["n_species"]) if row.get("n_species") is not None else None,
            department_count=int(row["n_dept"]) if row.get("n_dept") is not None else None,
            departments=self._distinct(profile.department, limit=200),
            regions=self._distinct(profile.region, limit=50),
            notes=list(profile.notes),
            sample_species=sample,
            meta=self._meta(row_count=1, warnings=warnings),
        )

    def _distinct(self, column: str | None, *, limit: int = 200) -> list[str]:
        """Valeurs distinctes non vides d'une colonne (aide à la découverte)."""
        if not column:
            return []
        try:
            rows = self._query(
                f"SELECT DISTINCT {ident(column)} AS v FROM {self._table()} "
                f"WHERE {ident(column)} IS NOT NULL AND TRIM({ident(column)}) != '' "
                f"ORDER BY 1 LIMIT {int(limit)}"
            )
        except Exception:  # noqa: BLE001
            return []
        return [str(r["v"]) for r in rows if r.get("v") is not None]

    def _dataset_metadata(self) -> tuple[str | None, bool, list[str]]:
        """Lit la table de provenance `dataset_metadata` si elle existe."""
        try:
            tables = engine.list_tables()
        except Exception:  # noqa: BLE001
            return None, False, []
        if "dataset_metadata" not in tables:
            return None, False, []
        try:
            rows = self._query("SELECT key, value FROM dataset_metadata")
        except Exception:  # noqa: BLE001
            return None, False, []
        meta = {str(r["key"]): str(r["value"]) for r in rows if r.get("key") is not None}
        label = meta.get("label")
        is_demo = meta.get("is_demo", "").strip().lower() in {"1", "true", "oui", "yes"}
        notes: list[str] = []
        if is_demo:
            notes.append("Jeu de démonstration synthétique : aucune valeur écologique réelle.")
        if meta.get("generated_at"):
            notes.append(f"Jeu de démonstration généré le {meta['generated_at']}.")
        return label, is_demo, notes

    # ── Outil : liste d'espèces ───────────────────────────────────────────

    def list_species(
        self,
        query: str | None = None,
        limit: int = DEFAULT_SPECIES_LIMIT,
        offset: int = 0,
    ) -> SpeciesListResult:
        profile = self.profile
        expr = profile.species_expr
        if expr is None:
            return SpeciesListResult(
                summary="Ce jeu de données ne contient aucune information d'espèce.",
                query=query,
                species=[],
                total=0,
                meta=self._meta(row_count=0, warnings=["Capacité 'species' absente."]),
            )

        limit_i, offset_i = limit_offset(limit, offset, max_limit=MAX_LIMIT)
        where = Where()
        if profile.species_where:
            where.add(profile.species_where)
        if query:
            where.add(f"LOWER({expr}) LIKE :q", q=f"%{query.strip().lower()}%")

        protection_expr = f"MAX({ident(profile.protection)})" if profile.protection else "NULL"
        sensitive_expr = self._sensitive_expr() or "0"
        count_expr = f"SUM({ident(profile.count)})" if profile.count else "NULL"
        date_min = f"MIN({ident(profile.event_date)})" if profile.event_date else "NULL"
        date_max = f"MAX({ident(profile.event_date)})" if profile.event_date else "NULL"
        table = self._table()

        # `total` = nombre d'espèces distinctes correspondant au filtre.
        total_rows = self._query(
            f"SELECT COUNT(DISTINCT {expr}) AS n FROM {table}{where.sql()}", where.params
        )
        total = int(total_rows[0]["n"]) if total_rows else 0

        rows = self._query(
            f"SELECT {expr} AS label, COUNT(*) AS n, {count_expr} AS individuals, "
            f"{protection_expr} AS protection, {sensitive_expr} AS sensitive, "
            f"{date_min} AS dmin, {date_max} AS dmax "
            f"FROM {table}{where.sql()} GROUP BY 1 ORDER BY n DESC LIMIT :lim OFFSET :off",
            {**where.params, "lim": limit_i, "off": offset_i},
        )

        species = [
            SpeciesInfo(
                id=str(r["label"]),
                label=str(r["label"]),
                protection=r.get("protection"),
                sensitive=privacy_rules.is_sensitive(
                    r.get("sensitive"), protected_values=profile.protected_values
                ),
                observation_count=int(r.get("n") or 0),
                individual_count=(
                    float(r["individuals"]) if r.get("individuals") is not None else None
                ),
                first_seen=str(r["dmin"]) if r.get("dmin") else None,
                last_seen=str(r["dmax"]) if r.get("dmax") else None,
            )
            for r in rows
        ]

        warnings: list[str] = []
        if profile.notes:
            warnings.extend(profile.notes[:2])
        exclusion_note = self._exclusion_note()
        if exclusion_note:
            warnings.append(exclusion_note)
        summary = (
            f"{_num(total)} espèce(s) distincte(s)"
            + (f" correspondant à « {query} »" if query else "")
            + f" ; {len(species)} affichée(s)."
        )
        if species:
            top = species[0]
            summary += (
                f" La plus observée : {top.label} ({_num(top.observation_count)} observations)."
            )
        return SpeciesListResult(
            summary=summary,
            query=query,
            species=species,
            total=total,
            meta=self._meta(row_count=len(species), warnings=warnings),
        )

    # ── Outil : observations détaillées ──────────────────────────────────

    def get_observations(
        self,
        species: Sequence[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        department: str | None = None,
        region: str | None = None,
        limit: int = DEFAULT_ROW_LIMIT,
        offset: int = 0,
        privacy_mode: str = "auto",
        include_sensitive: bool = True,
    ) -> ObservationPage:
        profile = self.profile
        expr = profile.species_expr
        if expr is None:
            raise QueryError("Ce jeu de données ne contient pas d'espèce.")

        limit_i, offset_i = limit_offset(limit, offset, max_limit=MAX_LIMIT)
        where = self._base_where(
            species=species,
            start_date=start_date,
            end_date=end_date,
            department=department,
            region=region,
            include_sensitive=include_sensitive,
        )
        table = self._table()

        def col(name: str | None) -> str:
            return ident(name) if name else "NULL"

        order = f" ORDER BY {ident(profile.event_date)} DESC" if profile.event_date else ""
        sql = (
            f"SELECT {col(profile.id_column)} AS id, {expr} AS species, "
            f"{col(profile.event_date)} AS event_date, {col(profile.count)} AS count, "
            f"{col(profile.department)} AS department, {col(profile.region)} AS region, "
            f"{col(profile.lat)} AS lat, {col(profile.lon)} AS lon, "
            f"{col(profile.protection)} AS protection, "
            f"{self._sensitive_expr() or '0'} AS sensitive "
            f"FROM {table}{where.sql()}{order} LIMIT :lim OFFSET :off"
        )
        rows = self._query(sql, {**where.params, "lim": limit_i, "off": offset_i})
        total_rows = self._query(f"SELECT COUNT(*) AS n FROM {table}{where.sql()}", where.params)
        total = int(total_rows[0]["n"]) if total_rows else 0

        observations: list[Observation] = []
        blurred = 0
        warnings: list[str] = []
        resolved_modes: set[str] = set()
        for row in rows:
            sensitive = privacy_rules.is_sensitive(
                row.get("sensitive"), protected_values=profile.protected_values
            )
            mode = privacy_rules.resolve_mode(privacy_mode, sensitive=sensitive)
            resolved_modes.add(mode)
            lat, lon = privacy_rules.apply_privacy(
                lat=row.get("lat"), lon=row.get("lon"), sensitive=sensitive, mode=mode
            )
            if mode != "exact":
                blurred += 1
            observations.append(
                Observation(
                    id=str(row["id"]) if row.get("id") is not None else None,
                    species=(
                        str(row["species"]) if row.get("species") is not None else None
                    ),
                    event_date=str(row["event_date"]) if row.get("event_date") else None,
                    count=float(row["count"]) if row.get("count") is not None else None,
                    department=row.get("department"),
                    region=row.get("region"),
                    lat=lat,
                    lon=lon,
                    protection=row.get("protection"),
                    sensitive=sensitive,
                    precision=mode,
                )
            )

        if not profile.capabilities & {"coordinates"}:
            warnings.append("Coordonnées absentes du schéma : aucune carte possible.")
        if not profile.count:
            warnings.append("Effectifs absents du schéma : seules les lignes sont comptées.")

        privacy_label = ",".join(sorted(resolved_modes)) or privacy_mode
        summary = (
            f"{_num(total)} observation(s) correspondent aux filtres ; "
            f"{len(observations)} renvoyée(s) à partir de la ligne {offset_i}."
        )
        if blurred:
            summary += " " + privacy_rules.privacy_summary(
                privacy_rules.resolve_mode(privacy_mode, sensitive=True), sensitive_rows=blurred
            )
        return ObservationPage(
            summary=summary,
            observations=observations,
            total=total,
            returned=len(observations),
            offset=offset_i,
            meta=self._meta(
                row_count=len(observations),
                warnings=warnings,
                truncated=total > offset_i + len(observations),
                privacy=privacy_label,
            ),
        )

    # ── Outil : agrégations ───────────────────────────────────────────────

    def aggregate_observations(
        self,
        group_by: str = "year",
        metric: str = "observations",
        species: Sequence[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        department: str | None = None,
        region: str | None = None,
        limit: int = 50,
        privacy_mode: str = "auto",
    ) -> AggregateResult:
        profile = self.profile
        group_by = check_choice(group_by, GROUP_BY, "group_by")
        metric = check_choice(metric, METRICS, "metric")
        limit_i, _ = limit_offset(limit, 0, max_limit=MAX_LIMIT)
        warnings: list[str] = []
        table = self._table()

        if metric == "individuals" and not profile.count:
            warnings.append(
                "Effectifs indisponibles dans ce schéma : "
                "métrique 'observations' utilisée à la place."
            )
            metric = "observations"

        if group_by == "grid" and not (profile.lat and profile.lon):
            return AggregateResult(
                summary=(
                    "Agrégation spatiale impossible : ce jeu de données ne contient pas de "
                    "coordonnées. Utiliser à la place un regroupement par département."
                ),
                group_by=group_by,
                metric=metric,
                points=[],
                total=0,
                meta=self._meta(
                    row_count=0,
                    warnings=warnings + ["Capacité 'coordinates' absente."],
                    privacy=privacy_mode,
                ),
            )
        if group_by in {"species_month", "species_year"} and (
            not profile.species_expr or not profile.event_date
        ):
            return AggregateResult(
                summary="Comparaison mensuelle impossible : espèce ou date absente du schéma.",
                group_by=group_by,
                metric=metric,
                points=[],
                total=0,
                meta=self._meta(
                    row_count=0,
                    warnings=warnings + ["Capacité 'species' ou 'temporal' absente."],
                    privacy=privacy_mode,
                ),
            )
        if group_by == "species" and not profile.species_expr:
            return AggregateResult(
                summary="Agrégation par espèce impossible : aucune information d'espèce.",
                group_by=group_by,
                metric=metric,
                points=[],
                total=0,
                meta=self._meta(
                    row_count=0,
                    warnings=warnings + ["Capacité 'species' absente."],
                    privacy=privacy_mode,
                ),
            )
        if group_by == "department" and not profile.department:
            return AggregateResult(
                summary="Agrégation par département impossible : colonne absente.",
                group_by=group_by,
                metric=metric,
                points=[],
                total=0,
                meta=self._meta(
                    row_count=0,
                    warnings=warnings + ["Capacité 'department' absente."],
                    privacy=privacy_mode,
                ),
            )

        where = self._base_where(
            species=species,
            start_date=start_date,
            end_date=end_date,
            department=department,
            region=region,
        )

        count_expr = f"SUM({ident(profile.count)})" if profile.count else "NULL"
        # Les alias ne sont pas réutilisables dans la même SELECT : on répète
        # l'expression agrégée pour la colonne `value`.
        value_expr = "COUNT(*)" if metric == "observations" else count_expr
        sensitive_expr = self._sensitive_expr()
        sensitive_select = f"MAX({sensitive_expr})" if sensitive_expr else "0"
        extra_cols = ""
        group_cols = ""
        order_clause = ""
        privacy_applied = privacy_mode

        if group_by in {"year", "month"}:
            expr, _label = time_bucket(profile.event_date or "", group_by, self.dialect)
            group_cols = f"{expr} AS k"
            order_clause = "ORDER BY k ASC"
        elif group_by in {"department", "region"}:
            column = profile.department if group_by == "department" else profile.region
            group_cols = f"{ident(column or '')} AS k"
            where.add(f"{ident(column or '')} IS NOT NULL AND TRIM({ident(column or '')}) != ''")
            order_clause = "ORDER BY value DESC"
        elif group_by == "species":
            if profile.species_where:
                where.add(profile.species_where)
            group_cols = f"{profile.species_expr} AS k"
            order_clause = "ORDER BY value DESC"
        elif group_by in {"species_month", "species_year"}:
            level = "month" if group_by == "species_month" else "year"
            expr, _label = time_bucket(profile.event_date or "", level, self.dialect)
            if profile.species_where:
                where.add(profile.species_where)
            group_cols = f"{profile.species_expr} AS series, {expr} AS k"
            # Tri chronologique d'abord : une éventuelle troncature reste
            # interprétable (elle coupe la fin de l'historique, pas une espèce).
            order_clause = "ORDER BY k ASC, series ASC"
        else:  # grid
            decimals = privacy_rules.blur_decimals()
            group_cols = (
                f"{round_expr(profile.lat or '', decimals, self.dialect)} AS lat, "
                f"{round_expr(profile.lon or '', decimals, self.dialect)} AS lon"
            )
            where.add(
                f"{ident(profile.lat or '')} IS NOT NULL AND {ident(profile.lon or '')} IS NOT NULL"
            )
            order_clause = "ORDER BY value DESC"
            privacy_applied = "blurred"

        sql = (
            f"SELECT {group_cols}, COUNT(*) AS obs, {count_expr} AS individuals, "
            f"{sensitive_select} AS sensitive, {value_expr} AS value{extra_cols} "
            f"FROM {table}{where.sql()} GROUP BY 1"
            + (", 2" if group_by in {"grid", "species_month", "species_year"} else "")
            + f" {order_clause} LIMIT :lim"
        )
        rows = self._query(sql, {**where.params, "lim": limit_i})

        totals = self._query(
            f"SELECT COUNT(*) AS obs, {count_expr} AS individuals FROM {table}{where.sql()}",
            where.params,
        )
        total_obs = int(totals[0]["obs"]) if totals else 0
        total_individuals = (
            float(totals[0]["individuals"])
            if totals and totals[0].get("individuals") is not None
            else None
        )
        total = float(total_obs if metric == "observations" else (total_individuals or 0))

        points: list[SeriesPoint] = []
        for row in rows:
            value = row.get("value")
            if value is None:
                value = 0
            if group_by == "grid":
                lat = float(row["lat"]) if row.get("lat") is not None else None
                lon = float(row["lon"]) if row.get("lon") is not None else None
                key_label = f"{lat}, {lon}"
            else:
                key_label = str(row.get("k"))
                lat = lon = None
            series = row.get("series") if group_by in {"species_month", "species_year"} else None
            points.append(
                SeriesPoint(
                    key=key_label,
                    label=key_label,
                    series=str(series) if series else None,
                    value=float(value),
                    observations=int(row.get("obs") or 0),
                    individuals=(
                        float(row["individuals"]) if row.get("individuals") is not None else None
                    ),
                    share=(float(value) / total) if total else None,
                    lat=lat,
                    lon=lon,
                    sensitive=privacy_rules.is_sensitive(
                        row.get("sensitive"), protected_values=profile.protected_values
                    ),
                )
            )

        if group_by == "grid" and points:
            warnings.append(
                "Coordonnées agrégées sur une maille d'environ "
                f"{privacy_rules.grid_size_km()} km (protection des espèces sensibles)."
            )
        if profile.notes and group_by in {"department", "region", "species"}:
            warnings.extend(profile.notes[:1])

        metric_label = "observations" if metric == "observations" else "individus"
        unit = {
            "year": "par an",
            "month": "par mois",
            "department": "par département",
            "region": "par région",
            "species": "par espèce",
            "grid": "par maille",
            "species_month": "par mois et par espèce",
            "species_year": "par an et par espèce",
        }[group_by]
        summary = f"{_num(total)} {metric_label} au total, {unit} ({len(points)} point(s))."
        if points:
            best = max(points, key=lambda p: p.value)
            summary += f" Maximum : {best.label} ({_num(best.value)})."

        return AggregateResult(
            summary=summary,
            group_by=group_by,
            metric=metric,
            points=points,
            total=total,
            total_observations=total_obs,
            meta=self._meta(
                row_count=len(points),
                warnings=warnings,
                truncated=len(rows) >= limit_i,
                privacy=privacy_applied,
            ),
        )

    # ── Outil : fiche espèce ──────────────────────────────────────────────

    def get_species_info(self, species: str) -> SpeciesDetail:
        profile = self.profile
        if not species or not str(species).strip():
            raise QueryError("Paramètre 'species' obligatoire.")
        name = str(species).strip()

        # Correspondance **exacte** uniquement : prendre « le premier résultat »
        # ferait répondre sur le Milan royal à une question sur le Héron.
        listing = self.list_species(query=name, limit=50)
        match = next((s for s in listing.species if s.label.lower() == name.lower()), None)
        if match is None and profile.species_is_numeric_key:
            digit_groups = re.findall(r"\d+", name)
            if digit_groups:
                wanted = f"Espèce #{int(digit_groups[0])}"
                match = next((s for s in listing.species if s.label == wanted), None)
        if match is None:
            available = ", ".join(s.label for s in self.list_species(limit=5).species)
            hint = ""
            if profile.species_where and self._unspecified_species_count():
                hint = (
                    f" {_num(self._unspecified_species_count())} enregistrement(s) ne "
                    f"portent aucune espèce ({NO_SPECIES_LABEL}) et ne peuvent donc pas "
                    "faire l'objet d'une fiche."
                )
            raise QueryError(
                f"Espèce « {name} » introuvable (correspondance exacte exigée). "
                f"Espèces disponibles : {available or 'aucune'}.{hint}"
            )

        monthly = self.aggregate_observations(group_by="month", species=[match.label], limit=120)
        by_dept = self.aggregate_observations(
            group_by="department", species=[match.label], limit=10
        )
        species_info = match.model_copy(
            update={
                "departments": [p.label for p in by_dept.points[:10]],
            }
        )

        warnings: list[str] = []
        if not profile.species_label:
            warnings.append(
                "Nom d'espèce absent du schéma : seul un identifiant numérique est disponible."
            )
        if not profile.protection:
            warnings.append("Statut de protection absent du schéma.")
        if match.sensitive:
            warnings.append(
                "Espèce sensible : les localisations précises ne doivent pas être diffusées."
            )

        summary = (
            f"{match.label} : {_num(match.observation_count)} observation(s)"
            + (f", {_num(match.individual_count)} individu(s)" if match.individual_count else "")
            + (
                f", de {match.first_seen[:10]} à {match.last_seen[:10]}"
                if match.first_seen and match.last_seen
                else ""
            )
            + f", {len(by_dept.points)} département(s) concerné(s)."
        )
        return SpeciesDetail(
            summary=summary,
            species=species_info,
            monthly=monthly.points,
            top_departments=by_dept.points,
            meta=self._meta(
                row_count=len(monthly.points),
                warnings=warnings + monthly.meta.warnings[:1],
                privacy=monthly.meta.privacy,
            ),
        )


_DEFAULT_SERVICE: LpoDataService | None = None


def get_service() -> LpoDataService:
    """Service partagé (profil détecté une seule fois)."""
    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = LpoDataService()
    return _DEFAULT_SERVICE


def reset_service() -> None:
    global _DEFAULT_SERVICE
    _DEFAULT_SERVICE = None
