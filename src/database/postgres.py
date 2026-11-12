"""Réglages PostgreSQL, isolés pour être testables sans serveur.

SQLite reste le moteur de développement (base générée) ; **la production tourne
sur PostgreSQL**. Ce module rassemble tout ce qui est propre à ce moteur, pour
que ``engine.py`` reste lisible et que ces règles soient vérifiées par des tests
unitaires, sans instance à disposition :

- **DSN** : ``postgres://`` et ``postgresql://`` sont normalisés vers le pilote
  psycopg 3 (``postgresql+psycopg://``), le seul déclaré dans
  ``requirements.txt``. Les hébergeurs fournissent souvent ``postgres://``, que
  SQLAlchemy ne sait pas résoudre (il chercherait un dialecte nommé « postgres »).
- **Paramètres de connexion** : délai de connexion borné, ``application_name``
  lisible dans ``pg_stat_activity``, ``search_path`` restreint au schéma métier
  (le serveur ne voit pas les schémas des autres applications du même cluster).
- **Garde-fous de session**, appliqués à chaque connexion avant toute requête :
  ``statement_timeout``, ``lock_timeout``,
  ``idle_in_transaction_session_timeout``, fuseau UTC, et
  ``default_transaction_read_only`` quand l'application est en lecture seule.

``default_transaction_read_only`` est un garde-fou, **pas une frontière de
sécurité** : qui détient le mot de passe peut le désactiver. La frontière réelle
est le rôle PostgreSQL en lecture seule décrit dans ``docs/02-exploitation.md``.
"""

from __future__ import annotations

import re

#: Schéma SQLAlchemy du pilote unique (psycopg 3).
PG_DRIVER_SCHEME = "postgresql+psycopg"

_SCHEMA_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_POSTGRES_URL_PREFIXES = ("postgres://", "postgresql://")


def normalize_database_url(url: str) -> str:
    """Retourne une URL SQLAlchemy exploitable pour le moteur visé.

    - ``postgres://…`` → ``postgresql+psycopg://…`` ;
    - ``postgresql://…`` → ``postgresql+psycopg://…`` ;
    - tout le reste (``sqlite:///…``, ``postgresql+psycopg2://…``) est rendu
      tel quel : un pilote explicitement choisi n'est jamais réécrit.
    """
    cleaned = (url or "").strip()
    for prefix in _POSTGRES_URL_PREFIXES:
        if cleaned.startswith(prefix):
            return f"{PG_DRIVER_SCHEME}://{cleaned[len(prefix) :]}"
    return cleaned


def is_postgres_url(url: str) -> bool:
    """Vrai si l'URL désigne PostgreSQL (quel que soit le pilote demandé)."""
    return normalize_database_url(url).startswith("postgresql")


def validate_schema(schema: str) -> str:
    """Valide un nom de schéma destiné à ``search_path``.

    ``search_path`` n'accepte pas de paramètre lié : la valeur est forcément
    concaténée dans les options de connexion. Elle vient de la configuration
    (jamais d'une requête), et reste malgré tout restreinte à un identifiant
    simple.
    """
    cleaned = (schema or "").strip()
    if not cleaned:
        return ""
    if not _SCHEMA_RE.match(cleaned):
        raise ValueError(f"Nom de schéma refusé : {schema!r} (attendu [A-Za-z_][A-Za-z0-9_$]*)")
    return cleaned


def connect_args(
    *,
    schema: str = "",
    application_name: str = "lpo-data-ai",
    connect_timeout_s: int = 10,
) -> dict[str, object]:
    """Arguments passés au pilote PostgreSQL à l'ouverture d'une connexion."""
    args: dict[str, object] = {}
    name = (application_name or "").strip()
    if name:
        # Visible dans `SELECT * FROM pg_stat_activity` : indispensable pour
        # distinguer nos connexions de celles du reste du cluster.
        args["application_name"] = name
    if connect_timeout_s and int(connect_timeout_s) > 0:
        args["connect_timeout"] = int(connect_timeout_s)
    validated = validate_schema(schema)
    if validated:
        args["options"] = f"-c search_path={validated},public"
    return args


def session_setup_statements(
    *,
    read_only: bool,
    statement_timeout_ms: int = 15_000,
    lock_timeout_ms: int = 3_000,
    idle_in_transaction_timeout_ms: int = 60_000,
    timezone: str = "UTC",
) -> tuple[str, ...]:
    """Instructions ``SET`` appliquées à chaque nouvelle connexion.

    Elles sont volontairement explicites (et non des valeurs par défaut du
    serveur) : un pool partagé, un proxy ou un rôle configuré autrement ne
    peuvent pas les contourner par héritage — sauf à les redéfinir en session,
    ce qui reste un acte délibéré et journalisé par PostgreSQL.
    """
    for label, value in (
        ("statement_timeout_ms", statement_timeout_ms),
        ("lock_timeout_ms", lock_timeout_ms),
        ("idle_in_transaction_timeout_ms", idle_in_transaction_timeout_ms),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{label} doit être un entier positif ou nul, reçu {value!r}")
    if not re.fullmatch(r"[A-Za-z_]+(/[A-Za-z_]+)?", timezone or ""):
        raise ValueError(f"Fuseau horaire refusé : {timezone!r}")

    statements = [
        # Les libellés temporels (`to_char`) ne doivent pas dépendre du fuseau
        # du serveur, sinon une même observation change d'année selon l'hôte.
        f"SET TIME ZONE '{timezone}'",
        f"SET statement_timeout = {statement_timeout_ms}",
        f"SET lock_timeout = {lock_timeout_ms}",
        f"SET idle_in_transaction_session_timeout = {idle_in_transaction_timeout_ms}",
        # Écrit explicitement dans les deux sens : un rôle dont le défaut serait
        # `on` empêcherait sinon les scripts d'exploitation (création d'index).
        "SET default_transaction_read_only = " + ("on" if read_only else "off"),
    ]
    return tuple(statements)


def redact_url(url: str) -> str:
    """URL expurgée de son mot de passe, pour les journaux.

    Passe par le parseur de SQLAlchemy plutôt que par un découpage sur ``@`` :
    un mot de passe peut contenir ``@``, ``:`` ou des caractères encodés.
    """
    from sqlalchemy.engine import make_url

    cleaned = normalize_database_url(url)
    try:
        parsed = make_url(cleaned)
        return parsed.render_as_string(hide_password=True)
    except Exception:  # noqa: BLE001 - URL non analysable (mot de passe mal encodé)
        # Repli : on garde le schéma et l'hôte, jamais la partie identifiants.
        # `rsplit` et non `split` : un mot de passe non encodé peut contenir « @ ».
        if "://" in cleaned:
            scheme, rest = cleaned.split("://", 1)
            host = rest.rsplit("@", 1)[-1]
            return f"{scheme}://***@{host}"
        return "(URL de base invalide)"
