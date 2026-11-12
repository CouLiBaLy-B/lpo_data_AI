#!/usr/bin/env python3
"""Génère un jeu de démonstration d'observations naturalistes (**synthétique**).

Pourquoi ce script ?
--------------------
Le dépôt ne contient que la table `sensitive_areas` (zones réglementaires issues
de l'API biodiv-sports), sans espèce nommée, sans effectif, sans coordonnées et
sans statut de protection. La cible fonctionnelle (observations naturalistes)
n'est donc pas représentable sur ces données : ce script crée une table
`observations` au schéma cible pour permettre une démonstration et des tests de
bout en bout.

**Aucune valeur écologique réelle.** Les lignes sont générées
pseudo-aléatoirement (graine fixe) à partir de noms d'espèces courants et de
boîtes englobantes départementales approximatives. La surface produite le
signale explicitement (« jeu de démonstration »).

Usage :
    python scripts/seed_demo_db.py                       # data/lpo_demo.db, 2400 lignes
    python scripts/seed_demo_db.py --rows 500 --db /tmp/demo.db
"""

from __future__ import annotations

import argparse
import math
import random
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "lpo_demo.db"

#: (nom, nom scientifique, protection, sensible, poids relatif, tendance)
SPECIES: list[tuple[str, str, str, bool, float, float]] = [
    ("Milan royal", "Milvus milvus", "protégée", True, 0.9, 0.25),
    ("Busard cendré", "Circus pygargus", "protégée", True, 0.6, -0.1),
    ("Faucon crécerelle", "Falco tinnunculus", "protégée", False, 1.4, 0.05),
    ("Buse variable", "Buteo buteo", "protégée", False, 1.8, 0.1),
    ("Héron cendré", "Ardea cinerea", "protégée", False, 1.1, 0.15),
    ("Cigogne blanche", "Ciconia ciconia", "protégée", False, 0.7, 0.35),
    ("Chouette effraie", "Tyto alba", "protégée", True, 0.5, -0.05),
    ("Martin-pêcheur d'Europe", "Alcedo atthis", "protégée", True, 0.55, 0.0),
    ("Crapaud commun", "Bufo bufo", "protégée", False, 1.2, -0.15),
    ("Triton palmé", "Lissotriton helveticus", "protégée", True, 0.45, -0.2),
    ("Chevreuil européen", "Capreolus capreolus", "non protégée", False, 1.6, 0.2),
    ("Renard roux", "Vulpes vulpes", "non protégée", False, 1.0, 0.1),
    ("Sanglier", "Sus scrofa", "non protégée", False, 0.9, 0.3),
    ("Lièvre d'Europe", "Lepus europaeus", "non protégée", False, 0.6, -0.25),
]

#: département → (lat_min, lat_max, lon_min, lon_max, région)
DEPARTMENTS: dict[str, tuple[float, float, float, float, str]] = {
    "Finistère": (47.75, 48.72, -4.80, -3.40, "Bretagne"),
    "Côtes-d'Armor": (48.00, 48.90, -3.70, -1.90, "Bretagne"),
    "Ille-et-Vilaine": (47.60, 48.65, -2.10, -1.00, "Bretagne"),
    "Morbihan": (47.30, 48.10, -3.40, -1.90, "Bretagne"),
    "Loire-Atlantique": (46.90, 47.80, -2.30, -1.00, "Pays de la Loire"),
    "Vendée": (46.25, 47.10, -2.40, -0.70, "Pays de la Loire"),
    "Deux-Sèvres": (46.10, 47.10, -0.75, 0.10, "Nouvelle-Aquitaine"),
    "Charente-Maritime": (45.10, 46.30, -1.20, 0.20, "Nouvelle-Aquitaine"),
    "Gironde": (44.20, 45.55, -1.25, 0.30, "Nouvelle-Aquitaine"),
    "Loiret": (47.55, 48.15, 1.55, 2.75, "Centre-Val de Loire"),
    "Indre-et-Loire": (46.95, 47.75, 0.05, 1.30, "Centre-Val de Loire"),
    "Bas-Rhin": (48.10, 49.10, 6.80, 8.20, "Grand Est"),
    "Hérault": (43.30, 43.95, 2.50, 4.10, "Occitanie"),
    "Bouches-du-Rhône": (43.20, 43.95, 4.60, 5.80, "Provence-Alpes-Côte d'Azur"),
}

START = date(2016, 1, 1)
END = date(2026, 10, 31)
SOURCES = (
    "LPO — réseau d'observateurs (démo)",
    "LPO — atlas communal (démo)",
    "LPO — base de démonstration",
)


def _seasonal_weight(day: date) -> float:
    """Pic d'observabilité au printemps (migration / reproduction)."""
    angle = 2 * math.pi * (day.timetuple().tm_yday - 100) / 365.0
    return 0.55 + 0.75 * max(0.0, math.cos(angle)) + 0.25 * max(0.0, math.cos(angle + math.pi))


def build_rows(rows: int, seed: int) -> list[tuple]:
    """Construit les lignes de la table `observations` (déterministe)."""
    rng = random.Random(seed)
    dept_names = list(DEPARTMENTS)
    weights = [1.6 if d in {"Finistère", "Gironde", "Hérault"} else 1.0 for d in dept_names]
    total_days = (END - START).days
    out: list[tuple] = []
    for index in range(1, rows + 1):
        name, latin, protection, sensitive, base_weight, trend = rng.choices(
            SPECIES, weights=[s[4] for s in SPECIES], k=1
        )[0]
        year = rng.randint(START.year, END.year)
        year_frac = (year - START.year) / max(1, (END.year - START.year))
        trend_factor = max(0.15, 1 + trend * year_frac * 2)

        # Échantillonnage temporel biaisé par la saison
        while True:
            day = START + timedelta(days=rng.randint(0, total_days))
            if day.year != year:
                continue
            if rng.random() < min(1.0, _seasonal_weight(day) * trend_factor / 1.4):
                break

        dept = rng.choices(dept_names, weights=weights, k=1)[0]
        lat_min, lat_max, lon_min, lon_max, region = DEPARTMENTS[dept]
        lat = round(rng.uniform(lat_min, lat_max), 3)
        lon = round(rng.uniform(lon_min, lon_max), 3)
        count = max(1, int(rng.lognormvariate(1.1, 0.7)))
        out.append(
            (
                index,
                name,
                latin,
                day.isoformat(),
                count,
                lat,
                lon,
                dept,
                region,
                protection,
                1 if sensitive else 0,
                rng.choice(SOURCES),
            )
        )
    out.sort(key=lambda r: r[3])
    return out


SCHEMA = """
DROP TABLE IF EXISTS observations;
CREATE TABLE observations (
    id INTEGER PRIMARY KEY,
    species TEXT NOT NULL,
    latin_name TEXT,
    event_date TEXT NOT NULL,          -- ISO AAAA-MM-JJ (SQLite)
    count INTEGER,
    lat REAL,
    lon REAL,
    department TEXT,
    region TEXT,
    protection_status TEXT,
    sensitive INTEGER,                 -- 0/1
    source TEXT
);
CREATE INDEX idx_obs_species ON observations(species);
CREATE INDEX idx_obs_date ON observations(event_date);
CREATE INDEX idx_obs_dept ON observations(department);

DROP TABLE IF EXISTS dataset_metadata;
CREATE TABLE dataset_metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def seed(db_path: Path, *, rows: int = 6000, seed: int = 20261112) -> Path:
    """Crée (ou recrée) la base de démonstration et renvoie son chemin."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    data = build_rows(rows, seed)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO observations (id, species, latin_name, event_date, count, lat, lon, "
            "department, region, protection_status, sensitive, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            data,
        )
        conn.executemany(
            "INSERT INTO dataset_metadata (key, value) VALUES (?, ?)",
            [
                ("label", "Jeu de démonstration synthétique (observations naturalistes)"),
                ("is_demo", "true"),
                ("license", "Données synthétiques — réutilisation libre, aucune valeur écologique"),
                ("seed", str(seed)),
                ("rows", str(len(data))),
                ("generated_at", datetime.now(tz=timezone.utc).date().isoformat()),
                ("notes", "Généré par scripts/seed_demo_db.py — ne pas citer comme données LPO."),
            ],
        )
        conn.commit()
    return db_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Génère la base de démonstration LPO.")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Chemin SQLite cible.")
    parser.add_argument("--rows", type=int, default=6000, help="Nombre d'observations.")
    parser.add_argument("--seed", type=int, default=20261112, help="Graine pseudo-aléatoire.")
    args = parser.parse_args()

    path = seed(Path(args.db), rows=args.rows, seed=args.seed)
    with sqlite3.connect(path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        span = conn.execute("SELECT MIN(event_date), MAX(event_date) FROM observations").fetchone()
    print(f"✓ {total} observations synthétiques écrites dans {path}")
    print(f"  Période : {span[0]} → {span[1]}")
    print("  ⚠ Données de démonstration : aucune valeur écologique réelle.")


if __name__ == "__main__":
    main()
