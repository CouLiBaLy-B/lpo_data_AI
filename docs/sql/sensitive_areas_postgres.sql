-- Schéma PostgreSQL de la table exposée par le serveur MCP.
--
-- Source : l'extrait réel du dépôt, `data/sensitive_areas.db` (169 lignes,
-- 12 colonnes, 50 zones distinctes, 18 départements, 6 régions). Ce fichier
-- fixe les types, les commentaires et les index attendus en production ; il ne
-- contient **aucune donnée** et ne modifie rien s'il est rejoué.
--
-- À exécuter par un rôle disposant de CREATE sur le schéma `lpo` (pas par le
-- rôle applicatif, qui n'a que SELECT — voir docs/02-exploitation.md §1).

CREATE SCHEMA IF NOT EXISTS lpo;

CREATE TABLE IF NOT EXISTS lpo.sensitive_areas (
    -- `id` identifie la zone, pas la ligne : une zone porte plusieurs pratiques,
    -- donc plusieurs lignes. L'unicité est portée par (id, practices), plus bas.
    id              integer,
    create_datetime timestamp,
    update_datetime timestamp,
    name            text,
    description     text,
    -- « zone reglementaire » (131 lignes) ou « Espece » (38 lignes).
    category        text,
    structure       integer,
    -- Référentiel externe : entier côté production (10), stocké en REAL dans
    -- l'extrait SQLite (10.0). NULL = zone purement réglementaire, sans espèce.
    species_id      integer,
    practices       integer,
    region          text,
    department      text,
    -- `Pays` dans l'extrait SQLite : renommé en minuscules, PostgreSQL repliant
    -- les identifiants non cités. Le profil MCP ne référence pas cette colonne.
    pays            text
);

COMMENT ON TABLE lpo.sensitive_areas IS
    'Zones sensibles (API biodiv-sports) : ni coordonnées, ni effectifs, ni noms d''espèces.';
COMMENT ON COLUMN lpo.sensitive_areas.species_id IS
    'Identifiant d''espèce du référentiel externe ; NULL pour une zone réglementaire.';
COMMENT ON COLUMN lpo.sensitive_areas.category IS
    'Nature de la zone (zone réglementaire ou liée à une espèce).';
COMMENT ON COLUMN lpo.sensitive_areas.create_datetime IS
    'Date de création de la zone — pas une date d''observation.';

-- Grain réel de l'extrait : une ligne par (zone, pratique). Idempotent, sans
-- CONCURRENTLY (contrainte d'unicité, exécutée à la création du schéma).
CREATE UNIQUE INDEX IF NOT EXISTS uq_sensitive_areas_zone_practice
    ON lpo.sensitive_areas (id, practices);

-- Index alignés sur `scripts/create_indexes.py` (mêmes noms : le script les
-- reconnaît comme déjà présents et ne les recrée pas). CONCURRENTLY permet de
-- les poser sur une table déjà chargée et lue par l'ETL.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sensitive_areas_event_date
    ON lpo.sensitive_areas (create_datetime);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sensitive_areas_species
    ON lpo.sensitive_areas (species_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sensitive_areas_species_date
    ON lpo.sensitive_areas (species_id, create_datetime);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sensitive_areas_area
    ON lpo.sensitive_areas (department, region);

-- Droits : le rôle applicatif lit, il n'écrit pas.
-- GRANT USAGE ON SCHEMA lpo TO lpo_ro;
-- GRANT SELECT ON lpo.sensitive_areas TO lpo_ro;
-- REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON lpo.sensitive_areas FROM lpo_ro;

-- Chargement depuis l'extrait (colonnes arrivant en texte, d'où les casts).
-- Les spécialités `\copy` sont des commandes psql, pas du SQL :
--
--   CREATE TABLE staging_sensitive_areas (LIKE lpo.sensitive_areas);
--   ALTER TABLE staging_sensitive_areas ALTER COLUMN id TYPE text,
--       ALTER COLUMN practices TYPE text, ALTER COLUMN structure TYPE text,
--       ALTER COLUMN species_id TYPE text,
--       ALTER COLUMN create_datetime TYPE text,
--       ALTER COLUMN update_datetime TYPE text;
--   \copy staging_sensitive_areas FROM 'sensitive_areas.csv' WITH (FORMAT csv, HEADER true, NULL '')
--   INSERT INTO lpo.sensitive_areas
--   SELECT NULLIF(id, '')::integer,
--          NULLIF(create_datetime, '')::timestamp,
--          NULLIF(update_datetime, '')::timestamp,
--          name, description, category,
--          NULLIF(structure, '')::integer,
--          NULLIF(species_id, '')::numeric::integer,
--          NULLIF(practices, '')::integer,
--          region, department, pays
--   FROM staging_sensitive_areas;
--   DROP TABLE staging_sensitive_areas;
--
-- Vérification attendue après chargement (mêmes valeurs que l'extrait) :
--   SELECT count(*) FROM lpo.sensitive_areas;                       -- 169
--   SELECT count(DISTINCT id) FROM lpo.sensitive_areas;             -- 50
--   SELECT count(*) FROM lpo.sensitive_areas WHERE species_id IS NULL; -- 131
