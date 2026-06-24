---
name: sql-lpo
description: Glossaire métier LPO et stratégie de JOIN pour les bases de zones sensibles (biodiversité). À lire quand la question porte sur les zones sensibles, espèces, structures, régions, ou nécessite des jointures multi-tables.
---

# Contexte métier LPO — zones sensibles

## Glossaire des concepts

- **Zone sensible** : périmètre géographique à enjeu de biodiversité (souvent table
  `sensitive_areas` ou équivalent). Peut être lié à une espèce ou être réglementaire.
- **category** : typiquement `'Espece'` (zone liée à une espèce) ou
  `'zone reglementaire'`. Quand la colonne `category` n'existe pas, la distinction
  se déduit de `species_id` : `NULL` ⇒ zone réglementaire, sinon ⇒ espèce.
- **structure** : organisation responsable de la donnée (parc, association, LPO…).
- **practices** : activités sportives concernées (randonnée, escalade, VTT…).
- **region / departement / pays** : localisation administrative française.

## Stratégie multi-tables

1. Ne présume jamais du schéma : `list_tables` puis `describe_tables` d'abord.
2. Pour relier deux tables, utilise les **clés étrangères** retournées par
   `describe_tables` (section `FK ... → table(col)`). Exemple de motif :
   `... FROM zones z JOIN especes e ON z.species_id = e.id ...`.
3. Si une dimension (espèce, structure) est dans une table séparée, joins-la pour
   récupérer son libellé plutôt que d'afficher un identifiant brut.

## Normalisation des libellés

- Filtre les valeurs vides : `WHERE region IS NOT NULL AND region != ''`.
- Regroupe en tenant compte de la casse/espaces si nécessaire
  (ex. `GROUP BY TRIM(region)`).

## Agrégations temporelles

- SQLite : `strftime('%Y-%m', create_datetime)` pour grouper par mois.
- PostgreSQL : `to_char(create_datetime, 'YYYY-MM')`.

## Exemples de référence

```sql
-- Zones par catégorie
SELECT category, COUNT(*) AS nb FROM sensitive_areas GROUP BY category ORDER BY nb DESC;

-- Top régions (libellés propres)
SELECT region, COUNT(*) AS nb FROM sensitive_areas
WHERE region IS NOT NULL AND region != '' GROUP BY region ORDER BY nb DESC LIMIT 10;

-- Déduction de catégorie si la colonne n'existe pas
SELECT CASE WHEN species_id IS NULL THEN 'zone reglementaire' ELSE 'Espece' END AS categorie,
       COUNT(*) AS nb
FROM sensitive_areas GROUP BY categorie;
```
