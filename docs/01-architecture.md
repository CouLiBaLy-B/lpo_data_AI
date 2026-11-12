# Architecture — `lpo_data_AI` × MCP × A2UI

> Toute affirmation sur A2UI ci-dessous renvoie à la page consultée :
> [`a2ui.org/specification/v1.0-a2ui/`](https://a2ui.org/specification/v1.0-a2ui/),
> [`a2ui.org/concepts/overview/`](https://a2ui.org/concepts/overview/) et
> [`a2ui.org/concepts/messages/`](https://a2ui.org/concepts/messages/), lues le
> **2026-11-12**. Version retenue : **A2UI v1.0** (spec publiée le 2025-11-20,
> mise à jour le 2026-06-08 ; v0.9.1 est la version précédente).

---

## 1. Vue d'ensemble

```
┌────────────────────────────┐
│ Utilisateur                │  question en français
└─────────────┬──────────────┘
              ▼
┌────────────────────────────────────────────────────────────┐
│ Client A2UI — web/static/                                   │
│  • index.html / app.js   : dialogue + envoi des actions     │
│  • renderer.js           : renderer du catalogue LPO        │
│  • Aucun HTML/CSS/JS ne vient du serveur                    │
└─────────────┬──────────────────────────────────────────────┘
              │ HTTP/JSON (REST) — messages A2UI v1.0 + texte de repli
              ▼
┌────────────────────────────────────────────────────────────┐
│ Agent orchestrateur — orchestrator/                        │
│  intent.py      : intention (heuristique ou LLM)            │
│  mcp_client.py  : client MCP (SDK officiel)                 │
│  views.py       : données MCP → modèle de vue              │
│  agent.py       : boucle question → outils → A2UI           │
└─────────────┬──────────────────────────────────────────────┘
              │ MCP (JSON-RPC) — stdio par défaut, streamable-http en option
              ▼
┌────────────────────────────────────────────────────────────┐
│ Serveur MCP `lpo-data` — lpo_data_mcp/                     │
│  server.py  : 5 outils typés, lecture seule                 │
│  service.py : agrégations, pagination, provenance           │
│  sql.py     : SQL paramétré + liste blanche d'identifiants  │
│  privacy.py : floutage des espèces sensibles                │
│  profiles.py: capacités du schéma exposé                    │
└─────────────┬──────────────────────────────────────────────┘
              ▼
        SQLAlchemy (lecture seule) → SQLite / PostgreSQL
```

La couche A2UI est dans `lpo_a2ui/` : catalogue JSON, gabarits de surface,
validateur.

---

## 2. Décisions d'architecture et justifications

### 2.1 SDK MCP : `mcp` (Python officiel) 2.3.0

- **Choix** : `MCPServer` (`mcp.server.mcpserver`) avec décorateurs `@mcp.tool`
  et sorties **Pydantic** (donc `structuredContent` + `outputSchema` générés).
- **Pourquoi** : c'est l'implémentation de référence, elle gère les deux
  transports voulus (`stdio`, `streamable-http`), la négociation de capacités et
  la sérialisation des sorties structurées. Aucun besoin de réinventer JSON-RPC.
- **Piège rencontré** : sur la 2.x, `mcp.server.fastmcp.FastMCP` n'existe plus
  (`ModuleNotFoundError`) — l'API est `MCPServer`. De même côté client,
  `mcp.Client(...)` remplace les anciens helpers `streamablehttp_client`.
- **Conséquence de conception** : chaque outil renvoie **données structurées +
  `summary`** (résumé texte en français) + `meta` (profil, avertissements,
  niveau de confidentialité, troncature). L'agent n'a donc jamais besoin de
  parser du texte libre pour récupérer un chiffre.

### 2.2 Transport MCP : `stdio` par défaut, `streamable-http` en option

- **Choix** : `stdio` (sous-processus) en local/démo ; `--transport
  streamable-http --port 8765` pour un déploiement réseau.
- **Pourquoi** : `stdio` supprime toute question d'authentification et de
  surface réseau pour une démonstration mono-machine, tout en gardant le
  serveur réellement séparé du processus appelant (pas d'import direct). Le
  client (`orchestrator/mcp_client.py`) choisit via `LPO_MCP_URL` (HTTP) ou
  `LPO_MCP_COMMAND` (stdio), et garde une **session persistante** protégée par
  un verrou asynchrone.
- **Vérifié** à deux niveaux :
  - `tests/test_mcp_server.py` lance un vrai sous-processus serveur via
    `mcp.Client(StdioServerParameters(...))` et vérifie noms d'outils,
    `structured_content` et le comportement d'erreur ;
  - `tests/test_mcp_client.py` exerce le **client de production**
    (`orchestrator/mcp_client.py`) sur ce même transport : connexion paresseuse,
    transmission de l'environnement au sous-processus, propagation du message
    d'erreur, fermeture idempotente et reconnexion.
- **Intégration complète** : `tests/test_web_e2e.py` ne substitue rien — le
  `lifespan` de FastAPI démarre le sous-processus MCP et les requêtes HTTP
  traversent HTTP → agent → client MCP → serveur → SQLite.

### 2.3 Renderer A2UI : renderer JavaScript autonome (`web/static/renderer.js`)

- **Constat** : Streamlit (front historique du dépôt) **n'est pas** un renderer
  A2UI : il n'existe pas de binding A2UI pour Streamlit.
- **Choix** : un renderer en JavaScript pur (~600 lignes), sans dépendance ni
  build, qui implémente **exactement** les 14 composants du catalogue LPO et
  rien d'autre.
- **Pourquoi** : le protocole exige que le renderer soit « verrouillé » sur un
  catalogue ; un renderer minimal garantit qu'aucun composant hors catalogue ne
  peut s'afficher, et il sert de **preuve d'implémentation** pour les composants
  métier (graphiques, carte) absents du catalogue de base. Il est aussi
  facilement remplaçable par un renderer React/Lit officiel puisque seule la
  liste des composants change.
- **Sécurité** : le renderer n'utilise **jamais** `innerHTML` pour du contenu
  serveur ; tout passe par `textContent` et par des éléments SVG construits par
  `createElementNS`. Il n'évalue aucun code, n'accepte ni HTML, ni CSS, ni JS
  fournis par l'agent.

### 2.4 Interface web : sobriété contrainte par des tests

L'interface (`web/static/index.html`, `styles.css`, `app.js`) est un outil
d'analyse, pas une vitrine : elle doit rendre les chiffres vérifiables et rester
lisible sur un écran de terrain.

- **Un seul accent** (vert profond `#1b5e3f`) et des états sémantiques (ambre,
  rouge) réservés aux alertes. Pas de dégradé, pas d'ombre portée : les
  séparations sont des filets d'un pixel.
- **Échelle typographique fixe** (11 / 12 / 12,5 / 13,5 / 14 / 24 px) et pile
  système — aucune police distante n'est chargée, donc aucun FOUT ni dépendance
  réseau supplémentaire.
- **États explicites** : chargement (barre fine + `aria-busy`), erreur (encart
  `role="alert"` avec la raison, sans chiffre), vide (message du composant
  `Notice` ou `#placeholder`), mode texte seul (repli sans renderer).
- **Accessibilité** : lien d'évitement, `role="log"` + `aria-live` sur la
  conversation, libellé de champ réel, `:focus-visible` à 2 px jamais supprimé,
  contrastes AA vérifiés, animations conditionnées à `prefers-reduced-motion`.
- **Traçabilité visible** : chaque tour expose un `<details>` listant les appels
  MCP (outil, arguments, durée, statut) afin qu'un chiffre affiché puisse être
  reconduit à l'appel qui l'a produit.

Ces propriétés sont mesurables, donc testées : `tests/test_frontend_quality.py`
vérifie les ratios de contraste des jetons réellement employés (texte ≥ 4,5:1,
bordures de contrôles ≥ 3:1), l'absence de dégradés, d'ombres, d'emoji
d'identité, de capitales espacées et de polices fantômes, la présence du focus
visible et de `prefers-reduced-motion`, la cohérence des identifiants lus par
`app.js`, et le fait que chaque composant du catalogue possède un style.

Le rendu reste vérifié sans navigateur : `lpo_a2ui/examples/render_snapshot.mjs`
rejoue les messages A2UI sur le renderer de production dans jsdom et échoue si
un composant n'est pas rendu.

L'interface a en outre été auditée avec **axe-core** dans Chromium, sur sept
états (accueil, tendance, carte, réponse hors périmètre, mode texte seul, en
1440 px et en 390 px) : **aucune violation**. Deux défauts réels ont été
corrigés à cette occasion — les zones défilantes n'étaient pas atteignables au
clavier, et la page entière défilait au lieu de ses panneaux.

Enfin, les graphiques ne sont pas des SVG étirés : `drawLineChart` et `drawMap`
reçoivent la largeur mesurée du conteneur et redessinent au redimensionnement.
Un SVG mis à l'échelle du cadre grossit aussi ses libellés d'axes (16 px sur
grand écran, 5 px sur téléphone) ; le redessin garde une typographie constante.

### 2.5 Format d'inférence et validation : JSON canonique + validation systématique

Le pipeline décrit par [a2ui.org/concepts/overview](https://a2ui.org/concepts/overview/)
est : `catalog.json` → « skill » A2UI pour le modèle → appels d'outils → le
**compiler** du SDK valide et émet `updateComponents`/`updateDataModel`.

Ici, le « compiler » est **déterministe** : l'agent n'écrit pas de JSON A2UI à
main levée, il remplit des **gabarits** (`lpo_a2ui/templates.py`) dont la
structure est figée. Le LLM, quand il est activé, ne produit que l'**intention**
(sortie structurée Pydantic) — jamais un composant.

Validation en trois couches (`lpo_a2ui/validation.py`), **avant envoi** :

| Couche | Ce qui est vérifié | Outil |
|---|---|---|
| 1. Schéma + catalogue | enveloppe `version`/clé unique, champs requis, composants et propriétés déclarés, types `Dynamic*` | SDK `a2ui-agent-sdk` (`validate_payload`) |
| 2. Séquences | identifiants uniques, cycle de vie des surfaces, topologie (`root` présent, enfants existants) | `RendererState` du SDK |
| 3. Contrat LPO | aucune mesure littérale dans un composant (tout doit être lié), listes cohérentes | `_check_lpo_contract` |

Si la validation échoue, l'agent **ne renvoie pas d'interface** : il renvoie le
texte de repli et marque `degraded: true` (testé :
`tests/test_orchestrator.py::test_degraded_mode_when_a2ui_is_invalid`).

---

## 3. Conformité au protocole A2UI v1.0

### 3.1 Messages utilisés

La spec v1.0 impose : un objet JSON avec `version: "1.0"` et **une seule** clé de
message
([spec v1.0](https://a2ui.org/specification/v1.0-a2ui/)).

| Message | Usage ici |
|---|---|
| `createSurface` | une surface par réponse (`surfaceId`, `catalogId`, `components`, `dataModel`, `sendDataModel: true`) |
| `updateDataModel` | **toutes** les interactions de filtre : données seules, surface conservée |
| `deleteSurface` | uniquement quand la visualisation change de nature (ex. `trend` → `ranking`), ou en repli si le renderer a perdu la surface |
| `updateComponents` | non utilisé : la structure est immuable pour un `kind` donné |

`createSurface` porte `components` **et** `dataModel`, ce que la spec autorise
explicitement (instanciation en un message ; le data model est appliqué avant
les composants). Cela donne une surface atomique : un seul message, donc aucun
état intermédiaire affichable.

Points de conformité vérifiés et imposés :

- `id: "root"` obligatoire dans chaque surface.
- `surfaceId` globalement unique : toute recréation est précédée d'un
  `deleteSurface` (sinon `INTEGRITY_ERROR`).
- `theme` **interdit** dans `createSurface` en v1.0 (le style appartient au
  thème natif du renderer) — le catalogue LPO ne le déclare pas.
- `@catalogId` par composant **interdit** en v1.0 — le catalogue est unique par
  surface.
- `sendDataModel` est une propriété de `createSurface` ; le data model complet
  revient au renderer dans les métadonnées, il n'est pas redéclaré par composant.
- `@accessibility` **refusé** par v1.0 : le SDK lève `A2uiCatalogError`. Écarté
  définitivement ; l'accessibilité passe par le thème du renderer (rôles ARIA
  posés par le renderer lui-même) et par le résumé texte.
- Bindings **typés** `DynamicString/Number/Boolean/StringList` : soit un
  littéral (libellés d'interface), soit `{"@path": "/..."}` (JSON Pointer dans
  le data model). `{"@call": ...}` est accepté par le protocole mais **non
  implémenté** par le renderer LPO : il est journalisé dans `renderer.errors`.
- `ChildList` : liste statique d'identifiants (utilisée partout, la structure
  étant fixe).

### 3.2 Séparation structure / données

C'est l'exigence centrale : **la structure ne contient aucune donnée**.

```
components (structure, immuable)          dataModel (données, mutable)
──────────────────────────────────        ────────────────────────────────
{ "id": "chart",                          { "header": {"title": "…"},
  "component": "LineChart",                 "kpis":   [ … ],
  "title":  {"@path": "/chart/title"},      "chart":  {"series": [ … ],
  "series": {"@path": "/chart/series"},               "xLabel": "Année"},
  "xLabel": {"@path": "/chart/xLabel"} }              "…" }
```

- Un changement de filtre produit **uniquement** des `updateDataModel`
  (chemins `/chart/series`, `/kpis`, `/header/title`, `/filters/...`) : la
  requête HTTP contient des messages dont l'unique clé est `updateDataModel`
  (testé : `tests/test_orchestrator.py::test_filters_update_data_model_without_recreating_surface`).
- Le validateur refuse toute mesure littérale : si un composant de graphique
  contient des nombres en clair au lieu d'un `@path`, la validation échoue
  (`tests/test_a2ui_templates.py::test_literal_measures_are_rejected_by_lpo_contract`).

### 3.3 Où sont les interactions ?

Le catalogue LPO déclare un composant `SelectField` avec une propriété
`action` : un select émet une action `{name: "set_filter", context: {filter,
value}}` vers l'agent. Conformément au protocole, **le renderer ne filtre
rien** : il renvoie l'action, l'agent rappelle le serveur MCP, puis met à jour
le data model de la **surface existante**.

Un élément décoratif ne porte jamais d'action : `MapChart` est en lecture seule
(pas de zoom serveur), les cartes sont rendues côté client à partir des points
du data model.

---

## 4. Catalogue A2UI personnalisé

Le catalogue de base (`basic_catalog_v1.json`, vendorisé depuis
`a2ui-project/a2ui`) ne contient **ni graphique, ni carte**. La spec impose de
déclarer un catalogue si l'on veut d'autres composants : c'est ce qui est fait.

`catalogId` :
`https://lpo-data-ai.example/a2ui/catalogs/lpo/v1/catalog.json`

| Composant | Type | Rôle | Exposition serveur |
|---|---|---|---|
| `Text` | base | libellés, paragraphes | littéral ou `@path` |
| `Row`, `Column`, `List` | base | mise en page (`ChildList`) | structure |
| `Card` | base | conteneur d'un bloc | structure |
| `Button` | base | action secondaire (`action`) | structure |
| `Divider` | base | séparateur | structure |
| `KpiCard` | métier | indicateur clé (libellé, valeur, indice) | valeur liée |
| `LineChart` | métier | tendance, comparaison multi-séries | `series` lié |
| `BarChart` | métier | classement / top N | `items` lié |
| `MapChart` | métier | carte de points (fond GeoJSON local) | `points` lié |
| `DataTable` | métier | détail tabulaire | `rows` lié |
| `Notice` | métier | avertissement, état vide, confidentialité | littéral + `@path` |
| `SelectField` | métier | filtre → action `set_filter` | options liées, `value` lié |

Le catalogue complet (JSON Schema par composant, `$defs`, types `Dynamic*`) est
dans `lpo_a2ui/catalog/lpo_catalog.json` et **servi au client** par
`GET /api/catalog` : le client et l'agent partagent donc littéralement le même
fichier.

### 4.1 Les trois modèles d'interface demandés

| Modèle | `kind` | Composition |
|---|---|---|
| **Tendance** | `trend` | en-tête + KPI + `LineChart` (1..n séries) + `DataTable` de détail + `Notice` |
| **Classement** | `ranking` | en-tête + KPI + `BarChart` horizontal + `DataTable` (part %) + `Notice` |
| **Carte** | `map` | en-tête + KPI + `MapChart` + `Notice` de confidentialité + `DataTable` (mailles) |

Deux surfaces supplémentaires couvrent les cas restants : `species` (fiche
espèce) et `empty` (état vide explicite). Toutes les surfaces embarquent la
barre de filtres (`species` / `period` / `area`) et un pied de page de
provenance.

---

## 5. Chaîne de traitement d'une question

```
1. Question ─► intent.py
      heuristique déterministe (défaut) ou LLM en sortie structurée
      → {kind, species[], start/end, department/region, metric, granularity,
         rank_by}  (axe d'un classement : species | department | region | category)
      → hors périmètre détecté → kind = "unknown"

2. Contexte (une fois par assistant)
      get_dataset_overview  → profil + capacités + volumétrie
      list_species(200)     → vocabulaire exact des libellés
      (si un outil échoue, repli documenté sur species_sample)

3. Appels MCP (selon kind)
      trend      → aggregate_observations(group_by=year|month, metric, filtres)
      ranking    → aggregate_observations(group_by=rank_by, limit=10)
                   rank_by = species (défaut) | department | region | category
      map        → aggregate_observations(group_by=grid, ...)
      comparison → aggregate_observations(group_by=species_month) puis agrégation
                   par libellé en séries distinctes
      species    → get_species_info + série mensuelle

4. views.py : données MCP → modèle de vue (jamais l'inverse)
      KPI calculés à partir des totaux renvoyés par le serveur
      état vide explicite si capacité manquante ou résultat nul

5. templates.py : modèle de vue → surface A2UI (createSurface)
      entêtes de tableau = structure, donc calculés ici (`table_columns`)
      à partir de l'unité MCP (« enregistrements ») et de l'axe de classement ;
      les valeurs, elles, restent dans le dataModel

6. validation.py : schéma + catalogue + séquences + contrat LPO
      échec → réponse texte seule, degraded = true

7. Réponse HTTP : {text, kind, surfaceId, mode, messages[], trace[], validation}
```

### 5.1 Garantie « aucun chiffre inventé »

- L'agent ne calcule **aucune** mesure métier : totaux, parts, minimum/maximum
  et troncatures viennent du serveur MCP (`summary`, `total`, `share`,
  `meta.truncated`).
- L'intention ne comporte **aucun champ numérique** (testé :
  `test_intent_contains_no_measure_field`).
- Les textes de réponse sont **formatés** à partir de ces mêmes valeurs
  (`fmt_int`, `fmt_pct`) ; le test
  `test_every_number_in_text_comes_from_mcp` vérifie la cohérence
  texte/données/total serveur.
- Le LLM, s'il est activé, ne voit **jamais** les données : il ne reçoit que la
  question, la date du jour et le vocabulaire.

### 5.2 États vides et ambiguïtés

Quatre cas produisent une surface `empty` ou une `Notice`, jamais une donnée
inventée :

1. question hors périmètre (`kind = "unknown"`) ;
2. capacité absente du profil (ex. pas de coordonnées → pas de carte) ;
3. agrégat vide (filtre trop restrictif) ;
4. libellé cité mais absent du jeu de données (l'agent refuse et liste les
   libellés réels).

Le point 4 mérite un détail, parce qu'il a été appris sur l'extrait réel : une
question sur le « Milan royal » adressée à `sensitive_areas` produisait une
courbe de 169 lignes — toutes les zones, dont 131 sans espèce — sous le titre
« Milan royal ». Les chiffres venaient bien de la base, la réponse était fausse.
Désormais, une mention capitalisée introduite par « de/du/des/d'/l' » qui ne
correspond ni au vocabulaire d'espèces ni aux zones connues est portée par
`Intent.unknown_species` ; l'agent **n'appelle alors aucun outil** et affiche un
état vide explicite. Le repli par mot distinctif ne s'applique plus aux
qualificatifs (« royal »), qui faisaient correspondre « Aigle royal » à « Milan
royal ».

### 5.3 Prompt système et few-shots A2UI

`orchestrator/prompts.py` contient :

- `SYSTEM_PROMPT` — les invariants de l'agent (aucun chiffre inventé, outils
  typés, catalogue fermé cité par son identifiant, séparation structure/données,
  actions → agent, repli texte, confidentialité) **et** la table de décision
  « intention → type de vue → agrégation » ;
- `FEW_SHOTS` — six exemples **complets et valides** (tendance, classement,
  carte, comparaison à deux séries, fiche espèce, hors périmètre). Chaque
  exemple donne la question, l'intention, les appels MCP exacts, les messages
  A2UI et le résumé texte.

Deux points de méthode :

1. **Les few-shots sont produits par les gabarits de production**
   (`lpo_a2ui.templates`), pas écrits à la main. Ils ne peuvent donc pas
   diverger du catalogue : `tests/test_prompts.py` rejoue chaque exemple dans
   le validateur (schéma + catalogue + séquences) et vérifie qu'aucune mesure
   n'est en clair dans la structure.
2. **Périmètre actuel** : le LLM ne sert aujourd'hui qu'à l'extraction
   d'intention (`intent.py`, sortie structurée). La construction des surfaces
   reste déterministe (`views.py` + `templates.py`), ce qui est plus sûr que de
   laisser un modèle écrire du JSON A2UI. `prompts.py` est donc la
   **spécification exécutable** du comportement attendu et le matériel prêt pour
   un passage à une orchestration LLM, sans changement de catalogue ni de
   validation.

Visualiser le prompt complet (système + exemples) :

```bash
python -m orchestrator --prompt
```

---

### 5.4 Réglages d'exécution : modèle et cible MCP

Les deux dépendances externes de l'agent (le fournisseur de modèle et le serveur
MCP) sont décrites par `orchestrator/settings.py` — un couple de dataclasses
immuables, validées et sérialisables :

```
RuntimeSettings
├── LlmSettings  : mode (auto|on|off), provider (liste fermée), model, clé (repr=False)
└── McpSettings  : transport (stdio|http), url, command, args, cwd
```

- **Deux sources, une seule vérité** : `from_env()` lit `.env` au démarrage ;
  `POST /api/settings` produit des réglages `source="ui"` pour la durée du
  processus. `{"reset": "env"}` revient aux valeurs d'environnement — un
  redémarrage suffit donc toujours à retrouver un état connu.
- **La cible MCP n'est adoptée que si elle répond** : le client candidat est
  connecté et interrogé (`get_dataset_overview`) avant de remplacer l'ancien.
  En cas d'échec, l'ancienne cible reste en place et l'API renvoie 502 avec le
  message d'erreur expurgé.
- **Un changement de cible vide les conversations** : leurs surfaces décrivent
  un autre jeu de données. Un changement de modèle, lui, conserve le contexte
  MCP (vocabulaire d'espèces, départements) : `LpoAssistant.configure_llm()`
  reconstruit le parseur sans re-questionner les outils.
- **Panne au démarrage = mode dégradé, pas crash** : si le MCP ne répond pas,
  l'application démarre quand même, `/api/health` publie `mcp.ok=false` et
  l'interface invite à ouvrir *Configuration*. C'est le seul moyen de corriger
  une cible erronée depuis le navigateur.
- **Le modèle reste optionnel** : la fabrique de modèle est importée
  paresseusement. Sans `langchain` ou sans clé, l'extraction d'intention
  retombe sur l'heuristique — `POST /api/settings/test` le dit explicitement
  (« dépendances du modèle absentes… ») au lieu d'échouer silencieusement.

Les entêtes utilisées : `GET /api/settings` (état expurgé), `POST /api/settings`
(patch), `POST /api/settings/test` (sonde sans enregistrement).

---

## 6. Sécurité et confidentialité

| Risque | Mesure | Emplacement |
|---|---|---|
| SQL arbitraire | aucun outil n'accepte de SQL ; `group_by`/`metric` sont des énumérations ; identifiants issus du profil et validés par liste blanche | `lpo_data_mcp/sql.py` |
| Injection | requêtes **paramétrées** (`run_select_params`), valeurs liées | `sql.py`, `src/database/engine.py` |
| Écriture | `assert_read_only` (analyse de la requête) + moteur d'exploitation en écriture jamais chargé + `default_transaction_read_only = on` posé à chaque connexion PostgreSQL | `sql.py`, `src/database/postgres.py` |
| Déni de service | `limit` plafonné à 500, agrégats bornés, pagination explicite | `sql.py` |
| Fuite de localisation | floutage à la maille ~11 km par défaut pour les espèces sensibles ; repli au département ; mode exact seulement si `LPO_ALLOW_EXACT_COORDINATES=1` | `lpo_data_mcp/privacy.py` |
| Liste de sensibilité imprécise | liste officielle fournie par `LPO_SENSITIVE_SPECIES` / `…_FILE`, appliquée en SQL (comparaison insensible à la casse, y compris accents), échappée | `privacy.py`, `service.py` |
| API exposée sans contrôle | jeton bearer optionnel `LPO_WEB_TOKEN` sur tous les accès aux données ; `/api/health` et `/api/catalog` restent publics | `web/app.py` |
| Fuite par agrégation | l'agrégation `grid` supprime lat/lon exactes des résultats | `service.py` |
| HTML/JS arbitraire | renderer sans `innerHTML`, catalogue fermé, validation avant envoi | `web/static/renderer.js` |
| Erreurs masquées | `_guarded` convertit toute exception en `ToolError` lisible | `server.py` |
| Configuration à chaud | `POST /api/settings` exige le même jeton que les données ; clés jamais renvoyées (`apiKeySet` seulement) ; URL http/https sans identifiants ; `LPO_WEB_SETTINGS=url-only` interdit l'exécution de commandes locales, `off` gèle la configuration | `orchestrator/settings.py`, `web/app.py` |

Le floutage est **annoncé dans l'interface** (composant `Notice`) : l'utilisateur
sait que les points sont approximatifs, et à quelle maille.

En production (PostgreSQL), la lecture seule est défendue en profondeur, sans
être confondue avec une frontière de sécurité : analyse de la requête
(`assert_read_only`), `default_transaction_read_only` sur chaque connexion, et
surtout un **rôle PostgreSQL restreint à `SELECT`** — le seul niveau qu'un
détenteur du mot de passe ne peut pas désactiver depuis la session. Les
`statement_timeout` / `lock_timeout` posés à la connexion bornent, eux, le coût
d'une requête lente sur une base partagée avec l'ETL.

---

## 7. Ce qui n'est pas fait (et pourquoi)

- **AG-UI comme transport** : la spec le recommande comme binding standard, mais
  le dépôt n'a pas d'infrastructure AG-UI. Le transport retenu (HTTP/JSON côté
  client, MCP côté données) reste conforme : AG-UI normalise le *transport*, pas
  les messages A2UI, qui sont ici strictement v1.0.
- **`callRendererFunction` / `agentFunctionResponse`** : non utilisés. Aucun
  besoin de fonctions renderer côté agent (pas de calcul client).
- **Streaming token par token** : hors périmètre ; la réponse est atomique.
- **Authentification** : voir questions ouvertes du rapport d'analyse.
