# CI — workflow proposé (à installer manuellement)

`python-app.yml` est le workflow recommandé. Il n'a **pas** pu être installé
automatiquement : le jeton GitHub utilisé pour pousser cette branche n'a pas la
permission `workflows`, et GitHub refuse toute modification de
`.github/workflows/*` dans ce cas.

## Pourquoi le remplacer

Le workflow actuellement en place (`.github/workflows/python-app.yml`) :

- utilise Python **3.10** alors que `pyproject.toml` exige `>=3.11` ;
- installe `requirements.txt` mais **ne lance aucun test** (flake8 seulement) ;
- affiche une clé API dans les journaux (`echo "Using API key … $HUGGINGFACE_API_KEY"`),
  ce qui expose un secret dans les logs publics du dépôt.

## Installation (au choix)

1. **Interface GitHub** : ouvrir `.github/workflows/python-app.yml` dans
   l'éditeur web et coller le contenu de `docs/ci/python-app.yml` ;
2. **Ligne de commande** avec un jeton ayant la permission `workflows` :

   ```bash
   cp docs/ci/python-app.yml .github/workflows/python-app.yml
   git commit -am "ci: lint + tests sur Python 3.11"
   git push
   ```

## Ce que fait le workflow proposé

```yaml
- Python 3.11 + cache pip
- pip install -r requirements.txt  (chaîne MCP/A2UI incluse)
- pip install pytest pytest-asyncio ruff
- ruff check .
- python -m pytest -q        # 201 tests, sans clé LLM ni base externe
```

Les tests démarrent eux-mêmes leurs serveurs MCP (stdio et streamable-http) et
génèrent leur base de démonstration : aucun service externe n'est nécessaire.
