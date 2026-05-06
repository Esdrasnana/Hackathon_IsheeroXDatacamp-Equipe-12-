# Hackathon_IsheeroXDatacamp-Equipe-12-

Ce projet explore les événements GDELT relatifs au Bénin et construit un pipeline de données + modèles ML pour extraire des insights utiles.

## Objectif

- Extraire et nettoyer les données GDELT pour le Bénin.
- Préparer un jeu de données Gold ready à l'analyse.
- Entraîner un modèle de classification du sentiment avec `RandomForestClassifier`.
- Explorer les événements via un clustering K-Means.

## Structure du dépôt

- `dashboard/` : application de visualisation (non modifiée ici).
- `data/` : données brutes et traitées.
  - `data/raw/`
  - `data/processed/`
- `src/` : code Python du projet.
  - `src/models.py` : script ML principal (préparation, entraînement, évaluation, clustering).
  - `src/pipeline.py` : extraction et nettoyage Bronze/Silver/Gold.
- `docs/` : documentation complémentaire.
- `notebooks/` : notebooks d'analyse exploratoire.

## Environnement requis

Installer les dépendances Python :

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

## Utilisation du modèle

Le script principal du modèle est `src/models.py`.

- Pour exécuter sur les données de démonstration :

```bash
python src/models.py --use-demo --clusters 4
```

- Pour exécuter sur le fichier réel si disponible :

```bash
python src/models.py
```

- Pour changer le nombre de clusters K :

```bash
python src/models.py --use-demo --clusters 5
```

## Résultats

Le script génère :

- des fichiers de visualisation dans `outputs/figures`
- des artefacts modèles dans `outputs/models`

## Branche de travail

Le code est disponible sur la branche `model` pour revue et partage.

## Notes

- `src/models.py` est désormais structuré autour d'un pipeline clair : chargement des données, préparation, entraînement, évaluation, clustering et sauvegarde des artefacts.
- Le script supporte un jeu de données de démonstration si le CSV réel est absent.
