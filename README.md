# OHLEBUT

OHLEBUT est un site de prédictions pour la Coupe du Monde 2026, basé sur l'ensemble des matchs internationnaux depuis la Coupe du Monde 2022. Il prédit les résultats des phases de groupes et du tableau final, et expose ces prédictions sous forme de site statique déployé sur GitHub Pages.

## Fonctionnalités

- Prédictions de phase de groupes (résultats match par match, classements)
- Prédictions du tableau final (huitièmes → finale) avec probabilités de victoire
- Modèle XGBoost calibré avec régression isotonique sur données temporelles
- Validation walk-forward (6 fenêtres chronologiques) pour éviter toute fuite de données
- Données issues de football-data.org, API-sports.io et classements FIFA officiels

## Hébergement

Déploiement via GitHub Pages.

## Stack Technique

| Catégorie | Technologies |
|---|---|
| Machine Learning | XGBoost · scikit-learn · Pandas · NumPy |
| Base de données | SQLite |
| Données football | football-data.org · API-sports.io · Scrapping FIFA Rankings |
| Frontend | HTML · CSS · JavaScript |
| Déploiement | GitHub Pages |

## Pipeline de données

Le projet s'exécute via une série de scripts séquentiels :

```
1. init_db.py               → Initialisation de la base SQLite
2. collect_WC2026_data.py   → Collecte des données World Cup 2026
3. collect_football_data.py → Collecte de l'historique des matchs
4. clean_database.py        → Nettoyage des données
5. build_features.py        → Construction des features ML
6. load_fifa_ranking.py     → Chargement des classements FIFA officiels
7. train_model.py           → Entraînement du modèle XGBoost
8. predict_groups.py        → Prédictions de la phase de groupes
9. predict_knockout.py      → Prédictions du tableau final
10. export_json.py           → Export des JSON pour le frontend
```

## Structure du projet

```bash
OhLeBut/                          # Répertoire racine du projet
├── data/                         # Répertoire de la base SQLite et des données exportées
│   ├── predictions.json          # Prédictions globales
│   ├── groups.json               # Résultats de groupes
│   ├── bracket.json              # Tableau final
│   ├── training.json             # Données d'entraînement
│   └── model.json                # Métriques du modèle
├── models/                       # Répertoire des modèles entraînés (.pkl)
├── scripts/                      # Scripts Python du pipeline
│   ├── init_db.py                # Initialisation de la base de données
│   ├── collect_WC2026_data.py    # Collecte des données WC2026
│   ├── collect_football_data.py  # Collecte de l'historique de matchs
│   ├── clean_database.py         # Nettoyage des données
│   ├── build_features.py         # Feature engineering
│   ├── load_fifa_ranking.py      # Import des classements FIFA
│   ├── predict_groups.py         # Prédictions phase de groupes
│   ├── predict_knockout.py       # Prédictions tableau final
│   ├── export_json.py            # Export JSON pour le site
│   ├── team_utils.py             # Utilitaires normalisation des noms d'équipes
│   └── train_model.py            # Entraînement du modèle
├── pages/                        # Site statique GitHub Pages
│   ├── assets.js                 # Constantes JavaScript
│   ├── groups.html               # Phase de groupes
│   ├── bracket.html              # Tableau final
│   ├── data.html                 # Données & sources
│   └── model.html                # Méthodologie ML
├── index.html                # Page d'accueil
├── .gitignore                    # Fichier .gitignore
├── .gitattributes                # Fichier .gitattributes
├── requirements.txt              # Liste des dépendances
└── README.md                     # Ce fichier
```

## Auteurs

© 2026 / Mike Leveleux