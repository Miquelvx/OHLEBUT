"""
Feature Engineering — WC2026 Predictor

Pour chaque match du training set, calcule les features suivantes
AU MOMENT DU MATCH (pas de data leakage) :

Features de classement:
  - home_fifa_ranking, away_fifa_ranking
  - ranking_gap (away - home, positif = home mieux classé)

Features de forme récente (fenêtres glissantes AVANT le match):
  - home_form_5, away_form_5       : points moyens sur 5 derniers matchs
  - home_form_10, away_form_10     : points moyens sur 10 derniers matchs
  - home_goals_scored_5, away_goals_scored_5   : buts marqués moy. sur 5
  - home_goals_conceded_5, away_goals_conceded_5 : buts encaissés moy. sur 5

Features contextuelles:
  - neutral_venue                  : terrain neutre (déjà en DB)
  - competition_weight             : importance de la compétition
  - is_knockout                    : phase finale (pas de nul possible)

Features head-to-head (5 dernières confrontations AVANT le match):
  - h2h_home_wins, h2h_draws, h2h_away_wins
  - h2h_home_goals_avg, h2h_away_goals_avg

Variables cibles:
  - result_90    : H/D/A à 90 minutes
  - result_winner: H/A vainqueur effectif
  - home_goals, away_goals : score final
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(__file__))
from init_db import get_connection

# ------------------------------------------------------------------
# Poids des compétitions
# ------------------------------------------------------------------

COMPETITION_WEIGHTS = {
    # Tournois continentaux majeurs
    "UEFA Euro":                                               0.8,
    "Copa América":                                            0.8,
    "Africa Cup of Nations":                                   0.8,
    "AFC Asian Cup":                                           0.8,
    "CONCACAF Gold Cup":                                       0.8,
    # Qualifications CdM
    "WC Qualification Europe":                                 0.7,
    "WC Qualification CAF":                                    0.7,
    "WC Qualification Asia":                                   0.7,
    "WC Qualification CONCACAF":                               0.7,
    "WC Qualification CONMEBOL":                               0.7,
    "WC Qualification OFC":                                    0.7,
    "WC Qualification Intercontinental":                       0.7,
    # Qualifications tournois continentaux
    "Euro 2024 Qualifications":                                0.6,
    # Nations League
    "UEFA Nations League":                                     0.5,
    "CONCACAF Nations League":                                 0.5,
    # Amicaux
    "International Friendlies":                                0.3,
}

KNOCKOUT_STAGES = {
    "LAST_16", "QUARTER_FINALS", "SEMI_FINALS",
    "FINAL", "THIRD_PLACE", "Round of 16",
    "Quarter-finals", "Semi-finals", "Final", "3rd Place Final",
    "Round of 32",
}

# Poids spécifiques pour les phases finales de la CdM
WC_KNOCKOUT_WEIGHT = 1.0
WC_GROUP_WEIGHT    = 0.9
# Poids pour les phases finales des autres compétitions majeures
MAJOR_KNOCKOUT_WEIGHT = 0.85
MAJOR_GROUP_WEIGHT    = 0.8

WC_COMPETITIONS = {"FIFA World Cup 2022"}


def get_competition_weight(competition: str, stage: str) -> float:
    """Retourne le poids de la compétition selon la compétition et la phase."""
    knockout = any(k in str(stage) for k in KNOCKOUT_STAGES)

    if competition in WC_COMPETITIONS:
        weight = WC_KNOCKOUT_WEIGHT if knockout else WC_GROUP_WEIGHT
    elif competition in COMPETITION_WEIGHTS:
        base = COMPETITION_WEIGHTS[competition]
        # Pour les compétitions majeures (poids de base 0.8), on applique
        # les poids spécifiques phases finales/groupes
        if base == 0.8:
            weight = MAJOR_KNOCKOUT_WEIGHT if knockout else MAJOR_GROUP_WEIGHT
        else:
            # Qualifications et Nations League : pas de bonus phase finale
            weight = base
    else:
        weight = 0.5

    # Arrondi pour éviter les flottants à 0.7999...
    return round(weight, 2)


def is_knockout(stage: str) -> int:
    """Retourne 1 si le match est une phase finale (pas de nul possible)."""
    return 1 if any(k in str(stage) for k in KNOCKOUT_STAGES) else 0


# ------------------------------------------------------------------
# Calcul de la forme récente
# ------------------------------------------------------------------

def compute_form(team_id: int, before_date: str, matches_df: pd.DataFrame,
                 window: int = 5) -> dict:
    """
    Calcule les stats de forme d'une équipe sur ses `window` derniers matchs
    AVANT `before_date`.

    Retourne un dict avec :
      - points_avg   : moyenne de points (victoire=3, nul=1, défaite=0)
      - goals_scored_avg  : moyenne de buts marqués
      - goals_conceded_avg: moyenne de buts encaissés
    """
    # Matchs de l'équipe avant la date du match courant
    team_matches = matches_df[
        ((matches_df["home_team_id"] == team_id) |
         (matches_df["away_team_id"] == team_id)) &
        (matches_df["match_date"] < before_date)
    ].sort_values("match_date", ascending=False).head(window)

    if len(team_matches) == 0:
        return {"points_avg": None, "goals_scored_avg": None, "goals_conceded_avg": None}

    points_list        = []
    goals_scored_list  = []
    goals_conceded_list = []

    for _, m in team_matches.iterrows():
        if m["home_team_id"] == team_id:
            scored    = m["home_goals"]
            conceded  = m["away_goals"]
            result    = m["result_90"]
            pts = 3 if result == "H" else (1 if result == "D" else 0)
        else:
            scored    = m["away_goals"]
            conceded  = m["home_goals"]
            result    = m["result_90"]
            pts = 3 if result == "A" else (1 if result == "D" else 0)

        points_list.append(pts)
        goals_scored_list.append(scored)
        goals_conceded_list.append(conceded)

    return {
        "points_avg":        round(np.mean(points_list), 4),
        "goals_scored_avg":  round(np.mean(goals_scored_list), 4),
        "goals_conceded_avg": round(np.mean(goals_conceded_list), 4),
    }


# ------------------------------------------------------------------
# Calcul du head-to-head
# ------------------------------------------------------------------

def compute_h2h(home_id: int, away_id: int, before_date: str,
                matches_df: pd.DataFrame, window: int = 5) -> dict:
    """
    Calcule les stats head-to-head entre deux équipes sur les `window`
    dernières confrontations AVANT `before_date`.
    """
    h2h = matches_df[
        (
            ((matches_df["home_team_id"] == home_id) & (matches_df["away_team_id"] == away_id)) |
            ((matches_df["home_team_id"] == away_id) & (matches_df["away_team_id"] == home_id))
        ) &
        (matches_df["match_date"] < before_date)
    ].sort_values("match_date", ascending=False).head(window)

    if len(h2h) == 0:
        return {
            "h2h_home_wins": None, "h2h_draws": None, "h2h_away_wins": None,
            "h2h_home_goals_avg": None, "h2h_away_goals_avg": None,
            "h2h_matches": 0,
        }

    home_wins  = 0
    draws      = 0
    away_wins  = 0
    home_goals = []
    away_goals = []

    for _, m in h2h.iterrows():
        if m["home_team_id"] == home_id:
            hg, ag = m["home_goals"], m["away_goals"]
            r = m["result_90"]
            if r == "H": home_wins += 1
            elif r == "D": draws += 1
            else: away_wins += 1
        else:
            # Match inversé : home_id était away dans ce match
            hg, ag = m["away_goals"], m["home_goals"]
            r = m["result_90"]
            if r == "A": home_wins += 1
            elif r == "D": draws += 1
            else: away_wins += 1
        home_goals.append(hg)
        away_goals.append(ag)

    n = len(h2h)
    return {
        "h2h_home_wins":      round(home_wins / n, 4),
        "h2h_draws":          round(draws / n, 4),
        "h2h_away_wins":      round(away_wins / n, 4),
        "h2h_home_goals_avg": round(np.mean(home_goals), 4),
        "h2h_away_goals_avg": round(np.mean(away_goals), 4),
        "h2h_matches":        n,
    }


# ------------------------------------------------------------------
# Création de la table features
# ------------------------------------------------------------------

def create_features_table(conn):
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS match_features")
    c.execute("""
        CREATE TABLE match_features (
            match_id                INTEGER PRIMARY KEY REFERENCES matches(match_id),
            match_date              TEXT,
            competition             TEXT,
            competition_weight      REAL,
            is_knockout             INTEGER,
            neutral_venue           INTEGER,

            home_team_id            INTEGER,
            away_team_id            INTEGER,

            -- Classement FIFA
            home_fifa_ranking       INTEGER,
            away_fifa_ranking       INTEGER,
            ranking_gap             INTEGER,   -- away_ranking - home_ranking

            -- Forme récente home (5 et 10 derniers matchs)
            home_form5_pts          REAL,
            home_form5_scored       REAL,
            home_form5_conceded     REAL,
            home_form10_pts         REAL,
            home_form10_scored      REAL,
            home_form10_conceded    REAL,

            -- Forme récente away (5 et 10 derniers matchs)
            away_form5_pts          REAL,
            away_form5_scored       REAL,
            away_form5_conceded     REAL,
            away_form10_pts         REAL,
            away_form10_scored      REAL,
            away_form10_conceded    REAL,

            -- Head-to-head
            h2h_home_wins           REAL,
            h2h_draws               REAL,
            h2h_away_wins           REAL,
            h2h_home_goals_avg      REAL,
            h2h_away_goals_avg      REAL,
            h2h_matches             INTEGER,

            -- Variables cibles
            home_goals              INTEGER,
            away_goals              INTEGER,
            goal_diff               INTEGER,
            result_90               TEXT,
            result_winner           TEXT
        )
    """)
    conn.commit()
    print("✅ Table match_features créée")


# ------------------------------------------------------------------
# Pipeline principal
# ------------------------------------------------------------------

def build_features():
    conn = get_connection()

    create_features_table(conn)

    # Charger tous les matchs en mémoire pour les calculs de fenêtre glissante
    matches_df = pd.read_sql_query("""
        SELECT match_id, match_date, competition, stage,
               home_team_id, away_team_id,
               home_goals, away_goals, goal_diff,
               result_90, result_winner,
               neutral_venue, status,
               competition_id
        FROM matches
        WHERE result_90 IS NOT NULL
        ORDER BY match_date ASC
    """, conn)

    # Charger les rankings FIFA (on utilise le ranking actuel comme proxy)
    teams_df = pd.read_sql_query("""
        SELECT team_id, fifa_ranking FROM teams
    """, conn)
    ranking_map = dict(zip(teams_df["team_id"], teams_df["fifa_ranking"]))

    print(f"📊 {len(matches_df)} matchs à traiter...")

    rows = []
    for idx, m in matches_df.iterrows():
        if idx % 200 == 0:
            print(f"   ... {idx}/{len(matches_df)}")

        match_date = m["match_date"]
        home_id    = m["home_team_id"]
        away_id    = m["away_team_id"]

        # Poids de la compétition
        weight     = get_competition_weight(m["competition"], m["stage"])
        knockout   = is_knockout(m["stage"])

        # Classements FIFA
        home_rank  = ranking_map.get(home_id)
        away_rank  = ranking_map.get(away_id)
        rank_gap   = (away_rank - home_rank) if (home_rank and away_rank) else None

        # Forme sur 5 matchs
        home_f5    = compute_form(home_id, match_date, matches_df, window=5)
        away_f5    = compute_form(away_id, match_date, matches_df, window=5)

        # Forme sur 10 matchs
        home_f10   = compute_form(home_id, match_date, matches_df, window=10)
        away_f10   = compute_form(away_id, match_date, matches_df, window=10)

        # Head-to-head
        h2h        = compute_h2h(home_id, away_id, match_date, matches_df, window=5)

        rows.append({
            "match_id":             m["match_id"],
            "match_date":           match_date,
            "competition":          m["competition"],
            "competition_weight":   weight,
            "is_knockout":          knockout,
            "neutral_venue":        m["neutral_venue"],
            "home_team_id":         home_id,
            "away_team_id":         away_id,
            "home_fifa_ranking":    home_rank,
            "away_fifa_ranking":    away_rank,
            "ranking_gap":          rank_gap,
            "home_form5_pts":       home_f5["points_avg"],
            "home_form5_scored":    home_f5["goals_scored_avg"],
            "home_form5_conceded":  home_f5["goals_conceded_avg"],
            "home_form10_pts":      home_f10["points_avg"],
            "home_form10_scored":   home_f10["goals_scored_avg"],
            "home_form10_conceded": home_f10["goals_conceded_avg"],
            "away_form5_pts":       away_f5["points_avg"],
            "away_form5_scored":    away_f5["goals_scored_avg"],
            "away_form5_conceded":  away_f5["goals_conceded_avg"],
            "away_form10_pts":      away_f10["points_avg"],
            "away_form10_scored":   away_f10["goals_scored_avg"],
            "away_form10_conceded": away_f10["goals_conceded_avg"],
            "h2h_home_wins":        h2h["h2h_home_wins"],
            "h2h_draws":            h2h["h2h_draws"],
            "h2h_away_wins":        h2h["h2h_away_wins"],
            "h2h_home_goals_avg":   h2h["h2h_home_goals_avg"],
            "h2h_away_goals_avg":   h2h["h2h_away_goals_avg"],
            "h2h_matches":          h2h["h2h_matches"],
            "home_goals":           m["home_goals"],
            "away_goals":           m["away_goals"],
            "goal_diff":            m["goal_diff"],
            "result_90":            m["result_90"],
            "result_winner":        m["result_winner"],
        })

    # Insertion en batch
    features_df = pd.DataFrame(rows)
    features_df.to_sql("match_features", conn, if_exists="replace",
                       index=False, method="multi")

    conn.close()

    # Résumé
    print(f"\n✅ {len(features_df)} lignes insérées dans match_features")
    print(f"\n📊 Distribution des résultats (result_90) :")
    print(features_df["result_90"].value_counts().to_string())
    print(f"\n📊 Features avec valeurs manquantes (NULL) :")
    nulls = features_df.isnull().sum()
    nulls = nulls[nulls > 0]
    print(nulls.to_string() if len(nulls) > 0 else "   Aucune ✅")


if __name__ == "__main__":
    print("=" * 55)
    print("  Feature Engineering — WC2026 Predictor")
    print("=" * 55)
    build_features()
    print("\n🎉 Feature engineering terminé.")