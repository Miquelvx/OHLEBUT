"""
Feature Engineering — WC2026 Predictor (v3)

Améliorations par rapport à v2 :
  1. Forme pondérée par la qualité de l'adversaire + confédération
     pts_pondérés = pts × conf_factor × (1 + (100 - rank_adversaire) / 200)
     → Victoire Iran vs Irak (AFC) vaut moins que victoire France vs Espagne (UEFA)

  2. Décroissance temporelle exponentielle (inchangé)

  3. Imputation H2H intelligente (inchangé)

  4. ranking_gap_adj : ranking FIFA + pénalité confédération
     → Iran #21 AFC devient effectivement #41, réduisant les upsets artificiels

  5. top20_ratio : ratio de matchs joués contre le top 20 FIFA
     → Capture l'expérience des grandes équipes contre l'élite mondiale
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(__file__))
from init_db import get_connection

# ── Paramètres ────────────────────────────────────────────────────
LAMBDA_DECAY = 0.001   # Décroissance temporelle : demi-vie ≈ 693 jours

# ── Force relative des confédérations ─────────────────────────────
# Calibré sur les résultats inter-confédérations depuis 2018
# UEFA=référence, les autres sont pénalisées proportionnellement
CONF_STRENGTH = {
    "UEFA":     1.00,
    "CONMEBOL": 0.95,
    "CAF":      0.75,
    "AFC":      0.70,
    "CONCACAF": 0.72,
    "OFC":      0.55,
}

# Pénalité de ranking ajouté au ranking FIFA brut
# Iran #21 AFC → ranking ajusté #41 (21 + 20)
# France #3 UEFA → ranking ajusté #3 (3 + 0)
CONF_RANKING_PENALTY = {
    "UEFA":     0,
    "CONMEBOL": 5,
    "CAF":      15,
    "AFC":      20,
    "CONCACAF": 15,
    "OFC":      30,
}

COMPETITION_WEIGHTS = {
    "UEFA Euro":                        0.8,
    "Copa América":                     0.8,
    "Africa Cup of Nations":            0.8,
    "AFC Asian Cup":                    0.8,
    "CONCACAF Gold Cup":                0.8,
    "WC Qualification Europe":          0.7,
    "WC Qualification CAF":             0.7,
    "WC Qualification Asia":            0.7,
    "WC Qualification CONCACAF":        0.7,
    "WC Qualification CONMEBOL":        0.7,
    "WC Qualification OFC":             0.7,
    "WC Qualification Intercontinental":0.7,
    "Euro 2024 Qualifications":         0.6,
    "UEFA Nations League":              0.5,
    "CONCACAF Nations League":          0.5,
    "International Friendlies":         0.3,
}

KNOCKOUT_STAGES = {
    "LAST_16","QUARTER_FINALS","SEMI_FINALS","FINAL","THIRD_PLACE",
    "Round of 16","Quarter-finals","Semi-finals","Final","3rd Place Final",
    "Round of 32",
}

WC_COMPETITIONS = {"FIFA World Cup 2022"}


def get_competition_weight(competition, stage):
    knockout = any(k in str(stage) for k in KNOCKOUT_STAGES)
    if competition in WC_COMPETITIONS:
        return 1.0 if knockout else 0.9
    base = COMPETITION_WEIGHTS.get(competition, 0.5)
    if base == 0.8:
        return 0.85 if knockout else 0.8
    return base


def is_knockout(stage):
    return 1 if any(k in str(stage) for k in KNOCKOUT_STAGES) else 0


def days_between(date1_str, date2_str):
    """Nombre de jours entre deux dates ISO."""
    try:
        d1 = datetime.strptime(date1_str[:10], "%Y-%m-%d")
        d2 = datetime.strptime(date2_str[:10], "%Y-%m-%d")
        return abs((d2 - d1).days)
    except:
        return 365


# ── Amélioration 1+2 : Forme pondérée par adversaire + décroissance ──

def compute_form_v2(team_id, before_date, matches_df, ranking_map,
                    conf_map, window=5):
    """
    Calcule la forme pondérée :
    - Pondération par confédération de l'adversaire (NOUVEAU)
    - Pondération par qualité de l'adversaire (ranking FIFA)
    - Pondération par récence (décroissance exponentielle)

    Retourne pts_avg, goals_scored_avg, goals_conceded_avg pondérés.
    """
    team_matches = matches_df[
        ((matches_df["home_team_id"] == team_id) |
         (matches_df["away_team_id"] == team_id)) &
        (matches_df["match_date"] < before_date)
    ].sort_values("match_date", ascending=False).head(window)

    if len(team_matches) == 0:
        return {"points_avg": None, "goals_scored_avg": None, "goals_conceded_avg": None}

    pts_list    = []
    sc_list     = []
    co_list     = []
    weights     = []

    for _, m in team_matches.iterrows():
        is_home = (m["home_team_id"] == team_id)
        opp_id  = m["away_team_id"] if is_home else m["home_team_id"]

        if is_home:
            scored   = m["home_goals"]; conceded = m["away_goals"]
            result   = m["result_90"]
            pts_raw  = 3 if result=="H" else (1 if result=="D" else 0)
        else:
            scored   = m["away_goals"]; conceded = m["home_goals"]
            result   = m["result_90"]
            pts_raw  = 3 if result=="A" else (1 if result=="D" else 0)

        # Pondération par confédération de l'adversaire
        opp_conf    = conf_map.get(int(opp_id), "")
        conf_factor = CONF_STRENGTH.get(opp_conf, 0.75)

        # Pondération par ranking de l'adversaire
        opp_rank   = ranking_map.get(opp_id, 100)
        rank_bonus  = 1.0 + (100 - min(opp_rank, 100)) / 200.0

        # Combinaison : conf × rank_bonus
        opp_weight   = conf_factor * rank_bonus
        pts_weighted = pts_raw * opp_weight

        # Décroissance temporelle
        days = days_between(m["match_date"], before_date)
        time_weight = np.exp(-LAMBDA_DECAY * days)

        pts_list.append(pts_weighted)
        sc_list.append(scored)
        co_list.append(conceded)
        weights.append(time_weight)

    w = np.array(weights)
    w_sum = w.sum()
    if w_sum == 0:
        return {"points_avg": None, "goals_scored_avg": None, "goals_conceded_avg": None}

    return {
        "points_avg":        round(np.dot(pts_list, w) / w_sum, 4),
        "goals_scored_avg":  round(np.dot(sc_list,  w) / w_sum, 4),
        "goals_conceded_avg":round(np.dot(co_list,  w) / w_sum, 4),
    }


# ── Amélioration 3 : H2H intelligent ─────────────────────────────

def ranking_gap_to_h2h_proba(gap):
    """
    Convertit un écart de ranking FIFA en probabilités H/D/A implicites.
    gap = away_rank - home_rank (positif = home favoris)

    Calibration empirique depuis les données historiques :
    - gap > 50  : home gagne souvent (~60%)
    - gap ≈ 0   : équilibre (~38% H, 28% D, 34% A)
    - gap < -50 : away gagne souvent (~60%)
    """
    # Fonction sigmoïde centrée sur le gap
    # prob_home = 0.38 + 0.22 × tanh(gap / 60)
    if gap is None:
        return 0.38, 0.28, 0.34

    prob_h = 0.38 + 0.22 * np.tanh(gap / 60)
    prob_a = 0.34 - 0.22 * np.tanh(gap / 60)
    prob_d = max(0.15, 1.0 - prob_h - prob_a)

    # Renormaliser
    total = prob_h + prob_d + prob_a
    return round(prob_h/total, 4), round(prob_d/total, 4), round(prob_a/total, 4)


def compute_h2h_v2(home_id, away_id, before_date, matches_df,
                   ranking_map, window=5):
    """
    H2H v2 : si pas d'historique, imputer depuis le ranking_gap
    plutôt que 0.33/0.33/0.33.
    """
    h2h = matches_df[
        (((matches_df["home_team_id"]==home_id)&(matches_df["away_team_id"]==away_id))|
         ((matches_df["home_team_id"]==away_id)&(matches_df["away_team_id"]==home_id))) &
        (matches_df["match_date"] < before_date)
    ].sort_values("match_date", ascending=False).head(window)

    if len(h2h) == 0:
        # Imputation intelligente depuis ranking_gap
        home_rank = ranking_map.get(home_id, 50)
        away_rank = ranking_map.get(away_id, 50)
        gap = away_rank - home_rank if (home_rank and away_rank) else 0
        ph, pd_, pa = ranking_gap_to_h2h_proba(gap)
        avg_goals = 1.2  # moyenne historique
        return {
            "h2h_home_wins":      ph,
            "h2h_draws":          pd_,
            "h2h_away_wins":      pa,
            "h2h_home_goals_avg": avg_goals,
            "h2h_away_goals_avg": avg_goals * 0.85,
            "h2h_matches":        0,
        }

    home_wins=draws=away_wins=0; hg_list=[]; ag_list=[]
    for _, m in h2h.iterrows():
        if m["home_team_id"] == home_id:
            hg,ag=m["home_goals"],m["away_goals"]
            r=m["result_90"]
            if r=="H": home_wins+=1
            elif r=="D": draws+=1
            else: away_wins+=1
        else:
            hg,ag=m["away_goals"],m["home_goals"]
            r=m["result_90"]
            if r=="A": home_wins+=1
            elif r=="D": draws+=1
            else: away_wins+=1
        hg_list.append(hg); ag_list.append(ag)

    n=len(h2h)
    return {
        "h2h_home_wins":      round(home_wins/n, 4),
        "h2h_draws":          round(draws/n, 4),
        "h2h_away_wins":      round(away_wins/n, 4),
        "h2h_home_goals_avg": round(np.mean(hg_list), 4),
        "h2h_away_goals_avg": round(np.mean(ag_list), 4),
        "h2h_matches":        n,
    }


# ── Pipeline principal ────────────────────────────────────────────

def create_features_table(conn):
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS match_features")
    c.execute("""
        CREATE TABLE match_features (
            match_id             INTEGER PRIMARY KEY,
            match_date           TEXT,
            competition          TEXT,
            competition_weight   REAL,
            is_knockout          INTEGER,
            neutral_venue        INTEGER,
            home_team_id         INTEGER,
            away_team_id         INTEGER,
            home_fifa_ranking    INTEGER,
            away_fifa_ranking    INTEGER,
            ranking_gap          INTEGER,
            ranking_gap_adj      INTEGER,
            home_rank_adj        INTEGER,
            away_rank_adj        INTEGER,
            home_top20_ratio     REAL,
            away_top20_ratio     REAL,
            home_form5_pts       REAL,
            home_form5_scored    REAL,
            home_form5_conceded  REAL,
            home_form10_pts      REAL,
            home_form10_scored   REAL,
            home_form10_conceded REAL,
            away_form5_pts       REAL,
            away_form5_scored    REAL,
            away_form5_conceded  REAL,
            away_form10_pts      REAL,
            away_form10_scored   REAL,
            away_form10_conceded REAL,
            h2h_home_wins        REAL,
            h2h_draws            REAL,
            h2h_away_wins        REAL,
            h2h_home_goals_avg   REAL,
            h2h_away_goals_avg   REAL,
            h2h_matches          INTEGER,
            home_goals           INTEGER,
            away_goals           INTEGER,
            goal_diff            INTEGER,
            result_90            TEXT,
            result_winner        TEXT
        )
    """)
    conn.commit()
    print("✅ Table match_features créée")


def build_features():
    conn = get_connection()
    create_features_table(conn)

    matches_df = pd.read_sql_query("""
        SELECT match_id, match_date, competition, stage,
               home_team_id, away_team_id,
               home_goals, away_goals, goal_diff,
               result_90, result_winner,
               neutral_venue, status, competition_id
        FROM matches
        WHERE result_90 IS NOT NULL
        ORDER BY match_date ASC
    """, conn)

    # ── Carte des confédérations par team_id ──────────────────────
    conf_df = pd.read_sql_query(
        "SELECT team_id, confederation FROM teams WHERE confederation IS NOT NULL",
        conn)
    conf_map = dict(zip(conf_df["team_id"].astype(int),
                        conf_df["confederation"].str.strip()))

    # Rankings FIFA : depuis match_features (déjà résolus par load_fifa_ranking.py)
    # Fallback : table teams si match_features vide
    ranking_map = {}

    # Source 1 : derniers rankings résolus depuis match_features
    try:
        mf_ranks = pd.read_sql_query("""
            SELECT home_team_id as tid, home_fifa_ranking as rank, match_date
            FROM match_features WHERE home_fifa_ranking IS NOT NULL
            UNION ALL
            SELECT away_team_id, away_fifa_ranking, match_date
            FROM match_features WHERE away_fifa_ranking IS NOT NULL
            ORDER BY match_date DESC
        """, conn)
        for _, r in mf_ranks.iterrows():
            tid = int(r["tid"])
            if tid not in ranking_map and r["rank"] is not None:
                ranking_map[tid] = int(r["rank"])
        print(f"   Rankings depuis match_features : {len(ranking_map)} équipes")
    except Exception:
        pass

    # Source 2 : table teams (fallback)
    for _, r in pd.read_sql_query(
        "SELECT team_id, fifa_ranking FROM teams WHERE fifa_ranking IS NOT NULL",
        conn).iterrows():
        tid = int(r["team_id"])
        if tid not in ranking_map:
            ranking_map[tid] = int(r["fifa_ranking"])

    print(f"📊 {len(matches_df)} matchs à traiter...")
    print(f"   Confédérations chargées : {len(conf_map)} équipes")

    def get_adj_rank(team_id):
        """Ranking FIFA + pénalité confédération."""
        rank = ranking_map.get(team_id, 100)
        conf = conf_map.get(team_id, "")
        penalty = CONF_RANKING_PENALTY.get(conf, 15)
        return rank + penalty

    def compute_top20_ratio(team_id, before_date, window=20):
        """
        Ratio de matchs joués contre le top 20 FIFA (ajusté)
        sur les `window` derniers matchs avant before_date.
        Capture l'expérience contre l'élite mondiale.
        """
        recent = matches_df[
            ((matches_df["home_team_id"] == team_id) |
             (matches_df["away_team_id"] == team_id)) &
            (matches_df["match_date"] < before_date)
        ].sort_values("match_date", ascending=False).head(window)

        if len(recent) == 0:
            return 0.0

        top20_count = 0
        for _, m in recent.iterrows():
            opp_id = m["away_team_id"] if m["home_team_id"] == team_id \
                     else m["home_team_id"]
            if get_adj_rank(opp_id) <= 20:
                top20_count += 1
        return round(top20_count / len(recent), 4)

    rows = []
    for idx, m in matches_df.iterrows():
        if idx % 200 == 0:
            print(f"   ... {idx}/{len(matches_df)}")

        match_date = m["match_date"]
        home_id    = int(m["home_team_id"])
        away_id    = int(m["away_team_id"])

        weight   = get_competition_weight(m["competition"], m["stage"])
        knockout = is_knockout(m["stage"])

        home_rank = ranking_map.get(home_id)
        away_rank = ranking_map.get(away_id)
        rank_gap  = (away_rank - home_rank) if (home_rank and away_rank) else None

        # Ranking ajusté : FIFA + pénalité confédération
        home_rank_adj = get_adj_rank(home_id)
        away_rank_adj = get_adj_rank(away_id)
        rank_gap_adj  = away_rank_adj - home_rank_adj

        # Forme v2 (pondérée + décroissance + confédération adversaire)
        home_f5  = compute_form_v2(home_id, match_date, matches_df, ranking_map,
                                   conf_map, 5)
        away_f5  = compute_form_v2(away_id, match_date, matches_df, ranking_map,
                                   conf_map, 5)
        home_f10 = compute_form_v2(home_id, match_date, matches_df, ranking_map,
                                   conf_map, 10)
        away_f10 = compute_form_v2(away_id, match_date, matches_df, ranking_map,
                                   conf_map, 10)

        # H2H v2 (imputation intelligente)
        h2h = compute_h2h_v2(home_id, away_id, match_date, matches_df, ranking_map, 5)

        # top20_ratio : expérience contre l'élite
        home_top20 = compute_top20_ratio(home_id, match_date)
        away_top20 = compute_top20_ratio(away_id, match_date)

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
            "ranking_gap_adj":      rank_gap_adj,       # NOUVEAU
            "home_rank_adj":        home_rank_adj,      # NOUVEAU
            "away_rank_adj":        away_rank_adj,      # NOUVEAU
            "home_top20_ratio":     home_top20,         # NOUVEAU
            "away_top20_ratio":     away_top20,         # NOUVEAU
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

    features_df = pd.DataFrame(rows)
    features_df.to_sql("match_features", conn, if_exists="replace",
                       index=False, method="multi")
    conn.close()

    print(f"\n✅ {len(features_df)} lignes insérées dans match_features")
    print(f"\n📊 Distribution des résultats :")
    print(features_df["result_90"].value_counts().to_string())
    print(f"\n📊 Valeurs manquantes :")
    nulls = features_df.isnull().sum()
    nulls = nulls[nulls > 0]
    print(nulls.to_string() if len(nulls) > 0 else "   Aucune ✅")
    print(f"\n📊 Comparaison H2H (matchs sans historique) :")
    no_h2h = (features_df["h2h_matches"] == 0).sum()
    print(f"   {no_h2h} matchs sans H2H ({no_h2h/len(features_df)*100:.1f}%)"
          f" → imputation intelligente appliquée")


if __name__ == "__main__":
    print("="*55)
    print("  Feature Engineering v2 — WC2026 Predictor")
    print("="*55)
    build_features()
    print("\n🎉 Feature engineering terminé.")