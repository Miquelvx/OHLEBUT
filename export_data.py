"""
Export JSON — WC2026 Predictor

Génère 5 fichiers JSON depuis la DB SQLite :
  - data/predictions.json  : résumé général (accueil)
  - data/groups.json       : phases de groupe + score_freq
  - data/bracket.json      : bracket phases finales + probabilités
  - data/training.json     : matchs d'entraînement
  - data/model.json        : infos modèle + métriques (lues depuis models/)
"""

import os, sys, json
import pandas as pd
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "src/collect"))
from init_db import get_connection

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def safe_float(v, d=4):
    if v is None: return None
    try: return round(float(v), d)
    except: return None

def safe_int(v):
    if v is None: return None
    try: return int(v)
    except: return None

def write_json(filename, data):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(',',':'), default=str)
    size_kb = os.path.getsize(path) / 1024
    print(f"  ✅ {filename:<25} {size_kb:>7.0f} KB")
    return size_kb


# ══════════════════════════════════════════════════════════════════
# predictions.json
# ══════════════════════════════════════════════════════════════════

def export_predictions(conn, metrics):
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM matches")
    nb_train = c.fetchone()[0]
    c.execute("SELECT MIN(match_date), MAX(match_date) FROM matches")
    d_min, d_max = c.fetchone()

    c.execute("""
        SELECT gs.team_id, gs.team_label, gs.group_name,
               gs.fifa_ranking, t.confederation, t.country_code
        FROM group_standings gs
        LEFT JOIN teams t ON t.team_id = gs.team_id
        ORDER BY gs.group_name, gs.position
    """)
    teams = [
        {"id":r[0],"name":r[1],"group":r[2],
         "ranking":r[3],"confederation":r[4],"code":r[5]}
        for r in c.fetchall()
    ]

    c.execute("SELECT COUNT(*) FROM wc2026_fixtures WHERE actual_result IS NOT NULL")
    played = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM wc2026_fixtures")
    total = c.fetchone()[0]

    c.execute("""
        SELECT stage, COUNT(*) as nb,
               SUM(CASE WHEN actual_result IS NOT NULL THEN 1 ELSE 0 END) as played
        FROM wc2026_fixtures GROUP BY stage ORDER BY MIN(match_date)
    """)
    stages = [{"stage":r[0],"total":r[1],"played":r[2]} for r in c.fetchall()]

    return {
        "generated_at": datetime.now().isoformat(),
        "model": {
            "nb_train_matches": nb_train,
            "train_period":     f"{d_min} → {d_max}",
            "accuracy":         metrics.get("accuracy"),
            "baseline_accuracy":metrics.get("baseline_accuracy"),
            "log_loss":         metrics.get("log_loss"),
            "nb_simulations":   10000,
        },
        "tournament": {
            "name":       "FIFA World Cup 2026",
            "start_date": "2026-06-11",
            "end_date":   "2026-07-19",
            "hosts":      ["États-Unis","Canada","Mexique"],
            "nb_teams":   48,
            "nb_groups":  12,
            "format":     "12 groupes de 4 → 32èmes → 16èmes → QF → SF → Finale",
        },
        "teams": teams,
        "progress": {
            "played":  played,
            "total":   total,
            "pct":     round(played/total*100, 1) if total > 0 else 0,
            "stages":  stages,
        },
    }


# ══════════════════════════════════════════════════════════════════
# groups.json
# ══════════════════════════════════════════════════════════════════

def export_groups(conn):
    standings = pd.read_sql_query("""
        SELECT team_id, group_name, position, team_label, fifa_ranking,
               points, played, won, drawn, lost,
               goals_for, goals_against, goal_diff, qualified,
               prob_1st, prob_2nd, prob_3rd, prob_4th, prob_qualify
        FROM group_standings ORDER BY group_name, position
    """, conn)

    # Récupérer les colonnes disponibles dans wc2026_fixtures
    import sqlite3
    c = conn.cursor()
    c.execute("PRAGMA table_info(wc2026_fixtures)")
    cols = {r[1] for r in c.fetchall()}
    has_score_freq = "pred_score_freq" in cols

    freq_col = ", pred_score_freq" if has_score_freq else ""
    fixtures = pd.read_sql_query(f"""
        SELECT fixture_id, group_name, match_date,
               home_team_id, away_team_id,
               home_team_label, away_team_label,
               pred_home_goals, pred_away_goals, pred_result,
               pred_proba_home, pred_proba_draw, pred_proba_away,
               actual_home_goals, actual_away_goals, actual_result
               {freq_col}
        FROM wc2026_fixtures
        WHERE stage='GROUP_STAGE' AND home_team_id IS NOT NULL
        ORDER BY group_name, match_date
    """, conn)

    thirds = pd.read_sql_query("""
        SELECT rank, group_name, team_id, team_label,
               points, goal_diff, goals_for, fifa_ranking, prob_best3
        FROM best_third_place ORDER BY rank
    """, conn)

    groups = {}
    for grp in sorted(standings["group_name"].unique()):
        teams = []
        for _, r in standings[standings["group_name"]==grp].iterrows():
            teams.append({
                "id":    r["team_id"],
                "name":  r["team_label"],
                "rank":  safe_int(r["fifa_ranking"]),
                "pos":   safe_int(r["position"]),
                "qual":  r["qualified"],
                "pts":   safe_int(r["points"]),
                "played":safe_int(r["played"]),
                "w":safe_int(r["won"]),"d":safe_int(r["drawn"]),"l":safe_int(r["lost"]),
                "gf":safe_int(r["goals_for"]),"ga":safe_int(r["goals_against"]),
                "gd":safe_int(r["goal_diff"]),
                "p1":safe_float(r["prob_1st"],3),"p2":safe_float(r["prob_2nd"],3),
                "p3":safe_float(r["prob_3rd"],3),"p4":safe_float(r["prob_4th"],3),
                "pq":safe_float(r["prob_qualify"],3),
            })

        matches = []
        for _, m in fixtures[fixtures["group_name"]==grp].iterrows():
            match = {
                "id":      safe_int(m["fixture_id"]),
                "date":    m["match_date"],
                "home":    m["home_team_label"],
                "away":    m["away_team_label"],
                "home_id": safe_int(m["home_team_id"]),
                "away_id": safe_int(m["away_team_id"]),
                "pred_h":  safe_int(m["pred_home_goals"]),
                "pred_a":  safe_int(m["pred_away_goals"]),
                "pred_res":m["pred_result"],
                "act_h":   safe_int(m["actual_home_goals"]),
                "act_a":   safe_int(m["actual_away_goals"]),
                "act_res": m["actual_result"],
                "ph":      safe_float(m["pred_proba_home"],3),
                "pd":      safe_float(m["pred_proba_draw"],3),
                "pa":      safe_float(m["pred_proba_away"],3),
            }
            # score_freq si disponible
            if has_score_freq:
                match["score_freq"] = safe_float(m.get("pred_score_freq"), 1)
            matches.append(match)

        groups[grp] = {"teams":teams,"matches":matches}

    best_thirds = [
        {"rank":safe_int(r["rank"]),"group":r["group_name"],
         "id":safe_int(r["team_id"]),"name":r["team_label"],
         "pts":safe_int(r["points"]),"gd":safe_int(r["goal_diff"]),
         "gf":safe_int(r["goals_for"]),"fifa_rank":safe_int(r["fifa_ranking"]),
         "prob":safe_float(r["prob_best3"],3)}
        for _, r in thirds.iterrows()
    ]

    return {"groups":groups,"best_thirds":best_thirds}


# ══════════════════════════════════════════════════════════════════
# bracket.json
# ══════════════════════════════════════════════════════════════════

def export_bracket(conn):
    fixtures = pd.read_sql_query("""
        SELECT match_id, round,
               team_a_id, team_a_name, team_b_id, team_b_name,
               pred_score_a, pred_score_b,
               prob_a_wins, prob_b_wins, score_freq,
               winner_id, winner_name
        FROM knockout_fixtures ORDER BY match_id
    """, conn)

    # prob_qualify peut venir de knockout_probabilities (simulate_tournament)
    # ou de group_standings (predict_groups) — on gère les deux cas
    try:
        probs = pd.read_sql_query("""
            SELECT kp.team_id, kp.team_name, kp.group_name, kp.fifa_ranking,
                   kp.prob_r16, kp.prob_r8, kp.prob_qf, kp.prob_sf,
                   kp.prob_final, kp.prob_champion,
                   COALESCE(kp.prob_qualify, gs.prob_qualify) as prob_qualify
            FROM knockout_probabilities kp
            LEFT JOIN group_standings gs USING(team_id)
            ORDER BY kp.prob_champion DESC
        """, conn)
    except Exception:
        probs = pd.read_sql_query("""
            SELECT kp.team_id, kp.team_name, kp.group_name, kp.fifa_ranking,
                   kp.prob_r16, kp.prob_r8, kp.prob_qf, kp.prob_sf,
                   kp.prob_final, kp.prob_champion,
                   gs.prob_qualify
            FROM knockout_probabilities kp
            LEFT JOIN group_standings gs USING(team_id)
            ORDER BY kp.prob_champion DESC
        """, conn)

    bracket = [
        {"id":m["match_id"],"round":m["round"],
         "team_a":{"id":safe_int(m["team_a_id"]),"name":m["team_a_name"],
                   "score":safe_int(m["pred_score_a"])},
         "team_b":{"id":safe_int(m["team_b_id"]),"name":m["team_b_name"],
                   "score":safe_int(m["pred_score_b"])},
         "prob_a":     safe_float(m["prob_a_wins"],3),
         "prob_b":     safe_float(m["prob_b_wins"],3),
         "score_freq": safe_float(m["score_freq"],1),
         "winner":{"id":safe_int(m["winner_id"]),"name":m["winner_name"]}}
        for _, m in fixtures.iterrows()
    ]

    probabilities = [
        {"id":    safe_int(r["team_id"]),
         "name":  r["team_name"],
         "group": r["group_name"],
         "rank":  safe_int(r["fifa_ranking"]),
         "pq":    safe_float(r["prob_qualify"],3),
         "r16":   safe_float(r["prob_r16"],3),
         "r8":    safe_float(r["prob_r8"],3),
         "qf":    safe_float(r["prob_qf"],3),
         "sf":    safe_float(r["prob_sf"],3),
         "fin":   safe_float(r["prob_final"],3),
         "champ": safe_float(r["prob_champion"],3)}
        for _, r in probs.iterrows()
    ]

    return {"bracket":bracket,"probabilities":probabilities}


# ══════════════════════════════════════════════════════════════════
# training.json
# ══════════════════════════════════════════════════════════════════

def export_training(conn):
    matches = pd.read_sql_query("""
        SELECT m.match_date, m.competition, m.season, m.stage,
               m.neutral_venue,
               th.team_name as home_team, ta.team_name as away_team,
               m.home_goals, m.away_goals, m.result_90
        FROM matches m
        JOIN teams th ON th.team_id = m.home_team_id
        JOIN teams ta ON ta.team_id = m.away_team_id
        ORDER BY m.match_date DESC
    """, conn)

    comp_dist = (matches.groupby("competition").size()
                 .reset_index(name="count")
                 .sort_values("count", ascending=False))

    return {
        "total": len(matches),
        "matches": [
            {"date":r["match_date"],"competition":r["competition"],
             "season":r["season"] or "","stage":r["stage"] or "",
             "home":r["home_team"],"away":r["away_team"],
             "home_goals":safe_int(r["home_goals"]),
             "away_goals":safe_int(r["away_goals"]),
             "result":r["result_90"],"neutral":bool(r["neutral_venue"])}
            for _, r in matches.iterrows()
        ],
        "by_competition": comp_dist.to_dict(orient="records"),
        "result_dist":    matches["result_90"].value_counts().to_dict(),
    }


# ══════════════════════════════════════════════════════════════════
# model.json
# ══════════════════════════════════════════════════════════════════

def export_model(conn, metrics, features_meta):
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM matches")
    nb = c.fetchone()[0]
    c.execute("SELECT MIN(match_date), MAX(match_date) FROM matches")
    d_min, d_max = c.fetchone()

    features = features_meta.get("features", [])
    fi_raw   = features_meta.get("feature_importance", [])

    FEATURE_LABELS = {
        "ranking_gap":           "Écart de ranking FIFA",
        "ranking_gap_adj":       "Écart de ranking ajusté (confédération)",
        "home_rank_adj":         "Ranking ajusté domicile",
        "away_rank_adj":         "Ranking ajusté extérieur",
        "home_top20_ratio":      "Ratio matchs vs top 20 domicile",
        "away_top20_ratio":      "Ratio matchs vs top 20 extérieur",
        "home_fifa_ranking":     "Ranking FIFA domicile",
        "away_fifa_ranking":     "Ranking FIFA extérieur",
        "home_form5_pts":        "Points domicile (5 matchs)",
        "home_form5_scored":     "Buts marqués domicile (5 matchs)",
        "home_form5_conceded":   "Buts encaissés domicile (5 matchs)",
        "home_form10_pts":       "Points domicile (10 matchs)",
        "home_form10_scored":    "Buts marqués domicile (10 matchs)",
        "home_form10_conceded":  "Buts encaissés domicile (10 matchs)",
        "away_form5_pts":        "Points extérieur (5 matchs)",
        "away_form5_scored":     "Buts marqués extérieur (5 matchs)",
        "away_form5_conceded":   "Buts encaissés extérieur (5 matchs)",
        "away_form10_pts":       "Points extérieur (10 matchs)",
        "away_form10_scored":    "Buts marqués extérieur (10 matchs)",
        "away_form10_conceded":  "Buts encaissés extérieur (10 matchs)",
        "h2h_home_wins":         "% victoires domicile H2H",
        "h2h_draws":             "% nuls H2H",
        "h2h_away_wins":         "% victoires extérieur H2H",
        "h2h_home_goals_avg":    "Buts domicile H2H (moyenne)",
        "h2h_away_goals_avg":    "Buts extérieur H2H (moyenne)",
        "h2h_matches":           "Nombre de matchs H2H",
        "neutral_venue":         "Terrain neutre",
        "competition_weight":    "Importance de la compétition",
        "is_knockout":           "Phase éliminatoire",
    }

    feature_importance = [
        {"feature":    fi["feature"],
         "importance": safe_float(fi["importance"], 4),
         "label":      FEATURE_LABELS.get(fi["feature"], fi["feature"])}
        for fi in sorted(fi_raw, key=lambda x: x["importance"], reverse=True)
    ]

    return {
        "summary": {
            "algorithm":        "XGBoost + calibration isotonique",
            "nb_matches":       nb,
            "period":           f"{d_min} → {d_max}",
            "accuracy":         metrics.get("accuracy"),
            "baseline_accuracy":metrics.get("baseline_accuracy"),
            "log_loss":         metrics.get("log_loss"),
            "baseline_log_loss":metrics.get("baseline_log_loss"),
            "mae_home":         metrics.get("mae_home"),
            "mae_away":         metrics.get("mae_away"),
            "nb_simulations":   10000,
            "nb_features":      len(features),
        },
        "feature_importance": feature_importance,
        "features": [
            {"category":"Classement FIFA","count":2,
             "description":"Ranking FIFA domicile/extérieur."},
            {"category":"Classement ajusté","count":3,
             "description":"Ranking FIFA + pénalité confédération. Iran #21 AFC → #41 effectif."},
            {"category":"Expérience élite","count":2,
             "description":"Ratio de matchs joués contre le top 20 ajusté sur les 20 derniers matchs."},
            {"category":"Forme récente","count":12,
             "description":"Points, buts marqués/encaissés sur 5 et 10 derniers matchs. Pondérés par confédération et récence."},
            {"category":"Head-to-head","count":6,
             "description":"Historique des 5 dernières confrontations directes."},
            {"category":"Contexte","count":3,
             "description":"Terrain neutre, poids de la compétition (CdM=1.0 → amicaux=0.3), phase éliminatoire."},
        ],
        "data_sources": [
            {"name":"FIFA World Cup 2022",             "type":"Tournoi"},
            {"name":"Qualifications CdM 2026",         "type":"Qualifications"},
            {"name":"UEFA Euro 2024 + qualifications", "type":"Tournoi"},
            {"name":"Copa América 2024",               "type":"Tournoi"},
            {"name":"AFCON 2023 + 2025",               "type":"Tournoi"},
            {"name":"AFC Asian Cup 2023",              "type":"Tournoi"},
            {"name":"Gold Cup 2023 + 2025",            "type":"Tournoi"},
            {"name":"UEFA Nations League 2022 + 2024", "type":"Nations League"},
            {"name":"CONCACAF Nations League",         "type":"Nations League"},
            {"name":"Matchs amicaux 2023-2026",        "type":"Amicaux"},
        ],
    }


# ══════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*55)
    print("  Export JSON — WC2026 Predictor")
    print("="*55)

    os.makedirs(DATA_DIR, exist_ok=True)

    metrics_path  = os.path.join(MODELS_DIR, "metrics.json")
    features_path = os.path.join(MODELS_DIR, "features.json")

    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        print(f"  Métriques : accuracy={metrics.get('accuracy')}"
              f"  log_loss={metrics.get('log_loss')}")
    else:
        print("  ⚠️  models/metrics.json introuvable")

    features_meta = {}
    if os.path.exists(features_path):
        with open(features_path) as f:
            features_meta = json.load(f)
        print(f"  Features  : {len(features_meta.get('features',[]))} features")
    else:
        print("  ⚠️  models/features.json introuvable")

    conn = get_connection()

    print("\n📦 Génération des fichiers JSON :")
    print(f"  {'Fichier':<25} {'Taille':>8}")
    print(f"  {'─'*35}")

    write_json("predictions.json", export_predictions(conn, metrics))
    write_json("groups.json",      export_groups(conn))
    write_json("bracket.json",     export_bracket(conn))
    write_json("training.json",    export_training(conn))
    write_json("model.json",       export_model(conn, metrics, features_meta))

    conn.close()

    total_kb = sum(
        os.path.getsize(os.path.join(DATA_DIR, f)) / 1024
        for f in ["predictions.json","groups.json","bracket.json",
                  "training.json","model.json"]
        if os.path.exists(os.path.join(DATA_DIR, f))
    )
    print(f"\n  Total : {total_kb:.0f} KB")
    print(f"  → {DATA_DIR}")
    print("\n🎉 Export terminé")