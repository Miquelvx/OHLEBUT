import os, sys, json
import pandas as pd
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "src/collect"))
from init_db import get_connection

DATA_DIR   = os.path.join(os.path.dirname(__file__), "../data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")

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

    # Liste des dates de tous les matchs (pour calcul de progression côté client)
    c.execute("SELECT match_date FROM wc2026_fixtures ORDER BY match_date")
    match_dates = [r[0][:10] for r in c.fetchall() if r[0]]

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
            "played":      played,
            "total":       total,
            "pct":         round(played/total*100, 1) if total > 0 else 0,
            "stages":      stages,
            "match_dates": match_dates,  # ← dates réelles de tous les matchs
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
    c = conn.cursor()
    c.execute("PRAGMA table_info(wc2026_fixtures)")
    cols = {r[1] for r in c.fetchall()}

    extra_cols = ""
    if "prob_win_a" in cols:
        extra_cols = ", prob_win_a, prob_win_b"

    c.execute(f"""
        SELECT fixture_id, stage, match_date,
               home_team_id, away_team_id,
               home_team_label, away_team_label,
               pred_proba_home, pred_proba_draw, pred_proba_away,
               pred_home_goals, pred_away_goals, pred_result,
               actual_home_goals, actual_away_goals, actual_result,
               pred_winner_id
               {extra_cols}
        FROM wc2026_fixtures
        WHERE stage != 'GROUP_STAGE'
        ORDER BY match_date
    """)
    rows = c.fetchall()
    col_names = [d[0] for d in c.description]

    bracket = []
    for row in rows:
        m = dict(zip(col_names, row))

        # Déterminer le vainqueur affiché
        winner_id = m.get("pred_winner_id")
        if m.get("actual_result") == "H":
            winner_id = m["home_team_id"]
        elif m.get("actual_result") == "A":
            winner_id = m["away_team_id"]

        # Prob de victoire côté knockout (sans nul)
        prob_a = safe_float(m.get("prob_win_a") or m.get("pred_proba_home"), 3)
        prob_b = safe_float(m.get("prob_win_b") or m.get("pred_proba_away"), 3)
        if prob_a and prob_b:
            s = prob_a + prob_b
            if s > 0:
                prob_a = round(prob_a / s, 3)
                prob_b = round(prob_b / s, 3)

        bracket.append({
            "id":      m["fixture_id"],
            "stage":   m["stage"],
            "date":    m["match_date"],
            "team_a":  {
                "id":    m["home_team_id"],
                "name":  m["home_team_label"] or "TBD",
                "score": safe_int(m["actual_home_goals"]),
            },
            "team_b":  {
                "id":    m["away_team_id"],
                "name":  m["away_team_label"] or "TBD",
                "score": safe_int(m["actual_away_goals"]),
            },
            "prob_a":  prob_a,
            "prob_b":  prob_b,
            "winner":  {"id": winner_id} if winner_id else None,
        })

    # Probabilités globales de titre (depuis bracket_results si elle existe)
    title_probs = []
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='bracket_results'")
    if c.fetchone():
        c.execute("""
            SELECT team_id, team_label, fifa_ranking,
                   prob_r32, prob_r16, prob_qf, prob_sf, prob_final, prob_champion
            FROM bracket_results
            ORDER BY prob_champion DESC
        """)
        for r in c.fetchall():
            title_probs.append({
                "id":           r[0],
                "name":         r[1],
                "rank":         safe_int(r[2]),
                "prob_r32":     safe_float(r[3], 3),
                "prob_r16":     safe_float(r[4], 3),
                "prob_qf":      safe_float(r[5], 3),
                "prob_sf":      safe_float(r[6], 3),
                "prob_final":   safe_float(r[7], 3),
                "prob_champion":safe_float(r[8], 3),
            })

    return {"bracket": bracket, "title_probs": title_probs}


# ══════════════════════════════════════════════════════════════════
# training.json
# ══════════════════════════════════════════════════════════════════

def export_training(conn):
    df = pd.read_sql_query("""
        SELECT m.match_date as date,
               m.competition, m.stage, m.season,
               th.canonical_name as home, ta.canonical_name as away,
               m.home_goals, m.away_goals, m.result
        FROM matches m
        LEFT JOIN teams th ON th.team_id = m.home_team_id
        LEFT JOIN teams ta ON ta.team_id = m.away_team_id
        ORDER BY m.match_date DESC
    """, conn)

    by_comp = df.groupby("competition").agg(
        count=("result","count"),
        seasons=("season", lambda x: sorted(x.dropna().unique().tolist()))
    ).reset_index()

    result_dist = df["result"].value_counts().to_dict()

    matches = []
    for _, r in df.iterrows():
        matches.append({
            "date":        r["date"],
            "competition": r["competition"],
            "stage":       r["stage"],
            "season":      r["season"],
            "home":        r["home"],
            "away":        r["away"],
            "home_goals":  safe_int(r["home_goals"]),
            "away_goals":  safe_int(r["away_goals"]),
            "result":      r["result"],
        })

    return {
        "matches":      matches,
        "by_competition": [
            {"competition": r["competition"],
             "count":       int(r["count"]),
             "seasons":     r["seasons"]}
            for _, r in by_comp.iterrows()
        ],
        "result_dist": {k: int(v) for k, v in result_dist.items()},
    }


# ══════════════════════════════════════════════════════════════════
# model.json
# ══════════════════════════════════════════════════════════════════

def export_model(conn, metrics, features_meta):
    c = conn.cursor()

    # Walk-forward validation windows
    wf_windows = metrics.get("walk_forward_windows", [])

    # Feature importance
    feature_importance = features_meta.get("feature_importance", [])

    return {
        "metrics": {
            "accuracy":          metrics.get("accuracy"),
            "baseline_accuracy": metrics.get("baseline_accuracy"),
            "log_loss":          metrics.get("log_loss"),
            "brier_score":       metrics.get("brier_score"),
            "nb_train_matches":  metrics.get("nb_train_matches"),
            "calibration_method":metrics.get("calibration_method", "isotonic"),
        },
        "walk_forward": wf_windows,
        "feature_importance": feature_importance,
        "features_meta": {
            "total":      len(features_meta.get("features", [])),
            "categories": [
                {"category":"Classement FIFA","count":3,
                 "description":"Ranking home/away et écart normalisé. Source : publications officielles FIFA."},
                {"category":"Forme récente","count":12,
                 "description":"Points, buts marqués/encaissés sur 5 et 10 derniers matchs. Pondérés par confédération et récence."},
                {"category":"Head-to-head","count":6,
                 "description":"Historique des 5 dernières confrontations directes."},
                {"category":"Contexte","count":3,
                 "description":"Terrain neutre, poids de la compétition (CdM=1.0 → amicaux=0.3), phase éliminatoire."},
            ],
        },
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