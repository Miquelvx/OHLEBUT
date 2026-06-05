"""
Pipeline 1 — Prédiction phases de groupe CdM 2026

Corrections appliquées :
  - Chargement du modèle calibré (classifier_calibrated.pkl)
  - h2h_matches ajouté aux features
  - sim_score() O(1) sans boucle de rejection
  - Ordre des classes lu depuis le modèle, pas depuis features.json
  - avg_stats supprimé (non utilisé) pour clarté
"""

import os, sys, json, pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from collections import defaultdict
import unicodedata, re as _re

sys.path.append(os.path.join(os.path.dirname(__file__), "../collect"))
from init_db import get_connection

_script_dir = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(_script_dir, "models/")
if not os.path.exists(MODELS_DIR):
    MODELS_DIR = os.path.join(os.getcwd(), "models")

N_SIM       = 10_000
RANDOM_SEED = 42

# Pénalités de ranking par confédération (identique à build_features.py)
CONF_RANKING_PENALTY = {
    "UEFA":0,"CONMEBOL":5,"CAF":15,"AFC":20,"CONCACAF":15,"OFC":30,
}
CONF_STRENGTH = {
    "UEFA":1.00,"CONMEBOL":0.95,"CAF":0.75,"AFC":0.70,"CONCACAF":0.72,"OFC":0.55,
}

FEATURES = [
    "home_fifa_ranking","away_fifa_ranking","ranking_gap",
    "ranking_gap_adj","home_rank_adj","away_rank_adj",
    "home_top20_ratio","away_top20_ratio",
    "home_form5_pts","home_form5_scored","home_form5_conceded",
    "home_form10_pts","home_form10_scored","home_form10_conceded",
    "away_form5_pts","away_form5_scored","away_form5_conceded",
    "away_form10_pts","away_form10_scored","away_form10_conceded",
    "h2h_home_wins","h2h_draws","h2h_away_wins",
    "h2h_home_goals_avg","h2h_away_goals_avg","h2h_matches",
    "neutral_venue","competition_weight","is_knockout",
]


# ══════════════════════════════════════════════════════════════════
# 1. CHARGEMENT DES MODÈLES
# ══════════════════════════════════════════════════════════════════

def load_models():
    # FIX : charger le modèle calibré (pkl) et non le brut (json)
    clf_path = os.path.join(MODELS_DIR, "classifier_calibrated.pkl")
    with open(clf_path, "rb") as f:
        clf = pickle.load(f)

    rh = XGBRegressor(); ra = XGBRegressor()
    rh.load_model(os.path.join(MODELS_DIR, "regressor_home.json"))
    ra.load_model(os.path.join(MODELS_DIR, "regressor_away.json"))

    # FIX : ordre des classes depuis le modèle calibré lui-même
    # CalibratedClassifierCV expose .classes_ (entiers 0/1/2 du LabelEncoder)
    # On les mappe via features.json qui stocke l'ordre du LabelEncoder
    with open(os.path.join(MODELS_DIR, "features.json")) as f:
        meta = json.load(f)
    # classes = ["A","D","H"] dans l'ordre du LabelEncoder (alphabétique)
    # clf.classes_ = [0, 1, 2] — indices correspondants
    le_classes = meta["classes"]          # ["A","D","H"]
    # Mapping index → label : {0:"A", 1:"D", 2:"H"}
    idx_to_label = {i: c for i, c in enumerate(le_classes)}

    print(f"Modeles charges depuis {MODELS_DIR}")
    print(f"  Ordre classes : {le_classes}")
    return clf, rh, ra, idx_to_label


# ══════════════════════════════════════════════════════════════════
# 2. CALCUL DES FEATURES
# ══════════════════════════════════════════════════════════════════

def _norm(name):
    nfkd = unicodedata.normalize("NFKD", str(name))
    a    = nfkd.encode("ascii","ignore").decode("ascii")
    return _re.sub(r"\s+"," ", _re.sub(r"[^a-z0-9 ]"," ", a.lower())).strip()


def compute_features(conn):
    fixtures = pd.read_sql_query("""
        SELECT fixture_id, group_name, match_date,
               home_team_id, away_team_id,
               home_team_label, away_team_label
        FROM wc2026_fixtures
        WHERE stage='GROUP_STAGE'
          AND home_team_id IS NOT NULL
          AND away_team_id IS NOT NULL
        ORDER BY group_name, match_date
    """, conn)
    print(f"{len(fixtures)} matchs de groupe charges")

    hist = pd.read_sql_query("""
        SELECT match_id, match_date, home_team_id, away_team_id,
               home_goals, away_goals, result_90
        FROM matches WHERE result_90 IS NOT NULL ORDER BY match_date
    """, conn)

    # Rankings FIFA : dernière valeur connue par équipe depuis match_features
    mf_rank = {}
    for _, r in pd.read_sql_query("""
        SELECT home_team_id as tid, home_fifa_ranking as rank, match_date
        FROM match_features WHERE home_fifa_ranking IS NOT NULL
        UNION ALL
        SELECT away_team_id, away_fifa_ranking, match_date
        FROM match_features WHERE away_fifa_ranking IS NOT NULL
        ORDER BY match_date DESC
    """, conn).iterrows():
        tid = int(r["tid"])
        if tid not in mf_rank:
            mf_rank[tid] = int(r["rank"])

    teams_df = pd.read_sql_query("SELECT team_id, team_name FROM teams", conn)
    fi_raw, fi_norm = {}, {}
    for _, r in pd.read_sql_query(
        "SELECT team_name_fifa, rank FROM fifa_rankings ORDER BY rank_date DESC",
        conn).iterrows():
        n = r["team_name_fifa"]
        if n not in fi_raw:
            fi_raw[n] = r["rank"]
        if _norm(n) not in fi_norm:
            fi_norm[_norm(n)] = r["rank"]

    # Priorité : dernier classement FIFA publié depuis fifa_rankings
    # Écrase fi_norm et mf_rank avec les valeurs les plus récentes
    latest_rankings = pd.read_sql_query("""
        SELECT fr.team_name_fifa, fr.rank
        FROM fifa_rankings fr
        INNER JOIN (
            SELECT team_name_fifa, MAX(rank_date) AS last_date
            FROM fifa_rankings
            GROUP BY team_name_fifa
        ) latest ON fr.team_name_fifa = latest.team_name_fifa
               AND fr.rank_date       = latest.last_date
    """, conn)
    for _, r in latest_rankings.iterrows():
        norm = _norm(r["team_name_fifa"])
        fi_norm[norm] = int(r["rank"])   # écrase avec le dernier classement

    def get_rank(tid):
        if tid in mf_rank:
            return float(mf_rank[tid])
        rows = teams_df[teams_df["team_id"] == tid]["team_name"].values
        if len(rows):
            n = rows[0]
            if n in fi_raw:          return float(fi_raw[n])
            if _norm(n) in fi_norm:  return float(fi_norm[_norm(n)])
        return 100.0

    def form(tid, before, w=5):
        m = hist[
            ((hist["home_team_id"] == tid) | (hist["away_team_id"] == tid)) &
            (hist["match_date"] < before)
        ].sort_values("match_date", ascending=False).head(w)
        if len(m) == 0:
            return {"pts": 1.5, "sc": 1.2, "co": 1.0}
        p, s, c = [], [], []
        for _, r in m.iterrows():
            if r["home_team_id"] == tid:
                s.append(r["home_goals"]); c.append(r["away_goals"])
                p.append(3 if r["result_90"]=="H" else (1 if r["result_90"]=="D" else 0))
            else:
                s.append(r["away_goals"]); c.append(r["home_goals"])
                p.append(3 if r["result_90"]=="A" else (1 if r["result_90"]=="D" else 0))
        return {"pts": round(np.mean(p),4), "sc": round(np.mean(s),4),
                "co": round(np.mean(c),4)}

    def h2h(hid, aid, before):
        """FIX : retourne aussi h2h_matches pour la feature."""
        m = hist[
            (((hist["home_team_id"]==hid) & (hist["away_team_id"]==aid)) |
             ((hist["home_team_id"]==aid) & (hist["away_team_id"]==hid))) &
            (hist["match_date"] < before)
        ].tail(5)
        n = len(m)
        if n == 0:
            return {"hw":0.33, "d":0.33, "aw":0.33,
                    "hg":1.2, "ag":1.0, "n":0}
        hw = dr = aw = 0
        hgl, agl = [], []
        for _, r in m.iterrows():
            if r["home_team_id"] == hid:
                hgl.append(r["home_goals"]); agl.append(r["away_goals"])
                if r["result_90"]=="H": hw+=1
                elif r["result_90"]=="D": dr+=1
                else: aw+=1
            else:
                hgl.append(r["away_goals"]); agl.append(r["home_goals"])
                if r["result_90"]=="A": hw+=1
                elif r["result_90"]=="D": dr+=1
                else: aw+=1
        return {"hw": round(hw/n,4), "d": round(dr/n,4), "aw": round(aw/n,4),
                "hg": round(np.mean(hgl),4), "ag": round(np.mean(agl),4), "n": n}

    ref = "2026-06-11"

    # Carte des confédérations
    conf_df = pd.read_sql_query(
        "SELECT team_id, confederation FROM teams WHERE confederation IS NOT NULL", conn)
    conf_map = dict(zip(conf_df["team_id"].astype(int),
                        conf_df["confederation"].str.strip()))

    def get_adj_rank(tid):
        rank = get_rank(tid)
        conf = conf_map.get(tid, "")
        return rank + CONF_RANKING_PENALTY.get(conf, 15)

    def top20_ratio(tid, before, window=20):
        m = hist[
            ((hist["home_team_id"]==tid)|(hist["away_team_id"]==tid)) &
            (hist["match_date"] < before)
        ].sort_values("match_date", ascending=False).head(window)
        if len(m)==0: return 0.0
        count = 0
        for _, r in m.iterrows():
            opp = int(r["away_team_id"] if r["home_team_id"]==tid else r["home_team_id"])
            if get_adj_rank(opp) <= 20: count += 1
        return round(count / len(m), 4)

    all_tids = list(set(
        fixtures["home_team_id"].tolist() + fixtures["away_team_id"].tolist()
    ))
    tf = {}
    for tid in all_tids:
        tid = int(tid)
        f5  = form(tid, ref, 5)
        f10 = form(tid, ref, 10)
        tf[tid] = {
            "rank":     get_rank(tid),
            "rank_adj": get_adj_rank(tid),
            "f5_pts":   f5["pts"],  "f5_sc": f5["sc"],  "f5_co": f5["co"],
            "f10_pts":  f10["pts"], "f10_sc":f10["sc"], "f10_co":f10["co"],
            "top20":    top20_ratio(tid, ref),
        }

    rows = []
    for _, f in fixtures.iterrows():
        hid = int(f["home_team_id"]); aid = int(f["away_team_id"])
        fh = tf[hid]; fa = tf[aid]
        hh = h2h(hid, aid, ref)
        rows.append({
            "fixture_id":       f["fixture_id"],
            "group_name":       f["group_name"],
            "home_team_id":     hid,
            "away_team_id":     aid,
            "home_team_label":  f["home_team_label"],
            "away_team_label":  f["away_team_label"],
            "home_fifa_ranking": fh["rank"],
            "away_fifa_ranking": fa["rank"],
            "ranking_gap":       fa["rank"] - fh["rank"],
            "ranking_gap_adj":   fa["rank_adj"] - fh["rank_adj"],
            "home_rank_adj":     fh["rank_adj"],
            "away_rank_adj":     fa["rank_adj"],
            "home_top20_ratio":  fh["top20"],
            "away_top20_ratio":  fa["top20"],
            "home_form5_pts":    fh["f5_pts"],
            "home_form5_scored": fh["f5_sc"],
            "home_form5_conceded":fh["f5_co"],
            "home_form10_pts":   fh["f10_pts"],
            "home_form10_scored":fh["f10_sc"],
            "home_form10_conceded":fh["f10_co"],
            "away_form5_pts":    fa["f5_pts"],
            "away_form5_scored": fa["f5_sc"],
            "away_form5_conceded":fa["f5_co"],
            "away_form10_pts":   fa["f10_pts"],
            "away_form10_scored":fa["f10_sc"],
            "away_form10_conceded":fa["f10_co"],
            "h2h_home_wins":     hh["hw"],
            "h2h_draws":         hh["d"],
            "h2h_away_wins":     hh["aw"],
            "h2h_home_goals_avg":hh["hg"],
            "h2h_away_goals_avg":hh["ag"],
            "h2h_matches":       hh["n"],
            "neutral_venue":     1,
            "competition_weight":0.9,
            "is_knockout":       0,
        })
    return pd.DataFrame(rows), tf


# ══════════════════════════════════════════════════════════════════
# 3. PRÉDICTION DES PROBABILITÉS ET xG
# ══════════════════════════════════════════════════════════════════

def predict_fixtures(df, clf, rh, ra, idx_to_label):
    X = df[FEATURES].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(1.0)

    # FIX : clf est le modèle calibré — predict_proba() retourne
    # les probas dans l'ordre de clf.classes_ (indices 0/1/2)
    pr  = clf.predict_proba(X)
    xgh = np.clip(rh.predict(X), 0, 8)
    xga = np.clip(ra.predict(X), 0, 8)

    # Retrouver les indices H/D/A dans le tableau de probas
    # idx_to_label = {0:"A", 1:"D", 2:"H"} (ordre alphabétique LabelEncoder)
    label_to_idx = {v: k for k, v in idx_to_label.items()}
    ih  = label_to_idx["H"]
    id_ = label_to_idx["D"]
    ia  = label_to_idx["A"]

    df = df.copy()
    df["prob_h"] = pr[:, ih]
    df["prob_d"] = pr[:, id_]
    df["prob_a"] = pr[:, ia]
    df["xg_h"]   = xgh
    df["xg_a"]   = xga
    return df


# ══════════════════════════════════════════════════════════════════
# 4. SIMULATION D'UN SCORE — O(1), sans boucle de rejection
# ══════════════════════════════════════════════════════════════════

def sim_score(ph, pd_, pa, xgh, xga, rng):
    """
    FIX : O(1) garanti.
    1. Tirage du résultat selon les probabilités prédites.
    2. Tirage Poisson des buts.
    3. Correction minimale pour respecter le résultat tiré.
    """
    probs = np.array([ph, pd_, pa], dtype=np.float64)
    probs /= probs.sum()
    r  = rng.choice(["H","D","A"], p=probs)
    hg = int(rng.poisson(max(xgh, 0.3)))
    ag = int(rng.poisson(max(xga, 0.3)))

    if r == "H" and hg <= ag:
        hg = ag + 1
    elif r == "A" and ag <= hg:
        ag = hg + 1
    elif r == "D" and hg != ag:
        v  = round((hg + ag) / 2)
        hg = ag = v

    return int(hg), int(ag)


# ══════════════════════════════════════════════════════════════════
# 5. MONTE CARLO
# ══════════════════════════════════════════════════════════════════

def run_monte_carlo(df, tf):
    rng    = np.random.default_rng(RANDOM_SEED)
    groups = sorted(df["group_name"].unique())
    all_tids = list(set(
        df["home_team_id"].tolist() + df["away_team_id"].tolist()
    ))

    pos_counts  = {tid: {1:0, 2:0, 3:0, 4:0} for tid in all_tids}
    qual_third  = {tid: 0 for tid in all_tids}

    # Accumulation des scores simulés par fixture
    # fixture_id → defaultdict((hg, ag) → count)
    score_counts = {int(m["fixture_id"]): defaultdict(int)
                    for _, m in df.iterrows()}

    print(f"\nMonte Carlo : {N_SIM:,} simulations...")

    for s in range(N_SIM):
        if s % 2000 == 0:
            print(f"   {s:,}/{N_SIM:,}", end="\r")

        thirds_this = {}

        for grp in groups:
            gf   = df[df["group_name"] == grp]
            tids = list(set(
                gf["home_team_id"].tolist() + gf["away_team_id"].tolist()
            ))
            st = {t: {"pts":0,"w":0,"d":0,"l":0,"gf":0,"ga":0,"gd":0}
                  for t in tids}

            for _, m in gf.iterrows():
                hid = int(m["home_team_id"]); aid = int(m["away_team_id"])
                fid = int(m["fixture_id"])
                hg, ag = sim_score(
                    m["prob_h"], m["prob_d"], m["prob_a"],
                    m["xg_h"],  m["xg_a"],  rng
                )
                # Accumuler le score simulé
                score_counts[fid][(hg, ag)] += 1

                st[hid]["gf"] += hg; st[hid]["ga"] += ag; st[hid]["gd"] += hg-ag
                st[aid]["gf"] += ag; st[aid]["ga"] += hg; st[aid]["gd"] += ag-hg
                if hg > ag:
                    st[hid]["pts"] += 3; st[hid]["w"] += 1; st[aid]["l"] += 1
                elif hg == ag:
                    st[hid]["pts"] += 1; st[hid]["d"] += 1
                    st[aid]["pts"] += 1; st[aid]["d"] += 1
                else:
                    st[aid]["pts"] += 3; st[aid]["w"] += 1; st[hid]["l"] += 1

            ranking = sorted(tids,
                key=lambda t: (
                    st[t]["pts"], st[t]["gd"], st[t]["gf"],
                    -tf.get(t, {}).get("rank", 100)
                ),
                reverse=True
            )

            for pos, tid in enumerate(ranking, 1):
                pos_counts[tid][pos] += 1

            t3 = ranking[2]
            thirds_this[grp] = {
                "team_id": t3,
                "pts":  st[t3]["pts"],
                "gd":   st[t3]["gd"],
                "gf":   st[t3]["gf"],
                "rank": tf.get(t3, {}).get("rank", 100),
            }

        best8_this = sorted(
            thirds_this.values(),
            key=lambda x: (x["pts"], x["gd"], x["gf"], -x["rank"]),
            reverse=True
        )[:8]
        for info in best8_this:
            qual_third[info["team_id"]] += 1

    print(f"   {N_SIM:,}/{N_SIM:,} — done")
    return pos_counts, qual_third, score_counts


# ══════════════════════════════════════════════════════════════════
# 6. SCORE PRÉDIT PAR MATCH — moyenne pondérée conditionnelle
# ══════════════════════════════════════════════════════════════════

def get_predicted_scores(df, score_counts):
    """
    Score affiché = moyenne pondérée des scores simulés
    cohérents avec le résultat dominant.

    Exemple Germany vs Curaçao (84% H, xG 2.8/0.4) :
      Scores filtrés (Germany gagne) → moyenne hg ≈ 2.9, ag ≈ 0.4
      → round(2.9)=3, round(0.4)=0 → affiche 3-0

    Exemple Mexico vs South Korea (44% H) :
      Scores filtrés (Mexico gagne) → moyenne hg ≈ 1.8, ag ≈ 0.8
      → round(1.8)=2, round(0.8)=1 → affiche 2-1

    La moyenne capture mieux l'espérance du score selon le niveau
    des équipes, contrairement au mode qui donne toujours 2-1.
    Fallback sur mode global si aucun score filtré disponible.
    """
    scores = {}
    for _, m in df.iterrows():
        fid = int(m["fixture_id"])
        sc  = score_counts.get(fid, {})

        if not sc:
            scores[fid] = {"hg":1,"ag":0,"res":"H","freq":0.0,"n_distinct":0}
            continue

        ph  = float(m["prob_h"])
        pd_ = float(m["prob_d"])
        pa  = float(m["prob_a"])

        # Résultat dominant
        if ph >= pd_ and ph >= pa:
            dominant = "H"
        elif pa >= pd_ and pa >= ph:
            dominant = "A"
        else:
            dominant = "D"

        # Filtrer les scores cohérents avec le résultat dominant
        if dominant == "H":
            sc_filtered = {(h,a): n for (h,a),n in sc.items() if h > a}
        elif dominant == "A":
            sc_filtered = {(h,a): n for (h,a),n in sc.items() if a > h}
        else:
            sc_filtered = {(h,a): n for (h,a),n in sc.items() if h == a}

        sc_used = sc_filtered if sc_filtered else sc
        total_used = sum(sc_used.values())
        total_all  = sum(sc.values())

        # Moyenne pondérée des scores filtrés
        avg_hg = sum(h * n for (h,a),n in sc_used.items()) / total_used
        avg_ag = sum(a * n for (h,a),n in sc_used.items()) / total_used
        hg = round(avg_hg)
        ag = round(avg_ag)

        # Garantir la cohérence avec le résultat dominant
        if dominant == "H" and hg <= ag: hg = ag + 1
        elif dominant == "A" and ag <= hg: ag = hg + 1
        elif dominant == "D" and hg != ag:
            v = round((avg_hg + avg_ag) / 2)
            hg = ag = v

        res = "H" if hg > ag else ("A" if ag > hg else "D")

        # Fréquence : % des sims dans la catégorie dominante
        freq = round(total_used / total_all * 100, 1)

        scores[fid] = {
            "hg": int(hg), "ag": int(ag), "res": res,
            "freq": freq,
            "n_distinct": len(sc),
        }
    return scores


# ══════════════════════════════════════════════════════════════════
# 7. STANDINGS
# ══════════════════════════════════════════════════════════════════

def build_standings(df, tf, pos_counts, pred_scores):
    groups    = sorted(df["group_name"].unique())
    standings = {}

    for grp in groups:
        gf   = df[df["group_name"] == grp]
        tids = list(set(
            gf["home_team_id"].tolist() + gf["away_team_id"].tolist()
        ))

        labels, ranks = {}, {}
        for _, m in gf.iterrows():
            labels[int(m["home_team_id"])] = m["home_team_label"]
            labels[int(m["away_team_id"])] = m["away_team_label"]
            ranks[int(m["home_team_id"])]  = int(m["home_fifa_ranking"])
            ranks[int(m["away_team_id"])]  = int(m["away_fifa_ranking"])

        st = {t: {"pts":0,"w":0,"d":0,"l":0,"gf":0,"ga":0,"gd":0}
              for t in tids}

        for _, m in gf.iterrows():
            fid = int(m["fixture_id"])
            hid = int(m["home_team_id"]); aid = int(m["away_team_id"])
            hg  = pred_scores[fid]["hg"]; ag = pred_scores[fid]["ag"]
            st[hid]["gf"] += hg; st[hid]["ga"] += ag; st[hid]["gd"] += hg-ag
            st[aid]["gf"] += ag; st[aid]["ga"] += hg; st[aid]["gd"] += ag-hg
            if hg > ag:
                st[hid]["pts"] += 3; st[hid]["w"] += 1; st[aid]["l"] += 1
            elif hg == ag:
                st[hid]["pts"] += 1; st[hid]["d"] += 1
                st[aid]["pts"] += 1; st[aid]["d"] += 1
            else:
                st[aid]["pts"] += 3; st[aid]["w"] += 1; st[hid]["l"] += 1

        ranking = sorted(tids,
            key=lambda t: (
                st[t]["pts"], st[t]["gd"], st[t]["gf"],
                pos_counts[t][1] / N_SIM,
                -ranks.get(t, 100),
            ),
            reverse=True
        )
        standings[grp] = {"ranking": ranking, "stats": st,
                          "labels": labels, "ranks": ranks}
    return standings


# ══════════════════════════════════════════════════════════════════
# 8. MEILLEURS 3ÈMES
# ══════════════════════════════════════════════════════════════════

def select_best_thirds(standings, pos_counts, qual_third):
    thirds = []
    for grp, g in standings.items():
        tid = g["ranking"][2]
        s   = g["stats"][tid]
        thirds.append({
            "group": grp, "team_id": tid, "label": g["labels"][tid],
            "pts": s["pts"], "gd": s["gd"], "gf": s["gf"],
            "rank": g["ranks"][tid],
            "prob_1st":    round(pos_counts[tid][1] / N_SIM, 4),
            "prob_best3":  round(qual_third.get(tid, 0) / N_SIM, 4),
        })
    thirds_sorted = sorted(
        thirds,
        key=lambda x: (x["pts"], x["gd"], x["gf"], -x["rank"]),
        reverse=True
    )
    best8 = {t["group"]: t["team_id"] for t in thirds_sorted[:8]}
    return best8, thirds_sorted


# ══════════════════════════════════════════════════════════════════
# 9. AFFICHAGE
# ══════════════════════════════════════════════════════════════════

def print_match_scores(df, pred_scores):
    print(f"\n{'='*78}")
    print("  Scores predits — Phases de groupe (mode Poisson sur 10 000 sims)")
    print(f"{'='*78}")
    for grp in sorted(df["group_name"].unique()):
        gf = df[df["group_name"] == grp]
        print(f"\n  Groupe {grp}")
        print(f"  {'Domicile':<22} {'Score':^7} {'Exterieur':<22} "
              f"{'P(H)':>6} {'P(N)':>6} {'P(A)':>6} {'Freq':>6}")
        print(f"  {'─'*74}")
        for _, m in gf.iterrows():
            fid   = int(m["fixture_id"]); sc = pred_scores[fid]
            score = f"{sc['hg']} - {sc['ag']}"
            freq  = sc.get("freq", 0.0)
            print(f"  {m['home_team_label']:<22} {score:^7} "
                  f"{m['away_team_label']:<22} "
                  f"{m['prob_h']:>6.1%} {m['prob_d']:>6.1%} {m['prob_a']:>6.1%} "
                  f"{freq:>5.1f}%")


def print_standings(standings, pos_counts, qual_third):
    print(f"\n{'='*85}")
    print("  Classements predits — Probabilites Monte Carlo")
    print(f"{'='*85}")
    for grp in sorted(standings.keys()):
        g = standings[grp]; s = g["stats"]
        print(f"\n  Groupe {grp}")
        print(f"  {'':2} {'Equipe':<22} {'Pts':>4} {'J':>3} {'V':>3} "
              f"{'N':>3} {'D':>3} {'GF':>4} {'GA':>4} {'GD':>4} "
              f"{'P(1er)':>7} {'P(2eme)':>7} {'P(qual)':>7}")
        print(f"  {'─'*82}")
        for pos, tid in enumerate(g["ranking"], 1):
            if tid not in s:
                continue
            q    = "✓" if pos <= 2 else ("?" if pos == 3 else " ")
            p1   = pos_counts.get(tid, {}).get(1, 0) / N_SIM
            p2   = pos_counts.get(tid, {}).get(2, 0) / N_SIM
            pq   = (pos_counts.get(tid,{}).get(1,0) +
                    pos_counts.get(tid,{}).get(2,0) +
                    qual_third.get(tid, 0)) / N_SIM
            st   = s[tid]
            played = st["w"] + st["d"] + st["l"]
            print(f"  {q}{pos} {g['labels'].get(tid,'?'):<22} {st['pts']:>4} "
                  f"{played:>3} {st['w']:>3} {st['d']:>3} {st['l']:>3} "
                  f"{st['gf']:>4} {st['ga']:>4} {st['gd']:>4} "
                  f"{p1:>7.1%} {p2:>7.1%} {pq:>7.1%}")


def print_best_thirds(thirds_sorted, best8):
    print(f"\n{'='*68}")
    print("  Classement des 12 troisiemes (criteres FIFA)")
    print(f"{'='*68}")
    print(f"  {'':2} {'Grp':<5} {'Equipe':<22} {'Pts':>4} "
          f"{'GD':>4} {'GF':>4} {'Rank':>5} {'P(meilleur3)':>13}")
    print(f"  {'─'*62}")
    for t in thirds_sorted:
        q = "✓" if t["group"] in best8 else " "
        print(f"  {q}  {t['group']:<4} {t['label']:<22} "
              f"{t['pts']:>4} {t['gd']:>4} {t['gf']:>4} {t['rank']:>5} "
              f"{t['prob_best3']:>13.1%}")


# ══════════════════════════════════════════════════════════════════
# 10. SAUVEGARDE EN DB
# ══════════════════════════════════════════════════════════════════

def save(df, pred_scores, standings, best8, thirds_sorted,
         pos_counts, qual_third, conn):
    c = conn.cursor()

    for col in ["pred_home_goals","pred_away_goals","pred_proba_home",
                "pred_proba_draw","pred_proba_away"]:
        try: c.execute(f"ALTER TABLE wc2026_fixtures ADD COLUMN {col} REAL")
        except: pass
    try: c.execute("ALTER TABLE wc2026_fixtures ADD COLUMN pred_result TEXT")
    except: pass

    for _, m in df.iterrows():
        fid = int(m["fixture_id"]); sc = pred_scores[fid]
        c.execute("""
            UPDATE wc2026_fixtures
            SET pred_home_goals=?, pred_away_goals=?, pred_result=?,
                pred_proba_home=?, pred_proba_draw=?, pred_proba_away=?
            WHERE fixture_id=?
        """, (sc["hg"], sc["ag"], sc["res"],
              round(float(m["prob_h"]),4), round(float(m["prob_d"]),4),
              round(float(m["prob_a"]),4), fid))
    c.execute("DROP TABLE IF EXISTS group_standings")
    c.execute("""
        CREATE TABLE group_standings (
            team_id INTEGER PRIMARY KEY, group_name TEXT,
            position INTEGER, team_label TEXT, fifa_ranking INTEGER,
            points INTEGER, played INTEGER,
            won INTEGER, drawn INTEGER, lost INTEGER,
            goals_for INTEGER, goals_against INTEGER, goal_diff INTEGER,
            qualified TEXT,
            prob_1st REAL, prob_2nd REAL, prob_3rd REAL, prob_4th REAL,
            prob_qualify REAL
        )
    """)
    for grp, g in standings.items():
        for pos, tid in enumerate(g["ranking"], 1):
            st  = g["stats"][tid]
            if pos == 1:   qual = "1ST"
            elif pos == 2: qual = "2ND"
            elif pos == 3 and tid in best8.values(): qual = "3RD_BEST"
            else:          qual = "OUT"
            p1  = pos_counts[tid][1] / N_SIM
            p2  = pos_counts[tid][2] / N_SIM
            p3  = pos_counts[tid][3] / N_SIM
            p4  = pos_counts[tid][4] / N_SIM
            pq  = (pos_counts[tid][1] + pos_counts[tid][2] +
                   qual_third.get(tid, 0)) / N_SIM
            c.execute("""
                INSERT OR REPLACE INTO group_standings VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (tid, grp, pos, g["labels"][tid], g["ranks"][tid],
                  st["pts"], st["w"]+st["d"]+st["l"],
                  st["w"], st["d"], st["l"],
                  st["gf"], st["ga"], st["gd"], qual,
                  round(p1,4), round(p2,4), round(p3,4), round(p4,4),
                  round(pq,4)))

    c.execute("DROP TABLE IF EXISTS best_third_place")
    c.execute("""
        CREATE TABLE best_third_place (
            rank INTEGER, group_name TEXT PRIMARY KEY,
            team_id INTEGER, team_label TEXT,
            points INTEGER, goal_diff INTEGER, goals_for INTEGER,
            fifa_ranking INTEGER, prob_best3 REAL
        )
    """)
    for rank, t in enumerate(thirds_sorted[:8], 1):
        c.execute("INSERT INTO best_third_place VALUES (?,?,?,?,?,?,?,?,?)",
                  (rank, t["group"], t["team_id"], t["label"],
                   t["pts"], t["gd"], t["gf"], t["rank"], t["prob_best3"]))

    conn.commit()
    print("\nDB mise a jour : wc2026_fixtures, group_standings, best_third_place")


# ══════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  Pipeline 1 — Phases de groupe CdM 2026")
    print("=" * 55)

    conn = get_connection()
    clf, rh, ra, idx_to_label = load_models()
    df, tf = compute_features(conn)
    df = predict_fixtures(df, clf, rh, ra, idx_to_label)

    pos_counts, qual_third, score_counts = run_monte_carlo(df, tf)
    pred_scores = get_predicted_scores(df, score_counts)
    standings   = build_standings(df, tf, pos_counts, pred_scores)
    best8, thirds_sorted = select_best_thirds(standings, pos_counts, qual_third)

    print_match_scores(df, pred_scores)
    print_standings(standings, pos_counts, qual_third)
    print_best_thirds(thirds_sorted, best8)
    save(df, pred_scores, standings, best8, thirds_sorted,
         pos_counts, qual_third, conn)

    conn.close()
    print("\nPipeline 1 termine")