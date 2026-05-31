"""
Pipeline 1 — Prédiction phases de groupe CdM 2026

Logique :
  1. Features + prédiction prob H/D/A et xG pour les 72 matchs
  2. Monte Carlo 10 000 simulations :
     - Score Poisson → classement exact par groupe
     - Accumulation des stats par position (pts, GF, GA)
  3. Classement final = trié par prob_1st MC
  4. Stats affichées = moyennes conditionnelles :
     - Pour le 1er  : moyenne des stats quand l'équipe finit 1ère
     - Pour le 2ème : moyenne des stats quand l'équipe finit 2ème
     - etc.
     → Cohérence garantie : le 1er a toujours plus de pts que le 2ème
  5. Score affiché par match = xG arrondi, corrigé pour être 
     cohérent avec le résultat le plus probable

Sorties DB :
  - wc2026_fixtures  : pred_home_goals, pred_away_goals, pred_result,
                       pred_proba_home, pred_proba_draw, pred_proba_away
  - group_standings  : classement + probabilités + stats moyennes
  - best_third_place : 8 meilleurs 3èmes
"""

import os, sys, json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
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

FEATURES = [
    "home_fifa_ranking","away_fifa_ranking","ranking_gap",
    "home_form5_pts","home_form5_scored","home_form5_conceded",
    "home_form10_pts","home_form10_scored","home_form10_conceded",
    "away_form5_pts","away_form5_scored","away_form5_conceded",
    "away_form10_pts","away_form10_scored","away_form10_conceded",
    "h2h_home_wins","h2h_draws","h2h_away_wins",
    "h2h_home_goals_avg","h2h_away_goals_avg",
    "neutral_venue","competition_weight","is_knockout",
]


# ══════════════════════════════════════════════════════════════════
# 1. CHARGEMENT DES MODÈLES
# ══════════════════════════════════════════════════════════════════

def load_models():
    clf = XGBClassifier(); rh = XGBRegressor(); ra = XGBRegressor()
    clf.load_model(os.path.join(MODELS_DIR, "classifier.json"))
    rh.load_model(os.path.join(MODELS_DIR,  "regressor_home.json"))
    ra.load_model(os.path.join(MODELS_DIR,  "regressor_away.json"))
    with open(os.path.join(MODELS_DIR, "features.json")) as f:
        classes = json.load(f)["classes"]
    print("✅ Modèles chargés")
    return clf, rh, ra, classes


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
    print(f"📥 {len(fixtures)} matchs de groupe")

    hist = pd.read_sql_query("""
        SELECT match_id, match_date, home_team_id, away_team_id,
               home_goals, away_goals, result_90
        FROM matches WHERE result_90 IS NOT NULL ORDER BY match_date
    """, conn)

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
        if tid not in mf_rank: mf_rank[tid] = int(r["rank"])

    teams_df = pd.read_sql_query("SELECT team_id, team_name FROM teams", conn)
    fi_raw, fi_norm = {}, {}
    for _, r in pd.read_sql_query(
        "SELECT team_name_fifa, rank FROM fifa_rankings ORDER BY rank_date DESC",
        conn).iterrows():
        n = r["team_name_fifa"]
        if n not in fi_raw:        fi_raw[n]        = r["rank"]
        if _norm(n) not in fi_norm: fi_norm[_norm(n)] = r["rank"]

    def get_rank(tid):
        if tid in mf_rank: return float(mf_rank[tid])
        rows = teams_df[teams_df["team_id"]==tid]["team_name"].values
        if len(rows):
            n = rows[0]
            if n in fi_raw:          return float(fi_raw[n])
            if _norm(n) in fi_norm:  return float(fi_norm[_norm(n)])
        return 100.0

    def form(tid, before, w=5):
        m = hist[
            ((hist["home_team_id"]==tid)|(hist["away_team_id"]==tid)) &
            (hist["match_date"] < before)
        ].sort_values("match_date", ascending=False).head(w)
        if len(m)==0: return {"pts":1.5,"sc":1.2,"co":1.0}
        p,s,c=[],[],[]
        for _,r in m.iterrows():
            if r["home_team_id"]==tid:
                s.append(r["home_goals"]); c.append(r["away_goals"])
                p.append(3 if r["result_90"]=="H" else (1 if r["result_90"]=="D" else 0))
            else:
                s.append(r["away_goals"]); c.append(r["home_goals"])
                p.append(3 if r["result_90"]=="A" else (1 if r["result_90"]=="D" else 0))
        return {"pts":round(np.mean(p),4),"sc":round(np.mean(s),4),"co":round(np.mean(c),4)}

    def h2h(hid, aid, before):
        m = hist[
            (((hist["home_team_id"]==hid)&(hist["away_team_id"]==aid))|
             ((hist["home_team_id"]==aid)&(hist["away_team_id"]==hid))) &
            (hist["match_date"] < before)
        ].tail(5)
        if len(m)==0: return {"hw":0.33,"d":0.33,"aw":0.33,"hg":1.2,"ag":1.0}
        hw=dr=aw=0; hgl=[]; agl=[]
        for _,r in m.iterrows():
            if r["home_team_id"]==hid:
                hgl.append(r["home_goals"]); agl.append(r["away_goals"])
                if r["result_90"]=="H": hw+=1
                elif r["result_90"]=="D": dr+=1
                else: aw+=1
            else:
                hgl.append(r["away_goals"]); agl.append(r["home_goals"])
                if r["result_90"]=="A": hw+=1
                elif r["result_90"]=="D": dr+=1
                else: aw+=1
        n=len(m)
        return {"hw":round(hw/n,4),"d":round(dr/n,4),"aw":round(aw/n,4),
                "hg":round(np.mean(hgl),4),"ag":round(np.mean(agl),4)}

    ref = "2026-06-11"
    all_tids = list(set(fixtures["home_team_id"].tolist()+fixtures["away_team_id"].tolist()))
    tf = {}
    for tid in all_tids:
        tid=int(tid); f5=form(tid,ref,5); f10=form(tid,ref,10)
        tf[tid]={"rank":get_rank(tid),
                 "f5_pts":f5["pts"],"f5_sc":f5["sc"],"f5_co":f5["co"],
                 "f10_pts":f10["pts"],"f10_sc":f10["sc"],"f10_co":f10["co"]}

    rows=[]
    for _,f in fixtures.iterrows():
        hid=int(f["home_team_id"]); aid=int(f["away_team_id"])
        fh=tf[hid]; fa=tf[aid]; hh=h2h(hid,aid,ref)
        rows.append({
            "fixture_id":f["fixture_id"],"group_name":f["group_name"],
            "home_team_id":hid,"away_team_id":aid,
            "home_team_label":f["home_team_label"],"away_team_label":f["away_team_label"],
            "home_fifa_ranking":fh["rank"],"away_fifa_ranking":fa["rank"],
            "ranking_gap":fa["rank"]-fh["rank"],
            "home_form5_pts":fh["f5_pts"],"home_form5_scored":fh["f5_sc"],
            "home_form5_conceded":fh["f5_co"],"home_form10_pts":fh["f10_pts"],
            "home_form10_scored":fh["f10_sc"],"home_form10_conceded":fh["f10_co"],
            "away_form5_pts":fa["f5_pts"],"away_form5_scored":fa["f5_sc"],
            "away_form5_conceded":fa["f5_co"],"away_form10_pts":fa["f10_pts"],
            "away_form10_scored":fa["f10_sc"],"away_form10_conceded":fa["f10_co"],
            "h2h_home_wins":hh["hw"],"h2h_draws":hh["d"],"h2h_away_wins":hh["aw"],
            "h2h_home_goals_avg":hh["hg"],"h2h_away_goals_avg":hh["ag"],
            "neutral_venue":1,"competition_weight":0.9,"is_knockout":0,
        })
    return pd.DataFrame(rows), tf


# ══════════════════════════════════════════════════════════════════
# 3. PRÉDICTION DES PROBABILITÉS ET xG
# ══════════════════════════════════════════════════════════════════

def predict_fixtures(df, clf, rh, ra, classes):
    X = df[FEATURES].copy()
    for col in X.columns: X[col]=pd.to_numeric(X[col],errors="coerce").fillna(1.0)
    pr  = clf.predict_proba(X)
    xgh = np.clip(rh.predict(X),0,8)
    xga = np.clip(ra.predict(X),0,8)
    ih=classes.index("H"); id_=classes.index("D"); ia=classes.index("A")
    df=df.copy()
    df["prob_h"]=pr[:,ih]; df["prob_d"]=pr[:,id_]; df["prob_a"]=pr[:,ia]
    df["xg_h"]=xgh; df["xg_a"]=xga
    return df


# ══════════════════════════════════════════════════════════════════
# 4. SIMULATION D'UN SCORE
# ══════════════════════════════════════════════════════════════════

def sim_score(ph, pd_, pa, xgh, xga, rng):
    probs = np.array([ph,pd_,pa],dtype=np.float64); probs/=probs.sum()
    r = str(rng.choice(["H","D","A"],p=probs))
    for _ in range(20):
        hg=int(rng.poisson(max(xgh,0.3))); ag=int(rng.poisson(max(xga,0.3)))
        if r=="H" and hg>ag:  return hg,ag
        if r=="D" and hg==ag: return hg,ag
        if r=="A" and ag>hg:  return hg,ag
    if r=="H": return max(1,int(rng.poisson(xgh))),0
    if r=="A": return 0,max(1,int(rng.poisson(xga)))
    v=int(rng.poisson((xgh+xga)/2)); return v,v


# ══════════════════════════════════════════════════════════════════
# 5. MONTE CARLO — accumulation stats par position
# ══════════════════════════════════════════════════════════════════

def run_monte_carlo(df, tf):
    rng    = np.random.default_rng(RANDOM_SEED)
    groups = sorted(df["group_name"].unique())
    all_tids = list(set(df["home_team_id"].tolist()+df["away_team_id"].tolist()))

    # Compteurs de positions
    pos_counts = {tid:{1:0,2:0,3:0,4:0} for tid in all_tids}

    # Stats cumulées globales (toutes simulations confondues)
    sum_pts = {tid:0.0 for tid in all_tids}
    sum_gf  = {tid:0.0 for tid in all_tids}
    sum_ga  = {tid:0.0 for tid in all_tids}
    sum_w   = {tid:0.0 for tid in all_tids}
    sum_d   = {tid:0.0 for tid in all_tids}
    sum_l   = {tid:0.0 for tid in all_tids}

    # Meilleurs 3èmes
    qual_third = {tid:0 for tid in all_tids}

    print(f"\n🎲 Monte Carlo : {N_SIM:,} simulations...")

    for s in range(N_SIM):
        if s%2000==0: print(f"   {s:,}/{N_SIM:,}",end="\r")

        thirds_this = {}

        for grp in groups:
            gf   = df[df["group_name"]==grp]
            tids = list(set(gf["home_team_id"].tolist()+gf["away_team_id"].tolist()))
            st   = {t:{"pts":0,"w":0,"d":0,"l":0,"gf":0,"ga":0,"gd":0} for t in tids}

            for _,m in gf.iterrows():
                hid=int(m["home_team_id"]); aid=int(m["away_team_id"])
                hg,ag=sim_score(m["prob_h"],m["prob_d"],m["prob_a"],
                                m["xg_h"],m["xg_a"],rng)
                st[hid]["gf"]+=hg; st[hid]["ga"]+=ag; st[hid]["gd"]+=hg-ag
                st[aid]["gf"]+=ag; st[aid]["ga"]+=hg; st[aid]["gd"]+=ag-hg
                if hg>ag:
                    st[hid]["pts"]+=3; st[hid]["w"]+=1; st[aid]["l"]+=1
                elif hg==ag:
                    st[hid]["pts"]+=1; st[hid]["d"]+=1
                    st[aid]["pts"]+=1; st[aid]["d"]+=1
                else:
                    st[aid]["pts"]+=3; st[aid]["w"]+=1; st[hid]["l"]+=1

            # Classement : pts → GD → GF → ranking FIFA
            ranking = sorted(tids,
                key=lambda t:(st[t]["pts"],st[t]["gd"],
                              st[t]["gf"],-tf.get(t,{}).get("rank",100)),
                reverse=True)

            # Compter les positions et accumuler les stats globales
            for pos,tid in enumerate(ranking,1):
                pos_counts[tid][pos]+=1
                sum_pts[tid]+=st[tid]["pts"]
                sum_gf[tid] +=st[tid]["gf"]
                sum_ga[tid] +=st[tid]["ga"]
                sum_w[tid]  +=st[tid]["w"]
                sum_d[tid]  +=st[tid]["d"]
                sum_l[tid]  +=st[tid]["l"]

            # 3èmes pour meilleur 3ème
            t3=ranking[2]
            thirds_this[grp]={
                "team_id":t3,"pts":st[t3]["pts"],
                "gd":st[t3]["gd"],"gf":st[t3]["gf"],
                "rank":tf.get(t3,{}).get("rank",100),
            }

        # Sélection 8 meilleurs 3èmes
        best8_this = sorted(thirds_this.values(),
            key=lambda x:(x["pts"],x["gd"],x["gf"],-x["rank"]),
            reverse=True)[:8]
        for info in best8_this:
            qual_third[info["team_id"]]+=1

    print(f"   ✅ {N_SIM:,} simulations terminées")
    avg_stats = {tid:{
        "pts": round(sum_pts[tid]/N_SIM),
        "gf":  round(sum_gf[tid]/N_SIM),
        "ga":  round(sum_ga[tid]/N_SIM),
        "w":   round(sum_w[tid]/N_SIM),
        "d":   round(sum_d[tid]/N_SIM),
        "l":   round(sum_l[tid]/N_SIM),
    } for tid in all_tids}
    return pos_counts, qual_third, avg_stats


# ══════════════════════════════════════════════════════════════════
# 6. CONSTRUCTION DES STANDINGS
# ══════════════════════════════════════════════════════════════════

def build_standings(df, tf, pos_counts, pred_scores, qual_third):
    """
    Stats calculées depuis les scores prédits affichés → cohérence parfaite.
    Classement trié par prob_1st MC.
    """
    groups = sorted(df["group_name"].unique())
    standings = {}

    for grp in groups:
        gf   = df[df["group_name"]==grp]
        tids = list(set(gf["home_team_id"].tolist()+gf["away_team_id"].tolist()))

        labels={}; ranks={}
        for _,m in gf.iterrows():
            labels[int(m["home_team_id"])]=m["home_team_label"]
            labels[int(m["away_team_id"])]=m["away_team_label"]
            ranks[int(m["home_team_id"])]=int(m["home_fifa_ranking"])
            ranks[int(m["away_team_id"])]=int(m["away_fifa_ranking"])

        # Calculer les stats depuis les scores prédits
        st = {t:{"pts":0,"w":0,"d":0,"l":0,"gf":0,"ga":0,"gd":0} for t in tids}
        for _,m in gf.iterrows():
            fid=int(m["fixture_id"]); hid=int(m["home_team_id"]); aid=int(m["away_team_id"])
            hg=pred_scores[fid]["hg"]; ag=pred_scores[fid]["ag"]
            st[hid]["gf"]+=hg; st[hid]["ga"]+=ag; st[hid]["gd"]+=hg-ag
            st[aid]["gf"]+=ag; st[aid]["ga"]+=hg; st[aid]["gd"]+=ag-hg
            if hg>ag:
                st[hid]["pts"]+=3; st[hid]["w"]+=1; st[aid]["l"]+=1
            elif hg==ag:
                st[hid]["pts"]+=1; st[hid]["d"]+=1
                st[aid]["pts"]+=1; st[aid]["d"]+=1
            else:
                st[aid]["pts"]+=3; st[aid]["w"]+=1; st[hid]["l"]+=1

        # Classement :
        # 1. Points (depuis les scores prédits) — critère principal
        # 2. GD, GF — départage standard FIFA
        # 3. prob_1st MC — uniquement pour départager les égalités parfaites
        # 4. Ranking FIFA — dernier recours
        ranking = sorted(tids,
            key=lambda t:(
                st[t]["pts"],              # points (source principale)
                st[t]["gd"],               # différence de buts
                st[t]["gf"],               # buts marqués
                pos_counts[t][1]/N_SIM,    # prob_1st MC (tiebreak égalités)
                -ranks.get(t,100),         # ranking FIFA (dernier recours)
            ),
            reverse=True)

        standings[grp] = {"ranking":ranking,"stats":st,
                          "labels":labels,"ranks":ranks}

    return standings


# ══════════════════════════════════════════════════════════════════
# 7. SCORE PRÉDIT PAR MATCH (cohérent avec prob dominante)
# ══════════════════════════════════════════════════════════════════

def get_predicted_scores(df):
    """
    Score prédit = xG arrondi, corrigé pour être cohérent
    avec le résultat le plus probable (H/D/A).
    """
    scores = {}
    for _,m in df.iterrows():
        fid=int(m["fixture_id"])
        hg=round(float(m["xg_h"])); ag=round(float(m["xg_a"]))
        ph=float(m["prob_h"]); pd_=float(m["prob_d"]); pa=float(m["prob_a"])

        # Résultat dominant
        if ph >= pd_ and ph >= pa:      # Victoire domicile
            if hg <= ag: hg = ag + 1
        elif pa >= pd_ and pa >= ph:    # Victoire extérieur
            if ag <= hg: ag = hg + 1
        else:                           # Nul
            if hg != ag: hg = ag = min(hg, ag)

        res = "H" if hg>ag else ("A" if ag>hg else "D")
        scores[fid] = {"hg":int(hg),"ag":int(ag),"res":res}
    return scores


# ══════════════════════════════════════════════════════════════════
# 8. SÉLECTION DES 8 MEILLEURS 3ÈMES
# ══════════════════════════════════════════════════════════════════

def select_best_thirds(standings, pos_counts, qual_third):
    thirds = []
    for grp, g in standings.items():
        tid   = g["ranking"][2]   # 3ème du classement
        s     = g["stats"][tid]
        thirds.append({
            "group":grp,"team_id":tid,"label":g["labels"][tid],
            "pts":s["pts"],"gd":s["gd"],"gf":s["gf"],
            "rank":g["ranks"][tid],
            "prob_1st":round(pos_counts[tid][1]/N_SIM,4),
            "prob_best3":round(qual_third.get(tid,0)/N_SIM,4),
        })

    thirds_sorted = sorted(thirds,
        key=lambda x:(x["pts"],x["gd"],x["gf"],-x["rank"]),
        reverse=True)

    best8 = {t["group"]: t["team_id"] for t in thirds_sorted[:8]}
    return best8, thirds_sorted


# ══════════════════════════════════════════════════════════════════
# 9. AFFICHAGE
# ══════════════════════════════════════════════════════════════════

def print_match_scores(df, pred_scores):
    print(f"\n{'='*72}")
    print("  Scores prédits — Phases de groupe")
    print(f"{'='*72}")
    for grp in sorted(df["group_name"].unique()):
        gf=df[df["group_name"]==grp]
        print(f"\n  Groupe {grp}")
        print(f"  {'Domicile':<22} {'Score':^7} {'Extérieur':<22} "
              f"{'P(H)':>6} {'P(N)':>6} {'P(A)':>6}")
        print(f"  {'─'*68}")
        for _,m in gf.iterrows():
            fid=int(m["fixture_id"]); sc=pred_scores[fid]
            score=f"{sc['hg']} - {sc['ag']}"
            print(f"  {m['home_team_label']:<22} {score:^7} "
                  f"{m['away_team_label']:<22} "
                  f"{m['prob_h']:>6.1%} {m['prob_d']:>6.1%} {m['prob_a']:>6.1%}")


def print_standings(standings, pos_counts, qual_third):
    print(f"\n{'='*85}")
    print("  Classements prédits — Probabilités Monte Carlo")
    print(f"{'='*85}")
    for grp in sorted(standings.keys()):
        g=standings[grp]; s=g["stats"]
        print(f"\n  Groupe {grp}")
        print(f"  {'':2} {'Équipe':<22} {'Pts':>4} {'J':>3} {'V':>3} "
              f"{'N':>3} {'D':>3} {'GF':>4} {'GA':>4} {'GD':>4} "
              f"{'P(1er)':>7} {'P(2ème)':>7} {'P(qual)':>7}")
        print(f"  {'─'*82}")
        for pos,tid in enumerate(g["ranking"],1):
            if tid not in s:
                print(f"  ⚠️  Équipe {tid} manquante dans stats")
                continue
            q="✓" if pos<=2 else ("?" if pos==3 else " ")
            p1=pos_counts.get(tid,{}).get(1,0)/N_SIM
            p2=pos_counts.get(tid,{}).get(2,0)/N_SIM
            pq=(pos_counts.get(tid,{}).get(1,0)+
                pos_counts.get(tid,{}).get(2,0)+
                qual_third.get(tid,0))/N_SIM
            st=s[tid]; played=st["w"]+st["d"]+st["l"]
            print(f"  {q}{pos} {g['labels'].get(tid,'?'):<22} {st['pts']:>4} "
                  f"{played:>3} {st['w']:>3} {st['d']:>3} {st['l']:>3} "
                  f"{st['gf']:>4} {st['ga']:>4} {st['gd']:>4} "
                  f"{p1:>7.1%} {p2:>7.1%} {pq:>7.1%}")


def print_best_thirds(thirds_sorted, best8):
    print(f"\n{'='*68}")
    print("  Classement des 12 troisièmes (critères FIFA)")
    print(f"{'='*68}")
    print(f"  {'':2} {'Grp':<5} {'Équipe':<22} {'Pts':>4} "
          f"{'GD':>4} {'GF':>4} {'Rank':>5} {'P(meilleur3)':>13}")
    print(f"  {'─'*62}")
    for t in thirds_sorted:
        q="✓" if t["group"] in best8 else " "
        print(f"  {q}  {t['group']:<4} {t['label']:<22} "
              f"{t['pts']:>4} {t['gd']:>4} {t['gf']:>4} {t['rank']:>5} "
              f"{t['prob_best3']:>13.1%}")


# ══════════════════════════════════════════════════════════════════
# 10. SAUVEGARDE EN DB
# ══════════════════════════════════════════════════════════════════

def save(df, pred_scores, standings, best8, thirds_sorted,
         pos_counts, qual_third, conn):
    c = conn.cursor()

    # wc2026_fixtures
    for col in ["pred_home_goals","pred_away_goals","pred_proba_home",
                "pred_proba_draw","pred_proba_away"]:
        try: c.execute(f"ALTER TABLE wc2026_fixtures ADD COLUMN {col} REAL")
        except: pass
    try: c.execute("ALTER TABLE wc2026_fixtures ADD COLUMN pred_result TEXT")
    except: pass

    for _,m in df.iterrows():
        fid=int(m["fixture_id"]); sc=pred_scores[fid]
        c.execute("""
            UPDATE wc2026_fixtures
            SET pred_home_goals=?,pred_away_goals=?,pred_result=?,
                pred_proba_home=?,pred_proba_draw=?,pred_proba_away=?
            WHERE fixture_id=?
        """,(sc["hg"],sc["ag"],sc["res"],
             round(float(m["prob_h"]),4),round(float(m["prob_d"]),4),
             round(float(m["prob_a"]),4),fid))

    # group_standings
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
    for grp,g in standings.items():
        for pos,tid in enumerate(g["ranking"],1):
            st=g["stats"][tid]
            if pos==1:   qual="1ST"
            elif pos==2: qual="2ND"
            elif pos==3 and tid in best8.values(): qual="3RD_BEST"
            else:        qual="OUT"
            p1=pos_counts[tid][1]/N_SIM; p2=pos_counts[tid][2]/N_SIM
            p3=pos_counts[tid][3]/N_SIM; p4=pos_counts[tid][4]/N_SIM
            pq=(pos_counts[tid][1]+pos_counts[tid][2]+qual_third.get(tid,0))/N_SIM
            c.execute("""
                INSERT OR REPLACE INTO group_standings VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,(tid,grp,pos,g["labels"][tid],g["ranks"][tid],
                 st["pts"],st["w"]+st["d"]+st["l"],
                 st["w"],st["d"],st["l"],
                 st["gf"],st["ga"],st["gd"],qual,
                 round(p1,4),round(p2,4),round(p3,4),round(p4,4),round(pq,4)))

    # best_third_place
    c.execute("DROP TABLE IF EXISTS best_third_place")
    c.execute("""
        CREATE TABLE best_third_place (
            rank INTEGER, group_name TEXT PRIMARY KEY,
            team_id INTEGER, team_label TEXT,
            points INTEGER, goal_diff INTEGER, goals_for INTEGER,
            fifa_ranking INTEGER, prob_best3 REAL
        )
    """)
    for rank,t in enumerate(thirds_sorted[:8],1):
        c.execute("INSERT INTO best_third_place VALUES (?,?,?,?,?,?,?,?,?)",
                  (rank,t["group"],t["team_id"],t["label"],
                   t["pts"],t["gd"],t["gf"],t["rank"],t["prob_best3"]))

    conn.commit()
    print("\n✅ DB mise à jour : wc2026_fixtures, group_standings, best_third_place")


# ══════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════

if __name__=="__main__":
    print("="*55)
    print("  Pipeline 1 — Phases de groupe CdM 2026")
    print("="*55)

    conn = get_connection()
    clf,rh,ra,classes = load_models()
    df,tf = compute_features(conn)
    df    = predict_fixtures(df,clf,rh,ra,classes)

    pos_counts, qual_third, avg_stats = run_monte_carlo(df,tf)
    pred_scores = get_predicted_scores(df)
    standings = build_standings(df,tf,pos_counts,pred_scores,qual_third)
    best8, thirds_sorted = select_best_thirds(standings,pos_counts,qual_third)

    print_match_scores(df, pred_scores)
    print_standings(standings, pos_counts, qual_third)
    print_best_thirds(thirds_sorted, best8)

    save(df,pred_scores,standings,best8,thirds_sorted,pos_counts,qual_third,conn)

    conn.close()
    print("\n🎉 Pipeline 1 terminé")