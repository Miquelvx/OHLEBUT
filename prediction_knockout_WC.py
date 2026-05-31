"""
Pipeline 2 — Prédiction phases finales CdM 2026

Étapes :
  1. Lire group_standings + best_third_place
  2. Construire le bracket R16 (attribution des 3èmes par contrainte croissante)
  3. Afficher le bracket pour validation
  4. Monte Carlo 10 000 simulations :
     - Tirer les qualifiés selon prob_1st/prob_2nd
     - Construire le bracket de la simulation
     - Simuler chaque match avec score Poisson
     - Propager jusqu'à la finale
  5. Afficher le bracket déterministe (matchup le plus fréquent + score prédit)
  6. Afficher les probabilités de progression par équipe

Bracket officiel FIFA CdM 2026 :
  R16 :
    M1  : 1erE  vs 3ème(A/B/C/D/F)   ┐
    M2  : 1erI  vs 3ème(C/D/F/G/H)   ┘→ R8_1 → QF1 → SF1 → FINALE
    M3  : 2èmeA vs 2èmeB              ┐
    M4  : 1erF  vs 2èmeC              ┘→ R8_2 → QF1
    M5  : 2èmeK vs 2èmeL              ┐
    M6  : 1erH  vs 2èmeJ              ┘→ R8_3 → QF2 → SF1
    M7  : 1erD  vs 3ème(B/E/F/I/J)   ┐
    M8  : 1erG  vs 3ème(A/E/H/I/J)   ┘→ R8_4 → QF2
    M9  : 1erC  vs 2èmeF              ┐
    M10 : 2èmeE vs 2èmeI              ┘→ R8_5 → QF3 → SF2 → FINALE
    M11 : 1erA  vs 3ème(C/E/F/H/I)   ┐
    M12 : 1erL  vs 3ème(E/H/I/J/K)   ┘→ R8_6 → QF3
    M13 : 1erJ  vs 2èmeH              ┐
    M14 : 1erB  vs 3ème(E/F/G/I/J)   ┘→ R8_7 → QF4 → SF2
    M15 : 1erK  vs 3ème(D/E/I/J/L)   ┐
    M16 : 2èmeD vs 2èmeG              ┘→ R8_8 → QF4

  R8  : W(M1)vsW(M2), W(M3)vsW(M4), W(M5)vsW(M6), W(M7)vsW(M8),
         W(M9)vsW(M10), W(M11)vsW(M12), W(M13)vsW(M14), W(M15)vsW(M16)
  QF  : W(R8_1)vsW(R8_2), W(R8_3)vsW(R8_4),
         W(R8_5)vsW(R8_6), W(R8_7)vsW(R8_8)
  SF  : W(QF1)vsW(QF2), W(QF3)vsW(QF4)
  3RD : L(SF1) vs L(SF2)
  FIN : W(SF1) vs W(SF2)
"""

import os, sys, json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from collections import defaultdict

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

# ── Bracket officiel FIFA ──────────────────────────────────────────
# Groupes éligibles pour chaque slot de 3ème
THIRD_ELIGIBLE = {
    "M1":  ["A","B","C","D","F"],
    "M2":  ["C","D","F","G","H"],
    "M7":  ["B","E","F","I","J"],
    "M8":  ["A","E","H","I","J"],
    "M11": ["C","E","F","H","I"],
    "M12": ["E","H","I","J","K"],
    "M15": ["E","F","G","I","J"],
    "M16": ["D","E","I","J","L"],
}

# Matchs fixes (sans 3èmes)
FIXED_R16 = {
    "M3":  ("2A","2B"),  "M4":  ("1F","2C"),
    "M5":  ("2K","2L"),  "M6":  ("1H","2J"),
    "M9":  ("1C","2F"),  "M10": ("2E","2I"),
    "M13": ("1J","2H"),  "M14": ("2D","2G"),
}

# Matchs avec 3èmes (slot → 1er du groupe)
THIRD_R16 = {
    "M1": "1E", "M2": "1I", "M7": "1D", "M8":  "1G",
    "M11":"1A", "M12":"1L", "M15":"1B", "M16": "1K",
}

# Arbre complet du bracket
R8_PAIRS  = [("M1","M2"),("M3","M4"),("M5","M6"),("M7","M8"),
             ("M9","M10"),("M11","M12"),("M13","M14"),("M15","M16")]
R8_IDS    = ["R8_1","R8_2","R8_3","R8_4","R8_5","R8_6","R8_7","R8_8"]

QF_PAIRS  = [("R8_1","R8_2"),("R8_3","R8_4"),
             ("R8_5","R8_6"),("R8_7","R8_8")]
QF_IDS    = ["QF1","QF2","QF3","QF4"]

SF_PAIRS  = [("QF1","QF2"),("QF3","QF4")]
SF_IDS    = ["SF1","SF2"]


# ══════════════════════════════════════════════════════════════════
# 1. CHARGEMENT
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


def load_group_data(conn):
    """Charge les classements de groupe et les meilleurs 3èmes."""
    standings = pd.read_sql_query("""
        SELECT team_id, group_name, position, team_label, fifa_ranking,
               points, won, drawn, lost, goals_for, goals_against, goal_diff,
               prob_1st, prob_2nd, prob_3rd, prob_4th, prob_qualify
        FROM group_standings ORDER BY group_name, position
    """, conn)
    if len(standings) == 0:
        print("❌ group_standings vide — lance d'abord predict_group_stage.py")
        sys.exit(1)

    thirds = pd.read_sql_query("""
        SELECT rank, group_name, team_id, team_label,
               points, goal_diff, goals_for, fifa_ranking
        FROM best_third_place ORDER BY rank
    """, conn)

    # Features par équipe depuis match_features
    mf = pd.read_sql_query("""
        SELECT home_team_id as tid,
               AVG(CAST(home_fifa_ranking    AS FLOAT)) as rank,
               AVG(CAST(home_form5_pts       AS FLOAT)) as f5_pts,
               AVG(CAST(home_form5_scored    AS FLOAT)) as f5_sc,
               AVG(CAST(home_form5_conceded  AS FLOAT)) as f5_co,
               AVG(CAST(home_form10_pts      AS FLOAT)) as f10_pts,
               AVG(CAST(home_form10_scored   AS FLOAT)) as f10_sc,
               AVG(CAST(home_form10_conceded AS FLOAT)) as f10_co
        FROM match_features
        WHERE home_team_id IN (SELECT team_id FROM teams WHERE is_wc2026=1)
          AND home_fifa_ranking IS NOT NULL
        GROUP BY home_team_id
        UNION
        SELECT away_team_id,
               AVG(CAST(away_fifa_ranking    AS FLOAT)),
               AVG(CAST(away_form5_pts       AS FLOAT)),
               AVG(CAST(away_form5_scored    AS FLOAT)),
               AVG(CAST(away_form5_conceded  AS FLOAT)),
               AVG(CAST(away_form10_pts      AS FLOAT)),
               AVG(CAST(away_form10_scored   AS FLOAT)),
               AVG(CAST(away_form10_conceded AS FLOAT))
        FROM match_features
        WHERE away_team_id IN (SELECT team_id FROM teams WHERE is_wc2026=1)
          AND away_fifa_ranking IS NOT NULL
        GROUP BY away_team_id
    """, conn)

    tf = {}
    seen = set()
    for _, r in mf.iterrows():
        tid = int(r["tid"])
        if tid not in seen:
            seen.add(tid)
            tf[tid] = {
                "rank":    float(r["rank"]    or 100),
                "f5_pts":  float(r["f5_pts"]  or 1.5),
                "f5_sc":   float(r["f5_sc"]   or 1.2),
                "f5_co":   float(r["f5_co"]   or 1.0),
                "f10_pts": float(r["f10_pts"] or 1.5),
                "f10_sc":  float(r["f10_sc"]  or 1.2),
                "f10_co":  float(r["f10_co"]  or 1.0),
            }
    # Fallback depuis standings
    for _, r in standings.iterrows():
        tid = int(r["team_id"])
        if tid not in tf:
            tf[tid] = {"rank":float(r["fifa_ranking"] or 100),
                       "f5_pts":1.5,"f5_sc":1.2,"f5_co":1.0,
                       "f10_pts":1.5,"f10_sc":1.2,"f10_co":1.0}

    label_map = dict(zip(standings["team_id"].astype(int),
                         standings["team_label"]))

    print(f"📥 {len(standings)} équipes, {len(thirds)} meilleurs 3èmes")
    return standings, thirds, tf, label_map


# ══════════════════════════════════════════════════════════════════
# 2. CONSTRUCTION DU BRACKET R16
# ══════════════════════════════════════════════════════════════════

def assign_thirds(thirds_df, qualified_groups):
    """
    Assigne les 8 meilleurs 3èmes aux slots du bracket.
    Algorithme : backtracking pour trouver une assignation valide
    qui respecte toutes les contraintes d'éligibilité.

    On essaie d'abord les meilleurs 3èmes (par rang FIFA) pour
    chaque slot en partant du slot le plus contraint.

    thirds_df : DataFrame trié par rang (meilleur 3ème en premier)
    qualified_groups : set des groupes dont le 3ème est qualifié

    Retourne : dict {slot: group_name}
    """
    available = {row["group_name"]: int(row["rank"])
                 for _, row in thirds_df.iterrows()
                 if row["group_name"] in qualified_groups}

    # Trier les slots par nombre d'éligibles croissant (plus contraint en premier)
    slots = sorted(THIRD_ELIGIBLE.keys(),
        key=lambda s: len([g for g in THIRD_ELIGIBLE[s] if g in available]))

    def backtrack(idx, assignment, used):
        if idx == len(slots):
            return assignment.copy()
        slot = slots[idx]
        eligible = [g for g in THIRD_ELIGIBLE[slot]
                    if g in available and g not in used]
        # Trier par rang FIFA (meilleur d'abord)
        eligible.sort(key=lambda g: available[g])
        for grp in eligible:
            assignment[slot] = grp
            used.add(grp)
            result = backtrack(idx+1, assignment, used)
            if result is not None:
                return result
            del assignment[slot]
            used.remove(grp)
        return None  # Pas de solution avec ce chemin

    result = backtrack(0, {}, set())
    if result is None:
        print("  ⚠️  Impossible de trouver une assignation valide — assignation partielle")
        result = {}
    return result


def build_r16_bracket(standings, thirds_df, label_map):
    """
    Construit le bracket des 16èmes de finale.
    Retourne : dict {match_id: (team_id_a, team_id_b)}
    """
    # Index des qualifiés par groupe et position
    def get_team(pos_code):
        """Résout '1A', '2B' en team_id."""
        pos   = int(pos_code[0])
        grp   = pos_code[1]
        row   = standings[(standings["group_name"]==grp) &
                          (standings["position"]==pos)]
        if len(row) == 0: return None
        return int(row.iloc[0]["team_id"])

    # Groupes qualifiés pour les 3èmes
    qualified_groups = set(thirds_df["group_name"].tolist())
    assignment = assign_thirds(thirds_df, qualified_groups)

    bracket = {}

    # Matchs fixes
    for mid, (code_a, code_b) in FIXED_R16.items():
        ta = get_team(code_a); tb = get_team(code_b)
        if ta and tb: bracket[mid] = (ta, tb)

    # Matchs avec 3èmes
    for mid, code_a in THIRD_R16.items():
        ta = get_team(code_a)
        grp = assignment.get(mid)
        if grp:
            row = thirds_df[thirds_df["group_name"]==grp]
            tb  = int(row.iloc[0]["team_id"]) if len(row) > 0 else None
        else:
            tb = None
        if ta and tb: bracket[mid] = (ta, tb)

    return bracket, assignment


# ══════════════════════════════════════════════════════════════════
# 3. PRÉDICTION ET SIMULATION D'UN MATCH KO
# ══════════════════════════════════════════════════════════════════

def predict_match(ta, tb, tf, clf, rh, ra, classes):
    """Prédit prob de victoire A/B (nul redistribué) et xG."""
    fa = tf.get(ta,{}); fb = tf.get(tb,{})
    X = pd.DataFrame([{
        "home_fifa_ranking":   fa.get("rank",100),
        "away_fifa_ranking":   fb.get("rank",100),
        "ranking_gap":         fb.get("rank",100)-fa.get("rank",100),
        "home_form5_pts":      fa.get("f5_pts",1.5),
        "home_form5_scored":   fa.get("f5_sc",1.2),
        "home_form5_conceded": fa.get("f5_co",1.0),
        "home_form10_pts":     fa.get("f10_pts",1.5),
        "home_form10_scored":  fa.get("f10_sc",1.2),
        "home_form10_conceded":fa.get("f10_co",1.0),
        "away_form5_pts":      fb.get("f5_pts",1.5),
        "away_form5_scored":   fb.get("f5_sc",1.2),
        "away_form5_conceded": fb.get("f5_co",1.0),
        "away_form10_pts":     fb.get("f10_pts",1.5),
        "away_form10_scored":  fb.get("f10_sc",1.2),
        "away_form10_conceded":fb.get("f10_co",1.0),
        "h2h_home_wins":0.33,"h2h_draws":0.33,"h2h_away_wins":0.33,
        "h2h_home_goals_avg":1.2,"h2h_away_goals_avg":1.0,
        "neutral_venue":1,"competition_weight":1.0,"is_knockout":1,
    }])
    for col in X.columns:
        X[col] = pd.to_numeric(X[col],errors="coerce").fillna(1.0)

    pr   = clf.predict_proba(X)[0]
    xgh  = float(np.clip(rh.predict(X)[0],0,8))
    xga  = float(np.clip(ra.predict(X)[0],0,8))
    ph   = float(pr[classes.index("H")])
    pd_  = float(pr[classes.index("D")])
    pa   = float(pr[classes.index("A")])
    tot  = (ph+pa) if (ph+pa)>0 else 1.0
    prob_a = ph + pd_*(ph/tot)
    prob_b = pa + pd_*(pa/tot)
    return prob_a, prob_b, xgh, xga


def sim_ko(prob_a, prob_b, xgh, xga, ta, tb, rng):
    """Simule un match KO avec score Poisson. Retourne (hg, ag, winner)."""
    p = np.array([prob_a,prob_b],dtype=np.float64); p/=p.sum()
    win_a = bool(rng.choice([True,False],p=p))
    for _ in range(20):
        hg=int(rng.poisson(max(xgh,0.3))); ag=int(rng.poisson(max(xga,0.3)))
        if win_a and hg>ag: return hg,ag,ta
        if not win_a and ag>hg: return hg,ag,tb
    return (max(1,int(rng.poisson(xgh))),0,ta) if win_a \
           else (0,max(1,int(rng.poisson(xga))),tb)


# ══════════════════════════════════════════════════════════════════
# 4. MONTE CARLO PHASES FINALES
# ══════════════════════════════════════════════════════════════════

def run_monte_carlo(standings, thirds_df, tf, clf, rh, ra, classes):
    rng    = np.random.default_rng(RANDOM_SEED)
    groups = sorted(standings["group_name"].unique())
    all_tids = standings["team_id"].astype(int).tolist()

    # Compteurs de progression
    reach = {stage:{tid:0 for tid in all_tids}
             for stage in ["R16","R8","QF","SF","3RD","FIN","CHAMP"]}

    # Compteurs de matchups et scores par slot
    matchup_counts = {mid:defaultdict(int) for mid in
        [f"M{i}" for i in range(1,17)] + R8_IDS + QF_IDS + SF_IDS +
        ["3RD","FIN"]}
    score_counts   = {mid:defaultdict(int) for mid in matchup_counts}

    # Distributions par groupe pour tirer les qualifiés
    gi = {}
    for grp in groups:
        g    = standings[standings["group_name"]==grp].sort_values("position")
        tids = g["team_id"].astype(int).tolist()
        p1   = np.array(g["prob_1st"].tolist(),dtype=np.float64); p1/=p1.sum()
        p2   = np.array(g["prob_2nd"].tolist(),dtype=np.float64); p2/=p2.sum()
        p3   = np.array(g["prob_3rd"].tolist(),dtype=np.float64)
        p3   = p3/p3.sum() if p3.sum()>0 else np.ones(len(p3))/len(p3)
        gi[grp] = {"tids":tids,"p1":p1,"p2":p2,"p3":p3}

    # Groupes dont le 3ème est potentiellement qualifié (depuis best_third_place)
    best_third_groups = set(thirds_df["group_name"].tolist())

    print(f"\n🎲 Monte Carlo phases finales : {N_SIM:,} simulations...")

    for s in range(N_SIM):
        if s%2000==0: print(f"   {s:,}/{N_SIM:,}",end="\r")

        # Tirer les qualifiés de chaque groupe
        winners_sim = {}; runners_sim = {}; thirds_sim_pool = {}

        for grp in groups:
            g    = gi[grp]; tids = g["tids"]; rem=list(range(len(tids)))
            p    = g["p1"].copy(); p/=p.sum()
            i1   = rng.choice(len(tids),p=p)
            winners_sim[grp]=tids[i1]; rem=[i for i in rem if i!=i1]
            p2   = g["p2"][rem].copy(); p2/=p2.sum()
            i2   = rng.choice(len(rem),p=p2)
            runners_sim[grp]=tids[rem[i2]]; rem=[i for i in rem if i!=rem[i2]]
            if grp in best_third_groups:
                p3=g["p3"][rem].copy()
                p3=p3/p3.sum() if p3.sum()>0 else np.ones(len(rem))/len(rem)
                i3=rng.choice(len(rem),p=p3)
                thirds_sim_pool[grp]=tids[rem[i3]]

        # Assigner les 3èmes selon les contraintes
        # Utiliser les mêmes groupes que best_third_place
        available_thirds = {grp:{"group_name":grp,
                                  "team_id":thirds_sim_pool.get(grp,0),
                                  "rank":tf.get(thirds_sim_pool.get(grp,0),
                                                {}).get("rank",100)}
                            for grp in best_third_groups
                            if grp in thirds_sim_pool}

        # Backtracking pour assigner les 3èmes
        avail_ranks = {g: available_thirds[g]["rank"] for g in available_thirds}
        slots_bt = sorted(THIRD_ELIGIBLE.keys(),
            key=lambda s:len([g for g in THIRD_ELIGIBLE[s] if g in avail_ranks]))

        def bt(idx, asgn, used):
            if idx==len(slots_bt): return asgn.copy()
            sl=slots_bt[idx]
            elig=sorted([g for g in THIRD_ELIGIBLE[sl]
                         if g in avail_ranks and g not in used],
                        key=lambda g:avail_ranks[g])
            for grp in elig:
                asgn[sl]=grp; used.add(grp)
                r=bt(idx+1,asgn,used)
                if r is not None: return r
                del asgn[sl]; used.remove(grp)
            return None

        third_assignment = bt(0,{},set()) or {}

        # Construire le bracket R16 de cette simulation
        def get(pos,grp):
            if pos==1: return winners_sim.get(grp)
            if pos==2: return runners_sim.get(grp)
            return None

        bracket_sim = {}
        for mid,(ca,cb) in FIXED_R16.items():
            ta=get(int(ca[0]),ca[1]); tb=get(int(cb[0]),cb[1])
            if ta and tb: bracket_sim[mid]=(ta,tb)
        for mid,ca in THIRD_R16.items():
            ta=get(int(ca[0]),ca[1])
            grp=third_assignment.get(mid)
            tb=available_thirds[grp]["team_id"] if grp else None
            if ta and tb: bracket_sim[mid]=(ta,tb)

        winners = {}

        # R16
        for mid,(ta,tb) in bracket_sim.items():
            pa,pb,xgh,xga=predict_match(ta,tb,tf,clf,rh,ra,classes)
            hg,ag,w=sim_ko(pa,pb,xgh,xga,ta,tb,rng)
            winners[mid]=w; reach["R16"][w]+=1
            matchup_counts[mid][(ta,tb)]+=1; score_counts[mid][(hg,ag)]+=1

        # R8
        for rid,(ma,mb) in zip(R8_IDS,R8_PAIRS):
            ta=winners.get(ma); tb=winners.get(mb)
            if not ta or not tb: continue
            pa,pb,xgh,xga=predict_match(ta,tb,tf,clf,rh,ra,classes)
            hg,ag,w=sim_ko(pa,pb,xgh,xga,ta,tb,rng)
            winners[rid]=w; reach["R8"][w]+=1
            matchup_counts[rid][(ta,tb)]+=1; score_counts[rid][(hg,ag)]+=1

        # QF
        for qid,(ma,mb) in zip(QF_IDS,QF_PAIRS):
            ta=winners.get(ma); tb=winners.get(mb)
            if not ta or not tb: continue
            pa,pb,xgh,xga=predict_match(ta,tb,tf,clf,rh,ra,classes)
            hg,ag,w=sim_ko(pa,pb,xgh,xga,ta,tb,rng)
            winners[qid]=w; reach["QF"][w]+=1
            matchup_counts[qid][(ta,tb)]+=1; score_counts[qid][(hg,ag)]+=1

        # SF
        sf_losers={}
        for sid,(ma,mb) in zip(SF_IDS,SF_PAIRS):
            ta=winners.get(ma); tb=winners.get(mb)
            if not ta or not tb: continue
            pa,pb,xgh,xga=predict_match(ta,tb,tf,clf,rh,ra,classes)
            hg,ag,w=sim_ko(pa,pb,xgh,xga,ta,tb,rng)
            winners[sid]=w; reach["SF"][w]+=1
            sf_losers[sid]=tb if w==ta else ta
            matchup_counts[sid][(ta,tb)]+=1; score_counts[sid][(hg,ag)]+=1

        # 3ème place
        ta=sf_losers.get("SF1"); tb=sf_losers.get("SF2")
        if ta and tb:
            pa,pb,xgh,xga=predict_match(ta,tb,tf,clf,rh,ra,classes)
            hg,ag,w=sim_ko(pa,pb,xgh,xga,ta,tb,rng)
            winners["3RD"]=w; reach["3RD"][w]+=1
            matchup_counts["3RD"][(ta,tb)]+=1; score_counts["3RD"][(hg,ag)]+=1

        # Finale
        ta=winners.get("SF1"); tb=winners.get("SF2")
        if ta and tb:
            pa,pb,xgh,xga=predict_match(ta,tb,tf,clf,rh,ra,classes)
            hg,ag,w=sim_ko(pa,pb,xgh,xga,ta,tb,rng)
            winners["FIN"]=w; reach["FIN"][w]+=1; reach["CHAMP"][w]+=1
            matchup_counts["FIN"][(ta,tb)]+=1; score_counts["FIN"][(hg,ag)]+=1

    print(f"   ✅ {N_SIM:,} simulations terminées")
    return reach, matchup_counts, score_counts


# ══════════════════════════════════════════════════════════════════
# 5. BRACKET DÉTERMINISTE (matchup + score les plus fréquents)
# ══════════════════════════════════════════════════════════════════

def build_det_bracket(r16_bracket, matchup_counts, score_counts,
                      reach, tf, clf, rh, ra, classes, label_map):
    """
    Construit le bracket déterministe :
    - R16 : bracket fixe (1ers/2èmes/3èmes déterministes)
    - R8 à FINALE : matchup le plus fréquent → vainqueur = meilleur prob
    Score prédit = xG arrondi corrigé pour le vainqueur.
    """
    results  = {}  # mid → {ta, tb, winner, hg, ag, prob_a, prob_b, freq}
    winners  = {}

    def play(mid, ta, tb):
        pa,pb,xgh,xga = predict_match(ta,tb,tf,clf,rh,ra,classes)
        w = ta if pa>=pb else tb
        hg=round(xgh); ag=round(xga)
        if w==ta and hg<=ag: hg=ag+1
        if w==tb and ag<=hg: ag=hg+1
        mc=matchup_counts[mid]; freq=mc.get((ta,tb),0)/N_SIM
        results[mid]={"ta":ta,"tb":tb,"winner":w,
                      "hg":int(hg),"ag":int(ag),
                      "prob_a":round(pa,4),"prob_b":round(pb,4),
                      "freq":round(freq,4)}
        winners[mid]=w

    # R16 déterministe
    for mid in sorted(r16_bracket.keys(),key=lambda x:int(x[1:])):
        ta,tb = r16_bracket[mid]
        play(mid,ta,tb)

    # R8
    for rid,(ma,mb) in zip(R8_IDS,R8_PAIRS):
        ta=winners.get(ma); tb=winners.get(mb)
        if ta and tb: play(rid,ta,tb)

    # QF
    for qid,(ma,mb) in zip(QF_IDS,QF_PAIRS):
        ta=winners.get(ma); tb=winners.get(mb)
        if ta and tb: play(qid,ta,tb)

    # SF
    sf_losers={}
    for sid,(ma,mb) in zip(SF_IDS,SF_PAIRS):
        ta=winners.get(ma); tb=winners.get(mb)
        if not ta or not tb: continue
        play(sid,ta,tb)
        sf_losers[sid]=results[sid]["tb"] if winners[sid]==results[sid]["ta"] \
                        else results[sid]["ta"]

    # 3ème place
    ta=sf_losers.get("SF1"); tb=sf_losers.get("SF2")
    if ta and tb: play("3RD",ta,tb)

    # Finale
    ta=winners.get("SF1"); tb=winners.get("SF2")
    if ta and tb: play("FIN",ta,tb)

    return results


# ══════════════════════════════════════════════════════════════════
# 6. AFFICHAGE
# ══════════════════════════════════════════════════════════════════

def print_r16_bracket(r16_bracket, assignment, thirds_df, standings, label_map):
    print(f"\n{'='*75}")
    print("  Bracket R16 — 16èmes de finale")
    print(f"{'='*75}")

    # Infos 3èmes assignés
    third_info = {row["group_name"]: row
                  for _,row in thirds_df.iterrows()}

    order = [f"M{i}" for i in range(1,17)]
    print(f"\n  {'Match':<5} {'Équipe A':<25} {'vs'} {'Équipe B':<25} {'(3ème groupe)'}")
    print(f"  {'─'*72}")
    for mid in order:
        if mid not in r16_bracket: continue
        ta,tb = r16_bracket[mid]
        la=label_map.get(ta,"?"); lb=label_map.get(tb,"?")
        note=""
        if mid in THIRD_R16:
            grp=assignment.get(mid,"?")
            note=f"← 3ème Grp {grp}"
        print(f"  {mid:<5} {la:<25} vs {lb:<25} {note}")


def print_det_bracket(results, label_map):
    rounds = [
        ("16èmes de finale", [f"M{i}" for i in range(1,17)]),
        ("8èmes de finale",  R8_IDS),
        ("Quarts de finale", QF_IDS),
        ("Demi-finales",     SF_IDS),
        ("3ème place",       ["3RD"]),
        ("🏆 FINALE",        ["FIN"]),
    ]
    print(f"\n{'='*82}")
    print("  Bracket prédit — Scénario le plus probable")
    print(f"{'='*82}")
    for title, mids in rounds:
        active = [m for m in mids if m in results]
        if not active: continue
        print(f"\n  ── {title} ──")
        print(f"  {'Match':<6} {'Équipe A':<22} {'Score':^7} {'Équipe B':<22} "
              f"{'P(A)':>6} {'P(B)':>6}  Vainqueur")
        print(f"  {'─'*78}")
        for mid in active:
            r=results[mid]
            la=label_map.get(r["ta"],"?"); lb=label_map.get(r["tb"],"?")
            lw=label_map.get(r["winner"],"?")
            score=f"{r['hg']} - {r['ag']}"
            print(f"  {mid:<6} {la:<22} {score:^7} {lb:<22} "
                  f"{r['prob_a']:>6.1%} {r['prob_b']:>6.1%}  → {lw}")

    # Résumé final
    fin=results.get("FIN",{}); third=results.get("3RD",{})
    champ=label_map.get(fin.get("winner"),"?")
    runner=label_map.get(fin.get("tb") if fin.get("winner")==fin.get("ta")
                         else fin.get("ta"),"?")
    third_w=label_map.get(third.get("winner"),"?")
    print(f"\n  🥇 Champion   : {champ}")
    print(f"  🥈 Finaliste  : {runner}")
    print(f"  🥉 3ème place : {third_w}")


def print_progression(reach, label_map, standings):
    n=N_SIM
    print(f"\n{'='*85}")
    print("  Probabilités de progression — Top 20")
    print(f"{'='*85}")
    print(f"  {'Équipe':<22} {'Grp':>4} {'Rnk':>4} "
          f"{'R16':>6} {'R8':>6} {'QF':>6} {'SF':>6} {'Fin':>6} {'Titre':>7}")
    print(f"  {'─'*80}")
    rows=[]
    for _,r in standings.iterrows():
        tid=int(r["team_id"])
        rows.append({
            "name":label_map.get(tid,"?"),"group":r["group_name"],
            "rank":int(r["fifa_ranking"]),
            "r16":  reach["R16"].get(tid,0)/n,
            "r8":   reach["R8"].get(tid,0)/n,
            "qf":   reach["QF"].get(tid,0)/n,
            "sf":   reach["SF"].get(tid,0)/n,
            "fin":  reach["FIN"].get(tid,0)/n,
            "champ":reach["CHAMP"].get(tid,0)/n,
        })
    rows.sort(key=lambda x:x["champ"],reverse=True)
    for r in rows[:20]:
        print(f"  {r['name']:<22} {r['group']:>4} {r['rank']:>4} "
              f"{r['r16']:>6.1%} {r['r8']:>6.1%} {r['qf']:>6.1%} "
              f"{r['sf']:>6.1%} {r['fin']:>6.1%} {r['champ']:>7.1%}")


# ══════════════════════════════════════════════════════════════════
# 7. SAUVEGARDE
# ══════════════════════════════════════════════════════════════════

def save(det_bracket, reach, standings, label_map, conn):
    c=conn.cursor(); n=N_SIM

    # knockout_fixtures
    c.execute("DROP TABLE IF EXISTS knockout_fixtures")
    c.execute("""
        CREATE TABLE knockout_fixtures (
            match_id     TEXT PRIMARY KEY, round TEXT,
            team_a_id    INTEGER, team_a_name TEXT,
            team_b_id    INTEGER, team_b_name TEXT,
            pred_score_a INTEGER, pred_score_b INTEGER,
            prob_a_wins  REAL,    prob_b_wins  REAL,
            winner_id    INTEGER, winner_name  TEXT
        )
    """)
    round_map={**{f"M{i}":"R16" for i in range(1,17)},
               **{x:"R8" for x in R8_IDS},
               **{x:"QF" for x in QF_IDS},
               **{x:"SF" for x in SF_IDS},
               "3RD":"3RD","FIN":"FINAL"}
    for mid,r in det_bracket.items():
        c.execute("INSERT OR REPLACE INTO knockout_fixtures VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",(
            mid,round_map.get(mid,"?"),
            r["ta"],label_map.get(r["ta"],"?"),
            r["tb"],label_map.get(r["tb"],"?"),
            r["hg"],r["ag"],r["prob_a"],r["prob_b"],
            r["winner"],label_map.get(r["winner"],"?"),
        ))

    # knockout_probabilities
    c.execute("DROP TABLE IF EXISTS knockout_probabilities")
    c.execute("""
        CREATE TABLE knockout_probabilities (
            team_id       INTEGER PRIMARY KEY, team_name TEXT,
            group_name    TEXT,                fifa_ranking INTEGER,
            prob_r16      REAL, prob_r8   REAL, prob_qf    REAL,
            prob_sf       REAL, prob_final REAL, prob_champion REAL
        )
    """)
    for _,r in standings.iterrows():
        tid=int(r["team_id"])
        c.execute("INSERT OR REPLACE INTO knockout_probabilities VALUES (?,?,?,?,?,?,?,?,?,?)",(
            tid,label_map.get(tid,"?"),r["group_name"],int(r["fifa_ranking"]),
            round(reach["R16"].get(tid,0)/n,4),
            round(reach["R8"].get(tid,0)/n,4),
            round(reach["QF"].get(tid,0)/n,4),
            round(reach["SF"].get(tid,0)/n,4),
            round(reach["FIN"].get(tid,0)/n,4),
            round(reach["CHAMP"].get(tid,0)/n,4),
        ))

    conn.commit()
    print("\n✅ knockout_fixtures et knockout_probabilities sauvegardés")


# ══════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════

if __name__=="__main__":
    print("="*55)
    print("  Pipeline 2 — Phases finales CdM 2026")
    print("="*55)

    conn = get_connection()
    clf,rh,ra,classes = load_models()
    standings,thirds_df,tf,label_map = load_group_data(conn)

    # Construire et afficher le bracket R16
    r16_bracket, assignment = build_r16_bracket(standings,thirds_df,label_map)
    print_r16_bracket(r16_bracket, assignment, thirds_df, standings, label_map)

    input("\n  ▶ Appuie sur Entrée pour lancer les simulations...")

    # Monte Carlo
    reach, matchup_counts, score_counts = run_monte_carlo(
        standings,thirds_df,tf,clf,rh,ra,classes)

    # Bracket déterministe
    det_bracket = build_det_bracket(
        r16_bracket,matchup_counts,score_counts,
        reach,tf,clf,rh,ra,classes,label_map)

    # Affichage et sauvegarde
    print_det_bracket(det_bracket, label_map)
    print_progression(reach, label_map, standings)
    save(det_bracket, reach, standings, label_map, conn)

    conn.close()
    print("\n🎉 Pipeline 2 terminé")