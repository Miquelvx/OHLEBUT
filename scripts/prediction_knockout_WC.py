import os, sys, json, pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), "../collect"))
from init_db import get_connection

_script_dir = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(_script_dir, "../models/")
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
    "h2h_home_goals_avg","h2h_away_goals_avg","h2h_matches",
    "neutral_venue","competition_weight","is_knockout",
]

THIRD_ELIGIBLE = {
    "M1":["A","B","C","D","F"],"M2":["C","D","F","G","H"],
    "M7":["B","E","F","I","J"],"M8":["A","E","H","I","J"],
    "M11":["C","E","F","H","I"],"M12":["E","H","I","J","K"],
    "M15":["E","F","G","I","J"],"M16":["D","E","I","J","L"],
}
FIXED_R16 = {
    "M3":("2A","2B"),"M4":("1F","2C"),"M5":("2K","2L"),"M6":("1H","2J"),
    "M9":("1C","2F"),"M10":("2E","2I"),"M13":("1J","2H"),"M14":("2D","2G"),
}
THIRD_R16 = {
    "M1":"1E","M2":"1I","M7":"1D","M8":"1G",
    "M11":"1A","M12":"1L","M15":"1B","M16":"1K",
}
R8_PAIRS = [("M1","M2"),("M3","M4"),("M5","M6"),("M7","M8"),
            ("M9","M10"),("M11","M12"),("M13","M14"),("M15","M16")]
R8_IDS   = ["R8_1","R8_2","R8_3","R8_4","R8_5","R8_6","R8_7","R8_8"]
QF_PAIRS = [("R8_1","R8_2"),("R8_3","R8_4"),("R8_5","R8_6"),("R8_7","R8_8")]
QF_IDS   = ["QF1","QF2","QF3","QF4"]
SF_PAIRS = [("QF1","QF2"),("QF3","QF4")]
SF_IDS   = ["SF1","SF2"]
ALL_MIDS = [f"M{i}" for i in range(1,17)] + R8_IDS + QF_IDS + SF_IDS + ["3RD","FIN"]


# ══════════════════════════════════════════════════════════════════
# 1. CHARGEMENT
# ══════════════════════════════════════════════════════════════════

def load_models():
    with open(os.path.join(MODELS_DIR, "classifier_calibrated.pkl"), "rb") as f:
        clf = pickle.load(f)
    rh = XGBRegressor(); ra = XGBRegressor()
    rh.load_model(os.path.join(MODELS_DIR, "regressor_home.json"))
    ra.load_model(os.path.join(MODELS_DIR, "regressor_away.json"))
    with open(os.path.join(MODELS_DIR, "features.json")) as f:
        meta = json.load(f)
    le_classes   = meta["classes"]
    idx_to_label = {i: c for i, c in enumerate(le_classes)}
    label_to_idx = {v: k for k, v in idx_to_label.items()}
    print(f"Modeles charges — classes : {le_classes}")
    return clf, rh, ra, idx_to_label, label_to_idx


def load_group_data(conn):
    standings = pd.read_sql_query("""
        SELECT team_id, group_name, position, team_label, fifa_ranking,
               points, won, drawn, lost, goals_for, goals_against, goal_diff,
               prob_1st, prob_2nd, prob_3rd, prob_4th, prob_qualify
        FROM group_standings ORDER BY group_name, position
    """, conn)
    if len(standings) == 0:
        print("group_standings vide — lance d'abord predict_groups.py")
        sys.exit(1)

    thirds = pd.read_sql_query("""
        SELECT rank, group_name, team_id, team_label,
               points, goal_diff, goals_for, fifa_ranking
        FROM best_third_place ORDER BY rank
    """, conn)

    mf_home = pd.read_sql_query("""
        SELECT mf.home_team_id AS tid,
               mf.home_fifa_ranking AS rank, mf.home_form5_pts AS f5_pts,
               mf.home_form5_scored AS f5_sc, mf.home_form5_conceded AS f5_co,
               mf.home_form10_pts AS f10_pts, mf.home_form10_scored AS f10_sc,
               mf.home_form10_conceded AS f10_co
        FROM match_features mf
        INNER JOIN (
            SELECT home_team_id, MAX(match_date) AS last_date
            FROM match_features
            WHERE home_team_id IN (SELECT team_id FROM teams WHERE is_wc2026=1)
              AND home_fifa_ranking IS NOT NULL
            GROUP BY home_team_id
        ) latest ON mf.home_team_id = latest.home_team_id
                AND mf.match_date   = latest.last_date
    """, conn)

    mf_away = pd.read_sql_query("""
        SELECT mf.away_team_id AS tid,
               mf.away_fifa_ranking AS rank, mf.away_form5_pts AS f5_pts,
               mf.away_form5_scored AS f5_sc, mf.away_form5_conceded AS f5_co,
               mf.away_form10_pts AS f10_pts, mf.away_form10_scored AS f10_sc,
               mf.away_form10_conceded AS f10_co
        FROM match_features mf
        INNER JOIN (
            SELECT away_team_id, MAX(match_date) AS last_date
            FROM match_features
            WHERE away_team_id IN (SELECT team_id FROM teams WHERE is_wc2026=1)
              AND away_fifa_ranking IS NOT NULL
            GROUP BY away_team_id
        ) latest ON mf.away_team_id = latest.away_team_id
                AND mf.match_date   = latest.last_date
    """, conn)

    tf = {}
    for df_mf in [mf_home, mf_away]:
        for _, r in df_mf.iterrows():
            tid = int(r["tid"])
            if tid not in tf:
                tf[tid] = {
                    "rank":   float(r["rank"]   or 100),
                    "f5_pts": float(r["f5_pts"] or 1.5),
                    "f5_sc":  float(r["f5_sc"]  or 1.2),
                    "f5_co":  float(r["f5_co"]  or 1.0),
                    "f10_pts":float(r["f10_pts"]or 1.5),
                    "f10_sc": float(r["f10_sc"] or 1.2),
                    "f10_co": float(r["f10_co"] or 1.0),
                }
    import unicodedata as _ud, re as _re2
    def _norm_fifa(name: str) -> str:
        nfkd = _ud.normalize("NFKD", str(name))
        a    = nfkd.encode("ascii","ignore").decode("ascii")
        return _re2.sub(r"\s+"," ", _re2.sub(r"[^a-z0-9 ]"," ", a.lower())).strip()

    _FIFA_ALIASES = {
        "Korea Republic":"south korea","Korea DPR":"north korea",
        "IR Iran":"iran","USA":"united states","China PR":"china",
        "Congo DR":"congo dr","Cabo Verde":"cape verde",
        "Kyrgyz Republic":"kyrgyzstan","Bosnia and Herzegovina":"bosnia and herzegovina",
        "North Macedonia":"north macedonia","Republic of Ireland":"republic of ireland",
        "Türkiye":"turkiye","St. Vincent and the Grenadines":"st vincent and the grenadines",
    }

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

    teams_norm = pd.read_sql_query(
        "SELECT team_id, team_name_normalized FROM teams", conn)
    norm_to_tid = dict(zip(teams_norm["team_name_normalized"],
                           teams_norm["team_id"].astype(int)))

    for _, r in latest_rankings.iterrows():
        fifa_name = r["team_name_fifa"]
        norm = _FIFA_ALIASES.get(fifa_name, _norm_fifa(fifa_name))
        tid  = norm_to_tid.get(norm)
        if tid is None:
            continue
        rank    = float(r["rank"])
        if tid in tf:
            tf[tid]["rank"] = rank
        else:
            tf[tid] = {"rank":rank,
                       "f5_pts":1.5,"f5_sc":1.2,"f5_co":1.0,
                       "f10_pts":1.5,"f10_sc":1.2,"f10_co":1.0}

    for _, r in standings.iterrows():
        tid = int(r["team_id"])
        if tid not in tf:
            rank    = float(r["fifa_ranking"] or 100)
            tf[tid] = {"rank":rank,
                       "f5_pts":1.5,"f5_sc":1.2,"f5_co":1.0,
                       "f10_pts":1.5,"f10_sc":1.2,"f10_co":1.0}

    label_map = dict(zip(standings["team_id"].astype(int), standings["team_label"]))
    print(f"{len(standings)} equipes, {len(thirds)} meilleurs 3emes")
    return standings, thirds, tf, label_map


# ══════════════════════════════════════════════════════════════════
# 2. ATTRIBUTION DES 3ÈMES
# ══════════════════════════════════════════════════════════════════

def assign_thirds(available, eligible_map=THIRD_ELIGIBLE):
    slots = sorted(eligible_map.keys(),
        key=lambda s: len([g for g in eligible_map[s] if g in available]))
    def backtrack(idx, assignment, used):
        if idx == len(slots): return assignment.copy()
        slot = slots[idx]
        eligible = sorted(
            [g for g in eligible_map[slot] if g in available and g not in used],
            key=lambda g: available[g])
        for grp in eligible:
            assignment[slot] = grp; used.add(grp)
            result = backtrack(idx+1, assignment, used)
            if result is not None: return result
            del assignment[slot]; used.remove(grp)
        return None
    return backtrack(0, {}, set()) or {}


def build_r16_bracket(standings, thirds_df, label_map):
    def get_team(code):
        pos = int(code[0]); grp = code[1:]
        row = standings[(standings["group_name"]==grp) & (standings["position"]==pos)]
        return int(row.iloc[0]["team_id"]) if len(row) > 0 else None

    available  = {row["group_name"]: int(row["fifa_ranking"])
                  for _, row in thirds_df.iterrows()}
    assignment = assign_thirds(available)
    bracket    = {}

    for mid, (ca, cb) in FIXED_R16.items():
        ta = get_team(ca); tb = get_team(cb)
        if ta and tb: bracket[mid] = (ta, tb)

    for mid, ca in THIRD_R16.items():
        ta  = get_team(ca)
        grp = assignment.get(mid)
        tb  = None
        if grp:
            row = thirds_df[thirds_df["group_name"]==grp]
            tb  = int(row.iloc[0]["team_id"]) if len(row) > 0 else None
        if ta and tb: bracket[mid] = (ta, tb)

    return bracket, assignment


# ══════════════════════════════════════════════════════════════════
# 3. PRÉDICTION — avec cache
# ══════════════════════════════════════════════════════════════════

_pred_cache = {}

def predict_match(ta, tb, tf, clf, rh, ra, label_to_idx):
    key = (ta, tb)
    if key in _pred_cache:
        return _pred_cache[key]
    fa = tf.get(ta, {}); fb = tf.get(tb, {})
    X = pd.DataFrame([{
        "home_fifa_ranking":    fa.get("rank",100),
        "away_fifa_ranking":    fb.get("rank",100),
        "ranking_gap":          fb.get("rank",100) - fa.get("rank",100),
        "home_form5_pts":       fa.get("f5_pts",1.5),
        "home_form5_scored":    fa.get("f5_sc",1.2),
        "home_form5_conceded":  fa.get("f5_co",1.0),
        "home_form10_pts":      fa.get("f10_pts",1.5),
        "home_form10_scored":   fa.get("f10_sc",1.2),
        "home_form10_conceded": fa.get("f10_co",1.0),
        "away_form5_pts":       fb.get("f5_pts",1.5),
        "away_form5_scored":    fb.get("f5_sc",1.2),
        "away_form5_conceded":  fb.get("f5_co",1.0),
        "away_form10_pts":      fb.get("f10_pts",1.5),
        "away_form10_scored":   fb.get("f10_sc",1.2),
        "away_form10_conceded": fb.get("f10_co",1.0),
        "h2h_home_wins":0.33,"h2h_draws":0.33,"h2h_away_wins":0.33,
        "h2h_home_goals_avg":1.2,"h2h_away_goals_avg":1.0,
        "h2h_matches":0,"neutral_venue":1,
        "competition_weight":1.0,"is_knockout":1,
    }])
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(1.0)

    X = X[FEATURES]  # forcer l'ordre exact attendu par le modèle

    pr  = clf.predict_proba(X)[0]
    xgh = float(np.clip(rh.predict(X)[0], 0, 8))
    xga = float(np.clip(ra.predict(X)[0], 0, 8))
    ph  = float(pr[label_to_idx["H"]])
    pd_ = float(pr[label_to_idx["D"]])
    pa  = float(pr[label_to_idx["A"]])
    tot = (ph + pa) if (ph + pa) > 0 else 1.0
    result = (ph + pd_*(ph/tot), pa + pd_*(pa/tot), xgh, xga)
    _pred_cache[key] = result
    return result


def sim_ko(prob_a, prob_b, xgh, xga, ta, tb, rng):
    p = np.array([prob_a, prob_b], dtype=np.float64); p /= p.sum()
    win_a = bool(rng.random() < p[0])
    hg = int(rng.poisson(max(xgh, 0.3)))
    ag = int(rng.poisson(max(xga, 0.3)))
    if win_a and hg <= ag:   hg = ag + 1
    elif not win_a and ag <= hg: ag = hg + 1
    return hg, ag, (ta if win_a else tb)


# ══════════════════════════════════════════════════════════════════
# 4. MONTE CARLO — stats indexées par (mid, ta, tb)
# ══════════════════════════════════════════════════════════════════

def run_monte_carlo(r16_bracket, tf, clf, rh, ra, label_to_idx):
    rng = np.random.default_rng(RANDOM_SEED)
    all_tids = list(set(
        [ta for ta,tb in r16_bracket.values()] +
        [tb for ta,tb in r16_bracket.values()]
    ))
    reach = {stage: {tid:0 for tid in all_tids}
             for stage in ["R16","R8","QF","SF","3RD","FIN","CHAMP"]}

    # FIX : indexation par (mid, ta, tb) pour conditionner sur la paire
    matchup_counts     = {mid: defaultdict(int) for mid in ALL_MIDS}
    pair_score_counts  = defaultdict(lambda: defaultdict(int))
    pair_winner_counts = defaultdict(lambda: defaultdict(int))

    # Pré-calcul R16
    r16_probs = {mid: predict_match(ta, tb, tf, clf, rh, ra, label_to_idx)
                 for mid, (ta, tb) in r16_bracket.items()}
    print(f"  Probabilites R16 pre-calculees ({len(r16_probs)} matchs)")
    print(f"\nMonte Carlo : {N_SIM:,} simulations...")

    def record(mid, ta, tb, hg, ag, w):
        matchup_counts[mid][(ta,tb)]          += 1
        pair_score_counts[(mid,ta,tb)][(hg,ag)] += 1
        pair_winner_counts[(mid,ta,tb)][w]      += 1

    for s in range(N_SIM):
        if s % 2000 == 0: print(f"   {s:,}/{N_SIM:,}", end="\r")
        winners = {}

        for mid, (ta, tb) in r16_bracket.items():
            pa, pb, xgh, xga = r16_probs[mid]
            hg, ag, w = sim_ko(pa, pb, xgh, xga, ta, tb, rng)
            winners[mid] = w; reach["R16"][w] += 1
            record(mid, ta, tb, hg, ag, w)

        for rid, (ma, mb) in zip(R8_IDS, R8_PAIRS):
            ta = winners.get(ma); tb = winners.get(mb)
            if not ta or not tb: continue
            pa, pb, xgh, xga = predict_match(ta, tb, tf, clf, rh, ra, label_to_idx)
            hg, ag, w = sim_ko(pa, pb, xgh, xga, ta, tb, rng)
            winners[rid] = w; reach["R8"][w] += 1
            record(rid, ta, tb, hg, ag, w)

        for qid, (ma, mb) in zip(QF_IDS, QF_PAIRS):
            ta = winners.get(ma); tb = winners.get(mb)
            if not ta or not tb: continue
            pa, pb, xgh, xga = predict_match(ta, tb, tf, clf, rh, ra, label_to_idx)
            hg, ag, w = sim_ko(pa, pb, xgh, xga, ta, tb, rng)
            winners[qid] = w; reach["QF"][w] += 1
            record(qid, ta, tb, hg, ag, w)

        sf_losers = {}
        for sid, (ma, mb) in zip(SF_IDS, SF_PAIRS):
            ta = winners.get(ma); tb = winners.get(mb)
            if not ta or not tb: continue
            pa, pb, xgh, xga = predict_match(ta, tb, tf, clf, rh, ra, label_to_idx)
            hg, ag, w = sim_ko(pa, pb, xgh, xga, ta, tb, rng)
            winners[sid] = w; reach["SF"][w] += 1
            sf_losers[sid] = tb if w == ta else ta
            record(sid, ta, tb, hg, ag, w)

        ta = sf_losers.get("SF1"); tb = sf_losers.get("SF2")
        if ta and tb:
            pa, pb, xgh, xga = predict_match(ta, tb, tf, clf, rh, ra, label_to_idx)
            hg, ag, w = sim_ko(pa, pb, xgh, xga, ta, tb, rng)
            winners["3RD"] = w; reach["3RD"][w] += 1
            record("3RD", ta, tb, hg, ag, w)

        ta = winners.get("SF1"); tb = winners.get("SF2")
        if ta and tb:
            pa, pb, xgh, xga = predict_match(ta, tb, tf, clf, rh, ra, label_to_idx)
            hg, ag, w = sim_ko(pa, pb, xgh, xga, ta, tb, rng)
            winners["FIN"] = w; reach["FIN"][w] += 1; reach["CHAMP"][w] += 1
            record("FIN", ta, tb, hg, ag, w)

    print(f"   {N_SIM:,}/{N_SIM:,} — done")
    print(f"  Duels uniques calcules par XGBoost : {len(_pred_cache)}")
    print(f"  (rapide = normal : cache evite les appels redondants)")
    return reach, matchup_counts, pair_score_counts, pair_winner_counts


# ══════════════════════════════════════════════════════════════════
# 5. BRACKET DÉTERMINISTE — conditionné sur la paire dominante
# ══════════════════════════════════════════════════════════════════

def build_det_bracket(r16_bracket, matchup_counts, pair_score_counts, pair_winner_counts, label_map):
    results = {}
    winners = {} 
    
    def play(mid, ta, tb):
        sc = pair_score_counts.get((mid, ta, tb), {})
        wc = pair_winner_counts.get((mid, ta, tb), {})
        # Essayer la paire inversée si nécessaire
        if not wc:
            sc = pair_score_counts.get((mid, tb, ta), {})
            wc = pair_winner_counts.get((mid, tb, ta), {})
            if wc: ta, tb = tb, ta
        if not wc:
            winners[mid] = ta
            results[mid] = {"ta":ta,"tb":tb,"winner":ta,
                            "hg":1,"ag":0,"prob_a":0.5,"prob_b":0.5}
            return ta
        w       = max(wc.items(), key=lambda x: x[1])[0]
        total_w = sum(wc.values())

        # Score = moyenne pondérée conditionnelle
        # Filtrer les scores cohérents avec le vainqueur, puis moyenner
        if sc:
            if w == ta:
                sc_filtered = {(h,a): n for (h,a),n in sc.items() if h > a}
            else:
                sc_filtered = {(h,a): n for (h,a),n in sc.items() if a > h}
            sc_used    = sc_filtered if sc_filtered else sc
            total_used = sum(sc_used.values())
            avg_hg = sum(h * n for (h,a),n in sc_used.items()) / total_used
            avg_ag = sum(a * n for (h,a),n in sc_used.items()) / total_used
            hg = min(round(avg_hg), 4)
            ag = min(round(avg_ag), 4)
            if w == ta and hg <= ag: hg = ag + 1
            elif w == tb and ag <= hg: ag = hg + 1
        else:
            hg, ag = 1, 0

        # Fréquence = % des sims où ce vainqueur gagne
        freq = round(wc.get(w, 0) / total_w * 100, 1) if total_w else 0.0

        results[mid] = {
            "ta": ta, "tb": tb, "winner": w,
            "hg": int(hg), "ag": int(ag),
            "prob_a": round(wc.get(ta, 0) / total_w, 4),
            "prob_b": round(wc.get(tb, 0) / total_w, 4),
            "score_freq": freq,   # % des sims avec ce score exact
        }
        winners[mid] = w
        return w

    # R16 — participants fixes
    for mid in sorted(r16_bracket.keys(), key=lambda x: int(x[1:])):
        ta, tb = r16_bracket[mid]
        play(mid, ta, tb)

    # R8, QF, SF — propagation déterministe
    for rid, (ma, mb) in zip(R8_IDS, R8_PAIRS):
        ta = winners.get(ma); tb = winners.get(mb)
        if ta and tb: play(rid, ta, tb)

    for qid, (ma, mb) in zip(QF_IDS, QF_PAIRS):
        ta = winners.get(ma); tb = winners.get(mb)
        if ta and tb: play(qid, ta, tb)

    sf_losers = {}
    for sid, (ma, mb) in zip(SF_IDS, SF_PAIRS):
        ta = winners.get(ma); tb = winners.get(mb)
        if ta and tb:
            w = play(sid, ta, tb)
            sf_losers[sid] = tb if w == ta else ta

    ta = sf_losers.get("SF1"); tb = sf_losers.get("SF2")
    if ta and tb: play("3RD", ta, tb)

    ta = winners.get("SF1"); tb = winners.get("SF2")
    if ta and tb: play("FIN", ta, tb)

    return results


# ══════════════════════════════════════════════════════════════════
# 6. AFFICHAGE
# ══════════════════════════════════════════════════════════════════

def print_r16_bracket(r16_bracket, assignment, label_map):
    print(f"\n{'='*75}")
    print("  Bracket R16 — 16emes de finale")
    print(f"{'='*75}")
    print(f"\n  {'Match':<5} {'Equipe A':<25} vs {'Equipe B':<25} {'(3eme groupe)'}")
    print(f"  {'─'*72}")
    for mid in [f"M{i}" for i in range(1,17)]:
        if mid not in r16_bracket: continue
        ta, tb = r16_bracket[mid]
        la = label_map.get(ta,"?"); lb = label_map.get(tb,"?")
        note = f"<- 3eme Grp {assignment.get(mid,'?')}" if mid in THIRD_R16 else ""
        print(f"  {mid:<5} {la:<25} vs {lb:<25} {note}")


def print_det_bracket(results, label_map):
    rounds = [
        ("16emes de finale", [f"M{i}" for i in range(1,17)]),
        ("8emes de finale",  R8_IDS),
        ("Quarts de finale", QF_IDS),
        ("Demi-finales",     SF_IDS),
        ("3eme place",       ["3RD"]),
        ("FINALE",           ["FIN"]),
    ]
    print(f"\n{'='*82}")
    print("  Bracket predit — Scenario le plus probable")
    print(f"{'='*82}")
    for title, mids in rounds:
        active = [m for m in mids if m in results]
        if not active: continue
        print(f"\n  ── {title} ──")
        print(f"  {'Match':<6} {'Equipe A':<22} {'Score':^7} {'Equipe B':<22} "
              f"{'P(A)':>6} {'P(B)':>6} {'Freq':>6}  Vainqueur")
        print(f"  {'─'*84}")
        for mid in active:
            r  = results[mid]
            la = label_map.get(r["ta"],"?"); lb = label_map.get(r["tb"],"?")
            lw = label_map.get(r["winner"],"?")
            score = f"{r['hg']} - {r['ag']}"
            freq  = r.get("score_freq", 0.0)
            print(f"  {mid:<6} {la:<22} {score:^7} {lb:<22} "
                  f"{r['prob_a']:>6.1%} {r['prob_b']:>6.1%} {freq:>5.1f}%  -> {lw}")

    fin   = results.get("FIN",{}); third = results.get("3RD",{})
    champ = label_map.get(fin.get("winner"),"?")
    runner= label_map.get(
        fin.get("tb") if fin.get("winner")==fin.get("ta") else fin.get("ta"),"?")
    print(f"\n  Champion   : {champ}")
    print(f"  Finaliste  : {runner}")
    print(f"  3eme place : {label_map.get(third.get('winner'),'?')}")


def print_progression(reach, label_map, standings):
    n = N_SIM
    print(f"\n{'='*85}")
    print("  Probabilites de progression — Top 20")
    print(f"{'='*85}")
    print(f"  {'Equipe':<22} {'Grp':>4} {'Rnk':>4} "
          f"{'R16':>6} {'R8':>6} {'QF':>6} {'SF':>6} {'Fin':>6} {'Titre':>7}")
    print(f"  {'─'*80}")
    rows = []
    for _, r in standings.iterrows():
        tid = int(r["team_id"])
        rows.append({
            "name":  label_map.get(tid,"?"), "group": r["group_name"],
            "rank":  int(r["fifa_ranking"]),
            "r16":   reach["R16"].get(tid,0)/n,
            "r8":    reach["R8"].get(tid,0)/n,
            "qf":    reach["QF"].get(tid,0)/n,
            "sf":    reach["SF"].get(tid,0)/n,
            "fin":   reach["FIN"].get(tid,0)/n,
            "champ": reach["CHAMP"].get(tid,0)/n,
        })
    rows.sort(key=lambda x: x["champ"], reverse=True)
    for r in rows[:20]:
        print(f"  {r['name']:<22} {r['group']:>4} {r['rank']:>4} "
              f"{r['r16']:>6.1%} {r['r8']:>6.1%} {r['qf']:>6.1%} "
              f"{r['sf']:>6.1%} {r['fin']:>6.1%} {r['champ']:>7.1%}")


# ══════════════════════════════════════════════════════════════════
# 7. SAUVEGARDE
# ══════════════════════════════════════════════════════════════════

def save(det_bracket, reach, standings, label_map, conn):
    c = conn.cursor(); n = N_SIM
    round_map = {**{f"M{i}":"R16" for i in range(1,17)},
                 **{x:"R8" for x in R8_IDS},
                 **{x:"QF" for x in QF_IDS},
                 **{x:"SF" for x in SF_IDS},
                 "3RD":"3RD","FIN":"FINAL"}

    c.execute("DROP TABLE IF EXISTS knockout_fixtures")
    c.execute("""
        CREATE TABLE knockout_fixtures (
            match_id TEXT PRIMARY KEY, round TEXT,
            team_a_id INTEGER, team_a_name TEXT,
            team_b_id INTEGER, team_b_name TEXT,
            pred_score_a INTEGER, pred_score_b INTEGER,
            prob_a_wins REAL, prob_b_wins REAL,
            score_freq REAL,
            winner_id INTEGER, winner_name TEXT
        )
    """)
    for mid, r in det_bracket.items():
        c.execute(
            "INSERT OR REPLACE INTO knockout_fixtures VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (mid, round_map.get(mid,"?"),
             r["ta"], label_map.get(r["ta"],"?"),
             r["tb"], label_map.get(r["tb"],"?"),
             r["hg"], r["ag"], r["prob_a"], r["prob_b"],
             r.get("score_freq", 0.0),
             r["winner"], label_map.get(r["winner"],"?")))

    c.execute("DROP TABLE IF EXISTS knockout_probabilities")
    c.execute("""
        CREATE TABLE knockout_probabilities (
            team_id INTEGER PRIMARY KEY, team_name TEXT,
            group_name TEXT, fifa_ranking INTEGER,
            prob_r16 REAL, prob_r8 REAL, prob_qf REAL,
            prob_sf REAL, prob_final REAL, prob_champion REAL
        )
    """)
    for _, r in standings.iterrows():
        tid = int(r["team_id"])
        c.execute(
            "INSERT OR REPLACE INTO knockout_probabilities VALUES (?,?,?,?,?,?,?,?,?,?)",
            (tid, label_map.get(tid,"?"), r["group_name"], int(r["fifa_ranking"]),
             round(reach["R16"].get(tid,0)/n,4),
             round(reach["R8"].get(tid,0)/n,4),
             round(reach["QF"].get(tid,0)/n,4),
             round(reach["SF"].get(tid,0)/n,4),
             round(reach["FIN"].get(tid,0)/n,4),
             round(reach["CHAMP"].get(tid,0)/n,4)))

    conn.commit()
    print("\nknockout_fixtures et knockout_probabilities sauvegardes")


# ══════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  Pipeline 2 — Phases finales CdM 2026")
    print("=" * 55)

    conn = get_connection()
    clf, rh, ra, idx_to_label, label_to_idx = load_models()
    standings, thirds_df, tf, label_map = load_group_data(conn)

    r16_bracket, assignment = build_r16_bracket(standings, thirds_df, label_map)
    print_r16_bracket(r16_bracket, assignment, label_map)

    input("\n  Appuie sur Entree pour lancer les simulations...")

    reach, matchup_counts, pair_score_counts, pair_winner_counts = run_monte_carlo(
        r16_bracket, tf, clf, rh, ra, label_to_idx)

    det_bracket = build_det_bracket(
        r16_bracket, matchup_counts,
        pair_score_counts, pair_winner_counts, label_map)

    print_det_bracket(det_bracket, label_map)
    print_progression(reach, label_map, standings)
    save(det_bracket, reach, standings, label_map, conn)

    conn.close()
    print("\nPipeline 2 termine")