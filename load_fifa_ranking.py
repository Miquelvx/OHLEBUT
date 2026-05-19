"""
Scraping des classements FIFA historiques depuis api.fifa.com.

Endpoint : https://api.fifa.com/api/v3/fifarankings/rankings/rankingsbyschedule
Paramètres : rankingScheduleId=FRS_Male_Football_YYYYMMDD, language=en
"""

import os
import sys
import time
import json
import re
import unicodedata
import requests
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../collect"))
from init_db import get_connection

RANKING_PAGE  = "https://www.fifa.com/fifa-world-ranking/men"
API_URL       = "https://api.fifa.com/api/v3/fifarankings/rankings/rankingsbyschedule"
REQUEST_DELAY = 2

HEADERS_PAGE = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

HEADERS_API = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.fifa.com/fifa-world-ranking/men",
    "Origin":  "https://www.fifa.com",
}


# ------------------------------------------------------------------
# FIFA_TO_NORM : nom FIFA exact → team_name_normalized (clé DB)
#
# Principe :
#   teams.team_name_normalized = canonical_name(nom_api) = normalize(nom_fifa_cible)
#   C'est toujours du texte en minuscules purs (cf. team_utils.NAME_ALIASES).
#
#   FIFA_TO_NORM[team_name_fifa] doit retourner exactement cette valeur,
#   pour que la jointure fifa_rankings → match_features fonctionne.
#
# Clé   : nom EXACT retourné par l'API FIFA (sensible à la casse et aux accents)
# Valeur: valeur de teams.team_name_normalized = normalize_name(nom_fifa_cible)
#
# IMPORTANT : les noms FIFA avec accents (Côte d'Ivoire, Türkiye) doivent
# être saisis avec leurs accents exacts, tels que l'API les retourne.
# ------------------------------------------------------------------
FIFA_TO_NORM = {
    # ── Noms FIFA → noms DB normalisés ─────────────────────────────
    # FIFA différent du nom DB : on traduit explicitement
    "Korea Republic":                  "south korea",
    "Korea DPR":                       "north korea",
    "IR Iran":                         "iran",
    "USA":                             "united states",
    "China PR":                        "china",
    "Côte d'Ivoire":                  "ivory coast",    # accent sur Ô obligatoire
    "Congo DR":                        "congo dr",
    "Syria":                           "syria",
    "Cabo Verde":                      "cape verde",
    "Kyrgyz Republic":                 "kyrgyzstan",
    "St. Kitts and Nevis":             "saint kitts and nevis",
    "St. Lucia":                       "saint lucia",

    # Bosnia : FIFA="Bosnia and Herzegovina", DB norm="bosnia and herzegovina"
    "Bosnia and Herzegovina":          "bosnia and herzegovina",

    # 7 équipes corrigées
    "Brunei Darussalam":               "brunei darussalam",   # DB="Brunei"
    "North Macedonia":                 "north macedonia",     # DB="FYR Macedonia"
    "Republic of Ireland":             "republic of ireland", # DB="Rep. Of Ireland"
    "Sao Tome e Principe":             "sao tome e principe", # DB="Sao Tome and Principe"
    "St Vincent and the Grenadines":   "st vincent and the grenadines",  # DB="St. Vincent / Grenadines"
    "St. Vincent and the Grenadines":  "st vincent and the grenadines",  # variante avec point
    "Türkiye":                         "turkiye",             # tréma sur U obligatoire
}


def _normalize(name: str) -> str:
    """
    Normalisation identique à team_utils.normalize_name.
    Dupliquée ici pour éviter l'import circulaire.
    Utilisée en fallback dans update_match_features() pour les équipes
    non listées dans FIFA_TO_NORM (dont le nom FIFA == nom normalisé DB).
    """
    nfkd    = unicodedata.normalize("NFKD", name)
    ascii_n = nfkd.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-z0-9 ]", " ", ascii_n.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


# ------------------------------------------------------------------
# 1. Créer la table fifa_rankings
# ------------------------------------------------------------------

def create_rankings_table():
    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS fifa_rankings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            rank_date       TEXT    NOT NULL,
            team_name_fifa  TEXT    NOT NULL,
            rank            INTEGER NOT NULL,
            points          REAL,
            confederation   TEXT,
            UNIQUE(rank_date, team_name_fifa)
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Table fifa_rankings créée")


# ------------------------------------------------------------------
# 2. Récupérer les dates depuis la page FIFA
# ------------------------------------------------------------------

def fetch_available_dates() -> list[dict]:
    print("\n📅 Récupération des dates de publication FIFA...")
    r = requests.get(RANKING_PAGE, headers=HEADERS_PAGE, timeout=30)

    if r.status_code != 200:
        print(f"  ❌ Erreur HTTP {r.status_code}")
        return []

    pattern = r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>'
    match   = re.search(pattern, r.text, re.DOTALL)
    if not match:
        print("  ❌ JSON embarqué non trouvé")
        return []

    data  = json.loads(match.group(1))
    dates = data["props"]["pageProps"]["pageData"]["ranking"]["dates"]

    flat = []
    for year_group in dates:
        for d in year_group.get("dates", []):
            flat.append({
                "id":   d["id"],
                "date": d["matchWindowEndDate"],
            })

    flat_2022 = [d for d in flat if d["date"] >= "2022-01-01"]
    print(f"   {len(flat_2022)} publications FIFA depuis 2022")
    return flat_2022


# ------------------------------------------------------------------
# 3. Scraper les classements par date
# ------------------------------------------------------------------

def fetch_ranking_for_date(schedule_id: str, date_str: str) -> list[dict]:
    r = requests.get(
        API_URL,
        headers=HEADERS_API,
        params={"rankingScheduleId": schedule_id, "language": "en"},
        timeout=30,
    )

    if r.status_code != 200 or not r.text.strip() or r.text.strip().startswith("<"):
        return []

    data     = r.json()
    results  = data.get("Results", [])
    rankings = []

    for item in results:
        name_list = item.get("TeamName", [])
        name = next(
            (n["Description"] for n in name_list if "en" in n.get("Locale", "").lower()),
            name_list[0]["Description"] if name_list else ""
        )
        rank   = item.get("Rank")
        points = item.get("TotalPoints")
        conf   = item.get("ConfederationName", "")

        if name and rank:
            rankings.append({
                "rank_date":      date_str,
                "team_name_fifa": name,
                "rank":           int(rank),
                "points":         float(points) if points else None,
                "confederation":  conf,
            })

    return rankings


def scrape_all_rankings(dates: list[dict]) -> int:
    conn     = get_connection()
    c        = conn.cursor()
    inserted = 0

    print(f"\n📥 Scraping de {len(dates)} publications FIFA...")

    for i, d in enumerate(dates):
        print(f"   [{i+1}/{len(dates)}] {d['date']}...", end=" ", flush=True)

        c.execute("SELECT COUNT(*) FROM fifa_rankings WHERE rank_date = ?", (d["date"],))
        if c.fetchone()[0] > 0:
            print("déjà en DB ✓")
            continue

        rankings = fetch_ranking_for_date(d["id"], d["date"])

        if not rankings:
            print("0 résultats ⚠️")
            continue

        for row in rankings:
            try:
                c.execute("""
                    INSERT OR IGNORE INTO fifa_rankings
                        (rank_date, team_name_fifa, rank, points, confederation)
                    VALUES (?, ?, ?, ?, ?)
                """, (row["rank_date"], row["team_name_fifa"],
                      row["rank"], row["points"], row["confederation"]))
                inserted += c.rowcount
            except Exception:
                pass

        conn.commit()
        print(f"{len(rankings)} équipes ✅")
        time.sleep(REQUEST_DELAY)

    conn.close()
    print(f"\n✅ {inserted} classements insérés au total")
    return inserted


# ------------------------------------------------------------------
# 4. Mettre à jour match_features
# ------------------------------------------------------------------

def update_match_features():
    """
    Injecte home_fifa_ranking, away_fifa_ranking et ranking_gap dans
    match_features pour chaque match, en utilisant le classement FIFA
    le plus récent AVANT la date du match.

    Stratégie de jointure :
      teams.team_name_normalized  ←→  FIFA_TO_NORM[fifa_rankings.team_name_fifa]

    team_name_normalized est toujours en minuscules purs (invariant garanti
    par canonical_name dans team_utils.py). FIFA_TO_NORM traduit chaque nom
    FIFA vers cette même clé normalisée.

    Fallback : si aucun classement n'est disponible AVANT la date du match
    (cas des matchs de juin 2022, antérieurs à la première publication),
    on utilise le classement le plus proche APRÈS la date.
    """
    conn = get_connection()

    features_df = pd.read_sql_query("""
        SELECT mf.match_id, mf.match_date,
               mf.home_team_id, mf.away_team_id,
               th.team_name_normalized AS home_norm,
               ta.team_name_normalized AS away_norm
        FROM match_features mf
        JOIN teams th ON th.team_id = mf.home_team_id
        JOIN teams ta ON ta.team_id = mf.away_team_id
    """, conn)

    rankings_df = pd.read_sql_query("""
        SELECT rank_date, team_name_fifa, rank
        FROM fifa_rankings
        ORDER BY rank_date ASC
    """, conn)

    if len(rankings_df) == 0:
        print("  ❌ Aucun classement en DB — lance d'abord scrape_all_rankings()")
        conn.close()
        return

    print(f"\n📊 Mise à jour de {len(features_df)} matchs...")

    # Construire l'index norm → DataFrame trié par date.
    # FIFA_TO_NORM traduit team_name_fifa → team_name_normalized.
    # Fallback : _normalize() pour les équipes dont le nom FIFA == norm DB
    # (la grande majorité des ~200 équipes non listées dans FIFA_TO_NORM).
    rankings_df["team_norm"] = rankings_df["team_name_fifa"].apply(
        lambda n: FIFA_TO_NORM.get(n, _normalize(n))
    )
    rankings_by_norm = {
        norm: group.sort_values("rank_date")
        for norm, group in rankings_df.groupby("team_norm")
    }

    def get_ranking(team_norm: str, match_date: str) -> int | None:
        group = rankings_by_norm.get(team_norm)
        if group is None:
            return None
        # Classement le plus récent avant la date
        past = group[group["rank_date"] <= match_date]
        if len(past) > 0:
            return int(past.iloc[-1]["rank"])
        # Fallback : premier classement disponible après la date
        future = group[group["rank_date"] > match_date]
        if len(future) > 0:
            return int(future.iloc[0]["rank"])
        return None

    c         = conn.cursor()
    updated   = 0
    not_found = set()

    for _, row in features_df.iterrows():
        home_rank = get_ranking(row["home_norm"], row["match_date"])
        away_rank = get_ranking(row["away_norm"], row["match_date"])
        rank_gap  = (away_rank - home_rank) if (home_rank is not None and away_rank is not None) else None

        if home_rank is None: not_found.add(row["home_norm"])
        if away_rank is None: not_found.add(row["away_norm"])

        c.execute("""
            UPDATE match_features
            SET home_fifa_ranking = ?,
                away_fifa_ranking = ?,
                ranking_gap       = ?
            WHERE match_id = ?
        """, (home_rank, away_rank, rank_gap, row["match_id"]))
        updated += c.rowcount

    conn.commit()
    conn.close()

    print(f"✅ {updated} matchs mis à jour")
    if not_found:
        print(f"\n⚠️  {len(not_found)} équipes sans classement FIFA :")
        for name in sorted(not_found):
            print(f"   - {name}")
        print("\n   → Ajouter les entrées manquantes dans FIFA_TO_NORM (load_fifa_ranking.py)")
        print("     et dans NAME_ALIASES (team_utils.py)")
    else:
        print("✅ Toutes les équipes ont un classement")


# ------------------------------------------------------------------
# Point d'entrée
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  Scraping classements FIFA — WC2026 Predictor")
    print("=" * 55)

    create_rankings_table()
    dates = fetch_available_dates()

    if not dates:
        print("❌ Impossible de récupérer les dates")
        exit(1)

    scrape_all_rankings(dates)
    update_match_features()

    print("\n🎉 Classements FIFA chargés et features mises à jour.")