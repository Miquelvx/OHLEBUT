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


FIFA_TO_NORM = {
    "Korea Republic":                  "south korea",
    "Korea DPR":                       "north korea",
    "IR Iran":                         "iran",
    "USA":                             "united states",
    "China PR":                        "china",
    "Côte d'Ivoire":                  "ivory coast",   
    "Côte d'Ivoire":                  "cote d ivoire", 
    "Congo DR":                        "congo dr",
    "Syria":                           "syria",
    "Cabo Verde":                      "cape verde",
    "Kyrgyz Republic":                 "kyrgyzstan",
    "St. Kitts and Nevis":             "saint kitts and nevis",
    "St. Lucia":                       "saint lucia",
    "Bosnia and Herzegovina":          "bosnia and herzegovina",
    "Brunei Darussalam":               "brunei darussalam",   
    "North Macedonia":                 "north macedonia",     
    "Republic of Ireland":             "republic of ireland", 
    "Sao Tome e Principe":             "sao tome e principe", 
    "St Vincent and the Grenadines":   "st vincent and the grenadines", 
    "St. Vincent and the Grenadines":  "st vincent and the grenadines",  
    "Türkiye":                         "turkiye",            
}


def _normalize(name: str) -> str:
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
# 5. Mettre à jour teams.fifa_ranking (dernier classement connu)
# ------------------------------------------------------------------

def update_teams_ranking():
    conn = get_connection()

    rankings_df = pd.read_sql_query("""
        SELECT team_name_fifa, rank, confederation
        FROM fifa_rankings
        ORDER BY rank_date DESC
    """, conn)

    if len(rankings_df) == 0:
        print("  ❌ Aucun classement en DB — lance d'abord scrape_all_rankings()")
        conn.close()
        return

    # Garder uniquement le classement le plus récent par équipe
    latest = rankings_df.groupby("team_name_fifa").first().reset_index()
    latest["team_norm"] = latest["team_name_fifa"].apply(
        lambda n: FIFA_TO_NORM.get(n, _normalize(n))
    )

    teams_df = pd.read_sql_query(
        "SELECT team_id, team_name_normalized FROM teams", conn
    )
    merged = teams_df.merge(
        latest[["team_norm", "rank", "confederation"]],
        left_on="team_name_normalized",
        right_on="team_norm",
        how="left",
    )

    c         = conn.cursor()
    updated   = 0
    not_found = []

    for _, row in merged.iterrows():
        if pd.notna(row["rank"]):
            c.execute("""
                UPDATE teams
                SET fifa_ranking  = ?,
                    confederation = ?
                WHERE team_id = ?
            """, (int(row["rank"]), row["confederation"], int(row["team_id"])))
            updated += c.rowcount
        else:
            not_found.append(row["team_name_normalized"])

    conn.commit()

    print(f"\n✅ teams.fifa_ranking mis à jour : {updated} équipes")

    if not_found:
        wc_nf = [n for n in not_found
                 if _is_wc2026(c, n)]
        if wc_nf:
            print(f"\n⚠️  {len(wc_nf)} équipes CdM 2026 sans ranking FIFA :")
            for name in sorted(wc_nf):
                print(f"   - {name}")
            print("\n   → Ajouter les entrées manquantes dans FIFA_TO_NORM")

    c.execute("""
        SELECT team_name, fifa_ranking
        FROM teams
        WHERE is_wc2026 = 1 AND fifa_ranking IS NOT NULL
        ORDER BY fifa_ranking
        LIMIT 10
    """)
    rows = c.fetchall()
    if rows:
        print("\n📊 Top 10 équipes CdM 2026 (vérification) :")
        for name, rank in rows:
            print(f"   #{rank:<4} {name}")

    conn.close()


def _is_wc2026(c, team_norm: str) -> bool:
    c.execute("""
        SELECT 1 FROM teams
        WHERE team_name_normalized = ? AND is_wc2026 = 1
    """, (team_norm,))
    return c.fetchone() is not None


# ------------------------------------------------------------------
# 6. Mettre à jour wc2026_fixtures avec le dernier ranking FIFA connu
# ------------------------------------------------------------------

def update_wc2026_rankings():
    conn = get_connection()

    # Dernière publication FIFA par équipe
    rankings_df = pd.read_sql_query("""
        SELECT team_name_fifa, rank
        FROM fifa_rankings
        ORDER BY rank_date DESC
    """, conn)

    if len(rankings_df) == 0:
        print("  ❌ Aucun classement en DB — lance d'abord scrape_all_rankings()")
        conn.close()
        return

    # Garder le ranking le plus récent par équipe
    latest = rankings_df.groupby("team_name_fifa").first().reset_index()
    latest["team_norm"] = latest["team_name_fifa"].apply(
        lambda n: FIFA_TO_NORM.get(n, _normalize(n))
    )
    # Dict norm → rank pour lookup rapide
    norm_to_rank = dict(zip(latest["team_norm"], latest["rank"].astype(int)))

    # Récupérer les fixtures CdM avec les team_id et noms normalisés
    fixtures_df = pd.read_sql_query("""
        SELECT f.fixture_id,
               th.team_name_normalized AS home_norm,
               ta.team_name_normalized AS away_norm
        FROM wc2026_fixtures f
        JOIN teams th ON th.team_id = f.home_team_id
        JOIN teams ta ON ta.team_id = f.away_team_id
        WHERE f.home_team_id IS NOT NULL
          AND f.away_team_id IS NOT NULL
    """, conn)

    if len(fixtures_df) == 0:
        print("  ⚠️  Aucune fixture CdM 2026 avec équipes assignées")
        conn.close()
        return

    # Ajouter les colonnes si elles n'existent pas
    c = conn.cursor()
    for col in ["home_fifa_ranking", "away_fifa_ranking"]:
        try:
            c.execute(f"ALTER TABLE wc2026_fixtures ADD COLUMN {col} INTEGER")
        except Exception:
            pass  # colonne déjà existante

    updated    = 0
    not_found  = set()

    for _, row in fixtures_df.iterrows():
        home_rank = norm_to_rank.get(row["home_norm"])
        away_rank = norm_to_rank.get(row["away_norm"])

        if home_rank is None: not_found.add(row["home_norm"])
        if away_rank is None: not_found.add(row["away_norm"])

        if home_rank is not None or away_rank is not None:
            c.execute("""
                UPDATE wc2026_fixtures
                SET home_fifa_ranking = ?,
                    away_fifa_ranking = ?
                WHERE fixture_id = ?
            """, (home_rank, away_rank, int(row["fixture_id"])))
            updated += c.rowcount

    conn.commit()

    print(f"\n✅ wc2026_fixtures mis à jour : {updated} fixtures")
    print(f"   Ranking home/away injecté depuis la dernière publication FIFA")

    if not_found:
        print(f"\n⚠️  {len(not_found)} équipes sans ranking FIFA dans wc2026_fixtures :")
        for name in sorted(not_found):
            print(f"   - {name}")
        print("   → Ajouter dans FIFA_TO_NORM (load_fifa_ranking.py)")

    # Vérification : top 5 matchs avec rankings
    c.execute("""
        SELECT home_team_label, home_fifa_ranking,
               away_team_label, away_fifa_ranking
        FROM wc2026_fixtures
        WHERE home_fifa_ranking IS NOT NULL
        ORDER BY home_fifa_ranking
        LIMIT 5
    """)
    rows = c.fetchall()
    if rows:
        print("\n📊 Vérification (5 meilleures équipes domicile) :")
        for home, hr, away, ar in rows:
            print(f"   #{hr:<4} {home:<25} vs #{ar:<4} {away}")

    conn.close()


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
    update_teams_ranking()
    update_wc2026_rankings()    

    print("\n🎉 Classements FIFA chargés et features mises à jour.")