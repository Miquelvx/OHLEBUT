import os
import time
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
import config as config
load_dotenv()

API_KEY  = config.API_KEY_FOOTBALL_DATA
BASE_URL = "https://api.football-data.org/v4"
HEADERS  = {"X-Auth-Token": API_KEY}
REQUEST_DELAY = 7

import sys
sys.path.append(os.path.dirname(__file__))
from init_db import get_connection
from team_utils import upsert_team


def _get(endpoint, params=None):
    url = f"{BASE_URL}{endpoint}"
    r = requests.get(url, headers=HEADERS, params=params)
    if r.status_code == 429:
        print("  ⏳ Rate limit atteint, pause 60s...")
        time.sleep(60)
        return _get(endpoint, params)
    if r.status_code != 200:
        print(f"  ❌ Erreur {r.status_code} sur {url}")
        return {}
    time.sleep(REQUEST_DELAY)
    return r.json()


# ------------------------------------------------------------------
# Collecte des 48 équipes qualifiées → table teams (is_wc2026 = 1)
# ------------------------------------------------------------------

def collect_teams():
    print("\n📥 Collecte des 48 équipes CdM 2026...")
    data = _get("/competitions/WC/teams", params={"season": 2026})

    if not data or "teams" not in data:
        print("  ⚠️  Aucune équipe récupérée.")
        return

    conn = get_connection()
    c    = conn.cursor()
    inserted = 0

    for team in data["teams"]:
        tid = upsert_team(c,
            team_name=team["name"],
            source="football_data",
            external_id=team["id"],
            country_code=team.get("tla"),
            is_wc2026=1,
        )
        if tid:
            inserted += 1

    conn.commit()
    conn.close()
    print(f"  ✅ {len(data['teams'])} équipes traitées ({inserted} nouvelles)")


# ------------------------------------------------------------------
# Collecte des fixtures → table wc2026_fixtures
# ------------------------------------------------------------------

def collect_fixtures():
    print("\n📥 Collecte des fixtures CdM 2026...")
    data = _get("/competitions/WC/matches", params={"season": 2026})

    if not data or "matches" not in data:
        print("  ⚠️  Aucun match récupéré.")
        return

    conn = get_connection()
    c    = conn.cursor()
    inserted = 0
    total    = len(data["matches"])

    for m in data["matches"]:
        home_id   = m["homeTeam"].get("id")
        away_id   = m["awayTeam"].get("id")

        # Label affiché : nom de l'équipe si connue, sinon placeholder API
        home_label = m["homeTeam"].get("name") or m["homeTeam"].get("shortName") or "TBD"
        away_label = m["awayTeam"].get("name") or m["awayTeam"].get("shortName") or "TBD"

        # Extraction du groupe depuis le stage (ex: "GROUP_STAGE" + group field)
        stage      = m.get("stage", "")
        group_name = m.get("group", None)
        # football-data.org renvoie ex : "GROUP_A" → on extrait la lettre
        if group_name and "_" in group_name:
            group_name = group_name.split("_")[-1]

        match_date = m["utcDate"][:10] if m.get("utcDate") else None

        # Scores réels si le match est déjà joué (CdM en cours)
        actual_home = m["score"]["fullTime"].get("home")
        actual_away = m["score"]["fullTime"].get("away")
        actual_result = None
        if actual_home is not None and actual_away is not None:
            if actual_home > actual_away:   actual_result = "H"
            elif actual_home < actual_away: actual_result = "A"
            else:                           actual_result = "D"

        # Résoudre les IDs internes via team_id_map
        internal_home_id = None
        internal_away_id = None
        if home_id:
            internal_home_id = upsert_team(c, home_label, "football_data", home_id)
        if away_id:
            internal_away_id = upsert_team(c, away_label, "football_data", away_id)

        c.execute("""
            INSERT INTO wc2026_fixtures (
                fixture_id, group_name, stage, match_date,
                home_team_id, away_team_id,
                home_team_label, away_team_label,
                actual_home_goals, actual_away_goals, actual_result
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fixture_id) DO UPDATE SET
                home_team_id      = excluded.home_team_id,
                away_team_id      = excluded.away_team_id,
                home_team_label   = excluded.home_team_label,
                away_team_label   = excluded.away_team_label,
                actual_home_goals = excluded.actual_home_goals,
                actual_away_goals = excluded.actual_away_goals,
                actual_result     = excluded.actual_result
        """, (
            m["id"], group_name, stage, match_date,
            internal_home_id, internal_away_id,
            home_label, away_label,
            actual_home, actual_away, actual_result,
        ))
        inserted += c.rowcount

    conn.commit()
    conn.close()
    print(f"  ✅ {inserted} fixtures insérées (sur {total} récupérées)")


# ------------------------------------------------------------------
# Point d'entrée
# ------------------------------------------------------------------

if __name__ == "__main__":
    if not API_KEY:
        print("❌ FOOTBALL_DATA_API_KEY manquant")
        exit(1)

    print("=" * 55)
    print("  Collecte football-data.org — CdM 2026")
    print("=" * 55)

    collect_teams()
    collect_fixtures()

    print("\n🎉 Collecte football-data.org terminée.")