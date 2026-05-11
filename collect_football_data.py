"""
Collecte des matchs via API-Football (dashboard.api-football.com).

Couvre :
  - Qualifications CdM 2026 (UEFA, CONMEBOL, CAF, AFC, CONCACAF, OFC)
  - UEFA Nations League 2022-23 & 2024-25
  - Euro 2024 qualifications + Euro 2024
  - Copa América 2024, AFCON 2023/2025, Gold Cup 2023, Asian Cup 2023
  - Matchs amicaux internationaux 2023-2025

Stratégie :
  - Checkpoint par (competition_id, season) dans collection_log
  - Arrêt automatique si quota proche de 100 req/jour
  - Reprise sans doublon grâce à ON CONFLICT DO NOTHING

Limite : 100 requêtes/jour sur le tier gratuit.
"""

import os
import time
import sqlite3
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
import config

load_dotenv()

API_KEY  = config.API_KEY_API_SPORT
BASE_URL = "https://v3.football.api-sports.io"
HEADERS  = {
    "x-rapidapi-host": "v3.football.api-sports.io",
    "x-rapidapi-key":  API_KEY,
}

# Délai entre requêtes (éviter burst)
REQUEST_DELAY  = 2   # secondes
# Seuil de sécurité : on s'arrête si quota restant < QUOTA_SAFETY
QUOTA_SAFETY   = 10

import sys
sys.path.append(os.path.dirname(__file__))
from init_db import get_connection
from team_utils import upsert_team, resolve_team_id

# Compteur de requêtes de la session courante
_requests_used    = 0
_requests_remaining = 100


# ------------------------------------------------------------------
# Helpers HTTP
# ------------------------------------------------------------------

def _get(endpoint: str, params: dict = None) -> dict:
    """Appel API avec suivi du quota et gestion des erreurs."""
    global _requests_used, _requests_remaining

    url = f"{BASE_URL}{endpoint}"
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=15)
    except requests.RequestException as e:
        print(f"  ❌ Erreur réseau : {e}")
        return {}

    # Lecture des headers de quota
    remaining = response.headers.get("x-ratelimit-requests-remaining")
    if remaining is not None:
        _requests_remaining = int(remaining)

    if response.status_code == 429:
        print("  ⏳ Rate limit atteint, pause 60s...")
        time.sleep(60)
        return _get(endpoint, params)

    if response.status_code != 200:
        print(f"  ❌ Erreur HTTP {response.status_code} sur {url}")
        return {}

    _requests_used += 1
    time.sleep(REQUEST_DELAY)
    return response.json()


def _quota_ok() -> bool:
    """Retourne False si on approche de la limite quotidienne."""
    if _requests_remaining <= QUOTA_SAFETY:
        print(f"\n⚠️  Quota presque épuisé ({_requests_remaining} restantes).")
        print("    Relance le script demain pour continuer.")
        return False
    return True


# ------------------------------------------------------------------
# Helpers DB
# ------------------------------------------------------------------

def _result(home: int, away: int) -> str | None:
    if home is None or away is None:
        return None
    if home > away:  return "H"
    if home < away:  return "A"
    return "D"


def _update_log(source, comp_id, season, status, count):
    conn = get_connection()
    c    = conn.cursor()
    now  = datetime.now(timezone.utc).isoformat()
    c.execute("""
        UPDATE collection_log
        SET status = ?, matches_collected = ?, last_updated = ?
        WHERE source = ? AND competition_id = ? AND season = ?
    """, (status, count, now, source, comp_id, season))
    conn.commit()
    conn.close()


def _get_pending() -> list[tuple]:
    """Retourne les entrées de collection_log en statut 'pending'."""
    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
        SELECT competition_id, competition_name, season
        FROM   collection_log
        WHERE  source = 'api_football' AND status = 'pending'
        ORDER  BY competition_id, season
    """)
    rows = c.fetchall()
    conn.close()
    return rows


# ------------------------------------------------------------------
# Collecte d'une compétition / saison
# ------------------------------------------------------------------

def collect_competition(comp_id: int, comp_name: str, season: str) -> bool:
    """
    Collecte tous les matchs d'une compétition pour une saison donnée.
    Gère la pagination automatiquement.
    Retourne False si le quota est épuisé.
    """
    print(f"\n📥  {comp_name} {season}  (id={comp_id})")

    if not _quota_ok():
        return False

    # Récupération des fixtures (paginées par 20 dans l'API)
    inserted = 0
    total    = 0

    conn = get_connection()
    c    = conn.cursor()

    if not _quota_ok():
        conn.close()
        return False

    data = _get("/fixtures", params={
        "league": comp_id,
        "season": season,
    })

    if not data or "response" not in data:
        conn.commit()
        conn.close()
        _update_log("api_football", comp_id, season, "error", 0)
        return True

    fixtures = data["response"]
    total    = data.get("results", len(fixtures))

    if fixtures:

        for f in fixtures:
            fixture  = f["fixture"]
            teams    = f["teams"]
            goals    = f["goals"]
            status   = fixture["status"]["short"]   # FT, NS, PST…

            # On ne garde QUE les matchs terminés (scores obligatoires dans matches)
            if status not in ("FT", "AET", "PEN"):
                continue

            home_goals = goals.get("home")
            away_goals = goals.get("away")

            # Ignorer si les scores sont manquants malgré le statut FT
            if home_goals is None or away_goals is None:
                continue

            # Terrain neutre : heuristique sur le nom du lieu
            venue_name = (fixture.get("venue") or {}).get("name", "") or ""
            neutral    = 1 if _is_neutral(comp_id, venue_name) else 0

            home_id    = teams["home"]["id"]
            away_id    = teams["away"]["id"]

            match_date = fixture["date"][:10] if fixture.get("date") else None
            stage      = f.get("league", {}).get("round", "")
            goal_diff  = home_goals - away_goals

            # Déduplication : résoudre les IDs internes via team_utils
            internal_home = upsert_team(c,
                team_name=teams["home"]["name"],
                source="api_football",
                external_id=home_id,
                country_code=teams["home"].get("code"),
            )
            internal_away = upsert_team(c,
                team_name=teams["away"]["name"],
                source="api_football",
                external_id=away_id,
                country_code=teams["away"].get("code"),
            )

            # Ignorer si l'une des équipes est filtrée (U21, club, etc.)
            if internal_home is None or internal_away is None:
                continue

            c.execute("""
                INSERT INTO matches (
                    match_id, source, competition, competition_id,
                    season, stage, match_date,
                    home_team_id, away_team_id,
                    home_goals, away_goals, goal_diff, result,
                    neutral_venue, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(match_id) DO NOTHING
            """, (
                fixture["id"],
                "api_football",
                comp_name,
                comp_id,
                season,
                stage,
                match_date,
                internal_home,
                internal_away,
                home_goals,
                away_goals,
                goal_diff,
                _result(home_goals, away_goals),
                neutral,
                status,
            ))
            inserted += c.rowcount

    conn.commit()
    conn.close()
    # Si 0 matchs insérés, on garde le statut "pending" pour pouvoir relancer plus tard
    status = "done" if inserted > 0 else "pending"
    _update_log("api_football", comp_id, season, status, inserted)
    if inserted == 0:
        print(f"  ⏸️   0 matchs insérés — statut maintenu en 'pending' (données pas encore disponibles)")
    else:
        print(f"  ✅  {inserted} matchs insérés (total API : {total})")
    return True


def _is_neutral(comp_id: int, venue_name: str) -> bool:
    """
    Heuristique simple : les grandes compétitions sur terrain neutre.
    Les tournois continentaux et la CdM sont toujours neutres.
    Pour les amicaux et qualifications, on garde home/away.
    """
    NEUTRAL_COMPETITIONS = {1, 4, 6, 7, 9, 22, 536}  # CdM 2022, Euros, AFCON, Asian Cup, Copa Am, Gold Cup, CONCACAF NL
    if comp_id in NEUTRAL_COMPETITIONS:
        return True
    return False


# ------------------------------------------------------------------
# Point d'entrée : collecte de toutes les compétitions en attente
# ------------------------------------------------------------------

def run():
    if not API_KEY:
        print("❌ API_FOOTBALL_KEY manquant dans le fichier .env")
        exit(1)

    pending = _get_pending()

    if not pending:
        print("✅  Toutes les compétitions ont déjà été collectées !")
        return

    print("=" * 55)
    print(f"  Collecte API-Football — {len(pending)} compétition(s) en attente")
    print(f"  Quota estimé disponible : {_requests_remaining} requêtes")
    print("=" * 55)

    for (comp_id, comp_name, season) in pending:
        ok = collect_competition(comp_id, comp_name, season)
        if not ok:
            break

    print(f"\n📊  Requêtes utilisées cette session : {_requests_used}")
    print(f"    Requêtes restantes aujourd'hui   : {_requests_remaining}")

    remaining_pending = _get_pending()
    if remaining_pending:
        print(f"\n⏳  {len(remaining_pending)} compétition(s) restantes.")
        print("    Relance le script demain pour continuer.")
    else:
        print("\n🎉  Collecte API-Football terminée !")


if __name__ == "__main__":
    run()
