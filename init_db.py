"""
Initialisation de la base de données SQLite du projet WC2026 Predictor.

Tables :
  - teams            : référentiel de toutes les équipes nationales connues
  - matches          : matchs terminés (training set, 2022-2026)
  - wc2026_fixtures  : matchs de la CdM 2026 (prédiction + résultats réels)
  - collection_log   : checkpoint de collecte par compétition/saison
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "data/wc2026.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(os.path.abspath(DB_PATH))


def init_db():
    conn = get_connection()
    c = conn.cursor()

    # ------------------------------------------------------------------
    # Table : teams
    # Référentiel de toutes les équipes nationales rencontrées.
    # On stocke bien TOUTES les équipes (pas seulement les 48 qualifiées)
    # pour éviter les matchs contre des équipes "inconnues" du modèle.
    # ------------------------------------------------------------------
    c.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            team_id             INTEGER PRIMARY KEY AUTOINCREMENT,  -- ID interne projet
            team_name           TEXT    NOT NULL UNIQUE,            -- clé de déduplication
            team_name_normalized TEXT,                              -- nom en minuscules sans accents
            country_code        TEXT,                               -- code à 3 lettres (ex : FRA)
            confederation       TEXT,                               -- UEFA, CONMEBOL, CAF, AFC, CONCACAF, OFC
            fifa_ranking        INTEGER,                            -- classement FIFA
            is_wc2026           INTEGER DEFAULT 0,                  -- 1 = qualifié pour la CdM 2026
            fd_id               INTEGER,                            -- ID football-data.org
            apif_id             INTEGER                             -- ID API-Football
        )
    """)

    # Table de correspondance externe → interne
    c.execute("""
        CREATE TABLE IF NOT EXISTS team_id_map (
            source      TEXT    NOT NULL,   -- 'football_data' | 'api_football'
            external_id INTEGER NOT NULL,
            team_id     INTEGER NOT NULL REFERENCES teams(team_id),
            PRIMARY KEY (source, external_id)
        )
    """)

    # ------------------------------------------------------------------
    # Table : matches  (training set)
    # Matchs TERMINÉS entre la CdM 2022 et la CdM 2026.
    # Les deux équipes sont toujours connues (home_team_id NOT NULL).
    # Usage : entraînement et validation du modèle ML.
    # ------------------------------------------------------------------
    c.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            match_id        INTEGER PRIMARY KEY,   -- ID natif API-Football
            source          TEXT    NOT NULL,      -- 'api_football'
            competition     TEXT    NOT NULL,      -- nom de la compétition
            competition_id  INTEGER,               -- ID dans API-Football
            season          TEXT,                  -- ex : '2022', '2023'
            stage           TEXT,                  -- 'Group Stage', 'Final', 'Friendly'...
            match_date      TEXT    NOT NULL,      -- format ISO : YYYY-MM-DD
            home_team_id    INTEGER NOT NULL REFERENCES teams(team_id),
            away_team_id    INTEGER NOT NULL REFERENCES teams(team_id),
            home_goals      INTEGER NOT NULL,      -- score final (matchs terminés uniquement)
            away_goals      INTEGER NOT NULL,
            goal_diff       INTEGER NOT NULL,      -- home_goals - away_goals
            result          TEXT    NOT NULL,      -- 'H' | 'D' | 'A'
            neutral_venue   INTEGER DEFAULT 0,     -- 1 = terrain neutre
            status          TEXT,                  -- 'FT', 'AET', 'PEN'
            penalty_home    INTEGER,               -- score tirs au but domicile (NULL si pas de PEN)
            penalty_away    INTEGER                -- score tirs au but extérieur (NULL si pas de PEN)
        )
    """)

    # ------------------------------------------------------------------
    # Table : wc2026_fixtures  (données de prédiction)
    # Tous les matchs de la CdM 2026, phases de groupe ET phases finales.
    # - home/away_team_id peuvent être NULL pour les phases finales
    #   (ex : "Winner Group A" avant que le groupe soit joué)
    # - Les colonnes pred_* sont remplies par le modèle ML
    # - Les colonnes actual_* sont remplies au fur et à mesure du tournoi
    # ------------------------------------------------------------------
    c.execute("""
        CREATE TABLE IF NOT EXISTS wc2026_fixtures (
            fixture_id          INTEGER PRIMARY KEY,   -- ID football-data.org
            group_name          TEXT,                  -- 'A'...'L' ou NULL (phases finales)
            stage               TEXT    NOT NULL,      -- 'GROUP_STAGE', 'ROUND_OF_32'...
            match_date          TEXT,                  -- YYYY-MM-DD
            home_team_id        INTEGER REFERENCES teams(team_id),
            away_team_id        INTEGER REFERENCES teams(team_id),
            home_team_label     TEXT,                  -- nom affiché (ex: "Winner Group A")
            away_team_label     TEXT,
            pred_proba_home     REAL,                  -- probabilité victoire domicile
            pred_proba_draw     REAL,                  -- probabilité match nul
            pred_proba_away     REAL,                  -- probabilité victoire extérieur
            pred_winner_id      INTEGER REFERENCES teams(team_id),
            actual_home_goals   INTEGER,               -- rempli après le match
            actual_away_goals   INTEGER,
            actual_result       TEXT                   -- 'H' | 'D' | 'A' après le match
        )
    """)

    # ------------------------------------------------------------------
    # Table : collection_log
    # Checkpoint par (source, competition_id, season).
    # Permet de reprendre la collecte API-Football sans tout relancer
    # si on atteint la limite de 100 req/jour.
    # ------------------------------------------------------------------
    c.execute("""
        CREATE TABLE IF NOT EXISTS collection_log (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            source           TEXT    NOT NULL,
            competition_id   INTEGER NOT NULL,
            competition_name TEXT,
            season           TEXT    NOT NULL,
            status           TEXT    NOT NULL,     -- 'pending' | 'done' | 'error'
            matches_collected INTEGER DEFAULT 0,
            last_updated     TEXT,                 -- timestamp ISO
            UNIQUE(source, competition_id, season)
        )
    """)

    conn.commit()
    conn.close()
    print("✅  Base de données initialisée : data/wc2026.db")
    print("    Tables : teams | matches | wc2026_fixtures | collection_log")


def seed_collection_log():
    """
    Pré-remplit collection_log avec toutes les compétitions à collecter
    via API-Football (training set uniquement).
    La CdM 2026 est gérée séparément par collect_football_data.py.
    """

    # Format : (source, competition_id, competition_name, season)
    COMPETITIONS = [
        # CdM 2022
        ("api_football",   1,  "FIFA World Cup 2022",          "2022"),

        # Qualifications CdM 2026 (IDs et saisons corrigés)
        ("api_football",  32,  "WC Qualification Europe",      "2024"),
        ("api_football",  34,  "WC Qualification CONMEBOL",    "2026"),
        ("api_football",  29,  "WC Qualification CAF",         "2023"),
        ("api_football",  30,  "WC Qualification Asia",        "2026"),
        ("api_football",  31,  "WC Qualification CONCACAF",    "2026"),
        ("api_football",  33,  "WC Qualification OFC",         "2026"),
        ("api_football",  37,  "WC Qualification Intercontinental", "2026"),

        # UEFA Nations League
        ("api_football",   5,  "UEFA Nations League",          "2022"),
        ("api_football",   5,  "UEFA Nations League",          "2024"),

        # Euro 2024
        ("api_football", 960,  "Euro 2024 Qualifications",     "2023"),
        ("api_football",   4,  "UEFA Euro",                    "2024"),

        # Tournois continentaux
        ("api_football",   9,  "Copa América",                 "2024"),
        ("api_football",   6,  "Africa Cup of Nations",        "2023"),
        ("api_football",   6,  "Africa Cup of Nations",        "2025"),
        ("api_football",   7,  "AFC Asian Cup",                "2023"),

        # CONCACAF Gold Cup (ID corrigé : 22, pas 16)
        ("api_football",  22,  "CONCACAF Gold Cup",            "2023"),
        ("api_football",  22,  "CONCACAF Gold Cup",            "2025"),

        # CONCACAF Nations League (3 éditions sur la période)
        ("api_football", 536,  "CONCACAF Nations League",      "2022"),
        ("api_football", 536,  "CONCACAF Nations League",      "2023"),
        ("api_football", 536,  "CONCACAF Nations League",      "2024"),

        # Matchs amicaux
        ("api_football",  10,  "International Friendlies",     "2023"),
        ("api_football",  10,  "International Friendlies",     "2024"),
        ("api_football",  10,  "International Friendlies",     "2025"),
    ]

    conn = get_connection()
    c = conn.cursor()
    inserted = 0

    for (source, comp_id, comp_name, season) in COMPETITIONS:
        try:
            c.execute("""
                INSERT OR IGNORE INTO collection_log
                    (source, competition_id, competition_name, season, status)
                VALUES (?, ?, ?, ?, 'pending')
            """, (source, comp_id, comp_name, season))
            inserted += c.rowcount
        except sqlite3.Error as e:
            print(f"  ⚠️  Erreur insertion {comp_name} {season} : {e}")

    conn.commit()
    conn.close()
    print(f"✅  collection_log initialisé : {inserted} compétitions en attente")


if __name__ == "__main__":
    init_db()
    seed_collection_log()