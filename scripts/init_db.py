import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "../data/wc2026.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(os.path.abspath(DB_PATH))


def init_db():
    conn = get_connection()
    c = conn.cursor()

    # ------------------------------------------------------------------
    # Table : teams
    # ------------------------------------------------------------------
    c.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            team_id             INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name           TEXT    NOT NULL UNIQUE,
            team_name_normalized TEXT,
            country_code        TEXT,
            confederation       TEXT,
            fifa_ranking        INTEGER,
            is_wc2026           INTEGER DEFAULT 0,
            fd_id               INTEGER,
            apif_id             INTEGER
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS team_id_map (
            source      TEXT    NOT NULL,
            external_id INTEGER NOT NULL,
            team_id     INTEGER NOT NULL REFERENCES teams(team_id),
            PRIMARY KEY (source, external_id)
        )
    """)

    # ------------------------------------------------------------------
    # Table : matches  (training set)
    # ------------------------------------------------------------------
    c.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            match_id        INTEGER PRIMARY KEY,
            source          TEXT    NOT NULL,
            competition     TEXT    NOT NULL,
            competition_id  INTEGER,
            season          TEXT,
            stage           TEXT,
            match_date      TEXT    NOT NULL,
            home_team_id    INTEGER NOT NULL REFERENCES teams(team_id),
            away_team_id    INTEGER NOT NULL REFERENCES teams(team_id),
            home_goals      INTEGER NOT NULL,
            away_goals      INTEGER NOT NULL,
            goal_diff       INTEGER NOT NULL,
            result          TEXT    NOT NULL,
            neutral_venue   INTEGER DEFAULT 0,
            status          TEXT,
            penalty_home    INTEGER,
            penalty_away    INTEGER
        )
    """)

    # ------------------------------------------------------------------
    # Table : wc2026_fixtures  (données de prédiction)
    # ------------------------------------------------------------------
    c.execute("""
        CREATE TABLE IF NOT EXISTS wc2026_fixtures (
            fixture_id          INTEGER PRIMARY KEY,
            group_name          TEXT,
            stage               TEXT    NOT NULL,
            match_date          TEXT,
            home_team_id        INTEGER REFERENCES teams(team_id),
            away_team_id        INTEGER REFERENCES teams(team_id),
            home_team_label     TEXT,
            away_team_label     TEXT,
            pred_proba_home     REAL,
            pred_proba_draw     REAL,
            pred_proba_away     REAL,
            pred_home_goals     REAL,
            pred_away_goals     REAL,
            pred_winner_id      INTEGER REFERENCES teams(team_id),
            actual_home_goals   INTEGER,
            actual_away_goals   INTEGER,
            actual_result       TEXT
        )
    """)

    # ------------------------------------------------------------------
    # Table : collection_log
    # ------------------------------------------------------------------
    c.execute("""
        CREATE TABLE IF NOT EXISTS collection_log (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            source           TEXT    NOT NULL,
            competition_id   INTEGER NOT NULL,
            competition_name TEXT,
            season           TEXT    NOT NULL,
            status           TEXT    NOT NULL,
            matches_collected INTEGER DEFAULT 0,
            last_updated     TEXT,
            UNIQUE(source, competition_id, season)
        )
    """)

    conn.commit()
    conn.close()
    print("✅  Base de données initialisée : data/wc2026.db")
    print("    Tables : teams | matches | wc2026_fixtures | collection_log")


def seed_collection_log():
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

        # CONCACAF Gold Cup
        ("api_football",  22,  "CONCACAF Gold Cup",            "2023"),
        ("api_football",  22,  "CONCACAF Gold Cup",            "2025"),

        # CONCACAF Nations League
        ("api_football", 536,  "CONCACAF Nations League",      "2022"),
        ("api_football", 536,  "CONCACAF Nations League",      "2023"),
        ("api_football", 536,  "CONCACAF Nations League",      "2024"),

        # Matchs amicaux
        ("api_football",  10,  "International Friendlies",     "2023"),
        ("api_football",  10,  "International Friendlies",     "2024"),
        ("api_football",  10,  "International Friendlies",     "2025"),
        ("api_football",  10,  "International Friendlies",     "2026"),
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