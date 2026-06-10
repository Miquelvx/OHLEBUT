import sys
import os
from datetime import datetime, timezone
sys.path.append(os.path.dirname(__file__))
from init_db import get_connection


def add_result_columns():
    conn = get_connection()
    c    = conn.cursor()

    # Ajouter les colonnes si elles n'existent pas encore
    for col in ["result_90", "result_winner"]:
        try:
            c.execute(f"ALTER TABLE matches ADD COLUMN {col} TEXT")
        except Exception:
            pass  # colonne déjà existante

    c.execute("""
        UPDATE matches SET result_90 = CASE
            WHEN status = 'PEN' THEN 'D'
            WHEN goal_diff > 0   THEN 'H'
            WHEN goal_diff < 0   THEN 'A'
            ELSE                      'D'
        END
    """)

    # Ajouter les colonnes penalty si elles n'existent pas (migration DB existante)
    for col in ["penalty_home", "penalty_away"]:
        try:
            c.execute(f"ALTER TABLE matches ADD COLUMN {col} INTEGER")
        except Exception:
            pass

    c.execute("""
        UPDATE matches SET result_winner = CASE
            WHEN status = 'PEN' AND penalty_home > penalty_away THEN 'H'
            WHEN status = 'PEN' AND penalty_away > penalty_home THEN 'A'
            WHEN goal_diff > 0                                  THEN 'H'
            WHEN goal_diff < 0                                  THEN 'A'
            ELSE NULL
        END
    """)

    conn.commit()

    # Vérification
    c.execute("SELECT status, result_90, result_winner, COUNT(*) FROM matches GROUP BY status, result_90, result_winner ORDER BY status")
    rows = c.fetchall()
    conn.close()

    print(f"🔄 Colonnes result_90 et result_winner calculées :")
    print(f"   {'Status':<8} {'result_90':<12} {'result_winner':<15} {'Nb matchs':>10}")
    print(f"   {'─'*50}")
    for status, r90, rw, nb in rows:
        print(f"   {str(status):<8} {str(r90):<12} {str(rw):<15} {nb:>10}")


def clean_friendlies():
    conn = get_connection()
    c    = conn.cursor()

    # Compter avant nettoyage
    c.execute("SELECT COUNT(*) FROM matches WHERE competition_id = 10")
    total_before = c.fetchone()[0]

    # Supprimer les amicaux où AUCUNE des deux équipes n'est qualifiée CdM 2026
    c.execute("""
        DELETE FROM matches
        WHERE competition_id = 10
        AND home_team_id NOT IN (SELECT team_id FROM teams WHERE is_wc2026 = 1)
        AND away_team_id NOT IN (SELECT team_id FROM teams WHERE is_wc2026 = 1)
    """)
    deleted = c.rowcount

    # Compter après nettoyage
    c.execute("SELECT COUNT(*) FROM matches WHERE competition_id = 10")
    total_after = c.fetchone()[0]

    conn.commit()
    conn.close()

    print(f"\n🧹 Nettoyage des matchs amicaux :")
    print(f"   Avant   : {total_before} matchs")
    print(f"   Supprimés : {deleted} matchs (aucune équipe CdM 2026)")
    print(f"   Après   : {total_after} matchs")


def update_collection_log():
    conn = get_connection()
    c    = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()

    # Recalculer le nombre réel de matchs par compétition/saison dans matches
    c.execute("""
        SELECT competition_id, season, COUNT(*) as nb
        FROM matches
        WHERE competition_id = 10
        GROUP BY competition_id, season
    """)
    rows = c.fetchall()

    updated = 0
    for (comp_id, season, nb) in rows:
        c.execute("""
            UPDATE collection_log
            SET matches_collected = ?, last_updated = ?
            WHERE source = 'api_football'
            AND competition_id = ?
            AND season = ?
        """, (nb, now, comp_id, season))
        updated += c.rowcount

    conn.commit()
    conn.close()
    print(f"📋 collection_log mis à jour : {updated} entrée(s) amicaux recalculées")


def print_summary():
    conn = get_connection()
    c    = conn.cursor()

    print(f"\n📊 Résumé du training set après nettoyage :")
    print(f"{'─'*55}")

    c.execute("""
        SELECT competition, COUNT(*) as nb
        FROM matches
        GROUP BY competition
        ORDER BY nb DESC
    """)
    rows = c.fetchall()
    total = 0
    for comp, nb in rows:
        print(f"   {comp:<45} {nb:>5} matchs")
        total += nb

    print(f"{'─'*55}")
    print(f"   {'TOTAL':<45} {total:>5} matchs")

    # Nombre d'équipes distinctes
    c.execute("SELECT COUNT(DISTINCT team_id) FROM teams")
    nb_teams = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM teams WHERE is_wc2026 = 1")
    nb_wc    = c.fetchone()[0]
    print(f"\n   Équipes dans la DB        : {nb_teams}")
    print(f"   Équipes qualifiées CdM 2026 : {nb_wc}")

    conn.close()


if __name__ == "__main__":
    print("=" * 55)
    print("  Nettoyage du training set")
    print("=" * 55)

    add_result_columns()
    clean_friendlies()
    update_collection_log()
    print_summary()

    print("\n✅ Nettoyage terminé.")