"""
Nettoyage du training set.

Règles appliquées :
  - Matchs amicaux (competition_id = 10) : garder uniquement les matchs
    où au moins une des deux équipes est qualifiée pour la CdM 2026
  - Toutes les autres compétitions officielles : on garde tout

Lance ce script UNE SEULE FOIS après la collecte complète.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))
from init_db import get_connection


def add_result_columns():
    """
    Ajoute les colonnes result_90 et result_winner à la table matches.

    - result_90   : résultat à 90 minutes (H/D/A), jamais influencé par les
                    prolongations ou les tirs au but
    - result_winner : vainqueur effectif (H/A uniquement), utile pour les
                    phases finales. Égal à result_90 sauf pour AET et PEN
                    où on détermine le vainqueur depuis le score réel.

    Statuts API-Football :
      FT  = Match Finished (90 min)
      AET = After Extra Time (prolongations)
      PEN = Penalties (tirs au but, score à 90min souvent à égalité)
    """
    conn = get_connection()
    c    = conn.cursor()

    # Ajouter les colonnes si elles n'existent pas encore
    for col in ["result_90", "result_winner"]:
        try:
            c.execute(f"ALTER TABLE matches ADD COLUMN {col} TEXT")
        except Exception:
            pass  # colonne déjà existante

    # result_90 : basé sur home_goals - away_goals à 90min
    # Pour AET/PEN, le score final inclut les prolongations mais pas les pénaltys
    # On utilise le goal_diff pour déterminer le résultat à 90min
    # Note : pour les PEN, le score à 90min est toujours nul → result_90 = D
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

    # result_winner : vainqueur effectif du match
    # - FT/AET : basé sur le goal_diff (prolongations incluses)
    # - PEN    : basé sur penalty_home vs penalty_away
    # - Nul en phase de groupe : NULL (pas de vainqueur)
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
    """Met à jour collection_log avec le bon nombre de matchs amicaux après nettoyage."""
    conn = get_connection()
    c    = conn.cursor()
    from datetime import datetime, timezone
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
    """Affiche un résumé du training set après nettoyage."""
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