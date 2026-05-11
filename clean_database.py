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

    clean_friendlies()
    update_collection_log()
    print_summary()

    print("\n✅ Nettoyage terminé.")