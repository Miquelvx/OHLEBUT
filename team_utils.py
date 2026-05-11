"""
Utilitaires pour la gestion des équipes dans la base de données.

Principe de déduplication :
  - Chaque équipe a un ID interne auto-incrémenté (teams.team_id)
  - La clé de déduplication est le nom normalisé (minuscules, sans accents)
  - Une table team_id_map fait le lien entre les IDs externes (football-data.org
    et API-Football) et l'ID interne
  - Toutes les FK dans matches et wc2026_fixtures pointent vers teams.team_id

Filtrage des équipes parasites :
  - On filtre les équipes U21, U23, U20, U19, U18, U17
  - On filtre les équipes de club (Real España, etc.) qui peuvent apparaître
    dans certaines compétitions sur API-Football
"""

import re
import unicodedata


# ------------------------------------------------------------------
# Mots-clés qui signalent une équipe NON senior / NON nationale
# ------------------------------------------------------------------
_EXCLUDED_KEYWORDS = [
    r"\bU\d{2}\b",          # U21, U23, U20, U19, U18, U17...
    r"\bUnder.?\d{2}\b",    # Under-21, Under 21
    r"\bOlympic\b",         # équipes olympiques
    r"\bOlympics\b",
    r"\bB\b",               # équipes B (ex: "France B")
    r"\bReserves?\b",       # équipes réserve
]
_EXCLUDED_RE = re.compile("|".join(_EXCLUDED_KEYWORDS), re.IGNORECASE)


def is_national_senior(team_name: str) -> bool:
    """Retourne True si le nom correspond à une équipe nationale senior."""
    return not bool(_EXCLUDED_RE.search(team_name))


# ------------------------------------------------------------------
# Normalisation du nom pour déduplication
# ------------------------------------------------------------------

def normalize_name(name: str) -> str:
    """
    Normalise un nom d'équipe pour la déduplication.
    Ex: 'Côte d\'Ivoire' → 'cote d ivoire'
        'Korea Republic' → 'korea republic'
    """
    # Décomposition unicode puis suppression des accents
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    # Minuscules et suppression des caractères spéciaux
    cleaned = re.sub(r"[^a-z0-9 ]", " ", ascii_name.lower())
    # Normalisation des espaces
    return re.sub(r"\s+", " ", cleaned).strip()


# ------------------------------------------------------------------
# Correspondances manuelles pour les noms qui diffèrent entre sources
# ------------------------------------------------------------------
# Format : {nom_normalisé_api_football: nom_normalisé_football_data}
# Permet de réconcilier les variantes de noms entre les deux APIs.

NAME_ALIASES = {
    # API-Football         → football-data.org
    "korea republic":       "south korea",
    "republic of ireland":  "ireland",
    "china pr":             "china",
    "usa":                  "united states",
    "ir iran":              "iran",
    "cape verde islands":   "cape verde",
    "dpr korea":            "north korea",
    "trinidad tobago":      "trinidad and tobago",
    "bosnia":               "bosnia and herzegovina",
    "czechia":              "czech republic",
}


def canonical_name(name: str) -> str:
    """Retourne le nom canonique après normalisation et résolution des alias."""
    norm = normalize_name(name)
    return NAME_ALIASES.get(norm, norm)


# ------------------------------------------------------------------
# Upsert d'une équipe avec déduplication
# ------------------------------------------------------------------

def upsert_team(c, team_name: str, source: str, external_id: int,
                country_code: str = None, is_wc2026: int = 0) -> int | None:
    """
    Insère ou retrouve une équipe dans la table teams par son nom canonique.
    Met à jour team_id_map avec la correspondance source → ID interne.

    Retourne l'ID interne (teams.team_id) ou None si l'équipe est filtrée.

    Args:
        c            : curseur SQLite actif
        team_name    : nom brut de l'équipe tel que renvoyé par l'API
        source       : 'football_data' | 'api_football'
        external_id  : ID de l'équipe dans la source externe
        country_code : code à 3 lettres (optionnel)
        is_wc2026    : 1 si qualifiée pour la CdM 2026
    """
    # Filtre équipes non seniors / non nationales
    if not is_national_senior(team_name):
        return None

    canon = canonical_name(team_name)

    # Vérifier si on connaît déjà cet ID externe
    c.execute("""
        SELECT team_id FROM team_id_map
        WHERE source = ? AND external_id = ?
    """, (source, external_id))
    row = c.fetchone()
    if row:
        return row[0]

    # Chercher par nom canonique dans teams
    c.execute("""
        SELECT team_id FROM teams
        WHERE team_name_normalized = ?
    """, (canon,))
    row = c.fetchone()

    if row:
        # Équipe déjà connue sous un autre ID externe → on ajoute le mapping
        internal_id = row[0]
    else:
        # Nouvelle équipe → insertion
        c.execute("""
            INSERT INTO teams (team_name, team_name_normalized, country_code, is_wc2026)
            VALUES (?, ?, ?, ?)
        """, (team_name, canon, country_code, is_wc2026))
        internal_id = c.lastrowid

    # Mettre à jour fd_id ou apif_id selon la source
    if source == "football_data":
        c.execute("UPDATE teams SET fd_id = ? WHERE team_id = ?",
                  (external_id, internal_id))
    elif source == "api_football":
        c.execute("UPDATE teams SET apif_id = ? WHERE team_id = ?",
                  (external_id, internal_id))

    # Enregistrer le mapping externe → interne
    c.execute("""
        INSERT OR IGNORE INTO team_id_map (source, external_id, team_id)
        VALUES (?, ?, ?)
    """, (source, external_id, internal_id))

    # Mettre à jour is_wc2026 si nécessaire
    if is_wc2026:
        c.execute("UPDATE teams SET is_wc2026 = 1 WHERE team_id = ?",
                  (internal_id,))

    return internal_id


def resolve_team_id(c, source: str, external_id: int) -> int | None:
    """
    Résout un ID externe vers l'ID interne.
    Retourne None si le mapping n'existe pas.
    """
    c.execute("""
        SELECT team_id FROM team_id_map
        WHERE source = ? AND external_id = ?
    """, (source, external_id))
    row = c.fetchone()
    return row[0] if row else None
