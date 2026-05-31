"""
Utilitaires pour la gestion des équipes dans la base de données.

Principe de déduplication :
  - Chaque équipe a un ID interne auto-incrémenté (teams.team_id)
  - La clé de déduplication est le nom normalisé (minuscules, sans accents)
  - Une table team_id_map fait le lien entre les IDs externes et l'ID interne
  - Toutes les FK dans matches et wc2026_fixtures pointent vers teams.team_id

Filtrage :
  - Équipes U21/U23/olympiques/réserves filtrées par regex
  - Clubs de football filtrés par liste explicite
  - Territoires non-FIFA filtrés par liste explicite
"""

import re
import unicodedata


# ------------------------------------------------------------------
# Mots-clés signalant une équipe non-senior
# ------------------------------------------------------------------
_EXCLUDED_KEYWORDS = [
    r"\bU\d{2}\b",
    r"\bUnder.?\d{2}\b",
    r"\bOlympic\b",
    r"\bOlympics\b",
    r"\bReserves?\b",
]
_EXCLUDED_RE = re.compile("|".join(_EXCLUDED_KEYWORDS), re.IGNORECASE)

# ------------------------------------------------------------------
# Clubs infiltrés via certaines compétitions CONCACAF/amicaux
# ------------------------------------------------------------------
EXCLUDED_CLUBS = {
    "Leon", "Los Angeles FC", "Philadelphia Union", "Tigres UANL",
    "Atlas", "Vancouver Whitecaps", "CD Motagua", "CD Olimpia",
    "CF Pachuca", "LD Alajuelense", "Orlando City SC", "Tauro FC",
    "Real Espana", "Alianza", "Austin", "Violette AC",
    "Alanyaspor", "Hull City", "Ghana B","FC Urartu"
}

# ------------------------------------------------------------------
# Territoires sans classement FIFA officiel
# ------------------------------------------------------------------
EXCLUDED_TERRITORIES = {
    "Guadeloupe", "Martinique", "French Guyana", "Sint Maarten",
    "Saint Martin", "Bonaire", "Basque Country", "Mação",
    "Kosovo B", "Catalonia",
}


def is_national_senior(team_name: str) -> bool:
    """Retourne True si l'équipe est une sélection nationale senior valide."""
    if team_name in EXCLUDED_CLUBS:
        return False
    if team_name in EXCLUDED_TERRITORIES:
        return False
    if _EXCLUDED_RE.search(team_name):
        return False
    return True


# ------------------------------------------------------------------
# Normalisation du nom pour déduplication
# ------------------------------------------------------------------

def normalize_name(name: str) -> str:
    """
    Transforme un nom d'équipe en clé de déduplication :
    minuscules, sans accents, sans ponctuation, espaces uniques.

    Ex : "Côte d'Ivoire"       → "cote d ivoire"
         "Bosnia-Herzegovina"  → "bosnia herzegovina"
         "Türkiye"             → "turkiye"
         "St. Lucia"           → "st lucia"
    """
    nfkd    = unicodedata.normalize("NFKD", name)
    ascii_n = nfkd.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-z0-9 ]", " ", ascii_n.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


# ------------------------------------------------------------------
# Correspondances manuelles : nom normalisé → nom normalisé canonique
#
# RÈGLE ABSOLUE :
#   - Les CLÉS   sont le résultat de normalize_name(nom_api)
#   - Les VALEURS sont le résultat de normalize_name(nom_fifa_exact)
#
# Cela garantit que team_name_normalized en DB est TOUJOURS en
# minuscules purs, ce qui permet la jointure fiable avec fifa_rankings
# via load_fifa_ranking.FIFA_TO_NORM.
#
# Cas couverts :
#   1. Deux APIs utilisent des noms différents pour la même équipe
#      → on unifie vers le nom normalisé FIFA
#   2. L'API retourne un nom plus long / abbrévié que le nom FIFA
#      → on mappe vers le normalize(nom_fifa)
# ------------------------------------------------------------------
NAME_ALIASES = {
    # ── Unification noms API → noms FIFA normalisés ────────────────
    # FIFA="Bosnia and Herzegovina" / API peut donner "Bosnia-Herzegovina"
    # normalize("Bosnia-Herzegovina") = "bosnia herzegovina"
    "bosnia herzegovina":              "bosnia and herzegovina",
    "bosnia and herzegovina":          "bosnia and herzegovina",

    # FIFA="Brunei Darussalam" / API donne parfois juste "Brunei"
    "brunei":                          "brunei darussalam",
    "brunei darussalam":               "brunei darussalam",

    # FIFA="North Macedonia" / API donne parfois "FYR Macedonia" (ancien nom)
    "fyr macedonia":                   "north macedonia",
    "north macedonia":                 "north macedonia",

    # FIFA="Republic of Ireland" / API donne "Rep. Of Ireland"
    # normalize("Rep. Of Ireland") = "rep of ireland"
    "rep of ireland":                  "republic of ireland",
    "republic of ireland":             "republic of ireland",

    # FIFA="Sao Tome e Principe" / API donne "Sao Tome and Principe"
    "sao tome and principe":           "sao tome e principe",
    "sao tome e principe":             "sao tome e principe",

    # FIFA="St Vincent and the Grenadines" / API donne "St. Vincent / Grenadines"
    # normalize("St. Vincent / Grenadines") = "st vincent grenadines"
    "st vincent grenadines":           "st vincent and the grenadines",
    "st vincent and the grenadines":   "st vincent and the grenadines",

    # FIFA="Syria" / API donne "Syrian Arab Republic"
    "syrian arab republic":            "syria",

    # ── Noms FIFA avec "PR", "DR", code pays → noms DB simplifiés ──
    # L'API retourne "Korea Republic" mais la DB stocke "South Korea"
    # normalize("Korea Republic") = "korea republic" → cible = "south korea"
    "korea republic":                  "south korea",
    "korea dpr":                       "north korea",
    "ir iran":                         "iran",
    "china pr":                        "china",
    "usa":                             "united states",

    # normalize("Côte d'Ivoire") = "cote d ivoire" → cible = "ivory coast"
    "cote d ivoire":                   "ivory coast",

    # normalize("Kyrgyz Republic") = "kyrgyz republic" → cible = "kyrgyzstan"
    "kyrgyz republic":                 "kyrgyzstan",

    # normalize("Cabo Verde") = "cabo verde" → cible = "cape verde"
    "cabo verde":                      "cape verde",

    # normalize("St. Kitts and Nevis") = "st kitts and nevis" → cible = "saint kitts and nevis"
    "st kitts and nevis":              "saint kitts and nevis",
    "st lucia":                        "saint lucia",

    # FIFA="Türkiye" → normalize → "turkiye" (pas d'alias nécessaire,
    # mais on le garde pour la lisibilité)
    "turkiye":                         "turkiye",
}


def canonical_name(name: str) -> str:
    """
    Retourne la clé de déduplication canonique d'un nom d'équipe.
    Résultat : toujours une chaîne en minuscules sans accents (norm pure).
    C'est cette valeur qui est stockée dans teams.team_name_normalized
    et utilisée comme clé de jointure avec fifa_rankings.
    """
    norm = normalize_name(name)
    return NAME_ALIASES.get(norm, norm)


# ------------------------------------------------------------------
# Upsert d'une équipe avec déduplication
# ------------------------------------------------------------------

def upsert_team(c, team_name: str, source: str, external_id: int,
                country_code: str = None, is_wc2026: int = 0) -> int | None:
    """
    Insère ou retrouve une équipe dans teams par son nom canonique.
    Retourne l'ID interne ou None si l'équipe est filtrée.
    """
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

    # Chercher par nom canonique
    c.execute("""
        SELECT team_id FROM teams WHERE team_name_normalized = ?
    """, (canon,))
    row = c.fetchone()

    if row:
        internal_id = row[0]
    else:
        c.execute("""
            INSERT INTO teams (team_name, team_name_normalized, country_code, is_wc2026)
            VALUES (?, ?, ?, ?)
        """, (team_name, canon, country_code, is_wc2026))
        internal_id = c.lastrowid

    # Mettre à jour fd_id ou apif_id
    if source == "football_data":
        c.execute("UPDATE teams SET fd_id = ? WHERE team_id = ?",
                  (external_id, internal_id))
    elif source == "api_football":
        c.execute("UPDATE teams SET apif_id = ? WHERE team_id = ?",
                  (external_id, internal_id))

    c.execute("""
        INSERT OR IGNORE INTO team_id_map (source, external_id, team_id)
        VALUES (?, ?, ?)
    """, (source, external_id, internal_id))

    if is_wc2026:
        c.execute("UPDATE teams SET is_wc2026 = 1 WHERE team_id = ?",
                  (internal_id,))

    return internal_id


def resolve_team_id(c, source: str, external_id: int) -> int | None:
    c.execute("""
        SELECT team_id FROM team_id_map
        WHERE source = ? AND external_id = ?
    """, (source, external_id))
    row = c.fetchone()
    return row[0] if row else None