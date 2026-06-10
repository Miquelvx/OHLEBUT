/* ══════════════════════════════════════════════════
   MAP DRAPEAUX — nom d'équipe → code ISO flagcdn
   Source unifiée (superset de toutes les pages)
══════════════════════════════════════════════════ */
const FLAGS = {
  /* ── Équipes CdM 2026 ── */
  'Mexico':'mx','South Korea':'kr','Czechia':'cz','South Africa':'za',
  'Canada':'ca','Switzerland':'ch','Bosnia-Herzegovina':'ba','Qatar':'qa',
  'Brazil':'br','Morocco':'ma','Scotland':'gb-sct','Haiti':'ht',
  'United States':'us','Turkey':'tr','Australia':'au','Paraguay':'py',
  'Germany':'de','Ecuador':'ec','Ivory Coast':'ci','Curaçao':'cw',
  'Netherlands':'nl','Japan':'jp','Tunisia':'tn','Sweden':'se',
  'Belgium':'be','Iran':'ir','Egypt':'eg','New Zealand':'nz',
  'Spain':'es','Uruguay':'uy','Saudi Arabia':'sa','Cape Verde Islands':'cv',
  'France':'fr','Senegal':'sn','Norway':'no','Iraq':'iq',
  'Argentina':'ar','Austria':'at','Algeria':'dz','Jordan':'jo',
  'Portugal':'pt','Colombia':'co','Congo DR':'cd','Uzbekistan':'uz',
  'Croatia':'hr','England':'gb-eng','Panama':'pa','Ghana':'gh',
  /* ── CONMEBOL / CONCACAF ── */
  'Chile':'cl','Venezuela':'ve','Bolivia':'bo','Peru':'pe',
  'Costa Rica':'cr','Honduras':'hn','El Salvador':'sv','Guatemala':'gt',
  'Jamaica':'jm','Trinidad and Tobago':'tt','Cuba':'cu','Nicaragua':'ni',
  'Dominican Republic':'do','Grenada':'gd','Suriname':'sr',
  'Belize':'bz','Bermuda':'bm','Antigua and Barbuda':'ag','Barbados':'bb',
  'Puerto Rico':'pr','Guyana':'gy','St. Vincent / Grenadines':'vc',
  'St. Kitts and Nevis':'kn','St. Lucia':'lc','Montserrat':'ms',
  'Aruba':'aw','Bahamas':'bs','Cayman Islands':'ky',
  'British Virgin Islands':'vg','US Virgin Islands':'vi',
  'Anguilla':'ai','Turks and Caicos Islands':'tc','Dominica':'dm',
  /* ── UEFA ── */
  'Serbia':'rs','Ukraine':'ua','Poland':'pl','Italy':'it',
  'Denmark':'dk','Finland':'fi','Rep. Of Ireland':'ie','Greece':'gr',
  'Romania':'ro','Hungary':'hu','Bulgaria':'bg','Albania':'al','Kosovo':'xk',
  'Georgia':'ge','Armenia':'am','Slovakia':'sk','Slovenia':'si','Belarus':'by',
  'Estonia':'ee','Latvia':'lv','Lithuania':'lt','Kazakhstan':'kz','Moldova':'md',
  'Iceland':'is','Malta':'mt','Cyprus':'cy','Luxembourg':'lu','Montenegro':'me',
  'FYR Macedonia':'mk','North Macedonia':'mk','Gibraltar':'gi','Andorra':'ad',
  'Liechtenstein':'li','San Marino':'sm','Faroe Islands':'fo','Azerbaijan':'az',
  'Wales':'gb-wls','Northern Ireland':'gb-nir','Russia':'ru',
  /* ── AFC ── */
  'China':'cn','United Arab Emirates':'ae','Kuwait':'kw','Oman':'om',
  'Bahrain':'bh','Palestine':'ps','Syria':'sy','Kyrgyzstan':'kg',
  'Tajikistan':'tj','Indonesia':'id','Vietnam':'vn','Thailand':'th',
  'India':'in','North Korea':'kp','Myanmar':'mm','Malaysia':'my',
  'Singapore':'sg','Philippines':'ph','Cambodia':'kh','Hong Kong':'hk',
  'Turkmenistan':'tm','Lebanon':'lb','Pakistan':'pk','Bangladesh':'bd',
  'Nepal':'np','Maldives':'mv','Sri Lanka':'lk','Bhutan':'bt','Laos':'la',
  'Mongolia':'mn','Afghanistan':'af','Brunei':'bn','Timor-Leste':'tl',
  'Chinese Taipei':'tw','Guam':'gu',
  /* ── CAF ── */
  'Nigeria':'ng','Cameroon':'cm','Algeria':'dz','Tunisia':'tn','Ghana':'gh',
  'Congo':'cg','Zambia':'zm','Zimbabwe':'zw','Uganda':'ug',
  'Kenya':'ke','Tanzania':'tz','Rwanda':'rw','Ethiopia':'et','Angola':'ao',
  'Mozambique':'mz','Botswana':'bw','Namibia':'na','Malawi':'mw',
  'Mauritius':'mu','Madagascar':'mg','Mali':'ml','Burkina Faso':'bf',
  'Guinea':'gn','Benin':'bj','Gabon':'ga','Equatorial Guinea':'gq',
  'Mauritania':'mr','Sudan':'sd','South Sudan':'ss','Somalia':'so',
  'Djibouti':'dj','Libya':'ly','Togo':'tg','Sierra Leone':'sl',
  'Liberia':'lr','Guinea-Bissau':'gw','Gambia':'gm','Niger':'ne',
  'Chad':'td','Central African Republic':'cf','Burundi':'bi',
  'Seychelles':'sc','Eswatini':'sz','Lesotho':'ls','Comoros':'km',
  'Sao Tome and Principe':'st','Rwanda':'rw','Malawi':'mw',
  /* ── OFC ── */
  'Fiji':'fj','Papua New Guinea':'pg','Solomon Islands':'sb',
  'Vanuatu':'vu','Samoa':'ws','Tahiti':'pf','New Caledonia':'nc',
};

/**
 * Retourne une balise <img> de drapeau pour un nom d'équipe.
 * @param {string} name  - Nom de l'équipe (ex. 'France')
 * @param {number} w     - Largeur en px (défaut 20)
 * @param {number} h     - Hauteur en px (défaut 13)
 * @param {string} cls   - Classe CSS optionnelle
 */
function flag(name, w=20, h=13, cls='') {
  const code = FLAGS[name];
  if (!code) return `<span style="font-size:.9rem;line-height:1;opacity:.4">🏳️</span>`;
  return `<img src="https://flagcdn.com/w40/${code}.png"`
       + ` style="width:${w}px;height:${h}px;object-fit:cover;border-radius:2px;vertical-align:middle;flex-shrink:0"`
       + (cls ? ` class="${cls}"` : '')
       + ` alt="${name}" loading="lazy">`;
}

/* ══════════════════════════════════════════════════
   COULEURS COMPÉTITIONS
══════════════════════════════════════════════════ */
const COMP_COLORS = {
  'FIFA World Cup 2022':              '#F97316',
  'UEFA Euro':                        '#3B82F6',
  'Copa América':                     '#8B5CF6',
  'Africa Cup of Nations':            '#16A34A',
  'AFC Asian Cup':                    '#0EA5E9',
  'CONCACAF Gold Cup':                '#EC4899',
  'UEFA Nations League':              '#6366F1',
  'CONCACAF Nations League':          '#F59E0B',
  'Euro 2024 Qualifications':         '#60A5FA',
  'WC Qualification Europe':          '#2563EB',
  'WC Qualification CAF':             '#22C55E',
  'WC Qualification CONMEBOL':        '#A78BFA',
  'WC Qualification Asia':            '#06B6D4',
  'WC Qualification CONCACAF':        '#F97316',
  'WC Qualification OFC':             '#14B8A6',
  'WC Qualification Intercontinental':'#84CC16',
  'International Friendlies':         '#A8A29E',
};

/** Retourne la couleur hex d'une compétition. */
function compColor(c) { return COMP_COLORS[c] || '#78716C'; }

/* ══════════════════════════════════════════════════
   UTILITAIRES PROBABILITÉS
══════════════════════════════════════════════════ */

/** Formate une probabilité en pourcentage. Ex: 0.573 → '57%' */
function pct(v) { return Math.round((v || 0) * 100) + '%'; }

/** Retourne la classe CSS de couleur selon le niveau de probabilité. */
function pctClass(v) {
  if (v >= .5) return 'pct-high';
  if (v >= .2) return 'pct-med';
  return 'pct-low';
}

/* ══════════════════════════════════════════════════
   CONFÉDÉRATIONS
══════════════════════════════════════════════════ */
const CONF_SHORT = {
  'UEFA':'UEFA','CONMEBOL':'CONMEBOL','CAF':'CAF',
  'AFC':'AFC','CONCACAF':'CONCACAF','OFC':'OFC',
};