"""
cleaning.py — Bénin Sentinel 360
Transformations Bronze → Silver → Gold selon l'architecture Médaillon.

    clean_dataframe()   Bronze → Silver  : types, doublons, valeurs manquantes
    polish_dataframe()  Silver → Gold    : feature engineering complet des 5 axes

Indices produits dans le Gold :
    AXE 1  zscore_volume, rolling_avg_7j, alert_level          Détection anomalies
    AXE 2  sentinel_score (0-100), cross_signal, goldstein_norm Score composite Sentinel
    AXE 3  interaction_type, interaction_tone_label             Dynamique acteurs
    AXE 4  source_type, media_lang_group, sentiment_score       Biais médias
    AXE 5  zone, tension_contribution, tension_norm             Heatmap géographique
"""

from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Chemins de sortie Silver & Gold
# ---------------------------------------------------------------------------
SILVER_OUTPUT_PARQUET_PATH = Path("data/processed/benin_events_silver.parquet")
SILVER_OUTPUT_CSV_PATH     = Path("data/processed/benin_events_silver.csv")

GOLD_OUTPUT_PARQUET_PATH   = Path("data/processed/benin_events_gold.parquet")
GOLD_OUTPUT_CSV_PATH       = Path("data/processed/benin_events_gold.csv")

# Agrégats exportés séparément pour le dashboard (lecture rapide)
DAILY_AGG_PATH   = Path("data/processed/benin_daily_agg.parquet")
MONTHLY_AGG_PATH = Path("data/processed/benin_monthly_agg.parquet")
ZONE_AGG_PATH    = Path("data/processed/benin_zone_agg.parquet")

# ---------------------------------------------------------------------------
# Constantes métier
# ---------------------------------------------------------------------------

CAMEO_ROOT_LABELS = {
    "01": "Déclarations verbales",   "02": "Appels à l'action",
    "03": "Coopération",             "04": "Consultation",
    "05": "Engagement diplomatique", "06": "Coopération matérielle",
    "07": "Aide humanitaire",        "08": "Coopération judiciaire",
    "09": "Investigations",          "10": "Demandes",
    "11": "Désapprobation",          "12": "Rejet",
    "13": "Menaces",                 "14": "Protestations",
    "15": "Coercition",              "16": "Agression",
    "17": "Violence",                "18": "Assaut",
    "19": "Combats",                 "20": "Violence de masse",
}

# EventRootCode → pilier Sentinel 360
_PILIER_MAP: dict[str, str] = {
    "18": "securite", "19": "securite", "20": "securite",
    "03": "economie", "04": "economie", "05": "economie", "06": "economie",
    "11": "social",   "12": "social",   "14": "social",
}

# QuadClass → poids de stabilité [0, 1]
# 1=coopération verbale, 2=coopération matérielle, 3=conflit verbal, 4=conflit physique
_QUADCLASS_WEIGHT = {1: 1.0, 2: 0.75, 3: 0.35, 4: 0.0}

# ---------------------------------------------------------------------------
# Overrides manuels domaine → source_type
# ---------------------------------------------------------------------------
# WORKFLOW D'ENRICHISSEMENT :
#   1. python inspect_domains.py        → génère le top 50 des domaines réels
#   2. Annoter les "inconnu" dans le tableau affiché
#   3. Copier le dictionnaire généré ici en remplacement
#   4. python pipeline.py --skip-extract → régénère le Gold
# ---------------------------------------------------------------------------
#
# Structure du dictionnaire :
#   "domaine.tld" : "national" | "regional" | "international"
#
# Priorité de classification (_source_type) :
#   1. Override exact  (ce dictionnaire)       ← plus fiable
#   2. TLD national    (.bj)
#   3. TLD régional    (.tg .sn .ci ...)
#   4. TLD international (.fr .uk .us ...)
#   5. Générique       (.com .org .net)        ← moins fiable
#
_DOMAIN_OVERRIDES: dict[str, str] = {
    # ------------------------------------------------------------------ national — Bénin
    "ortb.bj":               "national",   # Office de Radiodiffusion du Bénin
    "fraternite.bj":         "national",   # La Fraternité
    "24haubenin.com":        "national",   # 24h au Bénin
    "beninwebtv.com":        "national",   # Bénin Web TV
    "lanationbenin.com":     "national",   # La Nation
    "matinlibre.net":        "national",   # Le Matin Libre
    "beninactu.net":         "national",   # Bénin Actu
    "beninplus.com":         "national",   # Bénin Plus
    "golfeinfo.com":         "national",   # Golfe Info
    "republicain-benin.com": "national",   # Le Républicain
    "acotonou.com":          "national",   # aCotonou
    "benintimes.net":        "national",   # Bénin Times
    "beninbrasier.com":      "national",   # Bénin Brasier
    "lenouveaureveil.com":   "national",   # Le Nouveau Réveil
    "beninpresse.bj":        "national",   # Agence Bénin Presse
    # ------------------------------------------------------------------ regional — Afrique de l'Ouest / continent
    "rfi.fr":                "regional",   # Radio France Internationale
    "jeuneafrique.com":      "regional",   # Jeune Afrique
    "africanews.com":        "regional",   # Africa News
    "icilome.com":           "regional",   # Ici Lomé (Togo)
    "abidjan.net":           "regional",   # Abidjan.net (CI)
    "dakaractu.com":         "regional",   # Dakar Actu (Sénégal)
    "linfodrome.com":        "regional",   # L'Infodrome (CI)
    "nouvelles-dafrique.com":"regional",   # Nouvelles d'Afrique
    "afrik.com":             "regional",   # Afrik.com
    "togoweb.net":           "regional",   # Togo Web
    "republicoftogo.com":    "regional",   # Republic of Togo
    "koaci.com":             "regional",   # Koaci (CI)
    "connectionivoirienne.net":"regional", # Connection Ivoirienne
    "africatime.com":        "regional",   # Africa Time
    "guineenews.org":        "regional",   # Guinée News
    "maliweb.net":           "regional",   # Mali Web
    "lefaso.net":            "regional",   # Le Faso (Burkina)
    "ouestaf.com":           "regional",   # Ouest Afrique
    # ------------------------------------------------------------------ international
    "reuters.com":           "international",  # Reuters
    "bbc.com":               "international",  # BBC
    "apnews.com":            "international",  # AP News
    "bloomberg.com":         "international",  # Bloomberg
    "theguardian.com":       "international",  # The Guardian
    "voanews.com":           "international",  # Voice of America
    "aljazeera.com":         "international",  # Al Jazeera
    "lemonde.fr":            "international",  # Le Monde
    "dw.com":                "international",  # Deutsche Welle
    "france24.com":          "international",  # France 24
    "lefigaro.fr":           "international",  # Le Figaro
    "liberation.fr":         "international",  # Libération
    "nytimes.com":           "international",  # New York Times
    "washingtonpost.com":    "international",  # Washington Post
    # ------------------------------------------------------------------ national — top 50 réel (inspect_domains.py)
    "gouv.bj":               "national",   # Gouvernement du Bénin
    "lanouvelletribune.info":"national",   # La Nouvelle Tribune Bénin
    "24haubenin.info":       "national",   # 24h au Bénin (variante .info)
    # ------------------------------------------------------------------ regional — top 50 réel (presse nigériane + panafricaine)
    "dailypost.ng":          "regional",   # Daily Post Nigeria
    "leadership.ng":         "regional",   # Leadership Nigeria
    "guardian.ng":           "regional",   # The Guardian Nigeria
    "thesun.ng":             "regional",   # The Sun Nigeria
    "blueprint.ng":          "regional",   # Blueprint Nigeria
    "theeagleonline.com.ng": "regional",   # The Eagle Online Nigeria
    "thecable.ng":           "regional",   # The Cable Nigeria
    "legit.ng":              "regional",   # Legit Nigeria
    "thenewsnigeria.com.ng": "regional",   # The News Nigeria
    "newsghana.com.gh":      "regional",   # News Ghana
    "tell.ng":               "regional",   # Tell Nigeria
    "pulse.ng":              "regional",   # Pulse Nigeria
    "allafrica.com":         "regional",   # AllAfrica — agrégateur panafricain
    "fr.allafrica.com":      "regional",   # AllAfrica français
    "ecofinagency.com":      "regional",   # Ecofin Agency — éco africaine
    "africatopsuccess.com":  "regional",   # Africa Top Success
    "myjoyonline.com":       "regional",   # Joy Online Ghana
    "ghanaweb.com":          "regional",   # Ghana Web
    "africa-newsroom.com":   "regional",   # Africa Newsroom
    "afriquinfos.com":       "regional",   # Afriquinfos
    # ------------------------------------------------------------------ international — top 50 réel (presse nigériane à diffusion mondiale)
    "punchng.com":           "international",  # Punch Nigeria
    "nigerianobservernews.com":"international",# Nigerian Observer
    "saharareporters.com":   "international",  # Sahara Reporters
    "thisdaylive.com":       "international",  # This Day Live
    "quicknews-africa.net":  "international",  # Quick News Africa
    "premiumtimesng.com":    "international",  # Premium Times Nigeria
    "thenationonlineng.net": "international",  # The Nation Nigeria
    "tribuneonlineng.com":   "international",  # Tribune Nigeria
    "dailytrust.com":        "international",  # Daily Trust Nigeria
    "nationalaccordnewspaper.com":"international", # National Accord
    "promptnewsonline.com":  "international",  # Prompt News
    "levenementprecis.com":  "international",  # L'Événement Précis Bénin
    "yahoo.com":             "international",  # Yahoo News (agrégateur)
    "opinionnigeria.com":    "international",  # Opinion Nigeria
    "channelstv.com":        "international",  # Channels TV Nigeria
    "naija247news.com":      "international",  # Naija 247 News
    "informationng.com":     "international",  # Information Nigeria
    "nigerianeye.com":       "international",  # Nigerian Eye
    "peoplesdailyng.com":    "international",  # Peoples Daily Nigeria
    "tanzanianewsreports.com":"international", # Tanzania News Reports
    "naijanews.com":         "international",  # Naija News
    "en.antaranews.com":     "international",  # Antara News (Indonésie)
    "newtelegraphng.com":    "international",  # New Telegraph Nigeria
    "hallmarknews.com":      "international",  # Hallmark News Nigeria
    "eurasiareview.com":     "international",  # Eurasia Review
}

# ---------------------------------------------------------------------------
# TLDs de secours — utilisés si le domaine n'est pas dans _DOMAIN_OVERRIDES
# ---------------------------------------------------------------------------
_NATIONAL_TLDS      = (".bj",)
_REGIONAL_TLDS      = (".tg", ".sn", ".ci", ".ml", ".bf", ".ne",
                       ".cd", ".cm", ".gh", ".ng", ".za", ".gn", ".gw")
_INTERNATIONAL_TLDS = (".fr", ".uk", ".us", ".au", ".ca",
                       ".de", ".es", ".it", ".jp", ".cn", ".ru")

# Cache de compilation — évite de recalculer la classification à chaque appel
# Alimenté automatiquement lors du premier appel à _source_type()
_SOURCE_TYPE_CACHE: dict[str, str] = {}

# Zones géographiques du Bénin par latitude
_ZONE_BOUNDS = [
    (10.5,  90,  "Nord-Frontière"),
    ( 8.5, 10.5, "Centre"),
    (-90,   8.5, "Sud-Cotonou"),
]


# ---------------------------------------------------------------------------
# Helpers privés
# ---------------------------------------------------------------------------

def _assign_pilier(root_code) -> str:
    return _PILIER_MAP.get(str(root_code), "autre")


def _goldstein_label(score: float) -> str:
    if score >= 5:   return "stabilisant"
    if score >= 1:   return "légèrement_positif"
    if score >= -1:  return "neutre"
    if score >= -5:  return "déstabilisant"
    return "très_déstabilisant"


def _extract_domain(url: str) -> str:
    try:
        return urlparse(str(url)).netloc.replace("www.", "") or "inconnu"
    except Exception:
        return "inconnu"


def _source_type(domain: str) -> str:
    """
    Classifie un domaine en national / regional / international.

    Optimisations :
        - Cache dict en mémoire : évite de reclassifier le même domaine
          deux fois (GDELT produit ~200 domaines uniques sur 40 000 lignes)
        - Priorité : override exact > TLD national > TLD régional >
          TLD international > générique .com/.org/.net > inconnu

    Pour enrichir : lancer inspect_domains.py et copier le dictionnaire
    généré dans _DOMAIN_OVERRIDES ci-dessus.
    """
    d = domain.lower()

    # 1. Lecture du cache — O(1) pour les domaines déjà vus
    if d in _SOURCE_TYPE_CACHE:
        return _SOURCE_TYPE_CACHE[d]

    # 2. Classification
    if d in _DOMAIN_OVERRIDES:
        result = _DOMAIN_OVERRIDES[d]
    elif any(d.endswith(t) for t in _NATIONAL_TLDS):
        result = "national"
    elif any(d.endswith(t) for t in _REGIONAL_TLDS):
        result = "regional"
    elif any(d.endswith(t) for t in _INTERNATIONAL_TLDS):
        result = "international"
    elif d.endswith(".com") or d.endswith(".org") or d.endswith(".net"):
        result = "international"
    else:
        result = "inconnu"

    # 3. Mise en cache pour les prochains appels
    _SOURCE_TYPE_CACHE[d] = result
    return result


def _lang_group(domain: str) -> str:
    d = domain.lower()
    fr = (".fr", ".bj", ".tg", ".sn", ".ci", ".ml", ".bf", ".ne", ".cd", ".cm",
          "rfi.fr", "lemonde.fr", "jeuneafrique.com", "africanews.com")
    en = (".uk", ".us", ".au", ".ca", ".gh", ".ng", ".za",
          "bbc.com", "reuters.com", "apnews.com", "bloomberg.com")
    if any(d.endswith(s) or s in d for s in fr):
        return "francophone"
    if any(d.endswith(s) or s in d for s in en):
        return "anglophone"
    return "autre"


def _assign_zone(lat) -> str:
    if pd.isna(lat):
        return "inconnu"
    for low, high, label in _ZONE_BOUNDS:
        if low <= lat < high:
            return label
    return "inconnu"


def _minmax_series(s: pd.Series, min_val: float, max_val: float) -> pd.Series:
    """Normalise une série sur [0, 1] avec des bornes fixes."""
    return (s.clip(min_val, max_val) - min_val) / (max_val - min_val)


def _safe_save(df: pd.DataFrame, parquet_path: Path, csv_path: Path, label: str) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    size_mb = parquet_path.stat().st_size / 1e6
    print(f"  [{label}] {len(df):,} lignes — {size_mb:.1f} MB → {parquet_path}")


# ---------------------------------------------------------------------------
# Bronze → Silver
# ---------------------------------------------------------------------------

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformations Bronze → Silver.

    Opérations :
        1. Copie défensive
        2. Conversion des types (dates, numériques, clipping)
        3. Suppression des lignes avec valeurs critiques manquantes
        4. Suppression des doublons sur GLOBALEVENTID
        5. Standardisation des noms d'acteurs
        6. Sauvegarde Parquet + CSV
    """
    df = df.copy()
    print(f"  [SILVER] Entrée : {len(df):,} lignes")

    # --- 1. Dates --------------------------------------------------------
    df["DATEADDED"] = pd.to_datetime(
        df["DATEADDED"].astype(str), format="%Y%m%d%H%M%S", errors="coerce"
    )
    df["SQLDATE"] = pd.to_datetime(
        df["SQLDATE"].astype(str), format="%Y%m%d", errors="coerce"
    )

    # --- 2. Numériques ---------------------------------------------------
    df["GoldsteinScale"] = pd.to_numeric(df["GoldsteinScale"], errors="coerce").clip(-10, 10)
    df["AvgTone"]        = pd.to_numeric(df["AvgTone"],        errors="coerce").clip(-30, 30)
    df["NumMentions"]    = pd.to_numeric(df["NumMentions"],    errors="coerce").fillna(0).astype(int)
    df["NumSources"]     = pd.to_numeric(df["NumSources"],     errors="coerce").fillna(0).astype(int)
    df["NumArticles"]    = pd.to_numeric(df["NumArticles"],    errors="coerce").fillna(0).astype(int)
    df["ActionGeo_Lat"]  = pd.to_numeric(df["ActionGeo_Lat"],  errors="coerce")
    df["ActionGeo_Long"] = pd.to_numeric(df["ActionGeo_Long"], errors="coerce")
    df["_eventcode_num"] = pd.to_numeric(df["EventCode"],      errors="coerce")

    # --- 3. Valeurs critiques manquantes ---------------------------------
    print(f"  [SILVER] Lignes sans Actor1Name       : {df['Actor1Name'].isna().sum():,}")
    print(f"  [SILVER] Lignes sans DATEADDED valide : {df['DATEADDED'].isna().sum():,}")
    print(f"  [SILVER] EventCode hors [0, 2000]     : "
          f"{((df['_eventcode_num'] < 0) | (df['_eventcode_num'] > 2000)).sum():,}")

    df = df.dropna(subset=["Actor1Name", "DATEADDED", "EventCode", "GoldsteinScale"])
    df = df[df["_eventcode_num"].between(0, 2000)]
    df = df.drop(columns=["_eventcode_num"])

    # --- 4. Doublons -----------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset=["GLOBALEVENTID"])
    print(f"  [SILVER] Doublons supprimés : {before - len(df):,}")

    # --- 5. Standardisation acteurs --------------------------------------
    df["Actor1Name"] = df["Actor1Name"].str.strip().str.upper()
    df["Actor2Name"] = df["Actor2Name"].fillna("UNKNOWN").str.strip().str.upper()

    print(f"  [SILVER] Sortie : {len(df):,} lignes")
    _safe_save(df, SILVER_OUTPUT_PARQUET_PATH, SILVER_OUTPUT_CSV_PATH, "SILVER")
    return df


# ---------------------------------------------------------------------------
# Silver → Gold — Feature Engineering complet (5 axes)
# ---------------------------------------------------------------------------

def polish_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformations Silver → Gold.

    Colonnes produites :
    ┌──────────────────────────────────────────────────────────────────────┐
    │ AXE 1 — Détection d'anomalies                                        │
    │   event_date, period, week_label                                     │
    │   rolling_avg_7j   : moyenne mobile 7j du volume NumArticles         │
    │   zscore_volume    : Z-Score quotidien (signal faible si Z > 2)      │
    │   alert_level      : vert / orange / rouge                           │
    ├──────────────────────────────────────────────────────────────────────┤
    │ AXE 2 — Indice de Stabilité Sentinel                                 │
    │   pilier, pilier_label, goldstein_label, is_root_event               │
    │   goldstein_norm   : GoldsteinScale normalisé [0, 1]                │
    │   tone_norm        : AvgTone normalisé [0, 1]                       │
    │   cross_signal     : goldstein_norm × tone_norm (dissonance)        │
    │   quadclass_norm   : QuadClass → poids stabilité [0, 1]             │
    │   sentinel_score   : score composite [0, 100]                       │
    │   sentinel_label   : critique / instable / modéré / stable           │
    ├──────────────────────────────────────────────────────────────────────┤
    │ AXE 3 — Dynamique d'influence acteurs                                │
    │   interaction_type       : paire Actor1Type × Actor2Type             │
    │   interaction_tone_label : coopération / tension / neutre            │
    ├──────────────────────────────────────────────────────────────────────┤
    │ AXE 4 — Biais médias & Soft Power                                    │
    │   source_domain, media_lang_group                                    │
    │   source_type      : national / regional / international             │
    │   sentiment_proxy  : positif / neutre / négatif                     │
    │   sentiment_score  : AvgTone normalisé [-1, 1]                      │
    ├──────────────────────────────────────────────────────────────────────┤
    │ AXE 5 — Heatmap de résilience géographique                           │
    │   has_geo               : booléen coordonnées valides               │
    │   zone                  : Nord-Frontière / Centre / Sud-Cotonou     │
    │   tension_contribution  : |GoldsteinScale négatif| × NumArticles    │
    └──────────────────────────────────────────────────────────────────────┘
    """
    df = df.copy()
    print(f"  [GOLD] Entrée : {len(df):,} lignes")

    # ===================================================================
    # AXE 1 — Détection d'anomalies et signaux faibles
    # ===================================================================

    df["event_date"] = df["SQLDATE"]
    df["period"]     = df["event_date"].dt.to_period("M").astype(str)
    df["week_label"] = df["event_date"].dt.to_period("W").astype(str)

    # Agrégation quotidienne du volume pour calculer le Z-Score
    daily_vol = (
        df.groupby(df["event_date"].dt.date)["NumArticles"]
        .sum()
        .rename("daily_volume")
        .reset_index()
        .rename(columns={"event_date": "date_key"})
        .sort_values("date_key")
    )
    daily_vol["rolling_avg_7j"] = (
        daily_vol["daily_volume"].rolling(7, min_periods=1).mean()
    )
    daily_vol["rolling_std_7j"] = (
        daily_vol["daily_volume"]
        .rolling(7, min_periods=1).std()
        .fillna(1).replace(0, 1)
    )
    daily_vol["zscore_volume"] = (
        (daily_vol["daily_volume"] - daily_vol["rolling_avg_7j"])
        / daily_vol["rolling_std_7j"]
    ).round(3)

    def _alert(z: float) -> str:
        if z > 3:  return "rouge"
        if z > 2:  return "orange"
        return "vert"

    daily_vol["alert_level"] = daily_vol["zscore_volume"].apply(_alert)
    daily_vol["date_key"]    = pd.to_datetime(daily_vol["date_key"])

    # Join retour sur le Gold
    df["_date_key"] = df["event_date"].dt.normalize()
    df = df.merge(
        daily_vol[["date_key", "rolling_avg_7j", "zscore_volume", "alert_level"]],
        left_on="_date_key", right_on="date_key", how="left"
    ).drop(columns=["_date_key", "date_key"])

    print(f"  [GOLD][AXE 1] Alertes orange+rouge : "
          f"{df['alert_level'].isin(['orange','rouge']).sum():,} événements")

    # ===================================================================
    # AXE 2 — Score composite Sentinel (0–100)
    # ===================================================================

    df["pilier"]          = df["EventRootCode"].astype(str).apply(_assign_pilier)
    df["pilier_label"]    = df["EventRootCode"].astype(str).map(CAMEO_ROOT_LABELS).fillna("Autre")
    df["goldstein_label"] = df["GoldsteinScale"].apply(_goldstein_label)
    df["is_root_event"]   = df["IsRootEvent"].fillna(0).astype(bool)

    df["goldstein_norm"] = _minmax_series(df["GoldsteinScale"], -10, 10)
    df["tone_norm"]      = _minmax_series(df["AvgTone"], -30, 30)

    # Signal croisé — pénalise la dissonance coopération/ton négatif
    # Ex : accord diplomatique (Goldstein > 0) couvert négativement → cross_signal faible
    df["cross_signal"]   = (df["goldstein_norm"] * df["tone_norm"]).round(4)
    df["quadclass_norm"] = df["QuadClass"].map(_QUADCLASS_WEIGHT).fillna(0.5)

    # Score pondéré : Goldstein 40% | Cross-signal 35% | QuadClass 25%
    df["sentinel_score"] = (
        0.40 * df["goldstein_norm"]
      + 0.35 * df["cross_signal"]
      + 0.25 * df["quadclass_norm"]
    ).mul(100).round(1)

    def _sentinel_label(score: float) -> str:
        if score >= 70: return "stable"
        if score >= 50: return "modéré"
        if score >= 30: return "instable"
        return "critique"

    df["sentinel_label"] = df["sentinel_score"].apply(_sentinel_label)

    print(f"  [GOLD][AXE 2] Sentinel score moyen  : {df['sentinel_score'].mean():.1f}/100")
    print(f"  [GOLD][AXE 2] Distribution labels   : "
          f"{df['sentinel_label'].value_counts().to_dict()}")

    # ===================================================================
    # AXE 3 — Dynamique d'influence acteurs
    # ===================================================================

    df["interaction_type"] = (
        df["Actor1Type1Code"].fillna("UNK").str.strip()
        + "×"
        + df["Actor2Type1Code"].fillna("UNK").str.strip()
    )

    def _interaction_tone(tone: float) -> str:
        if tone > 2:   return "coopération"
        if tone < -2:  return "tension"
        return "neutre"

    df["interaction_tone_label"] = df["AvgTone"].apply(_interaction_tone)

    print(f"  [GOLD][AXE 3] Paires d'interaction uniques : "
          f"{df['interaction_type'].nunique():,}")

    # ===================================================================
    # AXE 4 — Biais médias & Soft Power
    # ===================================================================

    df["source_domain"]    = df["SOURCEURL"].apply(_extract_domain)
    df["media_lang_group"] = df["source_domain"].apply(_lang_group)
    df["source_type"]      = df["source_domain"].apply(_source_type)

    df["sentiment_proxy"] = df["AvgTone"].apply(
        lambda t: "positif" if t > 2 else ("négatif" if t < -2 else "neutre")
    )
    df["sentiment_score"] = (df["AvgTone"].clip(-10, 10) / 10).round(4)

    print(f"  [GOLD][AXE 4] Distribution source_type : "
          f"{df['source_type'].value_counts().to_dict()}")

    # ===================================================================
    # AXE 5 — Heatmap de résilience géographique
    # ===================================================================

    df["has_geo"] = (
        df["ActionGeo_Lat"].notna()
        & df["ActionGeo_Long"].notna()
        & df["ActionGeo_Lat"].between(-90, 90)
        & df["ActionGeo_Long"].between(-180, 180)
    )
    df["zone"] = df["ActionGeo_Lat"].apply(_assign_zone)

    # Contribution à la tension : GoldsteinScale négatif pondéré par NumArticles
    df["tension_contribution"] = (
        df["GoldsteinScale"].clip(-10, 0).abs() * df["NumArticles"]
    ).round(2)

    print(f"  [GOLD][AXE 5] Distribution zones         : "
          f"{df['zone'].value_counts().to_dict()}")
    print(f"  [GOLD][AXE 5] Événements géolocalisés    : "
          f"{df['has_geo'].sum():,} / {len(df):,}")

    # ===================================================================
    # Sauvegarde Gold
    # ===================================================================
    print(f"  [GOLD] Sortie : {len(df):,} lignes")
    _safe_save(df, GOLD_OUTPUT_PARQUET_PATH, GOLD_OUTPUT_CSV_PATH, "GOLD")

    # ===================================================================
    # Agrégats pré-calculés pour le dashboard
    # ===================================================================
    _build_aggregates(df)

    return df


# ---------------------------------------------------------------------------
# Agrégats pré-calculés — consommés directement par le dashboard Esdras
# ---------------------------------------------------------------------------

def _build_aggregates(df: pd.DataFrame) -> None:
    """
    Produit 3 tables agrégées légères pour le dashboard.
    Évite de recalculer à chaque rechargement Streamlit.

    Tables produites :
        benin_daily_agg.parquet    — Axe 1 : volume quotidien + Z-Score + alert
        benin_monthly_agg.parquet  — Axe 2 : score Sentinel mensuel + QuadClass
        benin_zone_agg.parquet     — Axe 5 : tension par zone et par mois
    """

    # --- Table quotidienne (Axe 1) ------------------------------------
    daily = (
        df.groupby(df["event_date"].dt.date)
        .agg(
            num_articles=("NumArticles",      "sum"),
            num_events=("GLOBALEVENTID",      "count"),
            goldstein_mean=("GoldsteinScale", "mean"),
            tone_mean=("AvgTone",             "mean"),
            zscore_volume=("zscore_volume",   "first"),
            alert_level=("alert_level",       "first"),
            rolling_avg_7j=("rolling_avg_7j", "first"),
        )
        .reset_index()
        .rename(columns={"event_date": "date"})
    )
    daily["date"]           = pd.to_datetime(daily["date"])
    daily["goldstein_mean"] = daily["goldstein_mean"].round(3)
    daily["tone_mean"]      = daily["tone_mean"].round(3)

    # Filet de sécurité : recalcule rolling_avg_7j si absente ou nulle
    if daily["rolling_avg_7j"].isna().all():
        daily["rolling_avg_7j"] = daily["num_articles"].rolling(7, min_periods=1).mean()
    daily["rolling_avg_7j"] = daily["rolling_avg_7j"].round(1)

    # Recalcule zscore et alert_level si absents
    if daily["zscore_volume"].isna().all():
        std = daily["num_articles"].rolling(7, min_periods=1).std().fillna(1).replace(0, 1)
        daily["zscore_volume"] = ((daily["num_articles"] - daily["rolling_avg_7j"]) / std).round(3)
        daily["alert_level"] = daily["zscore_volume"].apply(
            lambda z: "rouge" if z > 3 else ("orange" if z > 2 else "vert")
        )

    DAILY_AGG_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(DAILY_AGG_PATH, index=False)
    print(f"  [AGG] Daily   → {DAILY_AGG_PATH}  ({len(daily)} jours)")

    # --- Table mensuelle (Axe 2) --------------------------------------
    monthly = (
        df.groupby("period")
        .agg(
            sentinel_score_mean=("sentinel_score",   "mean"),
            goldstein_mean=("GoldsteinScale",         "mean"),
            tone_mean=("AvgTone",                     "mean"),
            num_articles=("NumArticles",              "sum"),
            num_events=("GLOBALEVENTID",              "count"),
            pct_securite=("pilier", lambda x: (x == "securite").mean() * 100),
            pct_economie=("pilier", lambda x: (x == "economie").mean() * 100),
            pct_social=(  "pilier", lambda x: (x == "social").mean()   * 100),
            pct_conflict=("QuadClass", lambda x: x.isin([3, 4]).mean() * 100),
        )
        .reset_index()
    )
    for col in ["sentinel_score_mean", "goldstein_mean", "tone_mean",
                "pct_securite", "pct_economie", "pct_social", "pct_conflict"]:
        monthly[col] = monthly[col].round(2)

    MONTHLY_AGG_PATH.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(MONTHLY_AGG_PATH, index=False)
    print(f"  [AGG] Monthly → {MONTHLY_AGG_PATH}  ({len(monthly)} mois)")

    # --- Table zones (Axe 5) ------------------------------------------
    zone_monthly = (
        df[df["has_geo"]]
        .groupby(["period", "zone"])
        .agg(
            tension_sum=("tension_contribution",  "sum"),
            goldstein_mean=("GoldsteinScale",      "mean"),
            num_events=("GLOBALEVENTID",           "count"),
            num_articles=("NumArticles",           "sum"),
        )
        .reset_index()
    )

    # Normalisation tension sur [0, 100] par période
    zone_monthly["tension_norm"] = (
        zone_monthly.groupby("period")["tension_sum"]
        .transform(lambda x: (
            ((x - x.min()) / (x.max() - x.min()) * 100)
            if x.max() > x.min()
            else pd.Series([50.0] * len(x), index=x.index)
        ))
    ).round(1)

    ZONE_AGG_PATH.parent.mkdir(parents=True, exist_ok=True)
    zone_monthly.to_parquet(ZONE_AGG_PATH, index=False)
    print(f"  [AGG] Zone    → {ZONE_AGG_PATH}  ({len(zone_monthly)} lignes)")