"""
cleaning.py — Bénin Sentinel 360
Transformations Bronze → Silver → Gold selon l'architecture Médaillon.

    clean_dataframe()   Bronze → Silver  : types, doublons, valeurs manquantes
    polish_dataframe()  Silver → Gold    : features engineered pour les 5 axes analytiques
"""

from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

# ---------------------------------------------------------------------------
# Chemins de sortie Silver & Gold
# ---------------------------------------------------------------------------
SILVER_OUTPUT_PARQUET_PATH = Path("data/processed/benin_events_silver.parquet")
SILVER_OUTPUT_CSV_PATH     = Path("data/processed/benin_events_silver.csv")

GOLD_OUTPUT_PARQUET_PATH   = Path("data/processed/benin_events_gold.parquet")
GOLD_OUTPUT_CSV_PATH       = Path("data/processed/benin_events_gold.csv")

# ---------------------------------------------------------------------------
# Constantes métier
# ---------------------------------------------------------------------------

# Mapping EventRootCode → pilier Sentinel 360
_PILIER_MAP: dict[str, str] = {
    "18": "securite", "19": "securite", "20": "securite",          # conflits / violence
    "03": "economie", "04": "economie", "05": "economie",          # coopération
    "06": "economie",                                               # aide matérielle
    "11": "social",   "12": "social",   "14": "social",            # protestations / rejet
}

# Domaines / TLDs francophones (priorité sur la règle anglophone)
_FR_DOMAINS = (
    ".fr", ".bj", ".tg", ".sn", ".ci", ".ml", ".bf", ".ne", ".cd", ".cm",
    "rfi.fr", "lemonde.fr", "jeuneafrique.com", "africanews.com",
    "24haubenin.com", "beninwebtv.com", "fraternite.bj", "lanationbenin.com",
)

# Domaines / TLDs anglophones
_EN_DOMAINS = (
    ".uk", ".us", ".au", ".ca", ".nz", ".in", ".gh", ".ng", ".za",
    "bbc.com", "reuters.com", "apnews.com", "aljazeera.com",
    "theguardian.com", "voanews.com", "bloomberg.com",
)


# ---------------------------------------------------------------------------
# Helpers privés
# ---------------------------------------------------------------------------

def _assign_pilier(root_code) -> str:
    """Mappe un EventRootCode CAMEO sur un pilier Sentinel 360."""
    return _PILIER_MAP.get(str(root_code), "autre")


def _goldstein_label(score: float) -> str:
    """Traduit un score Goldstein en label lisible pour les décideurs."""
    if score >= 5:   return "stabilisant"
    if score >= 1:   return "légèrement_positif"
    if score >= -1:  return "neutre"
    if score >= -5:  return "déstabilisant"
    return "très_déstabilisant"


def _extract_domain(url: str) -> str:
    """Extrait le nom de domaine d'une URL (sans www.)."""
    try:
        return urlparse(str(url)).netloc.replace("www.", "") or "inconnu"
    except Exception:
        return "inconnu"


def _lang_group(domain: str) -> str:
    """Classifie un domaine en groupe linguistique (francophone / anglophone / autre)."""
    d = domain.lower()
    if any(d.endswith(s) or s in d for s in _FR_DOMAINS):
        return "francophone"
    if any(d.endswith(s) or s in d for s in _EN_DOMAINS):
        return "anglophone"
    return "autre"


def _safe_save(df: pd.DataFrame, parquet_path: Path, csv_path: Path, label: str) -> None:
    """Sauvegarde un DataFrame en Parquet + CSV, crée les dossiers si nécessaire."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    size_mb = parquet_path.stat().st_size / 1e6
    print(f"  [{label}] {len(df):,} lignes sauvegardées — {size_mb:.1f} MB → {parquet_path}")


# ---------------------------------------------------------------------------
# Bronze → Silver
# ---------------------------------------------------------------------------

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformations Bronze → Silver.

    Opérations :
        1. Copie défensive pour ne pas muter le Bronze en mémoire
        2. Conversion des types (dates, numériques, clipping outliers)
        3. Suppression des lignes avec valeurs critiques manquantes
        4. Suppression des doublons sur GLOBALEVENTID
        5. Standardisation des noms d'acteurs (strip + upper)
        6. Sauvegarde Parquet + CSV

    Args:
        df : DataFrame Bronze brut issu de run_extraction()

    Returns:
        DataFrame Silver nettoyé
    """
    df = df.copy()
    print(f"  [SILVER] Entrée : {len(df):,} lignes")

    # --- 1. Conversion des types ----------------------------------------
    # Date : DATEADDED est au format YYYYMMDDHHMMSS (entier BigQuery)
    df["DATEADDED"] = pd.to_datetime(
        df["DATEADDED"].astype(str), format="%Y%m%d%H%M%S", errors="coerce"
    )
    df["SQLDATE"] = pd.to_datetime(
        df["SQLDATE"].astype(str), format="%Y%m%d", errors="coerce"
    )

    # Numérique : clipping des échelles GDELT
    df["GoldsteinScale"] = pd.to_numeric(df["GoldsteinScale"], errors="coerce").clip(-10, 10)
    df["AvgTone"]        = pd.to_numeric(df["AvgTone"],        errors="coerce").clip(-30, 30)
    df["NumMentions"]    = pd.to_numeric(df["NumMentions"],    errors="coerce").fillna(0).astype(int)
    df["NumSources"]     = pd.to_numeric(df["NumSources"],     errors="coerce").fillna(0).astype(int)
    df["NumArticles"]    = pd.to_numeric(df["NumArticles"],    errors="coerce").fillna(0).astype(int)

    # Géolocalisation
    df["ActionGeo_Lat"]  = pd.to_numeric(df["ActionGeo_Lat"],  errors="coerce")
    df["ActionGeo_Long"] = pd.to_numeric(df["ActionGeo_Long"], errors="coerce")

    # EventCode numérique pour validation
    df["_eventcode_num"] = pd.to_numeric(df["EventCode"], errors="coerce")

    # --- 2. Suppression des valeurs critiques manquantes ----------------
    missing_actor1   = df["Actor1Name"].isna().sum()
    missing_date     = df["DATEADDED"].isna().sum()
    invalid_codes    = ((df["_eventcode_num"] < 0) | (df["_eventcode_num"] > 2000)).sum()

    print(f"  [SILVER] Lignes sans Actor1Name          : {missing_actor1:,}")
    print(f"  [SILVER] Lignes sans DATEADDED valide    : {missing_date:,}")
    print(f"  [SILVER] EventCode hors plage [0, 2000]  : {invalid_codes:,}")

    df = df.dropna(subset=["Actor1Name", "DATEADDED", "EventCode", "GoldsteinScale"])
    df = df[df["_eventcode_num"].between(0, 2000)]
    df = df.drop(columns=["_eventcode_num"])

    # --- 3. Suppression des doublons ------------------------------------
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["GLOBALEVENTID"])
    print(f"  [SILVER] Doublons supprimés : {before_dedup - len(df):,}")

    # --- 4. Standardisation Actor1Name ---------------------------------
    df["Actor1Name"] = df["Actor1Name"].str.strip().str.upper()
    df["Actor2Name"] = df["Actor2Name"].fillna("UNKNOWN").str.strip().str.upper()

    print(f"  [SILVER] Sortie : {len(df):,} lignes")

    # --- 5. Sauvegarde --------------------------------------------------
    # Note : la sauvegarde "de secours" est également faite dans pipeline.py
    # via _save() — ici on garde la logique au plus proche du module.
    _safe_save(df, SILVER_OUTPUT_PARQUET_PATH, SILVER_OUTPUT_CSV_PATH, "SILVER")

    return df


# ---------------------------------------------------------------------------
# Silver → Gold
# ---------------------------------------------------------------------------

def polish_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformations Silver → Gold : feature engineering pour les 5 axes analytiques.

    Colonnes ajoutées :
        pilier            (securite / economie / social / autre)       — Axe 2, 3
        pilier_label      (libellé CAMEO lisible)                      — Axe 2
        sentiment_proxy   (positif / neutre / négatif)                 — Axes 2, 4
        sentiment_score   (AvgTone normalisé [-1, 1])                  — Axe 4
        goldstein_label   (label décideur depuis GoldsteinScale)       — Axe 2
        event_date        (datetime propre depuis DATEADDED)           — Axe 1
        period            (YYYY-MM pour agrégations mensuelles)        — Axe 1
        source_domain     (domaine extrait de SOURCEURL)               — Axe 4
        media_lang_group  (francophone / anglophone / autre)           — Axe 4
        is_root_event     (booléen depuis IsRootEvent)                 — Axe 2

    Args:
        df : DataFrame Silver nettoyé issu de clean_dataframe()

    Returns:
        DataFrame Gold enrichi
    """
    df = df.copy()
    print(f"  [GOLD] Entrée : {len(df):,} lignes")

    # --- Axe 2 · Piliers Sentinel 360 ----------------------------------
    df["pilier"] = df["EventRootCode"].astype(str).apply(_assign_pilier)

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
    df["pilier_label"] = df["EventRootCode"].astype(str).map(CAMEO_ROOT_LABELS).fillna("Autre")

    # --- Axe 2 · Stabilité Goldstein -----------------------------------
    df["goldstein_label"] = df["GoldsteinScale"].apply(_goldstein_label)

    # IsRootEvent → booléen propre
    df["is_root_event"] = df["IsRootEvent"].fillna(0).astype(bool)

    # --- Axe 4 · Sentiment média ---------------------------------------
    df["sentiment_proxy"] = df["AvgTone"].apply(
        lambda t: "positif" if t > 2 else ("négatif" if t < -2 else "neutre")
    )
    df["sentiment_score"] = df["AvgTone"].clip(-10, 10) / 10   # normalisé [-1, 1]

    # --- Axe 1 · Dates et périodes ------------------------------------
    # event_date peut déjà exister (passé depuis Silver) — on la reconstruit si absente
    if "event_date" not in df.columns:
        df["event_date"] = df["DATEADDED"]   # déjà converti en datetime dans clean_dataframe
    df["period"]     = df["event_date"].dt.to_period("M").astype(str)
    df["week_label"] = df["event_date"].dt.to_period("W").astype(str)  # Axe 1 · granularité hebdo

    # --- Axe 4 · Source médiatique ------------------------------------
    df["source_domain"]    = df["SOURCEURL"].apply(_extract_domain)
    df["media_lang_group"] = df["source_domain"].apply(_lang_group)

    # --- Axe 5 · Géographie : filtre coordonnées valides --------------
    df["has_geo"] = (
        df["ActionGeo_Lat"].notna()
        & df["ActionGeo_Long"].notna()
        & df["ActionGeo_Lat"].between(-90, 90)
        & df["ActionGeo_Long"].between(-180, 180)
    )

    # --- Log distribution piliers ------------------------------------
    print("  [GOLD] Distribution piliers :")
    for pilier, count in df["pilier"].value_counts().items():
        pct = count / len(df) * 100
        print(f"           {pilier:<12} {count:>6,}  ({pct:.1f}%)")
    print(f"  [GOLD] Événements géolocalisés : {df['has_geo'].sum():,} / {len(df):,}")

    print(f"  [GOLD] Sortie : {len(df):,} lignes")

    # --- Sauvegarde ---------------------------------------------------
    _safe_save(df, GOLD_OUTPUT_PARQUET_PATH, GOLD_OUTPUT_CSV_PATH, "GOLD")

    return df
