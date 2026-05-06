"""
pipeline.py — Bénin Sentinel 360
Orchestre l'extraction BigQuery + le nettoyage Bronze → Silver → Gold.

Usage :
    python pipeline.py                  # exécution complète
    python pipeline.py --dry-run        # estimation du coût uniquement
    python pipeline.py --skip-extract   # nettoyage seul (Bronze déjà présent)

Architecture Médaillon :
    Bronze  →  données brutes BigQuery (Parquet + CSV)
    Silver  →  données nettoyées       (Parquet + CSV)
    Gold    →  features engineered     (Parquet + CSV)
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

# ---------------------------------------------------------------------------
# Import des modules locaux
# ---------------------------------------------------------------------------
from extraction import (
    BRONZE_OUTPUT_CSV_PATH,
    BRONZE_OUTPUT_PARQUET_PATH,
    estimate_query_cost,
    get_table_partition_info,
    run_extraction,
)
from cleaning import (
    GOLD_OUTPUT_CSV_PATH,
    GOLD_OUTPUT_PARQUET_PATH,
    SILVER_OUTPUT_CSV_PATH,
    SILVER_OUTPUT_PARQUET_PATH,
    clean_dataframe,
    polish_dataframe,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

TABLE_ID = "gdelt-bq.gdeltv2.events_partitioned"
COST_THRESHOLD_GB = 0.5   # seuil au-delà duquel on demande confirmation

# Mapping EventRootCode → pilier Sentinel 360
PILIER_MAP = {
    "18": "securite", "19": "securite", "20": "securite",
    "03": "economie", "04": "economie", "05": "economie", "06": "economie",
    "14": "social",   "11": "social",   "12": "social",
}

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

# ---------------------------------------------------------------------------
# Requête SQL — table partitionnée, code pays BN (Bénin)
# Utilise _PARTITIONTIME pour l'élagage physique des partitions (< 200 MB)
# ---------------------------------------------------------------------------
QUERY = """
SELECT
    GLOBALEVENTID,
    DATEADDED,
    SQLDATE,
    YEAR,
    MonthYear,

    -- Événement
    IsRootEvent,
    EventCode,
    EventBaseCode,
    EventRootCode,
    QuadClass,
    GoldsteinScale,

    -- Acteur 1
    Actor1Name,
    Actor1CountryCode,
    Actor1Type1Code,

    -- Acteur 2
    Actor2Name,
    Actor2CountryCode,
    Actor2Type1Code,

    -- Volume médiatique
    NumMentions,
    NumSources,
    NumArticles,
    AvgTone,

    -- Géographie de l'action
    ActionGeo_FullName,
    ActionGeo_Type,
    ActionGeo_CountryCode,
    ActionGeo_ADM1Code,
    ActionGeo_Lat,
    ActionGeo_Long,

    -- Source
    SOURCEURL

FROM `gdelt-bq.gdeltv2.events_partitioned`
WHERE
    -- Filtre partitionné en premier (élagage physique, économise le quota)
    _PARTITIONTIME >= '2025-01-01'
    AND _PARTITIONTIME < '2025-12-31'
    AND ActionGeo_CountryCode = 'BN'
    AND GoldsteinScale IS NOT NULL
    AND NumArticles > 0
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(df: pd.DataFrame, parquet_path: Path, csv_path: Path, label: str) -> None:
    """Sauvegarde un DataFrame en Parquet ET CSV avec logs."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    size_mb = parquet_path.stat().st_size / 1e6
    print(f"  [{label}] {len(df):,} lignes — {size_mb:.1f} MB → {parquet_path}")


def _assign_pilier(root_code) -> str:
    """Assigne chaque événement à un pilier Sentinel 360."""
    return PILIER_MAP.get(str(root_code), "autre")


def _goldstein_label(score: float) -> str:
    """Traduit un score Goldstein en label décideur."""
    if score >= 5:   return "stabilisant"
    if score >= 1:   return "légèrement_positif"
    if score >= -1:  return "neutre"
    if score >= -5:  return "déstabilisant"
    return "très_déstabilisant"


# ---------------------------------------------------------------------------
# Étape 1 — Extraction Bronze
# ---------------------------------------------------------------------------

def step_extract(client: bigquery.Client, dry_run_only: bool) -> pd.DataFrame | None:
    """
    Interroge events_partitioned, sauvegarde en Bronze.
    Retourne le DataFrame ou None si dry_run_only.
    """
    print("\n=== ÉTAPE 1 : Extraction BigQuery (Bronze) ===")

    # Inspection du partitionnement (informatif)
    info = get_table_partition_info(client, TABLE_ID)
    print(f"  Partition : {info['partition_column']} ({info['partition_type']})")
    print(f"  Clustering : {info['clustering_fields']}")

    # Estimation du coût
    gb = estimate_query_cost(client, QUERY)
    print(f"  Données à scanner : {gb:.3f} GB")

    if dry_run_only:
        print("  [--dry-run] Estimation terminée. Aucune donnée extraite.")
        return None

    if gb > COST_THRESHOLD_GB:
        rep = input(f"  Attention : {gb:.2f} GB > {COST_THRESHOLD_GB} GB. Continuer ? (o/n) : ")
        if rep.strip().lower() != "o":
            print("  Requête annulée.")
            sys.exit(0)

    df_bronze = run_extraction(client, QUERY)
    _save(df_bronze, BRONZE_OUTPUT_PARQUET_PATH, BRONZE_OUTPUT_CSV_PATH, "BRONZE")
    return df_bronze


# ---------------------------------------------------------------------------
# Étape 2 — Nettoyage Silver (via cleaning.clean_dataframe)
# ---------------------------------------------------------------------------

def step_clean(df_bronze: pd.DataFrame) -> pd.DataFrame:
    """Bronze → Silver : types, doublons, valeurs manquantes critiques."""
    print("\n=== ÉTAPE 2 : Nettoyage Bronze → Silver ===")
    df_silver = clean_dataframe(df_bronze)
    # La sauvegarde Parquet/CSV est déjà gérée dans clean_dataframe,
    # mais les chemins y sont en dur (bug). On re-sauvegarde ici proprement.
    _save(df_silver, SILVER_OUTPUT_PARQUET_PATH, SILVER_OUTPUT_CSV_PATH, "SILVER")
    return df_silver


# ---------------------------------------------------------------------------
# Étape 3 — Feature Engineering Gold (via cleaning.polish_dataframe)
# ---------------------------------------------------------------------------

def step_polish(df_silver: pd.DataFrame) -> pd.DataFrame:
    """
    Silver → Gold : enrichissement Sentinel 360.
    - Colonne `pilier`           (securite / economie / social / autre)
    - Colonne `pilier_label`     (libellé CAMEO racine)
    - Colonne `sentiment_proxy`  (positif / neutre / négatif)
    - Colonne `sentiment_score`  (AvgTone normalisé [-1, 1])
    - Colonne `goldstein_label`  (décideur-friendly)
    - Colonne `event_date`       (datetime propre depuis DATEADDED)
    - Colonne `period`           (YYYY-MM pour agrégation mensuelle)
    - Colonne `source_domain`    (domaine extrait de SOURCEURL)
    - Colonne `media_lang_group` (francophone / anglophone / autre)
    """
    print("\n=== ÉTAPE 3 : Feature Engineering Silver → Gold ===")

    df_gold = polish_dataframe(df_silver)

    # --- Piliers Sentinel 360
    df_gold["pilier"] = df_gold["EventRootCode"].astype(str).apply(_assign_pilier)
    df_gold["pilier_label"] = df_gold["EventRootCode"].astype(str).map(CAMEO_ROOT_LABELS).fillna("Autre")

    # --- Sentiment proxy (AvgTone)
    df_gold["sentiment_proxy"] = df_gold["AvgTone"].apply(
        lambda t: "positif" if t > 2 else ("négatif" if t < -2 else "neutre")
    )
    df_gold["sentiment_score"] = df_gold["AvgTone"].clip(-10, 10) / 10

    # --- Label décideur pour GoldsteinScale
    df_gold["goldstein_label"] = df_gold["GoldsteinScale"].apply(_goldstein_label)

    # --- Dates
    if "event_date" not in df_gold.columns:
        df_gold["event_date"] = pd.to_datetime(
            df_gold["DATEADDED"].astype(str), format="%Y%m%d%H%M%S", errors="coerce"
        )
    df_gold["period"] = df_gold["event_date"].dt.to_period("M").astype(str)

    # --- Domaine source
    def _extract_domain(url: str) -> str:
        try:
            from urllib.parse import urlparse
            return urlparse(str(url)).netloc.replace("www.", "")
        except Exception:
            return "inconnu"

    df_gold["source_domain"] = df_gold["SOURCEURL"].apply(_extract_domain)

    # --- Groupe linguistique de la source (heuristique TLD / domaine)
    FR_DOMAINS = (
        ".fr", ".bj", ".tg", ".sn", ".ci", ".ml", ".bf", ".ne",
        "rfi.fr", "lemonde.fr", "jeuneafrique.com", "24haubenin.com",
        "beninwebtv.com", "fraternite.bj",
    )
    EN_DOMAINS = (
        ".com", ".org", ".net", ".uk", ".us", ".au",
        "bbc.com", "reuters.com", "apnews.com", "aljazeera.com",
    )

    def _lang_group(domain: str) -> str:
        d = domain.lower()
        if any(d.endswith(s) or s in d for s in FR_DOMAINS):
            return "francophone"
        if any(d.endswith(s) or s in d for s in EN_DOMAINS):
            return "anglophone"
        return "autre"

    df_gold["media_lang_group"] = df_gold["source_domain"].apply(_lang_group)

    _save(df_gold, GOLD_OUTPUT_PARQUET_PATH, GOLD_OUTPUT_CSV_PATH, "GOLD")
    return df_gold


# ---------------------------------------------------------------------------
# Récapitulatif final
# ---------------------------------------------------------------------------

def _print_summary(df_gold: pd.DataFrame) -> None:
    print("\n=== RÉSUMÉ SENTINEL 360 ===")
    print(f"  Événements Gold     : {len(df_gold):,}")
    print(f"  Période             : {df_gold['event_date'].min().date()} → {df_gold['event_date'].max().date()}")
    print(f"  GoldsteinScale moy. : {df_gold['GoldsteinScale'].mean():.2f}")
    print(f"  AvgTone moyen       : {df_gold['AvgTone'].mean():.2f}")
    print()
    print("  Distribution piliers :")
    for pilier, count in df_gold["pilier"].value_counts().items():
        pct = count / len(df_gold) * 100
        print(f"    {pilier:<12} {count:>6,}  ({pct:.1f}%)")
    print()
    print("  Distribution sentiment :")
    for label, count in df_gold["sentiment_proxy"].value_counts().items():
        pct = count / len(df_gold) * 100
        print(f"    {label:<12} {count:>6,}  ({pct:.1f}%)")
    print()
    print("  Top 5 acteurs (Actor1Name) :")
    for actor, count in df_gold["Actor1Name"].value_counts().head(5).items():
        print(f"    {actor:<30} {count:>5,}")


# ---------------------------------------------------------------------------
# Orchestrateur principal
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline Bénin Sentinel 360")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estime le coût BigQuery sans extraire les données")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Saute l'extraction (utilise le Bronze existant)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════╗")
    print("║     Bénin Sentinel 360 — Pipeline v1     ║")
    print("╚══════════════════════════════════════════╝")

    client = bigquery.Client()

    # --- Bronze
    if args.skip_extract:
        print("\n=== [--skip-extract] Chargement Bronze existant ===")
        if not BRONZE_OUTPUT_PARQUET_PATH.exists():
            print(f"  ERREUR : {BRONZE_OUTPUT_PARQUET_PATH} introuvable. Lancez sans --skip-extract.")
            sys.exit(1)
        df_bronze = pd.read_parquet(BRONZE_OUTPUT_PARQUET_PATH)
        print(f"  {len(df_bronze):,} événements chargés depuis {BRONZE_OUTPUT_PARQUET_PATH}")
    else:
        df_bronze = step_extract(client, dry_run_only=args.dry_run)
        if df_bronze is None:
            return  # dry-run uniquement

    # --- Silver
    df_silver = step_clean(df_bronze)

    # --- Gold
    df_gold = step_polish(df_silver)

    # --- Résumé
    _print_summary(df_gold)

    print("\n  Pipeline terminé avec succès.")
    print(f"  Gold disponible : {GOLD_OUTPUT_PARQUET_PATH}")


if __name__ == "__main__":
    main()
