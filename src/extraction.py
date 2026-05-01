"""
extraction.py — Bénin Sentinel 360
Gère l'authentification BigQuery, l'estimation du coût et l'extraction
des données brutes (couche Bronze) depuis gdelt-bq.gdeltv2.events_partitioned.
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

# ---------------------------------------------------------------------------
# Authentification
# ---------------------------------------------------------------------------
# Priorité 1 : variable d'environnement déjà définie (CI, Cloud Run, etc.)
# Priorité 2 : fichier .env local (développement)
# Priorité 3 : fallback sur le fichier JSON du service account
load_dotenv()

_SA_FILE = "hackaton-benin-insights-2026-f2afc452104d.json"
if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") and Path(_SA_FILE).exists():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _SA_FILE

# ---------------------------------------------------------------------------
# Chemins de sortie Bronze
# ---------------------------------------------------------------------------
BRONZE_OUTPUT_PARQUET_PATH = Path("data/raw/benin_events_bronze.parquet")
BRONZE_OUTPUT_CSV_PATH     = Path("data/raw/benin_events_bronze.csv")


# ---------------------------------------------------------------------------
# Fonctions publiques
# ---------------------------------------------------------------------------

def estimate_query_cost(client: bigquery.Client, query: str) -> float:
    """
    Estime le volume de données scanné par la requête (en GB) sans l'exécuter.

    Args:
        client : client BigQuery authentifié
        query  : chaîne SQL à évaluer

    Returns:
        Volume scanné en GB (float)
    """
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    dry_run = client.query(query, job_config=job_config)
    gb = dry_run.total_bytes_processed / 1e9
    return gb


def run_extraction(client: bigquery.Client, query: str) -> pd.DataFrame:
    """
    Exécute la requête BigQuery et retourne le résultat sous forme de DataFrame.

    Utilise le Storage Read API (via to_dataframe) pour un transfert
    plus rapide sur les gros volumes.

    Args:
        client : client BigQuery authentifié
        query  : requête SQL à exécuter

    Returns:
        DataFrame pandas avec les données brutes (Bronze)
    """
    print("  Exécution de la requête BigQuery...")
    job = client.query(query)
    df = job.to_dataframe(
        create_bqstorage_client=True,   # Storage Read API — plus rapide
        progress_bar_type="tqdm",       # barre de progression dans le terminal
    )
    print(f"  {len(df):,} événements extraits.")
    return df


def get_table_partition_info(client: bigquery.Client, table_id: str) -> dict:
    """
    Récupère les informations de partitionnement et de clustering d'une table BigQuery.

    Args:
        client   : client BigQuery authentifié
        table_id : identifiant complet de la table (ex: "project.dataset.table")

    Returns:
        dict avec les clés :
            - 'partition_column'  : nom de la colonne de partition (None si absente)
            - 'partition_type'    : type DAY / MONTH / YEAR / HOUR (None si absente)
            - 'clustering_fields' : liste des colonnes de clustering ([] si absent)
            - 'num_rows'          : nombre de lignes déclaré dans les métadonnées
            - 'size_mb'           : taille de la table en MB
    """
    table = client.get_table(table_id)
    partitioning = table.time_partitioning
    clustering   = table.clustering_fields

    info = {
        "partition_column":  partitioning.field if partitioning else None,
        "partition_type":    partitioning.type_ if partitioning else None,
        "clustering_fields": clustering if clustering else [],
        "num_rows":          table.num_rows,
        "size_mb":           round((table.num_bytes or 0) / 1e6, 1),
    }
    return info
