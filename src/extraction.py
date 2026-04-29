import os
from google.cloud import bigquery
from pathlib import Path
import pandas as pd

### Configuration de l'authentification
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "hackaton-benin-insights-2026-f2afc452104d.json"
### Configuration de l'Output Brut 
# Données Parquet
BRONZE_OUTPUT_PARQUET_PATH = Path("data/raw/benin_events_bronze.parquet")
# Données CSV
BRONZE_OUTPUT_CSV_PATH = Path("data/raw/benin_events_bronze.csv")

def estimate_query_cost(client: bigquery.Client, query: str) -> float:
    """Retourne le volume scanné en GB sans exécuter la requête."""
    
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    dry_run = client.query(query, job_config=job_config)
    gb = dry_run.total_bytes_processed / 1e9
    return gb


def run_extraction(client: bigquery.Client, query: str) -> pd.DataFrame:
    """Récupération des données selon l'architecture en Medaille --- Bronze - Silver - Gold."""

    print("Exécution de la requête BigQuery...")
    df = client.query(query).to_dataframe()
    print(f"{len(df):,} événements extraits.")
    return df

def get_table_partition_info(client: bigquery.Client, table_id: str) -> dict:
    """
    Récupère les informations de partitionnement et de clustering d'une table BigQuery.

    Args:
        client: client BigQuery
        table_id: identifiant complet de la table (ex: "project.dataset.table")

    Returns:
        dict avec les clés:
            - 'partition_column' : nom de la colonne de partition (None si non partitionnée)
            - 'partition_type'   : type (DAY, MONTH, YEAR, etc.)
            - 'clustering_fields': liste des colonnes de clustering (vide si non clusterisé)
    """
    table = client.get_table(table_id)
    partitioning = table.time_partitioning
    clustering = table.clustering_fields

    info = {
        'partition_column': None,
        'partition_type': None,
        'clustering_fields': clustering if clustering else []
    }

    
    if partitioning:
        info['partition_column'] = partitioning.field
        info['partition_type'] = partitioning.type_
    return info