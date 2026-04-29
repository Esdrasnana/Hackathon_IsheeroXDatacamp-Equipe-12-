
from numpy.compat import Path
import pandas as pd

### Configuration de l'Output Silver 
# Données Parquet
SILVER_OUTPUT_PARQUET_PATH = Path("data/processed/benin_events_silver.parquet")
# Données CSV
SILVER_OUTPUT_CSV_PATH = Path("data/processed/benin_events_silver.csv")

### Configuration de l'Output Gold 
# Données Parquet
GOLD_OUTPUT_PARQUET_PATH = Path("data/processed/benin_events_gold.parquet")
# Données CSV
GOLD_OUTPUT_CSV_PATH = Path("data/processed/benin_events_gold.csv")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    """Transformation des données selon l'architecture en Medaille --- Bronze to Silver : types, doublons, outliers."""

    # --- Récupération des données brutes [Bronze]
    benin_events_silver_df = df;
    benin_events_silver_df_count = benin_events_silver_df.count()
    print(f"Nombre d'événements avant traitement : {benin_events_silver_df_count}")

    # --- Conversion des types des colonnes essentielles
    benin_events_silver_df["DATEADDED"] = pd.to_datetime(benin_events_silver_df["DATEADDED"])
    benin_events_silver_df["GoldsteinScale"] = benin_events_silver_df["GoldsteinScale"].clip(-10, 10)
    benin_events_silver_df["AvgTone"] = benin_events_silver_df["AvgTone"].clip(-30, 30)
    benin_events_silver_df["ActionGeo_Lat"] = pd.to_numeric(benin_events_silver_df["ActionGeo_Lat"], errors="coerce")
    benin_events_silver_df["ActionGeo_Long"] = pd.to_numeric(benin_events_silver_df["ActionGeo_Long"], errors="coerce")

    # --- Suppression des lignes avec des valeurs manquantes critiques
    
    ### Colonne essentielle : Actor1Name
    benin_events_missing_actor1 = benin_events_silver_df['Actor1Name'].isna().sum()
    print(f"Nombre d'événements sans Actor1Name : {benin_events_missing_actor1}")
    # Supprimer ces lignes
    benin_events_silver_df = benin_events_silver_df.dropna(subset=['Actor1Name'])
    
    ### Colonne essentielle : EventCode
    benin_events_codes = pd.to_numeric(benin_events_silver_df['EventCode'], errors='coerce')
    benin_events_invalid_codes = ((benin_events_codes < 0) | (benin_events_codes > 2000)).sum()
    print(f"Nombre d'événements avec des codes d'événement invalides : {benin_events_invalid_codes}")
    # Supprimer ces lignes
    benin_events_silver_df = benin_events_silver_df.dropna(subset=['EventCode'])

    ### Colonne essentielle : DATEADDED
    benin_events_missing_date = benin_events_silver_df['DATEADDED'].isna().sum()
    print(f"Nombre d'événements sans date : {benin_events_missing_date}")
    
    # --- Suppression des doublons basée sur la colonne GlobalEventID
    benin_events_silver_df = benin_events_silver_df.drop_duplicates(["GLOBALEVENTID"])
    print(f"Nombre d'événements après suppression des doublons : {len(benin_events_silver_df)}")

    # --- Standardisation et nettoyage de la colonne ActorName1
    benin_events_silver_df['Actor1Name'] = benin_events_silver_df['Actor1Name'].str.strip().str.upper()

    # --- Enregistrement des données nettoyées [Silver] au format Parquet 
    benin_events_silver_df.to_parquet(f"SILVER_OUTPUT_PARQUET_PATH", index=False)

    # --- Enregistrement des données nettoyées [Silver] au format CSV
    benin_events_silver_df.to_csv(f"SILVER_OUTPUT_CSV_PATH", index=False)
        
    return benin_events_silver_df


def polish_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Transformation des données selon l'architecture en Medaille --- Silver to Gold : enrichissement, agrégation, features engineering."""
    
    # --- Récupération des données nettoyées [Silver]
    benin_events_gold_df = df;
    benin_events_gold_df_count = benin_events_gold_df.count()
    print(f"Nombre d'événements avant traitement : {benin_events_gold_df_count}")

    return benin_events_gold_df
    