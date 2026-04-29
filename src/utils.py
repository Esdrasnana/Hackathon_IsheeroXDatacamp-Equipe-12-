"""
Fonctions utilitaires partagées (Yawo, Esdras, Lucia, Roosvelt).
"""

import pandas as pd
from pathlib import Path

RAW_PATH      = Path("data/raw/benin_events_raw.parquet")
PROCESSED_PATH = Path("data/processed/benin_events_features.parquet")

# Mapping CAMEO root codes → libellés lisibles
CAMEO_ROOT_LABELS = {
    "01": "Déclarations verbales",
    "02": "Appels à l'action",
    "03": "Coopération",
    "04": "Consultation",
    "05": "Engagement diplomatique",
    "06": "Coopération matérielle",
    "07": "Aide humanitaire",
    "08": "Coopération judiciaire",
    "09": "Investigations",
    "10": "Demandes",
    "11": "Désapprobation",
    "12": "Rejet",
    "13": "Menaces",
    "14": "Protestations",
    "15": "Coercition",
    "16": "Agression",
    "17": "Violence",
    "18": "Assaut",
    "19": "Combats",
    "20": "Violence de masse",
}


def load_raw() -> pd.DataFrame:
    """Charge les données brutes depuis le Parquet."""
    return pd.read_parquet(RAW_PATH)


def load_processed() -> pd.DataFrame:
    """Charge les features engineered."""
    return pd.read_parquet(PROCESSED_PATH)


def goldstein_label(score: float) -> str:
    """Traduit un score Goldstein en label décideur."""
    if score >= 5:
        return "Stabilisant"
    elif score >= 1:
        return "Légèrement positif"
    elif score >= -1:
        return "Neutre"
    elif score >= -5:
        return "Déstabilisant"
    else:
        return "Très déstabilisant"


def get_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Agrégation mensuelle pour le dashboard."""
    df = df.copy()
    df["period"] = df["event_date"].dt.to_period("M").astype(str)
    return df.groupby("period").agg(
        goldstein_mean=("GoldsteinScale", "mean"),
        tone_mean=("AvgTone", "mean"),
        articles_total=("NumArticles", "sum"),
        mentions_total=("NumMentions", "sum"),
        event_count=("EventCode", "count"),
    ).reset_index()


def get_top_events(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top N événements par volume médiatique."""
    return (
        df.sort_values("NumArticles", ascending=False)
          .head(n)[["event_date", "EventRootCode", "Actor1Name",
                     "Actor2Name", "GoldsteinScale", "AvgTone",
                     "NumArticles", "ActionGeo_FullName", "SOURCEURL"]]
    )
