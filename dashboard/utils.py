"""
=============================================================================
SENTINEL 360 — Fonctions utilitaires
=============================================================================
Chargement, nettoyage, feature engineering avancé.
Inclut : Z-Score, indice de stabilité composite, macro-zones géographiques,
classification des sources.
=============================================================================
"""

import pandas as pd
import numpy as np
import os

# ── CONSTANTES ──────────────────────────────────────────────────────────

CAMEO_ROOT_LABELS = {
    "01": "Déclaration publique",
    "02": "Appel",
    "03": "Intention de coopérer",
    "04": "Consultation",
    "05": "Engagement diplomatique",
    "06": "Coopération matérielle",
    "07": "Aide",
    "08": "Concession",
    "09": "Investigation",
    "10": "Demande",
    "11": "Désapprobation",
    "12": "Rejet",
    "13": "Menace",
    "14": "Protestation",
    "15": "Force militaire",
    "16": "Réduction des relations",
    "17": "Coercition",
    "18": "Assaut",
    "19": "Combat",
    "20": "Force non conventionnelle",
}

QUADCLASS_LABELS = {
    1: "Coopération verbale",
    2: "Coopération matérielle",
    3: "Conflit verbal",
    4: "Conflit matériel",
}

ACTOR_TYPE_LABELS = {
    "GOV": "Gouvernement",
    "MIL": "Militaire",
    "REB": "Rebelle",
    "OPP": "Opposition",
    "PTY": "Parti politique",
    "AGR": "Agriculture",
    "BUS": "Business",
    "CRM": "Criminel",
    "CVL": "Société civile",
    "EDU": "Éducation",
    "ELI": "Élite",
    "ENV": "Environnement",
    "HLH": "Santé",
    "HRI": "Droits humains",
    "IGO": "Org. intergouvernementale",
    "JUD": "Judiciaire",
    "LAB": "Travail",
    "LEG": "Législatif",
    "MED": "Médias",
    "MNC": "Multinationale",
    "NGO": "ONG",
    "REF": "Réfugié",
    "REL": "Religion",
    "SPY": "Renseignement",
    "UAF": "Forces armées",
}

# Macro-zones géographiques du Bénin (par latitude)
def classify_zone(lat):
    """Classe une latitude dans une macro-zone du Bénin."""
    if pd.isna(lat):
        return "Non localisé"
    if lat >= 10.0:
        return "Nord . Zone frontaliere"
    elif lat >= 8.0:
        return "Centre . Borgou / Collines"
    elif lat >= 6.8:
        return "Sud . Plateau / Zou"
    else:
        return "Littoral . Cotonou / Porto-Novo"

ZONE_ORDER = [
    "Littoral . Cotonou / Porto-Novo",
    "Sud . Plateau / Zou",
    "Centre . Borgou / Collines",
    "Nord . Zone frontaliere",
]

# Note : la classification source_type (national / international)
# est deja presente dans le fichier Gold, creee par le Data Engineer.


# ── CHARGEMENT ──────────────────────────────────────────────────────────

def load_data():
    """Charge le fichier Gold et enrichit les données."""
    possible_paths = [
        "data/processed/benin_events_gold.parquet",
        "../data/processed/benin_events_gold.parquet",
        "benin_events_gold.parquet",
    ]

    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            break

    if df is None:
        raise FileNotFoundError(
            "Fichier benin_events_gold.parquet introuvable. "
            "Placez-le dans data/processed/"
        )

    # ── Dates ──
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    elif "SQLDATE" in df.columns:
        df["event_date"] = pd.to_datetime(df["SQLDATE"], errors="coerce")

    df["mois"] = df["event_date"].dt.to_period("M").astype(str)
    df["semaine"] = df["event_date"].dt.to_period("W").apply(lambda x: x.start_time)

    # ── Labels CAMEO ──
    if "event_root_label" not in df.columns:
        df["event_root_label"] = df["EventRootCode"].map(CAMEO_ROOT_LABELS).fillna("Autre")

    if "quad_class_label" not in df.columns:
        df["quad_class_label"] = df["QuadClass"].map(QUADCLASS_LABELS).fillna("Inconnu")

    if "actor1_type_label" not in df.columns:
        df["actor1_type_label"] = df["Actor1Type1Code"].map(ACTOR_TYPE_LABELS).fillna("Non spécifié")

    # ── Sentiment ──
    if "sentiment_proxy" not in df.columns:
        df["sentiment_proxy"] = pd.cut(
            df["AvgTone"],
            bins=[-100, -2, 2, 100],
            labels=["négatif", "neutre", "positif"]
        )

    # ── Macro-zones géographiques ──
    df["macro_zone"] = df["ActionGeo_Lat"].apply(classify_zone)

    # ── Indice de stabilité composite (0-100) ──
    df["stability_index"] = compute_stability_index(df)

    # ── Filtrer 2025 ──
    if "YEAR" in df.columns:
        df = df[df["YEAR"] == 2025].copy()

    return df


def compute_stability_index(df):
    """
    Calcule un indice de stabilité composite entre 0 (critique) et 100 (stable).
    Combine GoldsteinScale et AvgTone en gérant la divergence.
    
    Formule :
      1. Normaliser GoldsteinScale de [-10, +10] vers [0, 100]
      2. Normaliser AvgTone de [-15, +15] vers [0, 100] (clipé)
      3. Moyenne pondérée : 60% Goldstein + 40% Tone
      4. Bonus/malus de divergence : si les deux divergent fortement,
         on applique un malus (signal de tension cachée)
    """
    # Normaliser GoldsteinScale [-10, 10] → [0, 100]
    gs_norm = ((df["GoldsteinScale"].clip(-10, 10) + 10) / 20 * 100)

    # Normaliser AvgTone [-15, 15] → [0, 100]
    tone_norm = ((df["AvgTone"].clip(-15, 15) + 15) / 30 * 100)

    # Moyenne pondérée
    composite = 0.6 * gs_norm + 0.4 * tone_norm

    # Malus de divergence : si Goldstein positif mais tone très négatif (ou inverse)
    divergence = (gs_norm - tone_norm).abs()
    malus = (divergence > 40).astype(float) * 10  # -10 points si forte divergence

    result = (composite - malus).clip(0, 100)
    return result


def compute_weekly_zscore(df):
    """
    Calcule le Z-Score du volume hebdomadaire pour détecter les anomalies.
    Z > 2 = alerte orange, Z > 3 = alerte rouge.
    """
    weekly = df.groupby("semaine").agg(
        volume=("GLOBALEVENTID", "count"),
        tone_moyen=("AvgTone", "mean"),
        goldstein_moyen=("GoldsteinScale", "mean"),
        stability_moyen=("stability_index", "mean"),
        pct_conflit=("QuadClass", lambda x: (x.isin([3, 4])).mean() * 100),
    ).reset_index().sort_values("semaine")

    # Z-Score sur le volume
    mean_vol = weekly["volume"].mean()
    std_vol = weekly["volume"].std()
    weekly["z_score"] = (weekly["volume"] - mean_vol) / std_vol if std_vol > 0 else 0

    # Moyenne mobile 4 semaines
    weekly["volume_ma4"] = weekly["volume"].rolling(window=4, min_periods=1).mean()

    # Niveau d'alerte
    weekly["alerte"] = "Normal"
    weekly.loc[weekly["z_score"] > 2, "alerte"] = "Vigilance"
    weekly.loc[weekly["z_score"] > 3, "alerte"] = "Alerte"
    weekly.loc[weekly["z_score"] < -2, "alerte"] = "Creux inhabituel"

    return weekly


def compute_actor_interactions(df, min_count=5):
    """
    Calcule les interactions Actor1 → Actor2 avec le ton moyen.
    Retourne les paires ayant au moins min_count occurrences.
    """
    pairs = (
        df[df["Actor1Name"].notna() & df["Actor2Name"].notna()]
        .groupby(["Actor1Name", "Actor2Name"])
        .agg(
            interactions=("GLOBALEVENTID", "count"),
            tone_moyen=("AvgTone", "mean"),
            goldstein_moyen=("GoldsteinScale", "mean"),
            pct_conflit=("QuadClass", lambda x: (x.isin([3, 4])).mean() * 100),
        )
        .reset_index()
        .sort_values("interactions", ascending=False)
    )
    pairs = pairs[pairs["interactions"] >= min_count]
    return pairs


def compute_actor_type_crosstab(df):
    """
    Croise les types d'acteurs (Actor1Type vs Actor2Type) avec le ton moyen.
    Révèle les dynamiques de pouvoir.
    """
    df_typed = df[
        df["Actor1Type1Code"].notna()
        & df["Actor2Type1Code"].notna()
        & (df["Actor1Type1Code"] != "")
        & (df["Actor2Type1Code"] != "")
    ].copy()

    if len(df_typed) == 0:
        return pd.DataFrame()

    df_typed["a1_label"] = df_typed["Actor1Type1Code"].map(ACTOR_TYPE_LABELS).fillna(df_typed["Actor1Type1Code"])
    df_typed["a2_label"] = df_typed["Actor2Type1Code"].map(ACTOR_TYPE_LABELS).fillna(df_typed["Actor2Type1Code"])

    cross = (
        df_typed.groupby(["a1_label", "a2_label"])
        .agg(
            count=("GLOBALEVENTID", "count"),
            tone_moyen=("AvgTone", "mean"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )
    return cross


def compute_zone_stats(df):
    """Statistiques par macro-zone géographique."""
    zone_stats = (
        df[df["macro_zone"] != "Non localisé"]
        .groupby("macro_zone")
        .agg(
            nb_events=("GLOBALEVENTID", "count"),
            tone_moyen=("AvgTone", "mean"),
            goldstein_moyen=("GoldsteinScale", "mean"),
            stability_moyen=("stability_index", "mean"),
            pct_conflit=("QuadClass", lambda x: (x.isin([3, 4])).mean() * 100),
            pct_negatif=("sentiment_proxy", lambda x: (x.isin(["négatif", "negatif"])).mean() * 100),
        )
        .reset_index()
    )
    return zone_stats


def compute_source_bias(df):
    """Compare le ton entre sources nationales et internationales."""
    if "source_type" not in df.columns:
        return pd.DataFrame()

    bias = (
        df.groupby(["source_type", "pilier"])
        .agg(
            nb_events=("GLOBALEVENTID", "count"),
            tone_moyen=("AvgTone", "mean"),
        )
        .reset_index()
    )
    return bias


def compute_alert_feed(df, top_n=10):
    """
    Construit le fil d'alerte : les semaines les plus anormales
    avec le contexte (type d'evenement dominant, acteur principal, lieu, ton).
    """
    weekly = compute_weekly_zscore(df)
    
    # Garder les semaines anormales + les trier par z_score
    alerts = weekly[weekly["z_score"].abs() > 1.5].sort_values("z_score", ascending=False).head(top_n).copy()
    
    if len(alerts) == 0:
        return pd.DataFrame()
    
    # Pour chaque semaine anormale, trouver le contexte
    context_rows = []
    for _, week_row in alerts.iterrows():
        sem = week_row["semaine"]
        mask = df["semaine"] == sem
        week_df = df[mask]
        
        if len(week_df) == 0:
            continue
        
        # Type d'evenement dominant
        top_event = "N/A"
        if "event_root_label" in week_df.columns:
            top_event = week_df["event_root_label"].value_counts().index[0] if len(week_df["event_root_label"].dropna()) > 0 else "N/A"
        elif "EventRootCode" in week_df.columns:
            code = week_df["EventRootCode"].value_counts().index[0] if len(week_df["EventRootCode"].dropna()) > 0 else ""
            top_event = CAMEO_ROOT_LABELS.get(code, code)
        
        # Acteur principal
        top_actor = week_df["Actor1Name"].value_counts().index[0] if len(week_df["Actor1Name"].dropna()) > 0 else "N/A"
        
        # Lieu principal
        top_lieu = week_df["ActionGeo_FullName"].value_counts().index[0] if len(week_df["ActionGeo_FullName"].dropna()) > 0 else "N/A"
        
        # Pilier dominant
        top_pilier = week_df["pilier"].value_counts().index[0] if "pilier" in week_df.columns and len(week_df["pilier"].dropna()) > 0 else "N/A"
        
        # QuadClass dominant
        pct_conflit = (week_df["QuadClass"].isin([3, 4])).mean() * 100 if "QuadClass" in week_df.columns else 0
        
        context_rows.append({
            "semaine": sem,
            "volume": int(week_row["volume"]),
            "z_score": round(week_row["z_score"], 1),
            "tone_moyen": round(week_row["tone_moyen"], 2),
            "type_dominant": top_event,
            "acteur_principal": top_actor,
            "lieu_principal": top_lieu,
            "pilier": top_pilier,
            "pct_conflit": round(pct_conflit, 0),
            "alerte": week_row["alerte"],
        })
    
    return pd.DataFrame(context_rows)


# ── FILTRAGE ────────────────────────────────────────────────────────────

def get_date_range(df):
    return df["event_date"].min().date(), df["event_date"].max().date()


def filter_data(df, date_range, piliers=None, sentiments=None):
    mask = (
        (df["event_date"].dt.date >= date_range[0])
        & (df["event_date"].dt.date <= date_range[1])
    )
    if piliers and len(piliers) > 0:
        mask &= df["pilier"].isin(piliers)
    if sentiments and len(sentiments) > 0:
        mask &= df["sentiment_proxy"].isin(sentiments)
    return df[mask].copy()
