"""
Modèle ML pour le hackathon :
- Random Forest pour la classification du sentiment
- K-Means pour le clustering exploratoire

Ce script est devenu modulaire, reproductible et plus simple à maintenir.
"""

import argparse
import os
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OrdinalEncoder, StandardScaler)

warnings.filterwarnings("ignore")
plt.style.use("dark_background")
ACCENT = "#e040fb"

# Répertoires de sortie pour les fichiers générés
OUTPUT_DIR = Path("outputs")
FIG_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"

# Chemin par défaut du jeu de données
DATA_PATH = Path("data/gdelt_benin_clean.csv")

# Cible et colonnes utilisées pour l'entraînement
TARGET = "sentiment"
CATEGORICAL_FEATURES = ["event_category", "department", "Actor1Name"]
NUMERIC_FEATURES = ["GoldsteinScale", "NumArticles", "NumMentions",
                    "day_of_week", "day_of_month", "month_num", "quarter"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def ensure_output_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def make_demo_dataframe() -> pd.DataFrame:
    """Retourne un DataFrame de démonstration si le jeu de données réel est absent."""
    np.random.seed(42)
    n = 5000
    cats = ["Déclaration publique", "Appel à action", "Protestation",
            "Consulter / Négocier", "Aide humanitaire", "Coopération",
            "Violence", "Arrestation", "Accord diplomatique", "Coercition"]
    probs = [0.15, 0.12, 0.13, 0.15, 0.10, 0.10, 0.08, 0.07, 0.05, 0.05]
    depts = ["Cotonou", "Porto-Novo", "Parakou", "Abomey-Calavi",
             "Natitingou", "Lokossa", "Bohicon", "Kandi", "Ouidah", "Djougou"]
    actors = ["Gouvernement Bénin", "Patrice Talon", "CEDEAO", "ONU", "France",
              "Nigeria", "Syndicat", "Armée", "Société civile", "Union Africaine"]

    start = datetime(2025, 4, 27)
    dates = [start + timedelta(days=i % 365) for i in range(n)]
    tones = np.random.normal(-1.5, 3.8, n)

    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "event_category": np.random.choice(cats, n, p=probs),
        "AvgTone": tones,
        "GoldsteinScale": np.random.uniform(-10, 10, n),
        "NumArticles": np.random.randint(1, 150, n),
        "NumMentions": np.random.randint(1, 600, n),
        "department": np.random.choice(depts, n),
        "Actor1CountryCode": np.random.choice(depts, n),
        "Actor2CountryCode": np.random.choice(depts, n),
        "Actor1Name": np.random.choice(actors, n),
    })
    df[TARGET] = df["AvgTone"].apply(
        lambda t: "Positif" if t > 1 else ("Négatif" if t < -1 else "Neutre")
    )
    return df


def load_data(path: Path, use_demo: bool = True) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"], low_memory=False)
        print(f"✅ Données réelles chargées : {len(df):,} lignes")
        return df

    if use_demo:
        print("⚠️  Fichier non trouvé, utilisation d'un jeu de données de démonstration.")
        return make_demo_dataframe()

    raise FileNotFoundError(f"Le fichier {path} est introuvable.")


def enrich_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        if "DATEADDED" in df.columns:
            df["date"] = pd.to_datetime(df["DATEADDED"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
        else:
            raise ValueError("Aucune colonne de date n'est disponible.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["day_of_week"] = df["date"].dt.dayofweek.fillna(-1).astype(int)
    df["day_of_month"] = df["date"].dt.day.fillna(0).astype(int)
    df["month_num"] = df["date"].dt.month.fillna(0).astype(int)
    df["quarter"] = df["date"].dt.quarter.fillna(0).astype(int)
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "department" not in df.columns and "ActionGeo_FullName" in df.columns:
        df["department"] = df["ActionGeo_FullName"].astype(str)
    if "department" not in df.columns:
        df["department"] = "inconnu"
    return df


def build_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET not in df.columns:
        df[TARGET] = df["AvgTone"].apply(
            lambda t: "Positif" if t > 1 else ("Négatif" if t < -1 else "Neutre")
        )
    return df


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prépare X et y en appliquant les transformations de base."""
    # Normalisation des colonnes pour s'assurer que les noms utilisés existent
    df = normalize_columns(df.copy())
    # Création des variables temporelles à partir de la date
    df = enrich_datetime(df)
    # Construction du label sentiment si nécessaire
    df = build_sentiment(df)
    # Suppression des lignes incomplètes sur les features ou la cible
    df = df.dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Retourne le transformateur de colonnes pour l'encodage des catégoriques."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="passthrough",
    )


def build_classifier_pipeline() -> Pipeline:
    """Construit le pipeline de classification complet.

    Ce pipeline encode d'abord les colonnes catégorielles,
    puis entraîne un RandomForestClassifier.
    """
    preprocessor = build_preprocessor()
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_classifier(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Évalue le modèle sur le jeu de test et affiche les métriques principales."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy test : {accuracy*100:.1f}%")
    print("\nRapport de classification :\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    return {
        "accuracy": accuracy,
        "y_pred": y_pred,
    }


def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray, out_path: Path) -> None:
    labels = ["Négatif", "Neutre", "Positif"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax,
                xticklabels=labels, yticklabels=labels, linewidths=0.5)
    ax.set_xlabel("Prédiction", color="white")
    ax.set_ylabel("Réalité", color="white")
    ax.set_title("Matrice de confusion", color="white", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f0f13")
    plt.close(fig)
    print(f"   ✅ Sauvegardé : {out_path}")


def plot_feature_importance(model: Pipeline, out_path: Path) -> None:
    feature_names = model.named_steps["preprocessor"].get_feature_names_out(FEATURES)
    importances = model.named_steps["classifier"].feature_importances_
    importance_series = pd.Series(importances, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    importance_series.plot.barh(color=[ACCENT if i >= len(feature_names) - 3 else "#3d2b6b" for i in range(len(feature_names))], ax=ax)
    ax.set_title("Importance des variables", color="white")
    ax.set_xlabel("Importance (Gini)", color="white")
    for spine in ax.spines.values():
        spine.set_color("#333")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f0f13")
    plt.close(fig)
    print(f"   ✅ Sauvegardé : {out_path}")


def cross_validate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> tuple[float, float, list[float]]:
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return scores.mean(), scores.std(), scores.tolist()


def train_kmeans(X: pd.DataFrame, n_clusters: int) -> tuple[KMeans, StandardScaler, np.ndarray, ColumnTransformer]:
    """Prépare et entraîne un modèle K-Means sur les données prétraitées."""
    # Encodage des colonnes catégorielles avant la normalisation
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    # Normalisation nécessaire pour K-Means, qui dépend des distances
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_transformed)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(X_scaled)
    return km, scaler, X_scaled, preprocessor


def plot_elbow(inertias: list[float], k_values: range, out_path: Path) -> None:
    """Trace et enregistre la courbe d'inertie pour choisir le nombre de clusters."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(list(k_values), inertias, marker="o", color=ACCENT, linewidth=2.5, markersize=8)
    ax.set_xlabel("Nombre de clusters K", color="white")
    ax.set_ylabel("Inertie (plus bas = mieux)", color="white")
    ax.set_title("Méthode du coude — Choix du nombre de clusters", color="white", fontsize=12)
    ax.grid(alpha=0.2)
    for spine in ax.spines.values():
        spine.set_color("#333")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f0f13")
    plt.close(fig)
    print(f"   ✅ Sauvegardé : {out_path}")


def plot_clusters(X_scaled: np.ndarray, labels: np.ndarray, n_clusters: int, out_path: Path) -> None:
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(12, 8))
    palette = ["#e040fb", "#00e5ff", "#4caf7d", "#ff9800", "#ff5722", "#8bc34a"]
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=palette[cluster_id % len(palette)],
                   label=f"Cluster {cluster_id}",
                   alpha=0.5, s=20)
    ax.set_xlabel(f"Composante 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", color="white")
    ax.set_ylabel(f"Composante 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", color="white")
    ax.set_title("Clustering K-Means des événements", color="white", fontsize=13)
    ax.legend(facecolor="#1a1a2e", edgecolor="#333", fontsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f0f13")
    plt.close(fig)
    print(f"   ✅ Sauvegardé : {out_path}")


def save_artifacts(model: Pipeline, scaler: StandardScaler, kmeans: KMeans, kmeans_preprocessor: ColumnTransformer) -> None:
    with open(MODEL_DIR / "rf_pipeline.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODEL_DIR / "kmeans_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_DIR / "kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    with open(MODEL_DIR / "kmeans_preprocessor.pkl", "wb") as f:
        pickle.dump(kmeans_preprocessor, f)
    print("\n💾 Modèles sauvegardés :")
    print(f"   - {MODEL_DIR / 'rf_pipeline.pkl'}")
    print(f"   - {MODEL_DIR / 'kmeans_scaler.pkl'}")
    print(f"   - {MODEL_DIR / 'kmeans_model.pkl'}")
    print(f"   - {MODEL_DIR / 'kmeans_preprocessor.pkl'}")


def profile_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df["cluster"] = labels
    profile = df.groupby("cluster").agg(
        nb_events=("cluster", "count"),
        avg_tone=("AvgTone", "mean"),
        avg_articles=("NumArticles", "mean"),
        avg_goldstein=("GoldsteinScale", "mean"),
        top_event=("event_category", lambda x: x.mode().iloc[0] if len(x) else "-"),
        top_dept=("department", lambda x: x.mode().iloc[0] if len(x) else "-"),
    ).round(2)
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîne et évalue les modèles ML pour GDELT Bénin.")
    parser.add_argument("--data-path", default=DATA_PATH, help="Chemin vers le fichier CSV des données.")
    parser.add_argument("--use-demo", action="store_true", help="Utiliser un jeu de données de démonstration si le CSV est absent.")
    parser.add_argument("--clusters", type=int, default=4, help="Nombre de clusters K pour K-Means.")
    args = parser.parse_args()

    # Crée les dossiers de sortie si nécessaire
    ensure_output_directories()

    # Chargement des données et préparation du jeu X/y
    df = load_data(Path(args.data_path), use_demo=args.use_demo)
    X, y = prepare_dataset(df)

    print(f"\n🔧 Dataset préparé : {len(X):,} lignes, {len(FEATURES)} features")
    print("Répartition des classes :")
    print(y.value_counts(normalize=True).mul(100).round(1).to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_pipeline = build_classifier_pipeline()
    print("\n⏳ Entraînement du Random Forest...")
    rf_pipeline.fit(X_train, y_train)
    print("Entraînement terminé.")

    results = evaluate_classifier(rf_pipeline, X_test, y_test)
    plot_confusion_matrix(y_test, results["y_pred"], FIG_DIR / "ml_confusion_matrix.png")
    plot_feature_importance(rf_pipeline, FIG_DIR / "ml_feature_importance.png")

    cv_mean, cv_std, cv_scores = cross_validate_model(rf_pipeline, X, y, cv=5)
    print(f"\nValidation croisée : {cv_mean*100:.1f}% ± {cv_std*100:.1f}%")
    print(f"Scores : {[f'{s*100:.1f}%' for s in cv_scores]}")

    print("\n🔵 Entraînement du clustering K-Means...")
    # Appliquer le prétraitement et entraîner K-Means sur les données transformées
    kmeans, scaler, X_scaled, kmeans_preprocessor = train_kmeans(X, n_clusters=args.clusters)
    print(f"Clusters entraînés : K = {args.clusters}")

    # Calcul des inerties pour la méthode du coude
    inertias = []
    k_range = range(2, min(9, len(X)) if len(X) > 2 else 3)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    plot_elbow(inertias, k_range, FIG_DIR / "ml_kmeans_elbow.png")
    plot_clusters(X_scaled, kmeans.labels_, args.clusters, FIG_DIR / "ml_kmeans_clusters.png")

    # Affichage du profil de chaque cluster
    cluster_profile = profile_clusters(df, kmeans.labels_)
    print("\nProfil des clusters :")
    print(cluster_profile.to_string())

    # Sauvegarde des modèles et des transformateurs nécessaires pour l'inférence
    save_artifacts(rf_pipeline, scaler, kmeans, kmeans_preprocessor)

    print("\n✅ Résumé :")
    print(f"   Random Forest accuracy test : {results['accuracy']*100:.1f}%")
    print(f"   K-Means clustering : {args.clusters} clusters")
    print(f"   Figures sauvegardées dans {FIG_DIR}")
    print(f"   Modèles sauvegardés dans {MODEL_DIR}")


if __name__ == "__main__":
    main()
