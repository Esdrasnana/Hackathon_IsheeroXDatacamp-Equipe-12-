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

def compute_weekly_zscore(df, window=8):
    """
    Detection par percentiles + ton.
    Une semaine est signalee si :
    - Son volume depasse le 75e percentile (vigilance) ou le 90e (alerte)
    - OU son ton moyen est inferieur a -2.5 (vigilance) ou -3.5 (alerte)
    - OU son taux de conflit depasse 40% (vigilance) ou 55% (alerte)
    """
    weekly = df.groupby("semaine").agg(
        volume=("GLOBALEVENTID", "count"),
        tone_moyen=("AvgTone", "mean"),
        goldstein_moyen=("GoldsteinScale", "mean"),
        stability_moyen=("stability_index", "mean"),
        pct_conflit=("QuadClass", lambda x: (x.isin([3, 4])).mean() * 100),
    ).reset_index().sort_values("semaine")

    # Moyenne mobile pour la tendance visuelle
    weekly["volume_ma4"] = weekly["volume"].rolling(window=4, min_periods=1).mean()

    # Percentiles du volume
    p75 = weekly["volume"].quantile(0.75)
    p90 = weekly["volume"].quantile(0.90)

    # Score composite pour le tri (garde le nom z_score pour compatibilite)
    weekly["z_score"] = (
        (weekly["volume"] / weekly["volume"].median())
        + (weekly["tone_moyen"].clip(upper=0).abs() / 2)
        + (weekly["pct_conflit"] / 30)
    )

    # Raison principale
    weekly["raison"] = "Normal"

    # Niveau d'alerte : on check chaque critere
    weekly["alerte"] = "Normal"

    for idx, row in weekly.iterrows():
        reasons = []

        # Critere volume
        if row["volume"] >= p90:
            weekly.loc[idx, "alerte"] = "Alerte"
            reasons.append("Volume tres eleve")
        elif row["volume"] >= p75:
            if weekly.loc[idx, "alerte"] != "Alerte":
                weekly.loc[idx, "alerte"] = "Vigilance"
            reasons.append("Volume eleve")

        # Critere ton
        if row["tone_moyen"] < -3.5:
            weekly.loc[idx, "alerte"] = "Alerte"
            reasons.append("Ton tres negatif")
        elif row["tone_moyen"] < -2.5:
            if weekly.loc[idx, "alerte"] != "Alerte":
                weekly.loc[idx, "alerte"] = "Vigilance"
            reasons.append("Ton negatif")

        # Critere conflit
        if row["pct_conflit"] > 55:
            weekly.loc[idx, "alerte"] = "Alerte"
            reasons.append("Conflit eleve")
        elif row["pct_conflit"] > 40:
            if weekly.loc[idx, "alerte"] != "Alerte":
                weekly.loc[idx, "alerte"] = "Vigilance"
            reasons.append("Conflit notable")

        if reasons:
            weekly.loc[idx, "raison"] = " + ".join(reasons)

    return weekly

def analyze_peak_context(df, semaine):
    """
    Pour une semaine donnee, extrait le contexte detaille
    et genere automatiquement un resume narratif.
    """
    week_df = df[df["semaine"] == semaine].copy()
    if len(week_df) == 0:
        return None

    result = {
        "nb_events": len(week_df),
        "tone_moyen": round(week_df["AvgTone"].mean(), 2),
        "goldstein_moyen": round(week_df["GoldsteinScale"].mean(), 2),
    }

    # Top 3 types d'evenements
    if "event_root_label" in week_df.columns:
        top_types = week_df["event_root_label"].value_counts().head(3)
        result["top_types"] = [(t, int(c), round(c/len(week_df)*100)) for t, c in top_types.items()]
    elif "EventRootCode" in week_df.columns:
        code = week_df["EventRootCode"].value_counts().head(3)
        result["top_types"] = [(CAMEO_ROOT_LABELS.get(t, t), int(c), round(c/len(week_df)*100)) for t, c in code.items()]

    # Top 3 acteurs
    if "Actor1Name" in week_df.columns:
        top_actors = week_df["Actor1Name"].dropna().value_counts().head(3)
        result["top_actors"] = [(a, int(c)) for a, c in top_actors.items()]

    # Top acteur 2 (cible)
    if "Actor2Name" in week_df.columns:
        top_a2 = week_df["Actor2Name"].dropna().value_counts().head(3)
        result["top_actors_cibles"] = [(a, int(c)) for a, c in top_a2.items()]

    # Top 3 sources
    if "source_domain" in week_df.columns:
        top_sources = week_df["source_domain"].dropna().value_counts().head(5)
        result["top_sources"] = [(s, int(c)) for s, c in top_sources.items()]

    # Repartition piliers
    if "pilier" in week_df.columns:
        pilier_dist = week_df["pilier"].value_counts(normalize=True).mul(100).round(0)
        result["piliers"] = {p: int(v) for p, v in pilier_dist.items()}

    # Repartition conflit/cooperation
    if "QuadClass" in week_df.columns:
        pct_conflit = (week_df["QuadClass"].isin([3, 4])).mean() * 100
        result["pct_conflit"] = round(pct_conflit)

    # Lieu principal
    if "ActionGeo_FullName" in week_df.columns:
        top_lieux = week_df["ActionGeo_FullName"].dropna().value_counts().head(3)
        result["top_lieux"] = [(l, int(c)) for l, c in top_lieux.items()]
        if len(top_lieux) > 0:
            result["lieu"] = top_lieux.index[0]

    # Repartition source nationale vs internationale
    if "source_type" in week_df.columns:
        source_dist = week_df["source_type"].value_counts()
        result["source_type_dist"] = {s: int(c) for s, c in source_dist.items()}

    # Mots cles extraits des URLs
    if "SOURCEURL" in week_df.columns:
        urls = week_df["SOURCEURL"].dropna().tolist()
        keywords = extract_keywords_from_urls(urls)
        if keywords:
            result["url_keywords"] = keywords

    # Generer le resume narratif automatique
    result["narrative"] = generate_narrative(result)

    return result


def extract_keywords_from_urls(urls, top_n=10):
    """
    Extrait les mots cles les plus frequents des chemins d'URL.
    Les URLs contiennent souvent le titre de l'article dans le chemin.
    Exemple : https://aljazeera.com/news/benin-soldiers-killed-border-attack
    -> ["benin", "soldiers", "killed", "border", "attack"]
    """
    import re
    from collections import Counter

    # Mots a ignorer (stop words pour les URLs)
    stop_words = {
        "www", "com", "org", "net", "html", "htm", "php", "asp", "index",
        "news", "article", "story", "post", "blog", "page", "the", "and",
        "for", "with", "from", "that", "this", "have", "are", "was", "were",
        "been", "being", "http", "https", "2025", "2024", "2026", "000",
        "des", "les", "une", "dans", "pour", "par", "sur", "qui", "que",
        "est", "son", "aux", "avec", "plus", "pas", "tout", "mais",
    }

    word_counter = Counter()
    for url in urls[:200]:  # Limiter pour la performance
        try:
            # Extraire le chemin de l'URL
            path = url.split("//")[-1].split("?")[0]
            # Decouper en mots
            words = re.findall(r'[a-zA-Zéèêëàâùûôîïç]{4,}', path.lower())
            # Filtrer les stop words et les domaines
            words = [w for w in words if w not in stop_words and len(w) < 20]
            word_counter.update(words)
        except Exception:
            continue

    # Retourner les N mots les plus frequents
    return word_counter.most_common(top_n)


def generate_narrative(context):
    """
    Genere automatiquement un resume narratif d'une semaine
    a partir des donnees extraites.
    """
    parts = []

    nb = context.get("nb_events", 0)
    tone = context.get("tone_moyen", 0)

    # Phrase d'ouverture
    if tone < -3:
        parts.append(f"Semaine marquee par une forte tension mediatique ({nb} evenements, ton moyen de {tone}).")
    elif tone < -1.5:
        parts.append(f"Semaine a dominante negative ({nb} evenements, ton moyen de {tone}).")
    else:
        parts.append(f"Semaine d'activite soutenue ({nb} evenements, ton moyen de {tone}).")

    # Types d'evenements
    if "top_types" in context and len(context["top_types"]) > 0:
        type1 = context["top_types"][0]
        parts.append(f"Le type d'evenement dominant est « {type1[0]} » ({type1[2]}% des evenements).")
        if len(context["top_types"]) > 1:
            type2 = context["top_types"][1]
            parts.append(f"Suivi de « {type2[0]} » ({type2[2]}%).")

    # Acteurs
    if "top_actors" in context and len(context["top_actors"]) > 0:
        actors = [a[0] for a in context["top_actors"][:3]]
        parts.append("Les acteurs principaux sont " + ", ".join(actors) + ".")

    # Lieu
    if "top_lieux" in context and len(context["top_lieux"]) > 0:
        lieu1 = context["top_lieux"][0]
        if len(context["top_lieux"]) > 1:
            lieu2 = context["top_lieux"][1]
            parts.append(f"Les evenements se concentrent autour de {lieu1[0]} ({lieu1[1]} evt) et {lieu2[0]} ({lieu2[1]} evt).")
        else:
            parts.append(f"Les evenements se concentrent autour de {lieu1[0]}.")

    # Mots cles des URLs
    if "url_keywords" in context and len(context["url_keywords"]) > 0:
        top_kw = [kw[0] for kw in context["url_keywords"][:5]]
        parts.append("Mots cles des articles : " + ", ".join(top_kw) + ".")

    # Sources
    if "top_sources" in context and len(context["top_sources"]) > 0:
        src = [s[0] for s in context["top_sources"][:3]]
        parts.append("Principaux medias : " + ", ".join(src) + ".")

    # Conflit
    pct = context.get("pct_conflit", 0)
    if pct > 50:
        parts.append(f"Semaine a forte dominante conflictuelle ({pct}% des evenements).")
    elif pct > 35:
        parts.append(f"Proportion notable d'evenements conflictuels ({pct}%).")

    return " ".join(parts)

def extract_headlines_from_urls(df, semaine, top_n=5):
    """
    Extrait des titres approximatifs a partir des URLs des articles.
    Le dernier segment du chemin contient souvent le titre slugifie.
    """
    import re

    week_df = df[df["semaine"] == semaine]
    if "SOURCEURL" not in week_df.columns:
        return []

    # Recuperer les URLs avec leur ton
    url_data = week_df[["SOURCEURL", "AvgTone"]].dropna(subset=["SOURCEURL"]).drop_duplicates(subset=["SOURCEURL"])

    headlines = []
    for _, row in url_data.head(100).iterrows():
        url = row["SOURCEURL"]
        tone = round(row["AvgTone"], 1)

        try:
            path = url.split("//")[-1].split("?")[0]
            slug = path.split("/")[-1]
            slug = re.sub(r'\.(html|htm|php|asp|aspx)$', '', slug)
            slug = re.sub(r'[-_]', ' ', slug)
            slug = re.sub(r'[0-9]{8,}', '', slug)

            if len(slug) > 20 and len(slug.split()) >= 4:
                # Filtrer les poubelles
                if re.search(r'[0-9a-f]{8}[\s\-][0-9a-f]{4}', slug):
                    continue
                digits = sum(1 for c in slug if c.isdigit())
                if digits > len(slug) * 0.4:
                    continue
                if re.search(r'article\s+[0-9a-f]', slug.lower()):
                    continue

                slug = slug.strip().capitalize()
                source = url.split("//")[-1].split("/")[0].replace("www.", "")
                headlines.append({
                    "titre": slug,
                    "source": source,
                    "url": url,
                    "tone": tone,
                })
        except Exception:
            continue

    seen = set()
    unique = []
    for h in headlines:
        if h["titre"] not in seen:
            seen.add(h["titre"])
            unique.append(h)
    return unique[:top_n]

def fetch_headlines_gdelt(date_start, date_end, max_results=10):
    """
    Interroge l'API DOC de GDELT pour recuperer les titres reels
    des articles sur le Benin pour une periode donnee.
    """
    import requests

    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": "Benin OR Bénin",
        "mode": "artlist",
        "maxrecords": max_results,
        "format": "json",
        "startdatetime": date_start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": date_end.strftime("%Y%m%d%H%M%S"),
        "sort": "hybridrel",
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if "articles" not in data:
            return []

        articles = []
        for art in data["articles"][:max_results]:
            articles.append({
                "titre": art.get("title", ""),
                "source": art.get("domain", ""),
                "date": art.get("seendate", ""),
                "url": art.get("url", ""),
                "langue": art.get("language", ""),
                "tone": art.get("tone", 0),
            })
        return articles

    except Exception:
        return []

def get_peak_headlines(df, semaine, max_results=5):
    """
    Recupere les titres, filtre par langue et pertinence,
    regroupe par evenement, et genere un titre synthetique.
    """
    from datetime import timedelta
    import re
    from collections import Counter

    date_start = semaine
    date_end = semaine + timedelta(days=7)

    # Mots cles de pertinence Benin
    benin_keywords = {
        "benin", "bénin", "cotonou", "porto-novo", "talon", "parakou",
        "alibori", "natitingou", "bohicon", "abomey", "djougou", "kandi",
        "boko", "homeky", "cedeao", "ecowas", "beninese", "beninois",
        "parc", "coup", "jnim", "kidjo", "w national",
    }

    # Langues acceptees (detectees par domaine ou mots)
    french_indicators = {"fr", "bj", "rfi", "france24", "lefigaro", "lemonde", "jeune", "afrique"}
    english_indicators = {"com", "uk", "ng", "reuters", "aljazeera", "bbc", "guardian", "africa"}

    # Essayer l'API GDELT d'abord
    articles_raw = fetch_headlines_gdelt(date_start, date_end, max_results=30)

    # Si pas de resultats API, extraire des URLs
    if not articles_raw:
        url_headlines = extract_headlines_from_urls(df, semaine, top_n=30)
        articles_raw = [{"titre": h["titre"], "source": h["source"], "url": h["url"], "tone": 0, "langue": ""} for h in url_headlines]

    if not articles_raw:
        return {"events": [], "source": "aucune"}

    # ── Etape 1 : Filtrer par langue ──
    filtered = []
    for art in articles_raw:
        titre_lower = art.get("titre", "").lower()
        source_lower = art.get("source", "").lower()
        langue = art.get("langue", "").lower()

        # Detecter la langue
        is_french = langue.startswith("fr") or any(ind in source_lower for ind in french_indicators)
        is_english = langue.startswith("en") or any(ind in source_lower for ind in english_indicators)

        if is_french or is_english:
            art["langue_detectee"] = "francais" if is_french else "anglais"
            filtered.append(art)

    # Si trop peu apres filtrage langue, garder tout
    if len(filtered) < 3:
        filtered = articles_raw
        for art in filtered:
            art["langue_detectee"] = "autre"

    # ── Etape 2 : Scorer la pertinence Benin ──
    for art in filtered:
        titre_lower = art.get("titre", "").lower()
        source_lower = art.get("source", "").lower()
        url_lower = art.get("url", "").lower()

        texte = titre_lower + " " + url_lower
        score = sum(1 for kw in benin_keywords if kw in texte)
        art["pertinence"] = score

    # Garder seulement les articles pertinents (score >= 1)
    pertinents = [a for a in filtered if a["pertinence"] >= 1]
    if len(pertinents) < 2:
        pertinents = sorted(filtered, key=lambda x: x["pertinence"], reverse=True)[:10]

    # ── Etape 3 : Regrouper par evenement ──
    events = cluster_headlines(pertinents)

    return {"events": events[:5], "source": "API GDELT" if articles_raw else "URLs du dataset"}


def cluster_headlines(articles):
    """
    Regroupe les articles qui parlent du meme evenement
    en utilisant le chevauchement de mots cles.
    """
    import re

    if not articles:
        return []

    # Extraire les mots significatifs de chaque titre
    def get_words(text):
        stop = {"the","and","for","with","from","that","this","have","are","was",
                "des","les","une","dans","pour","par","sur","qui","que","est",
                "son","aux","avec","plus","pas","tout","mais","has","been",
                "after","says","will","its","over","about","their","into","also",
                "not","but","can","had","may","would","could","should","than",
                "new","year","day","two","first","last","more","other","many"}
        words = set(re.findall(r'[a-zA-Zéèêëàâùûôîïç]{3,}', text.lower()))
        return words - stop

    # Calculer les mots de chaque article
    for art in articles:
        art["words"] = get_words(art.get("titre", ""))

    # Regrouper par similarite
    groups = []
    used = set()

    for i, art in enumerate(articles):
        if i in used:
            continue

        group = [art]
        used.add(i)

        for j, other in enumerate(articles):
            if j in used:
                continue
            # Calculer le chevauchement
            overlap = len(art["words"] & other["words"])
            total = min(len(art["words"]), len(other["words"]))
            if total > 0 and overlap / total >= 0.3:
                group.append(other)
                used.add(j)

        groups.append(group)

    # Pour chaque groupe, generer un titre synthetique
    events = []
    for group in groups:
        if not group:
            continue

        # Le titre le plus pertinent du groupe (score le plus haut)
        best = max(group, key=lambda x: x.get("pertinence", 0))
        titre_principal = best.get("titre", "")

        # Collecter toutes les sources
        sources = []
        for art in group:
            sources.append({
                "source": art.get("source", ""),
                "url": art.get("url", ""),
                "langue": art.get("langue_detectee", ""),
                "tone": art.get("tone", 0),
            })

        # Mots cles communs au groupe
        all_words = set()
        for art in group:
            all_words.update(art.get("words", set()))
        common_words = set()
        for art in group:
            common_words = common_words | art.get("words", set()) if not common_words else common_words & art.get("words", set())

        events.append({
            "titre": titre_principal,
            "nb_articles": len(group),
            "sources": sources,
            "tone_moyen": round(sum(a.get("tone", 0) for a in group) / len(group), 2) if group else 0,
            "mots_cles": list(common_words)[:5],
        })

    # Trier par nombre d'articles (les plus couverts en premier)
    events.sort(key=lambda x: x["nb_articles"], reverse=True)
    return events

THEME_KEYWORDS = {
    "Crise politique": {
        "coup", "putsch", "overthrow", "renversement", "impeach",
        "destitution", "constitution", "mandat", "mandate", "senat",
        "senate", "parlement", "parliament", "assemblee", "election",
        "vote", "opposition", "democratie", "democracy", "politique",
        "political", "gouvernement", "government", "president", "talon",
        "boko", "homeky", "pouvoir", "power", "reforme", "reform",
    },
    "Securite et defense": {
        "attack", "attaque", "killed", "tues", "morts", "dead", "soldiers",
        "soldats", "militaire", "military", "armee", "army", "terroris",
        "jnim", "djihadis", "jihadis", "boko haram", "frontiere", "border",
        "sahel", "securite", "security", "violence", "conflit", "conflict",
        "combat", "guerre", "war", "arme", "weapon", "embuscade", "ambush",
        "parc", "park", "alibori", "natitingou", "explosion", "bomb",
    },
    "Diplomatie et relations internationales": {
        "diplomati", "cedeao", "ecowas", "union africaine", "african union",
        "onu", "united nations", "france", "nigeria", "chine", "china",
        "ambassad", "cooperation", "traite", "treaty", "accord", "summit",
        "sommet", "sanction", "aide", "partenaire", "partner",
    },
    "Economie et developpement": {
        "economi", "economic", "investis", "invest", "commerce", "trade",
        "port", "cotonou", "infrastructure", "developpement", "development",
        "croissance", "growth", "emploi", "employment", "banque", "bank",
        "finance", "budget", "dette", "debt", "pib", "gdp", "entreprise",
    },
    "Societe et culture": {
        "kidjo", "culture", "education", "sante", "health", "sport",
        "artiste", "musique", "music", "festival", "hollywood", "prix",
        "award", "humanitaire", "humanitarian", "droits", "rights",
        "femme", "women", "jeune", "youth", "social",
    },
    "Justice et gouvernance": {
        "justice", "tribunal", "court", "proces", "trial", "condamn",
        "sentenced", "prison", "arret", "arrest", "police", "corruption",
        "fraude", "fraud", "loi", "law", "juge", "judge", "avocat",
    },
}
 
 
def detect_theme(keywords, articles, context):
    """
    Detecte le theme dominant a partir des mots cles des URLs,
    des titres des articles et du contexte GDELT.
    """
    # Collecter tous les mots
    all_words = set()
 
    # Mots cles des URLs
    if keywords:
        for kw, count in keywords:
            all_words.add(kw.lower())
 
    # Mots des titres d'articles
    if articles:
        import re
        for art in articles:
            titre = art.get("titre", "").lower()
            words = re.findall(r'[a-zA-Zéèêëàâùûôîïç]{3,}', titre)
            all_words.update(words)
 
    # Mots du type d'evenement GDELT
    if "top_types" in context:
        for type_name, _, _ in context["top_types"]:
            all_words.update(type_name.lower().split())
 
    # Scorer chaque theme
    scores = {}
    for theme, theme_kws in THEME_KEYWORDS.items():
        score = 0
        matched = []
        for word in all_words:
            for kw in theme_kws:
                if kw in word or word in kw:
                    score += 1
                    matched.append(word)
                    break
        scores[theme] = {"score": score, "matched": matched}
 
    # Theme dominant = score le plus eleve
    if not scores:
        return {"theme": "Activite mediatique", "matched": [], "all_scores": {}}
 
    best = max(scores, key=lambda x: scores[x]["score"])
 
    # Si le meilleur score est 0, theme generique
    if scores[best]["score"] == 0:
        return {"theme": "Activite mediatique", "matched": [], "all_scores": scores}
 
    # Themes secondaires (score > 0 et > 50% du meilleur)
    secondary = [t for t, s in scores.items()
                 if t != best and s["score"] > 0 and s["score"] >= scores[best]["score"] * 0.5]
 
    return {
        "theme": best,
        "secondary": secondary,
        "matched": scores[best]["matched"],
        "all_scores": {t: s["score"] for t, s in scores.items() if s["score"] > 0},
    }
 
 
def find_peak_day(df, semaine):
    """
    Trouve le jour exact du pic dans une semaine donnee.
    Retourne la date, le volume et le ton de ce jour.
    """
    week_df = df[df["semaine"] == semaine].copy()
    if len(week_df) == 0 or "event_date" not in week_df.columns:
        return None
 
    daily = week_df.groupby(week_df["event_date"].dt.date).agg(
        volume=("GLOBALEVENTID", "count"),
        tone=("AvgTone", "mean"),
    ).reset_index()
 
    if len(daily) == 0:
        return None
 
    peak = daily.loc[daily["volume"].idxmax()]
    return {
        "date": peak["event_date"],
        "volume": int(peak["volume"]),
        "tone": round(peak["tone"], 2),
        "total_semaine": int(daily["volume"].sum()),
        "pct_du_total": round(peak["volume"] / daily["volume"].sum() * 100),
    }
 
 
def generate_structured_summary(context, theme_info, peak_day, articles_grouped):
    """
    Genere un resume analytique structure et concis.
    """
    nb = context.get("nb_events", 0)
    tone = context.get("tone_moyen", 0)
    theme = theme_info.get("theme", "Activite mediatique")
    pct_conflit = context.get("pct_conflit", 0)
 
    # Phrase d'accroche selon la gravite
    if tone < -3:
        accroche = "Couverture mediatique exceptionnellement negative"
    elif tone < -2:
        accroche = "Couverture mediatique a forte dominante negative"
    elif tone < -1:
        accroche = "Couverture mediatique moderement negative"
    elif tone > 2:
        accroche = "Couverture mediatique a dominante positive"
    else:
        accroche = "Couverture mediatique soutenue"
 
    # Construire le resume
    parts = [accroche]
 
    # Jour du pic
    if peak_day:
        parts.append(
            f"avec un pic le {peak_day['date'].strftime('%d/%m/%Y')} "
            f"({peak_day['volume']} evenements sur {peak_day['total_semaine']}, "
            f"soit {peak_day['pct_du_total']}% de la semaine)"
        )
 
    # Themes dominants
    if theme_info.get("secondary"):
        parts.append(
            f"dominee par les thematiques « {theme} » "
            f"et « {theme_info['secondary'][0]} »"
        )
    else:
        parts.append(f"centree sur la thematique « {theme} »")
 
    # Types d'evenements
    if "top_types" in context and len(context["top_types"]) > 0:
        type1 = context["top_types"][0]
        parts.append(f"Le type d'evenement le plus frequent est « {type1[0]} » ({type1[2]}%)")
 
    # Acteurs
    if "top_actors" in context and len(context["top_actors"]) > 0:
        actors = [a[0] for a in context["top_actors"][:3]]
        parts.append("impliquant principalement " + ", ".join(actors))
 
    # Conflit
    if pct_conflit > 50:
        parts.append(f"avec une forte proportion d'evenements conflictuels ({pct_conflit}%)")
    elif pct_conflit > 35:
        parts.append(f"avec une proportion notable d'evenements conflictuels ({pct_conflit}%)")
 
    return ". ".join(parts) + "."
 
 
def analyze_peak_complete(df, semaine):
    """
    Analyse complete d'un pic : contexte GDELT, theme detecte,
    jour exact, articles regroupes, resume structure.
    """
    from datetime import timedelta
 
    # 1. Contexte GDELT de base
    week_df = df[df["semaine"] == semaine].copy()
    if len(week_df) == 0:
        return None
 
    context = {}
    context["nb_events"] = len(week_df)
    context["tone_moyen"] = round(week_df["AvgTone"].mean(), 2)
    context["goldstein_moyen"] = round(week_df["GoldsteinScale"].mean(), 2)
 
    # Top types
    if "event_root_label" in week_df.columns:
        top_types = week_df["event_root_label"].value_counts().head(3)
        context["top_types"] = [(t, int(c), round(c/len(week_df)*100)) for t, c in top_types.items()]
    elif "EventRootCode" in week_df.columns:
        top_codes = week_df["EventRootCode"].value_counts().head(3)
        context["top_types"] = [(CAMEO_ROOT_LABELS.get(t, t), int(c), round(c/len(week_df)*100)) for t, c in top_codes.items()]
 
    # Top acteurs
    if "Actor1Name" in week_df.columns:
        top_actors = week_df["Actor1Name"].dropna().value_counts().head(3)
        context["top_actors"] = [(a, int(c)) for a, c in top_actors.items()]
 
    # Top sources
    if "source_domain" in week_df.columns:
        top_sources = week_df["source_domain"].dropna().value_counts().head(5)
        context["top_sources"] = [(s, int(c)) for s, c in top_sources.items()]
 
    # Lieux
    if "ActionGeo_FullName" in week_df.columns:
        top_lieux = week_df["ActionGeo_FullName"].dropna().value_counts().head(3)
        context["top_lieux"] = [(l, int(c)) for l, c in top_lieux.items()]
 
    # Conflit
    if "QuadClass" in week_df.columns:
        context["pct_conflit"] = round((week_df["QuadClass"].isin([3, 4])).mean() * 100)
 
    # Piliers
    if "pilier" in week_df.columns:
        context["piliers"] = {p: int(v) for p, v in week_df["pilier"].value_counts(normalize=True).mul(100).round(0).items()}
 
    # 2. Mots cles des URLs
    url_keywords = []
    if "SOURCEURL" in week_df.columns:
        url_keywords = extract_keywords_from_urls(week_df["SOURCEURL"].dropna().tolist())
 
    # 3. Articles regroupes
    headlines_data = get_peak_headlines(df, semaine, max_results=5)
    articles = headlines_data.get("events", [])
 
    # 4. Detection du theme
    theme_info = detect_theme(url_keywords, 
                              [{"titre": e.get("titre", "")} for e in articles],
                              context)
 
    # 5. Jour exact du pic
    peak_day = find_peak_day(df, semaine)
 
    # 6. Resume structure
    summary = generate_structured_summary(context, theme_info, peak_day, articles)
    
    topics = extract_discussion_topics(url_keywords,
        [{"titre": e.get("titre", "")} for e in articles], context)
    cause_title = generate_cause_title(topics, theme_info, articles)
    cause = generate_cause_explanation(theme_info, context,
        [{"titre": e.get("titre", "")} for e in articles], peak_day, topics)
    
    return {
        "context": context,
        "theme": theme_info,
        "peak_day": peak_day,
        "articles": articles,
        "url_keywords": url_keywords,
        "summary": summary,
        "cause": cause,
        "topics": topics,
        "cause_title": cause_title,
    }

def extract_discussion_topics(url_keywords, articles, context):
    """
    Extrait les sujets de discussion concrets a partir des mots cles
    et des titres d'articles. Retourne une liste de topics lisibles.
    """
    import re
    from collections import Counter
 
    # Collecter tous les mots significatifs
    all_words = Counter()
 
    # Mots cles des URLs
    if url_keywords:
        for kw, count in url_keywords:
            all_words[kw.lower()] += count
 
    # Mots des titres d'articles
    if articles:
        for art in articles:
            titre = art.get("titre", "").lower()
            words = re.findall(r'[a-zA-Zéèêëàâùûôîïç]{4,}', titre)
            for w in words:
                all_words[w] += 1
 
    # Mots a ignorer
    stop = {
        "benin", "bénin", "news", "article", "says", "world", "three",
        "over", "after", "from", "with", "that", "this", "have", "more",
        "been", "about", "also", "will", "into", "than", "first", "last",
        "dans", "pour", "avec", "plus", "sont", "tout", "comme", "faire",
        "cotonou", "republic",
    }
 
    # Mapping mots cles -> topics lisibles
    topic_mapping = {
        "coup": "tentative de coup d'Etat",
        "putsch": "tentative de coup d'Etat",
        "overthrow": "renversement de pouvoir",
        "arrest": "arrestations",
        "arrested": "arrestations",
        "detained": "arrestations",
        "detains": "arrestations",
        "prison": "condamnations",
        "sentenced": "condamnations",
        "trial": "proces",
        "proces": "proces",
        "condemned": "condamnations",
        "attack": "attaque armee",
        "attaque": "attaque armee",
        "killed": "victimes",
        "tues": "victimes",
        "dead": "victimes",
        "soldiers": "forces armees",
        "military": "tensions militaires",
        "militaire": "tensions militaires",
        "army": "forces armees",
        "armee": "forces armees",
        "security": "enjeux securitaires",
        "securite": "enjeux securitaires",
        "terrorism": "menace terroriste",
        "terroris": "menace terroriste",
        "jnim": "menace terroriste (JNIM)",
        "border": "tensions frontalieres",
        "frontiere": "tensions frontalieres",
        "election": "processus electoral",
        "vote": "vote parlementaire",
        "mandat": "reforme des mandats",
        "mandate": "reforme des mandats",
        "constitution": "reforme constitutionnelle",
        "governance": "questions de gouvernance",
        "gouvernance": "questions de gouvernance",
        "government": "action gouvernementale",
        "gouvernement": "action gouvernementale",
        "opposition": "tensions politiques internes",
        "president": "presidence de la republique",
        "talon": "presidence Talon",
        "diplomati": "activite diplomatique",
        "cedeao": "reaction de la CEDEAO",
        "ecowas": "reaction de la CEDEAO",
        "nigeria": "implication du Nigeria",
        "france": "relations avec la France",
        "china": "relations avec la Chine",
        "economy": "enjeux economiques",
        "economi": "enjeux economiques",
        "invest": "investissements",
        "trade": "commerce",
        "kidjo": "rayonnement culturel (Angelique Kidjo)",
        "police": "action policiere",
        "protest": "protestations",
        "failed": "echec",
        "foiled": "tentative dejouee",
        "sanctions": "sanctions internationales",
    }
 
    # Trouver les topics
    found_topics = []
    seen = set()
 
    for word, count in all_words.most_common(30):
        if word in stop:
            continue
        for key, topic in topic_mapping.items():
            if key in word or word in key:
                if topic not in seen:
                    found_topics.append(topic)
                    seen.add(topic)
                break
 
    return found_topics[:6]
 
 
def generate_cause_title(topics, theme_info, articles):
    """
    Genere un titre synthetique pour la cause a partir
    des topics et des titres d'articles.
    Exemple : "Tentative de coup d'Etat fortement mediatisee"
    """
    import re
 
    theme = theme_info.get("theme", "")
 
    # Chercher dans les topics le sujet principal
    if not topics:
        return theme if theme else "Activite mediatique inhabituelle"
 
    main_topic = topics[0]
 
    # Enrichir avec le contexte
    qualifiers = {
        "tentative de coup d'Etat": "Tentative de coup d'Etat fortement mediatisee",
        "attaque armee": "Attaque armee et reponse securitaire",
        "victimes": "Pertes humaines et reaction internationale",
        "arrestations": "Vague d'arrestations et tensions politiques",
        "condamnations": "Condamnations judiciaires a forte resonance mediatique",
        "proces": "Proces politique sous haute attention mediatique",
        "menace terroriste": "Menace terroriste et mobilisation securitaire",
        "menace terroriste (JNIM)": "Attaque du JNIM et crise securitaire",
        "tensions militaires": "Tensions militaires et instabilite",
        "processus electoral": "Processus electoral et debat public",
        "vote parlementaire": "Vote parlementaire sous tension",
        "reforme des mandats": "Reforme des mandats et debat constitutionnel",
        "reforme constitutionnelle": "Reforme constitutionnelle controversee",
        "reaction de la CEDEAO": "Intervention regionale et reaction de la CEDEAO",
        "implication du Nigeria": "Implication du Nigeria dans la crise",
        "rayonnement culturel (Angelique Kidjo)": "Angelique Kidjo, visibilite internationale positive",
        "questions de gouvernance": "Gouvernance et tensions institutionnelles",
        "tensions frontalieres": "Incidents frontaliers et securite regionale",
        "activite diplomatique": "Activite diplomatique soutenue",
        "enjeux economiques": "Enjeux economiques et attractivite",
        "action policiere": "Operations de police et ordre public",
        "protestations": "Protestations et tensions sociales",
    }
 
    return qualifiers.get(main_topic, main_topic.capitalize())
 
 
def generate_cause_explanation(theme_info, context, articles, peak_day, topics):
    """
    Genere une explication detaillee avec les sujets de discussion.
    """
    theme = theme_info.get("theme", "")
    tone = context.get("tone_moyen", 0)
    secondary = theme_info.get("secondary", [])
 
    # Phrase d'accroche basee sur le theme et les topics
    if not topics:
        accroche = "une concentration inhabituelle d'evenements mediatiques"
    elif len(topics) >= 2:
        accroche = topics[0] + " et " + topics[1] + " qui ont suivi"
    else:
        accroche = topics[0]
 
    cause = f"La hausse mediatique semble principalement liee a {accroche}."
 
    # Tonalite
    if tone < -3:
        cause += " La tonalite fortement negative traduit un traitement critique de la situation par la presse."
    elif tone < -2:
        cause += " Le traitement mediatique est globalement defavorable."
 
    return cause

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
