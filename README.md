# SENTINEL 360

**Intelligence mediatique sur le Benin . Annee 2025**

Analyse de 12 mois de couverture mediatique mondiale sur le Benin a partir de la base GDELT. Ce projet extrait, nettoie, analyse et restitue plus de 21 000 evenements pour trois publics : le journaliste, le chercheur et le decideur public.

Projet realise dans le cadre du Hackathon GDELT x iSHEERO 2026.

---

## Acces rapide

| Ressource | Lien |
|-----------|------|
| Dashboard en ligne | [https://dashboard-app-t2sakihhwibbvdhnqcuxdm.streamlit.app/](URL_A_COMPLETER) |
| Video pitch| [https://drive.google.com/file/d/1bx4pED0DgksbccdgRGkDuLi8szOkruDB/view?usp=drivesdk](URL_A_COMPLETER) |

---

## Installation et execution

### Prerequis

- Python 3.10 ou superieur
- pip

### 1. Cloner le depot

```bash
git clone https://github.com/Esdrasnana/Hackathon_IsheeroXDatacamp-Equipe-12-
cd Hackathon_IsheeroXDatacamp-Equipe-12-
```

### 2. Installer les dependances

```bash
pip install -r requirements.txt
```

### 3. Lancer le dashboard en local

```bash
cd dashboard
streamlit run app.py
```

Le navigateur s'ouvre automatiquement sur `http://localhost:8501`.

### 4. Ouvrir le notebook d'exploration

```bash
jupyter notebook notebooks/eda.ipynb
```

---

## Donnees

Les donnees proviennent du projet GDELT (Global Database of Events, Language and Tone), interroge via Google BigQuery sur les tables partitionnees `gdelt-bq.gdeltv2.events_partitioned`.

## Dashboard

Le dashboard Sentinel 360 est construit avec Streamlit et Plotly. Il comporte 4 pages interactives avec des filtres globaux (periode, pilier thematique, sentiment).

### Page 1 : Signaux et stabilite

Detection des periodes de couverture anormale et suivi de l'indice de stabilite composite Sentinel. Inclut les indicateurs cles, le graphique de detection d'anomalies, la jauge Sentinel, l'evolution mensuelle, l'equilibre cooperation/conflit, la repartition par pilier et l'evolution du sentiment.

### Page 2 : Dynamique d'influence

Analyse des acteurs impliques dans les evenements lies au Benin. Classement des acteurs, matrice d'interactions par type, paires d'acteurs bilaterales colorees par nature de la relation, et pays tiers les plus mentionnes.

### Page 3 : Medias et geographie

Comparaison du ton mediatique entre sources nationales et internationales. Analyse par groupe linguistique. Carte interactive des evenements decoupee en 4 macro-zones avec fiches de tension et evolution temporelle par zone.

### Page 4 : Fil d'alerte

Chronologie des semaines anormales avec contexte automatique (type d'evenement, acteur principal, lieu, pilier). Tableau explorable de l'ensemble des evenements et export CSV.

---

## Axes d'analyse

Le projet est structure autour de 5 axes repondant chacun a une question concrete :

1. **Veille et signaux faibles** : quelles sont les periodes inhabituelles dans la couverture du Benin ?
2. **Stabilite percue** : le Benin est-il percu comme stable ? Cette perception evolue-t-elle ?
3. **Acteurs et relations** : qui interagit avec le Benin et dans quel contexte ?
4. **Image internationale** : les medias etrangers couvrent-ils le Benin differemment de la presse locale ?
5. **Geographie de l'attention** : quelles regions du pays captent la couverture mediatique ?

---

## Indice Sentinel

L'indice Sentinel est un score composite entre 0 (critique) et 100 (stable) concu pour ce projet. Il combine deux dimensions :

- Le **GoldsteinScale** (poids 60%) mesure la nature reelle de l'evenement sur une echelle de -10 a +10.
- Le **AvgTone** (poids 40%) mesure le traitement mediatique de l'evenement.

Un malus de 10 points est applique lorsque les deux dimensions divergent fortement (ecart superieur a 40 points sur l'echelle normalisee). Ce malus detecte les tensions cachees : un evenement de cooperation couvert negativement, par exemple.

---

## Equipe

| Membre | Role | Responsabilite principale |
|--------|------|---------------------------|
| Yawo | Data Engineer | Pipeline de donnees, repo GitHub, reproductibilite |
| Esdras | Data Analyst | Notebook d'exploration, visualisations, insights |
| Roosvelt | Data Scientist | Modele ML, evaluation, interpretation |
| Lucia | ML Engineer | Dashboard Streamlit, deploiement, integration |

---

## Technologies utilisees

| Categorie | Outils |
|-----------|--------|
| Extraction | Google BigQuery, tables GDELT partitionnees |
| Traitement | Python, Pandas, PyArrow |
| Visualisation notebook | Seaborn, Matplotlib, Folium |
| Dashboard | Streamlit, Plotly |
| Machine Learning | Scikit-learn, XGBoost |
| Deploiement | Streamlit Cloud |
| Versioning | GitHub |

---

## Licence

Donnees : GDELT Project (acces libre).

---

## Documentation GDELT

- [Codebook Events v2.0](http://data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf)
- [Manuel CAMEO v1.1b3](http://data.gdeltproject.org/documentation/CAMEO.Manual.1.1b3.pdf)
- [Tables partitionnees BigQuery](https://blog.gdeltproject.org/announcing-partitioned-gdelt-bigquery-tables/)
- [Page centrale de documentation](https://www.gdeltproject.org/data.html)
