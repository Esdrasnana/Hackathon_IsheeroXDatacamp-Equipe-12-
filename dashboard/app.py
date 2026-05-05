"""
=============================================================================
SENTINEL 360 . Dashboard  Benin 2025
=============================================================================
Lancer :  streamlit run app.py
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    load_data, get_date_range, filter_data,
    compute_weekly_zscore, compute_actor_interactions,
    compute_actor_type_crosstab, compute_zone_stats,
    compute_source_bias, compute_alert_feed,
    ZONE_ORDER, ACTOR_TYPE_LABELS,
)

st.set_page_config(page_title="Sentinel 360", page_icon="S", layout="wide", initial_sidebar_state="expanded")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PALETTE DE COULEURS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Couleur principale : bleu roi
ROYAL = "#2C5F8A"
ROYAL_LIGHT = "#3A7CB8"
ROYAL_DARK = "#1E3A5F"

# Accents
TURQUOISE = "#4ECDC4"
CORAL = "#FF6B6B"
GOLD = "#F7DC6F"

# Semantique
C_POS = "#2ECC71"
C_NEU = "#F39C12"
C_NEG = "#E74C3C"

# Surfaces
CARD_BG = "#141820"
CARD_BORDER = "#1F2530"
TEXT_PRIMARY = "#EAEAEA"
TEXT_SECONDARY = "#9098A3"

QUADCLASS_COLORS = {"Coopération verbale":"#5DADE2","Coopération matérielle":"#2ECC71","Conflit verbal":"#F39C12","Conflit matériel":"#E74C3C"}
ZONE_COLORS = {"Littoral . Cotonou / Porto-Novo":"#3498DB","Sud . Plateau / Zou":"#27AE60","Centre . Borgou / Collines":"#F39C12","Nord . Zone frontaliere":"#E74C3C"}
SENTIMENT_COLORS = {"positif":"#2ECC71","neutre":"#F39C12","négatif":"#E74C3C","negatif":"#E74C3C"}
PILIER_COLORS = {"economie":"#3498DB","securite":"#E74C3C","social":"#2ECC71","autre":"#7F8C8D"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    #MainMenu {{ visibility: hidden; }}
    header {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    [data-testid="stHeader"] {{ display: none; }}
    [data-testid="stToolbar"] {{ display: none; }}

    .block-container {{ padding-top: 0.5rem; max-width: 1200px; }}
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

    /* ── Titre de page ── */
    .page-title {{
        font-size: 1.8rem;
        font-weight: 800;
        color: {ROYAL_LIGHT};
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin-bottom: 6px;
    }}
    .page-subtitle {{
        font-size: 0.88rem;
        font-weight: 400;
        color: {TEXT_SECONDARY};
        line-height: 1.45;
        margin-bottom: 22px;
        padding-bottom: 16px;
        border-bottom: 2px solid {ROYAL_DARK};
    }}

    /* ── Carte metrique ── */
    .kpi-card {{
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-top: 3px solid {ROYAL};
        border-radius: 6px;
        padding: 18px 20px 14px 20px;
        margin-bottom: 10px;
    }}
    .kpi-label {{
        font-size: 0.7rem;
        font-weight: 600;
        color: {TEXT_SECONDARY};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.55rem;
        font-weight: 700;
        color: {TEXT_PRIMARY};
        line-height: 1.2;
    }}
    .kpi-note {{ font-size: 0.76rem; margin-top: 4px; }}
    .kpi-note.neg {{ color: {C_NEG}; }}
    .kpi-note.pos {{ color: {C_POS}; }}
    .kpi-note.neu {{ color: {C_NEU}; }}

    /* ── Titre de section ── */
    .sec-title {{
        font-size: 1.1rem;
        font-weight: 700;
        color: {ROYAL_LIGHT};
        margin-bottom: 4px;
        letter-spacing: -0.01em;
    }}
    .sec-desc {{
        font-size: 0.8rem;
        font-weight: 400;
        color: {TEXT_SECONDARY};
        line-height: 1.5;
        margin-bottom: 16px;
    }}

    /* ── Encadre Insight ── */
    .insight {{
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-left: 4px solid {ROYAL};
        border-radius: 0 6px 6px 0;
        padding: 14px 18px;
        margin: 12px 0;
    }}
    .insight-tag {{
        font-size: 0.65rem;
        font-weight: 700;
        color: {ROYAL_LIGHT};
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 6px;
    }}
    .insight-body {{
        font-size: 0.86rem;
        color: #C0C4CC;
        line-height: 1.55;
    }}
    .insight-body strong {{
        color: {TEXT_PRIMARY};
        font-weight: 600;
    }}

    /* ── Encadre Alerte ── */
    .alert {{
        background: #1c1215;
        border: 1px solid #2a1a1e;
        border-left: 4px solid {C_NEG};
        border-radius: 0 6px 6px 0;
        padding: 14px 18px;
        margin: 12px 0;
    }}
    .alert.warning {{
        background: #1c1a12;
        border-color: #2a2518;
        border-left-color: {C_NEU};
    }}
    .alert-tag {{
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 6px;
        color: {C_NEG};
    }}
    .alert.warning .alert-tag {{ color: {C_NEU}; }}

    /* ── Zone card ── */
    .zone {{
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-radius: 0 6px 6px 0;
        padding: 14px 18px;
        margin-bottom: 8px;
    }}
    .zone-name {{ font-size: 0.84rem; font-weight: 700; margin-bottom: 5px; }}
    .zone-stats {{ display: flex; gap: 16px; font-size: 0.78rem; color: #B8BCC4; }}
    .zone-stats strong {{ color: {TEXT_PRIMARY}; }}
    .zone-detail {{ font-size: 0.72rem; color: {TEXT_SECONDARY}; margin-top: 5px; }}

    /* ── Utilitaires ── */
    .sep {{ border: none; border-top: 1px solid {CARD_BORDER}; margin: 22px 0; }}

    .fixed-footer {{
        position: fixed; bottom: 0; left: 0; right: 0;
        background: #0C0E12; border-top: 1px solid {CARD_BORDER};
        padding: 8px 0; text-align: center; z-index: 999;
    }}
    .fixed-footer span {{ font-size: 0.68rem; color: #444; font-family: 'Inter', sans-serif; }}
    .bottom-spacer {{ height: 45px; }}

    [data-testid="stSidebar"] {{ border-right: 1px solid {CARD_BORDER}; }}
    [data-testid="stMetric"] {{ background: transparent !important; border: none !important; padding: 0 !important; }}
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def kpi(label, value, note=None, note_type="neu"):
    n = f'<div class="kpi-note {note_type}">{note}</div>' if note else ""
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>{n}</div>', unsafe_allow_html=True)

def sec(title, desc=""):
    d = f'<div class="sec-desc">{desc}</div>' if desc else ""
    st.markdown(f'<div class="sec-title">{title}</div>{d}', unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight"><div class="insight-tag">A retenir</div><div class="insight-body">{text}</div></div>', unsafe_allow_html=True)

def alert(text, level="alert"):
    cls = "warning" if level == "warning" else ""
    tag = "Point de vigilance" if level == "warning" else "Signal detecte"
    st.markdown(f'<div class="alert {cls}"><div class="alert-tag">{tag}</div><div class="insight-body">{text}</div></div>', unsafe_allow_html=True)

def apply_style(fig, height=400):
    fig.update_layout(
        height=height, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#B0B4BC", size=12),
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", y=-0.15, font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#181C24", zerolinecolor="#1F2530"),
        yaxis=dict(gridcolor="#181C24", zerolinecolor="#1F2530"),
        hovermode="x unified",
    )
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DONNEES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data
def get_data():
    return load_data()

try:
    df_raw = get_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.sidebar:
    st.markdown(f"""
    <div style="padding: 4px 0 16px 0; border-bottom: 2px solid {ROYAL_DARK}; margin-bottom: 16px;">
        <div style="font-size: 1.2rem; font-weight: 800; color: {ROYAL_LIGHT}; letter-spacing: 0.04em;">SENTINEL 360</div>
        <div style="font-size: 0.72rem; color: {TEXT_SECONDARY}; margin-top: 3px; font-weight: 400;">Intelligence mediatique . Benin 2025</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation",
        ["Signaux et stabilite", "Dynamique d'influence", "Medias et geographie", "Fil d'alerte"],
        label_visibility="collapsed")

    st.markdown('<hr class="sep">', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-label">Filtres</div>', unsafe_allow_html=True)

    date_min, date_max = get_date_range(df_raw)
    date_range = st.date_input("Periode", value=(date_min, date_max), min_value=date_min, max_value=date_max)
    selected_dates = date_range if isinstance(date_range, tuple) and len(date_range) == 2 else (date_min, date_max)

    piliers_sel = st.multiselect("Pilier", sorted(df_raw["pilier"].dropna().unique().tolist()), default=sorted(df_raw["pilier"].dropna().unique().tolist()))
    sentiments_sel = st.multiselect("Sentiment", df_raw["sentiment_proxy"].dropna().unique().tolist(), default=df_raw["sentiment_proxy"].dropna().unique().tolist())

    df = filter_data(df_raw, selected_dates, piliers_sel, sentiments_sel)

    st.markdown('<hr class="sep">', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Evenements selectionnes</div><div class="kpi-value">{len(df):,}</div><div class="kpi-note neu">sur {len(df_raw):,} au total</div></div>'.replace(",", " "), unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 1 : SIGNAUX ET STABILITE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if page == "Signaux et stabilite":

    st.markdown('<div class="page-title">Signaux et stabilite</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Cette page identifie les periodes inhabituelles dans la couverture mediatique du Benin et mesure l\'evolution de la stabilite percue a travers un indice composite.</div>', unsafe_allow_html=True)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    avg_tone = df["AvgTone"].mean()
    avg_stab = df["stability_index"].mean()
    pct_neg = (df["sentiment_proxy"].isin(["négatif", "negatif"])).mean() * 100

    with c1: kpi("Evenements analyses", f"{len(df):,}".replace(",", " "))
    with c2:
        lb = "Situation stable" if avg_stab > 60 else ("Vigilance requise" if avg_stab > 40 else "Situation critique")
        tp = "pos" if avg_stab > 60 else ("neu" if avg_stab > 40 else "neg")
        kpi("Indice de stabilite", f"{avg_stab:.0f} / 100", lb, tp)
    with c3: kpi("Ton mediatique moyen", f"{avg_tone:.2f}", "Couverture a dominante negative" if avg_tone < 0 else "Couverture a dominante positive", "neg" if avg_tone < 0 else "pos")
    with c4: kpi("Part de couverture negative", f"{pct_neg:.1f}%", f"Soit {int(pct_neg*len(df)/100)} evenements sur la periode", "neg" if pct_neg > 50 else "neu")

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    # ── Anomalies ──
    sec("Detection des periodes inhabituelles",
        "Le graphique ci-dessous represente le volume d'evenements par semaine. Les barres orange et rouges signalent les semaines dont l'activite depasse significativement la normale (Z-Score superieur a 2). La courbe blanche materialise la tendance de fond sur 4 semaines.")

    weekly = compute_weekly_zscore(df)
    fig1 = go.Figure()
    for _, row in weekly[weekly["alerte"] != "Normal"].iterrows():
        c = "rgba(231,76,60,0.12)" if row["alerte"] == "Alerte" else "rgba(243,156,18,0.10)"
        fig1.add_vrect(x0=row["semaine"]-pd.Timedelta(days=3), x1=row["semaine"]+pd.Timedelta(days=3), fillcolor=c, line_width=0, layer="below")

    bar_colors = []
    for _, row in weekly.iterrows():
        if row["alerte"] == "Alerte": bar_colors.append(C_NEG)
        elif row["alerte"] == "Vigilance": bar_colors.append(C_NEU)
        elif row["alerte"] == "Creux inhabituel": bar_colors.append("#4A5060")
        else: bar_colors.append(ROYAL_LIGHT)

    fig1.add_trace(go.Bar(x=weekly["semaine"], y=weekly["volume"], name="Volume hebdomadaire", marker_color=bar_colors, opacity=0.85))
    fig1.add_trace(go.Scatter(x=weekly["semaine"], y=weekly["volume_ma4"], name="Tendance (moy. mobile 4 sem.)", line=dict(color="#FFF", width=2, dash="dot"), mode="lines"))
    apply_style(fig1, 400)
    fig1.update_layout(yaxis_title="Nombre d'evenements")
    st.plotly_chart(fig1, use_container_width=True)

    anomalies = weekly[weekly["alerte"].isin(["Alerte", "Vigilance"])].sort_values("z_score", ascending=False)
    if len(anomalies) > 0:
        top = anomalies.iloc[0]
        ratio = top["volume"] / weekly["volume"].mean()
        alert(f"La semaine du <strong>{top['semaine'].strftime('%d/%m/%Y')}</strong> presente une activite anormale : <strong>{int(top['volume'])} evenements</strong> enregistres, soit <strong>{ratio:.1f} fois</strong> la moyenne habituelle (Z-Score : {top['z_score']:.1f}).", level="alert" if top["z_score"] > 3 else "warning")
    else:
        insight("Aucune periode d'activite anormale n'a ete detectee sur la plage selectionnee. La couverture mediatique est restee dans les limites habituelles.")

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    # ── Indice Sentinel ──
    cg, ce = st.columns([1, 2])
    with cg:
        sec("Indice Sentinel", "Cet indicateur synthetise la nature des evenements et leur perception mediatique en un score unique, de 0 (situation critique) a 100 (stabilite).")
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=avg_stab,
            number=dict(font=dict(size=46, color="#FFF", family="JetBrains Mono")),
            gauge=dict(axis=dict(range=[0,100], tickcolor="#444", dtick=20), bar=dict(color=ROYAL_LIGHT), bgcolor="#141820", borderwidth=0,
                steps=[dict(range=[0,30],color="#2a1215"),dict(range=[30,50],color="#2a2212"),dict(range=[50,70],color="#1a2a15"),dict(range=[70,100],color="#122a1a")],
                threshold=dict(line=dict(color="#FFF",width=2),thickness=0.8,value=avg_stab))))
        fig_g.update_layout(height=260, margin=dict(l=30,r=30,t=30,b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#666"))
        st.plotly_chart(fig_g, use_container_width=True)

    with ce:
        sec("Trajectoire mensuelle de l'indice", "La courbe bleue retrace l'evolution de l'indice Sentinel. Les barres rouges mesurent l'ecart entre la realite des evenements et leur traitement mediatique : plus elles sont hautes, plus la perception diverge de la realite.")
        monthly = df.groupby("mois").agg(goldstein=("GoldsteinScale","mean"), tone=("AvgTone","mean"), stability=("stability_index","mean")).reset_index().sort_values("mois")
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(x=monthly["mois"], y=monthly["stability"], name="Indice Sentinel", line=dict(color=ROYAL_LIGHT,width=3), mode="lines+markers", marker=dict(size=7)))
        fig_e.add_trace(go.Bar(x=monthly["mois"], y=(monthly["goldstein"]-monthly["tone"]).abs(), name="Ecart realite / perception", marker_color="rgba(255,107,107,0.25)", yaxis="y2"))
        fig_e.add_hline(y=50, line_dash="dot", line_color="#444", annotation_text="Seuil de vigilance", annotation_font_color="#666")
        apply_style(fig_e, 280)
        fig_e.update_layout(yaxis=dict(title="Indice",range=[0,100]), yaxis2=dict(title="Ecart",overlaying="y",side="right",showgrid=False))
        st.plotly_chart(fig_e, use_container_width=True)

    if len(monthly) >= 2:
        f_s, l_s = monthly.iloc[0]["stability"], monthly.iloc[-1]["stability"]
        direction = "progresse" if l_s > f_s else "recule"
        insight(f"L'indice de stabilite a <strong>{direction} de {abs(f_s - l_s):.0f} points</strong> entre {monthly.iloc[0]['mois']} et {monthly.iloc[-1]['mois']}, passant de {f_s:.0f} a {l_s:.0f}. Les mois presentant un ecart eleve entre realite et perception meritent une attention particuliere : ils revelent des tensions que les chiffres bruts ne montrent pas.")

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    # ── Cooperation / Conflit + Pilier ──
    cq, cp = st.columns(2)
    with cq:
        sec("Equilibre entre cooperation et conflit", "Decomposition mensuelle des evenements selon la grille CAMEO : cooperation (verbale ou materielle) contre conflit (verbal ou materiel).")
        mq = df.groupby(["mois","quad_class_label"]).size().reset_index(name="count")
        fig_q = px.bar(mq, x="mois", y="count", color="quad_class_label", color_discrete_map=QUADCLASS_COLORS, labels={"count":"Evenements","mois":"","quad_class_label":""}, barmode="stack")
        apply_style(fig_q, 350)
        fig_q.update_layout(legend=dict(orientation="h",y=-0.18,title=""))
        st.plotly_chart(fig_q, use_container_width=True)

    with cp:
        sec("Repartition thematique", "Part relative de chaque pilier dans la couverture mediatique du Benin.")
        pc = df["pilier"].value_counts().reset_index(); pc.columns=["pilier","count"]
        fig_p = px.pie(pc, values="count", names="pilier", color="pilier", color_discrete_map=PILIER_COLORS, hole=0.45)
        apply_style(fig_p, 350)
        fig_p.update_layout(legend=dict(orientation="h",y=-0.1), margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig_p, use_container_width=True)

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    sec("Evolution du sentiment dans le temps", "Proportion mensuelle des evenements couverts positivement, de maniere neutre, ou negativement. Une hausse durable de la part negative constitue un signal de degradation de l'image.")
    ms = df.groupby(["mois","sentiment_proxy"]).size().reset_index(name="count")
    mt = ms.groupby("mois")["count"].transform("sum")
    ms["pct"] = (ms["count"]/mt*100).round(1)
    fig_s = px.bar(ms, x="mois", y="pct", color="sentiment_proxy", color_discrete_map=SENTIMENT_COLORS, labels={"pct":"Part (%)","mois":"","sentiment_proxy":"Sentiment"}, barmode="stack")
    apply_style(fig_s, 320)
    fig_s.update_layout(legend=dict(orientation="h",y=-0.18,title=""))
    st.plotly_chart(fig_s, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 2 : DYNAMIQUE D'INFLUENCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif page == "Dynamique d'influence":

    st.markdown('<div class="page-title">Dynamique d\'influence</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Cette page analyse les acteurs impliques dans les evenements lies au Benin : qui sont-ils, comment interagissent-ils, et quelles relations s\'averent les plus tendues ou les plus cooperatives.</div>', unsafe_allow_html=True)

    cl, cr = st.columns(2)
    with cl:
        sec("Principaux acteurs identifies", "Classement des acteurs apparaissant le plus frequemment comme initiateurs d'evenements dans la couverture mediatique du Benin.")
        ta = df["Actor1Name"].dropna().value_counts().head(12).reset_index(); ta.columns=["Acteur","N"]
        fig_a = px.bar(ta, x="N", y="Acteur", orientation="h", color_discrete_sequence=[ROYAL_LIGHT])
        apply_style(fig_a, 440)
        fig_a.update_layout(yaxis=dict(autorange="reversed",title=""), xaxis_title="Nombre d'evenements")
        st.plotly_chart(fig_a, use_container_width=True)

    with cr:
        sec("Matrice des interactions par type d'acteur", "Chaque bulle represente une interaction entre un type d'acteur initiateur (axe horizontal) et un type d'acteur cible (axe vertical). La taille reflete la frequence ; la couleur, le ton moyen de la relation.")
        cross = compute_actor_type_crosstab(df)
        if len(cross) > 0:
            ct = cross.head(15)
            fig_c = px.scatter(ct, x="a1_label", y="a2_label", size="count", color="tone_moyen",
                color_continuous_scale=[C_NEG, C_NEU, C_POS], range_color=[-5,5], size_max=35,
                labels={"a1_label":"Initiateur","a2_label":"Cible","count":"Interactions","tone_moyen":"Ton"})
            apply_style(fig_c, 440)
            fig_c.update_layout(xaxis=dict(tickangle=-45), coloraxis_colorbar=dict(title="Ton",len=0.6))
            st.plotly_chart(fig_c, use_container_width=True)
        else:
            st.info("Les donnees disponibles ne permettent pas de construire cette matrice avec les filtres actuels.")

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    sec("Relations bilaterales entre acteurs nommes", "Les paires d'acteurs les plus frequentes sont classees ci-dessous. La couleur traduit la nature de la relation : rouge pour les interactions tendues (ton inferieur a -2), vert pour la cooperation (ton superieur a +2), jaune pour les relations neutres.")

    pairs = compute_actor_interactions(df, min_count=3)
    if len(pairs) > 0:
        pt = pairs.head(15).copy()
        pt["paire"] = pt["Actor1Name"] + "  >  " + pt["Actor2Name"]
        pt["nature"] = pt["tone_moyen"].apply(lambda x: "Relation tendue" if x<-2 else ("Cooperation" if x>2 else "Relation neutre"))
        fig_pr = px.bar(pt, x="interactions", y="paire", orientation="h", color="nature",
            color_discrete_map={"Relation tendue":C_NEG, "Relation neutre":C_NEU, "Cooperation":C_POS},
            hover_data={"tone_moyen":":.2f","pct_conflit":":.1f"}, labels={"interactions":"Nombre d'interactions","paire":""})
        apply_style(fig_pr, 460)
        fig_pr.update_layout(yaxis=dict(autorange="reversed"), legend=dict(title=""))
        st.plotly_chart(fig_pr, use_container_width=True)

        tensest = pt[pt["nature"]=="Relation tendue"]
        if len(tensest) > 0:
            t = tensest.iloc[0]
            insight(f"La relation la plus tendue concerne <strong>{t['Actor1Name']}</strong> et <strong>{t['Actor2Name']}</strong>, avec un ton moyen de <strong>{t['tone_moyen']:.2f}</strong> sur {int(t['interactions'])} interactions. <strong>{t['pct_conflit']:.0f}%</strong> de ces interactions sont classees comme conflictuelles dans la grille CAMEO.")
    else:
        st.info("Les filtres actuels ne permettent pas d'identifier de paires d'acteurs significatives.")

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    sec("Pays tiers les plus impliques", "Pays etrangers apparaissant comme acteurs dans les evenements lies au Benin, que ce soit en tant qu'initiateurs ou en tant que cibles.")
    ac = pd.concat([df["Actor1CountryCode"].dropna(), df["Actor2CountryCode"].dropna()])
    fg = ac[~ac.isin(["BEN",""])].value_counts().head(10).reset_index(); fg.columns=["Pays","Mentions"]
    fig_f = px.bar(fg, x="Pays", y="Mentions", color_discrete_sequence=[GOLD])
    apply_style(fig_f, 300)
    fig_f.update_layout(xaxis_title="Code pays", yaxis_title="Nombre de mentions")
    st.plotly_chart(fig_f, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 3 : MEDIAS ET GEOGRAPHIE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif page == "Medias et geographie":

    st.markdown('<div class="page-title">Couverture mediatique et geographie</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Cette page compare la perception du Benin entre medias nationaux et internationaux, identifie les biais de couverture par langue et par theme, et cartographie la repartition geographique de l\'attention mediatique.</div>', unsafe_allow_html=True)

    cb, clg = st.columns(2)
    with cb:
        sec("Ecart de perception : presse nationale contre internationale", "Le ton moyen des articles est compare selon l'origine du media. Un ecart significatif revele un biais de couverture.")
        if "source_type" in df.columns:
            bias = df.groupby("source_type").agg(tone_moyen=("AvgTone","mean"), nb=("GLOBALEVENTID","count")).reset_index()
            fig_bi = go.Figure()
            for _, row in bias.iterrows():
                c = C_POS if row["tone_moyen"]>0 else C_NEG
                fig_bi.add_trace(go.Bar(x=[row["source_type"]], y=[row["tone_moyen"]], marker_color=c, text=f"{row['tone_moyen']:.2f}", textposition="outside", showlegend=False))
            fig_bi.add_hline(y=0, line_color="#333", line_dash="dot")
            apply_style(fig_bi, 340)
            fig_bi.update_layout(xaxis_title="", yaxis_title="Ton moyen")
            st.plotly_chart(fig_bi, use_container_width=True)
            if len(bias) >= 2:
                nat = bias[bias["source_type"].str.contains("national",case=False,na=False)]
                intl = bias[~bias["source_type"].str.contains("national",case=False,na=False)]
                if len(nat)>0 and len(intl)>0:
                    diff = abs(intl.iloc[0]["tone_moyen"]-nat.iloc[0]["tone_moyen"])
                    if diff > 0.5:
                        d = "plus negative" if intl.iloc[0]["tone_moyen"]<nat.iloc[0]["tone_moyen"] else "plus positive"
                        insight(f"Un ecart de <strong>{diff:.2f} points</strong> separe les sources internationales des sources nationales : la couverture etrangere est {d}. Ce biais peut affecter la perception du Benin aupres des investisseurs et des partenaires internationaux.")
        else:
            st.info("La classification des sources n'est pas disponible dans les donnees.")

    with clg:
        sec("Perception selon la langue des medias", "Le ton moyen varie selon le groupe linguistique du media source. Seuls les groupes comptant au moins 10 evenements sont retenus.")
        if "media_lang_group" in df.columns:
            lt = df.groupby("media_lang_group").agg(tone_moyen=("AvgTone","mean"), nb=("GLOBALEVENTID","count")).reset_index()
            lt = lt[lt["nb"]>=10].sort_values("tone_moyen")
            colors = [C_POS if v>0 else C_NEG for v in lt["tone_moyen"]]
            fig_lt = go.Figure(go.Bar(x=lt["media_lang_group"], y=lt["tone_moyen"], marker_color=colors, text=lt["tone_moyen"].round(2), textposition="outside"))
            fig_lt.add_hline(y=0, line_color="#333", line_dash="dot")
            apply_style(fig_lt, 340)
            fig_lt.update_layout(xaxis_title="", yaxis_title="Ton moyen")
            st.plotly_chart(fig_lt, use_container_width=True)

    sb = compute_source_bias(df)
    if len(sb) > 0:
        st.markdown('<hr class="sep">', unsafe_allow_html=True)
        sec("Repartition thematique selon le type de source", "La presse internationale et la presse nationale couvrent-elles les memes sujets ? Un desequilibre revele un biais editorial : si les sources etrangeres se concentrent sur la securite, elles donnent une image deformee du pays.")
        fig_sb = px.bar(sb, x="pilier", y="nb_events", color="source_type", barmode="group",
            color_discrete_sequence=[ROYAL_LIGHT, CORAL], labels={"nb_events":"Evenements","pilier":"","source_type":"Type de source"})
        apply_style(fig_sb, 300)
        st.plotly_chart(fig_sb, use_container_width=True)

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    sec("Repartition geographique de l'attention", "Le Benin est decoupe en quatre macro-zones (Littoral, Sud, Centre, Nord). La carte localise chaque evenement ; le tableau a droite synthetise le niveau de tension par zone.")
    cm, cz = st.columns([3,2])

    with cm:
        df_geo = df[df["ActionGeo_Lat"].notna() & df["ActionGeo_Long"].notna()].copy()
        if len(df_geo) > 0:
            dm = df_geo.head(5000)
            fig_mp = px.scatter_mapbox(dm, lat="ActionGeo_Lat", lon="ActionGeo_Long", color="macro_zone", color_discrete_map=ZONE_COLORS,
                hover_name="ActionGeo_FullName", hover_data={"pilier":True,"AvgTone":":.2f","event_date":True,"Actor1Name":True,"ActionGeo_Lat":False,"ActionGeo_Long":False,"macro_zone":False},
                zoom=5.5, center={"lat":9.3,"lon":2.3}, mapbox_style="carto-darkmatter", labels={"macro_zone":"Zone"}, category_orders={"macro_zone":ZONE_ORDER})
            fig_mp.update_traces(marker=dict(size=6,opacity=0.7))
            fig_mp.update_layout(height=480, margin=dict(l=0,r=0,t=0,b=0), legend=dict(orientation="h",y=-0.02,font=dict(size=10),bgcolor="rgba(0,0,0,0.5)"))
            st.plotly_chart(fig_mp, use_container_width=True)
        else:
            st.warning("Aucun evenement geolocalise disponible avec les filtres actuels.")

    with cz:
        sec("Indicateurs par zone")
        zs = compute_zone_stats(df)
        if len(zs) > 0:
            zs["order"] = zs["macro_zone"].map({z:i for i,z in enumerate(ZONE_ORDER)})
            zs = zs.sort_values("order")
            for _, row in zs.iterrows():
                zone = row["macro_zone"]
                color = ZONE_COLORS.get(zone, "#888")
                if row["pct_conflit"]>40: tension,tc = "Eleve","#E74C3C"
                elif row["pct_conflit"]>25: tension,tc = "Modere","#F39C12"
                else: tension,tc = "Faible","#2ECC71"
                st.markdown(f'''<div class="zone" style="border-left:4px solid {color};"><div class="zone-name" style="color:{color};">{zone}</div><div class="zone-stats"><span><strong>{int(row["nb_events"])}</strong> evt</span><span>Ton : <strong>{row["tone_moyen"]:.1f}</strong></span><span style="color:{tc};">Tension : <strong>{tension}</strong></span></div><div class="zone-detail">{row["pct_conflit"]:.0f}% conflictuel . {row["pct_negatif"]:.0f}% negatif . Stabilite : {row["stability_moyen"]:.0f}/100</div></div>''', unsafe_allow_html=True)

            mt_row = zs.loc[zs["pct_conflit"].idxmax()]
            ms_row = zs.loc[zs["pct_conflit"].idxmin()]
            insight(f"La zone <strong>{mt_row['macro_zone']}</strong> concentre le plus de tensions avec <strong>{mt_row['pct_conflit']:.0f}%</strong> d'evenements conflictuels et un ton moyen de {mt_row['tone_moyen']:.1f}. A l'inverse, <strong>{ms_row['macro_zone']}</strong> presente le profil le plus stable ({ms_row['pct_conflit']:.0f}% conflictuel ; indice de stabilite a {ms_row['stability_moyen']:.0f}/100).")

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    sec("Evolution du ton mediatique par zone geographique", "Cette courbe permet de suivre la trajectoire de chaque zone au fil des mois. Un passage sous la barre du zero indique une couverture a dominante negative pour la zone concernee.")
    zm = df[df["macro_zone"]!="Non localisé"].groupby(["mois","macro_zone"]).agg(tone_moyen=("AvgTone","mean")).reset_index()
    fig_ze = px.line(zm, x="mois", y="tone_moyen", color="macro_zone", color_discrete_map=ZONE_COLORS,
        labels={"tone_moyen":"Ton moyen","mois":"","macro_zone":"Zone"}, category_orders={"macro_zone":ZONE_ORDER}, markers=True)
    fig_ze.add_hline(y=0, line_dash="dot", line_color="#444")
    apply_style(fig_ze, 340)
    fig_ze.update_layout(legend=dict(orientation="h",y=-0.2,title=""))
    st.plotly_chart(fig_ze, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 4 : FIL D'ALERTE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif page == "Fil d'alerte":

    st.markdown('<div class="page-title">Fil d\'alerte</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Les periodes de couverture mediatique anormale sont listees ci-dessous avec leur contexte : type d\'evenement dominant, acteur principal, lieu et niveau de tension. Ce fil permet a un decideur ou un journaliste d\'identifier rapidement les moments cles de l\'annee.</div>', unsafe_allow_html=True)

    alert_feed = compute_alert_feed(df, top_n=10)

    if len(alert_feed) > 0:

        # KPIs du fil d'alerte
        ca1, ca2, ca3 = st.columns(3)
        nb_alertes = len(alert_feed[alert_feed["alerte"] == "Alerte"])
        nb_vigilances = len(alert_feed[alert_feed["alerte"] == "Vigilance"])

        with ca1: kpi("Periodes detectees", str(len(alert_feed)))
        with ca2: kpi("Alertes fortes", str(nb_alertes), "Z-Score superieur a 3" if nb_alertes > 0 else "Aucune", "neg" if nb_alertes > 0 else "pos")
        with ca3: kpi("Points de vigilance", str(nb_vigilances), "Z-Score entre 2 et 3" if nb_vigilances > 0 else "Aucun", "neu" if nb_vigilances > 0 else "pos")

        st.markdown('<hr class="sep">', unsafe_allow_html=True)

        sec("Chronologie des periodes inhabituelles", "Chaque fiche correspond a une semaine dont le volume d'evenements depasse significativement la moyenne. Les informations contextuelles sont extraites automatiquement des donnees GDELT.")

        # Afficher chaque alerte comme une fiche
        for i, row in alert_feed.iterrows():
            # Determiner le style selon le niveau
            if row["alerte"] == "Alerte":
                border_color = C_NEG
                badge_bg = "#2a1215"
                badge_color = C_NEG
                badge_text = "ALERTE"
            elif row["alerte"] == "Vigilance":
                border_color = C_NEU
                badge_bg = "#2a2212"
                badge_color = C_NEU
                badge_text = "VIGILANCE"
            else:
                border_color = ROYAL_LIGHT
                badge_bg = "#12202a"
                badge_color = ROYAL_LIGHT
                badge_text = "SIGNAL"

            # Ligne de detail pour la tension
            tension_text = "Dominante conflictuelle" if row["pct_conflit"] > 50 else ("Mixte" if row["pct_conflit"] > 30 else "Dominante cooperative")
            tension_color = C_NEG if row["pct_conflit"] > 50 else (C_NEU if row["pct_conflit"] > 30 else C_POS)

            st.markdown(f'''
            <div style="
                background: {CARD_BG};
                border: 1px solid {CARD_BORDER};
                border-left: 4px solid {border_color};
                border-radius: 0 8px 8px 0;
                padding: 18px 22px;
                margin-bottom: 12px;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div>
                        <span style="font-size: 1.05rem; font-weight: 700; color: {TEXT_PRIMARY};">
                            Semaine du {row["semaine"].strftime("%d/%m/%Y")}
                        </span>
                    </div>
                    <div style="
                        background: {badge_bg};
                        border: 1px solid {border_color};
                        border-radius: 4px;
                        padding: 3px 10px;
                        font-size: 0.65rem;
                        font-weight: 700;
                        color: {badge_color};
                        letter-spacing: 0.08em;
                    ">{badge_text}</div>
                </div>

                
            </div>
            ''', unsafe_allow_html=True)

        st.markdown('<hr class="sep">', unsafe_allow_html=True)

        # Tableau de donnees explorable
        sec("Explorer les evenements", "Le tableau ci-dessous permet de rechercher et filtrer les evenements bruts. Utilisez la barre de recherche pour trouver un acteur, un lieu ou un type d'evenement specifique.")

        # Colonnes a afficher
        display_cols = ["event_date", "Actor1Name", "Actor2Name", "event_root_label", "quad_class_label", "pilier", "AvgTone", "GoldsteinScale", "ActionGeo_FullName", "source_domain"]
        available_cols = [c for c in display_cols if c in df.columns]

        df_display = df[available_cols].copy()

        # Renommer pour lisibilite
        rename_map = {
            "event_date": "Date",
            "Actor1Name": "Acteur 1",
            "Actor2Name": "Acteur 2",
            "event_root_label": "Type",
            "quad_class_label": "Categorie",
            "pilier": "Pilier",
            "AvgTone": "Ton",
            "GoldsteinScale": "Goldstein",
            "ActionGeo_FullName": "Lieu",
            "source_domain": "Source",
        }
        df_display = df_display.rename(columns={k: v for k, v in rename_map.items() if k in df_display.columns})

        if "Ton" in df_display.columns:
            df_display["Ton"] = df_display["Ton"].round(2)
        if "Goldstein" in df_display.columns:
            df_display["Goldstein"] = df_display["Goldstein"].round(2)

        st.dataframe(
            df_display.sort_values("Date", ascending=False) if "Date" in df_display.columns else df_display,
            use_container_width=True,
            height=450,
        )

        # Bouton d'export
        st.markdown('<hr class="sep">', unsafe_allow_html=True)
        sec("Exporter les donnees", "Telechargez les evenements filtres au format CSV pour une analyse complementaire.")

        csv = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Telecharger les donnees filtrees (CSV)",
            data=csv,
            file_name="sentinel360_export.csv",
            mime="text/csv",
        )

    else:
        insight("Aucune periode anormale n'a ete detectee avec les filtres actuels. La couverture mediatique du Benin est restee dans les limites habituelles sur toute la plage selectionnee.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FOOTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown('<div class="bottom-spacer"></div>', unsafe_allow_html=True)
st.markdown('<div class="fixed-footer"><span>SENTINEL 360 . Hackathon Isheero x Benin 2025 . Donnees : GDELT Project</span></div>', unsafe_allow_html=True)
