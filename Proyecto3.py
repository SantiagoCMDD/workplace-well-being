# ============================================================
# DASHBOARD — BIENESTAR LABORAL EDA 2026-1
# Ejecutar con: python -m streamlit run dashboard_final.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import spearmanr, kruskal

# ── Configuración de página ──────────────────────────────────
st.set_page_config(
    page_title="Riesgos Psicosociales · EDA 2026-1",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Estilos globales ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background-color: #0f1117; color: #e2e8f0; }
    [data-testid="stSidebar"] { background-color: #161b27; border-right: 1px solid #1e2535; }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #161b27; border-radius: 8px;
        padding: 4px; gap: 4px; border: 1px solid #1e2535;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent; color: #94a3b8;
        border-radius: 6px; font-size: 0.82rem;
        font-weight: 500; letter-spacing: 0.03em; padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e2535 !important; color: #e2e8f0 !important;
    }
    .kpi-card {
        background: #161b27; border: 1px solid #1e2535;
        border-radius: 10px; padding: 20px 24px;
        position: relative; overflow: hidden;
    }
    .kpi-card::before {
        content: ''; position: absolute; top: 0; left: 0;
        width: 3px; height: 100%; border-radius: 10px 0 0 10px;
    }
    .kpi-card.riesgo::before  { background: #ef4444; }
    .kpi-card.alerta::before  { background: #f59e0b; }
    .kpi-card.ok::before      { background: #22c55e; }
    .kpi-card.neutro::before  { background: #64748b; }
    .kpi-label {
        font-size: 0.72rem; font-weight: 500; letter-spacing: 0.08em;
        text-transform: uppercase; color: #64748b; margin-bottom: 6px;
    }
    .kpi-valor {
        font-family: 'DM Mono', monospace; font-size: 1.9rem;
        font-weight: 500; color: #e2e8f0; line-height: 1; margin-bottom: 6px;
    }
    .kpi-badge {
        display: inline-block; font-size: 0.68rem; font-weight: 600;
        letter-spacing: 0.06em; text-transform: uppercase;
        padding: 2px 8px; border-radius: 4px;
    }
    .badge-riesgo { background: rgba(239,68,68,0.15);  color: #ef4444; }
    .badge-alerta { background: rgba(245,158,11,0.15); color: #f59e0b; }
    .badge-ok     { background: rgba(34,197,94,0.15);  color: #22c55e; }
    .insight-box {
        background: #161b27; border: 1px solid #1e2535;
        border-left: 3px solid #3b82f6; border-radius: 8px;
        padding: 14px 18px; margin-top: 16px;
        font-size: 0.88rem; color: #94a3b8; line-height: 1.6;
    }
    .insight-box strong { color: #e2e8f0; }
    .seccion-header {
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em;
        text-transform: uppercase; color: #475569;
        margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid #1e2535;
    }
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stDecoration"] { display: none; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Función para aplicar tema a cualquier figura ─────────────
def tema(fig, height=400, margin=None, extra=None):
    """Aplica el tema oscuro a una figura Plotly sin conflictos de keys."""
    if margin is None:
        margin = dict(t=24, b=16, l=16, r=16)
    layout_args = dict(
        paper_bgcolor='#161b27',
        plot_bgcolor='#0f1117',
        font=dict(family='DM Sans', color='#94a3b8', size=11),

        legend=dict(bgcolor='#161b27', bordercolor='#1e2535', borderwidth=1),
        height=height,
        margin=margin,
    )
    if extra:
        layout_args.update(extra)
    fig.update_layout(**layout_args)
    fig.update_xaxes(gridcolor='#1e2535', linecolor='#1e2535', tickcolor='#475569')
    fig.update_yaxes(gridcolor='#1e2535', linecolor='#1e2535', tickcolor='#475569')
    return fig

# ── Constantes de color ──────────────────────────────────────
C_DEMANDA = '#ef4444'
C_RECURSO = '#22c55e'
C_ACCENT  = '#3b82f6'

# ── Carga de datos ───────────────────────────────────────────
@st.cache_data
def cargar_datos():
    return pd.read_excel('bienestar_procesado.xlsx')

df = cargar_datos()

# ── Diccionarios ─────────────────────────────────────────────
DIMS = {
    'CTRL': 'Control del trabajo',
    'PT':   'Presión del tiempo',
    'CL':   'Compromiso del líder',
    'AC':   'Apoyo de compañeros',
    'CR':   'Claridad de rol',
    'CoR':  'Conflicto de rol',
    'GC':   'Gestión del cambio',
    'SM':   'Salud mental org.',
    'SAT':  'Satisfacción / Engagement',
    'IR':   'Intención de retiro',
    'FT':   'Conflicto F→T',
    'TF':   'Conflicto T→F',
    'BU':   'Burnout / Agotamiento',
    'BP':   'Bienestar percibido',
    'SOM':  'Somatización',
    'DL':   'Desgaste laboral',
}

ESCALA_MAX = {
    'DIM_SAT': 7, 'DIM_IR': 7, 'DIM_TF': 7, 'DIM_FT': 7,
    'DIM_BP':  7, 'DIM_SOM': 7, 'DIM_DL': 7,
}

DEMANDA = ['PT', 'CoR', 'FT', 'TF', 'BU', 'SOM', 'DL', 'IR']
RECURSO = ['CTRL', 'CL', 'AC', 'CR', 'GC', 'SM', 'SAT', 'BP']

PARES_CLAVE = [
    ('DIM_BU',  'DIM_DL',  'Burnout vs Desgaste laboral',
     'Ambas dimensiones son manifestaciones del mismo proceso de agotamiento crónico.'),
    ('DIM_BU',  'DIM_SOM', 'Burnout vs Somatización',
     'El agotamiento emocional se traduce en síntomas físicos.'),
    ('DIM_CL',  'DIM_SAT', 'Compromiso del líder vs Satisfacción',
     'El liderazgo comprometido es predictor directo del engagement.'),
    ('DIM_CL',  'DIM_BU',  'Compromiso del líder vs Burnout',
     'El líder actúa como amortiguador del burnout — la intervención de mayor retorno.'),
    ('DIM_SAT', 'DIM_IR',  'Satisfacción vs Intención de retiro',
     'La correlación más fuerte del dataset. Mayor satisfacción = menor rotación.'),
]

def escala_max_dim(col):
    return ESCALA_MAX.get(col, 5)

def normalizar(valor, col):
    return (valor - 1) / (escala_max_dim(col) - 1)

def nivel_riesgo_demanda(val, col):
    pct = (val - 1) / (escala_max_dim(col) - 1)
    if pct >= 0.6: return 'riesgo', 'badge-riesgo', 'Alto riesgo'
    if pct >= 0.4: return 'alerta', 'badge-alerta', 'Riesgo moderado'
    return 'ok', 'badge-ok', 'Riesgo bajo'

def nivel_recurso(val, col):
    pct = (val - 1) / (escala_max_dim(col) - 1)
    if pct >= 0.6: return 'ok',     'badge-ok',     'Favorable'
    if pct >= 0.4: return 'alerta', 'badge-alerta', 'Mejorable'
    return 'riesgo', 'badge-riesgo', 'Deficiente'

def kpi_html(label, valor_str, clase_card, clase_badge, texto_badge):
    return f"""
    <div class='kpi-card {clase_card}'>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-valor'>{valor_str}</div>
        <span class='kpi-badge {clase_badge}'>{texto_badge}</span>
    </div>
    """

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='margin-bottom:4px;font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;color:#475569;'>Estudio EDA 2026-1</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:1.05rem;font-weight:600;color:#e2e8f0;margin-bottom:20px;'>Riesgos Psicosociales<br>Laborales</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:#1e2535;margin-bottom:20px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='seccion-header'>Filtros de población</div>", unsafe_allow_html=True)

    sexo_opts   = ['Todos'] + sorted(df['Sexo'].dropna().unique().tolist())
    cargo_opts  = ['Todos'] + sorted(df['Tipo_Cargo'].dropna().unique().tolist())
    sector_opts = ['Todos'] + sorted(df['Sector'].dropna().unique().tolist())
    modal_opts  = ['Todos'] + sorted(df['Modalidad'].dropna().unique().tolist())

    sel_sexo   = st.selectbox("Sexo",          sexo_opts)
    sel_cargo  = st.selectbox("Tipo de cargo", cargo_opts)
    sel_sector = st.selectbox("Sector",        sector_opts)
    sel_modal  = st.selectbox("Modalidad",     modal_opts)

    dff = df.copy()
    if sel_sexo   != 'Todos': dff = dff[dff['Sexo']       == sel_sexo]
    if sel_cargo  != 'Todos': dff = dff[dff['Tipo_Cargo'] == sel_cargo]
    if sel_sector != 'Todos': dff = dff[dff['Sector']     == sel_sector]
    if sel_modal  != 'Todos': dff = dff[dff['Modalidad']  == sel_modal]

    n = len(dff)
    st.markdown("<div style='height:1px;background:#1e2535;margin:20px 0;'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-family:DM Mono,monospace;font-size:1.4rem;font-weight:500;color:#e2e8f0;'>{n}</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.72rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;'>trabajadores seleccionados</div>", unsafe_allow_html=True)

    filtros_activos = []
    if sel_sexo   != 'Todos': filtros_activos.append(f'Sexo: {sel_sexo}')
    if sel_cargo  != 'Todos': filtros_activos.append(f'Cargo: {sel_cargo}')
    if sel_sector != 'Todos': filtros_activos.append(f'Sector: {sel_sector}')
    if sel_modal  != 'Todos': filtros_activos.append(f'Modalidad: {sel_modal}')
    for f in filtros_activos:
        st.markdown(f"<div style='font-size:0.75rem;color:#64748b;padding:2px 0;'>— {f}</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1px;background:#1e2535;margin:20px 0;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.68rem;color:#334155;'>Grupo 6 · Facultad de Ingeniería</div>", unsafe_allow_html=True)

# ── Cabecera ─────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:24px;'>
    <div style='font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;color:#475569;margin-bottom:6px;'>
        Análisis Exploratorio de Datos
    </div>
    <div style='font-size:1.6rem;font-weight:600;color:#e2e8f0;'>
        Dashboard de Bienestar Laboral
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Resumen", "Perfil de muestra", "Dimensiones", "Correlaciones", "Grupos"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — RESUMEN
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='seccion-header'>Indicadores clave del grupo seleccionado</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if 'DIM_IR' in dff.columns:
            v = dff['DIM_IR'].mean()
            cl, ba, tx = nivel_riesgo_demanda(v, 'DIM_IR')
            st.markdown(kpi_html("Intención de retiro", f"{v:.2f} / 7", cl, ba, tx), unsafe_allow_html=True)
    with c2:
        if 'DIM_BU' in dff.columns:
            v = dff['DIM_BU'].mean()
            cl, ba, tx = nivel_riesgo_demanda(v, 'DIM_BU')
            st.markdown(kpi_html("Burnout", f"{v:.2f} / 5", cl, ba, tx), unsafe_allow_html=True)
    with c3:
        if 'DIM_SAT' in dff.columns:
            v = dff['DIM_SAT'].mean()
            cl, ba, tx = nivel_recurso(v, 'DIM_SAT')
            st.markdown(kpi_html("Satisfacción", f"{v:.2f} / 7", cl, ba, tx), unsafe_allow_html=True)
    with c4:
        if 'DIM_CL' in dff.columns:
            v = dff['DIM_CL'].mean()
            cl, ba, tx = nivel_recurso(v, 'DIM_CL')
            st.markdown(kpi_html("Compromiso del líder", f"{v:.2f} / 5", cl, ba, tx), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='seccion-header'>Perfil psicosocial completo — 16 dimensiones</div>", unsafe_allow_html=True)

    dims_radar = [f'DIM_{d}' for d in DIMS if f'DIM_{d}' in dff.columns]
    nombres_radar = [DIMS[c.replace('DIM_', '')] for c in dims_radar]
    valores_norm  = [normalizar(dff[c].mean(), c) for c in dims_radar]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=valores_norm + [valores_norm[0]],
        theta=nombres_radar + [nombres_radar[0]],
        fill='toself',
        fillcolor='rgba(59,130,246,0.1)',
        line=dict(color=C_ACCENT, width=1.5),
    ))
    fig_radar.update_layout(
        paper_bgcolor='#161b27',
        plot_bgcolor='#0f1117',
        font=dict(family='DM Sans', color='#94a3b8', size=11),
        polar=dict(
            bgcolor='#0f1117',
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickvals=[0.25, 0.5, 0.75],
                ticktext=['25 %', '50 %', '75 %'],
                gridcolor='#1e2535', linecolor='#1e2535',
                tickfont=dict(size=9, color='#475569')
            ),
            angularaxis=dict(gridcolor='#1e2535', linecolor='#1e2535')
        ),
        height=480, showlegend=False,
        margin=dict(t=20, b=20, l=40, r=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown("""
    <div class='insight-box'>
        Valores normalizados entre 0 y 1 para comparar dimensiones de distinta escala.
        Para <strong>demandas</strong>, valores altos indican mayor riesgo.
        Para <strong>recursos</strong>, valores altos indican mayor protección.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — PERFIL DE MUESTRA
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='seccion-header'>Distribución sociodemográfica y laboral</div>", unsafe_allow_html=True)

    vars_demo = [('Sexo','Sexo'),('Tipo_Cargo','Tipo de cargo'),('Sector','Sector'),('Modalidad','Modalidad')]
    col_a, col_b = st.columns(2)
    col_c, col_d = st.columns(2)
    cols_demo = [col_a, col_b, col_c, col_d]

    for col_ui, (var, titulo) in zip(cols_demo, vars_demo):
        with col_ui:
            conteo = dff[var].value_counts().reset_index()
            conteo.columns = [var, 'n']
            fig = px.bar(conteo, x='n', y=var, orientation='h', text='n',
                         color_discrete_sequence=[C_ACCENT],
                         labels={'n': 'Trabajadores', var: ''})
            fig.update_traces(textposition='outside', textfont_size=10, marker_opacity=0.85)
            tema(fig, height=220, margin=dict(t=36, b=8, l=8, r=30),
                 extra=dict(title=dict(text=titulo, font=dict(size=12, color='#e2e8f0', family='DM Sans')), showlegend=False))
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig, use_container_width=True)

    if 'Horas_Semana' in dff.columns:
        st.markdown("<div class='seccion-header' style='margin-top:8px;'>Jornada laboral — horas por semana</div>", unsafe_allow_html=True)
        fig_h = px.histogram(dff, x='Horas_Semana', nbins=20,
                             color_discrete_sequence=[C_ACCENT],
                             labels={'Horas_Semana': 'Horas por semana', 'count': 'Trabajadores'})
        media_h = dff['Horas_Semana'].mean()
        fig_h.add_vline(x=media_h, line_dash='dot', line_color='#f59e0b',
                        annotation_text=f'Media: {media_h:.1f} h',
                        annotation_font_color='#f59e0b', annotation_position='top right')
        tema(fig_h, height=260, margin=dict(t=16, b=16, l=8, r=8),
             extra=dict(bargap=0.05))
        st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("""
    <div class='insight-box'>
        <strong>Perfil predominante:</strong> mujer del sector público, con nivel educativo profesional o de posgrado,
        en modalidad presencial, cargo administrativo, con jornadas de 40 a 50 horas semanales.
        La baja representación de modalidad híbrida (≈15 %) limita la generalización de comparaciones por modalidad.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — DIMENSIONES
# ══════════════════════════════════════════════════════════════
with tab3:
    # Construir ranking
    filas = []
    for dim in DEMANDA:
        col = f'DIM_{dim}'
        if col in dff.columns:
            media = dff[col].mean()
            filas.append({'Dimensión': dim, 'Nombre': DIMS[dim],
                          'Tipo': 'Demanda', 'Media': round(media, 3), 'Riesgo': round(media, 3)})
    for dim in RECURSO:
        col = f'DIM_{dim}'
        if col in dff.columns:
            em    = escala_max_dim(col)
            media = dff[col].mean()
            filas.append({'Dimensión': dim, 'Nombre': DIMS[dim],
                          'Tipo': 'Recurso', 'Media': round(media, 3), 'Riesgo': round(em - media, 3)})
    ranking = (pd.DataFrame(filas)
               .sort_values('Riesgo', ascending=False)
               .reset_index(drop=True))
    ranking.index += 1

    st.markdown("<div class='seccion-header'>Ranking de dimensiones por nivel de riesgo</div>", unsafe_allow_html=True)

    fig_rank = px.bar(
        ranking, x='Riesgo', y='Nombre', color='Tipo',
        color_discrete_map={'Demanda': C_DEMANDA, 'Recurso': C_RECURSO},
        orientation='h', text='Riesgo',
        labels={'Riesgo': 'Nivel de riesgo', 'Nombre': '', 'Tipo': ''}
    )
    fig_rank.update_traces(texttemplate='%{text:.2f}', textposition='outside',
                           textfont_size=10, marker_opacity=0.85)
    tema(fig_rank, height=500, margin=dict(t=8, b=8, l=8, r=40),
         extra=dict(legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0)))
    fig_rank.update_yaxes(categoryorder='total ascending')
    fig_rank.update_xaxes(showgrid=True)
    st.plotly_chart(fig_rank, use_container_width=True)

    st.markdown("""
    <div class='insight-box'>
        <strong>Demandas:</strong> mayor puntaje = mayor riesgo directo.
        <strong>Recursos:</strong> mayor puntaje = recurso más debilitado.
        Los tres focos de intervención prioritaria son Intención de retiro, Conflicto T→F Salud Mental Organizacionl y Control del trabajo.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='seccion-header' style='margin-top:24px;'>Distribución individual por dimensión</div>", unsafe_allow_html=True)

    dim_sel = st.selectbox("Selecciona una dimensión:",
                           options=list(DIMS.keys()),
                           format_func=lambda x: f"{x}  —  {DIMS[x]}")

    col_dim = f'DIM_{dim_sel}'
    if col_dim in dff.columns:
        c1, c2 = st.columns([3, 1])
        with c1:
            color_hist = C_DEMANDA if dim_sel in DEMANDA else C_RECURSO
            fig_dist = px.histogram(dff, x=col_dim, nbins=20,
                                    color_discrete_sequence=[color_hist],
                                    labels={col_dim: 'Puntaje promedio', 'count': 'Trabajadores'})
            media_d   = dff[col_dim].mean()
            mediana_d = dff[col_dim].median()
            fig_dist.add_vline(x=media_d, line_dash='dot', line_color='#f59e0b',
                               annotation_text=f'Media {media_d:.2f}',
                               annotation_font_color='#f59e0b', annotation_position='top right')
            fig_dist.add_vline(x=mediana_d, line_dash='dash', line_color='#94a3b8',
                               annotation_text=f'Mediana {mediana_d:.2f}',
                               annotation_font_color='#94a3b8', annotation_position='top left')
            tema(fig_dist, height=320, margin=dict(t=16, b=16, l=8, r=8),
                 extra=dict(bargap=0.06, showlegend=False))
            st.plotly_chart(fig_dist, use_container_width=True)

        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            tipo_dim   = "Demanda" if dim_sel in DEMANDA else "Recurso"
            tipo_color = C_DEMANDA if dim_sel in DEMANDA else C_RECURSO
            st.markdown(f"""
            <div class='kpi-card neutro' style='border-left-color:{tipo_color};margin-bottom:12px;'>
                <div class='kpi-label'>Tipo</div>
                <div style='font-size:0.95rem;font-weight:600;color:#e2e8f0;'>{tipo_dim}</div>
            </div>
            """, unsafe_allow_html=True)
            stats_d = dff[col_dim].describe()
            for etiqueta, valor in [
                ('Media',      f"{stats_d['mean']:.3f}"),
                ('Mediana',    f"{mediana_d:.3f}"),
                ('Desv. est.', f"{stats_d['std']:.3f}"),
                ('Mín',        f"{stats_d['min']:.3f}"),
                ('Máx',        f"{stats_d['max']:.3f}"),
                ('N',          f"{int(stats_d['count'])}"),
            ]:
                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;padding:5px 0;
                            border-bottom:1px solid #1e2535;font-size:0.82rem;'>
                    <span style='color:#64748b;'>{etiqueta}</span>
                    <span style='font-family:DM Mono,monospace;color:#e2e8f0;'>{valor}</span>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — CORRELACIONES
# ══════════════════════════════════════════════════════════════
with tab4:
    dims_cols       = [f'DIM_{d}' for d in DIMS if f'DIM_{d}' in dff.columns]
    dim_labels_short = [c.replace('DIM_', '') for c in dims_cols]

    corr_mat         = dff[dims_cols].corr(method='spearman')
    corr_mat.index   = [DIMS[l] for l in dim_labels_short]
    corr_mat.columns = [DIMS[l] for l in dim_labels_short]

    st.markdown("<div class='seccion-header'>Matriz de correlaciones de Spearman — 16 dimensiones</div>", unsafe_allow_html=True)

    fig_heat = px.imshow(corr_mat, text_auto='.2f', aspect='auto',
                         color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                         labels=dict(color='rho'))
    fig_heat.update_traces(textfont_size=8)
    tema(fig_heat, height=540, margin=dict(t=8, b=8, l=8, r=8))
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("""
    <div class='insight-box'>
        Se usó Spearman porque los datos provienen de escalas ordinales (Likert) con distribuciones no normales.
        Correlaciones más relevantes: <strong>SAT ↔ IR (rho = -0.83)</strong>,
        <strong>CTRL ↔ PT (rho = -0.78)</strong>, <strong>CL ↔ GC (rho = +0.70)</strong>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='seccion-header' style='margin-top:24px;'>Análisis de pares clave</div>", unsafe_allow_html=True)

    par_sel_idx = st.selectbox("Selecciona un par:",
                               options=range(len(PARES_CLAVE)),
                               format_func=lambda i: PARES_CLAVE[i][2])

    x_col, y_col, titulo_par, interpretacion = PARES_CLAVE[par_sel_idx]

    if x_col in dff.columns and y_col in dff.columns:
        par_df = dff[[x_col, y_col]].dropna()
        rho_par, pval_par = spearmanr(par_df[x_col], par_df[y_col])
        x_label = DIMS[x_col.replace('DIM_', '')]
        y_label = DIMS[y_col.replace('DIM_', '')]

        c1, c2 = st.columns([3, 1])
        with c1:
            color_pts = C_DEMANDA if rho_par < 0 else C_ACCENT
            fig_par = go.Figure()
            fig_par.add_trace(go.Scatter(
                x=par_df[x_col], y=par_df[y_col],
                mode='markers',
                marker=dict(color=color_pts, opacity=0.4, size=5),
                name='Trabajadores'
            ))
            z    = np.polyfit(par_df[x_col], par_df[y_col], 1)
            p_fn = np.poly1d(z)
            xs   = np.linspace(par_df[x_col].min(), par_df[x_col].max(), 100)
            fig_par.add_trace(go.Scatter(
                x=xs, y=p_fn(xs), mode='lines',
                line=dict(color='#f59e0b', dash='dash', width=1.8),
                name='Tendencia'
            ))
            tema(fig_par, height=420, margin=dict(t=70, b=16, l=16, r=16),
                 extra=dict(
                     title=dict(
                         text=f"{titulo_par}<br><sup>ρ = {rho_par:.3f}   |   p = {pval_par:.4f}</sup>",
                         font=dict(size=14, color='#e2e8f0', family='DM Sans'),
                         y=0.97,
                     ),
                     xaxis_title=x_label,
                     yaxis_title=y_label,
                     legend=dict(orientation='h', yanchor='bottom', y=1.08),
                 ))
            st.plotly_chart(fig_par, use_container_width=True)

        with c2:
            abs_rho    = abs(rho_par)
            intensidad = "Fuerte" if abs_rho >= 0.7 else "Moderada" if abs_rho >= 0.4 else "Débil"
            direccion  = "Positiva" if rho_par > 0 else "Negativa"
            sig_txt    = "Significativa" if pval_par < 0.05 else "No significativa"
            st.markdown("<br>", unsafe_allow_html=True)
            for etiqueta, valor in [
                ('rho de Spearman', f"{rho_par:.3f}"),
                ('p-valor',         f"{pval_par:.4f}"),
                ('N',               f"{len(par_df)}"),
                ('Dirección',       direccion),
                ('Intensidad',      intensidad),
                ('Significancia',   sig_txt),
            ]:
                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;padding:6px 0;
                            border-bottom:1px solid #1e2535;font-size:0.82rem;'>
                    <span style='color:#64748b;'>{etiqueta}</span>
                    <span style='font-family:DM Mono,monospace;color:#e2e8f0;'>{valor}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class='insight-box' style='margin-top:16px;font-size:0.82rem;'>
                {interpretacion}
                <br><span style='color:#475569;font-size:0.75rem;'>Correlación no implica causalidad.</span>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 5 — GRUPOS
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='seccion-header'>Diferencias entre grupos — prueba de Kruskal-Wallis</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        dim_box = st.selectbox("Dimensión a comparar", list(DIMS.keys()),
                               format_func=lambda x: f"{x}  —  {DIMS[x]}", key='box_dim')
    with c2:
        grupo_box = st.selectbox("Agrupar por",
                                 ['Tipo_Cargo', 'Sector', 'Modalidad',
                                  'Personas_Cargo', 'Sexo', 'Nivel_Educativo'],
                                 key='box_grupo')

    col_box = f'DIM_{dim_box}'
    if col_box in dff.columns:
        data_box = dff[[col_box, grupo_box]].dropna()
        data_box = data_box.rename(columns={col_box: DIMS[dim_box]})

        fig_viol = px.violin(data_box, x=grupo_box, y=DIMS[dim_box],
                             box=True, color=grupo_box,
                             color_discrete_sequence=px.colors.qualitative.Set3,
                             labels={DIMS[dim_box]: 'Puntaje', grupo_box: ''})
        tema(fig_viol, height=420, margin=dict(t=16, b=16, l=8, r=8),
             extra=dict(showlegend=False))
        st.plotly_chart(fig_viol, use_container_width=True)

        grupos_vals = [g[DIMS[dim_box]].values
                       for _, g in data_box.groupby(grupo_box) if len(g) > 1]
        if len(grupos_vals) >= 2:
            stat_kw, pval_kw = kruskal(*grupos_vals)
            sig          = pval_kw < 0.05
            color_sig    = '#ef4444' if sig else '#22c55e'
            texto_sig    = "Diferencia significativa" if sig else "Sin diferencia significativa entre grupos"
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:12px;background:#161b27;
                        border:1px solid #1e2535;border-radius:8px;padding:12px 18px;margin-top:8px;'>
                <div style='font-family:DM Mono,monospace;font-size:0.85rem;color:#94a3b8;'>Kruskal-Wallis</div>
                <div style='font-family:DM Mono,monospace;font-size:0.85rem;color:#e2e8f0;'>H = {stat_kw:.2f}</div>
                <div style='font-family:DM Mono,monospace;font-size:0.85rem;color:#e2e8f0;'>p = {pval_kw:.4f}</div>
                <div style='font-size:0.78rem;font-weight:600;color:{color_sig};
                            background:rgba(0,0,0,0.2);padding:3px 10px;border-radius:4px;'>{texto_sig}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class='insight-box' style='margin-top:20px;'>
        <strong>Hallazgo central:</strong> ninguna comparación por grupo alcanzó significancia estadística
        (todos los p > 0.05). El riesgo psicosocial es <strong>transversal a toda la organización</strong> —
        las intervenciones deben ser sistémicas, no focalizadas en un cargo o modalidad particular.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<div style='margin-top:40px;padding-top:16px;border-top:1px solid #1e2535;
            display:flex;justify-content:space-between;align-items:center;
            font-size:0.72rem;color:#334155;'>
    <span>Proyecto Final EDA 2026-1 · Análisis de Riesgos Psicosociales Laborales</span>
    <span>Sara Montañez · Santiago Parada · David Peralta · Jorge Díaz · Grupo 6</span>
</div>
""", unsafe_allow_html=True)
