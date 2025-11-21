import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import signal
import time

# ==============================================================================
# 1. CONFIGURATION & INITIALISATION
# ==============================================================================
st.set_page_config(layout="wide", page_title="NYC Environmental Dashboard")

# Initialisation des états
if 'selected_geocode' not in st.session_state:
    st.session_state.selected_geocode = None
if 'dropdown_selector' not in st.session_state:
    st.session_state.dropdown_selector = "Tous quartiers"

# ==============================================================================
# 2. FONCTIONS UTILITAIRES
# ==============================================================================

@st.cache_data
def load_data():
    geo = gpd.read_file("dashboard_map.geojson")
    air = pd.read_parquet("dashboard_data_air.parquet")
    weather = pd.read_parquet("dashboard_data_weather.parquet")
    
    # --- NETTOYAGE & CONVERSIONS ---
    geo['GEOCODE'] = geo['GEOCODE'].astype(str)
    
    if 'LATITUDE_ZONE' not in geo.columns:
        try:
            geo_temp = geo.to_crs(epsg=2263)
            centroids = geo_temp.geometry.centroid.to_crs(epsg=4326)
        except:
            centroids = geo.geometry.centroid
        geo['LATITUDE_ZONE'] = centroids.y
        geo['LONGITUDE_ZONE'] = centroids.x
    
    air['DATE_OBSERVATION'] = pd.to_datetime(air['DATE_OBSERVATION'])
    weather['DATE'] = pd.to_datetime(weather['DATE'])
    
    # --- CONVERSION UNITÉS (Impérial -> Métrique) ---
    weather['TEMP'] = (weather['TEMP'] - 32) * 5.0/9.0
    weather['DEWP'] = (weather['DEWP'] - 32) * 5.0/9.0
    weather['WDSP'] = weather['WDSP'] * 1.852

    # --- FILTRAGE VALEURS EXTRÊMES ---
    weather = weather[weather['WDSP'] <= 150]
    weather = weather[weather['DEWP'] <= 40]
    
    stations = weather[['ID_STATION', 'NAME', 'LATITUDE', 'LONGITUDE']].drop_duplicates()
    return geo, air, weather, stations

def haversine_vectorized(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c 

def calculate_global_metrics(geo_df, weather_df, stations_df, radius_km):
    results = []
    weather_agg = weather_df.groupby('ID_STATION')[['TEMP', 'WDSP', 'DEWP']].mean().reset_index()
    stations_w_weather = stations_df.merge(weather_agg, on='ID_STATION')
    
    if stations_w_weather.empty:
        return pd.DataFrame()

    for _, row in geo_df.iterrows():
        lat_q, lon_q = row['LATITUDE_ZONE'], row['LONGITUDE_ZONE']
        dists = haversine_vectorized(
            lon_q, lat_q, 
            stations_w_weather['LONGITUDE'].values, 
            stations_w_weather['LATITUDE'].values
        )
        
        mask = dists <= radius_km
        nearby_stations = stations_w_weather[mask].copy()
        nearby_dists = dists[mask]
        
        if not nearby_stations.empty:
            weights = 1 / (nearby_dists + 0.1)
            w_temp = np.average(nearby_stations['TEMP'], weights=weights)
            w_wind = np.average(nearby_stations['WDSP'], weights=weights)
            w_dewp = np.average(nearby_stations['DEWP'], weights=weights)
            
            results.append({
                'GEOCODE': str(row['GEOCODE']),
                'W_TEMP': round(w_temp, 1),
                'W_WIND': round(w_wind, 1),
                'W_DEWP': round(w_dewp, 1),
                'NB_STATIONS': len(nearby_stations)
            })
        else:
            results.append({
                'GEOCODE': str(row['GEOCODE']),
                'W_TEMP': None, 'W_WIND': None, 'W_DEWP': None, 'NB_STATIONS': 0
            })
    return pd.DataFrame(results)

# ==============================================================================
# 3. CHARGEMENT & FILTRES
# ==============================================================================

geo, df_air, df_weather, df_stations = load_data()

# --- BOUTON D'ARRÊT (NOUVEAU) ---
st.sidebar.header("⚙️ Contrôle")
if st.sidebar.button("🛑 Arrêter le Dashboard"):
    st.sidebar.warning("Arrêt du serveur en cours...")
    time.sleep(1)
    os.kill(os.getpid(), signal.SIGTERM)

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Filtres & Paramètres")

# Dates
min_date, max_date = df_air['DATE_OBSERVATION'].min(), df_air['DATE_OBSERVATION'].max()
start_date, end_date = st.sidebar.date_input(
    "Période d'analyse", [min_date, max_date], min_value=min_date, max_value=max_date
)

# Filtre Polluants
mask_air_date = (df_air['DATE_OBSERVATION'].dt.date >= start_date) & (df_air['DATE_OBSERVATION'].dt.date <= end_date)
df_air_filtered = df_air[mask_air_date]

valid_pollutants = df_air_filtered[df_air_filtered['VALEUR'].notna()]['NOM_POLLUANT'].unique()
valid_pollutants = sorted(valid_pollutants)

if len(valid_pollutants) > 0:
    selected_polluant = st.sidebar.selectbox("Polluant (Dispo sur la période)", valid_pollutants)
else:
    st.sidebar.error("⚠️ Aucune donnée de pollution pour cette période.")
    selected_polluant = None

# Autres Filtres
radius = st.sidebar.slider("Rayon des stations météo (km)", 1, 100, 15)
meteo_vars = ['Température', 'Vitesse Vent', 'Point de Rosée']
selected_meteo_vars = st.sidebar.multiselect("Graphiques Météo (Comparaison & Boxplots)", meteo_vars, default=['Température'])

# ==============================================================================
# 4. ETL A LA VOLÉE (PRÉPARATION GLOBALE)
# ==============================================================================

if selected_polluant is None:
    st.warning("Veuillez élargir la plage de dates.")
    st.stop()

# Filtre Météo
mask_weather_date = (df_weather['DATE'].dt.date >= start_date) & (df_weather['DATE'].dt.date <= end_date)
df_weather_filtered = df_weather[mask_weather_date]

# --- INDICATEUR SIDEBAR ---
with st.sidebar:
    st.markdown("---")
    st.markdown("### ℹ️ Info Stations")
    active_stations = df_weather_filtered[['ID_STATION']].drop_duplicates()
    active_stations_coords = active_stations.merge(df_stations, on='ID_STATION')

    if st.session_state.selected_geocode is None:
        lat_center = geo['LATITUDE_ZONE'].mean()
        lon_center = geo['LONGITUDE_ZONE'].mean()
        if not active_stations_coords.empty:
            dists_s = haversine_vectorized(lon_center, lat_center, active_stations_coords['LONGITUDE'].values, active_stations_coords['LATITUDE'].values)
            nb_visible = np.sum(dists_s <= radius)
            st.metric(f"Stations (Centre NYC, {radius} km)", nb_visible)
        else:
            st.metric(f"Stations (Centre NYC, {radius} km)", 0)
    else:
        sel_geo = geo[geo['GEOCODE'] == st.session_state.selected_geocode]
        if not sel_geo.empty:
            lat_s = sel_geo.iloc[0]['LATITUDE_ZONE']
            lon_s = sel_geo.iloc[0]['LONGITUDE_ZONE']
            if not active_stations_coords.empty:
                dists_s = haversine_vectorized(lon_s, lat_s, active_stations_coords['LONGITUDE'].values, active_stations_coords['LATITUDE'].values)
                nb_visible = np.sum(dists_s <= radius)
                st.metric(f"Stations (Quartier, {radius} km)", nb_visible)
            else:
                st.metric(f"Stations (Quartier, {radius} km)", 0)

# --- PRÉPARATION DONNÉES CARTE ---
df_air_map = df_air_filtered[df_air_filtered['NOM_POLLUANT'] == selected_polluant]
if not df_air_map.empty:
    air_agg = df_air_map.groupby('GEOJOIN_ID')['VALEUR'].mean().reset_index()
    air_agg.columns = ['GEOCODE', 'MEAN_POLLUANT']
    air_agg['GEOCODE'] = air_agg['GEOCODE'].astype(str)
else:
    air_agg = pd.DataFrame(columns=['GEOCODE', 'MEAN_POLLUANT'])

weather_metrics_df = calculate_global_metrics(geo, df_weather_filtered, df_stations, radius)

gdf_display = geo.merge(air_agg, on='GEOCODE', how='left')
if not weather_metrics_df.empty:
    gdf_display = gdf_display.merge(weather_metrics_df, on='GEOCODE', how='left')

gdf_display['MEAN_POLLUANT'] = gdf_display['MEAN_POLLUANT'].fillna(0).round(2)
gdf_display['W_TEMP'] = gdf_display['W_TEMP'].fillna(0)
gdf_display['NB_STATIONS'] = gdf_display['NB_STATIONS'].fillna(0).astype(int)

# --- PRÉPARATION DONNÉES GRAPHIQUES & KPIs ---
current_title = ""
current_caption = ""
avg_polluant, avg_temp, avg_wind = 0, 0, 0

chart_air_src = pd.DataFrame()
chart_weather_src = pd.DataFrame()

if st.session_state.selected_geocode is None:
    # GLOBAL
    current_title = "New York City (Global)"
    current_caption = "Moyenne de tous les quartiers"
    
    valid_data = gdf_display[gdf_display['MEAN_POLLUANT'] > 0]
    if not valid_data.empty:
        avg_polluant = valid_data['MEAN_POLLUANT'].mean()
        avg_temp = valid_data['W_TEMP'].replace(0, np.nan).mean()
        avg_wind = valid_data['W_WIND'].replace(0, np.nan).mean()
    
    chart_air_src = df_air_filtered[df_air_filtered['NOM_POLLUANT'] == selected_polluant].copy()
    chart_weather_src = df_weather_filtered.copy()
else:
    # LOCAL
    current_geo_data = gdf_display[gdf_display['GEOCODE'] == st.session_state.selected_geocode].iloc[0]
    current_title = current_geo_data['GEONAME']
    current_caption = f"Borough: {current_geo_data['BOROUGH']} | Stations locales : {int(current_geo_data['NB_STATIONS'])}"
    
    avg_polluant = current_geo_data['MEAN_POLLUANT']
    avg_temp = current_geo_data['W_TEMP']
    avg_wind = current_geo_data['W_WIND']
    
    chart_air_src = df_air_filtered[
        (df_air_filtered['GEOJOIN_ID'] == st.session_state.selected_geocode) & 
        (df_air_filtered['NOM_POLLUANT'] == selected_polluant)
    ].copy()
    
    lat_q, lon_q = current_geo_data['LATITUDE_ZONE'], current_geo_data['LONGITUDE_ZONE']
    dists = haversine_vectorized(lon_q, lat_q, df_stations['LONGITUDE'].values, df_stations['LATITUDE'].values)
    nearby_ids = df_stations[dists <= radius]['ID_STATION'].unique()
    chart_weather_src = df_weather_filtered[df_weather_filtered['ID_STATION'].isin(nearby_ids)].copy()

# RESAMPLING
delta_days = (end_date - start_date).days
resample_rule = 'D'
if delta_days > 730: resample_rule = 'Q'
elif delta_days > 180: resample_rule = 'M'
elif delta_days > 60: resample_rule = 'W'

if not chart_air_src.empty:
    chart_air_final = chart_air_src.set_index('DATE_OBSERVATION').resample(resample_rule)['VALEUR'].mean().reset_index()
else:
    chart_air_final = pd.DataFrame()

if not chart_weather_src.empty:
    chart_weather_final = chart_weather_src.set_index('DATE').resample(resample_rule)[['TEMP', 'WDSP', 'DEWP']].mean().reset_index()
else:
    chart_weather_final = pd.DataFrame()

# ==============================================================================
# 5. UI PRINCIPALE (PARTIE HAUTE)
# ==============================================================================

col_map, col_details = st.columns([3, 2])

# --- COLONNE HAUT-GAUCHE : CARTE ---
with col_map:
    st.subheader(f"Carte : {selected_polluant}")
    
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="CartoDB positron")

    choropleth = folium.Choropleth(
        geo_data=gdf_display,
        data=gdf_display,
        columns=['GEOCODE', 'MEAN_POLLUANT'],
        key_on='feature.properties.GEOCODE',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f"Concentration {selected_polluant}",
        highlight=True
    )
    choropleth.add_to(m)

    style_function = lambda x: {'fillColor': '#ffffff', 'color':'#000000', 'fillOpacity': 0.0, 'weight': 0.1}
    tooltip_layer = folium.GeoJson(
        gdf_display,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['GEONAME', 'BOROUGH', 'MEAN_POLLUANT', 'W_TEMP', 'W_WIND', 'NB_STATIONS'],
            aliases=['Quartier:', 'Borough:', f'{selected_polluant}:', 'Temp (°C):', 'Vent (km/h):', 'Stations:'],
            localize=True
        )
    ).add_to(m)

    st_map = st_folium(m, width=None, height=650)
    
    # Synchro Clic
    geo_options_df = geo[['GEOCODE', 'GEONAME']].sort_values('GEONAME')
    if st_map and st_map.get('last_object_clicked'):
        last_clicked = st_map['last_object_clicked']
        if isinstance(last_clicked, dict) and 'properties' in last_clicked:
            props = last_clicked['properties']
            if props and 'GEOCODE' in props:
                clicked_code = str(props['GEOCODE'])
                name_match = geo_options_df[geo_options_df['GEOCODE'] == clicked_code]['GEONAME']
                if not name_match.empty:
                    clicked_name = name_match.values[0]
                    if st.session_state.dropdown_selector != clicked_name:
                        st.session_state.dropdown_selector = clicked_name
                        st.session_state.selected_geocode = clicked_code
                        st.rerun()

# --- COLONNE DROITE : DÉTAILS, KPIs ET TENDANCE ---
with col_details:
    # 1. Sélecteur & Titres
    st.markdown("### 📍 Détails")
    all_options = ["Tous quartiers"] + geo_options_df['GEONAME'].tolist()
    selected_option = st.selectbox("Sélectionner une zone", options=all_options, key="dropdown_selector")
    
    if selected_option == "Tous quartiers":
        st.session_state.selected_geocode = None
    else:
        code_match = geo_options_df[geo_options_df['GEONAME'] == selected_option]['GEOCODE']
        if not code_match.empty:
            st.session_state.selected_geocode = str(code_match.values[0])

    st.title(current_title)
    st.caption(current_caption)

    # 2. KPIs
    kpi1, kpi2, kpi3 = st.columns(3)
    val_p = f"{avg_polluant:.2f}" if pd.notnull(avg_polluant) else "N/A"
    val_t = f"{avg_temp:.1f} °C" if pd.notnull(avg_temp) else "N/A"
    val_w = f"{avg_wind:.1f} km/h" if pd.notnull(avg_wind) else "N/A"
    
    kpi1.metric(f"Moy. {selected_polluant}", val_p)
    kpi2.metric("Temp. Moy", val_t)
    kpi3.metric("Vent Moy", val_w)

    st.markdown("---")
    
    # 3. GRAPHIQUE TENDANCE (INTÉGRÉ ICI)
    st.subheader("📈 Analyses Temporelles")
    st.markdown(f"**Tendance : {selected_polluant}**")
    
    fig_main = go.Figure()
    if not chart_air_final.empty:
        fig_main.add_trace(go.Scatter(
            x=chart_air_final['DATE_OBSERVATION'], 
            y=chart_air_final['VALEUR'], 
            name=selected_polluant, 
            mode='lines+markers',
            marker=dict(size=8),
            line=dict(color='red', width=3)
        ))
        fig_main.update_layout(
            xaxis_title="Date", 
            yaxis=dict(title="Concentration"), 
            height=300, # Taille réduite pour bien tenir dans la colonne
            margin=dict(t=10, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_main, use_container_width=True)
    else:
        st.info("Pas de données suffisantes pour afficher l'évolution.")

# ==============================================================================
# 6. SECTION ANALYSE MÉTÉO (BAS DE PAGE)
# ==============================================================================
st.markdown("---")
st.subheader("☁️ Analyse Météo")

col_graphs, col_box = st.columns([2, 1])

meteo_config = {
    'Température': {'col': 'TEMP', 'color': 'orange', 'label': 'Temp (°C)'},
    'Vitesse Vent': {'col': 'WDSP', 'color': 'blue', 'label': 'Vent (km/h)'},
    'Point de Rosée': {'col': 'DEWP', 'color': 'green', 'label': 'Rosée (°C)'}
}

# --- COLONNE BAS-GAUCHE : GRAPHIQUES LINÉAIRES (CORRÉLATIONS) ---
with col_graphs:
    st.markdown("#### 📉 Corrélations")
    if not selected_meteo_vars:
        st.info("Sélectionnez des variables météo dans le menu.")
    else:
        for var_name in selected_meteo_vars:
            fig = go.Figure()
            # Air
            if not chart_air_final.empty:
                fig.add_trace(go.Scatter(
                    x=chart_air_final['DATE_OBSERVATION'], 
                    y=chart_air_final['VALEUR'], 
                    name=selected_polluant, 
                    mode='lines',
                    line=dict(color='red', width=1, dash='solid'),
                    opacity=0.5
                ))
            # Météo
            if not chart_weather_final.empty and var_name in meteo_config:
                conf = meteo_config[var_name]
                fig.add_trace(go.Scatter(
                    x=chart_weather_final['DATE'], 
                    y=chart_weather_final[conf['col']], 
                    name=conf['label'], 
                    mode='lines+markers',
                    marker=dict(size=4),
                    line=dict(color=conf['color'], width=2), 
                    yaxis='y2'
                ))

            fig.update_layout(
                title=f"{selected_polluant} vs {var_name}",
                xaxis_title="Date",
                yaxis=dict(title=selected_polluant, showgrid=False),
                yaxis2=dict(title=var_name, overlaying='y', side='right', showgrid=True),
                legend=dict(orientation="h", y=1.1),
                height=300, margin=dict(t=30, b=0, l=0, r=0)
            )
            st.plotly_chart(fig, use_container_width=True)

# --- COLONNE BAS-DROITE : BOXPLOTS (DISTRIBUTIONS) ---
with col_box:
    st.markdown("#### 📦 Distributions")
    if selected_meteo_vars and not chart_weather_src.empty:
        for var_name in selected_meteo_vars:
            if var_name in meteo_config:
                conf = meteo_config[var_name]
                
                fig_box = go.Figure()
                fig_box.add_trace(go.Box(
                    y=chart_weather_src[conf['col']],
                    name=conf['label'],
                    marker_color=conf['color']
                ))
                
                fig_box.update_layout(
                    title=f"{var_name}",
                    yaxis_title=conf['label'],
                    height=300,
                    margin=dict(t=30, b=20, l=0, r=0)
                )
                st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.write("Pas de données pour les distributions.")