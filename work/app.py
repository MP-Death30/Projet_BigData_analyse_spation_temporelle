
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import numpy as np
from math import radians, cos, sin, asin, sqrt

st.set_page_config(layout="wide", page_title="NYC Environmental Dashboard")

@st.cache_data
def load_data():
    geo = gpd.read_file("dashboard_map.geojson")
    air = pd.read_parquet("dashboard_data_air.parquet")
    weather = pd.read_parquet("dashboard_data_weather.parquet")
    
    air['DATE_OBSERVATION'] = pd.to_datetime(air['DATE_OBSERVATION'])
    weather['DATE'] = pd.to_datetime(weather['DATE'])
    
    stations = weather[['ID_STATION', 'NAME', 'LATITUDE', 'LONGITUDE']].drop_duplicates()
    return geo, air, weather, stations

def haversine_dist(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1) * cos(lat2) * sin((lon2-lon1)/2)**2
    return 6371 * 2 * asin(sqrt(a))

def get_weighted_weather(target_lat, target_lon, weather_df, station_ids):
    subset = weather_df[weather_df['ID_STATION'].isin(station_ids)].copy()
    if subset.empty: return None
    subset['dist'] = subset.apply(lambda x: haversine_dist(target_lon, target_lat, x['LONGITUDE'], x['LATITUDE']), axis=1)
    subset['weight'] = 1 / (subset['dist'] + 0.1)
    try:
        return np.average(subset['TEMP'], weights=subset['weight'])
    except:
        return None

geo, df_air, df_weather, df_stations = load_data()

st.sidebar.title("Filtres")
dates = st.sidebar.date_input("Période", [df_air['DATE_OBSERVATION'].min(), df_air['DATE_OBSERVATION'].max()])
polluant = st.sidebar.selectbox("Polluant", df_air['NOM_POLLUANT'].unique())

mask_air = (df_air['DATE_OBSERVATION'].dt.date >= dates[0]) & (df_air['DATE_OBSERVATION'].dt.date <= dates[1])
df_air_view = df_air[mask_air & (df_air['NOM_POLLUANT'] == polluant)]

col_map, col_stats = st.columns([3, 2])

with col_map:
    st.subheader("Carte des Quartiers")
    m = folium.Map([40.7, -74.0], zoom_start=10)
    folium.GeoJson(
        geo, 
        name="Quartiers",
        tooltip=folium.GeoJsonTooltip(fields=['GEONAME', 'BOROUGH']),
        style_function=lambda x: {'fillColor': '#3388ff', 'color': 'black', 'weight': 0.5, 'fillOpacity': 0.4},
        highlight_function=lambda x: {'weight': 3, 'color': 'red'}
    ).add_to(m)
    st_map = st_folium(m, width=None, height=600)

selected_geocode = None
selected_name = st.sidebar.selectbox("Ou choisir un quartier :", geo['GEONAME'].unique())
q_data = geo[geo['GEONAME'] == selected_name].iloc[0]
selected_geocode = q_data['GEOCODE']
centroid = q_data.geometry.centroid
lat_q, lon_q = centroid.y, centroid.x

with col_stats:
    st.title(selected_name)
    st.info(f"Borough: {q_data['BOROUGH']}")
    
    st.markdown("### 1. Qualité de l'Air")
    local_air = df_air_view[df_air_view['GEOJOIN_ID'] == selected_geocode]
    if not local_air.empty:
        st.metric(f"Moyenne {polluant}", f"{local_air['VALEUR'].mean():.2f}")
        fig = px.bar(local_air, x='DATE_OBSERVATION', y='VALEUR', title="Évolution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Pas de données.")

    st.markdown("### 2. Météo Pondérée")
    df_stations['dist'] = df_stations.apply(lambda x: haversine_dist(lon_q, lat_q, x['LONGITUDE'], x['LATITUDE']), axis=1)
    df_stations = df_stations.sort_values('dist')
    sel_stations = st.multiselect("Stations", df_stations['ID_STATION'].tolist(), format_func=lambda x: f"{df_stations[df_stations.ID_STATION==x].iloc[0]['NAME']} ({df_stations[df_stations.ID_STATION==x].iloc[0]['dist']:.1f}km)")
    
    if sel_stations:
        mask_w = (df_weather['DATE'].dt.date >= dates[0]) & (df_weather['DATE'].dt.date <= dates[1])
        w_temp = get_weighted_weather(lat_q, lon_q, df_weather[mask_w], sel_stations)
        if w_temp: st.metric("Température Pondérée", f"{w_temp:.1f} °F")
