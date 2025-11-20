"""
air_quality_pipeline_and_dash.py

Contenu :
1) Fonctions de pipeline pour :
   - Charger le dataset Air Quality (CSV local ou Socrata API)
   - Normaliser Start_Date au format YYYY-MM-DD
   - Charger GeoJSON des géographies (ex: UHF42, CD)
   - Calculer les centroïdes (LATITUDE, LONGITUDE)
   - Produire :
       - geo_raw.geojson  (polygones originaux)
       - geo_light.csv    (GEOCODE, GEONAME, BOROUGH, LATITUDE, LONGITUDE)
       - aq_with_geo.csv  (Air Quality joint avec GEOCODE + LAT/LON)

2) Une application Dash minimale (single-file) qui :
   - Affiche une carte interactive (dash-leaflet) des quartiers
   - Permet de sélectionner une zone (clic)
   - Montre les infos du quartier (nom, borough, qualité de l'air)
   - Permet de sélectionner une plage de dates et des éléments (ex: PM2.5, NO2)
   - Calcule une moyenne pondérée pour des stations choisies (selon distance)

INSTRUCTIONS GÉNÉRALES :
- Installer les dépendances (exemple) :
    pip install pandas geopandas shapely pyproj requests dash dash-leaflet geopy

- Préparer les fichiers :
    data/air_quality.csv    # export depuis NYC OpenData (ou utiliser Socrata)
    data/GEOS/u hf42.geojson  # ou tout autre geojson utilisé
    data/stations.csv       # (station_id, name, latitude, longitude, element, date, value)

Exécuter :
    python air_quality_pipeline_and_dash.py --prepare
    python air_quality_pipeline_and_dash.py --run-server

Remarque :
- Le script est conçu pour être modifié selon l'emplacement exact des colonnes de ton CSV.
- Le champ de jointure attendu dans air_quality.csv est "geo_join_id" (string ou int) ;
  le geojson doit contenir un champ nommé "GEOCODE" ou similaire (adaptable dans la fonction).

"""

import os
import argparse
from datetime import datetime
import json

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopy.distance import geodesic

# ---- CONFIG -------------------------------------------------
DATA_DIR = "data"
AIR_QUALITY_PATH = os.path.join(DATA_DIR, "air_quality.csv")
GEOJSON_PATH = os.path.join(DATA_DIR, "geo.json")  # remplacer par UHF42.geojson
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
STATIONS_PATH = os.path.join(DATA_DIR, "stations.csv")  # stations avec lat/lon

# noms de colonnes attendus -- adapter si nécessaire
COL_GEOJOIN = "geo_join_id"   # colonne dans air_quality.csv
COL_GEOCODE = "GEOCODE"      # champ dans geojson feature properties
COL_START = "Start_Date"     # colonne contenant la date de début

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ---- PARTIE PIPELINE ---------------------------------------

def normalize_start_date(df, col=COL_START):
    """Convertit la colonne date en format YYYY-MM-DD (string) et ajoute une colonne 'start_date_iso'."""
    df = df.copy()
    # essayer plusieurs formats si besoin
    df[col] = pd.to_datetime(df[col], errors='coerce')
    df['start_date_iso'] = df[col].dt.strftime('%Y-%m-%d')
    return df


def build_geo_light(geojson_path=GEOJSON_PATH, geocode_field=COL_GEOCODE):
    """Charge un geojson, calcule les centroïdes et renvoie un GeoDataFrame réduit.

    Retour : GeoDataFrame avec colonnes [GEOCODE, GEONAME, BOROUGH, geometry, LATITUDE, LONGITUDE]
    """
    gdf = gpd.read_file(geojson_path)
    # Adapter les noms selon le geojson fourni
    if geocode_field not in gdf.columns:
        # chercher un champ raisonnable
        candidates = [c for c in gdf.columns if c.lower() in ('geocode','geoid','boro_cd','cd')]
        if candidates:
            geocode_field = candidates[0]
        else:
            raise ValueError(f"Champ GEOCODE introuvable. Colonnes disponibles: {gdf.columns.tolist()}")

    # Assure projection en WGS84 (lon/lat)
    if gdf.crs is not None:
        try:
            gdf = gdf.to_crs(epsg=4326)
        except Exception:
            pass

    # Centroid
    gdf['centroid'] = gdf.geometry.centroid
    gdf['LONGITUDE'] = gdf.centroid.x
    gdf['LATITUDE'] = gdf.centroid.y

    # standardise les colonnes utiles
    out = gdf[[geocode_field, 'geometry']].copy()
    out = out.rename(columns={geocode_field: 'GEOCODE'})

    # tenter de récupérer un nom et borough
    for candidate in ('GEONAME','geoname','NAME','name','neighborhood','ntaname'):
        if candidate in gdf.columns:
            out['GEONAME'] = gdf[candidate]
            break
    for candidate in ('BOROUGH','borough','boro_name'):
        if candidate in gdf.columns:
            out['BOROUGH'] = gdf[candidate]
            break

    out['LATITUDE'] = gdf['LATITUDE']
    out['LONGITUDE'] = gdf['LONGITUDE']
    return out


def join_air_with_geo(aq_df, geo_light_df, left_on=COL_GEOJOIN, right_on='GEOCODE'):
    """Jointure simple entre air quality et geo_light sur les identifiants de zone.
    Retourne un DataFrame pandas.
    """
    aq_df = aq_df.copy()
    geo = geo_light_df.copy()
    # s'assurer du même type
    aq_df[left_on] = aq_df[left_on].astype(str)
    geo['GEOCODE'] = geo['GEOCODE'].astype(str)
    joined = aq_df.merge(geo.drop(columns=['geometry']), how='left', left_on=left_on, right_on='GEOCODE')
    return joined


def compute_weighted_value(stations_df, target_point, element_col='value', lat_col='latitude', lon_col='longitude', method='inverse_distance'):
    """Calcule une moyenne pondérée des stations par rapport à target_point=(lat, lon).

    stations_df doit contenir les colonnes lat/lon et la valeur mesurée.
    method : 'inverse_distance' ou 'inverse_distance_squared'
    """
    df = stations_df.copy()
    tgt = (target_point[0], target_point[1])
    def _dist(row):
        try:
            return geodesic(tgt, (row[lat_col], row[lon_col])).km
        except Exception:
            return float('nan')
    df['distance_km'] = df.apply(_dist, axis=1)
    df = df.dropna(subset=['distance_km', element_col])
    # éviter division par zéro
    df['distance_km'] = df['distance_km'].replace(0, 0.0001)
    if method == 'inverse_distance_squared':
        df['weight'] = 1.0 / (df['distance_km'] ** 2)
    else:
        df['weight'] = 1.0 / df['distance_km']
    weighted = (df[element_col] * df['weight']).sum() / df['weight'].sum()
    return weighted, df.sort_values('distance_km')


# ---- SCRIPT DE PREPARATION ---------------------------------

def prepare_all(air_path=AIR_QUALITY_PATH, geojson_path=GEOJSON_PATH, stations_path=STATIONS_PATH):
    print("Chargement Air Quality...", air_path)
    aq = pd.read_csv(air_path, dtype=str)
    print(f"Lignes Air Quality: {len(aq)}")

    # normaliser Start_Date
    if COL_START in aq.columns:
        aq = normalize_start_date(aq, col=COL_START)
    else:
        print(f"Attention: colonne {COL_START} introuvable dans le fichier Air Quality.")

    # charger geojson et construire geo_light
    geo_light_gdf = build_geo_light(geojson_path)
    # sauvegarder le geojson raw et le geo_light
    raw_out = os.path.join(PROCESSED_DIR, 'geo_raw.geojson')
    light_csv = os.path.join(PROCESSED_DIR, 'geo_light.csv')
    geo_light_gdf.to_file(raw_out, driver='GeoJSON')
    geo_light_gdf.drop(columns=['geometry']).to_csv(light_csv, index=False)
    print(f"Geo raw sauvegardé: {raw_out}")
    print(f"Geo light sauvegardé (csv): {light_csv}")

    # join
    joined = join_air_with_geo(aq, geo_light_gdf, left_on=COL_GEOJOIN, right_on='GEOCODE')
    out_joined = os.path.join(PROCESSED_DIR, 'aq_with_geo.csv')
    joined.to_csv(out_joined, index=False)
    print(f"Air Quality avec géo sauvegardé: {out_joined}")

    # stations (optionnel)
    if os.path.exists(stations_path):
        stations = pd.read_csv(stations_path)
        print(f"Stations chargées: {len(stations)}")
    else:
        print("Fichier stations introuvable; ignorer la partie stations pour l'instant.")

    return {
        'aq': aq,
        'geo_light': geo_light_gdf,
        'joined': joined
    }


# ---- DASH APP (minimal) -----------------------------------

def run_dash_server(processed_dir=PROCESSED_DIR):
    """Lance un serveur Dash minimal qui lit les fichiers produits par prepare_all()."""
    try:
        from dash import Dash, html, dcc, Output, Input, State
        import dash_leaflet as dl
        import dash_leaflet.express as dlx
        import plotly.express as px
    except Exception as e:
        print("Pour exécuter le dashboard, installez: dash, dash-leaflet, plotly")
        raise

    # Charger données
    path_joined = os.path.join(processed_dir, 'aq_with_geo.csv')
    path_geo = os.path.join(processed_dir, 'geo_raw.geojson')
    if not os.path.exists(path_joined) or not os.path.exists(path_geo):
        raise FileNotFoundError("Fichiers préparés introuvables. Lancez --prepare avant --run-server")

    df = pd.read_csv(path_joined)
    with open(path_geo, 'r', encoding='utf-8') as f:
        geojson = json.load(f)

    # construire GeoJSON pour dash-leaflet
    # crée un dict id -> properties pour le choropleth
    features = geojson.get('features', [])
    # map des GEOCODE -> feature
    id_prop = 'GEOCODE'
    for feat in features:
        props = feat.setdefault('properties', {})
        # s'assurer d'avoir GEOCODE
        if id_prop not in props:
            # essayer de deviner
            for k in list(feat['properties'].keys()):
                if k.lower() in ('geocode','geoid'):
                    props[id_prop] = feat['properties'][k]
                    break

    geojson_for_map = { 'type': 'FeatureCollection', 'features': features }

    app = Dash(__name__)
    server = app.server

    # Elements UI: carte + panneau droit
    app.layout = html.Div([
        html.H3("Air Quality - NYC (Dashboard prototype)"),
        html.Div([
            dl.Map(children=[
                dl.TileLayer(),
                dl.GeoJSON(data=geojson_for_map, id='geojson', zoomToBounds=True, options=dict(style=dict(weight=1, color='#444', fillOpacity=0.5))),
            ], id='map', style={'width': '70vw', 'height': '70vh', 'display': 'inline-block'}),

            html.Div([
                html.H4("Détails quartier"),
                html.Div(id='details'),
                html.Label("Sélection dates:"),
                dcc.DatePickerRange(id='date-range', start_date=df['start_date_iso'].min(), end_date=df['start_date_iso'].max()),
                html.Br(),
                dcc.Dropdown(id='element-select', options=[{'label':c,'value':c} for c in df['name'].unique()], multi=False, placeholder='Choisir un élément'),
                html.Div(id='side-charts')
            ], style={'width': '28vw', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '1rem'})
        ])
    ])

    # callback pour cliquer sur GeoJSON
    @app.callback(Output('details', 'children'), Input('geojson', 'click_feature'), State('date-range', 'start_date'), State('date-range', 'end_date'), State('element-select', 'value'))
    def on_click(feature, start_date, end_date, element):
        if feature is None:
            return "Cliquez sur un quartier sur la carte pour afficher les détails."
        props = feature.get('properties', {})
        gid = props.get('GEOCODE') or props.get('geocode') or props.get('GEOID')
        # filtrer df
        tmp = df[df['GEOCODE'].astype(str) == str(gid)] if 'GEOCODE' in df.columns else df[df[COL_GEOJOIN].astype(str) == str(gid)]
        if tmp.empty:
            return html.Div([html.P(f"Pas de données pour {gid}" )])
        # filter dates
        if start_date:
            tmp = tmp[tmp['start_date_iso'] >= start_date]
        if end_date:
            tmp = tmp[tmp['start_date_iso'] <= end_date]
        # si element
        if element:
            tmp2 = tmp[tmp['name'] == element]
        else:
            tmp2 = tmp
        # calculs simples
        n = len(tmp2)
        avg = tmp2['data_value'].astype(float).mean() if n>0 else None
        details = [html.P(f"GEOCODE: {gid}"), html.P(f"Nom: {props.get('GEONAME', props.get('name','-'))}"), html.P(f"Borough: {props.get('BOROUGH','-')}")]
        details.append(html.P(f"Observations filtrées: {n}"))
        details.append(html.P(f"Moyenne (valeur): {avg:.3f}" if avg is not None else html.P("Aucune valeur")))
        return details

    @app.callback(Output('side-charts', 'children'), Input('geojson', 'click_feature'), State('date-range', 'start_date'), State('date-range', 'end_date'), State('element-select', 'value'))
    def build_charts(feature, start_date, end_date, element):
        if feature is None:
            return ''
        props = feature.get('properties', {})
        gid = props.get('GEOCODE') or props.get('geocode') or props.get('GEOID')
        tmp = df[df['GEOCODE'].astype(str) == str(gid)] if 'GEOCODE' in df.columns else df[df[COL_GEOJOIN].astype(str) == str(gid)]
        if start_date:
            tmp = tmp[tmp['start_date_iso'] >= start_date]
        if end_date:
            tmp = tmp[tmp['start_date_iso'] <= end_date]
        if element:
            tmp = tmp[tmp['name'] == element]
        if tmp.empty:
            return html.Div([html.P('Aucune donnée pour ces filtres')])
        # histogramme par start_date_iso
        tmp['start_date_iso'] = pd.to_datetime(tmp['start_date_iso'], errors='coerce')
        hist = tmp.groupby(tmp['start_date_iso'].dt.date)['data_value'].mean().reset_index()
        fig = px.bar(hist, x='start_date_iso', y='data_value', labels={'start_date_iso':'Date','data_value':'Valeur moyenne'})
        return dcc.Graph(figure=fig)

    app.run_server(debug=True, port=8050)


# ---- CLI --------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true', help='Prépare les fichiers (formatage et jointures)')
    parser.add_argument('--run-server', action='store_true', help='Démarre le dashboard Dash (après préparation)')
    args = parser.parse_args()

    if args.prepare:
        prepare_all()
    if args.run_server:
        run_dash_server()

    if not args.prepare and not args.run_server:
        parser.print_help()
