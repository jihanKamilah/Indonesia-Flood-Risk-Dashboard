import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import folium
from streamlit_folium import st_folium
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from folium.plugins import HeatMap

# =============================
# FIX RANDOM (STABLE)
# =============================
random.seed(42)
np.random.seed(42)

# =============================
# CONFIG
# =============================
API_KEY = "ISI_API_MU"

st.set_page_config(layout="wide")
st.title("Indonesia Flood Risk Monitoring System")

# =============================
# SIDEBAR
# =============================
st.sidebar.header("Filter Panel")

province_map = {
    "DKI Jakarta": ["Jakarta"],
    "Jawa Barat": ["Bandung","Bogor","Depok","Bekasi","Cirebon","Sukabumi","Tasikmalaya","Garut","Subang"],
    "Jawa Tengah": ["Semarang","Solo","Magelang","Tegal","Purwokerto","Pekalongan","Salatiga","Kudus"],
    "Jawa Timur": ["Surabaya","Malang","Kediri","Madiun","Blitar","Jember","Banyuwangi","Probolinggo"],
    "Banten": ["Tangerang","Serang","Cilegon","Pandeglang"],
    "DI Yogyakarta": ["Yogyakarta"],
    "Sumatera Utara": ["Medan","Binjai","Pematangsiantar","Tebing Tinggi"],
    "Sumatera Barat": ["Padang","Bukittinggi","Payakumbuh","Solok"],
    "Riau": ["Pekanbaru","Dumai"],
    "Kepulauan Riau": ["Batam","Tanjung Pinang"],
    "Sumatera Selatan": ["Palembang","Lubuklinggau","Prabumulih"],
    "Lampung": ["Bandar Lampung","Metro"],
    "Jambi": ["Jambi","Sungai Penuh"],
    "Bengkulu": ["Bengkulu"],
    "Aceh": ["Banda Aceh","Langsa","Lhokseumawe"],
    "Kalimantan Barat": ["Pontianak","Singkawang"],
    "Kalimantan Tengah": ["Palangka Raya"],
    "Kalimantan Selatan": ["Banjarmasin","Banjarbaru"],
    "Kalimantan Timur": ["Balikpapan","Samarinda","Bontang"],
    "Kalimantan Utara": ["Tarakan"],
    "Sulawesi Selatan": ["Makassar","Parepare","Palopo"],
    "Sulawesi Utara": ["Manado","Bitung","Tomohon"],
    "Sulawesi Tengah": ["Palu"],
    "Sulawesi Tenggara": ["Kendari","Baubau"],
    "Gorontalo": ["Gorontalo"],
    "Sulawesi Barat": ["Mamuju"],
    "Bali": ["Denpasar","Singaraja"],
    "NTB": ["Mataram","Bima"],
    "NTT": ["Kupang","Maumere"],
    "Maluku": ["Ambon","Tual"],
    "Maluku Utara": ["Ternate","Tidore"],
    "Papua": ["Jayapura","Merauke"],
    "Papua Barat": ["Sorong","Manokwari"]
}

selected_province = st.sidebar.selectbox("Select Province", list(province_map.keys()))
cities = province_map[selected_province]
city = st.selectbox("Select City", cities)

risk_filter = st.sidebar.multiselect(
    "Select Risk Level",
    ["HIGH","MEDIUM","LOW"],
    default=["HIGH","MEDIUM","LOW"]
)

show_heatmap = st.sidebar.checkbox("Show Heatmap", True)
show_marker = st.sidebar.checkbox("Show Markers", True)

# =============================
# FETCH DATA
# =============================
@st.cache_data(ttl=600)
def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
    res = requests.get(url)
    data = res.json()

    if res.status_code != 200 or "list" not in data:
        return None, data

    return data, None

data, error = get_weather(city)

if error:
    st.error(error)
    st.stop()

df = pd.json_normalize(data['list'])
df['datetime'] = pd.to_datetime(df['dt_txt'])

# =============================
# FEATURE ENGINEERING
# =============================
if 'rain.3h' in df.columns:
    df['rainfall'] = df['rain.3h'].fillna(0)
else:
    df['rainfall'] = 0

df['humidity'] = df['main.humidity']
df['wind'] = df['wind.speed']
df['rain_24h'] = df['rainfall'].rolling(8).sum().fillna(0)

# =============================
# ML MODEL
# =============================
def create_label(row):
    # flood biasanya mulai relevan di >60mm/hari
    return 1 if row['rain_24h'] > 60 else 0

df['label'] = df.apply(create_label, axis=1)

# =============================
# REALISTIC FLOOD SCORING
# =============================
def flood_score(row):
    score = 0

    # 🌧️ Rainfall (MAIN DRIVER)
    if row['rain_24h'] < 20:
        score += 10
    elif row['rain_24h'] < 55:
        score += 30
    elif row['rain_24h'] < 100:
        score += 70
    else:
        score += 100

    # 💧 Humidity (supporting factor)
    if row['humidity'] > 80:
        score += 10
    elif row['humidity'] > 60:
        score += 5

    # 🌬️ Wind (storm indicator)
    if row['wind'] > 8:
        score += 10
    elif row['wind'] > 5:
        score += 5

    return min(score, 100)

df['ml_proba'] = df.apply(flood_score, axis=1)

# =============================
# CLASSIFICATION + RECOMMENDATION
# =============================
def classify(p):
    if p >= 70:
        return "HIGH"
    elif p >= 30:
        return "MEDIUM"
    else:
        return "LOW"

def recommendation(risk):
    if risk=="HIGH":
        return "⚠️ Avoid outdoor activities, stay alert for flood warnings."
    elif risk=="MEDIUM":
        return "☔ Bring umbrella & monitor weather updates."
    else:
        return "✅ Safe for activities, no major risk."

df['ml_risk'] = df['ml_proba'].apply(classify)

# =============================
# REALTIME DISPLAY
# =============================
st.subheader("Real-Time Weather")

col1,col2,col3 = st.columns(3)
col1.metric("Rainfall (Max 24h)", round(df['rainfall'].tail(8).max(),2))
col2.metric("Humidity", df['humidity'].iloc[-1])
col3.metric("Wind", round(df['wind'].iloc[-1],2))

col1,col2 = st.columns(2)
col1.plotly_chart(px.line(df, x='datetime', y='rainfall'), use_container_width=True)
col2.plotly_chart(px.line(df, x='datetime', y='rain_24h'), use_container_width=True)

# =============================
# FLOOD RISK
# =============================
st.subheader("Flood Risk Prediction")

latest_proba = float(df['ml_proba'].iloc[-1])
latest_risk = classify(latest_proba)

if latest_risk not in risk_filter:
    st.warning("⚠️ No data for selected risk filter")
else:
    col1,col2 = st.columns(2)
    col1.metric("Risk Level", latest_risk)
    col2.metric("Probability (%)", round(latest_proba,2))

    st.info(recommendation(latest_risk))

    st.plotly_chart(
        px.line(df, x='datetime', y='ml_proba'),
        use_container_width=True
    )


# =============================
# MAP CONTROL
# =============================
st.subheader("Flood Risk Map")

map_mode = st.radio(
    "Map View Mode",
    ["Province View","Single City View"],
    horizontal=True
)

# =============================
# MAP
# =============================
st.subheader("Flood Risk Map")

coords = {
    #DKI JAKARTA
    "Jakarta": [-6.2,106.8],
    # JAWA BARAT
    "Bandung": [-6.9,107.6], "Bogor": [-6.6,106.8],
    "Depok": [-6.4,106.82], "Bekasi": [-6.23,107.0],
    "Cirebon": [-6.7,108.55], "Sukabumi": [-6.92,106.93],
    "Tasikmalaya": [-7.35,108.22], "Garut": [-7.21,107.9],
    "Subang": [-6.57,107.76],
    # JAWA TENGAH
    "Semarang": [-7.0,110.4], "Solo": [-7.57,110.82],
    "Magelang": [-7.48,110.22], "Tegal": [-6.87,109.14],
    "Purwokerto": [-7.43,109.24], "Pekalongan": [-6.89,109.67],
    "Salatiga": [-7.33,110.5], "Kudus": [-6.8,110.84],
    # JAWA TIMUR
    "Surabaya": [-7.25,112.75], "Malang": [-7.98,112.63],
    "Kediri": [-7.82,112.0], "Madiun": [-7.63,111.52],
    "Blitar": [-8.1,112.17], "Jember": [-8.17,113.7],
    "Banyuwangi": [-8.22,114.36], "Probolinggo": [-7.75,113.22],
    # BANTEN
    "Tangerang": [-6.17,106.63], "Serang": [-6.12,106.15],
    "Cilegon": [-6.0,106.05], "Pandeglang": [-6.32,106.11],
    # SUMATERA
    "Medan": [3.6,98.6], "Binjai": [3.6,98.48],
    "Pematangsiantar": [2.96,99.06], "Tebing Tinggi": [3.33,99.16],
    "Padang": [-0.95,100.35], "Bukittinggi": [-0.31,100.37],
    "Payakumbuh": [-0.23,100.63], "Solok": [-0.79,100.65],
    "Pekanbaru": [0.5,101.45], "Dumai": [1.66,101.45],
    "Batam": [1.13,104.0], "Tanjung Pinang": [0.92,104.45],
    "Palembang": [-2.9,104.7], "Lubuklinggau": [-3.3,102.86],
    "Prabumulih": [-3.45,104.25],
    "Bandar Lampung": [-5.45,105.27], "Metro": [-5.11,105.3],
    "Jambi": [-1.6,103.6], "Sungai Penuh": [-2.08,101.38],
    "Bengkulu": [-3.8,102.26],
    "Banda Aceh": [5.55,95.32], "Langsa": [4.47,97.96],
    "Lhokseumawe": [5.18,97.15],
    # KALIMANTAN
    "Pontianak": [-0.02,109.3], "Singkawang": [0.91,108.98],
    "Palangka Raya": [-2.21,113.92],
    "Banjarmasin": [-3.3,114.6], "Banjarbaru": [-3.44,114.83],
    "Balikpapan": [-1.2,116.8], "Samarinda": [-0.5,117.15],
    "Bontang": [0.12,117.47], "Tarakan": [3.3,117.6],
    # SULAWESI
    "Makassar": [-5.1,119.4], "Parepare": [-4.01,119.62],
    "Palopo": [-3.0,120.2], "Manado": [1.49,124.84],
    "Bitung": [1.44,125.18], "Tomohon": [1.33,124.83],
    "Palu": [-0.9,119.87], "Kendari": [-3.99,122.52],
    "Baubau": [-5.47,122.63], "Gorontalo": [0.54,123.06],
    "Mamuju": [-2.68,118.89],
    # BALI & NUSA
    "Denpasar": [-8.65,115.22], "Singaraja": [-8.11,115.09],
    "Mataram": [-8.58,116.1], "Bima": [-8.45,118.73],
    "Kupang": [-10.16,123.6], "Maumere": [-8.62,122.21],
    # MALUKU
    "Ambon": [-3.7,128.18], "Tual": [-5.63,132.75],
    "Ternate": [0.79,127.38], "Tidore": [0.68,127.4],
    # PAPUA
    "Jayapura": [-2.53,140.7], "Merauke": [-8.49,140.4],
    "Sorong": [-0.86,131.25], "Manokwari": [-0.87,134.08]
}


if map_mode == "Single City View":
    center = coords.get(city, [-6.5,107])
    zoom = 10
    target_cities = [city]
else:
    valid_coords = [coords[c] for c in cities if c in coords]
    if len(valid_coords) > 0:
        avg_lat = sum([c[0] for c in valid_coords]) / len(valid_coords)
        avg_lon = sum([c[1] for c in valid_coords]) / len(valid_coords)
        center = [avg_lat, avg_lon]
    else:
        center = [-6.5,107]
    zoom = 7
    target_cities = cities

m = folium.Map(location=center, zoom_start=zoom)
heat_data = []

for c in target_cities:
    if c not in coords:
        continue

    if c == city:
        prob = float(latest_proba)
        rainfall = float(df['rainfall'].tail(8).max())
        rain24 = float(df['rain_24h'].iloc[-1])
    else:
        data_c,_ = get_weather(c)
        if not data_c:
            continue

        df_c = pd.json_normalize(data_c['list'])

        if 'rain.3h' in df_c.columns:
            df_c['rainfall'] = df_c['rain.3h'].fillna(0)
        else:
            df_c['rainfall'] = 0

        df_c['rain_24h'] = df_c['rainfall'].rolling(8).sum().fillna(0)

        rainfall = float(df_c['rainfall'].tail(8).max())
        rain24 = float(df_c['rain_24h'].iloc[-1])

        df_c['humidity'] = df_c['main.humidity']
        df_c['wind'] = df_c['wind.speed']

        df_c['ml_proba'] = df_c.apply(flood_score, axis=1)
        prob = float(df_c['ml_proba'].iloc[-1])

    risk = classify(prob)

    if risk not in risk_filter:
        continue

    heat_data.append([coords[c][0], coords[c][1], prob])

    tooltip = f"""
    <b>{c}</b><br>
    Probability: {round(prob,1)}%<br>
    Rainfall: {round(rainfall,2)} mm<br>
    24h Rain: {round(rain24,2)} mm<br>
    Risk: {risk}<br>
    {recommendation(risk)}
    """

    if show_marker:
        folium.CircleMarker(
            location=coords[c],
            radius=8,
            color="red" if prob>70 else "orange" if prob>40 else "green",
            fill=True,
            tooltip=folium.Tooltip(tooltip)
        ).add_to(m)

if show_heatmap and len(heat_data)>0:
    HeatMap(heat_data).add_to(m)

st_folium(m, use_container_width=True)

# =============================
# DATA
# =============================
st.subheader("Data Preview")
n = st.selectbox("Rows", [5,10,20,30])
st.dataframe(df.tail(n))
