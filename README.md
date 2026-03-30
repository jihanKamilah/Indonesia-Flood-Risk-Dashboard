# 🌊 Indonesia Flood Risk Monitoring Dashboard

An interactive dashboard to monitor and predict flood risk across Indonesia using real-time weather data.

---

## 🌧️ Background

Indonesia has been experiencing increasingly irregular rainfall patterns, with precipitation occurring more frequently and often unpredictably. Sudden rainfall events, even during typically dry periods, can significantly increase the risk of localized flooding.

These conditions highlight the need for accessible tools that can translate weather data into actionable insights. Without proper monitoring, individuals and decision-makers may struggle to assess potential risks in a timely manner.

This project aims to bridge that gap by transforming real-time weather forecasts into an intuitive flood risk monitoring system, combining data visualization and simple risk scoring to support better awareness and decision-making.

---

## 🚀 Live Demo

https://indonesia-flood-risk-dashboard.streamlit.app/

---

## 📌 Project Overview

This project provides a real-time flood risk monitoring and prediction system that combines:

* 🌧️ Weather forecast data (OpenWeather API)
* 📊 Time-series analysis
* 🧠 Rule-based flood risk prediction
* 🗺️ Interactive geospatial visualization

The system not only visualizes weather conditions but also generates a flood risk score (LOW, MEDIUM, HIGH) based on rainfall intensity and supporting environmental factors.

---

## ⚙️ Features

* Real-time weather data visualization
* 24-hour rainfall accumulation tracking
* Flood risk classification (LOW / MEDIUM / HIGH)
* Interactive map with:

  * Province view
  * City-level analysis
  * Heatmap visualization
* Risk-based recommendations

---

## 🧠 Methodology

Flood risk is calculated using a **rule-based scoring approach**:

### 🌧️ Rainfall (Primary Factor)

* < 20 mm → Low contribution
* 20–55 mm → Moderate contribution
* 55–100 mm → High contribution
* > 100 mm → Extreme contribution

### 💧 Humidity

* > 60% → increases risk

### 🌬️ Wind Speed

* Strong wind indicates storm potential

---

## 📦 Tech Stack

* Python
* Streamlit
* Plotly
* Folium
* OpenWeather API
* Scikit-learn (initial experimentation)

---

## ⚙️ Installation (Local)

```bash
git clone https://github.com/your-username/indonesia-flood-risk-dashboard.git
cd indonesia-flood-risk-dashboard
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔐 Environment Setup

Create `.streamlit/secrets.toml`:

```toml
API_KEY = "your_openweather_api_key"
```

---

## ☁️ Deployment

This app is designed to be deployed on **Streamlit Cloud**.

---

## 📊 Example Insight

* Low rainfall (~16 mm/day) → Low flood risk
* Moderate rainfall (~40 mm/day) → Medium risk
* Heavy rainfall (>60 mm/day) → High risk

---

## 🚀 Future Improvements

* Integration with BMKG data
* More complex machine learning-based prediction
* Flood historical dataset integration
* Early warning system based on trend detection

---

## 👩‍💻 Author

**Jihan Kamilah**
Data Analyst
