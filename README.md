# 🚑 SMART AMBULANCE & INTELLIGENT HOSPITAL ROUTING SYSTEM  
### AI-Powered Decision Support for Real-Time Emergency Response

This repository contains an end-to-end **Smart Ambulance Routing & Intelligent Hospital Recommendation System**, integrating:

- Machine Learning (XGBoost ETA Model)  
- Live Routing via OpenRouteService  
- Real-time Weather API (OpenWeather)  
- Browser/IP/Search-Based Ambulance Location  
- Mock Hospital API (FastAPI)  
- Facility-Aware Hospital Matching  
- Dynamic Scoring Engine  
- YOLO-Based Traffic Simulation (Optional Add-On)

The system finds the **best hospital in real time**, predicts ETA, evaluates conditions, shows routes, and performs intelligent decision-making using multiple factors.

---

## ⭐ FEATURES

### 🚑 Real-Time Ambulance Location
Supports multiple location sources:
- **Browser GPS (Geolocation API)**
- **IP-Based Location**
- **Address Search (OpenStreetMap Nominatim)**
- **Manual Coordinates Input**

### 🏥 Intelligent Hospital Ranking Engine
Evaluates hospitals using:
- ORS ETA  
- XGBoost ML ETA  
- ICU beds  
- Emergency load  
- Capacity utilization  
- Weather conditions  
- Facilities match (ICU, NICU, Oncology, etc.)
- Emergency severity (Low/Medium/High)

### 🧮 Scoring Formula

**High Emergency**
```
score = 0.4ETA + 0.2Capacity + 0.15Load + 0.25ICUBeds
```
**Low/Medium Emergency**
```
score = 0.3ETA + 0.25Capacity + 0.2Load + 0.25ICUBeds
```

### 🗺️ Route Visualization
- Folium maps  
- Red route → best hospital  
- Grey routes → other options  
- Markers for ambulance & hospitals  

---

## 🧠 MACHINE LEARNING ETA (XGBoost)

Features used:
- Distance  
- Temperature  
- Humidity  
- Weather condition  
- Weekday & Hour  
- Hospital load  
- Capacity utilization  
- Facility match  
- Emergency type  


---

## 🏥 MOCK HOSPITAL API (FASTAPI)

To simulate real hospital data (since real APIs are limited):

Simulated fields:
- ICU beds  
- Capacity utilization  
- Emergency load  
- Facilities (ICU/NICU/Oncology/etc.)  
- Coordinates  

API Endpoint:
GET /hospitals


---
# TO BE IMPLEMENTED NEXT:(UPCOMING)
## 🚦 YOLO-BASED TRAFFIC SIMULATION (OPTIONAL)

This module uses YOLOv8 to analyze:
- Vehicle count  
- Congestion  
- Lane density  
- Traffic blockages  
- Optional signal detection  

Example output:
```
segment_id, vehicle_count, density
road_1, 34, 0.72
road_2, 10, 0.21
```

Traffic can be used to **dynamically modify ETA**:
```
adjusted_eta = ors_eta * (1 + 0.3 * traffic_density)
```

---

## 🧱 SYSTEM ARCHITECTURE

+----------------------+
| Ambulance Location |
| GPS / IP / Search |
+-----------+----------+
|
v
+------------------------------+
| Streamlit Frontend |
| - Location UI |
| - Weather |
| - Hospital Ranking |
+------------+----------------+
|
v
+---------------------------------------+
| Routing + ML Decision Engine |
| - ORS ETA |
| - ML ETA |
| - Facility Matching |
| - Scoring System |
| - YOLO Traffic (optional) |
+----------------+----------------------+
|
v
+-----------------------+
| Best Hospital + Map |
+-----------------------+


---

## 🚀 RUN LOCALLY

### 1. Install Requirements
```
pip install -r requirements.txt
```

### 2. Start Mock API
```
uvicorn api.main:app --reload
```

### 3. Start Streamlit App
```
streamlit run ap.py
```

---

## 🌤️ APIS USED

| Feature | API |
|--------|-----|
| Routing & ETA | OpenRouteService |
| Weather | OpenWeather API |
| Address Search | OSM Nominatim |
| IP Geolocation | ipinfo.io |
| Traffic | YOLOv8 |

---

## 🧪 SIMULATIONS SUPPORTED

- Mock hospital load  
- Weather override  
- YOLO traffic congestion (UPCOMING) 
- GPS route simulation  
- Emergency severity & facility matching  

---

## 🔮 FUTURE ENHANCEMENTS

- Real-time GPS device integration  
- Real hospital APIs  
- Traffic signal priority system  
- Predictive hospital load modeling  
- Mobile app version  
- Full city-level integration  

---

## 🤝 CONTRIBUTIONS

Pull requests are welcome!

Steps:
1. Fork the repo  
2. Create a feature branch  
3. Submit a PR  

---

## ⭐ SUPPORT

If this project helped you, please **star the repository ⭐ on GitHub**!


