import polyline
import streamlit as st
import pandas as pd
import joblib
import requests
from datetime import datetime
import folium
import time
import streamlit.components.v1 as components
from streamlit_geolocation import streamlit_geolocation

# ----------------------------
# 1. Load ML model & scaler
# ----------------------------
try:
    model = joblib.load("models/eta_prediction_xgboost_model.pkl")
    scaler = joblib.load("processed/scaler.pkl")
except:
    st.error("❌ Model or scaler not found. Please train and preprocess first.")
    st.stop()

try:
    X_train_cols_df = pd.read_csv("processed/X_train.csv", nrows=0)
    EXPECTED_FEATURE_COLS = X_train_cols_df.columns.tolist()
except Exception as e:
    st.error(f"❌ Failed to load expected feature columns: {e}")
    st.stop()

# ORS API config
ORS_API_KEY = "your_key"
ORS_URL = "https://api.openrouteservice.org/v2/directions/driving-car"

# OpenWeather API config (replace with your key)
WEATHER_API_KEY = "your_key"
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

# ----------------------------
# 2. Fetch Hospital Data (Mock API)
# ----------------------------
try:
    hospital_data = requests.get("https://hospital-api-1-ob0v.onrender.com/hospitals").json()["hospitals"]
except:
    st.error("❌ Could not fetch hospital API data. Make sure FastAPI is running.")
    st.stop()

emergency_map = {"Low": 0, "Medium": 1, "High": 2}
for h in hospital_data:
    h["emergency_load_encoded"] = emergency_map[h["emergency_load"]]


def _is_valid_coord(lat, lon):
    try:
        if lat is None or lon is None:
            return False
        lat = float(lat)
        lon = float(lon)
        if pd.isna(lat) or pd.isna(lon):
            return False
        return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0
    except Exception:
        return False

# Simple great-circle distance fallback (in kilometers)
import math

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

DEFAULT_SPEED_KMH = 40.0

def _get_location_ipapi():
    try:
        resp = requests.get("https://ipapi.co/json/", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            lat = data.get("latitude")
            lon = data.get("longitude")
            if lat is not None and lon is not None:
                return float(lat), float(lon)
    except Exception:
        pass
    return None, None

# ----------------------------
# 3. Helper Functions
# ----------------------------
@st.cache_data(show_spinner=False, ttl=600)
def _ors_request_cached(start_lat, start_lon, end_lat, end_lon):
    headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
    coords = [[start_lon, start_lat], [end_lon, end_lat]]
    body = {"coordinates": coords}

    max_retries = 3
    backoff = 1.0
    last_err = None
    for _ in range(max_retries):
        try:
            response = requests.post(ORS_URL, json=body, headers=headers)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else backoff
                time.sleep(delay)
                backoff = min(backoff * 2, 8.0)
                continue
            response.raise_for_status()
            data = response.json()
            routes = data.get("routes", [])
            if not routes:
                raise ValueError("No routes returned from ORS")
            summary = routes[0]["summary"]
            geometry = routes[0]["geometry"]

            distance_km = round(summary["distance"] / 1000, 2)
            duration_min = round(summary["duration"] / 60, 2)
            return distance_km, duration_min, geometry
        except requests.exceptions.RequestException as e:
            last_err = e
            # If 429, try again with backoff
            if hasattr(e, "response") and e.response is not None and e.response.status_code == 429:
                retry_after = e.response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else backoff
                time.sleep(delay)
                backoff = min(backoff * 2, 8.0)
                continue
            # For other HTTP/network errors do not retry
            break
        except Exception as e:
            last_err = e
            break

    raise RuntimeError(f"ORS request failed after retries: {last_err}")


def get_route_data_ors(start_lat, start_lon, end_lat, end_lon):
    # Validate input coordinates before calling ORS
    if not _is_valid_coord(start_lat, start_lon) or not _is_valid_coord(end_lat, end_lon):
        return None, None, None
    # If start and end are effectively the same point, short-circuit
    if abs(start_lat - end_lat) < 1e-5 and abs(start_lon - end_lon) < 1e-5:
        return 0.0, 0.0, None

    # Round coordinates to improve cache hit rate
    r_start_lat = round(start_lat, 4)
    r_start_lon = round(start_lon, 4)
    r_end_lat = round(end_lat, 4)
    r_end_lon = round(end_lon, 4)
    try:
        return _ors_request_cached(r_start_lat, r_start_lon, r_end_lat, r_end_lon)
    except Exception as e:
        msg = str(e)
        if "400" in msg:
            if not st.session_state.get("ors_400_warned", False):
                st.warning("OpenRouteService rejected the request (400). Falling back to straight-line ETA for some hospitals.")
                st.session_state["ors_400_warned"] = True
        else:
            if not st.session_state.get("ors_fallback_warned", False):
                st.warning("Routing service error. Falling back to straight-line ETA for some hospitals.")
                st.session_state["ors_fallback_warned"] = True
        # Fallback ETA based on haversine distance and default speed
        d_km = _haversine_km(start_lat, start_lon, end_lat, end_lon)
        eta_min = round((d_km / DEFAULT_SPEED_KMH) * 60.0, 2) if DEFAULT_SPEED_KMH > 0 else None
        return round(d_km, 2), eta_min, None

def get_ml_eta(features_df):
    try:
        features_df = features_df[EXPECTED_FEATURE_COLS]
        scaled_features = scaler.transform(features_df)
        return model.predict(scaled_features)[0]
    except Exception as e:
        st.error(f"❌ ML ETA error: {e}")
        return None

def get_weather_data(lat, lon):
    """Fetch real-time weather from OpenWeatherMap."""
    try:
        params = {"lat": lat, "lon": lon, "appid": WEATHER_API_KEY, "units": "metric"}
        resp = requests.get(WEATHER_URL, params=params)
        resp.raise_for_status()
        response = resp.json()
        temp = response["main"]["temp"]
        humidity = response["main"]["humidity"]
        condition = response["weather"][0]["main"].lower()

        if "rain" in condition:
            weather_encoded = 2
        elif "storm" in condition or "thunder" in condition:
            weather_encoded = 3
        elif "cloud" in condition:
            weather_encoded = 1
        else:
            weather_encoded = 0

        return temp, humidity, weather_encoded, condition
    except Exception as e:
        st.error(f"❌ Weather API error: {e}")
        return 30, 70, 0, "Clear"

def evaluate_hospitals(amb_lat, amb_lon, temperature, humidity, weather_encoded, emergency_type, patient_condition):
    results = []

    ors_etas = []
    capacity_utilizations = []
    icu_beds_all = []
    emergency_all = []
    facilities_match_scores = []
    emergency_type_encoded_all = []
    skipped_invalid = 0

    for hosp in hospital_data:
        try:
            h_lat = float(hosp["lat"])
            h_lon = float(hosp["lon"])
        except Exception:
            skipped_invalid += 1
            continue

        if not _is_valid_coord(amb_lat, amb_lon) or not _is_valid_coord(h_lat, h_lon):
            skipped_invalid += 1
            continue

        distance_km, ors_eta, geometry = get_route_data_ors(amb_lat, amb_lon, h_lat, h_lon)
        if distance_km is None:
            continue

        icu_beds = hosp["icu_beds"]
        emergency_load_encoded = hosp["emergency_load_encoded"]
        capacity_utilization = hosp["capacity_utilization"]

        # Calculate facilities match score (1 if patient_condition in hospital facilities, else 0)
        facilities = hosp.get("facilities", "").lower()
        match_score = 1 if patient_condition.lower() in facilities else 0

        # Encode emergency type priority (High=2, Medium=1, Low=0)
        emergency_type_encoded = {"Low": 0, "Medium": 1, "High": 2}.get(emergency_type, 0)

        ors_etas.append(ors_eta)
        capacity_utilizations.append(capacity_utilization)
        icu_beds_all.append(icu_beds)
        emergency_all.append(emergency_load_encoded)
        facilities_match_scores.append(match_score)
        emergency_type_encoded_all.append(emergency_type_encoded)

        hour = datetime.now().hour
        day_of_week = datetime.now().weekday()
        is_weekend = 1 if day_of_week >= 5 else 0

        input_data = pd.DataFrame([{
            "ambulance_lat": amb_lat,
            "ambulance_lon": amb_lon,
            "hospital_lat": h_lat,
            "hospital_lon": h_lon,
            "distance_km": distance_km,
            "temperature": temperature,
            "humidity": humidity,
            "weather_encoded": weather_encoded,
            "icu_beds": icu_beds,
            "emergency_load_encoded": emergency_load_encoded,
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "capacity_utilization": capacity_utilization,
            "emergency_type_encoded": emergency_type_encoded,
            "patient_condition_encoded": 1 if match_score == 1 else 0
        }])

        ml_eta = get_ml_eta(input_data)

        results.append({
            "Hospital": hosp["name"],
            "Distance (km)": distance_km,
            "ORS ETA (min)": ors_eta,
            "ML ETA (min)": round(ml_eta, 2) if ml_eta else None,
            "ICU Beds": icu_beds,
            "Emergency Load": hosp["emergency_load"],
            "Capacity Utilization": capacity_utilization,
            "Facilities Match": match_score,
            "Emergency Type Priority": emergency_type_encoded,
            "Geometry": geometry
        })

    if skipped_invalid > 0:
        st.warning(f"Skipped {skipped_invalid} hospital(s) due to invalid coordinates.")
    if not results or not ors_etas:
        st.error("No routes available from ORS for the given inputs.")
        return []
    min_eta, max_eta = min(ors_etas), max(ors_etas)
    min_cap, max_cap = min(capacity_utilizations), max(capacity_utilizations)
    min_icu, max_icu = min(icu_beds_all), max(icu_beds_all)
    min_em, max_em = min(emergency_all), max(emergency_all)
    min_fac, max_fac = min(facilities_match_scores), max(facilities_match_scores)
    min_etp, max_etp = min(emergency_type_encoded_all), max(emergency_type_encoded_all)

    final_results = []
    for i, r in enumerate(results):
        norm_eta = (r["ORS ETA (min)"] - min_eta) / (max_eta - min_eta) if max_eta != min_eta else 0
        norm_cap = (r["Capacity Utilization"] - min_cap) / (max_cap - min_cap) if max_cap != min_cap else 0
        norm_em = (emergency_map[r["Emergency Load"]] - min_em) / (max_em - min_em) if max_em != min_em else 0
        norm_icu = 1 - ((r["ICU Beds"] - min_icu) / (max_icu - min_icu) if max_icu != min_icu else 0)

        # Adjust scoring based on emergency type
        if emergency_type == "High":
            # Prioritize ETA and ICU beds for high emergencies
            score = (0.4 * norm_eta) + (0.2 * norm_cap) + (0.15 * norm_em) + (0.25 * norm_icu)
        else:
            # Standard scoring for Low/Medium emergencies
            score = (0.3 * norm_eta) + (0.25 * norm_cap) + (0.2 * norm_em) + (0.25 * norm_icu)

        r["Score"] = round(score, 3)
        final_results.append(r)

    # Filter to only include hospitals with facilities match
    final_results = [r for r in final_results if r["Facilities Match"] == 1]

    if not final_results:
        st.warning("No hospitals found with facilities matching the patient's condition. Consider adjusting the patient condition or emergency type.")
        return []

    final_results.sort(key=lambda x: x["Score"])
    return final_results

def plot_routes_map(results, amb_lat, amb_lon):
    m = folium.Map(location=[amb_lat, amb_lon], zoom_start=12)
    folium.Marker([amb_lat, amb_lon], tooltip="Ambulance", icon=folium.Icon(color="blue")).add_to(m)
    best = results[0]["Hospital"]
    for hosp in results:
        if hosp["Geometry"]:
            coords = polyline.decode(hosp["Geometry"])
            if coords:
                col = "red" if hosp["Hospital"] == best else "gray"
                folium.PolyLine(coords, color=col, weight=5, opacity=0.8).add_to(m)
                folium.Marker(coords[-1], tooltip=f"{hosp['Hospital']} ({hosp['ORS ETA (min)']} min)",
                              icon=folium.Icon(color="green" if hosp["Hospital"] == best else "gray")).add_to(m)
    return m

# ----------------------------
# 4. Streamlit UI
# ----------------------------
st.title("🚑 Smart Ambulance Hospital Comparison (Real-Time)")

st.sidebar.header("Ambulance Location")

import streamlit as st
import requests
from streamlit_geolocation import streamlit_geolocation

# ----------------------------
# Initialize Session State
# ----------------------------
default_lat, default_lon = 20.2961, 85.8245
if "amb_lat" not in st.session_state:
    st.session_state["amb_lat"] = default_lat
if "amb_lon" not in st.session_state:
    st.session_state["amb_lon"] = default_lon
if "source" not in st.session_state:
    st.session_state["source"] = "Manual"

# ----------------------------
# Sidebar Layout
# ----------------------------
st.sidebar.header("Ambulance Location")

# GPS-based Location
st.sidebar.subheader("📍 GPS Location (Browser)")
location = streamlit_geolocation()

if location and location.get("latitude") and location.get("longitude"):
    lat, lon = float(location["latitude"]), float(location["longitude"])
    if (lat, lon) != (st.session_state["amb_lat"], st.session_state["amb_lon"]):
        st.session_state["amb_lat"] = lat
        st.session_state["amb_lon"] = lon
        st.session_state["source"] = "GPS"
        st.rerun()

# IP-based Location
st.sidebar.subheader("🌐 IP-based Location")
if st.sidebar.button("Use IP-based Location"):
    try:
        response = requests.get("https://ipinfo.io/json", timeout=5)
        data = response.json()
        if "loc" in data:
            lat, lon = map(float, data["loc"].split(","))
            st.session_state["amb_lat"] = lat
            st.session_state["amb_lon"] = lon
            st.session_state["source"] = "IP"
            st.rerun()
        else:
            st.sidebar.error("Could not retrieve IP-based location.")
    except Exception as e:
        st.sidebar.error(f"Error fetching IP location: {e}")

# Search Box Location
st.sidebar.subheader("🔍 Search by Address")
address = st.sidebar.text_input("Enter address or place name")

if st.sidebar.button("Search Location"):
    if not address.strip():
        st.sidebar.warning("Please enter a valid address.")
    else:
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {"q": address, "format": "json"}
            headers = {"User-Agent": "SmartAmbulanceApp"}
            response = requests.get(url, params=params, headers=headers, timeout=5)
            data = response.json()
            if data:
                lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
                st.session_state["amb_lat"] = lat
                st.session_state["amb_lon"] = lon
                st.session_state["source"] = "Search"
                st.rerun()
            else:
                st.sidebar.error("Could not find that address.")
        except Exception as e:
            st.sidebar.error(f"Error fetching location: {e}")

# Manual Coordinate Entry
st.sidebar.subheader("🧭 Manual Coordinates")

# No fixed key — dynamic sync
amb_lat = st.sidebar.number_input(
    "Latitude",
    value=st.session_state["amb_lat"],
    key=f"lat_{st.session_state['source']}",
    format="%.6f"
)
amb_lon = st.sidebar.number_input(
    "Longitude",
    value=st.session_state["amb_lon"],
    key=f"lon_{st.session_state['source']}",
    format="%.6f"
)

# If manually edited, update state
if amb_lat != st.session_state["amb_lat"] or amb_lon != st.session_state["amb_lon"]:
    st.session_state["amb_lat"] = amb_lat
    st.session_state["amb_lon"] = amb_lon
    st.session_state["source"] = "Manual"

# ----------------------------
# Display Source + Map
# ----------------------------
st.sidebar.markdown(f"### 📍 Source: **{st.session_state['source']}**")
st.markdown("### 🗺 Current Ambulance Location")

st.map(
    data=[{"lat": st.session_state["amb_lat"], "lon": st.session_state["amb_lon"]}],
    zoom=12,
    use_container_width=True
)



# ----------------------------

# New inputs for emergency type and patient condition
emergency_types = ["Low", "Medium", "High"]
patient_conditions = ["ICU", "NICU", "Cancer Centre", "Cardiology", "Neurology", "Oncology", "Gastroenterology", "Urology", "Orthopaedics", "Emergency", "Trauma Center", "PET CT", "Advanced Surgery", "24x7 Pharmacy"]

selected_emergency_type = st.sidebar.selectbox("Emergency Type", emergency_types, help="Select the urgency level: Low for non-critical, Medium for moderate, High for life-threatening emergencies.")
selected_patient_condition = st.sidebar.selectbox("Patient Condition", patient_conditions, help="Select the primary medical condition to prioritize hospitals with matching facilities/treatments.")

# Fetch real-time weather
temperature, humidity, weather_encoded, condition = get_weather_data(amb_lat, amb_lon)
st.sidebar.markdown(f"🌡️ **Weather:** {condition.capitalize()} ({temperature}°C, {humidity}% Humidity)")

# Session state
if "results" not in st.session_state:
    st.session_state.results = None
if "ors_error" not in st.session_state:
    st.session_state.ors_error = None
if "ors_400_warned" not in st.session_state:
    st.session_state.ors_400_warned = False
if "ors_fallback_warned" not in st.session_state:
    st.session_state.ors_fallback_warned = False

if st.button("Find Best Hospital"):
    with st.spinner("Evaluating hospitals and predicting ETAs..."):
        try:
            results = evaluate_hospitals(amb_lat, amb_lon, temperature, humidity, weather_encoded, selected_emergency_type, selected_patient_condition)
            st.session_state.results = results
            st.session_state.ors_error = None
        except Exception as e:
            st.session_state.ors_error = str(e)
            st.session_state.results = None

if st.session_state.ors_error:
    st.error(f"❌ ORS Error: {st.session_state.ors_error}")

elif st.session_state.results:
    best = st.session_state.results[0]
    st.success(f"🏥 Best Hospital: **{best['Hospital']}** "
               f"(ORS ETA: {best['ORS ETA (min)']} min, ML ETA: {best['ML ETA (min)']} min)")

    st.subheader("🏆 Hospital Ranking")
    df_results = pd.DataFrame(st.session_state.results).drop(columns=["Geometry"])
    st.dataframe(df_results, use_container_width=True)  # Improved for mobile responsiveness

    # Tooltip for scoring formula
    with st.expander("ℹ️ Scoring Formula Explanation"):
        st.markdown("""
        **Hospital scores are calculated based on the following weighted factors:**
        - **ETA (Estimated Time of Arrival):** Lower ETA is better.
        - **Capacity Utilization:** Lower utilization is better.
        - **Emergency Load:** Lower load is better.
        - **ICU Beds:** More beds are better.
        
        **Emergency Type Adjustments:**
        - **High:** Prioritizes ETA and ICU beds more.
        - **Medium/Low:** Balanced prioritization.
        """)
    
    st.subheader("🗺️ Routes")
    map_obj = plot_routes_map(st.session_state.results, amb_lat, amb_lon)
    map_html = map_obj._repr_html_()
    components.html(map_html, height=500, width=800)  # Adjusted width for better mobile responsiveness

