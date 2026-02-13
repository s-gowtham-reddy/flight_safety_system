"""
FLIGHT SAFETY INTELLIGENCE SYSTEM — STREAMLIT DASHBOARD
Continuous ML-powered risk prediction with auto-refreshing analytics.

Author: Syamala Gowtham Reddy
Institution: Sathyabama Institute of Science and Technology
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
import time
from datetime import datetime

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# =============================================================================
# CONSTANTS
# =============================================================================
AIRCRAFT_TYPES = ["Boeing 737", "Airbus A320", "Boeing 777", "Airbus A330", "Boeing 787"]
TIME_SLOTS = ["Morning", "Afternoon", "Evening", "Night"]
COMPLEXITY_LEVELS = ["Low", "Medium", "High"]
AIRPORT_CATEGORIES = ["Category 1", "Category 2", "Category 3"]
ROUTES = [
    "Mumbai  —  Delhi",
    "Delhi  —  Kolkata",
    "Chennai  —  Bangalore",
    "Hyderabad  —  Mumbai",
    "Bangalore  —  Chennai",
    "Kolkata  —  Mumbai",
    "Delhi  —  Chennai",
    "Mumbai  —  Bangalore",
    "Chennai  —  Delhi",
    "Pune  —  Delhi",
    "Jaipur  —  Mumbai",
    "Goa  —  Delhi",
    "Lucknow  —  Bangalore",
    "Ahmedabad  —  Chennai",
    "Bhopal  —  Hyderabad",
]

# Internal encoder-safe names (no spaces)
AIRCRAFT_INTERNAL = ["Boeing737", "Airbus320", "Boeing777", "Airbus330", "Boeing787"]
TIME_INTERNAL = ["Morning", "Afternoon", "Evening", "Night"]
COMPLEXITY_INTERNAL = ["Low", "Medium", "High"]
AIRPORT_INTERNAL = ["Category1", "Category2", "Category3"]

REFRESH_SECONDS = 5

# Color palette — professional, muted, corporate (LIGHT THEME)
COLOR_SAFE = "#2e7d32"      # Dark Green
COLOR_WARN = "#f57c00"      # Orange
COLOR_DANGER = "#c62828"    # Red
COLOR_PRIMARY = "#0277bd"   # Light Blue
COLOR_SECONDARY = "#455a64" # Blue Grey
COLOR_ACCENT = "#00838f"    # Cyan
COLOR_BG_CARD = "#ffffff"   # White

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Segoe UI, sans-serif", color="#263238", size=12),
    margin=dict(l=50, r=30, t=30, b=50),
    xaxis=dict(gridcolor="#e0e0e0", zerolinecolor="#bdbdbd", tickfont=dict(color="#455a64")),
    yaxis=dict(gridcolor="#e0e0e0", zerolinecolor="#bdbdbd", tickfont=dict(color="#455a64")),
)

# =============================================================================
# STYLING
# =============================================================================
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="st-"] {
            font-family: 'Inter', 'Segoe UI', sans-serif;
            color: #263238;
            background-color: #ffffff;
        }

        .main .block-container {
            max-width: 1400px;
            padding-top: 2rem;
        }

        h1, h2, h3, h4 {
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            color: #1a237e;
            letter-spacing: -0.3px;
        }

        /* KPI metric cards - Light Theme with Shadow */
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: all 0.2s ease;
        }
        div[data-testid="stMetric"]:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transform: translateY(-2px);
            border-color: #0277bd;
        }
        div[data-testid="stMetric"] > label {
            font-size: 0.85rem !important;
            color: #546e7a !important;
            font-weight: 600;
        }
        div[data-testid="stMetric"] > div {
            font-size: 1.8rem !important;
            font-weight: 800 !important;
            color: #0d47a1 !important;
        }

        /* Status indicator */
        .status-bar {
            background: #e3f2fd;
            border: 1px solid #bbdefb;
            border-left: 5px solid #0277bd;
            border-radius: 6px;
            padding: 12px 20px;
            margin-bottom: 1.5rem;
            font-size: 0.95rem;
            color: #01579b;
            font-weight: 500;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }

        /* Section headers */
        .section-header {
            font-size: 1.2rem;
            font-weight: 700;
            color: #263238;
            border-bottom: 2px solid #0277bd;
            padding-bottom: 8px;
            margin-bottom: 16px;
            margin-top: 10px;
        }

        /* Dashboard header */
        .dash-header {
            background: #ffffff;
            padding: 24px 30px;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            border-left: 8px solid #1a237e;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border: 1px solid #e0e0e0;
        }
        .dash-header h1 {
            color: #1a237e !important;
            font-size: 2rem;
            margin: 0;
            font-weight: 800;
        }
        .dash-header p {
            color: #546e7a;
            font-size: 1rem;
            margin-top: 4px;
            font-weight: 500;
        }
        .dash-header .live-indicator {
            display: inline-block;
            width: 10px; height: 10px;
            background: #00c853;
            border-radius: 50%;
            margin-right: 8px;
            animation: livePulse 2s infinite;
        }
        @keyframes livePulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        /* Prediction table risk colors */
        .risk-low { 
            background: #e8f5e9; color: #2e7d32; 
            padding: 3px 12px; border-radius: 4px; font-weight: 600; font-size: 0.82rem;
        }
        .risk-medium { 
            background: #fff3e0; color: #e65100;
            padding: 3px 12px; border-radius: 4px; font-weight: 600; font-size: 0.82rem;
        }
        .risk-high { 
            background: #ffebee; color: #c62828;
            padding: 3px 12px; border-radius: 4px; font-weight: 600; font-size: 0.82rem;
        }

        /* Log container */
        .log-box {
            background: #f5f5f5;
            color: #37474f;
            font-family: 'jetBrains Mono', 'Consolas', monospace;
            font-size: 0.8rem;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            max-height: 320px;
            overflow-y: auto;
            line-height: 1.6;
        }
        .log-box .latest { color: #000000; font-weight: 700; background: #e3f2fd; display: block; }

        /* Footer */
        .footer {
            text-align: center;
            padding: 30px 0;
            border-top: 1px solid #e0e0e0;
            margin-top: 3rem;
            color: #78909c;
            background: #fafafa;
        }
        .footer strong { color: #37474f; }

        /* Hide streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {visibility: hidden;}
        div[data-testid="stStatusWidget"] {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# DATA GENERATION
# =============================================================================
def create_aviation_dataset(n_samples=5000):
    np.random.seed(42)
    data = {
        "Aircraft_Age_Years": np.random.randint(1, 25, n_samples),
        "Days_Since_Maintenance": np.random.randint(0, 180, n_samples),
        "Flight_Hours": np.random.randint(1000, 50000, n_samples),
        "Aircraft_Type": np.random.choice(AIRCRAFT_INTERNAL, n_samples),
        "Flight_Distance_km": np.random.randint(200, 5000, n_samples),
        "Passenger_Count": np.random.randint(50, 350, n_samples),
        "Cargo_Weight_tons": np.random.uniform(0, 20, n_samples),
        "Departure_Weather_Severity": np.random.randint(0, 11, n_samples),
        "Destination_Weather_Severity": np.random.randint(0, 11, n_samples),
        "Wind_Speed_kmh": np.random.randint(0, 80, n_samples),
        "Pilot_Experience_Hours": np.random.randint(500, 15000, n_samples),
        "Time_of_Day": np.random.choice(TIME_INTERNAL, n_samples),
        "Route_Complexity": np.random.choice(COMPLEXITY_INTERNAL, n_samples),
        "Departure_Airport_Category": np.random.choice(AIRPORT_INTERNAL, n_samples),
        "Destination_Airport_Category": np.random.choice(AIRPORT_INTERNAL, n_samples),
    }
    df = pd.DataFrame(data)

    risk_score = (
        (df["Aircraft_Age_Years"] > 15).astype(int) * 10
        + (df["Days_Since_Maintenance"] > 90).astype(int) * 15
        + (df["Departure_Weather_Severity"] > 7).astype(int) * 20
        + (df["Destination_Weather_Severity"] > 7).astype(int) * 20
        + (df["Wind_Speed_kmh"] > 50).astype(int) * 15
        + (df["Pilot_Experience_Hours"] < 2000).astype(int) * 10
        + (df["Route_Complexity"] == "High").astype(int) * 10
        + np.random.randint(-10, 10, n_samples)
    )
    df["Risk_Level"] = pd.cut(
        risk_score, bins=[-np.inf, 30, 60, np.inf], labels=["Low", "Medium", "High"]
    )
    df["High_Risk"] = (df["Risk_Level"] == "High").astype(int)
    return df


# =============================================================================
# PREPROCESSING
# =============================================================================
def preprocess_data(df):
    X = df.drop(["Risk_Level", "High_Risk"], axis=1)
    y = df["High_Risk"]

    categorical_columns = [
        "Aircraft_Type", "Time_of_Day", "Route_Complexity",
        "Departure_Airport_Category", "Destination_Airport_Category",
    ]

    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    return X, y, scaler, label_encoders


# =============================================================================
# MODEL TRAINING
# =============================================================================
def train_all_models(X_train, y_train, X_test, y_test):
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1),
    }

    trained = {}
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
            "f1_score": round(f1_score(y_test, y_pred) * 100, 2),
            "roc_auc": round(roc_auc_score(y_test, y_proba) * 100, 2),
        }

    ensemble = VotingClassifier(
        estimators=[(n, m) for n, m in models.items()], voting="soft"
    )
    ensemble.fit(X_train, y_train)
    trained["Ensemble"] = ensemble

    y_pred_e = ensemble.predict(X_test)
    y_proba_e = ensemble.predict_proba(X_test)[:, 1]
    results["Ensemble"] = {
        "accuracy": round(accuracy_score(y_test, y_pred_e) * 100, 2),
        "f1_score": round(f1_score(y_test, y_pred_e) * 100, 2),
        "roc_auc": round(roc_auc_score(y_test, y_proba_e) * 100, 2),
    }

    return trained, results


# =============================================================================
# FLIGHT PREDICTION
# =============================================================================
# =============================================================================
# REALISTIC FLIGHT DATA DATABASE
# =============================================================================
REAL_FLIGHTS = [
    {"id": "AI-441", "route": "Delhi — Kochi", "aircraft": "Airbus A320neo", "time": "Morning"},
    {"id": "6E-5032", "route": "Mumbai — Bangalore", "aircraft": "Airbus A321", "time": "Morning"},
    {"id": "UK-945", "route": "Delhi — Mumbai", "aircraft": "Boeing 787-9", "time": "Morning"},
    {"id": "QP-1303", "route": "Bangalore — Pune", "aircraft": "Boeing 737 Max", "time": "Morning"},
    {"id": "SG-816", "route": "Chennai — Hyderabad", "aircraft": "Bombardier Q400", "time": "Morning"},
    {"id": "AI-101", "route": "Delhi — New York (JFK)", "aircraft": "Boeing 777-300ER", "time": "Afternoon"},
    {"id": "6E-205", "route": "Kolkata — Delhi", "aircraft": "Airbus A320", "time": "Afternoon"},
    {"id": "UK-820", "route": "Bangalore — Delhi", "aircraft": "Airbus A320neo", "time": "Afternoon"},
    {"id": "AI-540", "route": "Delhi — Tirupati", "aircraft": "Airbus A319", "time": "Afternoon"},
    {"id": "6E-6022", "route": "Hyderabad — Goa", "aircraft": "ATR 72-600", "time": "Afternoon"},
    {"id": "IX-1202", "route": "Dubai — Kochi", "aircraft": "Boeing 737-800", "time": "Evening"},
    {"id": "UK-963", "route": "Mumbai — Singapore", "aircraft": "Airbus A321neo", "time": "Evening"},
    {"id": "AI-308", "route": "Delhi — Melbourne", "aircraft": "Boeing 787-8", "time": "Night"},
    {"id": "6E-1412", "route": "Sharjah — Lucknow", "aircraft": "Airbus A320", "time": "Night"},
    {"id": "QP-1102", "route": "Mumbai — Ahmedabad", "aircraft": "Boeing 737 Max", "time": "Night"},
    {"id": "SG-182", "route": "Delhi — Dubai", "aircraft": "Boeing 737-800", "time": "Night"},
    {"id": "AI-865", "route": "Mumbai — Delhi", "aircraft": "Boeing 747-400", "time": "Morning"},
    {"id": "6E-356", "route": "Bangalore — Kolkata", "aircraft": "Airbus A321neo", "time": "Morning"},
    {"id": "UK-705", "route": "Delhi — Kolkata", "aircraft": "Airbus A320", "time": "Morning"},
    {"id": "AI-692", "route": "Hyderabad — Mumbai", "aircraft": "Airbus A319", "time": "Morning"},
    {"id": "6E-478", "route": "Chennai — Mumbai", "aircraft": "Airbus A320neo", "time": "Morning"},
    {"id": "I5-782", "route": "Bangalore — Jaipur", "aircraft": "Airbus A320", "time": "Afternoon"},
    {"id": "AI-469", "route": "Delhi — Raipur", "aircraft": "Airbus A320", "time": "Afternoon"},
    {"id": "UK-813", "route": "Bangalore — Delhi", "aircraft": "Airbus A320", "time": "Afternoon"},
    {"id": "6E-512", "route": "Pune — Nagpur", "aircraft": "ATR 72-600", "time": "Evening"},
    {"id": "QP-1402", "route": "Delhi — Bangalore", "aircraft": "Boeing 737 Max", "time": "Evening"},
    {"id": "AI-678", "route": "Mumbai — Kolkata", "aircraft": "Airbus A321", "time": "Evening"},
    {"id": "6E-672", "route": "Kochi — Doha", "aircraft": "Airbus A320neo", "time": "Night"},
    {"id": "IX-345", "route": "Kozhikode — Dubai", "aircraft": "Boeing 737-800", "time": "Night"},
    {"id": "AI-202", "route": "New York — Delhi", "aircraft": "Boeing 777-200LR", "time": "Night"},
    {"id": "UK-955", "route": "Delhi — Paris", "aircraft": "Boeing 787-9", "time": "Afternoon"},
    {"id": "LH-754", "route": "Frankfurt — Bangalore", "aircraft": "Boeing 747-8", "time": "Night"},
    {"id": "BA-119", "route": "London — Bangalore", "aircraft": "A350-1000", "time": "Morning"},
    {"id": "EK-500", "route": "Dubai — Mumbai", "aircraft": "Airbus A380", "time": "Morning"},
    {"id": "QR-570", "route": "Doha — Delhi", "aircraft": "Boeing 777", "time": "Morning"},
    {"id": "SQ-502", "route": "Singapore — Bangalore", "aircraft": "Airbus A350", "time": "Morning"},
    {"id": "EY-278", "route": "Abu Dhabi — Chennai", "aircraft": "Boeing 787-10", "time": "Evening"},
    {"id": "6E-7201", "route": "Lucknow — Mumbai", "aircraft": "Airbus A320", "time": "Afternoon"},
    {"id": "G8-320", "route": "Delhi — Patna", "aircraft": "Airbus A320neo", "time": "Evening"},
    {"id": "AI-803", "route": "Delhi — Bangalore", "aircraft": "Boeing 787-8", "time": "Morning"},
    {"id": "UK-970", "route": "Mumbai — Delhi", "aircraft": "Airbus A321", "time": "Afternoon"},
    {"id": "I5-543", "route": "Kolkata — Bagdogra", "aircraft": "Airbus A320", "time": "Afternoon"},
    {"id": "SG-8169", "route": "Guwahati — Delhi", "aircraft": "Boeing 737", "time": "Afternoon"},
    {"id": "6E-2433", "route": "Indore — Mumbai", "aircraft": "ATR 72", "time": "Evening"},
    {"id": "AI-506", "route": "Delhi — Bangalore", "aircraft": "Boeing 777", "time": "Night"},
]

def generate_random_flight():
    # Pick a random REAL flight from our database
    real_flight = np.random.choice(REAL_FLIGHTS)
    
    # Generate realistic metadata based on the flight profile
    is_long_haul = "777" in real_flight["aircraft"] or "787" in real_flight["aircraft"] or "A350" in real_flight["aircraft"]
    base_dist = 5000 if is_long_haul else 1200
    
    # Roll for flight scenario (85% Normal, 15% Issue)
    roll = np.random.random()
    
    # Default safe parameters
    weather_sev = int(np.random.choice([0,1,2,3], p=[0.4,0.3,0.2,0.1]))
    maintenance = int(np.random.randint(0, 30))
    wind = int(np.random.normal(15, 5))
    pilot_exp = int(np.random.normal(8000, 2000))
    age = int(np.random.randint(1, 12))
    
    if roll < 0.85:
        # Normal Operation (Safe)
        pass 
        
    elif roll < 0.90:
        # Scenario 1: Severe Weather Event (5% prob)
        weather_sev = int(np.random.randint(8, 11))
        wind = int(np.random.randint(60, 95))
        # Maintenance and pilot likely fine to show contrast
        
    elif roll < 0.95:
        # Scenario 2: Technical / Maintenance Issue (5% prob)
        maintenance = int(np.random.randint(95, 180))
        age = int(np.random.randint(18, 30))
        # Weather likely fine
        weather_sev = int(np.random.randint(0, 3))
        
    else:
        # Scenario 3: Operational / Pilot Risk (5% prob)
        pilot_exp = int(np.random.randint(200, 800)) # Rookie
        weather_sev = int(np.random.randint(4, 7))   # Moderate weather makes it worse
        wind = int(np.random.randint(30, 50))

    return {
        "Aircraft_Age_Years": age,
        "Days_Since_Maintenance": maintenance,
        "Flight_Hours": int(np.random.randint(1000, 30000)),
        "Aircraft_Type": real_flight["aircraft"],
        "Flight_Distance_km": int(np.random.normal(base_dist, 200)),
        "Passenger_Count": int(np.random.randint(120, 400)) if is_long_haul else int(np.random.randint(80, 180)),
        "Cargo_Weight_tons": round(float(np.random.uniform(5, 20)), 1) if is_long_haul else round(float(np.random.uniform(0.5, 4)), 1),
        "Departure_Weather_Severity": weather_sev,
        "Destination_Weather_Severity": int(np.random.randint(0, 11)),
        "Wind_Speed_kmh": wind,
        "Pilot_Experience_Hours": pilot_exp, # Ensure this doesn't go below 0
        "Time_of_Day": real_flight["time"],
        "Route_Complexity": "High" if is_long_haul else np.random.choice(COMPLEXITY_INTERNAL),
        "Departure_Airport_Category": "Category1",
        "Destination_Airport_Category": np.random.choice(AIRPORT_INTERNAL),
        "Real_ID": real_flight["id"],
        "Real_Route": real_flight["route"]
    }


def predict_single_flight(flight, trained_models, scaler, label_encoders, feature_names):
    df = pd.DataFrame([flight])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    cat_cols = [
        "Aircraft_Type", "Time_of_Day", "Route_Complexity",
        "Departure_Airport_Category", "Destination_Airport_Category",
    ]
    for col in cat_cols:
        if col in df.columns and col in label_encoders:
            try:
                df[col] = label_encoders[col].transform([flight[col]])[0]
            except ValueError:
                df[col] = 0

    X_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)

    preds = {}
    for name, model in trained_models.items():
        risk = model.predict_proba(X_scaled)[0][1] * 100
        preds[name] = round(risk, 2)

    avg = round(np.mean(list(preds.values())), 2)
    if avg < 30:
        level, action = "LOW", "Cleared for Operation"
    elif avg < 70:
        level, action = "MEDIUM", "Review Required"
    else:
        level, action = "HIGH", "Delay / Inspection Required"

    return {"avg_risk": avg, "level": level, "action": action, "models": preds}


# =============================================================================
# INITIALIZATION (cached in session state)
# =============================================================================
def initialize():
    if "initialized" in st.session_state:
        return

    df = create_aviation_dataset(5000)
    X, y, scaler, label_encoders = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    trained_models, results = train_all_models(X_train, y_train, X_test, y_test)

    # Feature importance
    rf = trained_models["Random Forest"]
    fi = sorted(
        zip(list(X.columns), rf.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )[:10]

    # Best model
    best_name = max(results, key=lambda m: results[m]["accuracy"])
    best_model = trained_models[best_name]
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Risk distribution
    risk_counts = df["Risk_Level"].value_counts()

    st.session_state.update({
        "initialized": True,
        "trained_models": trained_models,
        "results": results,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "feature_names": list(X.columns),
        "feature_importance": [{"feature": f, "importance": round(v * 100, 2)} for f, v in fi],
        "confusion_matrix": cm,
        "risk_distribution": {
            "Low": int(risk_counts.get("Low", 0)),
            "Medium": int(risk_counts.get("Medium", 0)),
            "High": int(risk_counts.get("High", 0)),
        },
        "best_model_name": best_name,
        "df": df,
        "predictions": [],
        "history": [],
        "activity_log": [],
        "cycle": 0,
        "total_processed": 0,
        "high_risk_total": 0,
        "start_time": time.time(),
    })


# =============================================================================
# PROCESS ONE CYCLE
# =============================================================================
def process_cycle():
    s = st.session_state
    s["cycle"] += 1
    cycle = s["cycle"]
    now = datetime.now().strftime("%H:%M:%S")

    flight = generate_random_flight()
    # Use real-world data from our database
    route = flight["Real_Route"]
    flight_id = flight["Real_ID"]

    result = predict_single_flight(
        flight, s["trained_models"], s["scaler"], s["label_encoders"], s["feature_names"]
    )

    record = {
        "Flight ID": flight_id,
        "Route": route,
        "Aircraft": flight["Aircraft_Type"],
        "Risk (%)": result["avg_risk"],
        "Level": result["level"],
        "Action": result["action"],
        "Weather Severity": flight["Departure_Weather_Severity"],
        "Wind (km/h)": flight["Wind_Speed_kmh"],
        "Maintenance (days)": flight["Days_Since_Maintenance"],
        "Pilot Experience (hrs)": flight["Pilot_Experience_Hours"],
        "Aircraft Age": flight["Aircraft_Age_Years"],
        "Timestamp": now,
    }

    s["predictions"].insert(0, record)
    s["predictions"] = s["predictions"][:50]

    s["history"].append({"cycle": cycle, "risk": result["avg_risk"], "level": result["level"], "time": now})
    s["history"] = s["history"][-200:]

    s["total_processed"] += 1
    if result["level"] == "HIGH":
        s["high_risk_total"] += 1
    
    # Update Risk Distribution with new flight
    current_level = result["level"].capitalize() # "Low", "Medium", "High"
    if current_level in s["risk_distribution"]:
        s["risk_distribution"][current_level] += 1

    # Update Top Risk Factors based on recent traffic (Dynamic!)
    risky_flights = [p for p in s["predictions"] if p["Level"] in ["HIGH", "MEDIUM"]]
    if risky_flights:
        factors = {
            "Severe Weather": 0, "High Winds": 0, "Maintenance Overdue": 0,
            "Inexperienced Pilot": 0, "Aging Aircraft": 0
        }
        for p in risky_flights:
            if p["Weather Severity"] > 7:
                factors["Severe Weather"] += 1
            if p["Wind (km/h)"] > 40:
                factors["High Winds"] += 1
            if p["Maintenance (days)"] > 90:
                factors["Maintenance Overdue"] += 1
            if p["Pilot Experience (hrs)"] < 2000:
                factors["Inexperienced Pilot"] += 1
            if p["Aircraft Age"] > 15:
                factors["Aging Aircraft"] += 1
        
        # Convert to % of risky flights
        total_risky = len(risky_flights)
        s["feature_importance"] = [
            {"feature": k, "importance": round((v / total_risky) * 100, 1)}
            for k, v in factors.items() if v > 0
        ]
        s["feature_importance"].sort(key=lambda x: x["importance"], reverse=True)

    level_tag = {"LOW": "SAFE", "MEDIUM": "REVIEW", "HIGH": "ALERT"}
    log_msg = f"[{now}]  Cycle {cycle:>4d}  |  {flight_id}  |  {route:<28s}  |  {level_tag.get(result['level'], '----')}  {result['avg_risk']:>6.2f}%"
    s["activity_log"].insert(0, log_msg)
    s["activity_log"] = s["activity_log"][:50]


# =============================================================================
# DASHBOARD RENDER
# =============================================================================
def render_dashboard():
    s = st.session_state
    inject_css()

    # ---- HEADER ----
    st.markdown(
        f"""
        <div class="dash-header">
            <h1>Flight Safety Intelligence System</h1>
            <p>
                <span class="live-indicator"></span>
                Continuous ML Risk Prediction  &nbsp;&bull;&nbsp;  Auto-refreshing every {REFRESH_SECONDS}s
                &nbsp;&bull;&nbsp;  Cycle #{s['cycle']}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- STATUS BAR ----
    best = s["best_model_name"]
    best_acc = s["results"][best]["accuracy"]
    uptime = int(time.time() - s["start_time"])
    h, m, sec = uptime // 3600, (uptime % 3600) // 60, uptime % 60
    st.markdown(
        f"""<div class="status-bar">
            System Online  &nbsp;|&nbsp;  Best Model: <strong>{best}</strong> ({best_acc}% accuracy)
            &nbsp;|&nbsp;  Uptime: {h:02d}:{m:02d}:{sec:02d}
        </div>""",
        unsafe_allow_html=True,
    )

    # ---- KPI ROW ----
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Flights Processed", f"{s['total_processed']:,}")
    with k2:
        st.metric("High Risk Detected", s["high_risk_total"])
    with k3:
        st.metric("Best Model Accuracy", f"{best_acc}%")
    with k4:
        avg_risk = 0
        if s["predictions"]:
            avg_risk = np.mean([p["Risk (%)"] for p in s["predictions"]])
        st.metric("Avg Risk Score", f"{avg_risk:.1f}%")
    with k5:
        safe_rate = 0
        if s["total_processed"] > 0:
            safe_rate = ((s["total_processed"] - s["high_risk_total"]) / s["total_processed"]) * 100
        st.metric("Safe Flight Rate", f"{safe_rate:.1f}%")

    st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

    # ---- ROW 1: GAUGE + MODEL PERFORMANCE ----
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Latest Flight Risk Assessment</div>', unsafe_allow_html=True)
        if s["predictions"]:
            latest = s["predictions"][0]
            risk_val = latest["Risk (%)"]
            bar_color = COLOR_SAFE if risk_val < 30 else COLOR_WARN if risk_val < 70 else COLOR_DANGER

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_val,
                number={"suffix": "%", "font": {"size": 48, "color": "#1a1a2e", "family": "Inter"}},
                title={"text": f"{latest['Flight ID']}  —  {latest['Route']}", "font": {"size": 13, "color": "#78909c"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#90a4ae", "tickfont": {"color": "#78909c"}},
                    "bar": {"color": bar_color, "thickness": 0.25},
                    "bgcolor": "#f5f5f5",
                    "bordercolor": "#e0e0e0",
                    "steps": [
                        {"range": [0, 30], "color": "#e8f5e9"},
                        {"range": [30, 70], "color": "#fff3e0"},
                        {"range": [70, 100], "color": "#ffebee"},
                    ],
                    "threshold": {"line": {"color": COLOR_DANGER, "width": 3}, "thickness": 0.8, "value": risk_val},
                },
            ))
            # Update layout by merging dicts manually to avoid duplicate kwargs
            layout = PLOTLY_LAYOUT.copy()
            layout.update(height=370, margin=dict(l=30, r=30, t=60, b=10))
            fig_gauge.update_layout(**layout)
            st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{s['cycle']}")
        else:
            st.info("Waiting for first prediction...")

    with col_b:
        st.markdown('<div class="section-header">Model Performance Comparison</div>', unsafe_allow_html=True)
        if s["results"]:
            names = list(s["results"].keys())
            accs = [s["results"][n]["accuracy"] for n in names]
            f1s = [s["results"][n]["f1_score"] for n in names]
            aucs = [s["results"][n]["roc_auc"] for n in names]

            fig_model = go.Figure()
            fig_model.add_trace(go.Bar(x=names, y=accs, name="Accuracy (%)", marker_color=COLOR_PRIMARY,
                                       text=[f"{v}%" for v in accs], textposition="outside", textfont=dict(size=10)))
            fig_model.add_trace(go.Bar(x=names, y=f1s, name="F1 Score (%)", marker_color=COLOR_ACCENT,
                                       text=[f"{v}%" for v in f1s], textposition="outside", textfont=dict(size=10)))
            fig_model.add_trace(go.Bar(x=names, y=aucs, name="ROC-AUC (%)", marker_color=COLOR_SECONDARY,
                                       text=[f"{v}%" for v in aucs], textposition="outside", textfont=dict(size=10)))
            
            layout = PLOTLY_LAYOUT.copy()
            layout.update(
                barmode="group", height=370,
                yaxis=dict(range=[0, 112], gridcolor="#e0e0e0", title="Percentage (%)"),
                legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
            )
            fig_model.update_layout(**layout)
            st.plotly_chart(fig_model, use_container_width=True, key="model_perf")

    # ---- ROW 2: FEATURE IMPORTANCE + RISK DISTRIBUTION ----
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-header">Prevalent Risk Factors (Live)</div>', unsafe_allow_html=True)
        if s["feature_importance"]:
            fi = s["feature_importance"]
            f_names = [f["feature"] for f in fi][::-1]
            f_vals = [f["importance"] for f in fi][::-1]

            fig_feat = go.Figure(go.Bar(
                y=f_names, x=f_vals, orientation="h",
                marker=dict(
                    color=f_vals,
                    colorscale=[[0, "#e3f2fd"], [0.5, "#1565c0"], [1, "#0d47a1"]],
                    line=dict(width=0),
                ),
                text=[f"{v:.1f}%" for v in f_vals],
                textposition="outside",
                textfont=dict(size=10, color="#37474f"),
            ))
            
            layout = PLOTLY_LAYOUT.copy()
            layout.update(
                height=370,
                margin=dict(l=180, r=60, t=10, b=40),
                xaxis=dict(title="Frequency in Risky Flights (%)", gridcolor="#e0e0e0"),
            )
            fig_feat.update_layout(**layout)
            st.plotly_chart(fig_feat, use_container_width=True, key="features")
        else:
            st.info("Waiting for high-risk flights to analyze patterns...")

    with col_d:
        st.markdown('<div class="section-header">Risk Distribution (Live Session)</div>', unsafe_allow_html=True)
        rd = s["risk_distribution"]
        fig_dist = go.Figure(go.Pie(
            values=[rd["Low"], rd["Medium"], rd["High"]],
            labels=["Low Risk", "Medium Risk", "High Risk"],
            hole=0.5,
            marker=dict(
                colors=[COLOR_SAFE, COLOR_WARN, COLOR_DANGER],
                line=dict(color="white", width=3),
            ),
            textinfo="label+percent",
            textfont=dict(size=12, color="#37474f"),
            pull=[0, 0, 0.04],
        ))
        total = rd["Low"] + rd["Medium"] + rd["High"]
        
        layout = PLOTLY_LAYOUT.copy()
        layout.update(
            height=370,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", y=-0.08, font=dict(size=11)),
            annotations=[dict(text=f"<b>{total:,}</b><br>Flights", showarrow=False,
                              font=dict(size=16, color="#1a1a2e", family="Inter"))],
        )
        fig_dist.update_layout(**layout)
        st.plotly_chart(fig_dist, use_container_width=True, key="risk_dist")

    # ---- ROW 3: PREDICTION HISTORY TIMELINE ----
    st.markdown('<div class="section-header">Risk Prediction Timeline</div>', unsafe_allow_html=True)
    if s["history"]:
        hist = s["history"]
        cycles = [h["cycle"] for h in hist]
        risks = [h["risk"] for h in hist]
        colors = [COLOR_SAFE if r < 30 else COLOR_WARN if r < 70 else COLOR_DANGER for r in risks]

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=cycles, y=risks, mode="lines", name="Risk %",
            line=dict(color=COLOR_PRIMARY, width=2, shape="spline"),
            fill="tozeroy", fillcolor="rgba(21,101,192,0.06)",
        ))
        fig_hist.add_trace(go.Scatter(
            x=cycles, y=risks, mode="markers", showlegend=False,
            marker=dict(color=colors, size=5, line=dict(color="white", width=1)),
        ))
        # Threshold lines
        fig_hist.add_hline(y=30, line_dash="dash", line_color="#e65100", opacity=0.4,
                           annotation_text="Medium threshold", annotation_font_color="#e65100", annotation_font_size=10)
        fig_hist.add_hline(y=70, line_dash="dash", line_color="#c62828", opacity=0.4,
                           annotation_text="High threshold", annotation_font_color="#c62828", annotation_font_size=10)

        layout = PLOTLY_LAYOUT.copy()
        layout.update(
            height=320,
            xaxis=dict(title="Processing Cycle", gridcolor="#e0e0e0"),
            yaxis=dict(title="Risk (%)", range=[0, 105], gridcolor="#e0e0e0"),
            legend=dict(orientation="h", y=-0.2),
        )
        fig_hist.update_layout(**layout)
        st.plotly_chart(fig_hist, use_container_width=True, key=f"history_{s['cycle']}")
    else:
        st.info("Timeline will appear after the first prediction cycle.")

    # ---- ROW 4: CONFUSION MATRIX + ACTIVITY LOG ----
    # ---- ROW 4: PROCESSING ACTIVITY LOG (Full Width) ----
    st.markdown('<div class="section-header">Processing Activity Log</div>', unsafe_allow_html=True)
    if s["activity_log"]:
        # Reverse log to show newest first (already done in insertion, but ensure display is clean)
        log_html = "".join(
            f'<div class="{"latest" if i == 0 else ""}" style="border-bottom: 1px solid #e0e0e0; padding: 4px 0;">{entry}</div>'
            for i, entry in enumerate(s["activity_log"])
        )
        st.markdown(f'<div class="log-box" style="height: 300px;">{log_html}</div>', unsafe_allow_html=True)
    else:
        st.info("Log will appear after the first prediction cycle.")

    # ---- ROW 5: PREDICTIONS TABLE ----
    st.markdown('<div class="section-header">Live Flight Risk Predictions</div>', unsafe_allow_html=True)
    if s["predictions"]:
        df_pred = pd.DataFrame(s["predictions"])

        def color_risk(val):
            if val == "LOW":
                return "background-color: #e8f5e9; color: #2e7d32; font-weight: 600"
            elif val == "MEDIUM":
                return "background-color: #fff3e0; color: #e65100; font-weight: 600"
            else:
                return "background-color: #ffebee; color: #c62828; font-weight: 600"

        def color_risk_pct(val):
            if val < 30:
                return "color: #2e7d32; font-weight: 700"
            elif val < 70:
                return "color: #e65100; font-weight: 700"
            else:
                return "color: #c62828; font-weight: 700"

        styled = df_pred.style.applymap(color_risk, subset=["Level"]).applymap(color_risk_pct, subset=["Risk (%)"])
        st.dataframe(styled, use_container_width=True, height=460, hide_index=True)
    else:
        st.info("Predictions will appear after the first cycle.")

    # ---- FOOTER ----
    st.markdown(
        """
        <div class="footer">
            <p><strong>Developed by:</strong> Syamala Gowtham Reddy</p>
            <p>Sathyabama Institute of Science and Technology &bull; B.E. Computer Science & Engineering (Data Science)</p>
            <p style="margin-top: 6px;">Integrated Flight Safety & Risk Analysis System &bull; Machine Learning Powered</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# MAIN
# =============================================================================
def main():
    st.set_page_config(
        page_title="Flight Safety Intelligence System",
        page_icon="",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    initialize()
    process_cycle()
    render_dashboard()

    # Auto-refresh
    time.sleep(REFRESH_SECONDS)
    st.rerun()


if __name__ == "__main__":
    main()
