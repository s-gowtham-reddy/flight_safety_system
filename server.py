"""
FLIGHT SAFETY SYSTEM — CONTINUOUS LIVE SERVER
Trains ML models on startup, then continuously generates & predicts
new flight data every 5 seconds. Serves a professional live dashboard.

Author: Syamala Gowtham Reddy
"""

import pandas as pd
import numpy as np
import warnings
import json
import threading
import time
from datetime import datetime, timedelta
from flask import Flask, jsonify, send_file, Response

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# =============================================================================
# GLOBAL STATE
# =============================================================================
app = Flask(__name__)

state = {
    "trained_models": {},
    "results": {},
    "scaler": None,
    "label_encoders": {},
    "feature_names": [],
    "X_train": None,
    "y_train": None,
    "X_test": None,
    "y_test": None,
    "df": None,
    "feature_importance": [],
    "confusion_matrix": [],
    "predictions": [],       # latest batch of predictions
    "history": [],           # timeline of risk scores
    "activity_log": [],      # processing log
    "cycle": 0,
    "start_time": None,
    "total_flights_processed": 0,
    "high_risk_total": 0,
    "lock": threading.Lock(),
}

AIRCRAFT_TYPES = ['Boeing737', 'Airbus320', 'Boeing777', 'Airbus330', 'Boeing787']
TIME_OF_DAY = ['Morning', 'Afternoon', 'Evening', 'Night']
ROUTE_COMPLEXITY = ['Low', 'Medium', 'High']
AIRPORT_CATS = ['Category1', 'Category2', 'Category3']
ROUTES = [
    'Mumbai → Delhi', 'Delhi → Kolkata', 'Chennai → Bangalore',
    'Hyderabad → Mumbai', 'Bangalore → Chennai', 'Kolkata → Mumbai',
    'Delhi → Chennai', 'Mumbai → Bangalore', 'Chennai → Delhi',
    'Pune → Delhi', 'Jaipur → Mumbai', 'Goa → Delhi',
    'Lucknow → Bangalore', 'Ahmedabad → Chennai', 'Bhopal → Hyderabad',
]

# =============================================================================
# DATA GENERATION
# =============================================================================
def create_aviation_dataset(n_samples=5000):
    np.random.seed(42)
    data = {
        'Aircraft_Age_Years': np.random.randint(1, 25, n_samples),
        'Days_Since_Maintenance': np.random.randint(0, 180, n_samples),
        'Flight_Hours': np.random.randint(1000, 50000, n_samples),
        'Aircraft_Type': np.random.choice(AIRCRAFT_TYPES, n_samples),
        'Flight_Distance_km': np.random.randint(200, 5000, n_samples),
        'Passenger_Count': np.random.randint(50, 350, n_samples),
        'Cargo_Weight_tons': np.random.uniform(0, 20, n_samples),
        'Departure_Weather_Severity': np.random.randint(0, 11, n_samples),
        'Destination_Weather_Severity': np.random.randint(0, 11, n_samples),
        'Wind_Speed_kmh': np.random.randint(0, 80, n_samples),
        'Pilot_Experience_Hours': np.random.randint(500, 15000, n_samples),
        'Time_of_Day': np.random.choice(TIME_OF_DAY, n_samples),
        'Route_Complexity': np.random.choice(ROUTE_COMPLEXITY, n_samples),
        'Departure_Airport_Category': np.random.choice(AIRPORT_CATS, n_samples),
        'Destination_Airport_Category': np.random.choice(AIRPORT_CATS, n_samples),
    }
    df = pd.DataFrame(data)

    risk_score = (
        (df['Aircraft_Age_Years'] > 15).astype(int) * 10 +
        (df['Days_Since_Maintenance'] > 90).astype(int) * 15 +
        (df['Departure_Weather_Severity'] > 7).astype(int) * 20 +
        (df['Destination_Weather_Severity'] > 7).astype(int) * 20 +
        (df['Wind_Speed_kmh'] > 50).astype(int) * 15 +
        (df['Pilot_Experience_Hours'] < 2000).astype(int) * 10 +
        (df['Route_Complexity'] == 'High').astype(int) * 10 +
        np.random.randint(-10, 10, n_samples)
    )
    df['Risk_Level'] = pd.cut(risk_score, bins=[-np.inf, 30, 60, np.inf], labels=['Low', 'Medium', 'High'])
    df['High_Risk'] = (df['Risk_Level'] == 'High').astype(int)
    return df

# =============================================================================
# PREPROCESSING
# =============================================================================
def preprocess_data(df):
    X = df.drop(['Risk_Level', 'High_Risk'], axis=1)
    y = df['High_Risk']

    categorical_columns = ['Aircraft_Type', 'Time_of_Day', 'Route_Complexity',
                           'Departure_Airport_Category', 'Destination_Airport_Category']

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
def train_models(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1),
    }

    trained_models = {}
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
            'f1_score': round(f1_score(y_test, y_pred) * 100, 2),
            'roc_auc': round(roc_auc_score(y_test, y_proba) * 100, 2),
        }

    # Ensemble
    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    trained_models['Ensemble'] = ensemble

    y_pred_ens = ensemble.predict(X_test)
    y_proba_ens = ensemble.predict_proba(X_test)[:, 1]
    results['Ensemble'] = {
        'accuracy': round(accuracy_score(y_test, y_pred_ens) * 100, 2),
        'f1_score': round(f1_score(y_test, y_pred_ens) * 100, 2),
        'roc_auc': round(roc_auc_score(y_test, y_proba_ens) * 100, 2),
    }

    return trained_models, results

# =============================================================================
# GENERATE A SINGLE RANDOM TEST FLIGHT
# =============================================================================
def generate_random_flight():
    return {
        'Aircraft_Age_Years': int(np.random.randint(1, 25)),
        'Days_Since_Maintenance': int(np.random.randint(0, 180)),
        'Flight_Hours': int(np.random.randint(1000, 50000)),
        'Aircraft_Type': np.random.choice(AIRCRAFT_TYPES),
        'Flight_Distance_km': int(np.random.randint(200, 5000)),
        'Passenger_Count': int(np.random.randint(50, 350)),
        'Cargo_Weight_tons': round(float(np.random.uniform(0, 20)), 1),
        'Departure_Weather_Severity': int(np.random.randint(0, 11)),
        'Destination_Weather_Severity': int(np.random.randint(0, 11)),
        'Wind_Speed_kmh': int(np.random.randint(0, 80)),
        'Pilot_Experience_Hours': int(np.random.randint(500, 15000)),
        'Time_of_Day': np.random.choice(TIME_OF_DAY),
        'Route_Complexity': np.random.choice(ROUTE_COMPLEXITY),
        'Departure_Airport_Category': np.random.choice(AIRPORT_CATS),
        'Destination_Airport_Category': np.random.choice(AIRPORT_CATS),
    }

# =============================================================================
# PREDICT RISK FOR A SINGLE FLIGHT
# =============================================================================
def predict_flight(flight_data):
    df = pd.DataFrame([flight_data])
    feature_names = state["feature_names"]
    label_encoders = state["label_encoders"]
    scaler = state["scaler"]

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    categorical_columns = ['Aircraft_Type', 'Time_of_Day', 'Route_Complexity',
                           'Departure_Airport_Category', 'Destination_Airport_Category']
    for col in categorical_columns:
        if col in df.columns and col in label_encoders:
            try:
                df[col] = label_encoders[col].transform([flight_data[col]])[0]
            except ValueError:
                df[col] = 0

    X_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)

    predictions = {}
    for name, model in state["trained_models"].items():
        risk = model.predict_proba(X_scaled)[0][1] * 100
        predictions[name] = round(risk, 2)

    avg_risk = round(np.mean(list(predictions.values())), 2)

    if avg_risk < 30:
        level, action = "LOW", "Cleared"
    elif avg_risk < 70:
        level, action = "MEDIUM", "Review Required"
    else:
        level, action = "HIGH", "Delay / Inspect"

    return {
        'avg_risk': avg_risk,
        'level': level,
        'action': action,
        'model_predictions': predictions,
    }

# =============================================================================
# BACKGROUND PROCESSING LOOP — runs every 5 seconds
# =============================================================================
def process_cycle():
    with state["lock"]:
        state["cycle"] += 1
        cycle = state["cycle"]
        now = datetime.now().strftime("%H:%M:%S")

        # Generate a random flight
        flight = generate_random_flight()
        route = np.random.choice(ROUTES)
        flight_id = f"AI-{2000 + cycle}"

        # Predict
        result = predict_flight(flight)

        # Build prediction record
        record = {
            "id": flight_id,
            "route": route,
            "aircraft": flight["Aircraft_Type"],
            "risk": result["avg_risk"],
            "level": result["level"],
            "action": result["action"],
            "weather": flight["Departure_Weather_Severity"],
            "wind": flight["Wind_Speed_kmh"],
            "maintenance_days": flight["Days_Since_Maintenance"],
            "pilot_exp": flight["Pilot_Experience_Hours"],
            "time": now,
            "models": result["model_predictions"],
        }

        # Add to predictions (keep last 30)
        state["predictions"].insert(0, record)
        state["predictions"] = state["predictions"][:30]

        # Add to history timeline (keep last 200)
        state["history"].append({
            "cycle": cycle,
            "risk": result["avg_risk"],
            "level": result["level"],
            "time": now,
        })
        state["history"] = state["history"][-200:]

        # Update totals
        state["total_flights_processed"] += 1
        if result["level"] == "HIGH":
            state["high_risk_total"] += 1

        # Activity log (keep last 50)
        level_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
        log_msg = f"[{now}] Cycle #{cycle} — {flight_id} ({route}) → {level_emoji.get(result['level'], '⚪')} {result['level']} ({result['avg_risk']}%)"
        state["activity_log"].insert(0, log_msg)
        state["activity_log"] = state["activity_log"][:50]

        print(f"  Cycle {cycle}: {flight_id} | {route} | Risk: {result['avg_risk']}% ({result['level']})")

    # Schedule next cycle
    timer = threading.Timer(5.0, process_cycle)
    timer.daemon = True
    timer.start()

# =============================================================================
# INITIAL TRAINING
# =============================================================================
def initialize_system():
    print("=" * 70)
    print("  FLIGHT SAFETY SYSTEM — CONTINUOUS LIVE SERVER")
    print("=" * 70)

    print("\n  [1/4] Generating aviation dataset...")
    df = create_aviation_dataset(5000)
    print(f"        ✓ {df.shape[0]} flights generated")

    print("\n  [2/4] Preprocessing data...")
    X, y, scaler, label_encoders = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"        ✓ Train: {len(X_train)}, Test: {len(X_test)}")

    print("\n  [3/4] Training ML models...")
    trained_models, results = train_models(X_train, y_train, X_test, y_test)
    for name, m in results.items():
        print(f"        {name}: Acc={m['accuracy']}% | F1={m['f1_score']}% | AUC={m['roc_auc']}%")

    # Feature importance from Random Forest
    rf = trained_models['Random Forest']
    fi = sorted(
        zip(list(X.columns), rf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )[:10]

    # Confusion matrix from best model
    best_name = max(results, key=lambda m: results[m]['accuracy'])
    best_model = trained_models[best_name]
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Risk distribution from dataset
    risk_counts = df['Risk_Level'].value_counts()
    risk_dist = {
        "Low": int(risk_counts.get('Low', 0)),
        "Medium": int(risk_counts.get('Medium', 0)),
        "High": int(risk_counts.get('High', 0)),
    }

    # Store state
    state["trained_models"] = trained_models
    state["results"] = results
    state["scaler"] = scaler
    state["label_encoders"] = label_encoders
    state["feature_names"] = list(X.columns)
    state["X_train"] = X_train
    state["y_train"] = y_train
    state["X_test"] = X_test
    state["y_test"] = y_test
    state["df"] = df
    state["feature_importance"] = [{"feature": f, "importance": round(v * 100, 2)} for f, v in fi]
    state["confusion_matrix"] = cm
    state["risk_distribution"] = risk_dist
    state["start_time"] = time.time()
    state["best_model_name"] = best_name

    print(f"\n  [4/4] Starting continuous prediction loop (every 5s)...")
    print(f"        Best Model: {best_name} ({results[best_name]['accuracy']}%)")
    print(f"\n  ✅ System ready! Open http://localhost:5000 in your browser")
    print("=" * 70)

    # Start background loop
    timer = threading.Timer(2.0, process_cycle)
    timer.daemon = True
    timer.start()

# =============================================================================
# API ROUTES
# =============================================================================
@app.route('/')
def serve_dashboard():
    return send_file('dashboard.html')

@app.route('/api/dashboard')
def api_dashboard():
    with state["lock"]:
        uptime = 0
        if state["start_time"]:
            uptime = int(time.time() - state["start_time"])

        data = {
            "cycle": state["cycle"],
            "uptime": uptime,
            "total_processed": state["total_flights_processed"],
            "high_risk_total": state["high_risk_total"],
            "models": state["results"],
            "best_model": state.get("best_model_name", ""),
            "features": state["feature_importance"],
            "confusion_matrix": state["confusion_matrix"],
            "risk_distribution": state.get("risk_distribution", {}),
            "predictions": state["predictions"],
            "history": state["history"],
            "activity_log": state["activity_log"],
        }
    return jsonify(data)

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    initialize_system()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
