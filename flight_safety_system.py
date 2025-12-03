"""
INTEGRATED FLIGHT SAFETY AND RISK ANALYSIS SYSTEM
Author: Syamala Gowtham Reddy
Institution: Sathyabama Institute of Science and Technology

Performance Achieved:
- Accuracy: 97.20% (Target: 91.2%) ✅
- F1-Score: 0.7143
- ROC-AUC: 0.9847
"""

# ============================================================================
# IMPORTS
# ============================================================================

import pandas as pd
import numpy as np
import warnings
import webbrowser
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

# ============================================================================
# DATA GENERATION
# ============================================================================

def create_aviation_dataset(n_samples=5000):
    """Create realistic aviation safety dataset"""
    np.random.seed(42)

    data = {
        'Aircraft_Age_Years': np.random.randint(1, 25, n_samples),
        'Days_Since_Maintenance': np.random.randint(0, 180, n_samples),
        'Flight_Hours': np.random.randint(1000, 50000, n_samples),
        'Aircraft_Type': np.random.choice(['Boeing737', 'Airbus320', 'Boeing777', 'Airbus330', 'Boeing787'], n_samples),
        'Flight_Distance_km': np.random.randint(200, 5000, n_samples),
        'Passenger_Count': np.random.randint(50, 350, n_samples),
        'Cargo_Weight_tons': np.random.uniform(0, 20, n_samples),
        'Departure_Weather_Severity': np.random.randint(0, 11, n_samples),
        'Destination_Weather_Severity': np.random.randint(0, 11, n_samples),
        'Wind_Speed_kmh': np.random.randint(0, 80, n_samples),
        'Pilot_Experience_Hours': np.random.randint(500, 15000, n_samples),
        'Time_of_Day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n_samples),
        'Route_Complexity': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'Departure_Airport_Category': np.random.choice(['Category1', 'Category2', 'Category3'], n_samples),
        'Destination_Airport_Category': np.random.choice(['Category1', 'Category2', 'Category3'], n_samples),
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

# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess_data(df):
    """Preprocess and prepare data for ML models"""
    X = df.drop(['Risk_Level', 'High_Risk'], axis=1)
    y = df['High_Risk']
    
    # Create label encoders for categorical columns
    categorical_columns = ['Aircraft_Type', 'Time_of_Day', 'Route_Complexity', 
                          'Departure_Airport_Category', 'Destination_Airport_Category']
    
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Create and fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    # RETURN SCALER AND ENCODERS
    return X, y, scaler, label_encoders


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_models(X_train, y_train):
    """Train all ML models"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1)
    }

    trained_models = {}
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }

    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    trained_models['Ensemble'] = ensemble

    y_pred_ens = ensemble.predict(X_test)
    y_proba_ens = ensemble.predict_proba(X_test)[:, 1]
    results['Ensemble'] = {
        'accuracy': accuracy_score(y_test, y_pred_ens),
        'f1_score': f1_score(y_test, y_pred_ens),
        'roc_auc': roc_auc_score(y_test, y_proba_ens)
    }

    return trained_models, results

# ============================================================================
# COMPREHENSIVE DASHBOARD GENERATION
# ============================================================================

def generate_comprehensive_dashboard(df, X, y, X_test, y_test, trained_models, results):
    """Generate complete professional dashboard with all visualizations working"""
    
    total_flights = len(df)
    high_risk_count = df["High_Risk"].sum()
    safe_count = total_flights - high_risk_count
    high_risk_pct = round((high_risk_count / total_flights) * 100, 1)
    
    best_model_name = max(results, key=lambda m: results[m]["accuracy"])
    best_model = trained_models[best_model_name]
    best_acc = round(results[best_model_name]["accuracy"] * 100, 2)
    best_f1 = round(results[best_model_name]["f1_score"], 4)
    best_roc = round(results[best_model_name]["roc_auc"], 4)
    
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    rf_model = trained_models['Random Forest']
    feature_names = list(X.columns)
    feature_importance = list(rf_model.feature_importances_)
    
    feature_data = list(zip(feature_names, feature_importance))
    feature_data_sorted = sorted(feature_data, key=lambda x: x[1], reverse=True)[:10]
    top_features = [f[0] for f in feature_data_sorted]
    top_importance = [round(f[1] * 100, 2) for f in feature_data_sorted]
    
    sample_predictions = []
    routes = ['Mumbai→Delhi', 'Delhi→Kolkata', 'Chennai→Bangalore', 
              'Hyderabad→Mumbai', 'Bangalore→Chennai', 'Kolkata→Mumbai',
              'Delhi→Chennai', 'Mumbai→Bangalore', 'Chennai→Delhi', 'Pune→Delhi']
    
    for i in range(min(15, len(X_test))):
        flight_data = X_test.iloc[i:i+1]
        risk_proba = best_model.predict_proba(flight_data)[0][1] * 100
        actual = y_test.iloc[i]
        
        if risk_proba < 30:
            level, action = 'LOW', 'Cleared'
        elif risk_proba < 70:
            level, action = 'MEDIUM', 'Review'
        else:
            level, action = 'HIGH', 'Delay/Inspect'
        
        sample_predictions.append({
            'id': f'AI-{1000+i}',
            'route': routes[i % len(routes)],
            'risk': round(risk_proba, 2),
            'level': level,
            'action': action,
            'actual': 'High Risk' if actual == 1 else 'Safe'
        })
    
    model_names = list(results.keys())
    model_accs = [round(results[m]["accuracy"] * 100, 2) for m in model_names]
    model_f1s = [round(results[m]["f1_score"] * 100, 2) for m in model_names]
    
    sample_flight_risk = sample_predictions[0]['risk'] if sample_predictions else 50
    gauge_color = '#27ae60' if sample_flight_risk < 30 else '#f39c12' if sample_flight_risk < 70 else '#e74c3c'
    
    # Convert to JSON-safe format for JavaScript
    import json
    model_names_json = json.dumps(model_names)
    model_accs_json = json.dumps(model_accs)
    model_f1s_json = json.dumps(model_f1s)
    top_features_json = json.dumps(top_features)
    top_importance_json = json.dumps(top_importance)
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Flight Safety Dashboard</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI';background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:20px;min-height:100vh}}
.container{{max-width:1600px;margin:0 auto}}
.header{{background:white;padding:30px;border-radius:15px;box-shadow:0 8px 32px rgba(0,0,0,0.1);margin-bottom:30px;text-align:center}}
.header h1{{color:#2c3e50;font-size:2.5em;margin-bottom:10px}}
.header p{{color:#7f8c8d;font-size:1.1em}}
.stats-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:20px;margin-bottom:30px}}
.stat-card{{background:white;padding:25px;border-radius:15px;box-shadow:0 8px 32px rgba(0,0,0,0.1);text-align:center;transition:transform 0.3s}}
.stat-card:hover{{transform:translateY(-5px)}}
.stat-value{{font-size:2.5em;font-weight:bold;margin:10px 0}}
.stat-label{{color:#7f8c8d;font-size:0.9em;text-transform:uppercase;letter-spacing:1px}}
.green{{color:#27ae60}}.yellow{{color:#f39c12}}.red{{color:#e74c3c}}.blue{{color:#3498db}}
.chart-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:25px;margin-bottom:30px}}
@media(max-width:1200px){{.chart-grid{{grid-template-columns:1fr}}}}
.chart-container{{background:white;padding:25px;border-radius:15px;box-shadow:0 8px 32px rgba(0,0,0,0.1);min-height:450px;display:flex;flex-direction:column}}
.chart-title{{font-size:1.3em;font-weight:bold;color:#2c3e50;margin-bottom:15px;padding-bottom:10px;border-bottom:2px solid #667eea}}
.chart-inner{{flex:1;position:relative}}
.full-width{{grid-column:1/-1}}
.alert-box{{background:#d4edda;border-left:5px solid #27ae60;padding:20px;border-radius:10px;margin-bottom:30px;color:#155724;font-weight:500}}
.flight-table{{width:100%;border-collapse:collapse;margin-top:20px}}
.flight-table th{{background:#3498db;color:white;padding:15px;text-align:left;font-weight:600}}
.flight-table td{{padding:12px 15px;border-bottom:1px solid #ecf0f1;font-size:0.95em}}
.flight-table tr:hover{{background:#f8f9fa}}
.risk-badge{{padding:5px 15px;border-radius:20px;font-weight:bold;font-size:0.85em;display:inline-block}}
.risk-low{{background:#d4edda;color:#27ae60}}
.risk-medium{{background:#fff3cd;color:#f39c12}}
.risk-high{{background:#f8d7da;color:#e74c3c}}
.confusion-matrix{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:15px}}
.matrix-cell{{padding:20px;text-align:center;border-radius:10px;font-weight:bold;color:white;font-size:1.1em}}
.tn{{background:#27ae60}}.fp{{background:#e74c3c}}.fn{{background:#f39c12}}.tp{{background:#3498db}}
.footer{{text-align:center;color:white;margin-top:30px;padding:20px}}
</style></head><body>
<div class="container">
<div class="header">
<h1>✈️ Flight Safety & Risk Analysis Dashboard</h1>
<p>Real-time ML Model Results - Live Run</p>
</div>

<div class="alert-box">
<strong>✅ STATUS:</strong> All {len(model_names)} models trained successfully. 
Best Model: <strong>{best_model_name}</strong> | Accuracy: <strong>{best_acc}%</strong>
</div>

<div class="stats-grid">
<div class="stat-card">
<div class="stat-label">Best Accuracy</div>
<div class="stat-value blue">{best_acc}%</div>
<p style="color:#7f8c8d;margin-top:10px">{best_model_name}</p>
</div>
<div class="stat-card">
<div class="stat-label">Total Flights</div>
<div class="stat-value">{total_flights:,}</div>
</div>
<div class="stat-card">
<div class="stat-label">High Risk</div>
<div class="stat-value red">{high_risk_count}</div>
<p style="color:#7f8c8d;margin-top:10px">{high_risk_pct}%</p>
</div>
<div class="stat-card">
<div class="stat-label">F1-Score / ROC-AUC</div>
<div class="stat-value green">{best_f1} / {best_roc}</div>
</div>
</div>

<div class="chart-grid">
<div class="chart-container">
<div class="chart-title">📊 Model Performance</div>
<div class="chart-inner" id="modelChart"></div>
</div>

<div class="chart-container">
<div class="chart-title">🎯 Risk Distribution</div>
<div class="chart-inner" id="riskDist"></div>
</div>

<div class="chart-container">
<div class="chart-title">🔍 Top 10 Risk Factors</div>
<div class="chart-inner" id="featureChart"></div>
</div>

<div class="chart-container">
<div class="chart-title">📊 Confusion Matrix</div>
<div class="confusion-matrix">
<div class="matrix-cell tn">TN<br><strong>{cm[0][0]}</strong></div>
<div class="matrix-cell fp">FP<br><strong>{cm[0][1]}</strong></div>
<div class="matrix-cell fn">FN<br><strong>{cm[1][0]}</strong></div>
<div class="matrix-cell tp">TP<br><strong>{cm[1][1]}</strong></div>
</div>
</div>
</div>

<div class="chart-container full-width">
<div class="chart-title">🛫 Live Predictions (Test Set Sample)</div>
<table class="flight-table">
<thead>
<tr>
<th>Flight ID</th>
<th>Route</th>
<th>Risk %</th>
<th>Risk Level</th>
<th>Action</th>
<th>Actual Outcome</th>
</tr>
</thead>
<tbody>
"""
    
    for f in sample_predictions:
        rc = f"risk-{f['level'].lower()}"
        html += f"<tr><td><strong>{f['id']}</strong></td><td>{f['route']}</td><td><strong>{f['risk']}%</strong></td><td><span class='risk-badge {rc}'>{f['level']}</span></td><td>{f['action']}</td><td>{f['actual']}</td></tr>"
    
    html += f"""</tbody>
</table>
</div>

<div class="footer">
<p><strong>Developed by:</strong> Syamala Gowtham Reddy</p>
<p>Sathyabama Institute | B.E. Computer Science & Engineering (Data Science)</p>
</div>
</div>

<script>
// Model Performance Chart
var modelData = {{
    x: {model_names_json},
    y: {model_accs_json},
    type: 'bar',
    marker: {{color: '#3498db'}}
}};
Plotly.newPlot('modelChart', [modelData], {{
    title: {{}},
    xaxis: {{title: 'Model'}},
    yaxis: {{title: 'Accuracy (%)', range: [0, 105]}},
    plot_bgcolor: '#f8f9fa',
    margin: {{l: 60, r: 20, t: 30, b: 60}}
}}, {{responsive: true, displayModeBar: false}});

// Risk Distribution Pie Chart
var riskData = {{
    values: [{safe_count}, {high_risk_count}],
    labels: ['Safe Flights', 'High Risk Flights'],
    type: 'pie',
    marker: {{colors: ['#27ae60', '#e74c3c']}},
    textinfo: 'label+percent+value',
    hole: 0.4
}};
Plotly.newPlot('riskDist', [riskData], {{
    title: {{}},
    paper_bgcolor: 'white',
    margin: {{l: 20, r: 20, t: 30, b: 20}}
}}, {{responsive: true, displayModeBar: false}});

// Top Risk Factors - Horizontal Bar Chart
var featureData = {{
    y: {top_features_json},
    x: {top_importance_json},
    type: 'bar',
    orientation: 'h',
    marker: {{
        color: {top_importance_json},
        colorscale: 'Reds',
        showscale: false
    }},
    text: {[str(x) + '%' for x in top_importance]},
    textposition: 'outside'
}};
Plotly.newPlot('featureChart', [featureData], {{
    title: {{}},
    xaxis: {{title: 'Impact (%)'}},
    yaxis: {{title: ''}},
    plot_bgcolor: '#f8f9fa',
    margin: {{l: 200, r: 80, t: 30, b: 50}}
}}, {{responsive: true, displayModeBar: false}});
</script>

</body></html>"""
    
    with open("flight_safety_dashboard_live.html", "w", encoding="utf-8") as f:
        f.write(html)
    
    print("\n🌐 Launching comprehensive dashboard...")
    try:
        webbrowser.open("flight_safety_dashboard_live.html")
    except:
        print("   ✓ Open flight_safety_dashboard_live.html manually in your browser")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("INTEGRATED FLIGHT SAFETY AND RISK ANALYSIS SYSTEM")
    print("="*70)

    print("\n1. Creating aviation dataset...")
    df = create_aviation_dataset(5000)
    print(f"   ✓ Dataset created: {df.shape[0]} flights")

    print("\n2. Preprocessing data...")
    X, y, scaler, label_encoders = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"   ✓ Training: {len(X_train)}, Test: {len(X_test)}")

    print("\n3. Training ML models...")
    trained_models, results = train_models(X_train, y_train)
    print("   ✓ All models trained")

    print("\n4. Model Performance:")
    for name, metrics in results.items():
        print(f"   {name}:")
        print(f"      Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"      F1-Score: {metrics['f1_score']:.4f}")

    print("\n✅ System Ready for Deployment!")

    generate_comprehensive_dashboard(df, X, y, X_test, y_test, trained_models, results)
    # ADD AT END OF flight_safety_system.py
import joblib
import os

print("\n" + "="*70)
print("SAVING TRAINED MODELS FOR FUTURE USE")
print("="*70)

# Create saved_models folder
os.makedirs("saved_models", exist_ok=True)

# Save all trained models
joblib.dump(trained_models, "saved_models/trained_models.pkl")
print("✓ Trained models saved")

# Save the scaler
joblib.dump(scaler, "saved_models/scaler.pkl")
print("✓ StandardScaler saved")

# Save label encoders
joblib.dump(label_encoders, "saved_models/label_encoders.pkl")
print("✓ Label encoders saved")

# Save feature names
joblib.dump(list(X.columns), "saved_models/feature_names.pkl")
print("✓ Feature names saved")

print("\n✅ ALL MODELS SAVED!")
print("   Location: saved_models/")
print("   Now use 'predict_new_flights.py' for predictions")
