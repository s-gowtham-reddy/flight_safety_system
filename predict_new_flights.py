"""
PREDICT NEW FLIGHTS
Load pre-trained models and predict risk for new flights
"""

import pandas as pd
import numpy as np
import joblib
import webbrowser
import json

print("="*70)
print("LOADING TRAINED MODELS")
print("="*70)

try:
    trained_models = joblib.load("saved_models/trained_models.pkl")
    scaler = joblib.load("saved_models/scaler.pkl")
    label_encoders = joblib.load("saved_models/label_encoders.pkl")
    feature_names = joblib.load("saved_models/feature_names.pkl")
    print("✓ All models loaded successfully!")
except FileNotFoundError:
    print("❌ Models not found! Run flight_safety_system.py first.")
    exit()

def prepare_flight(flight_data):
    """Prepare new flight data for prediction"""
    df = pd.DataFrame([flight_data])
    
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    df = df[feature_names]
    
    categorical_columns = ['Aircraft_Type', 'Time_of_Day', 'Route_Complexity', 
                          'Departure_Airport_Category', 'Destination_Airport_Category']
    
    for col in categorical_columns:
        if col in df.columns and col in label_encoders:
            df[col] = label_encoders[col].transform([flight_data[col]])[0]
    
    X_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
    return X_scaled

def predict_risk(flight_data):
    """Get risk prediction"""
    X_new = prepare_flight(flight_data)
    
    predictions = {}
    for name, model in trained_models.items():
        risk = model.predict_proba(X_new)[0][1] * 100
        predictions[name] = round(risk, 2)
    
    avg_risk = np.mean(list(predictions.values()))
    
    if avg_risk < 30:
        level, color, action = "LOW", "#27ae60", "✅ CLEARED"
    elif avg_risk < 70:
        level, color, action = "MEDIUM", "#f39c12", "⚠️ REVIEW"
    else:
        level, color, action = "HIGH", "#e74c3c", "🚫 DELAY"
    
    return {
        'avg_risk': round(avg_risk, 2),
        'level': level,
        'color': color,
        'action': action,
        'predictions': predictions
    }

def show_results(result, flight_data):
    """Display results in browser"""
    model_names = list(result['predictions'].keys())
    model_risks = list(result['predictions'].values())
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Flight Risk Prediction</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        *{{margin:0;padding:0;box-sizing:border-box}}
        body{{font-family:'Segoe UI';background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:20px;min-height:100vh}}
        .container{{max-width:1000px;margin:0 auto}}
        .header{{background:white;padding:30px;border-radius:15px;margin-bottom:30px;box-shadow:0 8px 32px rgba(0,0,0,0.1)}}
        .header h1{{color:#2c3e50;margin-bottom:10px}}
        .alert{{background:{result['color']};color:white;padding:25px;border-radius:15px;margin-bottom:30px;font-size:1.2em;font-weight:bold}}
        .risk-card{{background:white;padding:30px;border-radius:15px;text-align:center;margin-bottom:30px;border-top:5px solid {result['color']};box-shadow:0 8px 32px rgba(0,0,0,0.1)}}
        .risk-value{{font-size:3.5em;font-weight:bold;color:{result['color']};margin:20px 0}}
        .card{{background:white;padding:25px;border-radius:15px;box-shadow:0 8px 32px rgba(0,0,0,0.1);margin-bottom:30px}}
        .card h3{{color:#2c3e50;margin-bottom:20px;border-bottom:2px solid #667eea;padding-bottom:10px}}
        .chart{{height:350px}}
        .models{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:15px;margin-top:20px}}
        .model-box{{background:#f8f9fa;padding:20px;border-radius:10px;text-align:center;border-top:3px solid #667eea}}
        .model-risk{{font-size:1.8em;font-weight:bold;margin:10px 0;color:#e74c3c}}
        .footer{{text-align:center;color:white;margin-top:30px}}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>✈️ Flight Risk Prediction</h1>
        <p>{flight_data.get('Aircraft_Type','Unknown')} | {flight_data.get('Flight_Distance_km',0)}km</p>
    </div>
    
    <div class="alert">{result['action']} - Risk: {result['avg_risk']}%</div>
    
    <div class="risk-card">
        <div style="font-size:1.5em;color:#2c3e50;font-weight:bold">{result['level']} RISK</div>
        <div class="risk-value">{result['avg_risk']}%</div>
    </div>
    
    <div class="card">
        <h3>📊 Model Predictions</h3>
        <div id="chart" class="chart"></div>
    </div>
    
    <div class="card">
        <h3>🎯 Individual Models</h3>
        <div class="models">
"""
    
    for name, risk in zip(model_names, model_risks):
        color = "#27ae60" if risk < 30 else "#f39c12" if risk < 70 else "#e74c3c"
        html += f'<div class="model-box" style="border-top-color:{color}"><div>{name}</div><div class="model-risk" style="color:{color}">{risk}%</div></div>'
    
    html += f"""
        </div>
    </div>
    
    <div class="footer">
        <p>Prediction using trained ensemble models</p>
    </div>
</div>

<script>
var data = {{
    x: {json.dumps(model_names)},
    y: {json.dumps(model_risks)},
    type: 'bar',
    marker: {{color: {json.dumps(model_risks)}, colorscale: 'Reds', showscale: true}}
}};
Plotly.newPlot('chart', [data], {{yaxis:{{range:[0,105]}}}}, {{responsive:true}});
</script>
</body>
</html>"""
    
    with open("flight_prediction.html", "w", encoding="utf-8") as f:f.write(html)

    webbrowser.open("flight_prediction.html")

if __name__ == "__main__":
    # Example: Test new flight
    new_flight = {
        'Aircraft_Age_Years': 8,
        'Days_Since_Maintenance': 20,
        'Flight_Hours': 15000,
        'Aircraft_Type': 'Boeing737',
        'Flight_Distance_km': 1200,
        'Passenger_Count': 180,
        'Cargo_Weight_tons': 5.5,
        'Departure_Weather_Severity': 4,
        'Destination_Weather_Severity': 7,
        'Wind_Speed_kmh': 35,
        'Pilot_Experience_Hours': 6500,
        'Time_of_Day': 'Afternoon',
        'Route_Complexity': 'Medium',
        'Departure_Airport_Category': 'Category1',
        'Destination_Airport_Category': 'Category2',
    }
    
    print("\n📍 Predicting for new flight...")
    result = predict_risk(new_flight)
    print(f"   Risk: {result['avg_risk']}% - {result['level']}")
    print(f"   Action: {result['action']}")
    
    show_results(result, new_flight)
