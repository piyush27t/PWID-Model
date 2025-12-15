import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os

app = FastAPI(title="PWID Risk Prediction API")

# ---------------------------------------------------------
# 1. LOAD THE SAVED MODEL
# ---------------------------------------------------------
MODEL_FILE = "pwid_risk_model_clinical_v2.joblib"

if not os.path.exists(MODEL_FILE):
    raise RuntimeError(f"CRITICAL ERROR: {MODEL_FILE} not found! Did you upload it?")

print(f"Loading model from {MODEL_FILE}...")
model = joblib.load(MODEL_FILE)
print("✅ Model loaded successfully.")

# ---------------------------------------------------------
# 2. INPUT SCHEMAS
# ---------------------------------------------------------
class PatientInput(BaseModel):
    person_id: int
    sleep_quality: int       
    has_pain_or_gut_issue: bool 
    sensory_overload: bool
    emotional_distress: bool 
    routine_change: bool

class PredictionOutput(BaseModel):
    person_id: int
    risk_probability: float
    risk_level: str
    contributing_factors: List[str]

# ---------------------------------------------------------
# 3. HELPER: MOCK HISTORY
# ---------------------------------------------------------
def get_patient_history(person_id: int):
    # In real deployment, connect to MongoDB/SQL here
    return pd.DataFrame({
        'sleep_quality': np.random.randint(1, 6, size=6),
        'bowel_issue': np.random.binomial(1, 0.3, size=6),
        'sensory_overload': np.random.binomial(1, 0.2, size=6)
    })

# ---------------------------------------------------------
# 4. PREDICT ENDPOINT
# ---------------------------------------------------------
@app.post("/predict", response_model=PredictionOutput)
def predict_risk(input_data: PatientInput):
    
    # A. Fetch History
    history_df = get_patient_history(input_data.person_id)

    # B. Map User Inputs to Features
    current_data = {
        'sleep_hours': input_data.sleep_quality * 1.5 + 2, 
        'sleep_quality': input_data.sleep_quality,
        'bowel_issue': 1 if input_data.has_pain_or_gut_issue else 0,
        'pain_signs': 1 if input_data.has_pain_or_gut_issue else 0,
        'sensory_overload': 1 if input_data.sensory_overload else 0,
        'routine_disruption': 1 if input_data.routine_change else 0,
        'comm_frustration': 1 if input_data.emotional_distress else 0
    }

    # C. Calculate Rolling Averages
    full_sleep = np.append(history_df['sleep_quality'].values, current_data['sleep_quality'])
    full_bowel = np.append(history_df['bowel_issue'].values, current_data['bowel_issue'])
    full_sensory = np.append(history_df['sensory_overload'].values, current_data['sensory_overload'])

    current_data['sleep_quality_avg_7d'] = full_sleep.mean()
    current_data['bowel_issue_avg_7d'] = full_bowel.mean()
    current_data['sensory_overload_avg_7d'] = full_sensory.mean()

    # D. Prepare Final DataFrame (Order matters!)
    model_columns = [
        "sleep_hours", "sleep_quality", "bowel_issue", "pain_signs", 
        "sensory_overload", "routine_disruption", "comm_frustration",
        "sleep_quality_avg_7d", "bowel_issue_avg_7d", "sensory_overload_avg_7d"
    ]
    
    input_df = pd.DataFrame([current_data])[model_columns]

    # E. Predict
    probability = model.predict_proba(input_df)[0][1]
    
    risk_level = "High" if probability > 0.7 else "Moderate" if probability > 0.4 else "Low"
    
    factors = []
    if current_data['sleep_quality_avg_7d'] < 3: factors.append("Poor Sleep History")
    if current_data['bowel_issue'] == 1: factors.append("Physical Discomfort")
    if probability > 0.6 and not factors: factors.append("Combined Triggers")

    return {
        "person_id": input_data.person_id,
        "risk_probability": round(probability, 2),
        "risk_level": risk_level,
        "contributing_factors": factors
    }