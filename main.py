import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os

app = FastAPI(title="PWID Risk Prediction API")

# 1. LOAD MODEL
MODEL_FILE = "pwid_risk_model_clinical_v2.joblib"
if not os.path.exists(MODEL_FILE):
    raise RuntimeError(f"Model file {MODEL_FILE} not found!")
model = joblib.load(MODEL_FILE)

# 2. DEFINE INPUT SCHEMAS
class DailyInput(BaseModel):
    sleep_quality: int       # 1-5
    has_pain_or_gut_issue: bool
    sensory_overload: bool
    emotional_distress: bool
    routine_change: bool

class PredictionRequest(BaseModel):
    person_id: int
    current_input: DailyInput
    # We now demand the history from the backend!
    history_last_6_days: List[DailyInput] 

class PredictionOutput(BaseModel):
    person_id: int
    risk_probability: float
    risk_level: str
    contributing_factors: List[str]

@app.post("/predict", response_model=PredictionOutput)
def predict_risk(request: PredictionRequest):
    
    # A. PREPARE DATA
    # Convert the list of Pydantic objects into a list of dictionaries
    history_data = [item.dict() for item in request.history_last_6_days]
    current_data = request.current_input.dict()
    
    # Create a DataFrame of all 7 days (History + Current)
    all_days_df = pd.DataFrame(history_data + [current_data])

    # Check if we actually have enough data
    if len(all_days_df) < 1:
        raise HTTPException(status_code=400, detail="Not enough data to predict.")

    # B. FEATURE ENGINEERING (The Python Logic)
    # We map the simple inputs to the model's complex features for the CURRENT day
    row_for_model = {
        'sleep_hours': current_data['sleep_quality'] * 1.5 + 2, 
        'sleep_quality': current_data['sleep_quality'],
        'bowel_issue': 1 if current_data['has_pain_or_gut_issue'] else 0,
        'pain_signs': 1 if current_data['has_pain_or_gut_issue'] else 0,
        'sensory_overload': 1 if current_data['sensory_overload'] else 0,
        'routine_disruption': 1 if current_data['routine_change'] else 0,
        'comm_frustration': 1 if current_data['emotional_distress'] else 0
    }

    # C. CALCULATE ROLLING AVERAGES (The Magic)
    # We take the mean of the column 'sleep_quality' from the 7-day DataFrame
    row_for_model['sleep_quality_avg_7d'] = all_days_df['sleep_quality'].mean()
    
    # Map boolean inputs to 0/1 for averages
    all_days_df['bowel_int'] = all_days_df['has_pain_or_gut_issue'].astype(int)
    row_for_model['bowel_issue_avg_7d'] = all_days_df['bowel_int'].mean()

    all_days_df['sensory_int'] = all_days_df['sensory_overload'].astype(int)
    row_for_model['sensory_overload_avg_7d'] = all_days_df['sensory_int'].mean()

    # D. FINALIZE INPUT FOR MODEL
    model_columns = [
        "sleep_hours", "sleep_quality", "bowel_issue", "pain_signs", 
        "sensory_overload", "routine_disruption", "comm_frustration",
        "sleep_quality_avg_7d", "bowel_issue_avg_7d", "sensory_overload_avg_7d"
    ]
    
    # Create final DataFrame (1 row)
    input_df = pd.DataFrame([row_for_model])[model_columns]

    # E. PREDICT
    probability = model.predict_proba(input_df)[0][1]
    
    # F. INTERPRETATION
    risk_level = "High" if probability > 0.7 else "Moderate" if probability > 0.4 else "Low"
    
    factors = []
    if row_for_model['sleep_quality_avg_7d'] < 3: factors.append("Consistent Poor Sleep")
    if row_for_model['bowel_issue'] == 1: factors.append("Physical Discomfort")
    if probability > 0.6 and not factors: factors.append("Accumulated Stress")

    return {
        "person_id": request.person_id,
        "risk_probability": round(probability, 2),
        "risk_level": risk_level,
        "contributing_factors": factors
    }