import joblib
import pandas as pd
from pathlib import Path
import json

MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'heart_model.pkl'

def load_model(path: str = None):
    p = Path(path) if path else MODEL_PATH
    data = joblib.load(p)
    return data['model'], data['columns']

MODEL, COLUMNS = None, None

def ensure_model_loaded():
    global MODEL, COLUMNS
    if MODEL is None or COLUMNS is None:
        MODEL, COLUMNS = load_model()

def predict_heart_disease(input_dict):
    """Input: dict of feature_name: value.
    Returns prediction label and probability.
    """
    ensure_model_loaded()

    # Create DataFrame with all expected columns. Missing columns will be filled with 0.
    row = {col: input_dict.get(col, 0) for col in COLUMNS}
    df = pd.DataFrame([row], columns=COLUMNS)
    proba = MODEL.predict_proba(df)[0][1] if hasattr(MODEL, 'predict_proba') else None
    pred = MODEL.predict(df)[0]
    return {
        'prediction': int(pred),
        'probability': float(proba) if proba is not None else None,
        'input_used': row
    }

# helper to parse user input provided as string (e.g. from LangChain)
def parse_input_string(s: str):
    try:
        # allow JSON-like or Python dict-like strings
        return json.loads(s)
    except Exception:
        # fallback to eval (be cautious)
        return eval(s)
