import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

xgb = joblib.load(os.path.join(PROJECT_ROOT, "models", "xgb_model.pkl"))
lgbm = joblib.load(os.path.join(PROJECT_ROOT, "models", "lgbm_model.pkl"))
dnn = load_model(os.path.join(PROJECT_ROOT, "models", "dnn_model.h5"))
meta = joblib.load(os.path.join(PROJECT_ROOT, "models", "meta_model.pkl"))

with open(os.path.join(PROJECT_ROOT, "models", "threshold.txt"), "r") as f:
    THRESHOLD = float(f.read())

scaler_path = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    scaler = None

def predict_transaction(tx: np.ndarray):
    """
    Input: tx -> np.ndarray of shape (1, 30)
    Output: risk score (0-1), label (0=Legit, 1=Fraud)
    """
    if scaler:
        tx = scaler.transform(tx)

    base_preds = [
        xgb.predict_proba(tx)[:,1][0],
        lgbm.predict_proba(tx)[:,1][0],
        dnn.predict(tx)[0][0]
    ]

    risk = meta.predict_proba([base_preds])[0][1]
    label = int(risk > THRESHOLD)
    return risk, label
