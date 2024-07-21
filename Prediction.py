import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.preprocessing import StandardScaler

# Load .pkl files generated from modelling.py
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')