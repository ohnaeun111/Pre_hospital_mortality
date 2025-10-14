import numpy as np
import joblib
import os

import os
import joblib
import numpy as np

def eval_model(input_data, SavePath):
    print(f"SavePath: {SavePath}")
    print(f"input_data shape: {input_data.shape}")
    
    if input_data.shape[1] != 149:
        print(f"Warning: Expected 149 features, but got {input_data.shape[1]}")

    model_path = os.path.join(SavePath, 'Prehospital_AI_model.pkl')
    print(f"Trying to load model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Error: File does not exist - {model_path}")

    xgb_model, lgbm_model, rf_model = joblib.load(model_path)

    xgb_test_proba = xgb_model.predict_proba(input_data)[:, 1]
    lgbm_test_proba = lgbm_model.predict_proba(input_data)[:, 1]
    rf_test_proba = rf_model.predict_proba(input_data)[:, 1]

    final_test_proba = (xgb_test_proba + lgbm_test_proba + rf_test_proba) / 3

    return final_test_proba
