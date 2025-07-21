import numpy as np
import joblib
import os

def eval_model(input_data, SavePath):
    print(f"SavePath: {SavePath}")
    n_folds = 5 
    all_fold_models = []
    print(f"input__data shape: {input_data.shape}")
    if input_data.shape[1] != 149:
        print(f"but {input_data.shape[1]}")


    for num_folds in range(1, n_folds + 1):
        model_path = os.path.join(SavePath, f'Prehospital_AI_model_fold{num_folds}.pkl')
        print(f"Trying to load model from: {model_path}")
        if not os.path.exists(model_path):
            print(f"Error: File does not exist - {model_path}")
            continue

        loaded_models = joblib.load(model_path)
        all_fold_models.append(loaded_models)

    if not all_fold_models:
        raise FileNotFoundError("No models could be loaded. Please check the model paths.")

    test_eval_probas = []

    for xgb_model, lgbm_model, rf_model in all_fold_models:
        
        xgb_test_proba = xgb_model.predict_proba(input_data)[:, 1]
        lgbm_test_proba = lgbm_model.predict_proba(input_data)[:, 1]
        rf_test_proba = rf_model.predict_proba(input_data)[:, 1]

        avg_test_proba = (xgb_test_proba + lgbm_test_proba + rf_test_proba) / 3
        test_eval_probas.append(avg_test_proba)

    final_test_proba = np.mean(test_eval_probas, axis=0)

    return final_test_proba