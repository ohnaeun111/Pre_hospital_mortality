import os
import joblib
import numpy as np

def load_models(model_dir, n_folds=5):
    models = []
    for fold in range(1, n_folds + 1):
        path = os.path.join(model_dir, f'Prehospital_AI_model_fold{fold}.pkl')
        models.append(joblib.load(path))  # returns (xgb, lgbm, rf)
    return models


def predict_ensemble(models, data):
    all_probas = []
    for xgb, lgbm, rf in models:
        xgb_p = xgb.predict_proba(data)[:, 1]
        lgbm_p = lgbm.predict_proba(data)[:, 1]
        rf_p = rf.predict_proba(data)[:, 1]
        all_probas.append((xgb_p + lgbm_p + rf_p) / 3)
    return np.mean(all_probas, axis=0)


if __name__ == '__main__':
    from sklearn.datasets import load_svmlight_file

    model_directory = 'models'
    models = load_models(model_directory)

    data_test, _ = load_svmlight_file('data/test.svm')  # replace with actual test data path
    predictions = predict_ensemble(models, data_test)

    
