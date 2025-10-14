import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.model import get_models
from src.data_loader import load_data, preprocess_data
from src.evaluation import get_clf_eval

X, y = load_data()
X_train_full, X_test, y_train_full, y_test = preprocess_data(X, y)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=77
)

smote = SMOTE(random_state=77)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

sample_weights_over1 = np.where(y_train_over == 0, 0.2, 0.8)
sample_weights_over2 = np.where(y_train_over == 0, 0.3, 0.7)

model1, model2, model3 = get_models()

model1.fit(X_train_over, y_train_over, sample_weight=sample_weights_over1)
model2.fit(X_train_over, y_train_over, sample_weight=sample_weights_over2)
model3.fit(X_train_over, y_train_over)

xgb_valid_proba = model1.predict_proba(X_valid)[:, 1]
lgbm_valid_proba = model2.predict_proba(X_valid)[:, 1]
rf_valid_proba = model3.predict_proba(X_valid)[:, 1]

valid_eval_proba = (xgb_valid_proba + lgbm_valid_proba + rf_valid_proba) / 3
valid_eval = np.where(valid_eval_proba > 0.5, 1, 0)

accuracy, precision, recall, spec, f1, roc_auc, bal_acc = get_clf_eval(y_valid, valid_eval, valid_eval_proba)
print('\nValidation Performance:')
print('Accuracy: {:.4f}, Precision: {:.4f}, Sensitivity: {:.4f}, Specificity: {:.4f}, F1: {:.4f}, AUC: {:.4f}, Balanced: {:.4f}'
      .format(accuracy, precision, recall, spec, f1, roc_auc, bal_acc))
