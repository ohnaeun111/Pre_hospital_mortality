import os
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from src.model import get_models
from src.data_loader import load_data, preprocess_data
from src.evaluation import get_clf_eval

# Load and preprocess data
X, y = load_data()
X_train, X_test, y_train, y_test = preprocess_data(X, y)

# Set up Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=77)

# Lists to store evaluation metrics
acc_list, pre_list, sen_list, sp_list, f1_list, auc_list, bal_list = [], [], [], [], [], [], []
test_eval_probas = []

# Train models using K-Fold cross-validation
for n_iter, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f'-------------------- Fold {n_iter} -------------------')

    # Split data into training and validation sets
    X_train_fold, X_valid_fold = X_train[train_idx], X_train[valid_idx]
    y_train_fold, y_valid_fold = y_train[train_idx], y_train[valid_idx]

    # Apply SMOTE for handling class imbalance
    smote = SMOTE()
    X_train_over, y_train_over = smote.fit_resample(X_train_fold, y_train_fold)
    sample_weights_over = np.where(y_train_over == 0, 0.28, 0.72)  # Assign different weights to classes

    # Load models
    model1, model2, model3 = get_models()

    # Train models
    model1.fit(X_train_over, y_train_over)
    model2.fit(X_train_over, y_train_over)
    model3.fit(X_train_over, y_train_over, sample_weight=sample_weights_over)

    # Validation predictions
    xgb_valid_proba = model1.predict_proba(X_valid_fold)[:, 1]
    lgbm_valid_proba = model2.predict_proba(X_valid_fold)[:, 1]
    gbm_valid_proba = model3.predict_proba(X_valid_fold)[:, 1]

    # Ensemble prediction (average of all models)
    valid_eval_proba = (xgb_valid_proba + lgbm_valid_proba + gbm_valid_proba) / 3
    valid_eval = np.where(valid_eval_proba > 0.5, 1, 0)

    # Evaluate validation performance
    accuracy, precision, recall, spec, f1, roc_auc, bal_acc = get_clf_eval(y_valid_fold, valid_eval, valid_eval_proba)
    acc_list.append(accuracy)
    pre_list.append(precision)
    sen_list.append(recall)
    sp_list.append(spec)
    f1_list.append(f1)
    auc_list.append(roc_auc)
    bal_list.append(bal_acc)

    # Store test predictions for final evaluation
    test_eval_probas.append((model1.predict_proba(X_test)[:, 1] + 
                             model2.predict_proba(X_test)[:, 1] + 
                             model3.predict_proba(X_test)[:, 1]) / 3)


# Print validation performance summary
print('Validation Performance Summary:')
print('Accuracy: {:.4f}, Precision: {:.4f}, Sensitivity: {:.4f}, Specificity: {:.4f}, F1 Score: {:.4f}, AUC: {:.4f}, Balanced Accuracy: {:.4f}'
      .format(np.mean(acc_list), np.mean(pre_list), np.mean(sen_list), np.mean(sp_list), np.mean(f1_list), np.mean(auc_list), np.mean(bal_list)))
